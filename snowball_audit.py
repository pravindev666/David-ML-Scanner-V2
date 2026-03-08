
import pandas as pd
import numpy as np
import os
import sys
from collections import defaultdict
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from data_engine import load_all_data
from feature_forge import engineer_features
from models.ensemble_classifier import EnsembleClassifier
from utils import UP, DOWN, SIDEWAYS

# ============================================================
# REALISTIC CONSTANTS (Bug Fixes Applied)
# ============================================================
SPREAD_WIN_PER_LOT   = 2500    # Fix #3: Realistic 1-lot Nifty spread profit
SPREAD_LOSS_PER_LOT  = 2500    # Fix #3: Symmetric loss
CONDOR_WIN_PER_LOT   = 1500    # Fix #3: Realistic IC premium collected
CONDOR_LOSS_PER_LOT  = 5000    # Fix #3: Realistic IC blowout
BROKERAGE_PER_TRADE  = 200     # Fix #4: ₹40 brokerage + ₹160 STT/taxes
MAX_LOTS             = 10      # Fix #5: Liquidity cap
TARGET_PCT           = 0.007   # Fix #2: Symmetric ±0.7% (~170 points)
STOP_PCT             = 0.007   # Fix #2: Symmetric ±0.7%

def run_realistic_audit(start_date="2025-01-01", initial_capital=100000):
    print(f"🚀 Starting REALISTIC ₹1L AUDIT (2025 - Present)...")
    print(f"   Fixes Applied: Cooldown, Symmetric Stops, Realistic P/L, Brokerage, Lot Cap")
    
    df_raw = load_all_data()
    df_full_feat, feature_cols = engineer_features(df_raw)
    df_test_full = df_full_feat[df_full_feat['date'] >= start_date].copy()
    
    styles = ["Greedy", "Gambler"]
    modes = ["Fixed (1 Lot)", "Snowball (Compound)"]
    results = defaultdict(lambda: defaultdict(list))

    for style in styles:
        for mode in modes:
            print(f"\n--- Style: {style} | Mode: {mode} ---")
            
            capital = initial_capital
            unique_months = sorted(df_test_full['date'].dt.strftime('%Y-%m').unique())
            current_year = None
            ensemble = None
            total_trades = 0
            total_wins = 0
            total_brokerage = 0
            
            for month in unique_months:
                df_month = df_test_full[df_test_full['date'].dt.strftime('%Y-%m') == month].copy()
                year_of_month = int(month.split('-')[0])
                
                if year_of_month != current_year:
                    print(f"   Training David for {year_of_month}...")
                    train_start = f"{year_of_month-10}-01-01"
                    train_end = f"{year_of_month-1}-12-31"
                    df_train = df_full_feat[(df_full_feat['date'] >= train_start) & (df_full_feat['date'] <= train_end)].copy()
                    ensemble = EnsembleClassifier()
                    ensemble.train(df_train, feature_cols, verbose=False)
                    current_year = year_of_month
                
                month_pl = 0
                month_trades = 0
                month_brokerage = 0
                
                # ====================================================
                # FIX #1: Trade Cooldown — skip days after entering
                # ====================================================
                cooldown_until = 0  # index of next allowed trade day
                
                for i in range(len(df_month)):
                    if i < cooldown_until:
                        continue  # Skip — still in a trade
                    
                    row = df_month.iloc[i]
                    date = row['date']
                    price_now = row['close']
                    
                    pred = ensemble.predict(row)
                    verdict = pred['direction']
                    conf = pred['confidence'] * 100
                    
                    # Whipsaw
                    adx = row.get('adx', 25)
                    bb_width_pct = row.get('bb_width', 0)
                    whipsaw_prob = 0
                    if adx < 20: whipsaw_prob += 40
                    if bb_width_pct < df_full_feat['bb_width'].quantile(0.2): whipsaw_prob += 30
                    
                    trade_type = None
                    if style == "Greedy":
                        if verdict in [UP, DOWN] and conf > 40:
                            if whipsaw_prob < 45: trade_type = "SPREAD"
                            elif whipsaw_prob < 60: trade_type = "CONDOR"
                    else: # Gambler
                        if verdict in [UP, DOWN] and conf > 40: trade_type = "SPREAD"
                        elif whipsaw_prob > 55: trade_type = "CONDOR"
                    
                    if trade_type:
                        # FIX #5: Lot Calculation with Cap
                        if mode == "Fixed (1 Lot)":
                            lots = 1
                        else:
                            lots = min(MAX_LOTS, max(1, int(capital / 150000)))
                        
                        full_idx = df_full_feat.index[df_full_feat['date'] == date][0]
                        max_hold = 15 if trade_type == "SPREAD" else 5
                        if style == "Gambler" and trade_type == "SPREAD": max_hold = 10
                        
                        future = df_full_feat.iloc[full_idx + 1 : full_idx + max_hold + 1]
                        if len(future) < 2: continue
                        
                        trade_pl = 0
                        hold_days = max_hold  # default: held to expiry
                        
                        if trade_type == "SPREAD":
                            # FIX #2: Symmetric target/stop
                            for d in range(len(future)):
                                f_row = future.iloc[d]
                                if verdict == UP:
                                    if f_row['high'] >= price_now * (1 + TARGET_PCT):
                                        trade_pl = SPREAD_WIN_PER_LOT * lots
                                        hold_days = d + 1
                                        break
                                    if f_row['low'] <= price_now * (1 - STOP_PCT):
                                        trade_pl = -SPREAD_LOSS_PER_LOT * lots
                                        hold_days = d + 1
                                        break
                                else: # DOWN
                                    if f_row['low'] <= price_now * (1 - TARGET_PCT):
                                        trade_pl = SPREAD_WIN_PER_LOT * lots
                                        hold_days = d + 1
                                        break
                                    if f_row['high'] >= price_now * (1 + STOP_PCT):
                                        trade_pl = -SPREAD_LOSS_PER_LOT * lots
                                        hold_days = d + 1
                                        break
                            
                            # If neither target nor stop hit, close at expiry
                            if trade_pl == 0:
                                final_p = future.iloc[-1]['close']
                                if verdict == UP:
                                    pct_move = (final_p - price_now) / price_now
                                else:
                                    pct_move = (price_now - final_p) / price_now
                                # Scale P/L proportionally, capped
                                trade_pl = int(pct_move / TARGET_PCT * SPREAD_WIN_PER_LOT * lots)
                                trade_pl = np.clip(trade_pl, -SPREAD_LOSS_PER_LOT * lots, SPREAD_WIN_PER_LOT * lots)
                        
                        else: # CONDOR
                            if future['high'].max() <= price_now * 1.015 and future['low'].min() >= price_now * 0.985:
                                trade_pl = CONDOR_WIN_PER_LOT * lots
                            else:
                                trade_pl = -CONDOR_LOSS_PER_LOT * lots
                        
                        # FIX #4: Deduct Brokerage
                        brokerage = BROKERAGE_PER_TRADE * lots
                        trade_pl -= brokerage
                        
                        capital += trade_pl
                        month_pl += trade_pl
                        month_trades += 1
                        month_brokerage += brokerage
                        total_trades += 1
                        total_brokerage += brokerage
                        if trade_pl > 0: total_wins += 1
                        
                        # FIX #1: Set cooldown
                        cooldown_until = i + hold_days
                
                results[style][mode].append({
                    "month": month, "pl": month_pl, "capital": capital,
                    "trades": month_trades, "brokerage": month_brokerage
                })
            
            win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
            print(f"   Total Trades: {total_trades} | Win Rate: {win_rate:.1f}% | Brokerage Paid: ₹{total_brokerage:,.0f}")

    # Generate Report
    print("\n🚀 GENERATING REALISTIC AUDIT REPORT...")
    md = ["# 🛡️ David Oracle: REALISTIC ₹1L Capital Audit (2025-2026)"]
    md.append(f"\nAudit: 2025-01-01 to Present | Capital: ₹1,00,000")
    md.append("\n**Bug Fixes Applied**: Trade Cooldown, Symmetric Stops (±0.7%), Realistic P/L (₹2,500/lot), Brokerage (₹200/trade), Lot Cap (10 max)")
    
    for style in styles:
        md.append(f"\n## 📈 Style: {style}")
        md.append("| Month | Trades | Fixed P/L | Snowball P/L | Snowball Capital |")
        md.append("|:---|:---:|:---:|:---:|:---|")
        
        fixed_data = results[style]["Fixed (1 Lot)"]
        snow_data = results[style]["Snowball (Compound)"]
        
        for i in range(len(fixed_data)):
            f = fixed_data[i]
            s = snow_data[i]
            md.append(f"| {f['month']} | {f['trades']} | ₹{f['pl']:>+10,.0f} | ₹{s['pl']:>+10,.0f} | ₹{s['capital']:>12,.0f} |")
        
        fixed_profit = sum(x['pl'] for x in fixed_data)
        snow_profit = snow_data[-1]['capital'] - initial_capital
        fixed_trades = sum(x['trades'] for x in fixed_data)
        snow_trades = sum(x['trades'] for x in snow_data)
        
        md.append(f"\n**Fixed: {fixed_trades} trades → ₹{fixed_profit:,.0f} profit**")
        md.append(f"**Snowball: {snow_trades} trades → ₹{snow_profit:,.0f} profit (Capital: ₹{snow_data[-1]['capital']:,.0f})**")

    report_content = "\n".join(md)
    with open("lakh_audit_2025_realistic.md", "w", encoding="utf-8") as f:
        f.write(report_content)
    
    print("\n✅ REALISTIC Audit Complete.")
    print(report_content)

if __name__ == "__main__":
    run_realistic_audit()
