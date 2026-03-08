
import pandas as pd
import numpy as np
import os
import sys
from collections import defaultdict
from datetime import datetime

# Add root to sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from data_engine import load_all_data
from feature_forge import engineer_features
from models.ensemble_classifier import EnsembleClassifier
from utils import UP, DOWN, SIDEWAYS

def run_yearly_comparison(years=[2021, 2022, 2023, 2024, 2025]):
    print(f"🚀 Starting YEARLY COMPARISON ({years[0]}-{years[-1]})...")
    
    # 1. Load data
    df_raw = load_all_data()
    df_full_feat, feature_cols = engineer_features(df_raw)
    
    # Define Rules to Test
    # Tuple: (min_conf, use_whipsaw_guard_for_spreads, name)
    rule_configs = [
        (60, True, "Golden (Safe)"),
        (40, True, "Greedy (Moderate)"),
        (40, False, "Gambler (Aggressive)")
    ]
    
    # Data structure: final_results[rule_name][year] = {pl, trades, wins, max_dd}
    final_results = {}

    for min_conf, use_whipsaw, rule_name in rule_configs:
        print(f"\n--- Testing Rule: {rule_name} ---")
        final_results[rule_name] = {}
        
        for year in years:
            print(f" Analyzing {year}...")
            # 10-year training window
            train_start = f"{year-10}-01-01"
            train_end = f"{year-1}-12-31"
            test_start = f"{year}-01-01"
            test_end = f"{year}-12-31"
            
            df_train = df_full_feat[(df_full_feat['date'] >= train_start) & (df_full_feat['date'] <= train_end)].copy()
            df_test = df_full_feat[(df_full_feat['date'] >= test_start) & (df_full_feat['date'] <= test_end)].copy()
            
            if len(df_train) < 1000 or len(df_test) < 20: continue
            
            ensemble = EnsembleClassifier()
            ensemble.train(df_train, feature_cols, verbose=False)
            
            equity = 100000
            peak = 100000
            pl_year = 0
            trades_year = 0
            wins_year = 0
            max_dd_year = 0

            for i in range(len(df_test)):
                row = df_test.iloc[i]
                price_now = row['close']
                date = row['date']
                
                pred = ensemble.predict(row)
                verdict = pred['direction']
                conf = pred['confidence'] * 100
                
                # Whipsaw Logic
                adx = row.get('adx', 25)
                bb_width_pct = row.get('bb_width', 0)
                whipsaw_prob = 0
                if adx < 20: whipsaw_prob += 40
                if bb_width_pct < df_full_feat['bb_width'].quantile(0.2): whipsaw_prob += 30
                
                type = None
                if verdict in [UP, DOWN] and conf > min_conf:
                    if not use_whipsaw or whipsaw_prob < 40:
                        type = "SPREAD"
                elif (verdict == SIDEWAYS or whipsaw_prob > 55):
                    type = "CONDOR"
                
                if type:
                    full_idx = df_full_feat.index[df_full_feat['date'] == date][0]
                    max_hold = 15 if type == "SPREAD" else 5
                    future = df_full_feat.iloc[full_idx + 1 : full_idx + max_hold + 1]
                    if len(future) < 2: continue
                    
                    pl = 0
                    win = False
                    if type == "SPREAD":
                        target = 0.005 if verdict == UP else -0.005
                        stop = -0.01 if verdict == UP else 0.01
                        for d in range(len(future)):
                            f_row = future.iloc[d]
                            if verdict == UP:
                                if f_row['high'] >= price_now * (1 + target): pl = 5000; win = True; break
                                if f_row['low'] <= price_now * (1 + stop): pl = -5000; break
                            else: # DOWN
                                if f_row['low'] <= price_now * (1 + target): pl = 5000; win = True; break
                                if f_row['high'] >= price_now * (1 + stop): pl = -5000; break
                        if pl == 0:
                            pl = (future.iloc[-1]['close'] - price_now) / price_now * 500000 if verdict==UP else \
                                 (price_now - future.iloc[-1]['close']) / price_now * 500000
                            pl = np.clip(pl, -5000, 5000)
                            if pl > 1000: win = True
                    else: # CONDOR
                        if future['high'].max() <= price_now * 1.015 and future['low'].min() >= price_now * 0.985:
                            pl = 2500; win = True
                        else:
                            pl = -7500
                            
                    equity += pl
                    trades_year += 1
                    if win: wins_year += 1
                    pl_year += pl
                    if equity > peak: peak = equity
                    dd = (peak - equity) / peak * 100
                    max_dd_year = max(max_dd_year, dd)

            final_results[rule_name][year] = {
                "pl": pl_year, "trades": trades_year, "win_rate": (wins_year/trades_year*100) if trades_year>0 else 0, "max_dd": max_dd_year
            }

    # Generate Report Table
    print("\n--- STRESS TEST YEARLY COMPARISON ---")
    for rule_name in final_results:
        print(f"\nRule Set: {rule_name}")
        print("| Year | Trades | Win Rate | Net P/L (INR) | Max DD |")
        print("|:---|:---:|:---:|:---:|:---|")
        total_pl = 0
        for year in years:
            res = final_results[rule_name][year]
            print(f"| {year} | {res['trades']} | {res['win_rate']:.1f}% | ₹{res['pl']:>+10,.0f} | {res['max_dd']:.1f}% |")
            total_pl += res['pl']
        print(f"| **TOTAL** | | | **₹{total_pl:>+12,.0f}** | |")

if __name__ == "__main__":
    run_yearly_comparison()
