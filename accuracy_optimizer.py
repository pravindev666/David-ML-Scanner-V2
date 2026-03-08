
import pandas as pd
import numpy as np
import os
import sys
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from data_engine import load_all_data
from feature_forge import engineer_features
from models.ensemble_classifier import EnsembleClassifier
from utils import UP, DOWN, SIDEWAYS

def test_accuracy(df_full_feat, df_test, ensemble, filters, label=""):
    """Test accuracy of David's verdict under given filters."""
    correct = 0
    total = 0
    
    for i in range(len(df_test)):
        row = df_test.iloc[i]
        date = row['date']
        price_now = row['close']
        
        pred = ensemble.predict(row)
        verdict = pred['direction']
        conf = pred['confidence'] * 100
        
        # Apply filters
        skip = False
        for f_name, f_check in filters.items():
            if not f_check(row, conf, verdict, df_full_feat):
                skip = True
                break
        if skip:
            continue
        
        # Check actual outcome (5-day)
        full_idx = df_full_feat.index[df_full_feat['date'] == date][0]
        future = df_full_feat.iloc[full_idx + 1 : full_idx + 6]
        if len(future) < 3:
            continue
        
        future_close = future.iloc[-1]['close']
        pct_change = (future_close - price_now) / price_now * 100
        
        is_correct = False
        if verdict == UP and pct_change > 0:
            is_correct = True
        elif verdict == DOWN and pct_change < 0:
            is_correct = True
        elif verdict == SIDEWAYS and abs(pct_change) < 1.0:
            is_correct = True
        
        total += 1
        if is_correct:
            correct += 1
    
    accuracy = (correct / total * 100) if total > 0 else 0
    return {"correct": correct, "total": total, "accuracy": accuracy}


def run_optimizer():
    print("🔬 ACCURACY OPTIMIZER: Testing Filter Combinations")
    print("=" * 60)
    
    df_raw = load_all_data()
    df_full_feat, feature_cols = engineer_features(df_raw)
    
    # Test across 2021-2025
    years = [2021, 2022, 2023, 2024, 2025]
    
    # ============================================================
    # DEFINE ALL FILTER COMBINATIONS TO TEST
    # ============================================================
    combos = {
        # --- BASE STYLES (for reference) ---
        "Base: Conf > 60%": {
            "conf": lambda row, conf, v, df: conf > 60,
        },
        "Base: Conf > 40%": {
            "conf": lambda row, conf, v, df: conf > 40,
        },
        
        # --- CONFIDENCE VARIATIONS ---
        "Conf > 55%": {
            "conf": lambda row, conf, v, df: conf > 55,
        },
        "Conf > 50%": {
            "conf": lambda row, conf, v, df: conf > 50,
        },
        "Conf > 65%": {
            "conf": lambda row, conf, v, df: conf > 65,
        },
        
        # --- VIX FILTERS ---
        "Conf>50 + Low VIX (<20)": {
            "conf": lambda row, conf, v, df: conf > 50,
            "vix": lambda row, conf, v, df: row.get('vix', 15) < 20,
        },
        "Conf>50 + High VIX (>20)": {
            "conf": lambda row, conf, v, df: conf > 50,
            "vix": lambda row, conf, v, df: row.get('vix', 15) > 20,
        },
        "Conf>40 + Low VIX (<18)": {
            "conf": lambda row, conf, v, df: conf > 40,
            "vix": lambda row, conf, v, df: row.get('vix', 15) < 18,
        },
        
        # --- ADX (TREND STRENGTH) FILTERS ---
        "Conf>50 + ADX>25 (Strong Trend)": {
            "conf": lambda row, conf, v, df: conf > 50,
            "adx": lambda row, conf, v, df: row.get('adx', 20) > 25,
        },
        "Conf>50 + ADX>30 (Very Strong Trend)": {
            "conf": lambda row, conf, v, df: conf > 50,
            "adx": lambda row, conf, v, df: row.get('adx', 20) > 30,
        },
        "Conf>40 + ADX>25": {
            "conf": lambda row, conf, v, df: conf > 40,
            "adx": lambda row, conf, v, df: row.get('adx', 20) > 25,
        },
        
        # --- RSI CONFIRMATION ---
        "Conf>50 + RSI Confirms (UP=RSI>50, DOWN=RSI<50)": {
            "conf": lambda row, conf, v, df: conf > 50,
            "rsi": lambda row, conf, v, df: (v == UP and row.get('rsi_14', 50) > 50) or (v == DOWN and row.get('rsi_14', 50) < 50) or v == SIDEWAYS,
        },
        "Conf>40 + RSI Confirms": {
            "conf": lambda row, conf, v, df: conf > 40,
            "rsi": lambda row, conf, v, df: (v == UP and row.get('rsi_14', 50) > 50) or (v == DOWN and row.get('rsi_14', 50) < 50) or v == SIDEWAYS,
        },
        
        # --- SMA TREND CONFIRMATION ---
        "Conf>50 + Price Above SMA20 for UP": {
            "conf": lambda row, conf, v, df: conf > 50,
            "sma": lambda row, conf, v, df: (v == UP and row['close'] > row.get('sma_20', row['close'])) or (v == DOWN and row['close'] < row.get('sma_20', row['close'])) or v == SIDEWAYS,
        },
        "Conf>40 + Price Above SMA50 for UP": {
            "conf": lambda row, conf, v, df: conf > 40,
            "sma": lambda row, conf, v, df: (v == UP and row['close'] > row.get('sma_50', row['close'])) or (v == DOWN and row['close'] < row.get('sma_50', row['close'])) or v == SIDEWAYS,
        },
        
        # --- MACD CONFIRMATION ---
        "Conf>50 + MACD Confirms (UP=MACD>Signal)": {
            "conf": lambda row, conf, v, df: conf > 50,
            "macd": lambda row, conf, v, df: (v == UP and row.get('macd', 0) > row.get('macd_signal', 0)) or (v == DOWN and row.get('macd', 0) < row.get('macd_signal', 0)) or v == SIDEWAYS,
        },
        
        # --- COMBO: ADX + VIX ---
        "Conf>50 + ADX>25 + VIX<20": {
            "conf": lambda row, conf, v, df: conf > 50,
            "adx": lambda row, conf, v, df: row.get('adx', 20) > 25,
            "vix": lambda row, conf, v, df: row.get('vix', 15) < 20,
        },
        
        # --- COMBO: RSI + ADX ---
        "Conf>50 + RSI Confirms + ADX>25": {
            "conf": lambda row, conf, v, df: conf > 50,
            "rsi": lambda row, conf, v, df: (v == UP and row.get('rsi_14', 50) > 50) or (v == DOWN and row.get('rsi_14', 50) < 50) or v == SIDEWAYS,
            "adx": lambda row, conf, v, df: row.get('adx', 20) > 25,
        },
        
        # --- COMBO: RSI + SMA ---
        "Conf>50 + RSI + SMA20 Confirm": {
            "conf": lambda row, conf, v, df: conf > 50,
            "rsi": lambda row, conf, v, df: (v == UP and row.get('rsi_14', 50) > 50) or (v == DOWN and row.get('rsi_14', 50) < 50) or v == SIDEWAYS,
            "sma": lambda row, conf, v, df: (v == UP and row['close'] > row.get('sma_20', row['close'])) or (v == DOWN and row['close'] < row.get('sma_20', row['close'])) or v == SIDEWAYS,
        },
        
        # --- TRIPLE COMBO ---
        "Conf>50 + RSI + ADX>25 + VIX<20": {
            "conf": lambda row, conf, v, df: conf > 50,
            "rsi": lambda row, conf, v, df: (v == UP and row.get('rsi_14', 50) > 50) or (v == DOWN and row.get('rsi_14', 50) < 50) or v == SIDEWAYS,
            "adx": lambda row, conf, v, df: row.get('adx', 20) > 25,
            "vix": lambda row, conf, v, df: row.get('vix', 15) < 20,
        },
        
        # --- UP-ONLY STRATEGIES ---
        "UP Only + Conf>50": {
            "conf": lambda row, conf, v, df: conf > 50,
            "direction": lambda row, conf, v, df: v == UP,
        },
        "UP Only + Conf>50 + RSI>50": {
            "conf": lambda row, conf, v, df: conf > 50,
            "direction": lambda row, conf, v, df: v == UP,
            "rsi": lambda row, conf, v, df: row.get('rsi_14', 50) > 50,
        },
        "UP Only + Conf>50 + ADX>25": {
            "conf": lambda row, conf, v, df: conf > 50,
            "direction": lambda row, conf, v, df: v == UP,
            "adx": lambda row, conf, v, df: row.get('adx', 20) > 25,
        },
    }
    
    # ============================================================
    # RUN ALL COMBOS ACROSS ALL YEARS
    # ============================================================
    leaderboard = []
    
    for combo_name, filters in combos.items():
        print(f"\n  Testing: {combo_name}...")
        grand_correct = 0
        grand_total = 0
        yearly = {}
        
        for year in years:
            train_start = f"{year-10}-01-01"
            train_end = f"{year-1}-12-31"
            test_start = f"{year}-01-01"
            test_end = f"{year}-12-31"
            
            df_train = df_full_feat[(df_full_feat['date'] >= train_start) & (df_full_feat['date'] <= train_end)].copy()
            df_test = df_full_feat[(df_full_feat['date'] >= test_start) & (df_full_feat['date'] <= test_end)].copy()
            
            if len(df_train) < 1000 or len(df_test) < 20: continue
            
            ensemble = EnsembleClassifier()
            ensemble.train(df_train, feature_cols, verbose=False)
            
            result = test_accuracy(df_full_feat, df_test, ensemble, filters, combo_name)
            yearly[year] = result
            grand_correct += result['correct']
            grand_total += result['total']
        
        grand_acc = (grand_correct / grand_total * 100) if grand_total > 0 else 0
        leaderboard.append({
            "name": combo_name, "accuracy": grand_acc, 
            "correct": grand_correct, "total": grand_total,
            "yearly": yearly
        })
        print(f"    → {grand_correct}/{grand_total} = {grand_acc:.1f}%")
    
    # Sort by accuracy
    leaderboard.sort(key=lambda x: x['accuracy'], reverse=True)
    
    # ============================================================
    # GENERATE REPORT
    # ============================================================
    print("\n\n🚀 GENERATING OPTIMIZER REPORT...")
    md = ["# 🔬 David Oracle: Accuracy Optimizer Report"]
    md.append("\nTested **" + str(len(combos)) + " filter combinations** across 2021-2025.")
    md.append("Each combo adds extra filters ON TOP of David's base AI verdict.")
    
    md.append("\n## 🏆 Leaderboard (Ranked by 5-Year Accuracy)")
    md.append("| Rank | Strategy | Signals | Correct | Accuracy |")
    md.append("|:---:|:---|:---:|:---:|:---:|")
    
    for i, entry in enumerate(leaderboard):
        emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i+1}."
        md.append(f"| {emoji} | {entry['name']} | {entry['total']} | {entry['correct']} | **{entry['accuracy']:.1f}%** |")
    
    # Top 3 detailed breakdown
    md.append("\n---\n## 📊 Top 3: Year-by-Year Breakdown")
    for i, entry in enumerate(leaderboard[:3]):
        rank = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
        md.append(f"\n### {rank} {entry['name']}")
        md.append("| Year | Signals | Correct | Accuracy |")
        md.append("|:---|:---:|:---:|:---:|")
        for year in years:
            if year in entry['yearly']:
                r = entry['yearly'][year]
                md.append(f"| {year} | {r['total']} | {r['correct']} | {r['accuracy']:.1f}% |")
    
    report = "\n".join(md)
    with open("accuracy_optimizer.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    print("\n✅ Optimizer Complete.")
    print(report)

if __name__ == "__main__":
    run_optimizer()
