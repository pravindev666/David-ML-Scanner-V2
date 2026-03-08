"""
DAVID ORACLE — Enhanced Accuracy Audit
========================================
Tests TWO architectural improvements:
  1. 1-Day prediction horizon (instead of 5-day)
  2. Regime-specific models (separate models for trending vs choppy markets)
"""

import pandas as pd
import numpy as np
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from data_engine import load_all_data
from feature_forge import engineer_features
from models.ensemble_classifier import EnsembleClassifier
from utils import UP, DOWN, SIDEWAYS, DIRECTION_THRESHOLD


def engineer_features_custom(df_raw, horizon=1):
    """Re-engineer features with a custom prediction horizon."""
    from feature_forge import engineer_features as _ef
    # We call the standard pipeline but override the target
    df, feature_cols = _ef(df_raw, target_horizon=horizon)
    return df, feature_cols


def classify_regime(row):
    """Classify market regime using ADX and realized volatility."""
    adx = row.get('adx', 20)
    vol = row.get('realized_vol_20', 0.15)
    
    if adx > 25:
        return "TRENDING"
    elif vol > 0.25:
        return "VOLATILE"
    else:
        return "CHOPPY"


def run_enhanced_audit():
    print("=" * 60)
    print("  ENHANCED ACCURACY AUDIT (Last 1 Year)")
    print("  1-Day Horizon + Regime-Specific Models")
    print("=" * 60)
    
    df_raw = load_all_data()
    
    # ============================================================
    # TEST CONFIGURATIONS
    # ============================================================
    configs = {
        "Baseline: 5-Day, Single Model, Conf>50": {"horizon": 5, "regime": False, "min_conf": 50},
        "Upgrade 1: 1-Day, Single Model, Conf>50": {"horizon": 1, "regime": False, "min_conf": 50},
        "Upgrade 2: 1-Day + Regime Models, Conf>50": {"horizon": 1, "regime": True, "min_conf": 50},
        "Upgrade 3: 1-Day + Regime, Conf>55": {"horizon": 1, "regime": True, "min_conf": 55},
        "Upgrade 4: 1-Day + Regime, Conf>60": {"horizon": 1, "regime": True, "min_conf": 60},
    }
    
    all_results = {}
    
    for config_name, cfg in configs.items():
        print(f"\n{'='*60}")
        print(f"  TESTING: {config_name}")
        print(f"{'='*60}")
        
        # Engineer features with the specified horizon
        df_feat, feature_cols = engineer_features_custom(df_raw, horizon=cfg["horizon"])
        
        # Train on everything before Mar 2025, test on last ~1 year
        train_end = "2025-03-01"
        test_start = "2025-03-01"
        
        df_train = df_feat[df_feat['date'] < train_end].copy()
        df_test = df_feat[df_feat['date'] >= test_start].copy()
        
        print(f"  Train: {len(df_train)} rows | Test: {len(df_test)} rows")
        
        if cfg["regime"]:
            # Train 3 separate models for each regime
            models = {}
            for regime in ["TRENDING", "CHOPPY", "VOLATILE"]:
                df_regime_train = df_train[df_train.apply(classify_regime, axis=1) == regime].copy()
                if len(df_regime_train) < 200:
                    df_regime_train = df_train.copy()
                
                m = EnsembleClassifier()
                m.train(df_regime_train, feature_cols, verbose=False)
                models[regime] = m
                print(f"    Trained {regime} model on {len(df_regime_train)} rows")
        else:
            single_model = EnsembleClassifier()
            single_model.train(df_train, feature_cols, verbose=False)
        
        correct = 0
        total = 0
        regime_stats = {"TRENDING": [0, 0], "CHOPPY": [0, 0], "VOLATILE": [0, 0]}
        
        for i in range(len(df_test)):
            row = df_test.iloc[i]
            date = row['date']
            price_now = row['close']
            
            regime = classify_regime(row)
            
            if cfg["regime"]:
                model = models.get(regime, models.get("TRENDING"))
            else:
                model = single_model
            
            pred = model.predict(row)
            verdict = pred['direction']
            conf = pred['confidence'] * 100
            
            if conf < cfg["min_conf"]:
                continue
            
            # Check actual outcome
            full_idx = df_feat.index[df_feat['date'] == date]
            if len(full_idx) == 0:
                continue
            full_idx = full_idx[0]
            
            lookahead = cfg["horizon"]
            future = df_feat.iloc[full_idx + 1 : full_idx + 1 + lookahead]
            if len(future) < 1:
                continue
            
            future_close = future.iloc[-1]['close']
            pct_change = (future_close - price_now) / price_now * 100
            
            is_correct = False
            if verdict == UP and pct_change > 0:
                is_correct = True
            elif verdict == DOWN and pct_change < 0:
                is_correct = True
            elif verdict == SIDEWAYS and abs(pct_change) < 0.5:
                is_correct = True
            
            total += 1
            regime_stats[regime][1] += 1
            if is_correct:
                correct += 1
                regime_stats[regime][0] += 1
        
        accuracy = (correct / total * 100) if total > 0 else 0
        all_results[config_name] = {
            "accuracy": accuracy, "correct": correct, "total": total,
            "regime_stats": regime_stats
        }
        print(f"  >>> Result: {correct}/{total} = {accuracy:.1f}%")
        for r_name, (r_c, r_t) in regime_stats.items():
            if r_t > 0:
                print(f"      {r_name}: {r_c}/{r_t} = {r_c/r_t*100:.1f}%")
    
    # ============================================================
    # GENERATE REPORT
    # ============================================================
    print("\n\n" + "=" * 60)
    print("  LEADERBOARD")
    print("=" * 60)
    
    md = ["# Enhanced Accuracy Audit: 1-Day Horizon + Regime Detection"]
    md.append("\nTest period: **Mar 2025 — Mar 2026** (last 1 year)\n")
    md.append("## Leaderboard")
    md.append("| Rank | Configuration | Signals | Correct | Accuracy |")
    md.append("|:---:|:---|:---:|:---:|:---:|")
    
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    for i, (name, res) in enumerate(sorted_results):
        rank = f"{i+1}."
        emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else ""
        print(f"  {rank} {name}: {res['accuracy']:.1f}%")
        md.append(f"| {emoji} {rank} | {name} | {res['total']} | {res['correct']} | **{res['accuracy']:.1f}%** |")
    
    # Regime breakdown for winner
    winner_name, winner = sorted_results[0]
    md.append(f"\n## Winner Breakdown by Regime: {winner_name}")
    md.append("| Regime | Signals | Correct | Accuracy |")
    md.append("|:---|:---:|:---:|:---:|")
    for r_name, (r_c, r_t) in winner['regime_stats'].items():
        if r_t > 0:
            md.append(f"| {r_name} | {r_t} | {r_c} | {r_c/r_t*100:.1f}% |")
    
    report = "\n".join(md)
    with open("enhanced_accuracy_audit.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"\nReport saved to enhanced_accuracy_audit.md")
    print(report)



if __name__ == "__main__":
    run_enhanced_audit()
