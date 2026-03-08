"""
DAVID ORACLE — LSTM Accuracy Audit
====================================
Tests the LSTM sequence model against the tree-ensemble baseline.
Also tests a hybrid (LSTM + Trees averaged) approach.
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from data_engine import load_all_data
from feature_forge import engineer_features
from models.ensemble_classifier import EnsembleClassifier
from models.lstm_classifier import LSTMClassifier
from utils import UP, DOWN, SIDEWAYS


def classify_regime(row):
    adx = row.get('adx', 20)
    vol = row.get('realized_vol_20', 0.15)
    if adx > 25: return "TRENDING"
    elif vol > 0.25: return "VOLATILE"
    else: return "CHOPPY"


def run_lstm_audit():
    print("=" * 60)
    print("  LSTM ACCURACY AUDIT (Last 1 Year)")
    print("  Tree Ensemble vs LSTM vs Hybrid")
    print("=" * 60)
    
    df_raw = load_all_data()
    df_feat, feature_cols = engineer_features(df_raw, target_horizon=1)
    
    train_end = "2025-03-01"
    df_train = df_feat[df_feat['date'] < train_end].copy()
    df_test = df_feat[df_feat['date'] >= train_end].copy()
    
    print(f"  Train: {len(df_train)} | Test: {len(df_test)}")
    
    # ============================================================
    # 1. Train Tree Ensemble (regime-specific, current production)
    # ============================================================
    print(f"\n{'='*60}")
    print("  Training: Tree Ensemble (Regime-Specific)")
    print(f"{'='*60}")
    
    tree_models = {}
    for regime in ["TRENDING", "CHOPPY", "VOLATILE"]:
        df_regime = df_train[df_train.apply(classify_regime, axis=1) == regime].copy()
        if len(df_regime) < 200:
            df_regime = df_train.copy()
        m = EnsembleClassifier()
        m.train(df_regime, feature_cols, verbose=False)
        tree_models[regime] = m
        print(f"    {regime}: {len(df_regime)} samples")
    
    # ============================================================
    # 2. Train LSTM
    # ============================================================
    print(f"\n{'='*60}")
    print("  Training: LSTM Sequence Model")
    print(f"{'='*60}")
    
    lstm = LSTMClassifier(seq_len=10, hidden_size=64, num_layers=2, lr=0.001, epochs=80, batch_size=64)
    lstm.train(df_train, feature_cols, verbose=True)
    
    # ============================================================
    # 3. Evaluate all approaches
    # ============================================================
    print(f"\n{'='*60}")
    print("  Evaluating...")
    print(f"{'='*60}")
    
    # Get LSTM predictions for entire test set
    lstm_preds, lstm_probs = lstm.predict(df_test)
    # lstm_preds starts from index seq_len in df_test
    lstm_offset = lstm.seq_len
    
    results = {
        "Trees (Regime)": {"correct": 0, "total": 0},
        "LSTM Only": {"correct": 0, "total": 0},
        "Hybrid (Trees+LSTM avg)": {"correct": 0, "total": 0},
        "Hybrid, Conf>55": {"correct": 0, "total": 0},
        "Hybrid, Conf>60": {"correct": 0, "total": 0},
    }
    
    for i in range(lstm_offset, len(df_test)):
        row = df_test.iloc[i]
        date = row['date']
        price_now = row['close']
        regime = classify_regime(row)
        
        # Check actual 1-day outcome
        full_idx = df_feat.index[df_feat['date'] == date]
        if len(full_idx) == 0:
            continue
        full_idx = full_idx[0]
        full_pos = df_feat.index.get_loc(full_idx)
        if full_pos + 1 >= len(df_feat):
            continue
        
        future_close = df_feat.iloc[full_pos + 1]['close']
        pct_change = (future_close - price_now) / price_now * 100
        
        # Tree prediction
        tree_pred = tree_models.get(regime, tree_models["TRENDING"]).predict(row)
        tree_direction = tree_pred['direction']
        tree_conf = tree_pred['confidence']
        tree_probs_arr = np.array([tree_pred['prob_up'], tree_pred['prob_down'], tree_pred['prob_sideways']])
        
        # LSTM prediction
        lstm_idx = i - lstm_offset
        lstm_prob = lstm_probs[lstm_idx]
        lstm_pred_class = lstm_preds[lstm_idx]
        lstm_direction = {0: UP, 1: DOWN, 2: SIDEWAYS}[lstm_pred_class]
        lstm_conf = float(lstm_prob[lstm_pred_class])
        
        # Hybrid: average tree and LSTM probabilities
        hybrid_probs = (tree_probs_arr + lstm_prob) / 2.0
        hybrid_class = int(np.argmax(hybrid_probs))
        hybrid_direction = {0: UP, 1: DOWN, 2: SIDEWAYS}[hybrid_class]
        hybrid_conf = float(hybrid_probs[hybrid_class])
        
        def check_correct(verdict, pct):
            if verdict == UP and pct > 0: return True
            if verdict == DOWN and pct < 0: return True
            if verdict == SIDEWAYS and abs(pct) < 0.5: return True
            return False
        
        # Trees
        if tree_conf * 100 >= 50:
            results["Trees (Regime)"]["total"] += 1
            if check_correct(tree_direction, pct_change):
                results["Trees (Regime)"]["correct"] += 1
        
        # LSTM only
        if lstm_conf * 100 >= 50:
            results["LSTM Only"]["total"] += 1
            if check_correct(lstm_direction, pct_change):
                results["LSTM Only"]["correct"] += 1
        
        # Hybrid
        if hybrid_conf * 100 >= 50:
            results["Hybrid (Trees+LSTM avg)"]["total"] += 1
            if check_correct(hybrid_direction, pct_change):
                results["Hybrid (Trees+LSTM avg)"]["correct"] += 1
        
        if hybrid_conf * 100 >= 55:
            results["Hybrid, Conf>55"]["total"] += 1
            if check_correct(hybrid_direction, pct_change):
                results["Hybrid, Conf>55"]["correct"] += 1
                
        if hybrid_conf * 100 >= 60:
            results["Hybrid, Conf>60"]["total"] += 1
            if check_correct(hybrid_direction, pct_change):
                results["Hybrid, Conf>60"]["correct"] += 1
    
    # ============================================================
    # LEADERBOARD
    # ============================================================
    print("\n\n" + "=" * 60)
    print("  LEADERBOARD")
    print("=" * 60)
    
    md = ["# LSTM Accuracy Audit"]
    md.append("\nTest period: **Mar 2025 - Mar 2026** (1 year)")
    md.append("Comparing Tree Ensemble vs LSTM vs Hybrid.\n")
    md.append("## Leaderboard")
    md.append("| Rank | Model | Signals | Correct | Accuracy |")
    md.append("|:---:|:---|:---:|:---:|:---:|")
    
    sorted_results = sorted(
        results.items(), 
        key=lambda x: (x[1]['correct'] / max(1, x[1]['total'])), 
        reverse=True
    )
    
    for i, (name, res) in enumerate(sorted_results):
        acc = (res['correct'] / max(1, res['total'])) * 100
        rank = f"{i+1}."
        emoji = "\U0001f947" if i == 0 else "\U0001f948" if i == 1 else "\U0001f949" if i == 2 else ""
        print(f"  {rank} {name}: {res['correct']}/{res['total']} = {acc:.1f}%")
        md.append(f"| {emoji} {rank} | {name} | {res['total']} | {res['correct']} | **{acc:.1f}%** |")
    
    report = "\n".join(md)
    with open("lstm_accuracy_audit.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"\nReport saved to lstm_accuracy_audit.md")
    print(report)


if __name__ == "__main__":
    run_lstm_audit()
