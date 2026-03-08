"""
DAVID ORACLE — V2 Backtest (Train 2011-2024, Test 2025-2026)
==============================================================
Trains all models on data up to Dec 2024.
Tests predictions on Jan 2025 — Mar 2026 (fully out-of-sample).
Reports accuracy, avg holding days, and spread strategy performance.
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


def run_v2_backtest():
    print("=" * 65)
    print("  DAVID V2 BACKTEST")
    print("  Train: 2011 — Dec 2024 | Test: Jan 2025 — Mar 2026")
    print("  Architecture: Regime Trees + LSTM Hybrid")
    print("=" * 65)

    df_raw = load_all_data()
    df, feature_cols = engineer_features(df_raw, target_horizon=1)

    train_end = "2025-01-01"
    df_train = df[df['date'] < train_end].copy()
    df_test = df[df['date'] >= train_end].copy()

    print(f"  Train: {len(df_train)} samples ({df_train['date'].min().strftime('%Y-%m-%d')} → {df_train['date'].max().strftime('%Y-%m-%d')})")
    print(f"  Test:  {len(df_test)} samples ({df_test['date'].min().strftime('%Y-%m-%d')} → {df_test['date'].max().strftime('%Y-%m-%d')})")

    # ── Train Regime-Specific Trees ──
    print(f"\n{'='*65}")
    print("  Training Regime-Specific Tree Ensembles...")
    print(f"{'='*65}")

    regime_models = {}
    for regime in ["TRENDING", "CHOPPY", "VOLATILE"]:
        df_regime = df_train[df_train.apply(classify_regime, axis=1) == regime].copy()
        if len(df_regime) < 200:
            df_regime = df_train.copy()
        m = EnsembleClassifier()
        m.train(df_regime, feature_cols, verbose=False)
        regime_models[regime] = m
        print(f"    {regime}: {len(df_regime)} samples")

    # ── Train LSTM ──
    print(f"\n{'='*65}")
    print("  Training LSTM Sequence Model...")
    print(f"{'='*65}")

    lstm = LSTMClassifier(seq_len=10, hidden_size=64, num_layers=2, lr=0.001, epochs=80, batch_size=64)
    lstm.train(df_train, feature_cols, verbose=True)

    # ── Get LSTM predictions ──
    lstm_preds, lstm_probs = lstm.predict(df_test)
    lstm_offset = lstm.seq_len

    # ── Evaluate ──
    print(f"\n{'='*65}")
    print("  Evaluating on 2025-2026 Test Set...")
    print(f"{'='*65}")

    results = {
        "Trees (Regime)": {"correct": 0, "total": 0, "trades": []},
        "LSTM Only": {"correct": 0, "total": 0, "trades": []},
        "Hybrid (Trees+LSTM)": {"correct": 0, "total": 0, "trades": []},
    }

    for i in range(lstm_offset, len(df_test)):
        row = df_test.iloc[i]
        date = row['date']
        price_now = row['close']
        regime = classify_regime(row)

        # Future 1-day return
        full_idx = df.index[df['date'] == date]
        if len(full_idx) == 0: continue
        full_pos = df.index.get_loc(full_idx[0])
        if full_pos + 1 >= len(df): continue

        future_close = df.iloc[full_pos + 1]['close']
        pct_change = (future_close - price_now) / price_now * 100

        # Tree prediction
        tree_pred = regime_models.get(regime, regime_models["TRENDING"]).predict(row)
        tree_dir = tree_pred['direction']
        tree_conf = tree_pred['confidence']
        tree_probs = np.array([tree_pred['prob_up'], tree_pred['prob_down'], tree_pred['prob_sideways']])

        # LSTM prediction
        lstm_idx = i - lstm_offset
        lstm_prob = lstm_probs[lstm_idx]
        lstm_class = lstm_preds[lstm_idx]
        lstm_dir = {0: UP, 1: DOWN, 2: SIDEWAYS}[lstm_class]
        lstm_conf = float(lstm_prob[lstm_class])

        # Hybrid
        hybrid_probs = (tree_probs + lstm_prob) / 2.0
        hybrid_class = int(np.argmax(hybrid_probs))
        hybrid_dir = {0: UP, 1: DOWN, 2: SIDEWAYS}[hybrid_class]
        hybrid_conf = float(hybrid_probs[hybrid_class])

        def check(verdict, pct):
            if verdict == UP and pct > 0: return True
            if verdict == DOWN and pct < 0: return True
            if verdict == SIDEWAYS and abs(pct) < 0.5: return True
            return False

        # Trees
        if tree_conf * 100 >= 40:
            c = check(tree_dir, pct_change)
            results["Trees (Regime)"]["total"] += 1
            if c: results["Trees (Regime)"]["correct"] += 1
            results["Trees (Regime)"]["trades"].append({
                "date": date, "dir": tree_dir, "conf": tree_conf * 100,
                "pct": pct_change, "correct": c, "regime": regime
            })

        # LSTM
        if lstm_conf * 100 >= 40:
            c = check(lstm_dir, pct_change)
            results["LSTM Only"]["total"] += 1
            if c: results["LSTM Only"]["correct"] += 1
            results["LSTM Only"]["trades"].append({
                "date": date, "dir": lstm_dir, "conf": lstm_conf * 100,
                "pct": pct_change, "correct": c, "regime": regime
            })

        # Hybrid
        if hybrid_conf * 100 >= 40:
            c = check(hybrid_dir, pct_change)
            results["Hybrid (Trees+LSTM)"]["total"] += 1
            if c: results["Hybrid (Trees+LSTM)"]["correct"] += 1
            results["Hybrid (Trees+LSTM)"]["trades"].append({
                "date": date, "dir": hybrid_dir, "conf": hybrid_conf * 100,
                "pct": pct_change, "correct": c, "regime": regime
            })

    # ── REPORT ──
    print("\n\n" + "=" * 65)
    print("  BACKTEST RESULTS: Jan 2025 — Mar 2026")
    print("=" * 65)

    md = ["# David V2 Backtest Report"]
    md.append(f"\n**Train Period**: 2011 — Dec 2024")
    md.append(f"**Test Period**: Jan 2025 — Mar 2026 (fully out-of-sample)")
    md.append(f"**Architecture**: Regime-Specific Trees + LSTM Hybrid\n")

    md.append("## Accuracy Leaderboard")
    md.append("| Model | Signals | Correct | Accuracy |")
    md.append("|:---|:---:|:---:|:---:|")

    sorted_r = sorted(results.items(), key=lambda x: x[1]['correct'] / max(1, x[1]['total']), reverse=True)

    for i, (name, r) in enumerate(sorted_r):
        acc = (r['correct'] / max(1, r['total'])) * 100
        emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
        print(f"  {emoji} {name}: {r['correct']}/{r['total']} = {acc:.1f}%")
        md.append(f"| {emoji} {name} | {r['total']} | {r['correct']} | **{acc:.1f}%** |")

    # ── Spread Strategy Analysis ──
    md.append("\n## Spread Strategy Simulation")
    md.append("*Based on taking Bull/Bear spreads on directional signals*\n")

    best_model = sorted_r[0][0]
    trades = sorted_r[0][1]['trades']

    # Simulate spread P&L
    wins = [t for t in trades if t['correct']]
    losses = [t for t in trades if not t['correct']]

    avg_win_pct = np.mean([abs(t['pct']) for t in wins]) if wins else 0
    avg_loss_pct = np.mean([abs(t['pct']) for t in losses]) if losses else 0

    # Spread assumptions
    spread_premium_collected = 50  # avg Nifty spread premium per lot
    spread_max_loss = 150          # max loss on spread per lot

    spread_wins = len(wins)
    spread_losses = len(losses)
    total_pnl = (spread_wins * spread_premium_collected) - (spread_losses * spread_max_loss)
    pnl_per_lot = total_pnl / max(1, len(trades))

    md.append(f"| Metric | Value |")
    md.append(f"|:---|:---|")
    md.append(f"| Best Model | **{best_model}** |")
    md.append(f"| Total Signals | {len(trades)} |")
    md.append(f"| Winning Days | {spread_wins} ({spread_wins/max(1,len(trades))*100:.0f}%) |")
    md.append(f"| Losing Days | {spread_losses} ({spread_losses/max(1,len(trades))*100:.0f}%) |")
    md.append(f"| Avg Move (Win) | {avg_win_pct:.2f}% |")
    md.append(f"| Avg Move (Loss) | {avg_loss_pct:.2f}% |")
    md.append(f"| Estimated P&L (per lot) | **₹{total_pnl:,.0f}** |")
    md.append(f"| Avg P&L per Trade | ₹{pnl_per_lot:,.0f} |")

    print(f"\n  Spread P&L ({best_model}):")
    print(f"    Wins: {spread_wins} | Losses: {spread_losses}")
    print(f"    Estimated P&L/lot: ₹{total_pnl:,.0f}")

    # ── By Regime Breakdown ──
    md.append("\n## Accuracy by Regime")
    md.append("| Regime | Signals | Accuracy |")
    md.append("|:---|:---:|:---:|")

    for regime in ["TRENDING", "CHOPPY", "VOLATILE"]:
        rt = [t for t in trades if t['regime'] == regime]
        rw = [t for t in rt if t['correct']]
        acc = len(rw) / max(1, len(rt)) * 100
        md.append(f"| {regime} | {len(rt)} | **{acc:.0f}%** |")
        print(f"    {regime}: {len(rw)}/{len(rt)} = {acc:.0f}%")

    # ── By Confidence Tier ──
    md.append("\n## Accuracy by Confidence Tier")
    md.append("| Confidence | Signals | Accuracy |")
    md.append("|:---|:---:|:---:|")

    for lo, hi, label in [(40, 50, "40-50%"), (50, 60, "50-60%"), (60, 100, "60%+")]:
        ct = [t for t in trades if lo <= t['conf'] < hi]
        cw = [t for t in ct if t['correct']]
        acc = len(cw) / max(1, len(ct)) * 100
        md.append(f"| {label} | {len(ct)} | **{acc:.0f}%** |")

    report = "\n".join(md)
    out_path = os.path.join(BASE_DIR, "backtest_v2_2025_2026.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n  Report saved to: {out_path}")
    print(report)


if __name__ == "__main__":
    run_v2_backtest()
