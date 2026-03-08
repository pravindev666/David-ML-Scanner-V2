
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

def run_accuracy_audit(years=[2021, 2022, 2023, 2024, 2025]):
    """
    Pure accuracy test: Does David's verdict (UP/DOWN/SIDEWAYS) match 
    what the market ACTUALLY did over the next 5 trading days?
    
    No P/L. No spreads. Just: Was David RIGHT or WRONG?
    
    "Right" means:
      - David said UP   → Nifty closed HIGHER after 5 days
      - David said DOWN → Nifty closed LOWER after 5 days
      - David said SIDEWAYS → Nifty stayed within ±1% after 5 days
    """
    print("🔍 PURE AI SIGNAL ACCURACY AUDIT")
    print("=" * 60)
    
    df_raw = load_all_data()
    df_full_feat, feature_cols = engineer_features(df_raw)
    
    # Style filters
    styles = {
        "Golden (Conf > 60%)": {"min_conf": 60, "use_whipsaw": True, "max_whipsaw": 35},
        "Greedy (Conf > 40%)": {"min_conf": 40, "use_whipsaw": True, "max_whipsaw": 45},
        "Gambler (Conf > 40%, No Filter)": {"min_conf": 40, "use_whipsaw": False, "max_whipsaw": 100},
    }
    
    # Results: style -> year -> {correct, wrong, total, by_verdict}
    all_results = {}
    
    for style_name, style_config in styles.items():
        print(f"\n{'='*60}")
        print(f"TESTING: {style_name}")
        print(f"{'='*60}")
        
        all_results[style_name] = {}
        
        for year in years:
            print(f"\n  Year {year}...")
            
            # Train on 10 years prior
            train_end = f"{year-1}-12-31"
            train_start = f"{year-10}-01-01"
            test_start = f"{year}-01-01"
            test_end = f"{year}-12-31"
            
            df_train = df_full_feat[(df_full_feat['date'] >= train_start) & (df_full_feat['date'] <= train_end)].copy()
            df_test = df_full_feat[(df_full_feat['date'] >= test_start) & (df_full_feat['date'] <= test_end)].copy()
            
            if len(df_train) < 1000 or len(df_test) < 20:
                print(f"  Skipping {year} — insufficient data")
                continue
            
            ensemble = EnsembleClassifier()
            ensemble.train(df_train, feature_cols, verbose=False)
            
            correct = 0
            wrong = 0
            total = 0
            skipped = 0
            
            # Track per-verdict accuracy
            verdict_stats = {
                UP: {"correct": 0, "wrong": 0, "total": 0},
                DOWN: {"correct": 0, "wrong": 0, "total": 0},
                SIDEWAYS: {"correct": 0, "wrong": 0, "total": 0},
            }
            
            for i in range(len(df_test)):
                row = df_test.iloc[i]
                date = row['date']
                price_now = row['close']
                
                pred = ensemble.predict(row)
                verdict = pred['direction']
                conf = pred['confidence'] * 100
                
                # Whipsaw check
                adx = row.get('adx', 25)
                bb_width_pct = row.get('bb_width', 0)
                whipsaw_prob = 0
                if adx < 20: whipsaw_prob += 40
                if bb_width_pct < df_full_feat['bb_width'].quantile(0.2): whipsaw_prob += 30
                
                # Apply style filter
                if conf < style_config["min_conf"]:
                    skipped += 1
                    continue
                if style_config["use_whipsaw"] and whipsaw_prob > style_config["max_whipsaw"]:
                    skipped += 1
                    continue
                
                # Check actual market outcome over next 5 days
                full_idx = df_full_feat.index[df_full_feat['date'] == date][0]
                future = df_full_feat.iloc[full_idx + 1 : full_idx + 6]  # 5 trading days
                if len(future) < 3:
                    skipped += 1
                    continue
                
                future_close = future.iloc[-1]['close']
                pct_change = (future_close - price_now) / price_now * 100
                
                # Was David right?
                is_correct = False
                if verdict == UP and pct_change > 0:
                    is_correct = True
                elif verdict == DOWN and pct_change < 0:
                    is_correct = True
                elif verdict == SIDEWAYS and abs(pct_change) < 1.0:
                    is_correct = True
                
                total += 1
                verdict_stats[verdict]["total"] += 1
                if is_correct:
                    correct += 1
                    verdict_stats[verdict]["correct"] += 1
                else:
                    wrong += 1
                    verdict_stats[verdict]["wrong"] += 1
            
            accuracy = (correct / total * 100) if total > 0 else 0
            all_results[style_name][year] = {
                "correct": correct, "wrong": wrong, "total": total,
                "skipped": skipped, "accuracy": accuracy,
                "verdict_stats": verdict_stats
            }
            print(f"  → {year}: {correct}/{total} correct = {accuracy:.1f}% (Skipped: {skipped})")
    
    # ============================================================
    # GENERATE REPORT
    # ============================================================
    print("\n\n🚀 GENERATING ACCURACY REPORT...")
    md = ["# 🔍 David Oracle: AI Signal Accuracy Audit"]
    md.append("\nThis audit answers ONE question: **When David says UP, DOWN, or SIDEWAYS, is he RIGHT?**")
    md.append("\n**Method**: For each trading day, David makes a prediction. We check the actual Nifty close 5 trading days later.")
    md.append("- **UP correct** = Nifty closed higher after 5 days")
    md.append("- **DOWN correct** = Nifty closed lower after 5 days")
    md.append("- **SIDEWAYS correct** = Nifty stayed within ±1% after 5 days")
    
    for style_name in all_results:
        md.append(f"\n---\n## 📈 {style_name}")
        md.append("| Year | Signals | Correct | Wrong | Accuracy |")
        md.append("|:---|:---:|:---:|:---:|:---:|")
        
        grand_correct = 0
        grand_total = 0
        
        for year in years:
            if year not in all_results[style_name]: continue
            r = all_results[style_name][year]
            emoji = "✅" if r['accuracy'] >= 55 else "⚠️" if r['accuracy'] >= 50 else "❌"
            md.append(f"| {year} | {r['total']} | {r['correct']} | {r['wrong']} | {emoji} **{r['accuracy']:.1f}%** |")
            grand_correct += r['correct']
            grand_total += r['total']
        
        grand_acc = (grand_correct / grand_total * 100) if grand_total > 0 else 0
        md.append(f"| **5-Year Total** | **{grand_total}** | **{grand_correct}** | **{grand_total - grand_correct}** | **{grand_acc:.1f}%** |")
        
        # Per-verdict breakdown for last available year
        last_year = years[-1]
        if last_year in all_results[style_name]:
            vs = all_results[style_name][last_year]['verdict_stats']
            md.append(f"\n**{last_year} Verdict Breakdown:**")
            md.append("| Verdict | Signals | Correct | Accuracy |")
            md.append("|:---|:---:|:---:|:---:|")
            for v_name in [UP, DOWN, SIDEWAYS]:
                v = vs[v_name]
                v_acc = (v['correct'] / v['total'] * 100) if v['total'] > 0 else 0
                label = "UP 📈" if v_name == UP else "DOWN 📉" if v_name == DOWN else "SIDEWAYS ↔️"
                md.append(f"| {label} | {v['total']} | {v['correct']} | {v_acc:.1f}% |")
    
    report = "\n".join(md)
    with open("accuracy_audit.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    print("\n✅ Accuracy Audit Complete.")
    print(report)

if __name__ == "__main__":
    run_accuracy_audit()
