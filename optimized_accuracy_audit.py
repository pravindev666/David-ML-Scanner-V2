"""
DAVID ORACLE — Optimized Accuracy Audit
=========================================
Tests 3 FREE accuracy upgrades over the last 1 year:
  1. Tuned Hyperparameters (lower learning rate, more regularization, fewer trees)
  2. Feature Selection (top 25 by importance, drop noisy ones)
  3. Volatility-Adjusted Target Labels (dynamic threshold based on realized vol)
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
from utils import UP, DOWN, SIDEWAYS, C

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None
try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None


def classify_regime(row):
    adx = row.get('adx', 20)
    vol = row.get('realized_vol_20', 0.15)
    if adx > 25: return "TRENDING"
    elif vol > 0.25: return "VOLATILE"
    else: return "CHOPPY"


def build_tuned_models():
    """Optimized hyperparameters for financial time series."""
    models = {}
    
    if XGBClassifier:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,       # fewer to reduce overfitting
            max_depth=4,            # shallower trees (was 6)
            learning_rate=0.03,     # slower learning (was 0.05)
            subsample=0.7,          # more randomness
            colsample_bytree=0.6,   # use fewer features per tree
            reg_alpha=0.5,          # more L1 regularization (was 0.1)
            reg_lambda=3.0,         # more L2 regularization (was 1.0)
            min_child_weight=10,    # larger min samples (was 5)
            gamma=0.1,             # minimum loss reduction
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            use_label_encoder=False,
            random_state=42,
            verbosity=0,
        )
    
    if LGBMClassifier:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.7,
            colsample_bytree=0.6,
            reg_alpha=0.5,
            reg_lambda=3.0,
            min_child_samples=30,   # larger (was 20)
            num_class=3,
            objective="multiclass",
            metric="multi_logloss",
            random_state=42,
            verbose=-1,
        )
    
    if CatBoostClassifier:
        models["CatBoost"] = CatBoostClassifier(
            iterations=200,
            depth=4,
            learning_rate=0.03,
            l2_leaf_reg=5.0,        # more regularization (was 3.0)
            loss_function="MultiClass",
            classes_count=3,
            random_seed=42,
            verbose=0,
        )
    
    return models


def select_top_features(df_train, feature_cols, n_top=25):
    """Train a quick model and return top N features by importance."""
    X = df_train[feature_cols].values
    y = df_train["target"].values.astype(int)
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    
    model = XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        objective="multi:softprob", num_class=3, verbosity=0, random_state=42
    ) if XGBClassifier else LGBMClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        objective="multiclass", verbose=-1, random_state=42
    )
    model.fit(X_s, y)
    
    imp = pd.Series(model.feature_importances_, index=feature_cols)
    top = imp.nlargest(n_top).index.tolist()
    print(f"    Selected {len(top)} features from {len(feature_cols)}")
    print(f"    Top 5: {top[:5]}")
    return top


def engineer_with_vol_adjusted_targets(df_raw, horizon=1):
    """Engineer features with volatility-adjusted direction thresholds."""
    df, feature_cols = engineer_features(df_raw, target_horizon=horizon)
    
    # Override the fixed threshold with a volatility-adjusted one
    # Instead of fixed ±0.3%, use ±(0.3 * realized_vol / median_vol)
    if "realized_vol_10" in df.columns:
        vol = df["realized_vol_10"].copy()
        median_vol = vol.median()
        # Dynamic threshold: scales with volatility
        # In calm markets (vol=0.10), threshold is ~0.15% → more decisive
        # In volatile markets (vol=0.25), threshold is ~0.75% → needs bigger move to count
        dynamic_threshold = 0.003 * (vol / max(median_vol, 0.001))
        dynamic_threshold = dynamic_threshold.clip(0.001, 0.015)  # between 0.1% and 1.5%
    else:
        dynamic_threshold = 0.003
    
    # Recompute targets with dynamic threshold
    future_return = df["close"].shift(-horizon) / df["close"] - 1
    df["future_return"] = future_return
    
    df["target"] = np.where(
        future_return > dynamic_threshold, 0,   # UP
        np.where(future_return < -dynamic_threshold, 1, 2)  # DOWN, SIDEWAYS
    )
    df["target_label"] = df["target"].map({0: UP, 1: DOWN, 2: SIDEWAYS})
    
    # Re-clean
    df = df.dropna(subset=feature_cols + ["target"]).reset_index(drop=True)
    
    return df, feature_cols


def train_and_predict(df_train, df_test, feature_cols, use_tuned=False, use_feature_select=False):
    """Train models and return predictions + confidence."""
    
    if use_feature_select:
        feature_cols = select_top_features(df_train, feature_cols, n_top=25)
    
    X_train = df_train[feature_cols].values
    y_train = df_train["target"].values.astype(int)
    X_test = df_test[feature_cols].values
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    if use_tuned:
        models = build_tuned_models()
    else:
        # Default models (same as current production)
        models = {}
        if XGBClassifier:
            models["XGBoost"] = XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                min_child_weight=5, objective="multi:softprob", num_class=3,
                eval_metric="mlogloss", use_label_encoder=False, random_state=42, verbosity=0
            )
        if LGBMClassifier:
            models["LightGBM"] = LGBMClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                min_child_samples=20, num_class=3, objective="multiclass",
                metric="multi_logloss", random_state=42, verbose=-1
            )
        if CatBoostClassifier:
            models["CatBoost"] = CatBoostClassifier(
                iterations=300, depth=6, learning_rate=0.05, l2_leaf_reg=3.0,
                loss_function="MultiClass", classes_count=3, random_seed=42, verbose=0
            )
    
    # Train all models
    for name, model in models.items():
        model.fit(X_train_s, y_train)
    
    # Predict with soft voting
    all_probs = []
    for name, model in models.items():
        probs = model.predict_proba(X_test_s)
        all_probs.append(probs)
    
    avg_probs = np.mean(all_probs, axis=0)
    predictions = np.argmax(avg_probs, axis=1)
    confidences = np.max(avg_probs, axis=1)
    
    return predictions, confidences, feature_cols


def run_optimized_audit():
    print("=" * 60)
    print("  OPTIMIZED ACCURACY AUDIT (Last 1 Year)")
    print("  Testing: Tuned HPs + Feature Selection + Vol-Adjusted Targets")
    print("=" * 60)
    
    df_raw = load_all_data()
    
    configs = {
        "A: Current Production (1-Day, Default)": {
            "vol_targets": False, "tuned": False, "feat_select": False, "regime": True
        },
        "B: + Tuned Hyperparams": {
            "vol_targets": False, "tuned": True, "feat_select": False, "regime": True
        },
        "C: + Feature Selection (Top 25)": {
            "vol_targets": False, "tuned": True, "feat_select": True, "regime": True
        },
        "D: + Vol-Adjusted Targets": {
            "vol_targets": True, "tuned": True, "feat_select": True, "regime": True
        },
        "E: All Upgrades + Conf>60": {
            "vol_targets": True, "tuned": True, "feat_select": True, "regime": True,
            "min_conf_override": 60
        },
    }
    
    all_results = {}
    
    for config_name, cfg in configs.items():
        print(f"\n{'='*60}")
        print(f"  {config_name}")
        print(f"{'='*60}")
        
        min_conf = cfg.get("min_conf_override", 50)
        
        if cfg["vol_targets"]:
            df_feat, feature_cols = engineer_with_vol_adjusted_targets(df_raw, horizon=1)
        else:
            df_feat, feature_cols = engineer_features(df_raw, target_horizon=1)
        
        train_end = "2025-03-01"
        df_train = df_feat[df_feat['date'] < train_end].copy()
        df_test = df_feat[df_feat['date'] >= train_end].copy()
        
        print(f"  Train: {len(df_train)} | Test: {len(df_test)}")
        
        if cfg["regime"]:
            # regime-specific
            regime_preds = {}
            regime_confs = {}
            used_features = feature_cols
            
            for regime in ["TRENDING", "CHOPPY", "VOLATILE"]:
                df_regime_train = df_train[df_train.apply(classify_regime, axis=1) == regime].copy()
                if len(df_regime_train) < 200:
                    df_regime_train = df_train.copy()
                
                df_regime_test = df_test[df_test.apply(classify_regime, axis=1) == regime].copy()
                if len(df_regime_test) == 0:
                    continue
                
                preds, confs, used_features = train_and_predict(
                    df_regime_train, df_regime_test, feature_cols,
                    use_tuned=cfg["tuned"], use_feature_select=cfg["feat_select"]
                )
                for i, idx in enumerate(df_regime_test.index):
                    regime_preds[idx] = preds[i]
                    regime_confs[idx] = confs[i]
        
        correct = 0
        total = 0
        
        for i in range(len(df_test)):
            idx = df_test.index[i]
            row = df_test.iloc[i]
            date = row['date']
            price_now = row['close']
            actual_target = int(row['target'])
            
            if cfg["regime"]:
                if idx not in regime_preds:
                    continue
                pred_class = regime_preds[idx]
                conf = regime_confs[idx] * 100
            else:
                continue
            
            if conf < min_conf:
                continue
            
            # Check actual 1-day outcome
            full_idx_pos = df_feat.index.get_loc(idx)
            if full_idx_pos + 1 >= len(df_feat):
                continue
            
            future_close = df_feat.iloc[full_idx_pos + 1]['close']
            pct_change = (future_close - price_now) / price_now * 100
            
            verdict = {0: UP, 1: DOWN, 2: SIDEWAYS}[pred_class]
            
            is_correct = False
            if verdict == UP and pct_change > 0:
                is_correct = True
            elif verdict == DOWN and pct_change < 0:
                is_correct = True
            elif verdict == SIDEWAYS and abs(pct_change) < 0.5:
                is_correct = True
            
            total += 1
            if is_correct:
                correct += 1
        
        accuracy = (correct / total * 100) if total > 0 else 0
        all_results[config_name] = {"accuracy": accuracy, "correct": correct, "total": total}
        print(f"  >>> {correct}/{total} = {accuracy:.1f}%")
    
    # ============================================================
    # LEADERBOARD
    # ============================================================
    print("\n\n" + "=" * 60)
    print("  LEADERBOARD")
    print("=" * 60)
    
    md = ["# Optimized Accuracy Audit: Free Upgrades"]
    md.append("\nTest period: **Mar 2025 — Mar 2026** (1 year)")
    md.append("All upgrades are FREE — no paid data required.\n")
    md.append("## Leaderboard")
    md.append("| Rank | Configuration | Signals | Correct | Accuracy |")
    md.append("|:---:|:---|:---:|:---:|:---:|")
    
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    for i, (name, res) in enumerate(sorted_results):
        rank = f"{i+1}."
        emoji = "\U0001f947" if i == 0 else "\U0001f948" if i == 1 else "\U0001f949" if i == 2 else ""
        print(f"  {rank} {name}: {res['accuracy']:.1f}%")
        md.append(f"| {emoji} {rank} | {name} | {res['total']} | {res['correct']} | **{res['accuracy']:.1f}%** |")
    
    report = "\n".join(md)
    with open("optimized_accuracy_audit.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"\nReport saved to optimized_accuracy_audit.md")
    print(report)


if __name__ == "__main__":
    run_optimized_audit()
