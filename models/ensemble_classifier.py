"""
DAVID PROPHETIC ORACLE — Ensemble Direction Classifier
=======================================================
XGBoost + LightGBM + CatBoost soft-voting ensemble.
Outputs probability for UP / DOWN / SIDEWAYS.
Walk-forward validation built in.
"""

import numpy as np
import pandas as pd
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
    print("[WARN] xgboost not installed. Install with: pip install xgboost")

try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None
    print("[WARN] lightgbm not installed. Install with: pip install lightgbm")

try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None
    print("[WARN] catboost not installed. Install with: pip install catboost")

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import MODEL_DIR, UP, DOWN, SIDEWAYS, C


TARGET_MAP = {0: UP, 1: DOWN, 2: SIDEWAYS}
LABEL_MAP = {UP: 0, DOWN: 1, SIDEWAYS: 2}


class EnsembleClassifier:
    """
    Soft-voting ensemble of XGBoost + LightGBM + CatBoost.
    Each model outputs class probabilities.
    Final prediction = weighted average of probabilities.
    """
    
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.is_trained = False
        self.validation_scores = {}
        self.feature_importance = None
    
    def _build_models(self):
        """Initialize the 3 gradient-boosted classifiers."""
        models = {}
        
        if XGBClassifier is not None:
            models["XGBoost"] = XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                min_child_weight=5,
                objective="multi:softprob",
                num_class=3,
                eval_metric="mlogloss",
                use_label_encoder=False,
                random_state=42,
                verbosity=0,
            )
        
        if LGBMClassifier is not None:
            models["LightGBM"] = LGBMClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                min_child_samples=20,
                num_class=3,
                objective="multiclass",
                metric="multi_logloss",
                random_state=42,
                verbose=-1,
            )
        
        if CatBoostClassifier is not None:
            models["CatBoost"] = CatBoostClassifier(
                iterations=300,
                depth=6,
                learning_rate=0.05,
                l2_leaf_reg=3.0,
                loss_function="MultiClass",
                classes_count=3,
                random_seed=42,
                verbose=0,
            )
        
        if not models:
            raise RuntimeError("No ML libraries installed! Install xgboost, lightgbm, or catboost.")
        
        return models
    
    def train(self, df, feature_cols, verbose=True):
        """
        Train all models on the full dataset.
        Uses walk-forward cross-validation to estimate performance.
        """
        self.feature_cols = feature_cols
        X = df[feature_cols].values
        y = df["target"].values.astype(int)
        
        # ─── Walk-Forward Cross-Validation ────────────────────────────────
        tscv = TimeSeriesSplit(n_splits=5)
        fold_scores = {name: [] for name in ["XGBoost", "LightGBM", "CatBoost"]}
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train_raw, X_test_raw = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Scale features ONLY on training data for this fold
            fold_scaler = StandardScaler()
            X_train = fold_scaler.fit_transform(X_train_raw)
            X_test = fold_scaler.transform(X_test_raw)
            
            fold_models = self._build_models()
            
            for name, model in fold_models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                fold_scores[name].append(acc)
            
            if verbose:
                accs = " | ".join(f"{n}: {fold_scores[n][-1]:.1%}" for n in fold_scores)
                print(f"  Fold {fold+1}/5: {accs}")
        
        # Calculate model weights based on CV performance
        for name in fold_scores:
            avg = np.mean(fold_scores[name])
            self.validation_scores[name] = avg
        
        total_score = sum(self.validation_scores.values())
        if total_score > 0:
            for name in self.validation_scores:
                self.weights[name] = self.validation_scores[name] / total_score
        else:
            # Fallback to equal weights if all scores are 0
            for name in self.validation_scores:
                self.weights[name] = 1.0 / len(self.validation_scores)
        
        if verbose:
            print(f"\n  {C.CYAN}Model Weights (performance-based):{C.RESET}")
            for name, w in self.weights.items():
                print(f"    {name:>10}: {w:.3f} (CV accuracy: {self.validation_scores[name]:.1%})")
        
        # ─── Train Final Models on Full Data ──────────────────────────────
        # Use full historical scaling for the final production model
        X_scaled = self.scaler.fit_transform(X)
        
        self.models = self._build_models()
        for name, model in self.models.items():
            model.fit(X_scaled, y)
            if verbose:
                print(f"  {C.GREEN}[OK] {name} trained on full dataset{C.RESET}")
        
        # Feature importance (average across models)
        importances = np.zeros(len(feature_cols))
        for name, model in self.models.items():
            if hasattr(model, "feature_importances_"):
                imp = model.feature_importances_
                importances += imp / imp.sum()
        importances /= len(self.models)
        
        self.feature_importance = pd.DataFrame({
            "feature": feature_cols,
            "importance": importances
        }).sort_values("importance", ascending=False)
        
        self.is_trained = True
        
        # Final walk-forward accuracy
        avg_acc = np.mean([np.mean(v) for v in fold_scores.values()])
        if verbose:
            print(f"\n  {C.highlight(f'Walk-Forward Accuracy: {avg_acc:.1%}')}")
            print(f"  {C.DIM}(Random baseline = 33.3%){C.RESET}")
        
        return avg_acc
    
    def predict(self, X_row):
        """
        Predict direction for a single row or batch.
        Returns: (direction, probabilities_dict, confidence)
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained! Call train() first.")
        
        if isinstance(X_row, pd.Series):
            X_row = X_row[self.feature_cols].values.reshape(1, -1)
        elif isinstance(X_row, pd.DataFrame):
            X_row = X_row[self.feature_cols].values
        
        X_scaled = self.scaler.transform(X_row)
        
        # Get weighted probabilities from each model
        combined_probs = np.zeros((X_scaled.shape[0], 3))
        
        for name, model in self.models.items():
            probs = model.predict_proba(X_scaled)
            combined_probs += probs * self.weights.get(name, 1.0 / len(self.models))
        
        # Normalize
        combined_probs /= combined_probs.sum(axis=1, keepdims=True)
        
        results = []
        for i in range(len(combined_probs)):
            prob = combined_probs[i]
            pred_class = np.argmax(prob)
            direction = TARGET_MAP[pred_class]
            confidence = float(prob[pred_class])
            
            results.append({
                "direction": direction,
                "confidence": confidence,
                "prob_up": float(prob[0]),
                "prob_down": float(prob[1]),
                "prob_sideways": float(prob[2]),
            })
        
        if len(results) == 1:
            return results[0]
        return results
    
    def predict_today(self, df):
        """Predict direction for the latest row in the dataframe."""
        latest = df.iloc[-1:]
        return self.predict(latest)
    
    def get_top_features(self, n=15):
        """Return top N most important features."""
        if self.feature_importance is None:
            return []
        return self.feature_importance.head(n)
    
    def save(self, path=None):
        """Save the trained ensemble to disk."""
        if path is None:
            path = os.path.join(MODEL_DIR, "ensemble_classifier.pkl")
        with open(path, "wb") as f:
            pickle.dump({
                "models": self.models,
                "weights": self.weights,
                "scaler": self.scaler,
                "feature_cols": self.feature_cols,
                "validation_scores": self.validation_scores,
                "feature_importance": self.feature_importance,
            }, f)
        print(f"  {C.GREEN}[SAVED] Ensemble → {path}{C.RESET}")
    
    def load(self, path=None):
        """Load a previously trained ensemble from disk."""
        if path is None:
            path = os.path.join(MODEL_DIR, "ensemble_classifier.pkl")
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.models = data["models"]
        self.weights = data["weights"]
        self.scaler = data["scaler"]
        self.feature_cols = data["feature_cols"]
        self.validation_scores = data["validation_scores"]
        self.feature_importance = data.get("feature_importance")
        self.is_trained = True
        print(f"  {C.GREEN}[LOADED] Ensemble from {path}{C.RESET}")
        return True
    
    def detailed_backtest(self, df, feature_cols, train_end_year=2023):
        """
        Run a proper out-of-sample backtest.
        Train on data up to train_end_year, test on everything after.
        """
        train_mask = df["date"].dt.year <= train_end_year
        test_mask = df["date"].dt.year > train_end_year
        
        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()
        
        if len(test_df) == 0:
            print(f"  {C.YELLOW}[WARN] No test data after {train_end_year}{C.RESET}")
            return None
        
        print(f"\n{C.header('OUT-OF-SAMPLE BACKTEST')}")
        print(f"  Train: 2015–{train_end_year} ({len(train_df)} rows)")
        print(f"  Test:  {train_end_year+1}–now ({len(test_df)} rows)")
        
        # Train on training data
        X_train = train_df[feature_cols].values
        y_train = train_df["target"].values.astype(int)
        
        temp_scaler = StandardScaler()
        X_train_scaled = temp_scaler.fit_transform(X_train)
        
        temp_models = self._build_models()
        for name, model in temp_models.items():
            model.fit(X_train_scaled, y_train)
        
        # Test on unseen data
        X_test = test_df[feature_cols].values
        y_test = test_df["target"].values.astype(int)
        X_test_scaled = temp_scaler.transform(X_test)
        
        # Weighted predictions
        combined_probs = np.zeros((len(X_test), 3))
        for name, model in temp_models.items():
            probs = model.predict_proba(X_test_scaled)
            combined_probs += probs / len(temp_models)
        
        y_pred = np.argmax(combined_probs, axis=1)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        
        print(f"\n  {C.highlight(f'Out-of-Sample Accuracy: {acc:.1%}')}")
        print(f"  {C.highlight(f'Weighted F1 Score:      {f1:.3f}')}")
        print(f"\n  Classification Report:")
        
        report = classification_report(
            y_test, y_pred, 
            target_names=["UP", "DOWN", "SIDEWAYS"],
            zero_division=0
        )
        print(report)
        
        return {
            "accuracy": acc,
            "f1": f1,
            "y_test": y_test,
            "y_pred": y_pred,
            "test_dates": test_df["date"].values,
        }
