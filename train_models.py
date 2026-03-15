"""
DAVID ORACLE — Model Training Script (for CI/CD)
===================================================
Pre-trains all models and saves them as .pkl files.
Designed to run in GitHub Actions so Streamlit loads instantly.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from data_engine import load_all_data
from feature_forge import engineer_features
from models.ensemble_classifier import EnsembleClassifier
from models.regime_detector import RegimeDetector
from models.range_predictor import RangePredictor
def classify_regime(row):
    adx = row.get('adx', 20)
    vol = row.get('realized_vol_20', 0.15)
    if adx > 25: return "TRENDING"
    elif vol > 0.25: return "VOLATILE"
    else: return "CHOPPY"


def train_all():
    print("=" * 60)
    print("  DAVID MODEL TRAINER")
    print("  Pre-training all models for fast Streamlit startup")
    print("=" * 60)

    # 1. Load data and engineer features
    df_raw = load_all_data()
    df, features = engineer_features(df_raw)

    # 2. Train default ensemble
    print("\n[1/4] Training Ensemble Classifier...")
    ensemble = EnsembleClassifier()
    ensemble.train(df, features)
    ensemble.save()
    print("  ✅ Ensemble saved")

    # 3. Train regime detector
    print("\n[2/4] Training Regime Detector...")
    regime = RegimeDetector()
    regime.train(df)
    regime.save()
    print("  ✅ Regime Detector saved")

    # 4. Train range predictor
    print("\n[3/4] Training Range Predictor...")
    range_pred = RangePredictor()
    range_pred.train(df, features)
    range_pred.save()
    print("  ✅ Range Predictor saved")

    # 5. Train regime-specific models (for the routing logic)
    print("\n[4/4] Training Regime-Specific Ensembles...")
    import joblib
    from utils import MODEL_DIR
    regime_models = {}
    for r_name in ["TRENDING", "CHOPPY", "VOLATILE"]:
        df_regime = df[df.apply(classify_regime, axis=1) == r_name].copy()
        if len(df_regime) < 50:
            df_regime = df.copy()
        m = EnsembleClassifier()
        m.train(df_regime, features, verbose=False)
        regime_models[r_name] = m
        print(f"    ✅ {r_name}: {len(df_regime)} samples")
    
    joblib.dump(regime_models, os.path.join(MODEL_DIR, "regime_models.pkl"))
    print("  ✅ Regime models saved")

    print("\n" + "=" * 60)
    print("  ALL MODELS TRAINED AND SAVED ✅")
    print("=" * 60)


if __name__ == "__main__":
    train_all()
