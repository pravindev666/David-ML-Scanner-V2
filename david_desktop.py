"""
DAVID ORACLE v2.0 — Desktop Backend Engine
===========================================
Powers the PyQt5 desktop app. All heavy operations run in threads.

Functions:
    fetch_spot()        → grabs live NIFTY + VIX (2 seconds)
    sync_all_data()     → downloads full CSV history (10 seconds)
    train_all_models()  → trains 5 ML models from CSVs (3 minutes)
    predict_now()       → loads .pkl + CSVs + live spot → full prediction
    get_data_status()   → returns CSV freshness info
"""

import os
import sys
import time
import traceback
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Ensure imports work from this directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (
    DATA_DIR, MODEL_DIR, C,
    NIFTY_SYMBOL, VIX_SYMBOL, SP500_SYMBOL,
    DATA_START_YEAR, UP, DOWN, SIDEWAYS
)

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING CALLBACK
# ═══════════════════════════════════════════════════════════════════════════════

_log_callback = None

def set_log_callback(callback):
    """Set a callback function that receives log messages for the GUI."""
    global _log_callback
    _log_callback = callback

def log(msg):
    """Print to console AND send to GUI log panel."""
    print(msg)
    if _log_callback:
        try:
            _log_callback(msg)
        except Exception:
            pass

# ═══════════════════════════════════════════════════════════════════════════════
# 1. FETCH SPOT — Lightning fast (2 seconds)
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_spot():
    """
    Fetch ONLY the latest NIFTY spot price and India VIX.
    This is the fastest operation — just 2 tiny API calls.
    
    Returns:
        dict with keys: nifty_price, vix_value, timestamp, success
    """
    import yfinance as yf
    
    result = {
        "nifty_price": None,
        "vix_value": None,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "success": False,
        "error": None
    }
    
    try:
        log("🔴 Fetching NIFTY spot price...")
        nifty = yf.download(NIFTY_SYMBOL, period="5d", progress=False)
        if not nifty.empty:
            # Handle MultiIndex columns
            if isinstance(nifty.columns, pd.MultiIndex):
                nifty.columns = nifty.columns.get_level_values(0)
            elif len(nifty.columns) > 0 and isinstance(nifty.columns[0], tuple):
                nifty.columns = [c[0] for c in nifty.columns]
            
            result["nifty_price"] = float(nifty["Close"].iloc[-1])
            log(f"  ✅ NIFTY: {result['nifty_price']:,.2f}")
        else:
            log("  ⚠️ No NIFTY data returned")
    except Exception as e:
        log(f"  ❌ NIFTY fetch failed: {e}")
        result["error"] = str(e)
    
    try:
        log("🔴 Fetching India VIX...")
        vix = yf.download(VIX_SYMBOL, period="5d", progress=False)
        if not vix.empty:
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = vix.columns.get_level_values(0)
            elif len(vix.columns) > 0 and isinstance(vix.columns[0], tuple):
                vix.columns = [c[0] for c in vix.columns]
            
            result["vix_value"] = float(vix["Close"].iloc[-1])
            log(f"  ✅ VIX: {result['vix_value']:.2f}")
        else:
            log("  ⚠️ No VIX data returned")
    except Exception as e:
        log(f"  ❌ VIX fetch failed: {e}")
    
    result["success"] = result["nifty_price"] is not None
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SYNC ALL DATA — Downloads full CSV history (~10 seconds)
# ═══════════════════════════════════════════════════════════════════════════════

def sync_all_data():
    """
    Download all market data (NIFTY, VIX, SP500, PCR, FII/DII) and save to CSVs.
    Uses the existing data_engine.py with live_sentiment=True.
    
    Returns:
        dict with keys: success, row_counts, error
    """
    result = {"success": False, "row_counts": {}, "error": None}
    
    try:
        log("📊 Syncing all market data...")
        log("═" * 50)
        
        log("☁️  Checking for updates from Cloud (GitHub)...")
        import subprocess
        try:
            # First try pulling the latest pre-fetched CSVs from the GitHub repository 
            # (which are generated every 15 mins by GitHub Actions)
            # Use strict host key checking = no to prevent ssh prompts if ssh is used
            subprocess.run(["git", "-c", "core.sshCommand=ssh -o StrictHostKeyChecking=no", "pull", "origin", "main"], 
                           check=True, capture_output=True, text=True)
            log("  ✅ Successfully pulled latest pre-fetched CSVs from GitHub.")
        except Exception as e:
            log("  ⚠️ Git pull skipped (offline, no updates, or not a git repo).")
            log(f"  Debug: {e}")

        log("\nStarting local data engine fallback sync...")
        from data_engine import load_all_data
        df = load_all_data(live_sentiment=True)
        
        result["row_counts"] = {
            "total_rows": len(df),
            "date_range": f"{df['date'].min().date()} → {df['date'].max().date()}",
            "latest_close": float(df["close"].iloc[-1])
        }
        result["success"] = True
        
        log(f"\n✅ Sync complete! {len(df)} trading days loaded.")
        log(f"   Latest: {df['date'].max().date()} — ₹{df['close'].iloc[-1]:,.2f}")
        
    except Exception as e:
        result["error"] = str(e)
        log(f"❌ Sync failed: {e}")
        log(traceback.format_exc())
    
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TRAIN ALL MODELS — Builds .pkl files (~3 minutes)
# ═══════════════════════════════════════════════════════════════════════════════

def train_all_models():
    """
    Train all 5 ML models from the CSV data and save as .pkl files.
    Uses the existing train_models.py logic.
    
    Returns:
        dict with keys: success, models_trained, error
    """
    result = {"success": False, "models_trained": [], "error": None}
    
    try:
        log("🧠 Training all models...")
        log("═" * 50)
        
        # Import and run the training pipeline
        from data_engine import load_all_data
        from feature_forge import engineer_features
        from models.ensemble_classifier import EnsembleClassifier
        from models.regime_detector import RegimeDetector
        from models.range_predictor import RangePredictor
        # NOTE: LSTMClassifier import deferred to avoid torch DLL crash
        import joblib
        
        # Load data
        log("\n[1/5] Loading data...")
        df_raw = load_all_data(live_sentiment=False)  # Use cached CSVs
        df, features = engineer_features(df_raw)
        log(f"  ✅ Data loaded: {len(df)} rows, {len(features)} features")
        
        # Train Ensemble
        log("\n[2/5] Training Ensemble Classifier...")
        ensemble = EnsembleClassifier()
        ensemble.train(df, features)
        ensemble.save()
        result["models_trained"].append("ensemble_classifier")
        log("  ✅ Ensemble saved")
        
        # Train Regime Detector
        log("\n[3/5] Training Regime Detector...")
        regime = RegimeDetector()
        regime.train(df, features)
        regime.save()
        result["models_trained"].append("regime_detector")
        log("  ✅ Regime Detector saved")
        
        # Train Range Predictor
        log("\n[4/5] Training Range Predictor...")
        range_pred = RangePredictor()
        range_pred.train(df, features)
        range_pred.save()
        result["models_trained"].append("range_predictor")
        log("  ✅ Range Predictor saved")
        
        # Train Regime-Specific Ensembles
        log("\n[5/5] Training Regime-Specific Ensembles...")
        regime_models = {}
        for regime_name in ["TRENDING", "CHOPPY", "VOLATILE"]:
            def classify_regime(row):
                adx = row.get('adx', 20)
                vol = row.get('realized_vol_20', 0.15)
                if adx > 25: return "TRENDING"
                elif vol > 0.25: return "VOLATILE"
                else: return "CHOPPY"
            
            mask = df.apply(classify_regime, axis=1) == regime_name
            df_regime = df[mask]
            if len(df_regime) > 100:
                model = EnsembleClassifier()
                model.train(df_regime, features)
                regime_models[regime_name] = model
                log(f"  ✅ {regime_name}: {len(df_regime)} samples")
            else:
                log(f"  ⚠️ {regime_name}: Only {len(df_regime)} samples, skipping")
        
        regime_models_path = os.path.join(MODEL_DIR, "regime_models.pkl")
        joblib.dump(regime_models, regime_models_path)
        result["models_trained"].append("regime_models")
        log("  ✅ Regime models saved")
        
        # Try LSTM (optional, may fail on some systems due to torch DLL issues)
        try:
            log("\n[Bonus] Training LSTM Sequence Model...")
            from models.lstm_classifier import LSTMClassifier
            lstm = LSTMClassifier()
            lstm.train(df, features)
            lstm.save()
            result["models_trained"].append("lstm_classifier")
            log("  ✅ LSTM saved")
        except Exception as e:
            log(f"  ⚠️ LSTM skipped (torch issue): {e}")
            log("  ℹ️ This is normal — core models still work perfectly.")
        
        result["success"] = True
        log(f"\n{'═' * 50}")
        log(f"✅ ALL MODELS TRAINED! ({len(result['models_trained'])} models)")
        
    except Exception as e:
        result["error"] = str(e)
        log(f"❌ Training failed: {e}")
        log(traceback.format_exc())
    
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PREDICT NOW — Instant prediction using .pkl + live spot
# ═══════════════════════════════════════════════════════════════════════════════

def predict_now(spot_price=None, vix_value=None):
    """
    Load pre-trained .pkl models and cached CSVs, inject live spot/VIX,
    and run the full prediction pipeline.
    
    Args:
        spot_price: Live NIFTY price (if None, uses latest from CSV)
        vix_value: Live VIX value (if None, uses latest from CSV)
    
    Returns:
        dict with full prediction results
    """
    result = {
        "success": False,
        "error": None,
        "spot_price": spot_price,
        "vix_value": vix_value,
        "tree_prediction": None,
        "lstm_prediction": None,
        "ensemble_prediction": None,
        "regime": None,
        "regime_info": None,
        "whipsaw": None,
        "ranges": None,
        "supports": None,
        "resistances": None,
        "pcr": None,
        "fii_net": None,
        "dii_net": None,
        "df": None,
        "df_raw": None,
        "features": None,
    }
    
    try:
        log("🦅 Running prediction pipeline...")
        
        import joblib
        from data_engine import load_all_data
        from feature_forge import engineer_features
        from models.ensemble_classifier import EnsembleClassifier
        from models.regime_detector import RegimeDetector
        from models.range_predictor import RangePredictor
        # NOTE: LSTMClassifier import deferred to try/except below
        from models.sr_engine import SREngine
        from analyzers.whipsaw_detector import WhipsawDetector
        from analyzers.iron_condor_analyzer import IronCondorAnalyzer
        from analyzers.bounce_analyzer import BounceAnalyzer
        
        # Load cached data (no network calls)
        log("  Loading cached data...")
        df_raw = load_all_data(live_sentiment=False)
        
        # Inject live spot price if provided
        if spot_price is not None:
            log(f"  Injecting live spot: ₹{spot_price:,.2f}")
            df_raw.loc[df_raw.index[-1], "close"] = spot_price
        if vix_value is not None:
            log(f"  Injecting live VIX: {vix_value:.2f}")
            df_raw.loc[df_raw.index[-1], "vix"] = vix_value
        
        result["df_raw"] = df_raw
        result["spot_price"] = spot_price or float(df_raw["close"].iloc[-1])
        result["vix_value"] = vix_value or float(df_raw.get("vix", pd.Series([15.0])).iloc[-1])
        
        # Engineer features
        log("  Engineering features...")
        df, features = engineer_features(df_raw)
        result["df"] = df
        result["features"] = features
        
        # Sentiment values
        result["pcr"] = float(df_raw["pcr"].iloc[-1]) if "pcr" in df_raw.columns else 1.0
        result["fii_net"] = float(df_raw["fii_net"].iloc[-1]) if "fii_net" in df_raw.columns else 0.0
        result["dii_net"] = float(df_raw["dii_net"].iloc[-1]) if "dii_net" in df_raw.columns else 0.0
        
        # Regime classification
        latest_row = df.iloc[-1]
        adx = latest_row.get('adx', 20)
        vol = latest_row.get('realized_vol_20', 0.15)
        if adx > 25:
            current_regime = "TRENDING"
        elif vol > 0.25:
            current_regime = "VOLATILE"
        else:
            current_regime = "CHOPPY"
        result["regime"] = current_regime
        
        # Load models
        log("  Loading pre-trained models...")
        
        # Ensemble
        ensemble = EnsembleClassifier()
        try:
            ensemble.load()
            log("    ✅ Ensemble loaded")
        except Exception:
            log("    ⚠️ Ensemble not found, skipping")
            ensemble = None
        
        # Regime-specific models
        regime_models_path = os.path.join(MODEL_DIR, "regime_models.pkl")
        regime_models = {}
        try:
            regime_models = joblib.load(regime_models_path)
            log(f"    ✅ Regime models loaded ({len(regime_models)} regimes)")
        except Exception:
            log("    ⚠️ Regime models not found")
        
        # LSTM (may fail on systems with broken torch)
        lstm = None
        try:
            from models.lstm_classifier import LSTMClassifier
            lstm = LSTMClassifier()
            lstm.load()
            log("    ✅ LSTM loaded")
        except Exception:
            log("    ⚠️ LSTM not available (torch issue, skipping)")
            lstm = None
        
        # Regime detector
        regime_detector = RegimeDetector()
        try:
            regime_detector.load()
            log("    ✅ Regime detector loaded")
        except Exception:
            log("    ⚠️ Regime detector not found, using rule-based")
        
        # Range predictor
        range_pred = RangePredictor()
        try:
            if not range_pred.load():
                range_pred = None
            else:
                log("    ✅ Range predictor loaded")
        except Exception:
            range_pred = None
            log("    ⚠️ Range predictor not found")
        
        # Run predictions
        log("  Running predictions...")
        
        # Tree prediction
        if ensemble:
            regime_model = regime_models.get(current_regime, ensemble)
            tree_pred = regime_model.predict_today(df)
            result["tree_prediction"] = tree_pred
            log(f"    🌳 Tree: {tree_pred['direction']} ({tree_pred['confidence']*100:.0f}%)")
        
        # LSTM prediction
        if lstm:
            try:
                lstm_pred = lstm.predict_today(df, features)
                result["lstm_prediction"] = lstm_pred
                log(f"    🧠 LSTM: {lstm_pred['direction']} ({lstm_pred['confidence']*100:.0f}%)")
            except Exception:
                log("    ⚠️ LSTM prediction failed")
        
        # Ensemble verdict (combine tree + LSTM)
        if result["tree_prediction"]:
            result["ensemble_prediction"] = result["tree_prediction"]  # Primary
        
        # Regime info
        try:
            regime_info = regime_detector.get_regime_with_micro_direction(df, result["tree_prediction"])
            result["regime_info"] = regime_info
        except Exception:
            result["regime_info"] = {"regime": current_regime, "micro": "N/A"}
        
        # Whipsaw
        try:
            whipsaw = WhipsawDetector()
            result["whipsaw"] = whipsaw.analyze(df)
        except Exception:
            pass
        
        # Ranges
        if range_pred:
            try:
                result["ranges"] = range_pred.predict_range(df, result["spot_price"])
            except Exception:
                pass
        
        # Support & Resistance
        try:
            sr = SREngine()
            supports, resistances = sr.find_levels(df_raw)
            result["supports"] = supports
            result["resistances"] = resistances
        except Exception:
            pass
        
        result["success"] = True
        log(f"\n✅ Prediction complete!")
        
    except Exception as e:
        result["error"] = str(e)
        log(f"❌ Prediction failed: {e}")
        log(traceback.format_exc())
    
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 5. DATA STATUS — Instant CSV freshness check
# ═══════════════════════════════════════════════════════════════════════════════

def get_data_status():
    """
    Check the freshness and size of all CSV data files.
    
    Returns:
        list of dicts with: name, rows, latest_date, file_size, is_stale
    """
    csv_files = {
        "nifty": "nifty_daily.csv",
        "vix": "vix_daily.csv",
        "sp500": "sp500_daily.csv",
        "pcr": "pcr_daily.csv",
        "fii_dii": "fii_dii_daily.csv",
    }
    
    status = []
    today = datetime.now().date()
    
    for name, filename in csv_files.items():
        path = os.path.join(DATA_DIR, filename)
        info = {
            "name": name,
            "filename": filename,
            "exists": False,
            "rows": 0,
            "latest_date": "N/A",
            "file_size": "0 KB",
            "is_stale": True,
        }
        
        if os.path.exists(path):
            info["exists"] = True
            info["file_size"] = f"{os.path.getsize(path) / 1024:.0f} KB"
            
            try:
                df = pd.read_csv(path, parse_dates=["date"])
                info["rows"] = len(df)
                latest = df["date"].max()
                info["latest_date"] = latest.strftime("%Y-%m-%d")
                
                # Stale if latest date is more than 2 days old (accounting for weekends)
                days_old = (today - latest.date()).days
                info["is_stale"] = days_old > 3
            except Exception:
                info["latest_date"] = "Error reading"
        
        status.append(info)
    
    # Also check model files
    model_files = {
        "ensemble": "ensemble_classifier.pkl",
        "regime_models": "regime_models.pkl",
        "regime_detector": "regime_detector.pkl",
        "range_predictor": "range_predictor.pkl",
        "lstm": "lstm_classifier.pkl",
    }
    
    model_status = []
    for name, filename in model_files.items():
        path = os.path.join(MODEL_DIR, filename)
        info = {
            "name": name,
            "filename": filename,
            "exists": os.path.exists(path),
            "file_size": f"{os.path.getsize(path) / 1024:.0f} KB" if os.path.exists(path) else "0 KB",
            "modified": datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M") if os.path.exists(path) else "N/A",
        }
        model_status.append(info)
    
    return {"csv": status, "models": model_status}


# ═══════════════════════════════════════════════════════════════════════════════
# 6. IRON CONDOR ANALYZER (for Strategy Lab tab)
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_strike(df_raw, strike_price, days=5):
    """Run iron condor analysis on a strike price."""
    from analyzers.iron_condor_analyzer import IronCondorAnalyzer
    condor = IronCondorAnalyzer()
    return condor.analyze_strike(df_raw, strike_price, days)


def analyze_bounce(df_raw, target_price):
    """Run bounce-back analysis for a target price."""
    from analyzers.bounce_analyzer import BounceAnalyzer
    bounce = BounceAnalyzer()
    return bounce.analyze(df_raw, target_price)


# ═══════════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  DAVID ORACLE DESKTOP — Backend Self-Test")
    print("=" * 60)
    
    print("\n1. Data Status:")
    status = get_data_status()
    for s in status["csv"]:
        icon = "✅" if s["exists"] and not s["is_stale"] else "⚠️" if s["exists"] else "❌"
        print(f"   {icon} {s['name']:>8}: {s['rows']:>5} rows | Latest: {s['latest_date']} | {s['file_size']}")
    
    print("\n2. Model Status:")
    for m in status["models"]:
        icon = "✅" if m["exists"] else "❌"
        print(f"   {icon} {m['name']:>16}: {m['file_size']} | Modified: {m['modified']}")
    
    print("\n3. Running cached prediction (no network)...")
    pred = predict_now()
    if pred["success"]:
        print(f"   Spot: ₹{pred['spot_price']:,.2f}")
        print(f"   Regime: {pred['regime']}")
        if pred["tree_prediction"]:
            tp = pred["tree_prediction"]
            print(f"   Verdict: {tp['direction']} ({tp['confidence']*100:.0f}%)")
    else:
        print(f"   Failed: {pred['error']}")
    
    print("\nDone!")
