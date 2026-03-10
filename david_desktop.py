"""
DAVID ORACLE v2.0 — Desktop Backend Engine
===========================================
Powers the PyQt5 desktop app. All heavy operations run in threads.

Functions:
    fetch_spot()        → grabs live NIFTY + VIX (2 seconds)
    sync_all_data()     → downloads full CSV history (10 seconds)
    train_all_models()  → trains 4 ML models from CSVs (3 minutes)
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
            
            vix_date = pd.to_datetime(vix.index[-1]).tz_localize(None)
            yf_vix = float(vix["Close"].iloc[-1])
            
            # Start with YF as the baseline
            best_date = vix_date
            best_vix = yf_vix
            source = "YF"
            
            # Check Daily CSV
            daily_path = os.path.join(DATA_DIR, "vix_daily.csv")
            if os.path.exists(daily_path):
                df_daily = pd.read_csv(daily_path)
                daily_last_date = pd.to_datetime(df_daily["date"].iloc[-1]).tz_localize(None)
                if daily_last_date > best_date:
                    best_date = daily_last_date
                    best_vix = float(df_daily["close"].iloc[-1])
                    source = "Daily CSV"
                    
            # Check 15m CSV
            m15_path = os.path.join(DATA_DIR, "INDIAVIX_15minute_2001_now.csv")
            if os.path.exists(m15_path):
                df_15m = pd.read_csv(m15_path)
                last_15m_date = pd.to_datetime(df_15m["date"].iloc[-1]).tz_localize(None)
                if last_15m_date > best_date:
                    best_date = last_15m_date
                    best_vix = float(df_15m["Close"].iloc[-1])
                    source = "15m CSV"
            
            result["vix_value"] = best_vix
            log(f"  ✅ VIX: {result['vix_value']:.2f} (Source: {source})")

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
        
        log("\nStarting local data engine sync...")
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
    Train all 4 ML models from the CSV data and save as .pkl files.
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
        from models.intraday_classifier import IntradayClassifier
        import joblib
        
        # Load daily data
        log("\n[1/6] Loading daily data...")
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
        
        # Train Intraday 15M Classifier
        log("\n[6/6] Training 15M Intraday Classifier...")
        intra_model = IntradayClassifier()
        try:
            acc = intra_model.train(log_fn=log)
            intra_model.save()
            result["models_trained"].append("intraday_classifier")
            log(f"  ✅ Intraday Classifier saved (Test Acc: {acc*100:.1f}%)")
        except FileNotFoundError:
            log("  ⚠️ Missing 15m data CSVs. Intraday model skipped (Sync 15m Data first).")
        except Exception as e:
            log(f"  ❌ Failed to train Intraday model: {e}")
            
        result["success"] = True
        log(f"\n{'═' * 50}")
        log(f"✅ ALL CORE MODELS TRAINED! ({len(result['models_trained'])} models)")
        
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
        
        # Ensemble verdict (primary is tree/ensemble wrapper)
        if result["tree_prediction"]:
            result["ensemble_prediction"] = result["tree_prediction"]
        
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
        "nifty_15m": "nifty_15m_2001_to_now.csv",
        "vix_15m": "INDIAVIX_15minute_2001_now.csv",
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
# 7. SYNC 15-MINUTE DATA
# ═══════════════════════════════════════════════════════════════════════════════

def sync_15m_data():
    """
    Fetch and append 15-minute NIFTY + VIX candles using fetch_15m.py.
    Returns dict with sync results.
    """
    from fetch_15m import sync_15m_data as _sync
    return _sync(log_fn=log)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. HOLD / EXIT SIGNAL — The Backbone
# ═══════════════════════════════════════════════════════════════════════════════

def get_hold_exit_signal(entry_direction, entry_price, entry_date, prediction=None):
    """
    Compare the current prediction against the user's open position.
    Returns HOLD / HEDGE / EXIT with reasoning.
    
    Args:
        entry_direction: "UP", "DOWN", or "SIDEWAYS"
        entry_price: float, NIFTY level at entry
        entry_date: str, "YYYY-MM-DD"
        prediction: dict, existing prediction result (avoids re-running predict_now)
    
    Returns:
        dict with: signal, confidence, message, recovery_prob, days_held
    """
    result = {
        "signal": "HOLD",
        "confidence": 50,
        "message": "",
        "recovery_prob": None,
        "days_held": 0,
        "current_regime": None,
        "current_direction": None,
    }
    
    try:
        # Use provided prediction or run fresh
        pred = prediction
        if pred is None or not pred.get("success"):
            pred = predict_now()
        if not pred.get("success"):
            result["signal"] = "HOLD"
            result["message"] = "Cannot fetch prediction. Hold current position."
            return result
        
        current_price = pred["spot_price"]
        current_direction = None
        current_confidence = 0
        
        if pred.get("tree_prediction"):
            current_direction = pred["tree_prediction"]["direction"]
            current_confidence = pred["tree_prediction"]["confidence"] * 100
        
        result["current_direction"] = current_direction
        result["current_regime"] = pred.get("regime", "UNKNOWN")
        result["confidence"] = current_confidence
        
        # Days held
        try:
            entry_dt = datetime.strptime(entry_date, "%Y-%m-%d")
            result["days_held"] = (datetime.now() - entry_dt).days
        except Exception:
            pass
        
        # Recovery probability
        recovery = get_recovery_probability(current_price, entry_price, entry_direction)
        result["recovery_prob"] = recovery
        
        # ─── Decision Logic ──────────────────────────────────────────
        entry_dir_full = entry_direction.upper()
        is_credit = "CREDIT" in entry_dir_full
        
        if "UP" in entry_dir_full: base_dir = "UP"
        elif "DOWN" in entry_dir_full: base_dir = "DOWN"
        else: base_dir = "SIDEWAYS"
        
        # 1. Special Case: Credit Spread in Sideways Market
        if is_credit and current_direction == "SIDEWAYS" and base_dir != "SIDEWAYS":
            if current_confidence >= 50:
                result["signal"] = "HOLD"
                result["message"] = f"✅ HOLD — Sideways regime ({current_confidence:.0f}%). Your credit spread captures theta decay."
            else:
                result["signal"] = "HOLD"
                result["message"] = f"⚠️ HOLD (cautious) — Chop expected. Credit spread survives sideways action."
                
        # 2. Case: Direction still agrees perfectly
        elif current_direction and base_dir == current_direction:
            if current_confidence >= 50:
                result["signal"] = "HOLD"
                result["message"] = f"✅ HOLD — Direction still {current_direction} ({current_confidence:.0f}%). Stay in."
            elif current_confidence >= 40:
                result["signal"] = "HOLD"
                result["message"] = f"⚠️ HOLD (cautious) — Direction agrees but conviction weakening ({current_confidence:.0f}%)."
            else:
                result["signal"] = "HEDGE"
                result["message"] = f"⚠️ HEDGE — Direction agrees but conviction very low ({current_confidence:.0f}%). Protect it."
        
        # 3. Case: Direction flipped against you
        elif current_direction:
            if current_confidence >= 55:
                result["signal"] = "EXIT"
                result["message"] = f"🔴 EXIT — Direction flipped to {current_direction} ({current_confidence:.0f}%). Close position."
            elif current_confidence >= 40:
                result["signal"] = "HEDGE"
                result["message"] = f"⚠️ HEDGE — Direction shifting to {current_direction} ({current_confidence:.0f}%). Consider hedging."
            else:
                result["signal"] = "HOLD"
                result["message"] = f"🟡 HOLD — Direction unclear ({current_direction} at {current_confidence:.0f}%). Wait for clarity."
        
        # 4. Case: Regime danger
        if pred.get("regime") == "VOLATILE" and base_dir != "SIDEWAYS":
            if result["signal"] != "EXIT":
                result["signal"] = "HEDGE"
                result["message"] += " ⚠️ Regime is VOLATILE — tighten stops."
        
    except Exception as e:
        result["message"] = f"Error: {e}"
    
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 9. RECOVERY PROBABILITY — Historical drawdown analysis
# ═══════════════════════════════════════════════════════════════════════════════

def get_recovery_probability(current_price, entry_price, direction):
    """
    Scan 15 years of NIFTY history for similar drawdowns.
    Returns recovery probability and average recovery days.
    
    Args:
        current_price: Current NIFTY level
        entry_price: Entry NIFTY level
        direction: "UP" or "DOWN"
    """
    result = {
        "drawdown_pct": 0.0,
        "recovery_prob": 0.0,
        "avg_recovery_days": 0.0,
        "max_recovery_days": 0,
        "similar_scenarios": 0,
        "message": "",
    }
    
    try:
        from data_engine import load_all_data
        df = load_all_data(live_sentiment=False)
        
        # Calculate current drawdown
        dir_upper = direction.upper()
        if "UP" in dir_upper:
            dd_pct = ((current_price - entry_price) / entry_price) * 100
        else:
            dd_pct = ((entry_price - current_price) / entry_price) * 100
        
        result["drawdown_pct"] = round(dd_pct, 2)
        
        if dd_pct >= 0:
            result["recovery_prob"] = 100.0
            result["message"] = "You're in profit! No recovery needed."
            return result
        
        # Find similar historical drawdowns
        abs_dd = abs(dd_pct)
        dd_tolerance = max(0.5, abs_dd * 0.3)  # 30% tolerance band
        
        closes = df["close"].values
        recoveries = []
        
        for i in range(50, len(closes) - 30):
            # Look for similar drops
            for window in [1, 2, 3, 5]:
                if i + window >= len(closes):
                    continue
                drop = ((closes[i + window] - closes[i]) / closes[i]) * 100
                
                if "UP" in dir_upper:
                    historical_dd = drop  # negative = drawdown for bulls
                else:
                    historical_dd = -drop  # positive move = drawdown for bears
                
                if abs(historical_dd - (-abs_dd)) < dd_tolerance:
                    # Found similar drawdown — check recovery
                    recovery_found = False
                    for j in range(i + window + 1, min(i + window + 31, len(closes))):
                        if "UP" in dir_upper:
                            recovery = ((closes[j] - closes[i]) / closes[i]) * 100
                        else:
                            recovery = ((closes[i] - closes[j]) / closes[i]) * 100
                        
                        if recovery >= 0:
                            recoveries.append(j - (i + window))
                            recovery_found = True
                            break
                    
                    if not recovery_found:
                        recoveries.append(-1)  # did not recover within 30 days
        
        if recoveries:
            recovered = [r for r in recoveries if r > 0]
            result["similar_scenarios"] = len(recoveries)
            result["recovery_prob"] = round((len(recovered) / len(recoveries)) * 100, 1)
            
            if recovered:
                result["avg_recovery_days"] = round(np.mean(recovered), 1)
                result["max_recovery_days"] = max(recovered)
            
            if result["recovery_prob"] >= 70:
                result["message"] = f"📊 {result['recovery_prob']:.0f}% of similar {abs_dd:.1f}% dips recovered in ~{result['avg_recovery_days']:.0f} days. HOLD."
            elif result["recovery_prob"] >= 50:
                result["message"] = f"⚠️ {result['recovery_prob']:.0f}% recovery rate for {abs_dd:.1f}% dips. Monitor closely."
            else:
                result["message"] = f"🔴 Only {result['recovery_prob']:.0f}% recovery rate. This may be a regime shift, not a dip."
        else:
            result["message"] = "Not enough historical data for this drawdown level."
    
    except Exception as e:
        result["message"] = f"Recovery analysis error: {e}"
    
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 10. OPTIMAL EXPIRY SELECTOR
# ═══════════════════════════════════════════════════════════════════════════════

def get_optimal_expiry(confidence, vix_value, regime):
    """
    Recommend near or far expiry based on conviction, VIX, and regime.
    
    Returns:
        dict with: recommendation, expiry_type, reasoning
    """
    reasons = []
    score = 0  # positive = near expiry, negative = far expiry
    
    # Confidence factor
    if confidence >= 65:
        score += 2
        reasons.append(f"High confidence ({confidence:.0f}%) → near expiry captures more premium")
    elif confidence >= 50:
        score += 0
        reasons.append(f"Moderate confidence ({confidence:.0f}%) → either works")
    else:
        score -= 2
        reasons.append(f"Low confidence ({confidence:.0f}%) → far expiry for wider breakeven")
    
    # VIX factor
    if vix_value > 22:
        score -= 2
        reasons.append(f"VIX high ({vix_value:.1f}) → far expiry (vol crush will help)")
    elif vix_value > 16:
        score -= 1
        reasons.append(f"VIX moderate ({vix_value:.1f}) → slight preference for far")
    else:
        score += 1
        reasons.append(f"VIX low ({vix_value:.1f}) → near expiry fine")
    
    # Regime factor
    if regime == "TRENDING":
        score += 1
        reasons.append("TRENDING regime → near expiry (directional moves are fast)")
    elif regime == "VOLATILE":
        score -= 2
        reasons.append("VOLATILE regime → far expiry (need time buffer)")
    elif regime == "CHOPPY":
        score -= 1
        reasons.append("CHOPPY regime → far expiry (choppy markets need patience)")
    
    # Decision
    if score >= 2:
        expiry_type = "NEAR"
        recommendation = "📅 NEAR EXPIRY (current week / next week)"
    elif score <= -2:
        expiry_type = "FAR"
        recommendation = "📅 FAR EXPIRY (2-3 weeks out)"
    else:
        expiry_type = "MEDIUM"
        recommendation = "📅 MEDIUM EXPIRY (1-2 weeks out)"
    
    return {
        "recommendation": recommendation,
        "expiry_type": expiry_type,
        "score": score,
        "reasoning": reasons,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 11. 15-MIN INTRADAY REGIME
# ═══════════════════════════════════════════════════════════════════════════════

def get_intraday_regime():
    """Get the current intraday regime from 15-minute data."""
    from analyzers.regime_15m import Regime15mDetector
    detector = Regime15mDetector()
    return detector.analyze()


def predict_intraday_now():
    """
    Get prediction from the Intraday XGBoost classifier trained on 15m data.
    """
    from models.intraday_classifier import IntradayClassifier
    model = IntradayClassifier()
    if not model.load():
        return {"error": "Intraday model not trained yet. Click Train Models."}
    return model.predict_now()


# ═══════════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# ZERO ANXIETY SYSTEM — Traffic Light, Strategy Recommender, Morning Briefing
# ═══════════════════════════════════════════════════════════════════════════════

def compute_traffic_light(pred):
    """
    Analyze all signals and return a single GREEN / YELLOW / RED verdict.
    
    GREEN: Strong signal, safe to trade
    YELLOW: Mixed signals, reduce size or wait
    RED: Do not trade
    """
    if pred is None or not pred.get("success"):
        return {"color": "RED", "reason": "No prediction data available."}
    
    score = 0  # -10 to +10 scale
    reasons = []
    
    # 1. Direction confidence
    confidence = 0.0
    direction = "SIDEWAYS"
    tree = pred.get("tree_prediction")
    ensemble = pred.get("ensemble_prediction")
    
    if ensemble:
        confidence = ensemble.get("confidence", 0.5) * 100
        direction = ensemble.get("direction", "SIDEWAYS")
    elif tree:
        confidence = tree.get("confidence", 0.5) * 100
        direction = tree.get("direction", "SIDEWAYS")
    
    if confidence > 65:
        score += 3
        reasons.append(f"Strong confidence: {confidence:.0f}%")
    elif confidence > 50:
        score += 1
        reasons.append(f"Moderate confidence: {confidence:.0f}%")
    else:
        score -= 2
        reasons.append(f"Low confidence: {confidence:.0f}%")
    
    # 2. VIX level
    vix = pred.get("vix_value", 15.0) or 15.0
    if vix < 15:
        score += 2
        reasons.append(f"Low VIX ({vix:.1f}) — calm market")
    elif vix < 20:
        score += 1
        reasons.append(f"Normal VIX ({vix:.1f})")
    elif vix < 25:
        score -= 1
        reasons.append(f"Elevated VIX ({vix:.1f}) — caution")
    else:
        score -= 3
        reasons.append(f"High VIX ({vix:.1f}) — dangerous conditions")
    
    # 3. Regime alignment
    regime = pred.get("regime", "CHOPPY")
    if regime == "TRENDING" and direction != "SIDEWAYS":
        score += 2
        reasons.append(f"Trending regime — directional bet supported")
    elif regime == "VOLATILE":
        score -= 2
        reasons.append(f"Volatile regime — unpredictable")
    elif regime == "CHOPPY":
        score -= 1
        reasons.append(f"Choppy regime — whipsaws possible")
    
    # 4. Whipsaw check
    whipsaw = pred.get("whipsaw")
    if whipsaw and whipsaw.get("is_whipsaw"):
        score -= 2
        reasons.append("⚠️ Whipsaw detected — conflicting signals")
    
    # 5. PCR support
    pcr = pred.get("pcr", 1.0)
    if pcr and pcr > 1.2:
        score += 1
        reasons.append(f"Bullish PCR ({pcr:.2f}) — support from options data")
    elif pcr and pcr < 0.7:
        score += 1
        reasons.append(f"Bearish PCR ({pcr:.2f}) — options confirm bearishness")
    
    # Determine color
    if score >= 4:
        color = "GREEN"
    elif score >= 0:
        color = "YELLOW"
    else:
        color = "RED"
    
    return {
        "color": color,
        "score": score,
        "direction": direction,
        "confidence": confidence,
        "reasons": reasons,
    }


def recommend_strategy(pred):
    """
    Recommend the best options strategy based on current market conditions.
    
    Returns:
        dict with: strategy, reasoning, risk_level, direction
    """
    if pred is None or not pred.get("success"):
        return {
            "strategy": "NO TRADE",
            "reasoning": "No prediction data available.",
            "risk_level": "UNKNOWN",
            "direction": "NONE",
        }
    
    direction = "SIDEWAYS"
    confidence = 50.0
    tree = pred.get("tree_prediction")
    ensemble = pred.get("ensemble_prediction")
    
    if ensemble:
        direction = ensemble.get("direction", "SIDEWAYS")
        confidence = ensemble.get("confidence", 0.5) * 100
    elif tree:
        direction = tree.get("direction", "SIDEWAYS")
        confidence = tree.get("confidence", 0.5) * 100
    
    vix = pred.get("vix_value", 15.0) or 15.0
    regime = pred.get("regime", "CHOPPY")
    high_vix = vix > 18
    
    # Strategy selection matrix
    if direction == "UP":
        if high_vix:
            strategy = "Bull Put Spread (Credit)"
            reasoning = (
                f"Market predicted UP ({confidence:.0f}% confidence). "
                f"VIX is elevated ({vix:.1f}), so SELL premium with a Bull Put Spread. "
                f"Theta works in your favor. You profit if NIFTY stays above the sold strike."
            )
            risk_level = "MODERATE"
        else:
            strategy = "Bull Call Spread (Debit)"
            reasoning = (
                f"Market predicted UP ({confidence:.0f}% confidence). "
                f"VIX is low ({vix:.1f}), so options are cheap — BUY a Bull Call Spread. "
                f"Lower cost entry with defined risk."
            )
            risk_level = "LOW"
    
    elif direction == "DOWN":
        if high_vix:
            strategy = "Bear Call Spread (Credit)"
            reasoning = (
                f"Market predicted DOWN ({confidence:.0f}% confidence). "
                f"VIX elevated ({vix:.1f}) — SELL premium with a Bear Call Spread. "
                f"You profit if NIFTY stays below the sold strike."
            )
            risk_level = "MODERATE"
        else:
            strategy = "Bear Put Spread (Debit)"
            reasoning = (
                f"Market predicted DOWN ({confidence:.0f}% confidence). "
                f"VIX is low ({vix:.1f}) — BUY a Bear Put Spread for cheap directional exposure."
            )
            risk_level = "LOW"
    
    else:  # SIDEWAYS
        strategy = "Iron Condor (Credit)"
        reasoning = (
            f"Market predicted SIDEWAYS ({confidence:.0f}% confidence). "
            f"Sell both sides with an Iron Condor. "
            f"You profit if NIFTY stays within the range. Time decay is your friend."
        )
        risk_level = "LOW" if vix < 15 else "MODERATE"
    
    return {
        "strategy": strategy,
        "reasoning": reasoning,
        "risk_level": risk_level,
        "direction": direction,
    }


def generate_morning_briefing(pred):
    """
    Combine all signals into a single morning briefing summary.
    
    Returns:
        dict with all the info needed for the Command Center
    """
    traffic = compute_traffic_light(pred)
    strategy = recommend_strategy(pred)
    
    briefing = {
        "date": datetime.now().strftime("%A, %b %d %Y"),
        "traffic_light": traffic,
        "strategy": strategy,
        "regime": pred.get("regime", "UNKNOWN") if pred else "UNKNOWN",
        "spot_price": pred.get("spot_price", 0) if pred else 0,
        "vix": pred.get("vix_value", 0) if pred else 0,
        "pcr": pred.get("pcr", 0) if pred else 0,
        "supports": [],
        "resistances": [],
        "signal_summary": "",
    }
    
    if pred and pred.get("success"):
        briefing["supports"] = pred.get("supports", [])
        briefing["resistances"] = pred.get("resistances", [])
        
        # Build human-readable summary
        color_emoji = {"GREEN": "🟢", "YELLOW": "🟡", "RED": "🔴"}.get(traffic["color"], "⚪")
        briefing["signal_summary"] = (
            f"{color_emoji} Signal: {traffic['color']} | "
            f"Regime: {briefing['regime']} | "
            f"Strategy: {strategy['strategy']}"
        )
    else:
        briefing["signal_summary"] = "⚠️ No prediction data. Run Fetch + Predict first."
    
    return briefing


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
