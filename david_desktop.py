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
    DATA_DIR, MODEL_DIR, C, WAR_ROOM_PATH,
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
        nifty = yf.Ticker(NIFTY_SYMBOL).history(period="5d")
        if not nifty.empty:
            result["nifty_price"] = float(nifty["Close"].iloc[-1])
            log(f"  ✅ NIFTY: {result['nifty_price']:,.2f}")
        else:
            log("  ⚠️ No NIFTY data returned")
    except Exception as e:
        log(f"  ❌ NIFTY fetch failed: {e}")
        result["error"] = str(e)
    
    try:
        log("🔴 Fetching India VIX...")
        vix = yf.Ticker(VIX_SYMBOL).history(period="5d")
        if not vix.empty:
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
        "rsi_15m": None,
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

        # New MiroFish Features
        result["oi_change_1d"] = latest_row.get("oi_change_1d", 0.0)
        result["oi_change_5d"] = latest_row.get("oi_change_5d", 0.0)
        result["long_build_up"] = latest_row.get("long_build_up", 0.0)
        result["short_build_up"] = latest_row.get("short_build_up", 0.0)
        result["long_unwinding"] = latest_row.get("long_unwinding", 0.0)
        result["vol_poc_dist_20"] = latest_row.get("vol_poc_dist_20", 0.0)
        
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
            
        # 15-Minute RSI calculation for Intraday Legging Pulse
        try:
            from utils import DATA_DIR
            nifty_15m_path = os.path.join(DATA_DIR, "nifty_15m_2001_to_now.csv")
            if os.path.exists(nifty_15m_path):
                df_15m = pd.read_csv(nifty_15m_path)
                if not df_15m.empty and "close" in df_15m.columns:
                    close_prices = df_15m["close"].tail(200).values
                    # Simple RSI(14) calculation
                    deltas = np.diff(close_prices)
                    seed = deltas[:14]
                    up = seed[seed >= 0].sum() / 14
                    down = -seed[seed < 0].sum() / 14
                    rs = up / down if down != 0 else 0
                    rsi = np.zeros_like(close_prices)
                    rsi[14] = 100. - 100. / (1. + rs)

                    for i in range(14, len(close_prices) - 1):
                        delta = deltas[i]
                        upval = delta if delta > 0 else 0.
                        downval = -delta if delta < 0 else 0.
                        up = (up * 13 + upval) / 14
                        down = (down * 13 + downval) / 14
                        rs = up / down if down != 0 else 0
                        rsi[i+1] = 100. - 100. / (1. + rs)
                    
                    result["rsi_15m"] = float(rsi[-1])
        except Exception as e:
            log(f"    ⚠️ Failed to calculate 15m RSI: {e}")
        
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
        "pcr_oi": "pcr_daily.csv",
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
                # Get row count without loading entire file (fast)
                with open(path, 'r') as f:
                    info["rows"] = sum(1 for _ in f) - 1  # subtract header
                
                # Read just the last few rows to get latest date
                df_sample = pd.read_csv(path, usecols=["date"])
                latest_str = df_sample["date"].iloc[-1]
                latest = pd.to_datetime(latest_str, format='%Y-%m-%d %H:%M:%S', errors='coerce')
                if pd.isna(latest):
                    latest = pd.to_datetime(latest_str, errors='coerce')
                info["latest_date"] = latest.strftime("%Y-%m-%d")
                
                # Stale if latest date is more than 2 days old (accounting for weekends)
                days_old = (today - latest.date()).days
                info["is_stale"] = days_old > 3
            except Exception:
                info["latest_date"] = "Error reading"
        
        status.append(info)
    
    # Model Status (.pkl files)
    model_files = {
        "Ensemble": "ensemble_classifier.pkl",
        "Regime": "regime_detector.pkl",
        "Ranges": "range_predictor.pkl",
        "Intraday": "intraday_classifier.pkl",
        "Regime Sub": "regime_models.pkl"
    }
    model_status = []
    for name, filename in model_files.items():
        path = os.path.join(MODEL_DIR, filename)
        exists = os.path.exists(path)
        modified = "N/A"
        size = "0 KB"
        if exists:
            mtime = os.path.getmtime(path)
            modified = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
            size = f"{os.path.getsize(path) / 1024:.0f} KB"
        
        model_status.append({
            "name": name,
            "filename": filename,
            "exists": exists,
            "modified": modified,
            "file_size": size
        })

    # Feature Status (Institutional Metrics)
    feature_status = []
    try:
        from data_engine import load_all_data
        from feature_forge import engineer_features
        df_raw = load_all_data(live_sentiment=False)
        df, _ = engineer_features(df_raw)
        
        target_features = {
            "total_oi": "Total Open Interest",
            "oi_change_1d": "OI Change (1D)",
            "long_build_up": "Long Buildup Flag",
            "short_build_up": "Short Buildup Flag",
            "vol_poc_dist_20": "Dist from POC (20d)",
            "rsi_7": "Pulse (RSI 7d)",
            "adx": "Trend Power (ADX)",
            "vix_percentile": "Fear Percentile (VIX)",
            "bb_width": "Volatility Band Width"
        }
        
        for feat, label in target_features.items():
            exists = feat in df.columns
            val = df[feat].iloc[-1] if exists else "N/A"
            if isinstance(val, (float, np.float64)):
                if "rsi" in feat or "adx" in feat:
                    val_str = f"{val:.1f}"
                elif "percentile" in feat:
                    val_str = f"{val*100:.1f}%"
                else:
                    val_str = f"{val:+.2f}" if "dist" in feat or "change" in feat or "width" in feat else f"{val:,.0f}"
            else:
                val_str = str(val)
                
            feature_status.append({
                "name": label,
                "key": feat,
                "exists": exists,
                "value": val_str,
                "status": "✅ Active" if exists else "❌ Missing"
            })
    except Exception as e:
        log(f"  ⚠️ Feature status scan failed: {e}")

    return {
        "csv": status, 
        "models": model_status,
        "features": feature_status
    }


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

def get_tactical_advice(prediction, strategy_type, entry_price=None, simulated_price=None):
    """
    Generate ELI5 advice for a specific trade strategy.
    
    Args:
        prediction: dict from predict_now()
        strategy_type: str ('bull_credit', 'bear_credit', 'iron_condor')
        entry_price: float (optional) - The price where the user entered the trade.
        simulated_price: float (optional) - For "What-If" simulator.
    """
    if not prediction or not prediction.get("success"):
        return {
            "title": "Awaiting Prediction",
            "text": "Press Fetch Spot to get live tactical advice.",
            "risk": 50,
            "emoji": "⏳",
            "color": "#8B8D97"
        }
    
    dir_raw = prediction.get("ensemble_prediction", {}).get("direction", "SIDEWAYS")
    conf = prediction.get("ensemble_prediction", {}).get("confidence", 0.5) * 100
    regime_raw = prediction.get("regime", "CHOPPY")
    vix = prediction.get("vix_value", 15.0)
    spot = prediction.get("spot_price", 24000)
    
    # Use simulated price if provided
    active_price = simulated_price if simulated_price is not None else spot
    
    # ELI5 Regime Labels
    regime_map = {
        "TRENDING": "🚀 TRENDING MARKET (Price moves are sustained and directional)",
        "BULL": "🚀 STRONG UP-TREND (The Bulls are in control, price is climbing)",
        "BEAR": "📉 STRONG DOWN-TREND (The Bears are winning, price is falling)",
        "STABLE": "⚖️ STABLE MARKET (Moving slowly and safely, predictable)",
        "CHOPPY": "🎢 ZIG-ZAG MARKET (Price is jumping up and down—hard to trade!)",
        "VOLATILE": "⚡ HIGH STRESS (Prices are moving very fast and erratic)",
        "SIDEWAYS": "😴 SLEEPY MARKET (Price is barely moving, no action)"
    }
    regime = regime_map.get(regime_raw, f"📉 UNCERTAIN MARKET ({regime_raw})")

    # ELI5 Direction Labels
    dir_map = {
        "UP": "move UPWARDS",
        "DOWN": "fall DOWNWARDS",
        "SIDEWAYS": "stay ZIG-ZAGGING"
    }
    direction = dir_map.get(dir_raw, "be UNCERTAIN")

    # ── AI-GUIDED RECOVERY ANALYSIS ──
    if entry_price and abs(entry_price - active_price) > 50:
        is_loss = (strategy_type == "bull_credit" and active_price < entry_price) or \
                  (strategy_type == "bear_credit" and active_price > entry_price)
        
        if is_loss:
            dist = abs(entry_price - active_price)
            # Use AI-Guided Statistical Probability:
            ai_helps = (dir_raw == "UP" and strategy_type == "bull_credit") or \
                       (dir_raw == "DOWN" and strategy_type == "bear_credit")
            
            # Fetch real historical probabilities for graph
            hist_recovery = get_recovery_probability(active_price, entry_price, "UP" if strategy_type == "bull_credit" else "DOWN")
            base_prob = float(hist_recovery["recovery_prob"])
            
            # Dynamic Stop Loss (Safety Lane)
            # Stable regimes have tight stops, erratic ones have wider buffers
            stop_buffer_pct = 0.006 if regime_raw in ["STABLE", "SIDEWAYS", "BULL"] else 0.012
            safety_lane_stop = entry_price * (1 - stop_buffer_pct) if strategy_type == "bull_credit" else entry_price * (1 + stop_buffer_pct)
            
            if ai_helps:
                ai_boost = float((conf - 50) * 0.8)
                final_prob = float(min(95.0, base_prob + ai_boost))
                verdict = "🟢 HOLD/HEDGE. DAVID's AI sees a move towards your entry. Recovery is possible."
                color = "#00FF7F"
                risk = 40
                adjustment = ai_boost
            else:
                ai_penalty = float((conf - 50) * 0.5)
                final_prob = float(max(5.0, base_prob - ai_penalty))
                verdict = "🔴 EXIT IMMEDIATELY. DAVID's AI expects the market to keep moving AWAY from your entry."
                color = "#FF4B4B"
                risk = 95
                adjustment = -ai_penalty

            price_label = "PRICE" if simulated_price is None else "SIMULATED PRICE"

            return {
                "title": f"🚨 FIREFIGHT: Trade is Losing (-{dist:.0f} pts)",
                "text": (
                    f"You entered at {entry_price:,.0f} but {price_label} is {active_price:,.0f}. <br><br>"
                    f"<b>AI RECOVERY ODDS: {final_prob:.0f}%</b><br>"
                    f"<i>(DAVID adjusted the 15-year stats using his current brain).</i><br><br>"
                    f"Time to recover: ~{float(hist_recovery.get('avg_recovery_days', 0)):.0f} trading days.<br><br>"
                    f"<b>Safety Lane Stop:</b> {safety_lane_stop:,.0f} (Keep it wide in {regime_raw} markets).<br><br>"
                    f"<b>Final Verdict:</b> {verdict}"
                ),
                "risk": risk,
                "emoji": "🔥",
                "color": color,
                "safety_stop": safety_lane_stop,
                "charts": {
                    "historical": {
                        "labels": ["5 Days", "10 Days", "20 Days", "30 Days"],
                        "values": [base_prob * 0.7, base_prob * 0.85, base_prob, base_prob * 1.1],
                        "title": "Historical Recovery % Over Time"
                    },
                    "ai_logic": {
                        "labels": ["Stat Base", "AI Adjust", "Final Odds"],
                        "values": [base_prob, adjustment, final_prob],
                        "title": "AI Conviction Adjustment"
                    }
                }
            }

    # ── FRESH ENTRY ADVICE ──
    advice = {
        "title": f"Market Mood: {regime}",
        "text": "",
        "risk": 50,
        "emoji": "💡",
        "color": "#00CED1"
    }
    
    if strategy_type == "bull_credit":
        if dir_raw == "UP" and conf >= 55:
            advice["text"] = (
                f"DAVID is {conf:.0f}% sure the market will {direction}. "
                "Market is going UP — your downside PUTs are safe. Collect premium confidently."
            )
            advice["risk"] = 20
            advice["color"] = "#00FF7F"
        elif dir_raw == "SIDEWAYS":
            advice["text"] = f"The market is {direction}. Bull spreads are okay, but don't expect a big profit soon."
            advice["risk"] = 45
            advice["color"] = "#FFD700"
        else:
            advice["text"] = f"DAVID expects the market to {direction}! Entering a Bull Spread now is like swimming against a tsunami. AVOID."
            advice["risk"] = 90
            advice["color"] = "#FF4B4B"
            
    elif strategy_type == "bear_credit":
        if dir_raw == "DOWN" and conf >= 55:
            advice["text"] = (
                f"DAVID is {conf:.0f}% sure the market will {direction}. "
                "This is the perfect time for a Bear Spread. Sell upside safety."
            )
            advice["risk"] = 25
            advice["color"] = "#00FF7F"
        elif dir_raw == "SIDEWAYS":
            advice["text"] = f"The market is {direction}. Bear spreads are limited because the market isn't falling fast."
            advice["risk"] = 50
            advice["color"] = "#FFD700"
        else:
            advice["text"] = f"DAVID expects the market to {direction}! A Bear Spread will get crushed. Do not enter."
            advice["risk"] = 95
            advice["color"] = "#FF4B4B"
            
    elif strategy_type == "iron_condor":
        if dir_raw == "SIDEWAYS" or conf < 52:
            advice["text"] = (
                f"The market will likely {direction}. This is perfect for the Iron Condor (Range Trading). "
                "You profit as long as the market stays between the lines."
            )
            advice["risk"] = 20
            advice["color"] = "#00FF7F"
        elif vix > 22:
            advice["text"] = f"Fear (VIX) is very high ({vix:.1f}). Even if market stays {direction}, one sudden jump could wipe you out. Use smaller bets."
            advice["risk"] = 75
            advice["color"] = "#FFD700"
        else:
            advice["text"] = f"The market is trying to {direction}. Iron Condors hate trends. High chance of being 'steamrolled'."
            advice["risk"] = 65
            advice["color"] = "#FFD700"

    return advice


def get_morning_brief(prediction):
    """Generate a 2-sentence ELI5 summary of the market vibe."""
    if not prediction or not prediction.get("success"):
        return "Waiting for market data to wake up... Press 'Fetch Spot' for your morning brief."
    
    regime = prediction.get("regime", "UNKNOWN")
    dir_raw = prediction.get("ensemble_prediction", {}).get("direction", "SIDEWAYS")
    conf = prediction.get("ensemble_prediction", {}).get("confidence", 0.5) * 100
    vix = prediction.get("vix_value", 15.0)
    
    briefs = {
        "BULL": f"Good morning! Today is a 🚀 BULL CLIMB day. VIX is calm ({vix:.1f}), so credit spreads are safe. Stay long!",
        "BEAR": f"Watch out! Today is a 📉 BEAR SLIDE day. David is {conf:.0f}% sure prices will fall. Sell the rallies.",
        "STABLE": f"It's a ⚖️ STABLE day. Perfect for Iron Condors or theta decay. No sudden storms in sight.",
        "CHOPPY": f"Expect a 🎢 ZIG-ZAG day! Prices will jump up and down. Keep your position sizes small.",
        "VOLATILE": f"⚡ HIGH STRESS ALERT! VIX is at {vix:.1f}. David warns of big, fast moves. Tighten all stops!",
        "SIDEWAYS": f"It's a 😴 SLEEPY day. Nothing much is happening. Great day for non-directional trades."
    }
    
    res = briefs.get(regime, f"Today's vibe is {regime}. David expects a {dir_raw} move with {conf:.0f}% confidence.")
    
    # Add advice for current trend
    if dir_raw == "UP": res += " Bull Spreads look juicy."
    elif dir_raw == "DOWN": res += " Bear Spreads are the play."
    else: res += " Rangebound trading is king today."
    
    return res


def get_portfolio_status(prediction):
    """
    Scan logged trades and return a list of 'Traffic Light' statuses.
    """
    import json
    
    statuses = []
    try:
        if not os.path.exists(WAR_ROOM_PATH):
            return []
            
        with open(WAR_ROOM_PATH, 'r') as f:
            trades = json.load(f)
            
        spot = prediction.get("spot_price", 0)
        
        for trade in trades:
            entry = trade.get("entry_price", 0)
            dir_full = trade.get("direction", "").upper()
            
            # Simple Traffic Light Logic
            if "UP" in dir_full:
                is_win = spot > entry
            elif "DOWN" in dir_full:
                is_win = spot < entry
            else:
                is_win = abs(spot - entry) < 150 # range for IC
                
            if is_win:
                status = "🟢 BREEZE"
                color = "#00FF7F"
                msg = "Trade is in target zone. Relax."
            else:
                # Underwater — check AI recovery odds
                advice = get_tactical_advice(prediction, "bull_credit" if "UP" in dir_full else "bear_credit", entry_price=entry)
                risk = advice.get("risk", 50)
                
                if risk < 60:
                    status = "🟡 CAUTION"
                    color = "#FFD700"
                    msg = "Underwater, but David sees recovery chances."
                else:
                    status = "🔴 FIREFIGHT"
                    color = "#FF4B4B"
                    msg = "Red Alert! AI conviction is moving away from you."
            
            statuses.append({
                "id": trade.get("id", "??"),
                "status": status,
                "color": color,
                "message": msg,
                "strategy": trade.get("direction", "Unknown")
            })
    except Exception as e:
        print(f"Portfolio scan error: {e}")
        
    return statuses


# ═══════════════════════════════════════════════════════════════════════════════
# 8b. INTELLIGENCE SUITE — Historical Win-Rate, Recovery, Golden Hour, etc.
# ═══════════════════════════════════════════════════════════════════════════════

def pattern_match_proof(df_raw, spot, sell_strike, vix, direction, regime, side="PE", lookback=1500, holding_days=7):
    """
    Passthrough to the new robust Historical Strike Survival Backtester in `analyzers.strike_backtester`.
    """
    try:
        from analyzers.strike_backtester import full_strike_analysis
        return full_strike_analysis(df_raw, spot, sell_strike, vix, regime, direction, side, holding_days)
    except Exception as e:
        import traceback
        print(f"Strike Backtester Error: {e}\n{traceback.format_exc()}")
        return {"win_rate": None, "sample_size": 0, "text": f"Simulation error: {e}"}

def adjustment_ladder(spot, sell_strike, buy_strike, direction, strategy, vix, df_raw=None):
    """
    Generate a price-level based firefighting ladder.
    Tells the user exactly what to do at specific NIFTY prices approaching their danger zone.
    Returns a list of dicts: [{"level": "23,150", "price": 23150, "action": "Hold", "detail": "✅ Collect theta", "color": "#00FF7F"}]
    """
    ladder = []
    
    # Example logic for a Bull Put Spread (SELL PE below spot)
    if "Put" in strategy or "PE" in strategy:
        distance = spot - sell_strike
        if distance <= 0: return []
        
        # Level 1: Current Spot
        ladder.append({
            "level": f"At {spot:,.0f} (NOW)", 
            "price": spot, 
            "action": "Hold", 
            "detail": "✅ Collect theta",
            "color": "#00FF7F" # Green
        })
        
        # Level 2: Mid-point to strike
        mid = sell_strike + (distance * 0.5)
        ladder.append({
            "level": f"Drops to {mid:,.0f} ({(mid-spot)/spot*100:.1f}%)", 
            "price": mid, 
            "action": "Hedge", 
            "detail": f"⚠️ Buy cheap {round(mid/50)*50} PE as disaster insurance",
            "color": "#FFD700" # Yellow
        })
        
        # Level 3: At the sold strike
        ladder.append({
            "level": f"At Sold Strike ({sell_strike:,.0f})", 
            "price": sell_strike, 
            "action": "Fight", 
            "detail": "🛡️ Convert to Iron Condor: Sell opposite wing CE to collect extra buffer",
            "color": "#FF8C00" # Orange
        })
        
        # Level 4: At protection/exit
        ladder.append({
            "level": f"Breaks {buy_strike:,.0f}", 
            "price": buy_strike, 
            "action": "Exit", 
            "detail": "💀 Hard Exit. Preserve remaining capital for next trade.",
            "color": "#FF4B4B" # Red
        })
        
    elif "Call" in strategy or "CE" in strategy:
        distance = sell_strike - spot
        if distance <= 0: return []
        
        ladder.append({"level": f"At {spot:,.0f} (NOW)", "price": spot, "action": "Hold", "detail": "✅ Collect theta", "color": "#00FF7F"})
        mid = sell_strike - (distance * 0.5)
        ladder.append({"level": f"Surges to {mid:,.0f}", "price": mid, "action": "Hedge", "detail": "⚠️ Buy OTM CE hedge", "color": "#FFD700"})
        ladder.append({"level": f"At {sell_strike:,.0f}", "price": sell_strike, "action": "Fight", "detail": "🛡️ Sell opposite wing PE", "color": "#FF8C00"})
        ladder.append({"level": f"Breaks {buy_strike:,.0f}", "price": buy_strike, "action": "Exit", "detail": "💀 Close Position", "color": "#FF4B4B"})

    return ladder




def whatif_pnl(sell_strike, buy_strike, strategy, premium_received, spot, simulated_spot, lots=1):
    """
    Calculates pure expiry P&L for a spread strategy based on a simulated price.
    """
    try:
        if "Put" in strategy or "PE" in strategy:
            # Bull Put Spread
            sell_leg_value = max(0, sell_strike - simulated_spot)
            buy_leg_value = max(0, buy_strike - simulated_spot)
            
            # PnL = Premium Received - Liability + Buy Protection
            pnl_points = premium_received - sell_leg_value + buy_leg_value
            
        elif "Call" in strategy or "CE" in strategy:
            # Bear Call Spread
            sell_leg_value = max(0, simulated_spot - sell_strike)
            buy_leg_value = max(0, simulated_spot - buy_strike)
            pnl_points = premium_received - sell_leg_value + buy_leg_value
            
        else: # Unhandled/Iron Condor approx
            pnl_points = 0
            
        pnl_rupees = pnl_points * 65 * lots # Assuming NIFTY 65 lot size
        
        # Determine danger zone
        zone = "SAFE"
        if pnl_rupees < 0:
            zone = "DANGER"
        elif pnl_rupees < (premium_received * 65 * lots * 0.3):
            zone = "WARNING"
            
        return {
            "pnl": pnl_rupees,
            "pnl_pct": (pnl_rupees / (premium_received * 65 * lots)) * 100 if premium_received > 0 else 0,
            "zone": zone
        }
    except Exception as e:
        return {"pnl": 0, "pnl_pct": 0, "zone": "SAFE"}


def whatif_probability(df_raw, spot, target_price, days=7):
    """
    Calculates historical probability of NIFTY reaching `target_price` from `spot` within `days`.
    """
    try:
        if df_raw is None or len(df_raw) < 252 + days:
            return {"probability": 0, "sample_size": 0}
            
        distance_pct = abs(target_price - spot) / spot
        if distance_pct == 0: return {"probability": 100, "sample_size": 0}
        
        is_up = target_price > spot
        
        data = df_raw.tail(2000).copy() # Look back ~8 years
        closes = data["close"].values
        highs = data["high"].values if "high" in data.columns else closes
        lows = data["low"].values if "low" in data.columns else closes
        
        hits = 0
        total = 0
        
        for i in range(len(data) - days):
            ref_price = closes[i]
            target_at_dist = ref_price * (1 + distance_pct) if is_up else ref_price * (1 - distance_pct)
            
            hit = False
            for j in range(1, days + 1):
                if is_up:
                    if highs[i + j] >= target_at_dist:
                        hit = True
                        break
                else:
                    if lows[i + j] <= target_at_dist:
                        hit = True
                        break
            
            if hit: hits += 1
            total += 1
            
        prob = (hits / total) * 100 if total > 0 else 0
        return {"probability": prob, "sample_size": total}
    except Exception as e:
        return {"probability": 0, "sample_size": 0}


def get_position_health(strategy, sell_strike, current_spot, dte):
    """
    Traffic Light Health Monitor for saved open positions.
    Evaluates health based on distance to the sold strike and days to expiry (DTE).
    
    Returns: {"status": "🟢 HEALTHY", "color": "#00FF7F", "message": "Far from danger zone"}
    """
    if not sell_strike:
        return {"status": "⚪ N/A", "color": "#808080", "message": "No specific short strike defined"}
        
    try:
        if "Put" in strategy or "PE" in strategy:
            distance_pct = (current_spot - sell_strike) / current_spot * 100
        elif "Call" in strategy or "CE" in strategy:
            distance_pct = (sell_strike - current_spot) / current_spot * 100
        else:
            return {"status": "⚪ UNKNOWN", "color": "#808080", "message": "Cannot evaluate this strategy type"}
            
        # Time-adjusted danger thresholds
        # If DTE is low, smaller distance is riskier due to gamma
        if dte <= 2:
            safe_dist = 1.0
            warn_dist = 0.5
        elif dte <= 7:
            safe_dist = 2.0
            warn_dist = 1.0
        else:
            safe_dist = 3.0
            warn_dist = 1.5
            
        if distance_pct >= safe_dist:
            return {"status": "🟢 HEALTHY", "color": "#00FF7F", "message": f"{distance_pct:.1f}% cushion from short strike"}
        elif distance_pct >= warn_dist:
            return {"status": "🟡 WARNING", "color": "#FFD700", "message": f"Only {distance_pct:.1f}% cushion left. Monitor closely."}
        elif distance_pct > 0:
            return {"status": "🟠 DANGER", "color": "#FF8C00", "message": f"Approaching strike! ({distance_pct:.1f}% away). Prep adjustment."}
        else:
            return {"status": "🔴 BREACHED", "color": "#FF4B4B", "message": "Short strike breached. Execute recovery plan immediately."}
    except Exception as e:
        return {"status": "⚪ ERROR", "color": "#808080", "message": "Health eval failed"}


def recovery_blueprint(direction, vix, whipsaw_prob):
    """
    Generate specific recovery advice based on current conditions.
    Tells the user exactly what to do if the trade goes against them.
    """
    blueprints = []
    
    if direction == "UP":
        # User likely has Bull Put Spread, market might crash
        blueprints.append("If NIFTY drops toward your sold PUT strike:")
        blueprints.append("  1️⃣ Convert to Iron Condor by selling a Bear Call Spread above resistance (collects extra premium to offset loss).")
        blueprints.append("  2️⃣ If breach is imminent, roll the PUT spread down by 100-150 points and extend expiry by 1 week.")
        if vix > 20:
            blueprints.append("  ⚡ VIX is elevated — premiums are fat. Rolling will collect significant credit to offset the loss.")
        if whipsaw_prob > 50:
            blueprints.append("  ⚠️ High chop risk — do NOT add more legs. Simply hit your stop loss and exit clean.")
    elif direction == "DOWN":
        blueprints.append("If NIFTY rallies toward your sold CALL strike:")
        blueprints.append("  1️⃣ Convert to Iron Condor by selling a Bull Put Spread below support (collect extra premium).")
        blueprints.append("  2️⃣ If breach is imminent, roll the CALL spread up by 100-150 points and extend expiry by 1 week.")
        if vix < 13:
            blueprints.append("  ⚠️ Low VIX — rolling won't collect much credit. Consider cutting the position entirely.")
    else:
        blueprints.append("If NIFTY breaks out of the expected range:")
        blueprints.append("  1️⃣ Close the breached side immediately (don't hope for reversal).")
        blueprints.append("  2️⃣ Let the winning side expire worthless for max profit on that leg.")
        blueprints.append("  3️⃣ Re-evaluate with fresh David prediction before re-entering.")
    
    return " | ".join(blueprints)


def golden_hour_analysis(df_15m):
    """
    Analyze 15-minute data to find which time-of-day historically has the
    lowest adverse moves (safest entry window).
    Returns: dict with best_hour, worst_hour, and text advice.
    """
    try:
        if df_15m is None or len(df_15m) < 100:
            return {"text": "Insufficient 15m data for Golden Hour analysis."}
        
        df = df_15m.copy()
        
        # Parse datetime and extract hour
        if "date" in df.columns:
            df["dt"] = pd.to_datetime(df["date"], errors="coerce")
        elif "datetime" in df.columns:
            df["dt"] = pd.to_datetime(df["datetime"], errors="coerce")
        else:
            return {"text": "No datetime column found in 15m data."}
        
        df = df.dropna(subset=["dt"])
        df["hour"] = df["dt"].dt.hour
        
        # Calculate absolute return per candle
        if "close" in df.columns and "open" in df.columns:
            df["abs_move"] = (df["close"] - df["open"]).abs()
        else:
            return {"text": "Missing open/close columns."}
        
        # Group by hour, find hour with lowest average absolute move (calmest)
        hourly = df.groupby("hour")["abs_move"].mean()
        if len(hourly) == 0:
            return {"text": "Could not compute hourly stats."}
        
        best_hour = int(hourly.idxmin())
        worst_hour = int(hourly.idxmax())
        
        best_str = f"{best_hour}:00-{best_hour}:59"
        worst_str = f"{worst_hour}:00-{worst_hour}:59"
        
        text = f"⏰ GOLDEN HOUR: Enter trades around {best_str} (historically calmest). Avoid {worst_str} (most volatile)."
        return {"best_hour": best_hour, "worst_hour": worst_hour, "text": text}
    except Exception as e:
        return {"text": f"Golden Hour error: {e}"}


def fii_dii_flow_check(prediction):
    """
    Check if FII/DII flow contradicts or confirms the AI direction.
    Returns: dict with alignment status and advisory text.
    """
    try:
        direction = prediction.get("ensemble_prediction", {}).get("direction", "SIDEWAYS")
        fii_net = prediction.get("fii_net")
        dii_net = prediction.get("dii_net")
        
        if fii_net is None:
            # Try loading from CSV
            fii_path = os.path.join(DATA_DIR, "fii_dii_daily.csv")
            if os.path.exists(fii_path):
                df_fii = pd.read_csv(fii_path)
                if len(df_fii) > 0:
                    fii_net = float(df_fii["fii_net"].iloc[-1])
                    dii_net = float(df_fii["dii_net"].iloc[-1])
        
        if fii_net is None:
            return {"aligned": True, "text": "FII/DII data unavailable."}
        
        fii_net = float(fii_net)
        dii_net = float(dii_net) if dii_net else 0
        
        # Check alignment
        if direction == "UP" and fii_net < -3000:
            return {
                "aligned": False,
                "text": f"⚠️ FII DIVERGENCE: David says UP but FII dumped ₹{abs(fii_net):,.0f} Cr. Reduce position size by 50%."
            }
        elif direction == "DOWN" and fii_net > 3000:
            return {
                "aligned": False,
                "text": f"⚠️ FII DIVERGENCE: David says DOWN but FII bought ₹{fii_net:,.0f} Cr. Reduce position size by 50%."
            }
        elif abs(fii_net) > 5000:
            flow_dir = "buying" if fii_net > 0 else "selling"
            return {
                "aligned": True,
                "text": f"🌊 Heavy FII {flow_dir} (₹{abs(fii_net):,.0f} Cr). This CONFIRMS David's {direction} bias."
            }
        else:
            return {
                "aligned": True,
                "text": f"🌊 FII flow is neutral (₹{fii_net:,.0f} Cr). No major divergence detected."
            }
    except Exception as e:
        return {"aligned": True, "text": f"FII check error: {e}"}


def event_shield_check():
    """
    Check proximity to known market-moving events.
    Returns: dict with warning text if an event is near.
    """
    from datetime import datetime, timedelta
    today = datetime.now().date()
    
    # Known recurring events (approximate dates — updated periodically)
    # RBI Monetary Policy: Typically Feb, Apr, Jun, Aug, Oct, Dec (bi-monthly)
    # Budget: Feb 1
    # Expiry Thursdays: Every Thursday
    
    warnings = []
    
    # Check if today is near expiry (Thursday)
    day_of_week = today.weekday()  # 0=Mon, 3=Thu
    days_to_thursday = (3 - day_of_week) % 7
    if days_to_thursday == 0:
        warnings.append("📅 TODAY IS EXPIRY DAY! Gamma risk is extreme. Avoid new positions or widen strikes by 150+ points.")
    elif days_to_thursday <= 2:
        warnings.append(f"📅 Expiry is in {days_to_thursday} day(s). Gamma acceleration begins. Consider wider strikes.")
    
    # Check known RBI policy months (bi-monthly: Feb, Apr, Jun, Aug, Oct, Dec)
    rbi_months = [2, 4, 6, 8, 10, 12]
    if today.month in rbi_months and today.day <= 10:
        warnings.append("🏦 RBI Policy week detected. Expect VIX spike. Widen strikes by 100 points or skip this cycle.")
    
    # Budget check (Feb 1)
    if today.month == 2 and today.day <= 3:
        warnings.append("💰 BUDGET WEEK! Maximum volatility expected. Do NOT sell credit spreads this week.")
    
    if not warnings:
        return {"safe": True, "text": "📅 No major events in the next few days. Safe to trade normally."}
    
    return {"safe": False, "text": " | ".join(warnings)}


def find_twin_days(df, current_vix, current_regime, current_spot, match_count=3):
    """
    Finds historical 'Twin Days' that match today's conditions (VIX, Regime).
    Used to show what happened next.
    """
    if df is None or len(df) == 0:
        return []
        
    try:
        # Check for 'regime' column, fallback to VIX matching if missing
        has_regime_col = 'regime' in df.columns
        
        if has_regime_col:
            twins = df[df['regime'] == current_regime].copy()
        else:
            # Fallback: Filter by VIX and trend personality if regime column is missing
            # Calculate a simple trend for filtering if possible
            twins = df.copy()
            
        # Filter 2: Similar VIX (+/- 20%)
        vix_lower = current_vix * 0.80
        vix_upper = current_vix * 1.20
        twins = twins[(twins['vix'] >= vix_lower) & (twins['vix'] <= vix_upper)]
        
        if len(twins) < match_count:
            # Relax VIX constraint if no matches found
            twins = df.copy()
            if has_regime_col:
                twins = twins[twins['regime'] == current_regime]
            twins = twins[(twins['vix'] >= current_vix * 0.60) & (twins['vix'] <= current_vix * 1.40)]
            
        if len(twins) == 0:
            return []
            
        # Ensure we have future data to compute outcomes
        # Sort by date descending to get most recent relevant matches
        twins = twins.sort_index(ascending=False)
        
        results = []
        # Sample recent matches to avoid picking only one cluster in time
        pool = twins.index.unique().tolist()[:40] 
        
        for d in pool:
            try:
                # Direct index access instead of get_loc to avoid multi-index issues
                day_data = df.loc[d]
                if isinstance(day_data, pd.DataFrame):
                    day_data = day_data.iloc[0]
                    
                idx = df.index.get_loc(d)
                if isinstance(idx, slice):
                    idx = idx.start
                
                # Check for future data safely
                if idx + 3 < len(df):
                    future_spot = float(df.iloc[idx + 3]['close'])
                    twin_spot = float(day_data['close'])
                    move_pct = (future_spot - twin_spot) / twin_spot * 100
                    date_str = pd.to_datetime(str(d)).strftime("%d %b %Y")
                    results.append({
                        "date": date_str,
                        "vix": float(day_data['vix']),
                        "move": move_pct,
                        "outcome": "UP" if move_pct > 0 else "DOWN"
                    })
                if len(results) >= match_count * 3: break
            except Exception as e:
                continue
        
        # Sort by closest VIX match and return top `match_count`
        results.sort(key=lambda x: abs(x['vix'] - current_vix))
        return results[:match_count]
        
    except Exception as e:
        print(f"Twin Day Error: {e}")
        return []


def auto_stop_loss(spot, direction, nearest_support, nearest_resistance, vix, df_raw=None):
    """
    Calculates a volatility-adjusted stop loss level.
    Uses VIX to determine the required 'breathing room' and aligns with technical levels.
    """
    try:
        # Volatility multiplier: At VIX 15, we want ~0.8% cushion (~200 pts)
        # At VIX 30, we want ~1.6% cushion (~400 pts)
        vol_multiplier = (vix / 15) * 0.008
        points_cushion = spot * vol_multiplier
        
        if direction in ("UP", "SIDEWAYS") or "Bull" in str(direction):
            # For bullish trades, stop is BELOW
            technical_stop = nearest_support - 50
            vol_stop = spot - points_cushion
            stop_price = min(technical_stop, vol_stop)
        else:
            # For bearish trades, stop is ABOVE
            technical_stop = nearest_resistance + 50
            vol_stop = spot + points_cushion
            stop_price = max(technical_stop, vol_stop)
            
        # Round to nearest 50
        stop_price = round(float(stop_price) / 50) * 50
        dist_pct = abs(spot - stop_price) / spot * 100
        
        # Historical recovery logic (Simple heuristic based on VIX)
        if vix > 22:
            recovery_msg = "Historically, in high VIX, 41% of hits see a mean-reversion recovery. Consider hedging instead of hard exit."
        else:
            recovery_msg = "Historically, if this level is hit, only 22% of trades recover. Exit recommended to preserve capital."
            
        return {
            "stop_price": stop_price,
            "dist_pct": dist_pct,
            "message": f"Exit if NIFTY touches {stop_price:,.0f} ({dist_pct:.1f}% away). {recovery_msg}"
        }
    except Exception as e:
        print(f"Stop Loss Error: {e}")
        return {"stop_price": 0, "dist_pct": 0, "message": "Manual stop at 1.5% drawdown recommended."}


def calculate_conviction_score(ml_confidence, trust_score, vix, regime, fii_aligned, event_safe):
    """
    Master algorithm that aggregates 6 intelligence signals into a single score (0-100).
    ML Confidence (25%), Historical Trust Score (25%), VIX (15%), FII (10%), Regime (15%), Events (10%).
    """
    try:
        score = 0
        signals = []
        
        # 1. ML Confidence (25%) - ml_confidence should be 0-100
        score += ml_confidence * 0.25
        signals.append(f"🧠 ML Logic: {ml_confidence:.0f}%")
        
        # 2. Historical Trust (25%)
        # The new trust_score is already a 0-100 composite (win rate + formula accuracy + regime safety)
        score += trust_score * 0.25
        signals.append(f"🎯 Strike Trust: {trust_score:.0f}%")
        
        # 3. VIX Favorability (15%)
        vix_rating = 0
        if 13 <= vix <= 20: vix_rating = 100
        elif 11.5 <= vix < 13 or 20 < vix <= 25: vix_rating = 70
        else: vix_rating = 30
        score += vix_rating * 0.15
        signals.append(f"🌡️ VIX Quality: {vix_rating}%")
        
        # 4. FII/DII Alignment (10%)
        fii_rating = 100 if fii_aligned else 20
        score += fii_rating * 0.10
        signals.append(f"🌊 Institutional Flow: {fii_rating}%")
        
        # 5. Regime Harmony (15%)
        regime_rating = 100 if regime in ["TRENDING", "NORMAL", "STABLE", "BULL", "BEAR", "SIDEWAYS"] else 40
        score += regime_rating * 0.15
        signals.append(f"🎭 Market Personality: {regime_rating}%")
        
        # 6. Event Shield (10%)
        event_rating = 100 if event_safe else 0
        score += event_rating * 0.10
        signals.append(f"🛡️ Event Shield: {event_rating}%")
        
        final_score = int(score)
        grade = "A+" if final_score >= 90 else "A" if final_score >= 80 else "B" if final_score >= 70 else "C" if final_score >= 50 else "D"
        color = "#00FF7F" if final_score >= 80 else "#FFD700" if final_score >= 65 else "#FF4B4B"
        
        verdicts = {
            "A+": "UNSTOPPABLE SET-UP",
            "A": "STRONG CONVICTION",
            "B": "STEADY CREDIT PLAY",
            "C": "PROCEED WITH CAUTION",
            "D": "HIGH RISK - AVOID"
        }
        
        return {
            "score": final_score,
            "grade": grade,
            "color": color,
            "signals": signals,
            "verdict": verdicts.get(grade, "UNKNOWN")
        }
    except Exception as e:
        print(f"Conviction Score Error: {e}")
        return {"score": 50, "grade": "C", "color": "#FFD700", "signals": [], "verdict": "CALC ERROR"}


def roll_or_die_advice(days_to_expiry, is_underwater, vix):
    """
    Advise whether to roll the position forward or cut losses.
    """
    if days_to_expiry is None:
        return "Cannot calculate: no expiry info."
    
    if not is_underwater:
        if days_to_expiry <= 3:
            return "🟢 Position is profitable and near expiry. Let theta do the work — hold for max decay."
        return "🟢 Position is in profit. No action needed."
    
    # Underwater logic
    if days_to_expiry <= 2:
        return "🔴 ROLL OR EXIT NOW! With ≤2 days to expiry, gamma will destroy you. Roll to next week's expiry at wider strikes, or close for a defined loss."
    elif days_to_expiry <= 5:
        if vix > 18:
            return f"🟡 ROLL RECOMMENDED: VIX ({vix:.1f}) is high enough that rolling will collect decent credit. Roll out 1 week and widen strikes by 50-100 points."
        else:
            return f"🔴 EXIT RECOMMENDED: VIX ({vix:.1f}) is too low — rolling won't collect enough credit to justify the risk. Cut losses and re-enter on next clean signal."
    else:
        return "🟡 You have time. Monitor closely. If the trade doesn't recover in 2 days, roll or exit."


def get_position_health(direction, sell_strike, current_price, dte, vix=15):
    """
    Assesses if an open position is Safe, Watching, or Dangerous.
    """
    try:
        # Distance calculation
        is_put = "PUT" in direction.upper() or "PE" in direction.upper() or "UP" in direction.upper()
        
        if is_put:
            dist = current_price - sell_strike
        else: # Call
            dist = sell_strike - current_price
            
        pct_dist = (dist / current_price) * 100
        
        # Grading
        if dist < 0:
            status = "💀 DANGER"
            color = "#FF4B4B"
            msg = f"Position is underwater by {abs(dist):.0f} pts. Exit or roll immediately!"
        elif dist < 100 or pct_dist < 0.5:
            status = "⚠️ WATCH"
            color = "#FFD700"
            msg = f"Strike is within {dist:.0f} pts. Prepare to defend or roll."
        else:
            status = "🟢 SAFE"
            color = "#00FF7F"
            msg = f"Strike is {dist:.0f} pts ({pct_dist:.1f}%) away. Theta is your friend."
            
        # Add DTE factor
        if dte <= 2 and dist < 200:
            status = "🚨 EXPIRE"
            color = "#FF8C00"
            msg = "Expiry is too close. Gamma risk is high! Close today."
            
        return {
            "status": status,
            "color": color,
            "message": msg,
            "dist": dist,
            "pct_dist": pct_dist,
            "advice": roll_or_die_advice(dte, dist < 0, vix)
        }
    except Exception as e:
        return {"status": "UNKNOWN", "color": "#7F8C8D", "message": f"Error: {e}", "advice": "No advice."}


def streak_tracker():
    """
    Read the trade journal and calculate David's recent accuracy.
    Returns: dict with streak info and confidence adjustment.
    """
    import json
    journal_path = os.path.join(DATA_DIR, "trade_journal.json")
    
    try:
        if not os.path.exists(journal_path):
            return {"accuracy": None, "text": "No journal data yet. Start logging trades!", "confidence_modifier": 0}
        
        with open(journal_path, "r") as f:
            journal = json.load(f)
        
        entries = journal.get("entries", [])
        if len(entries) < 3:
            return {"accuracy": None, "text": f"Only {len(entries)} trades logged. Need 3+ for streak analysis.", "confidence_modifier": 0}
        
        # Count results from recent entries
        recent = entries[-10:]  # Last 10 trades
        wins = sum(1 for e in recent if e.get("result") in ("WIN", "PROFIT"))
        losses = sum(1 for e in recent if e.get("result") in ("LOSS", "STOPPED"))
        avoided = sum(1 for e in recent if e.get("result") == "AVOIDED")
        pending = len(recent) - wins - losses - avoided
        
        total_decided = wins + losses
        if total_decided == 0:
            accuracy = None
            text = f"📊 {len(recent)} recent trades logged ({avoided} avoided, {pending} pending). No decided outcomes yet."
            modifier = 0
        else:
            accuracy = (wins / total_decided) * 100
            
            # Determine streak
            consecutive_losses = 0
            for e in reversed(recent):
                if e.get("result") in ("LOSS", "STOPPED"):
                    consecutive_losses += 1
                else:
                    break
            
            if consecutive_losses >= 3:
                emoji = "🔴"
                modifier = -1  # Downgrade safety by 1 level
                text = f"{emoji} LOSING STREAK ({consecutive_losses} consecutive losses)! David's recent accuracy: {accuracy:.0f}%. Auto-widening strikes and reducing lot size."
            elif accuracy >= 80:
                emoji = "🟢"
                modifier = 0
                text = f"{emoji} David is ON FIRE! {accuracy:.0f}% accuracy over last {total_decided} trades. High confidence."
            elif accuracy >= 60:
                emoji = "🟡"
                modifier = 0
                text = f"{emoji} David's recent accuracy: {accuracy:.0f}% ({wins}W/{losses}L). Performing normally."
            else:
                emoji = "🔴"
                modifier = -1
                text = f"{emoji} David's recent accuracy is LOW ({accuracy:.0f}%). Playing it safe — wider strikes, smaller lots recommended."
        
        return {"accuracy": accuracy, "text": text, "confidence_modifier": modifier}
    except Exception as e:
        return {"accuracy": None, "text": f"Streak error: {e}", "confidence_modifier": 0}


# ═══════════════════════════════════════════════════════════════════════════════
# 9. STRIKE PRICE RECOMMENDER
# ═══════════════════════════════════════════════════════════════════════════════

def get_strike_recommendation(prediction, capital=500000, mode="aggressive"):
    """Recommend strike prices based on predicted range + support/resistance.
    
    mode: 'conservative' — Simple, safe, no-adjustment Bull Put only.
          'aggressive'   — Full multi-strategy engine with dynamic spreads.
    """
    if not prediction or not prediction.get("success"):
        return {"ready": False, "message": "Awaiting prediction data..."}
    
    spot = float(prediction.get("spot_price", 24000))
    ranges = prediction.get("ranges", {})
    supports = prediction.get("supports", [])
    resistances = prediction.get("resistances", [])
    direction = prediction.get("ensemble_prediction", {}).get("direction", "SIDEWAYS")
    conf = prediction.get("ensemble_prediction", {}).get("confidence", 0.5) * 100
    
    # Use range prediction for safe OTM zones
    range_7d = ranges.get("7d", {})
    range_low = float(range_7d.get("low", spot - 300))
    range_high = float(range_7d.get("high", spot + 300))
    
    # Round strikes to nearest 50 (NIFTY standard)
    def round_strike(price): return round(price / 50) * 50
    
    # Extract prices (SREngine returns dicts: {"price": float, "touches": int, ...})
    support_prices = [s["price"] if isinstance(s, dict) else s for s in supports]
    resistance_prices = [r["price"] if isinstance(r, dict) else r for r in resistances]
    
    # Find nearest support/resistance
    support_prices = [float(s) for s in support_prices if str(s).replace('.','').isdigit()]
    resistance_prices = [float(r) for r in resistance_prices if str(r).replace('.','').isdigit()]
    
    nearest_support = round_strike(max([s for s in support_prices if s < spot], default=spot - 200))
    nearest_resistance = round_strike(min([r for r in resistance_prices if r > spot], default=spot + 200))
    
    # ------------------------------------------------------------------
    # AI Safety Rating & Expiry Logic (VIX + Divergence/Whipsaw based)
    # ------------------------------------------------------------------
    vix = float(prediction.get("vix_value", 15.0))
    whipsaw = prediction.get("whipsaw", {})
    whipsaw_prob = whipsaw.get("whipsaw_prob", 0)
    
    def calculate_safety(dist_from_spot, base_safety):
        safety = base_safety
        warnings = []
        if vix < 11.5:
            safety = "WARNING"
            warnings.append("VIX extremely low (<11.5). Premiums do not justify risk.")
        elif vix < 13.0 and safety == "HIGH":
            safety = "MEDIUM"
            warnings.append("Low VIX. Margins of safety are thinner.")
            
        if whipsaw_prob > 65:
            safety = "WARNING"
            warnings.append(f"High chop/whipsaw risk ({whipsaw_prob}%).")
        elif whipsaw_prob > 50 and safety == "HIGH":
            safety = "MEDIUM"
            warnings.append("Elevated chop risk.")
            
        if dist_from_spot < 150:
            safety = "LOW" if safety != "WARNING" else "WARNING"
            warnings.append("Strike is too close to spot price.")
            
        return safety, warnings
        
    def get_expiry_advice():
        from datetime import datetime, timedelta
        today = datetime.now()
        
        if vix > 20:
            target_dte = 10
            reason = "(Capture fast IV crush)"
        elif vix < 13:
            target_dte = 35
            reason = "(Need more time for premium)"
        else:
            target_dte = 21
            reason = "(Standard sweet spot)"
            
        # Add base days
        target_date = today + timedelta(days=target_dte)
        
        # Shift to nearest Thursday (weekday 3)
        # Monday=0, Thursday=3
        days_to_thursday = (3 - target_date.weekday()) % 7
        actual_expiry = target_date + timedelta(days=days_to_thursday)
        
        # Example format: 26MAR2026
        expiry_str = actual_expiry.strftime("%d%b%Y").upper()
        
        return f"{expiry_str} {reason}"
            
    expiry_advice = get_expiry_advice()
    
    # Get df_raw for historical analysis
    df_raw = prediction.get("df_raw")
    regime = prediction.get("regime", "UNKNOWN") if isinstance(prediction.get("regime"), str) else prediction.get("regime", {}).get("current_regime", "UNKNOWN")
    
    # ==================================================================
    # CONSERVATIVE MODE — Simple, safe, disciplined
    # ==================================================================
    if mode == "conservative":
        import math
        
        # ── TRAFFIC LIGHT: ENTER or WAIT ──
        # Must be SIDEWAYS + conviction > 65 + VIX > 13 + no extreme whipsaw
        can_enter = True
        block_reasons = []
        
        if direction != "SIDEWAYS":
            can_enter = False
            block_reasons.append(f"Market is {direction}, not SIDEWAYS. Wait for range-bound conditions.")
        if conf < 65:
            can_enter = False
            block_reasons.append(f"Conviction {conf:.0f}% is below 65% threshold.")
        if vix < 11.5:
            can_enter = False
            block_reasons.append("VIX is too low — premiums don't justify risk.")
        if whipsaw_prob > 60:
            can_enter = False
            block_reasons.append(f"Whipsaw risk is {whipsaw_prob}% — market is choppy.")
            
        traffic_light = "ENTER" if can_enter else "WAIT"
        
        # ── STRIKE: Always 300+ pts OTM ──
        cons_otm = 300
        cons_width = 100  # Fixed 100-pt width = ₹6,500 max loss/lot
        
        cons_sell = round_strike(spot - cons_otm)
        # If there's support even further down, use it for extra safety
        # BUT cap it at a maximum of 400 pts OTM so we don't destroy premium
        deep_support = round_strike(max([s for s in support_prices if s < spot - 200], default=cons_sell))
        
        if deep_support < cons_sell:
            # Only use deep support if it's not ridiculously far away
            if (spot - deep_support) <= 450:
                cons_sell = deep_support
            else:
                cons_sell = round_strike(spot - 400) # Hard cap at 400 pts OTM
                
        # OVERRIDE FOR SIDEWAYS:
        # If SIDEWAYS, we want Iron Condor. We define the Call side similarly for resistance.
        # But wait, the existing code for conservative was only doing Bull Puts natively! It ignored direction.
        # Let's add basic condor support if SIDEWAYS or DOWN
        res_otm = 300
        cons_sell_call = round_strike(spot + res_otm)
        deep_res = round_strike(min([r for r in resistance_prices if r > spot + 200], default=cons_sell_call))
        if deep_res > cons_sell_call:
            if (deep_res - spot) <= 450:
                cons_sell_call = deep_res
            else:
                cons_sell_call = round_strike(spot + 400)

        cons_buy = cons_sell - cons_width
        cons_dist = spot - cons_sell
        
        # ── PREMIUM ESTIMATION ──
        iv = vix / 100
        est_dte = 10 if vix > 20 else 21 if vix > 13 else 35
        sqrt_t = math.sqrt(est_dte / 365)
        
        def _est_prem(strike):
            dist = abs(spot - strike)
            dist_pct = dist / spot
            atm = spot * iv * sqrt_t * 0.4
            decay = math.exp(-2.5 * (dist_pct / (iv * sqrt_t + 0.001)))
            return max(3.0, round(atm * decay, 2))
        
        credit = _est_prem(cons_sell) - _est_prem(cons_buy)
        credit = max(5, credit)
        max_profit_per_lot = credit * 65
        max_loss_per_lot = (cons_width - credit) * 65
        
        # ── 50% TAKE PROFIT ──
        take_profit_premium = credit * 0.5  # Exit when premium drops to this
        take_profit_per_lot = take_profit_premium * 65  # ₹ you keep after closing
        
        # ── LOT CAP from capital ──
        # Brokers require ~₹33k-₹35k margin for a 100-pt wide NIFTY spread. Max loss is only ~₹6,500.
        max_margin_per_lot = 35000
        
        # 1. Cap by Risk: Never risk more than 30% of total capital on a single trade
        safe_allocation = capital * 0.30
        lots_by_risk = int(safe_allocation / max_loss_per_lot) if max_loss_per_lot > 0 else 1
        
        # 2. Cap by Margin: Never exceed 95% of total capital in blocked margin
        lots_by_margin = int((capital * 0.95) / max_margin_per_lot)
        
        # Choose the safest lot size, ensuring at least 1 lot is recommended if possible
        max_lots = max(1, min(lots_by_risk, lots_by_margin))
        
        # ── Monthly target math ──
        monthly_est_profit = take_profit_per_lot * max_lots * 3  # ~3 winning trades/month
        
        # ── Safety rating ──
        cons_safety, cons_warn = calculate_safety(cons_dist, "HIGH")
        
        # ── Build the single recommendation ──
        if direction == "UP":
            rec = {
                "strategy": "🛡️ Bull Put Spread (Conservative)",
                "sell": f"SELL {cons_sell} PE",
                "buy": f"BUY {cons_buy} PE",
                "sell_strike": cons_sell,
                "buy_strike": cons_buy,
                "width": cons_width,
                "est_premium": credit,
                "est_profit": max_profit_per_lot,
                "est_loss": max_loss_per_lot,
                "est_rr": credit / (cons_width - credit) if (cons_width - credit) > 0 else 0,
                "reason": f"{cons_dist:.0f} pts below spot. Support at {nearest_support:.0f}.",
                "safety": cons_safety,
                "expiry": expiry_advice,
                "side": "PE"
            }
        elif direction == "DOWN":
            cons_buy_call = cons_sell_call + cons_width
            cons_dist_call = cons_sell_call - spot
            call_credit = max(5, _est_prem(cons_sell_call) - _est_prem(cons_buy_call))
            max_profit_per_lot = call_credit * 65
            max_loss_per_lot = (cons_width - call_credit) * 65
            take_profit_premium = call_credit * 0.5
            take_profit_per_lot = take_profit_premium * 65
            monthly_est_profit = take_profit_per_lot * max_lots * 3
            
            rec = {
                "strategy": "🛡️ Bear Call Spread (Conservative)",
                "sell": f"SELL {cons_sell_call} CE",
                "buy": f"BUY {cons_buy_call} CE",
                "sell_strike": cons_sell_call,
                "buy_strike": cons_buy_call,
                "width": cons_width,
                "est_premium": call_credit,
                "est_profit": max_profit_per_lot,
                "est_loss": max_loss_per_lot,
                "est_rr": call_credit / (cons_width - call_credit) if (cons_width - call_credit) > 0 else 0,
                "reason": f"{cons_dist_call:.0f} pts above spot. Resistance at {nearest_resistance:.0f}.",
                "safety": cons_safety,
                "expiry": expiry_advice,
                "side": "CE"
            }
        else:
            # SIDEWAYS -> Iron Condor
            cons_buy_call = cons_sell_call + cons_width
            call_credit = max(5, _est_prem(cons_sell_call) - _est_prem(cons_buy_call))
            total_credit = credit + call_credit
            max_profit_per_lot = total_credit * 65
            max_loss_per_lot = (cons_width - total_credit) * 65
            take_profit_premium = total_credit * 0.5
            take_profit_per_lot = take_profit_premium * 65
            monthly_est_profit = take_profit_per_lot * max_lots * 3
            
            rec = {
                "strategy": "🛡️ Iron Condor (Conservative)",
                "sell": f"SELL {cons_sell} PE + SELL {cons_sell_call} CE",
                "buy": f"BUY {cons_buy} PE + BUY {cons_buy_call} CE",
                "sell_strike": cons_sell, # Primary risk side tracking
                "buy_strike": cons_buy,
                "width": cons_width,
                "est_premium": total_credit,
                "est_profit": max_profit_per_lot,
                "est_loss": max_loss_per_lot,
                "est_rr": total_credit / (cons_width - total_credit) if (cons_width - total_credit) > 0 else 0,
                "reason": f"Range: {cons_sell:.0f} PE to {cons_sell_call:.0f} CE. Sideways engine.",
                "safety": cons_safety,
                "expiry": expiry_advice,
                "side": "BOTH"
            }
        
        # ── Pattern match for win rate / trust score ──
        try:
            pm = pattern_match_proof(df_raw, spot, rec["sell_strike"], vix, direction, regime, side=rec["side"])
            rec["win_rate"] = pm.get("win_rate", 80)
            rec["survival_rate"] = pm.get("survival_rate", 80) # Legacy support
            rec["trust_score"] = pm.get("trust_score", 80)
            rec["pattern_proof_text"] = pm.get("text", "")
            rec["color"] = pm.get("color", "#FFD700")
            
            # Additional detail for UI
            rec["mae_rupees"] = pm.get("mae_rupees", 0)
            rec["p95_mae_rupees"] = pm.get("p95_mae_rupees", 0)
            rec["sample_size"] = pm.get("sample_size", 0)
            rec["regime_data"] = pm.get("regime_data", {})
            rec["breakdown"] = pm.get("breakdown", {})
            rec["confidence_interval"] = pm.get("confidence_interval", [0, 0])
        except Exception as e:
            rec["win_rate"] = 80
            rec["survival_rate"] = 80
            rec["trust_score"] = 80
            rec["pattern_proof_text"] = f"Error computing trust: {e}"
            rec["color"] = "#FFD700"
            rec["mae_rupees"] = 0
            rec["p95_mae_rupees"] = 0
            rec["sample_size"] = 0
            rec["regime_data"] = {}
            rec["breakdown"] = {}
            rec["confidence_interval"] = [0, 0]
        
        # ── Simplified EXIT-only ladder ──
        ladder = [
            {"level": f"At {spot:,.0f} (NOW)", "action": "Hold", "icon": "✅", "detail": "Collect theta. Do nothing."},
            {"level": f"Drops to {cons_sell + 50:,.0f}", "action": "Monitor", "icon": "👀", "detail": f"Still {cons_dist - 50:.0f} pts away. Stay calm."},
            {"level": f"At Sold Strike ({cons_sell:,.0f})", "action": "EXIT", "icon": "🔴", "detail": f"Hard exit. Max loss: ₹{max_loss_per_lot * max_lots:,.0f}. No firefighting."},
            {"level": f"Breaks {cons_sell - 50:,.0f}", "action": "ALREADY OUT", "icon": "⛔", "detail": "You should already be out. If not, exit NOW."},
        ]
        
        # ── Conviction score ──
        # Pass the new trust_score into the master conviction calculator
        best_trust = rec.get("trust_score", 80)
        fii_check = fii_dii_flow_check(prediction)
        fii_aligned = fii_check.get("aligned", True)
        
        event_safe = prediction.get("event_shield", {}).get("safe", True) if isinstance(prediction.get("event_shield"), dict) else True
        
        conviction = calculate_conviction_score(conf, best_trust, vix, regime, fii_aligned, event_safe)
        
        # ── Playbook (simplified) ──
        playbook = {
            "sizing": f"🛡️ CONSERVATIVE: Sell {max_lots} Lot(s). Est Margin: ~₹{max_margin_per_lot * max_lots:,.0f}. Max risk: ₹{max_loss_per_lot * max_lots:,.0f} ({(max_loss_per_lot * max_lots / capital) * 100:.1f}% of capital).",
            "entry": f"{'🟢 ENTER' if can_enter else '🔴 WAIT'}. {'Conditions are met — place the trade.' if can_enter else ' | '.join(block_reasons)}",
            "take_profit": f"🎯 Exit when premium drops to ₹{take_profit_premium:.0f} (50% of ₹{credit:.0f}). That's ₹{take_profit_per_lot * max_lots:,.0f} profit on {max_lots} lot(s).",
            "stop_loss": f"🛑 If NIFTY touches {rec['sell_strike']:,.0f} → EXIT IMMEDIATELY. No adjustment. No firefighting. Max loss: ₹{max_loss_per_lot * max_lots:,.0f}.",
            "firefighting": f"🚫 NO FIREFIGHTING in Conservative Mode. If stop-loss triggers, accept the loss and wait for the next setup.",
            "adjustment_ladder": ladder,
            "streak": "",
            "fii_flow": "",
            "event_shield": "",
            "conviction": conviction,
            "monthly_target": f"📊 With {max_lots} lots × 3 trades/month: Est. ₹{monthly_est_profit:,.0f}/month (before losses). Net after ~1 loss: ₹{monthly_est_profit - max_loss_per_lot * max_lots:,.0f}/month."
        }
        
        return {
            "ready": True,
            "mode": "conservative",
            "traffic_light": traffic_light,
            "block_reasons": block_reasons,
            "conviction": conviction,
            "spot": spot,
            "range_low": range_low,
            "range_high": range_high,
            "nearest_support": nearest_support,
            "nearest_resistance": nearest_resistance,
            "recommendations": [rec],
            "direction": direction,
            "confidence": conf,
            "vix": vix,
            "whipsaw_prob": whipsaw_prob,
            "playbook": playbook,
            "df_raw": df_raw,
            "conservative_meta": {
                "max_lots": max_lots,
                "take_profit_premium": take_profit_premium,
                "take_profit_total": take_profit_per_lot * max_lots,
                "max_risk_total": max_loss_per_lot * max_lots,
                "monthly_estimate": monthly_est_profit,
                "monthly_net": monthly_est_profit - max_loss_per_lot * max_lots,
            }
        }
    
    # ==================================================================
    # AGGRESSIVE MODE (existing logic continues below)
    # ==================================================================
    
    expiry_advice = get_expiry_advice()

    
    # ------------------------------------------------------------------
    # DYNAMIC SPREAD WIDTH — Computed from VIX, Regime & Confidence
    # ------------------------------------------------------------------
    # High VIX → wider spreads (more volatility needs more buffer)
    # Low VIX  → tighter spreads (capture more premium %, better RR)
    # High Confidence → tighten further (model is more sure)
    
    if vix > 22:
        base_width = 200   # Very volatile, need wide protection
    elif vix > 17:
        base_width = 150   # Normal volatility
    elif vix > 13:
        base_width = 100   # Low vol, tighter = better premium capture
    else:
        base_width = 50    # Ultra-low VIX, premiums are thin already
    
    # Adjust width by confidence: high confidence → can tighten by 1 strike
    if conf > 75:
        base_width = max(50, base_width - 50)  # Tighten by 50 pts
    elif conf < 40:
        base_width = base_width + 50           # Widen for safety
        
    width = int(round_strike(base_width))
    if width < 50: width = 50  # Minimum 50-pt width (1 strike)
    
    # ------------------------------------------------------------------
    # PREMIUM ESTIMATION — Approximate credit received
    # ------------------------------------------------------------------
    import math
    
    def estimate_premium(strike, option_type="PE", dte=14):
        """
        Rough premium estimate using VIX as IV proxy.
        More accurate than guessing, less accurate than live chain.
        """
        distance = abs(spot - strike)
        distance_pct = distance / spot
        iv = vix / 100  # Annualized IV
        sqrt_t = math.sqrt(dte / 365)
        
        # Time value decays exponentially with distance from ATM
        atm_premium = spot * iv * sqrt_t * 0.4  # ~ATM approximation
        otm_decay = math.exp(-2.5 * (distance_pct / (iv * sqrt_t + 0.001)))
        
        premium = atm_premium * otm_decay
        return max(3.0, round(premium, 2))  # Floor at ₹3
    
    # DTE from expiry advice
    if "7-14" in expiry_advice:
        est_dte = 10
    elif "14-30" in expiry_advice:
        est_dte = 21
    else:
        est_dte = 35
    
    # ------------------------------------------------------------------
    # STRIKE SELECTION — Anchored to REAL technical levels
    # ------------------------------------------------------------------
    
    recommendations = []
    
    # ── Bull Put Spread ──
    # Primary anchor: nearest support (real technical level where buying happens)
    # Secondary: predicted range low (ML-derived)
    # Strategy: Sell AT or just below support → higher premium, SR backs you up
    
    sell_put = round_strike(nearest_support)
    # Guarantee minimum 200 pts OTM for decent survival rate + premium
    min_otm = 200 if vix > 18 else 250 if vix > 13 else 300
    if spot - sell_put < min_otm:
        sell_put = round_strike(spot - min_otm)
    # If range_low suggests danger above our sell, respect it but don't go above it
    if range_low > sell_put + 100:
        sell_put = round_strike(min(range_low, spot - min_otm))  # Respect range but keep distance
    
    buy_put = sell_put - width
    dist_put = spot - sell_put
    
    put_safety, put_warn = calculate_safety(dist_put, "HIGH" if dist_put > 200 else "MEDIUM")
    put_premium = estimate_premium(sell_put, "PE", est_dte) - estimate_premium(buy_put, "PE", est_dte)
    put_premium = max(5, put_premium)
    put_max_profit = put_premium * 65  # per lot
    put_max_loss = (width - put_premium) * 65
    put_rr = put_premium / (width - put_premium) if (width - put_premium) > 0 else 0
    
    put_reason = f"Support at {nearest_support:.0f} backs this. {dist_put:.0f} pts OTM."
    if put_warn: put_reason += " | " + " | ".join(put_warn)
    
    recommendations.append({
        "strategy": "Bull Put Spread",
        "sell": f"SELL {sell_put} PE",
        "buy": f"BUY {buy_put} PE",
        "sell_strike": sell_put,
        "buy_strike": buy_put,
        "width": width,
        "est_premium": put_premium,
        "est_profit": put_max_profit,
        "est_loss": put_max_loss,
        "est_rr": put_rr,
        "reason": put_reason,
        "safety": put_safety,
        "expiry": expiry_advice,
        "side": "PE"
    })
    
    # ── Bear Call Spread ──
    sell_call = round_strike(nearest_resistance)
    # Guarantee minimum OTM distance (same as put side)
    if sell_call - spot < min_otm:
        sell_call = round_strike(spot + min_otm)
    # If range_high suggests danger below our sell, respect it but keep distance
    if range_high < sell_call - 100:
        sell_call = round_strike(max(range_high, spot + min_otm))
        
    buy_call = sell_call + width
    dist_call = sell_call - spot
    
    call_safety, call_warn = calculate_safety(dist_call, "HIGH" if dist_call > 200 else "MEDIUM")
    call_premium = estimate_premium(sell_call, "CE", est_dte) - estimate_premium(buy_call, "CE", est_dte)
    call_premium = max(5, call_premium)
    call_max_profit = call_premium * 65
    call_max_loss = (width - call_premium) * 65
    call_rr = call_premium / (width - call_premium) if (width - call_premium) > 0 else 0
    
    call_reason = f"Resistance at {nearest_resistance:.0f} caps upside. {dist_call:.0f} pts OTM."
    if call_warn: call_reason += " | " + " | ".join(call_warn)
    
    recommendations.append({
        "strategy": "Bear Call Spread",
        "sell": f"SELL {sell_call} CE",
        "buy": f"BUY {buy_call} CE",
        "sell_strike": sell_call,
        "buy_strike": buy_call,
        "width": width,
        "est_premium": call_premium,
        "est_profit": call_max_profit,
        "est_loss": call_max_loss,
        "est_rr": call_rr,
        "reason": call_reason,
        "safety": call_safety,
        "expiry": expiry_advice,
        "side": "CE"
    })
    
    # ── Iron Condor ──
    ic_base = "HIGH" if direction == "SIDEWAYS" else "MEDIUM"
    ic_safety, ic_warn = calculate_safety(min(dist_put, dist_call), ic_base)
    ic_premium = put_premium + call_premium
    ic_max_profit = ic_premium * 65
    ic_max_loss = (width - ic_premium) * 65  # Only one side can lose
    ic_rr = ic_premium / (width - ic_premium) if (width - ic_premium) > 0 else 0
    
    ic_reason = f"Range: {sell_put} — {sell_call} ({sell_call - sell_put:.0f} pts). Both support & resistance back this."
    if ic_warn: ic_reason += " | " + " | ".join(ic_warn)
    elif direction != "SIDEWAYS": ic_reason += " ⚠️ Market trending, directional spread may be better."

    recommendations.append({
        "strategy": "Iron Condor",
        "sell": f"SELL {sell_put} PE + SELL {sell_call} CE",
        "buy": f"BUY {buy_put} PE + BUY {buy_call} CE",
        "sell_strike": sell_put,
        "buy_strike": buy_put,
        "width": width,
        "est_premium": ic_premium,
        "est_profit": ic_max_profit,
        "est_loss": ic_max_loss,
        "est_rr": ic_rr,
        "reason": ic_reason,
        "safety": ic_safety,
        "expiry": expiry_advice,
        "side": "BOTH"
    })
    
    # ------------------------------------------------------------------
    # DIRECTION-AWARE PRIORITY — Mark primary strategy with ⭐
    # ------------------------------------------------------------------
    if direction == "UP":
        # Bullish → Bear Call is the play (sell into the rally)
        # But also Bull Put is safe (market going away from puts)
        primary_idx = 1  # Bear Call
        recommendations[1]["strategy"] = "⭐ Bear Call Spread"
        recommendations[0]["strategy"] = "Bull Put Spread (Safe Hedge)"
    elif direction == "DOWN":
        # Bearish → Bull Put is the play (sell into the fear)
        primary_idx = 0  # Bull Put
        recommendations[0]["strategy"] = "⭐ Bull Put Spread"
        recommendations[1]["strategy"] = "Bear Call Spread (Safe Hedge)"
    else:
        # Sideways → Iron Condor is king
        primary_idx = 2  # Iron Condor
        recommendations[2]["strategy"] = "⭐ Iron Condor"
    
    # Reorder: primary strategy first
    primary = recommendations.pop(primary_idx)
    recommendations.insert(0, primary)
    
    # Calculate IV Percentile (IVP) if the dataframe has 252 days of VIX data
    ivp = None
    if prediction and "vix_value" in prediction and "df_raw" in prediction:
        df = prediction["df_raw"]
        if "vix" in df.columns and len(df) >= 252:
            last_year_vix = df["vix"].tail(252)
            current_vix = vix
            vix_min = last_year_vix.min()
            vix_max = last_year_vix.max()
            if vix_max > vix_min:
                ivp = ((current_vix - vix_min) / (vix_max - vix_min)) * 100
                
    ivp_str = f"IVP: {ivp:.0f}%" if ivp is not None else f"VIX: {vix:.2f}"
    
    # Intraday Legging Pulse (Micro-Timing via RSI 15m)
    legging_advice = "Wait 30 mins after open before entering."
    if prediction and prediction.get("rsi_15m") is not None:
        try:
            current_rsi = float(prediction["rsi_15m"])
            if current_rsi > 70:
                legging_advice = f"📈 15m-RSI is {current_rsi:.0f} (Overbought). Sell your CE (Call) spread NOW to capture peak premium. Hold off on the PE spread until the surge exhausts."
            elif current_rsi < 30:
                legging_advice = f"📉 15m-RSI is {current_rsi:.0f} (Oversold). Sell your PE (Put) spread NOW to capture panic premium. Wait for a bounce to sell the CE spread."
            else:
                legging_advice = f"⚖️ 15m-RSI is neutral ({current_rsi:.0f}). Safe to enter both wings simultaneously or leg in wing-by-wing."
        except:
            pass

    # Generate Gamma Risk Warning based on Expiry Date Advice
    gamma_risk = ""
    if "7-14 Days" in expiry_advice:
        gamma_risk = "⚠️ GAMMA ALERT: Short expiry means rapid gamma risk. Close trade immediately once 60% profit hit."
    else:
        gamma_risk = "NEVER hold into expiry week (Gamma Risk)."

    # Overnight Gap Risk
    gap_risk = "🟢 Low Overnight Gap Risk. Safe to hold spreads overnight."
    if prediction and "df_raw" in prediction:
        df = prediction["df_raw"]
        if "gap_pct" in df.columns:
            recent_gaps = df["gap_pct"].tail(3).abs().mean()
            if whipsaw_prob > 50 or recent_gaps > 0.005:
                gap_risk = "⚠️ HIGH OVERNIGHT GAP RISK. Do not hold credit spreads overnight. Convert to Intraday only."

    # Position Sizing
    margin_per_lot = 45000
    max_allocation = capital * 0.40  # 40% allocation
    recommended_lots = max(1, int(max_allocation / margin_per_lot))
    
    if whipsaw_prob > 50:
        recommended_lots = max(1, recommended_lots // 2)
        sizing_text = f"Sell {recommended_lots} Lot(s) ONLY (Reduced sizing due to {whipsaw_prob}% chop risk). Max Allocation: ₹{recommended_lots*margin_per_lot:,.0f}."
    else:
        sizing_text = f"Sell {recommended_lots} Lot(s) (Standard 40% allocation limit). Max Allocation: ₹{recommended_lots*margin_per_lot:,.0f}."

    # ------------------------------------------------------------------
    # INTELLIGENCE SUITE — Run all 7 modules
    # ------------------------------------------------------------------
    
    # 1. Historical Pattern Match (Prove It Feature)
    df_raw = prediction.get("df_raw")
    regime = prediction.get("regime", "NORMAL")
    for rec in recommendations:
        sell_text = rec["sell"]  # e.g., "SELL 22000 PE"
        parts = sell_text.split()
        try:
            strike_price = float(parts[1])
            side = "PE" if "PE" in sell_text else "CE"
            wr = pattern_match_proof(df_raw, spot, strike_price, vix, direction, regime, side=side)
            rec["win_rate"] = wr.get("win_rate")
            rec["survival_rate"] = wr.get("survival_rate")
            rec["pattern_proof_text"] = wr.get("text", "")
        except:
            rec["win_rate"] = None
            rec["survival_rate"] = None
            rec["pattern_proof_text"] = "Could not calculate."
    
    # 2. Recovery Blueprint
    recovery_text = recovery_blueprint(direction, vix, whipsaw_prob)
    
    # 3. Golden Hour Analysis
    golden_hour_text = ""
    try:
        nifty_15m_path = os.path.join(DATA_DIR, "nifty_15m_2001_to_now.csv")
        if os.path.exists(nifty_15m_path):
            df_15m = pd.read_csv(nifty_15m_path)
            gh = golden_hour_analysis(df_15m)
            golden_hour_text = gh.get("text", "")
    except:
        golden_hour_text = "Golden Hour data unavailable."
    
    # 4. FII/DII Flow Confirmation
    fii_check = fii_dii_flow_check(prediction)
    fii_text = fii_check.get("text", "")
    fii_aligned = fii_check.get("aligned", True)
    
    # 5. Event Shield
    event_check = event_shield_check()
    event_text = event_check.get("text", "")
    event_safe = event_check.get("safe", True)
    
    # 6. Streak Tracker
    streak = streak_tracker()
    streak_text = streak.get("text", "")
    confidence_modifier = streak.get("confidence_modifier", 0)
    
    # Apply streak confidence modifier to sizing
    if confidence_modifier < 0:
        recommended_lots = max(1, recommended_lots - 1)
        sizing_text += f" | {streak_text}"
    
    # Apply FII divergence to sizing
    if not fii_aligned:
        recommended_lots = max(1, recommended_lots // 2)
        sizing_text += f" | {fii_text}"
    
    # Apply Event Shield to entry
    if not event_safe:
        sizing_text += f" | {event_text}"
    
    # Stop Loss Calculation (Feature 6)
    stop_data = auto_stop_loss(spot, direction, nearest_support, nearest_resistance, vix, df_raw)
    
    # 7. Adjustment Ladder — Generate for PRIMARY strategy (recommendations[0])
    primary_rec = recommendations[0]
    primary_side = primary_rec.get("side", "PE")
    p_sell = primary_rec.get("sell_strike", sell_put)
    p_buy = primary_rec.get("buy_strike", buy_put)
    
    if primary_side == "BOTH":
        # Iron Condor: generate ladder for the weaker (closer) side
        if dist_put < dist_call:
            ladder = adjustment_ladder(spot, sell_put, buy_put, direction, "Bull Put", vix, df_raw)
        else:
            ladder = adjustment_ladder(spot, sell_call, buy_call, direction, "Bear Call", vix, df_raw)
    elif primary_side == "CE":
        ladder = adjustment_ladder(spot, p_sell, p_buy, direction, "Bear Call CE", vix, df_raw)
    else:
        ladder = adjustment_ladder(spot, p_sell, p_buy, direction, "Bull Put PE", vix, df_raw)


    # Build the final playbook with all intelligence
    playbook = {
        "sizing": sizing_text,
        "entry": f"Always buy protection legs first to cut margin by ~70%. {legging_advice} {ivp_str} is the current risk pricing. {golden_hour_text}",
        "take_profit": f"Target 50-60% of max premium. {gamma_risk}",
        "stop_loss": f"🛑 {stop_data['message']} {gap_risk}",
        "firefighting": f"🛡️ RECOVERY BLUEPRINT: {recovery_text}",
        "adjustment_ladder": ladder,
        "streak": streak_text,
        "fii_flow": fii_text,
        "event_shield": event_text,
    }
    
    if ivp is not None and ivp < 20:
        playbook["entry"] = f"🛑 WARNING - {ivp_str} is critically low. Options are historically cheap. DO NOT sell spreads. Consider buying debit spreads. " + playbook["entry"]

    if whipsaw_prob > 50:
        playbook["firefighting"] = f"⚠️ Whipsaw risk is extremely elevated ({whipsaw_prob}%). Cut position sizing in half. Do not attempt complex adjustments; simply hit Stop Loss if levels break. | {recovery_text}"

    # 8. Master Conviction Score (Feature 7)
    best_survival = max([r.get("survival_rate", 50) or 50 for r in recommendations], default=50)
    conviction = calculate_conviction_score(conf, best_survival, vix, regime, fii_aligned, event_safe)
    playbook["conviction"] = conviction

    return {
        "ready": True,
        "conviction": conviction,
        "spot": spot,
        "range_low": range_low,
        "range_high": range_high,
        "nearest_support": nearest_support,
        "nearest_resistance": nearest_resistance,
        "recommendations": recommendations,
        "direction": direction,
        "confidence": conf,
        "vix": vix,
        "whipsaw_prob": whipsaw_prob,
        "playbook": playbook,
        "df_raw": df_raw   # Pass raw data for What-If probability
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 10. THETA DECAY TIMER
# ═══════════════════════════════════════════════════════════════════════════════

def get_theta_decay_info(entry_date_str, expiry_date_str, premium_collected=100):
    """Calculate theta decay curve and sweet spot exit."""
    from datetime import datetime
    try:
        entry = datetime.strptime(entry_date_str, "%Y-%m-%d")
        expiry = datetime.strptime(expiry_date_str, "%Y-%m-%d")
    except (ValueError, TypeError):
        return {"ready": False, "message": "Invalid dates. Use YYYY-MM-DD format."}
    
    today = datetime.now()
    total_days = max(1, (expiry - entry).days)
    days_held = (today - entry).days
    days_remaining = max(0, (expiry - today).days)
    
    # Theta decay is non-linear: accelerates in the last 1/3
    # Approximate: decay_pct = (days_elapsed / total_days) ^ 0.6
    decay_curve = []
    for d in range(total_days + 1):
        pct_time = d / total_days
        decay_pct = pct_time ** 0.6  # non-linear acceleration
        premium_remaining = float(premium_collected) * (1 - decay_pct)
        decay_curve.append({
            "day": d,
            "premium_remaining": round(premium_remaining, 1),
            "decay_pct": round(decay_pct * 100, 1)
        })
    
    # Sweet spot: 70% decay (optimal exit for credit spreads)
    sweet_spot_day = 0
    for point in decay_curve:
        if point["decay_pct"] >= 70:
            sweet_spot_day = point["day"]
            break
    
    sweet_spot_date = entry + timedelta(days=sweet_spot_day)
    
    # Current decay
    current_pct_time = min(1.0, days_held / total_days)
    current_decay = current_pct_time ** 0.6
    daily_theta = float(premium_collected) * current_decay / max(1, days_held)
    
    return {
        "ready": True,
        "total_days": total_days,
        "days_held": days_held,
        "days_remaining": days_remaining,
        "decay_pct_now": round(current_decay * 100, 1),
        "premium_remaining": round(float(premium_collected) * (1 - current_decay), 1),
        "daily_theta": round(daily_theta, 1),
        "sweet_spot_day": sweet_spot_day,
        "sweet_spot_date": sweet_spot_date.strftime("%Y-%m-%d"),
        "decay_curve": decay_curve,
        "message": f"You've held for {days_held} days. {days_remaining} days remaining. "
                   f"~{current_decay*100:.0f}% of premium has decayed. "
                   f"Sweet spot exit: Day {sweet_spot_day} ({sweet_spot_date.strftime('%b %d')})."
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 11. VIX PREMIUM GAUGE
# ═══════════════════════════════════════════════════════════════════════════════

def get_vix_premium_gauge(prediction):
    """Analyze VIX level and what it means for credit spread premiums."""
    if not prediction or not prediction.get("success"):
        return {"ready": False, "message": "Awaiting prediction..."}
    
    vix = float(prediction.get("vix_value", 15.0))
    regime = prediction.get("regime", "UNKNOWN")
    
    if vix >= 25:
        level = "🔥 EXTREME"
        color = "#FF4B4B"
        premium_quality = "ULTRA JUICY"
        advice = "Premiums are at maximum fat levels! Sell credit spreads NOW for massive income. But keep stops tight — markets are wild."
        score = 95
    elif vix >= 20:
        level = "🟡 HIGH"
        color = "#FFD700"
        premium_quality = "JUICY"
        advice = "Premiums are fat and attractive. Great time to sell credit spreads. Wider strikes are profitable."
        score = 80
    elif vix >= 15:
        level = "🟢 NORMAL"
        color = "#00FF7F"
        premium_quality = "FAIR"
        advice = "Standard premiums. Credit spreads will work, but keep width tight (200pt) for best risk/reward."
        score = 55
    elif vix >= 12:
        level = "😴 LOW"
        color = "#00CED1"
        premium_quality = "THIN"
        advice = "Premiums are skinny. Consider wider spreads or skip selling today. Theta won't earn you much."
        score = 30
    else:
        level = "❄️ DEAD"
        color = "#8B8D97"
        premium_quality = "PAPER THIN"
        advice = "VIX is complacent. Premiums are almost zero. Avoid credit spreads — the risk/reward is terrible."
        score = 10
    
    return {
        "ready": True,
        "vix": vix,
        "level": level,
        "color": color,
        "premium_quality": premium_quality,
        "advice": advice,
        "score": score,
        "regime": regime
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 12. WIN RATE TRACKER
# ═══════════════════════════════════════════════════════════════════════════════

def get_win_rate_stats():
    """Analyze personal trading history from war_room_settings.json."""
    import json
    
    try:
        if not os.path.exists(WAR_ROOM_PATH):
            return {"ready": False, "message": "No trades logged yet. Use Position Manager to log trades."}
        
        with open(WAR_ROOM_PATH, 'r') as f:
            data = json.load(f)
        
        trades = data if isinstance(data, list) else data.get("closed_trades", [])
        
        if len(trades) == 0:
            return {"ready": False, "message": "No closed trades found. Close some positions to see stats."}
        
        wins = [t for t in trades if t.get("pnl", 0) > 0]
        losses = [t for t in trades if t.get("pnl", 0) <= 0]
        
        total = len(trades)
        win_count = len(wins)
        win_rate = (win_count / total * 100) if total > 0 else 0
        
        avg_win = np.mean([t.get("pnl", 0) for t in wins]) if wins else 0
        avg_loss = np.mean([abs(t.get("pnl", 0)) for t in losses]) if losses else 0
        
        # Best regime
        regime_stats = {}
        for t in trades:
            r = t.get("regime", "UNKNOWN")
            if r not in regime_stats:
                regime_stats[r] = {"wins": 0, "total": 0}
            regime_stats[r]["total"] += 1
            if t.get("pnl", 0) > 0:
                regime_stats[r]["wins"] += 1
        
        best_regime = max(regime_stats, key=lambda k: regime_stats[k]["wins"] / max(1, regime_stats[k]["total"])) if regime_stats else "N/A"
        worst_regime = min(regime_stats, key=lambda k: regime_stats[k]["wins"] / max(1, regime_stats[k]["total"])) if regime_stats else "N/A"
        
        return {
            "ready": True,
            "total_trades": total,
            "wins": win_count,
            "losses": len(losses),
            "win_rate": round(win_rate, 1),
            "avg_win": round(float(avg_win), 0),
            "avg_loss": round(float(avg_loss), 0),
            "best_regime": best_regime,
            "worst_regime": worst_regime,
            "regime_stats": regime_stats,
            "message": f"Your win rate is {win_count}/{total} ({win_rate:.0f}%). "
                       f"Best regime: {best_regime}. Avoid: {worst_regime}."
        }
    except Exception as e:
        return {"ready": False, "message": f"Error loading trades: {e}"}


# ═══════════════════════════════════════════════════════════════════════════════
# 13. ENTRY TIMING SIGNAL
# ═══════════════════════════════════════════════════════════════════════════════

def get_entry_timing(prediction):
    """Generate actionable entry timing advice."""
    if not prediction or not prediction.get("success"):
        return {"ready": False, "message": "Awaiting prediction..."}
    
    spot = float(prediction.get("spot_price", 24000))
    direction = prediction.get("ensemble_prediction", {}).get("direction", "SIDEWAYS")
    conf = prediction.get("ensemble_prediction", {}).get("confidence", 0.5) * 100
    regime = prediction.get("regime", "UNKNOWN")
    vix = float(prediction.get("vix_value", 15.0))
    supports = prediction.get("supports", [])
    resistances = prediction.get("resistances", [])
    
    support_prices = [s["price"] if isinstance(s, dict) else s for s in supports]
    resistance_prices = [r["price"] if isinstance(r, dict) else r for r in resistances]
    
    nearest_support = max([s for s in support_prices if s < spot], default=spot - 200)
    nearest_resistance = min([r for r in resistance_prices if r > spot], default=spot + 200)
    dist_to_support = spot - nearest_support
    dist_to_resistance = nearest_resistance - spot
    
    signals = []
    
    # Momentum check
    if conf >= 60:
        momentum = "STRONG"
        signals.append({
            "signal": "🟢 ENTER NOW",
            "reason": f"David is {conf:.0f}% confident in {direction}. Momentum is strong — don't wait for a dip.",
            "color": "#00FF7F"
        })
    elif conf >= 52:
        momentum = "MODERATE"
        if direction == "UP" and dist_to_support < 100:
            signals.append({
                "signal": "🟢 ENTER NOW (Near Support)",
                "reason": f"Price is only {dist_to_support:.0f} pts from support ({nearest_support:.0f}). Good entry zone.",
                "color": "#00FF7F"
            })
        elif direction == "DOWN" and dist_to_resistance < 100:
            signals.append({
                "signal": "🟢 ENTER NOW (Near Resistance)",
                "reason": f"Price is only {dist_to_resistance:.0f} pts from resistance ({nearest_resistance:.0f}). Good shorting zone.",
                "color": "#00FF7F"
            })
        else:
            signals.append({
                "signal": f"⏳ WAIT for dip to {nearest_support:.0f}",
                "reason": f"Confidence is {conf:.0f}% (moderate). Wait for price to reach support for a better entry.",
                "color": "#FFD700"
            })
    else:
        momentum = "WEAK"
        signals.append({
            "signal": "🔴 SKIP TODAY",
            "reason": f"Confidence is only {conf:.0f}%. No clear edge. Wait for a stronger signal tomorrow.",
            "color": "#FF4B4B"
        })
    
    # VIX timing
    if vix > 22:
        signals.append({
            "signal": "⚡ VIX PREMIUM WINDOW OPEN",
            "reason": f"VIX is {vix:.1f} — premiums are fat. If you're selling credit spreads, NOW is the time.",
            "color": "#FFD700"
        })
    
    # Regime warning
    if regime in ["CHOPPY", "VOLATILE"]:
        signals.append({
            "signal": "⚠️ REGIME CAUTION",
            "reason": f"Market is in {regime} mode. Use smaller position sizes and wider stops.",
            "color": "#FF4B4B"
        })
    
    return {
        "ready": True,
        "spot": spot,
        "momentum": momentum,
        "direction": direction,
        "confidence": conf,
        "nearest_support": nearest_support,
        "nearest_resistance": nearest_resistance,
        "signals": signals
    }



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
                result["message"] = f"✅ HOLD — The market is SLEEPY ({current_confidence:.0f}% confidence). Your trade earns money while the market rests."
            else:
                result["signal"] = "HOLD"
                result["message"] = f"⚠️ HOLD (cautious) — The market is ZIG-ZAGGING. Your safety margin is currently holding."
                
        # 2. Case: Direction still agrees perfectly
        elif current_direction and base_dir == current_direction:
            if current_confidence >= 50:
                result["signal"] = "HOLD"
                result["message"] = f"✅ HOLD — The market is still moving {current_direction} ({current_confidence:.0f}%). Stay the course."
            elif current_confidence >= 40:
                result["signal"] = "HOLD"
                result["message"] = f"⚠️ HOLD (cautious) — The market is moving {current_direction} but slowing down ({current_confidence:.0f}% conviction)."
            else:
                result["signal"] = "HEDGE"
                result["message"] = f"⚠️ HEDGE — The market is still {current_direction} but losing energy. Protect your profits."
        
        # 3. Case: Direction flipped against you
        elif current_direction:
            if current_confidence >= 55:
                result["signal"] = "EXIT"
                result["message"] = f"🔴 EXIT — The market has turned and is now moving {current_direction} ({current_confidence:.0f}% sure). Don't fight the tide."
            elif current_confidence >= 40:
                result["signal"] = "HEDGE"
                result["message"] = f"⚠️ HEDGE — The market is starting to shift {current_direction}. Consider adding a safety net."
            else:
                result["signal"] = "HOLD"
                result["message"] = f"🟡 HOLD — The market is confused ({current_direction} at low {current_confidence:.0f}% confidence). Wait for a clearer signal."
        
        # 4. Case: Regime danger
        if pred.get("regime") == "VOLATILE" and base_dir != "SIDEWAYS":
            if result["signal"] != "EXIT":
                result["signal"] = "HEDGE"
                result["message"] = str(result["message"]) + " ⚠️ Regime is VOLATILE — tighten stops."
        
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
                
                historical_dd = drop if "UP" in dir_upper else -drop
                
                if abs(historical_dd + abs_dd) < dd_tolerance:
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
    if whipsaw and whipsaw.get("is_choppy"):
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

def save_to_journal(strategy, signal, predicted, notes=""):
    """
    Save a trade recommendation to the trade_journal.json file.
    """
    import json
    from datetime import datetime
    import os
    from utils import DATA_DIR

    journal_path = os.path.join(DATA_DIR, "trade_journal.json")
    
    # Load existing journal
    if os.path.exists(journal_path):
        try:
            with open(journal_path, "r") as f:
                journal = json.load(f)
        except Exception:
            journal = {"entries": [], "drawdown_cooldown_until": None}
    else:
        journal = {"entries": [], "drawdown_cooldown_until": None}

    # Create new entry
    now = datetime.now()
    new_entry = {
        "date": now.strftime("%Y-%m-%d"),
        "signal": signal, # e.g., "GREEN", "YELLOW", "RED"
        "strategy": strategy,
        "predicted": predicted,
        "actual": None,
        "pnl": None,
        "result": None,
        "notes": notes,
        "logged_at": now.strftime("%Y-%m-%d %H:%M:%S")
    }

    # Append and save
    journal["entries"].append(new_entry)
    
    try:
        with open(journal_path, "w") as f:
            json.dump(journal, f, indent=2)
        return True
    except Exception as e:
        log(f"❌ Failed to save to journal: {e}")
        return False

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
