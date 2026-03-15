"""
DAVID PROPHETIC ORACLE — Feature Forge
========================================
Clean, leak-free feature engineering pipeline.
~45 features across 8 categories. No redundancy. No future leakage.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# pandas_ta removed (not used in core math)
from utils import DIRECTION_THRESHOLD, UP, DOWN, SIDEWAYS


def engineer_features(df, target_horizon=1):
    """
    Build the full feature matrix from raw OHLCV + VIX + S&P data.
    
    Args:
        df: DataFrame with columns [date, open, high, low, close, volume, vix, sp_close]
        target_horizon: Days ahead for target variable (default 1 = next day)
    
    Returns:
        df: DataFrame with all features + target columns
        feature_cols: List of feature column names (safe to use for ML)
    """
    df = df.copy()
    
    # ═══════════════════════════════════════════════════════════════════════
    # 1. PRICE ACTION (PRUNED)
    # ═══════════════════════════════════════════════════════════════════════
    df["returns_1d"] = df["close"].pct_change(1)
    
    # Gap (overnight)
    df["gap_pct"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)
    
    # Wick ratios
    df["upper_wick"] = (df["high"] - df[["close", "open"]].max(axis=1)) / (df["high"] - df["low"]).replace(0, np.nan)
    df["lower_wick"] = (df[["close", "open"]].min(axis=1) - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)
    
    # ═══════════════════════════════════════════════════════════════════════
    # 2. VOLATILITY (PRUNED)
    # ═══════════════════════════════════════════════════════════════════════
    df["realized_vol_10"] = df["returns_1d"].rolling(10).std() * np.sqrt(252)
    df["realized_vol_20"] = df["returns_1d"].rolling(20).std() * np.sqrt(252)
    df["vol_of_vol"] = df["realized_vol_20"].rolling(20).std()
    
    # ATR
    tr = pd.DataFrame({
        "hl": df["high"] - df["low"],
        "hc": (df["high"] - df["close"].shift(1)).abs(),
        "lc": (df["low"] - df["close"].shift(1)).abs()
    }).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()
    
    # Bollinger Band width
    sma20 = df["close"].rolling(20).mean()
    std20 = df["close"].rolling(20).std()
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma20
    
    # ═══════════════════════════════════════════════════════════════════════
    # 3. MOMENTUM (PRUNED)
    # ═══════════════════════════════════════════════════════════════════════
    # RSI
    delta = df["close"].diff()
    gain7 = delta.clip(lower=0).rolling(7).mean()
    loss7 = (-delta.clip(upper=0)).rolling(7).mean()
    rs7 = gain7 / loss7.replace(0, np.nan)
    df["rsi_7"] = 100 - (100 / (1 + rs7))
    
    # ═══════════════════════════════════════════════════════════════════════
    # 4. TREND (PRUNED)
    # ═══════════════════════════════════════════════════════════════════════
    # ADX (Average Directional Index) — proper filtering
    high_diff = df["high"].diff()
    low_diff = -df["low"].diff()
    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)
    
    atr_smooth = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr_smooth.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr_smooth.replace(0, np.nan))
    di_diff = (plus_di - minus_di).abs()
    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * di_diff / di_sum
    df["adx"] = dx.rolling(14).mean()
    
    # ═══════════════════════════════════════════════════════════════════════
    # 5. MARKET STRUCTURE (PRUNED)
    # ═══════════════════════════════════════════════════════════════════════
    # Removed due to lack of importance
    
    # ═══════════════════════════════════════════════════════════════════════
    # 6. VIX FEATURES (PRUNED)
    # ═══════════════════════════════════════════════════════════════════════
    if "vix" in df.columns:
        df["vix_percentile"] = df["vix"].rolling(252).rank(pct=True)
        df["vix_change"] = df["vix"].pct_change()
    
    # ═══════════════════════════════════════════════════════════════════════
    # 7. CROSS-MARKET (3 features)
    # ═══════════════════════════════════════════════════════════════════════
    if "sp_close" in df.columns:
        df["sp_return"] = df["sp_close"].pct_change()
        df["sp_nifty_corr_20"] = df["returns_1d"].rolling(20).corr(df["sp_return"])
        df["sp_return_lag1"] = df["sp_return"].shift(1)  # Previous day S&P (overnight signal)
    
    # ═══════════════════════════════════════════════════════════════════════
    # 8. CALENDAR (PRUNED)
    # ═══════════════════════════════════════════════════════════════════════
    # Removed due to lack of importance

    
    # ═══════════════════════════════════════════════════════════════════════
    # 9. VOLUME FEATURES (2 features)
    # ═══════════════════════════════════════════════════════════════════════
    if "volume" in df.columns and df["volume"].sum() > 0:
        df["vol_ratio_20"] = df["volume"] / df["volume"].rolling(20).mean().replace(0, np.nan)
        df["obv_trend"] = (np.sign(df["returns_1d"]) * df["volume"]).cumsum()
        df["obv_trend"] = df["obv_trend"].pct_change(5).replace([np.inf, -np.inf], 0)  # OBV momentum
    else:
        df["vol_ratio_20"] = 1.0
        df["obv_trend"] = 0.0
    
    # ═══════════════════════════════════════════════════════════════════════
    # 10. SENTIMENT (7 features)
    # ═══════════════════════════════════════════════════════════════════════
    if "pcr" in df.columns:
        df["pcr_val"] = df["pcr"]
        df["pcr_sma_5"] = df["pcr"].rolling(5).mean()
        df["pcr_change"] = df["pcr"].diff()
    else:
        df["pcr_val"] = 1.0
        df["pcr_sma_5"] = 1.0
        df["pcr_change"] = 0.0
        
    if "fii_net" in df.columns and "dii_net" in df.columns:
        df["fii_flow"] = df["fii_net"] / 1000.0  # scale for ML
        df["dii_flow"] = df["dii_net"] / 1000.0
        df["inst_net_flow"] = df["fii_flow"] + df["dii_flow"]
        df["fii_trend_5"] = df["fii_flow"].rolling(5).sum()
    else:
        df["fii_flow"] = 0.0
        df["dii_flow"] = 0.0
        df["inst_net_flow"] = 0.0
        df["fii_trend_5"] = 0.0

    # ═══════════════════════════════════════════════════════════════════════
    # 11. ADVANCED MIROFISH INSIGHTS (OI & VOLUME PROFILE)
    # ═══════════════════════════════════════════════════════════════════════
    # A. Open Interest (OI) Velocity
    if "total_oi" in df.columns and df["total_oi"].mean() > 0:
        df["oi_change_1d"] = df["total_oi"].pct_change(1)
        df["oi_change_5d"] = df["total_oi"].pct_change(5)
        
        # Price-OI Divergence: Long build-up (Price Up, OI Up)
        df["long_build_up"] = ((df["returns_1d"] > 0) & (df["oi_change_1d"] > 0)).astype(int)
        df["short_build_up"] = ((df["returns_1d"] < 0) & (df["oi_change_1d"] > 0)).astype(int)
        df["long_unwinding"] = ((df["returns_1d"] < 0) & (df["oi_change_1d"] < 0)).astype(int)
    else:
        df["oi_change_1d"] = 0.0
        df["oi_change_5d"] = 0.0
        df["long_build_up"] = 0
        df["short_build_up"] = 0
        df["long_unwinding"] = 0

    # B. Simplified Volume Profile (Distance to POC)
    # Price level with maximum volume over the last 20 days
    def get_poc_dist(series_price, series_vol, window=20):
        if len(series_price) < window: return 0.0
        # Simple binning to find Point of Control (POC)
        bins = 10
        prices = series_price.iloc[-window:]
        vols = series_vol.iloc[-window:]
        hist, bin_edges = np.histogram(prices, bins=bins, weights=vols)
        poc_idx = np.argmax(hist)
        poc_price = (bin_edges[poc_idx] + bin_edges[poc_idx+1]) / 2
        return (series_price.iloc[-1] - poc_price) / poc_price

    df["vol_poc_dist_20"] = [get_poc_dist(df["close"].iloc[:i+1], df["volume"].iloc[:i+1]) if i > 20 else 0 for i in range(len(df))]

    # ═══════════════════════════════════════════════════════════════════════
    # TARGET VARIABLE (NOT a feature — excluded from ML input)
    # ═══════════════════════════════════════════════════════════════════════
    df["future_return"] = df["close"].shift(-target_horizon) / df["close"] - 1
    df["target"] = np.where(
        df["future_return"] > DIRECTION_THRESHOLD, 0,   # UP
        np.where(df["future_return"] < -DIRECTION_THRESHOLD, 1, 2)  # DOWN, SIDEWAYS
    )
    df["target_label"] = df["target"].map({0: UP, 1: DOWN, 2: SIDEWAYS})
    
    # ═══════════════════════════════════════════════════════════════════════
    # CLEANUP
    # ═══════════════════════════════════════════════════════════════════════
    # Define feature columns (EXCLUDE targets, raw prices, dates, and specifically prune to top 20 ONLY)
    exclude = [
        "date", "open", "high", "low", "close", "volume",
        "vix", "sp_close",
        "future_return", "target", "target_label",
        "bb_upper", "bb_lower",
    ]
    
    # The refined elite list:
    TOP_FEATURES = [
        'sp_return', 'vol_ratio_20', 'pcr_change', 'upper_wick', 'obv_trend', 
        'gap_pct', 'lower_wick', 'sp_return_lag1', 'inst_net_flow', 'bb_width', 
        'adx', 'vix_change', 'fii_trend_5', 'rsi_7', 'vix_percentile', 
        'atr_14', 'sp_nifty_corr_20', 'realized_vol_10', 'vol_of_vol', 'returns_1d'
    ]
    
    feature_cols = [c for c in df.columns if c in TOP_FEATURES]
    
    # Drop rows with NaN in features (warmup period)
    df = df.dropna(subset=feature_cols + ["target"])
    df = df.reset_index(drop=True)
    
    # Replace any remaining infinities
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df[feature_cols] = df[feature_cols].fillna(0)
    
    print(f"  {len(feature_cols)} features engineered across {len(df)} rows")
    print(f"  Feature list: {feature_cols[:10]}... ({len(feature_cols)} total)")
    
    return df, feature_cols


def get_target_distribution(df):
    """Print target class balance."""
    counts = df["target_label"].value_counts()
    total = len(df)
    print(f"\n  Target Distribution:")
    for label in [UP, DOWN, SIDEWAYS]:
        ct = counts.get(label, 0)
        print(f"    {label:>10}: {ct:>5} ({ct/total*100:.1f}%)")
    return counts


if __name__ == "__main__":
    from data_engine import load_all_data
    df = load_all_data()
    df, feature_cols = engineer_features(df)
    print(f"\nFeature columns ({len(feature_cols)}):")
    for i, fc in enumerate(feature_cols):
        print(f"  {i+1:>3}. {fc}")
    get_target_distribution(df)
