"""
DAVID PROPHETIC ORACLE — Feature Forge
========================================
Clean, leak-free feature engineering pipeline.
~45 features across 8 categories. No redundancy. No future leakage.
"""

import pandas as pd
import numpy as np

# pandas_ta removed (not used in core math)
from utils import DIRECTION_THRESHOLD, UP, DOWN, SIDEWAYS


def engineer_features(df, target_horizon=1):
    """
    Build the full feature matrix from raw OHLCV + VIX + S&P data.
    
    Args:
        df: DataFrame with columns [date, open, high, low, close, volume, vix, sp_close]
        target_horizon: Days ahead for target variable (default 5 = weekly)
    
    Returns:
        df: DataFrame with all features + target columns
        feature_cols: List of feature column names (safe to use for ML)
    """
    df = df.copy()
    
    # ═══════════════════════════════════════════════════════════════════════
    # 1. PRICE ACTION (5 features)
    # ═══════════════════════════════════════════════════════════════════════
    df["returns_1d"] = df["close"].pct_change(1)
    df["returns_5d"] = df["close"].pct_change(5)
    df["returns_10d"] = df["close"].pct_change(10)
    df["returns_20d"] = df["close"].pct_change(20)
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    
    # Gap (overnight)
    df["gap_pct"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)
    
    # Body and wick ratios
    df["body_ratio"] = (df["close"] - df["open"]).abs() / (df["high"] - df["low"]).replace(0, np.nan)
    df["upper_wick"] = (df["high"] - df[["close", "open"]].max(axis=1)) / (df["high"] - df["low"]).replace(0, np.nan)
    df["lower_wick"] = (df[["close", "open"]].min(axis=1) - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)
    
    # ═══════════════════════════════════════════════════════════════════════
    # 2. VOLATILITY (6 features)
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
    df["atr_ratio"] = df["atr_14"] / df["close"]
    
    # Bollinger Band width
    sma20 = df["close"].rolling(20).mean()
    std20 = df["close"].rolling(20).std()
    df["bb_upper"] = sma20 + 2 * std20
    df["bb_lower"] = sma20 - 2 * std20
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / sma20
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
    
    # ═══════════════════════════════════════════════════════════════════════
    # 3. MOMENTUM (8 features)
    # ═══════════════════════════════════════════════════════════════════════
    # RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))
    
    gain7 = delta.clip(lower=0).rolling(7).mean()
    loss7 = (-delta.clip(upper=0)).rolling(7).mean()
    rs7 = gain7 / loss7.replace(0, np.nan)
    df["rsi_7"] = 100 - (100 / (1 + rs7))
    
    # MACD
    ema12 = df["close"].ewm(span=12).mean()
    ema26 = df["close"].ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    
    # Stochastic %K
    low14 = df["low"].rolling(14).min()
    high14 = df["high"].rolling(14).max()
    df["stoch_k"] = 100 * (df["close"] - low14) / (high14 - low14).replace(0, np.nan)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()
    
    # Williams %R
    df["williams_r"] = -100 * (high14 - df["close"]) / (high14 - low14).replace(0, np.nan)
    
    # Rate of Change
    df["roc_10"] = (df["close"] / df["close"].shift(10) - 1) * 100
    
    # ═══════════════════════════════════════════════════════════════════════
    # 4. TREND (7 features)
    # ═══════════════════════════════════════════════════════════════════════
    for p in [20, 50, 200]:
        df[f"sma_{p}"] = df["close"].rolling(p).mean()
        df[f"dist_sma_{p}"] = (df["close"] - df[f"sma_{p}"]) / df[f"sma_{p}"]
    
    # SMA cross signals
    df["sma_20_50_cross"] = np.where(df["sma_20"] > df["sma_50"], 1, -1)
    
    # ADX (Average Directional Index) — simplified
    plus_dm = df["high"].diff().clip(lower=0)
    minus_dm = (-df["low"].diff()).clip(lower=0)
    
    atr_smooth = tr.rolling(14).mean()
    plus_di = 100 * (plus_dm.rolling(14).mean() / atr_smooth.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(14).mean() / atr_smooth.replace(0, np.nan))
    di_diff = (plus_di - minus_di).abs()
    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * di_diff / di_sum
    df["adx"] = dx.rolling(14).mean()
    
    # ═══════════════════════════════════════════════════════════════════════
    # 5. MARKET STRUCTURE (4 features)
    # ═══════════════════════════════════════════════════════════════════════
    # Higher highs / Lower lows count (last 10 bars)
    df["higher_high_count"] = (df["high"] > df["high"].shift(1)).rolling(10).sum()
    df["lower_low_count"] = (df["low"] < df["low"].shift(1)).rolling(10).sum()
    
    # Consecutive up/down days
    df["consec_up"] = (df["close"] > df["close"].shift(1)).astype(int)
    consec_groups = (df["consec_up"] != df["consec_up"].shift(1)).cumsum()
    df["consec_streak"] = df.groupby(consec_groups)["consec_up"].cumsum()
    df.loc[df["consec_up"] == 0, "consec_streak"] = -df.groupby(consec_groups)["consec_up"].transform(lambda x: (x == 0).cumsum())
    
    # Distance from 52-week high/low
    df["dist_52w_high"] = (df["close"] - df["high"].rolling(252).max()) / df["high"].rolling(252).max()
    df["dist_52w_low"] = (df["close"] - df["low"].rolling(252).min()) / df["low"].rolling(252).min()
    
    # ═══════════════════════════════════════════════════════════════════════
    # 6. VIX FEATURES (4 features)
    # ═══════════════════════════════════════════════════════════════════════
    if "vix" in df.columns:
        df["vix_sma_10"] = df["vix"].rolling(10).mean()
        df["vix_ratio"] = df["vix"] / df["vix_sma_10"].replace(0, np.nan)
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
    # 8. CALENDAR (3 features)
    # ═══════════════════════════════════════════════════════════════════════
    df["day_of_week"] = df["date"].dt.dayofweek / 4.0  # Mon=0, Fri=1
    df["month"] = df["date"].dt.month / 12.0
    df["is_expiry_week"] = ((df["date"].dt.dayofweek == 3) | 
                            (df["date"].dt.dayofweek == 2) | 
                            (df["date"].dt.dayofweek == 4)).astype(int)
    
    # ═══════════════════════════════════════════════════════════════════════
    # 9. VOLUME FEATURES (2 features)
    # ═══════════════════════════════════════════════════════════════════════
    if "volume" in df.columns and df["volume"].sum() > 0:
        df["vol_ratio_20"] = df["volume"] / df["volume"].rolling(20).mean().replace(0, np.nan)
        df["obv_trend"] = (np.sign(df["returns_1d"]) * df["volume"]).cumsum()
        df["obv_trend"] = df["obv_trend"].pct_change(5)  # OBV momentum
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
    # Define feature columns (EXCLUDE targets, raw prices, dates)
    exclude = [
        "date", "open", "high", "low", "close", "volume",
        "vix", "sp_close",
        "future_return", "target", "target_label",
        "bb_upper", "bb_lower",  # Absolute prices, not features
        "sma_20", "sma_50", "sma_200",  # Absolute prices
        "consec_up",  # Intermediate calc
    ]
    
    feature_cols = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64, np.float32, np.int32]]
    
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
