"""
DAVID PROPHETIC ORACLE — Whipsaw & Flip Detector
==================================================
Classifies market as TRENDING or CHOPPY.
Predicts whipsaw probability and expected chop range.
Detects regime flips and warns about imminent reversals.
"""

import numpy as np
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import C


class WhipsawDetector:
    """
    Whipsaw/Chop detection based on multiple volatility and trend signals.
    
    Signals used:
        1. BB Squeeze (width percentile < 20% = compression → breakout coming)
        2. ADX < 20 = no trend = whipsaw zone
        3. ATR ratio vs 20D avg (expansion/contraction)
        4. Consecutive direction flip count (candle flips)
        5. Realized vol vs implied vol divergence
    """
    
    def analyze(self, df):
        """
        Analyze the latest market state for whipsaw/chop conditions.
        
        Returns:
            dict with whipsaw_prob, is_choppy, chop_range, signals, flip_risk
        """
        latest = df.iloc[-1]
        recent = df.tail(20)
        
        signals = {}
        
        # ─── Signal 1: Bollinger Band Squeeze ────────────────────────────
        bb_width = latest.get("bb_width", 0.04)
        bb_width_pctile = df["bb_width"].tail(252).rank(pct=True).iloc[-1] if "bb_width" in df.columns else 0.5
        
        squeeze = bb_width_pctile < 0.20
        signals["bb_squeeze"] = {
            "value": float(bb_width_pctile),
            "signal": "COMPRESSION (breakout imminent)" if squeeze else "Normal",
            "weight": 0.75 if squeeze else 0.0,
        }
        
        # ─── Signal 2: ADX Trend Strength ────────────────────────────────
        adx = latest.get("adx", 25)
        no_trend = adx < 20
        signals["adx_trend"] = {
            "value": float(adx),
            "signal": "NO TREND (chop zone)" if no_trend else f"Trending ({adx:.0f})",
            "weight": 0.8 if no_trend else 0.0,
        }
        
        # ─── Signal 3: ATR Expansion/Contraction ─────────────────────────
        atr_val = latest.get("atr_14", 0.01)
        atr_mean = df["atr_14"].tail(50).mean() if "atr_14" in df.columns else atr_val
        atr_expanding = atr_val > atr_mean * 1.3
        
        signals["atr_volatility"] = {
            "value": float(atr_val / atr_mean) if atr_mean > 0 else 1.0,
            "signal": "EXPANDING (reversal risk)" if atr_expanding else "Normal",
            "weight": 0.6 if atr_expanding else 0.0,
        }
        
        # ─── Signal 4: Direction Flip Count ──────────────────────────────
        if len(recent) >= 10:
            candle_dirs = (recent["close"] > recent["open"]).astype(int).values
            flips = sum(candle_dirs[i] != candle_dirs[i-1] for i in range(1, len(candle_dirs)))
            flip_rate = flips / max(1, len(candle_dirs) - 1)
        else:
            flip_rate = 0.5
            flips = 5
        
        high_flips = flip_rate > 0.6
        signals["candle_flips"] = {
            "value": float(flip_rate),
            "signal": f"HIGH CHOP ({flips} flips in 20 bars)" if high_flips else f"Normal ({flips} flips)",
            "weight": 0.7 if high_flips else 0.0,
        }
        
        # ─── Signal 5: VIX-RV Divergence ─────────────────────────────────
        vix = latest.get("vix", 15)
        rv = latest.get("realized_vol_20", 0.15)
        rv_annual = rv  # already annualized in feature_forge
        vix_rv_ratio = (vix / 100) / max(rv_annual, 0.01) if rv_annual > 0 else 1.0
        
        vix_overpriced = vix_rv_ratio > 1.3
        signals["vix_rv_divergence"] = {
            "value": float(vix_rv_ratio),
            "signal": "VIX > RV (mean reversion expected)" if vix_overpriced else "Normal",
            "weight": 0.4 if vix_overpriced else 0.0,
        }
        
        # ─── Aggregate Whipsaw Probability ───────────────────────────────
        total_weight = sum(s["weight"] for s in signals.values())
        max_weight = len(signals) * 0.8  # max possible
        whipsaw_prob = min(1.0, total_weight / max_weight) * 100
        
        is_choppy = whipsaw_prob > 55
        
        # Chop range (based on ATR)
        spot = float(df["close"].iloc[-1])
        atr_val = float(latest.get("atr_14", spot * 0.01))
        chop_low = spot - atr_val * 1.5
        chop_high = spot + atr_val * 1.5
        
        # Flip risk — probability of regime change in next 5 days
        # Based on historical regime transition frequency
        returns_std = df["returns_1d"].tail(20).std()
        flip_risk = min(100, float(returns_std * 1000))  # Crude but effective
        
        return {
            "whipsaw_prob": round(whipsaw_prob, 1),
            "is_choppy": is_choppy,
            "chop_range": (round(chop_low, 2), round(chop_high, 2)),
            "flip_risk": round(flip_risk, 1),
            "signals": signals,
            "atr": round(atr_val, 2),
        }
    
    def format_analysis(self, result, current_price):
        """Pretty print whipsaw analysis."""
        lines = []
        lines.append(f"\n  {C.header('WHIPSAW & FLIP ANALYSIS')}")
        lines.append(f"  {'─'*50}")
        
        # Whipsaw meter
        prob = result["whipsaw_prob"]
        bar_len = int(prob / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        
        if prob > 70:
            color = C.RED
            status = "DANGER — HIGH CHOP"
        elif prob > 50:
            color = C.YELLOW
            status = "CAUTION — CHOPPY"
        else:
            color = C.GREEN
            status = "CLEAR — TRENDING"
        
        lines.append(f"\n  Whipsaw Probability: {color}{C.BOLD}{prob:.0f}%{C.RESET}")
        lines.append(f"  [{color}{bar}{C.RESET}] {status}")
        
        # Chop range
        clow, chigh = result["chop_range"]
        lines.append(f"\n  Expected Chop Range:")
        lines.append(f"    Low:  {C.down(f'{clow:,.0f}')}")
        lines.append(f"    High: {C.up(f'{chigh:,.0f}')}")
        lines.append(f"    ATR:  {result['atr']:,.0f} pts")
        
        # Flip risk
        lines.append(f"\n  Regime Flip Risk (5-day): {C.pct(result['flip_risk'])}")
        
        # Individual signals
        lines.append(f"\n  Signal Breakdown:")
        for name, sig in result["signals"].items():
            icon = "⚠️" if sig["weight"] > 0.3 else "✅"
            lines.append(f"    {icon} {name:>18}: {sig['signal']}")
        
        return "\n".join(lines)
