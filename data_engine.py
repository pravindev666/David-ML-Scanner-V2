"""
DAVID PROPHETIC ORACLE — Data Engine
=====================================
Fetches NIFTY, VIX, S&P 500 daily OHLCV from yfinance (2015–now).
Caches to local CSVs with incremental sync.
Falls back to v3 CSVs if yfinance fails.
"""

import os
import time
import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

try:
    import yfinance as yf
except ImportError:
    raise ImportError("yfinance is required. Install with: pip install yfinance")

try:
    from nsepython import nse_fiidii
except ImportError:
    nse_fiidii = None
    raise ImportError("yfinance is required. Install with: pip install yfinance")

from utils import DATA_DIR, NIFTY_SYMBOL, VIX_SYMBOL, SP500_SYMBOL, DATA_START_YEAR, C


def _csv_path(name):
    return os.path.join(DATA_DIR, f"{name}_daily.csv")


def _v3_fallback_path(name):
    """Try to find v3 CSV as fallback."""
    v3_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "v3", "data")
    mapping = {
        "nifty": "NIFTY_50.csv",
        "vix": "VIX.csv",
        "sp500": "SP500.csv",
    }
    if name in mapping:
        path = os.path.join(v3_dir, mapping[name])
        if os.path.exists(path):
            return path
    return None


def fetch_symbol(symbol, name, start_year=DATA_START_YEAR):
    """
    Fetch daily OHLCV for a symbol from yfinance.
    Uses incremental sync — only downloads new data if CSV already exists.
    """
    csv_path = _csv_path(name)
    start_date = f"{start_year}-01-01"
    
    existing_df = None
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path, parse_dates=["date"])
        last_date = existing_df["date"].max()
        # Only fetch from last date onward
        start_date = (last_date - timedelta(days=5)).strftime("%Y-%m-%d")
        print(f"  {C.DIM}[SYNC] {name}: Incremental from {start_date}{C.RESET}")
    else:
        print(f"  {C.CYAN}[FETCH] {name}: Full download from {start_date}{C.RESET}")

    import time
    import random
    
    df = None
    max_retries = 3
    for attempt in range(max_retries):
        try:
            df = yf.download(symbol, start=start_date, auto_adjust=True, progress=False)
            if not df.empty:
                break
            else:
                raise ValueError(f"No data returned for {symbol}")
        except Exception as e:
            if attempt < max_retries - 1:
                sleep_time = random.uniform(1, 2)
                print(f"  {C.YELLOW}[RETRY] Rate limited or failed for {name}. Retrying in {sleep_time:.1f}s ({attempt+1}/{max_retries})...{C.RESET}")
                time.sleep(sleep_time)
            else:
                # Let the main exception handler catch it on the last attempt
                raise e
    
    try:
        if df is None or df.empty:
            raise ValueError(f"No data returned for {symbol} after {max_retries} attempts.")
        
        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        df = df.reset_index()
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        
        # Ensure we have the right columns
        required = ["date", "open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                if col == "volume":
                    df["volume"] = 0
                else:
                    raise ValueError(f"Missing column: {col}")
        
        df = df[required].copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.dropna(subset=["close"])
        
        # Merge with existing
        if existing_df is not None:
            combined = pd.concat([existing_df, df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["date"], keep="last")
            combined = combined.sort_values("date").reset_index(drop=True)
            df = combined
        
        df.to_csv(csv_path, index=False)
        print(f"  {C.GREEN}[OK] {name}: {len(df)} rows saved{C.RESET}")
        return df
        
    except Exception as e:
        print(f"  {C.YELLOW}[WARN] yfinance failed for {name}: {e}{C.RESET}")
        
        # Try v3 fallback
        fallback = _v3_fallback_path(name)
        if fallback:
            print(f"  {C.CYAN}[FALLBACK] Using v3 CSV: {fallback}{C.RESET}")
            df = pd.read_csv(fallback, parse_dates=["date"] if "date" in pd.read_csv(fallback, nrows=1).columns else [0])
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
            if df.columns[0] != "date":
                df = df.rename(columns={df.columns[0]: "date"})
            return df
        
        # Try existing cached CSV
        if existing_df is not None:
            print(f"  {C.YELLOW}[CACHE] Using cached data: {len(existing_df)} rows{C.RESET}")
            return existing_df
        
        raise RuntimeError(f"Cannot load data for {name}. No cache, no fallback.")


def fetch_sentiment_data():
    """
    Fetch the latest PCR and FII/DII data.
    If fetching fails, it simply falls back to last known value or uses generated history.
    """
    print(f"  {C.CYAN}[FETCH] Sentiment: FII/DII and PCR{C.RESET}")
    
    # 1. FII / DII
    fiidii_path = _csv_path("fii_dii")
    df_fii = None
    if os.path.exists(fiidii_path):
        df_fii = pd.read_csv(fiidii_path)
    
    today_str = datetime.now().strftime("%d-%b-%Y")
    
    try:
        if nse_fiidii is not None:
            f_data = nse_fiidii()
            if not f_data.empty:
                f_date = f_data["date"].iloc[0]
                dii_net = float(f_data[f_data["category"] == "DII"]["netValue"].iloc[0])
                fii_net = float(f_data[f_data["category"].str.contains("FII")]["netValue"].iloc[0])
                
                new_row = pd.DataFrame([{"date": pd.to_datetime(f_date).strftime("%Y-%m-%d"), "fii_net": fii_net, "dii_net": dii_net}])
                
                if df_fii is not None:
                    # Drop if exists, append new
                    df_fii = df_fii[df_fii["date"] != new_row["date"].iloc[0]]
                    df_fii = pd.concat([df_fii, new_row], ignore_index=True)
                else:
                    df_fii = new_row
                    
                df_fii.to_csv(fiidii_path, index=False)
                print(f"  {C.GREEN}[OK] FII/DII updated for {f_date}{C.RESET}")
    except Exception as e:
        print(f"  {C.YELLOW}[WARN] Failed to fetch live FII/DII: {e}{C.RESET}")

    # 2. PCR
    pcr_path = _csv_path("pcr")
    df_pcr = None
    if os.path.exists(pcr_path):
        df_pcr = pd.read_csv(pcr_path)
        
    try:
        # NSE Option chain via requests
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': '*/*','Accept-Language': 'en-US,en;q=0.5'
        }
        url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
        # Need a session to grab cookies first
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=headers, timeout=2.0)
        resp = session.get(url, headers=headers, timeout=3.0)
        
        if resp.status_code == 200:
            data = resp.json()
            records = data.get("records", {})
            data_list = records.get("data", [])
            pe_oi = sum(r.get("PE", {}).get("openInterest", 0) for r in data_list)
            ce_oi = sum(r.get("CE", {}).get("openInterest", 0) for r in data_list)
            pcr_val = pe_oi / max(1, ce_oi)
            
            today_iso = datetime.now().strftime("%Y-%m-%d")
            new_row = pd.DataFrame([{"date": today_iso, "pcr": round(pcr_val, 3)}])
            
            if df_pcr is not None:
                # Drop if exists, append new
                df_pcr = df_pcr[df_pcr["date"] != new_row["date"].iloc[0]]
                df_pcr = pd.concat([df_pcr, new_row], ignore_index=True)
            else:
                df_pcr = new_row
                
            df_pcr.to_csv(pcr_path, index=False)
            print(f"  {C.GREEN}[OK] PCR updated: {pcr_val:.3f}{C.RESET}")
    except Exception as e:
        print(f"  {C.CYAN}[FALLBACK] Live PCR timeout ({e}), using daily cache{C.RESET}")
        
    return df_fii, df_pcr

def load_all_data():
    """
    Fetch and merge NIFTY + VIX + S&P 500 into a single DataFrame.
    Returns a clean, merged DataFrame ready for feature engineering.
    """
    print(f"\n{C.header('DATA ENGINE: Loading Market Data')}")
    print(f"{'─'*50}")
    
    nifty = fetch_symbol(NIFTY_SYMBOL, "nifty")
    vix = fetch_symbol(VIX_SYMBOL, "vix")
    sp500 = fetch_symbol(SP500_SYMBOL, "sp500")
    
    # Rename columns for merging
    vix_cols = vix[["date", "close"]].rename(columns={"close": "vix"})
    sp_cols = sp500[["date", "close"]].rename(columns={"close": "sp_close"})
    
    # Merge on date
    df = nifty.merge(vix_cols, on="date", how="left")
    df = df.merge(sp_cols, on="date", how="left")
    
    # Fetch and merge sentiment features (PCR, FII/DII)
    df_fii, df_pcr = fetch_sentiment_data()
    
    if df_fii is not None:
        df_fii["date"] = pd.to_datetime(df_fii["date"])
        df = df.merge(df_fii, on="date", how="left")
        df["fii_net"] = df["fii_net"].ffill().fillna(0)
        df["dii_net"] = df["dii_net"].ffill().fillna(0)
    else:
        df["fii_net"] = 0.0
        df["dii_net"] = 0.0
        
    if df_pcr is not None:
        df_pcr["date"] = pd.to_datetime(df_pcr["date"])
        df = df.merge(df_pcr, on="date", how="left")
        df["pcr"] = df["pcr"].ffill().fillna(1.0)
    else:
        df["pcr"] = 1.0
    
    # Forward-fill VIX and S&P (they may have different trading calendars)
    df["vix"] = df["vix"].ffill().bfill()
    df["sp_close"] = df["sp_close"].ffill().bfill()
    
    # Sort and clean
    df = df.sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=["close"])
    
    print(f"\n  {C.GREEN}[OK] Merged dataset: {len(df)} trading days{C.RESET}")
    print(f"  {C.DIM}     Date range: {df['date'].min().date()} → {df['date'].max().date()}{C.RESET}")
    print(f"  {C.DIM}     Latest close: {df['close'].iloc[-1]:,.2f}{C.RESET}")
    
    return df


if __name__ == "__main__":
    df = load_all_data()
    print(f"\nColumns: {list(df.columns)}")
    print(df.tail())
