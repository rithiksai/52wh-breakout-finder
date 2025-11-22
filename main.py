#!/usr/bin/env python3
"""
main.py - Revised Orchestrator (compatible with improved pattern_detector)

Flow:
    1) Load tickers from CSV
    2) Add manual tickers (example: DYNAMATECH)
    3) For each ticker:
         - If not cached, download using download_for_ticker()
         - Load cached OHLC via pattern_detector loader
         - Run detect_breakout_pattern()
    4) Save accepted tickers to results/breakout_candidates.csv
"""

import os
import argparse
import datetime
import pandas as pd

from get_tickers import load_52w_high_tickers
from historical_downloader import download_for_ticker

# NEW imports (use the improved pattern module)
from pattern_detector import (
    load_ohlc_from_cache,
    detect_breakout_pattern,
    get_reference_image_paths,
)

# ------------------------------
# Directories
# ------------------------------
CSV_PATH = "52WeekHigh.csv"
CACHE_DIR = "data"
RESULTS_DIR = "results"
RESULTS_CSV = os.path.join(RESULTS_DIR, "breakout_candidates.csv")

# ------------------------------
# Pattern detection settings
# (These directly map to detect_breakout_pattern)
# ------------------------------
MIN_DISTANCE = 60          # prior high must be at least 60 candles before end
MIN_TROUGH_DROP = 0.20     # require at least 20% correction after prior high
BREAKOUT_THRESHOLD = 0.00  # price > prior_high (0% margin)
REQUIRE_EMA_DIR = True     # require EMA20 > EMA50
REQUIRE_VOLUME_SPIKE = False
VOLUME_SPIKE_FACTOR = 1.2
MIN_HISTORY_DAYS = 30

# ------------------------------
def ensure_dirs():
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

def is_cached_and_valid(ticker: str) -> bool:
    """Return True if cached file exists and successfully loads."""
    try:
        df = load_ohlc_from_cache(CACHE_DIR, ticker)
        return (df is not None) and (not df.empty)
    except:
        return False

# ------------------------------
def process_ticker(ticker: str, force_download: bool = False):
    """
    1) Ensure cached file exists (download if required)
    2) Load OHLC
    3) Run detect_breakout_pattern()
    4) Return accepted record or None
    """

    cache_path = os.path.join(CACHE_DIR, f"{ticker}.csv")

    # ------------------------------
    # Step 1 — Download if needed
    # ------------------------------
    if force_download or not is_cached_and_valid(ticker):
        print(f"[DOWNLOAD] {ticker} -> {cache_path}")
        _, df = download_for_ticker(ticker, cache_dir=CACHE_DIR, force=force_download)

        if df is None or df.empty:
            print(f"[NODATA] {ticker}: no data after download")
            return None
    else:
        print(f"[CACHE HIT] {ticker}")

    # ------------------------------
    # Step 2 — Load processed OHLC
    # ------------------------------
    df = load_ohlc_from_cache(CACHE_DIR, ticker)
    if df is None or df.empty:
        print(f"[ERROR] {ticker}: unreadable cached file")
        return None

    if len(df) < MIN_HISTORY_DAYS:
        print(f"[INFO] {ticker}: short history ({len(df)} rows). Still evaluating.")

    # ------------------------------
    # Step 3 — Run pattern detector
    # ------------------------------
    accepted, diag = detect_breakout_pattern(
        df,
        min_distance=MIN_DISTANCE,
        min_trough_drop=MIN_TROUGH_DROP,
        breakout_threshold=BREAKOUT_THRESHOLD,
        require_ema_dir=REQUIRE_EMA_DIR,
        require_volume_spike=REQUIRE_VOLUME_SPIKE,
        volume_spike_factor=VOLUME_SPIKE_FACTOR
    )

    # ------------------------------
    # Step 4 — Return result
    # ------------------------------
    if accepted:
        print(f"[ACCEPT] {ticker}: breakout pattern found.")
        rec = {"ticker": ticker, "cache_path": cache_path, "rows": len(df)}
        rec.update(diag)
        return rec
    else:
        print(f"[REJECT] {ticker}: {diag.get('reason')}")
        return None

# ------------------------------
def orchestrate(tickers, force_download=False, limit=None):
    ensure_dirs()
    candidates = []

    if limit:
        tickers = tickers[:limit]

    for i, t in enumerate(tickers, start=1):
        print(f"\n=== ({i}/{len(tickers)}) Processing {t} ===")
        rec = process_ticker(t, force_download=force_download)
        if rec:
            candidates.append(rec)

    if candidates:
        pd.DataFrame(candidates).to_csv(RESULTS_CSV, index=False)
        print(f"\nSaved {len(candidates)} candidates → {RESULTS_CSV}")
    else:
        print("\nNo breakout candidates found.")

    # Show your example images for visual correctness
    ref1, ref2 = get_reference_image_paths()
    print("\nReference images for expected pattern:")
    print(" -", ref1)
    print(" -", ref2)

# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Breakout pattern scanner (improved version)")
    parser.add_argument("--csv", type=str, default=CSV_PATH)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    # 1) Load tickers
    tickers = load_52w_high_tickers(args.csv)

    # 2) Manually include DYNAMATECH
    tickers.append("DYNAMATECH")

    print("Loaded tickers:", tickers)

    orchestrate(tickers, force_download=args.force, limit=args.limit)

# ------------------------------
if __name__ == "__main__":
    main()
