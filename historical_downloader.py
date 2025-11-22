"""
historical_downloader.py

Purpose:
    Download & cache daily OHLCV data for NSE tickers using yfinance only.
    No fallback. Pure & simple.

Features:
    - Automatically adds .NS suffix
    - Caches each ticker in data/<TICKER>.csv
    - Skips download if cached file exists (unless force=True)
    - Batch multithreaded downloader
"""

import os
import time
import datetime
from typing import List, Tuple, Dict, Optional
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed


# ==========================
# CONFIG
# ==========================
DEFAULT_CACHE_DIR = "data"
DEFAULT_YEARS = 10
DEFAULT_WORKERS = 6
REQUEST_DELAY = 0.35      # polite delay between downloads


# ==========================
# INTERNAL HELPERS
# ==========================
def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _yf_symbol(ticker: str) -> str:
    """
    yfinance NSE tickers need the .NS suffix.
    """
    t = ticker.upper().strip()
    if t.endswith(".NS"):
        return t
    return f"{t}.NS"


def _cache_path(cache_dir: str, ticker: str) -> str:
    """
    Convert 'RELIANCE' -> 'data/RELIANCE.csv'
    """
    safe = ticker.upper().strip()
    return os.path.join(cache_dir, f"{safe}.csv")


def _read_cache(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, index_col=0)
        df.index = pd.to_datetime(df.index)
        df.sort_index(inplace=True)
        return df
    except Exception:
        return None


def _save_cache(df: pd.DataFrame, path: str):
    df.to_csv(path, index=True)


# ==========================
# MAIN FUNCTION (single ticker)
# ==========================
def download_for_ticker(
    ticker: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    cache_dir: str = DEFAULT_CACHE_DIR,
    force: bool = False
) -> Tuple[str, Optional[pd.DataFrame]]:
    """
    Download OHLCV for ONE ticker using yfinance.

    Returns:
        (cache_path, df or None)
    """

    _ensure_dir(cache_dir)
    cache_path = _cache_path(cache_dir, ticker)

    # Load cache unless forcing fresh download
    if not force:
        cached = _read_cache(cache_path)
        if cached is not None:
            print(f"[CACHE] {ticker}: {cache_path}")
            return cache_path, cached

    # Compute start/end dates
    if end is None:
        end_date = datetime.date.today()
    else:
        end_date = pd.to_datetime(end).date()

    if start is None:
        start_date = end_date - datetime.timedelta(days=DEFAULT_YEARS * 365)
    else:
        start_date = pd.to_datetime(start).date()

    yf_symbol = _yf_symbol(ticker)
    print(f"[DOWNLOAD] {ticker} ({start_date} → {end_date})")

    try:
        df = yf.download(
            yf_symbol,
            start=start_date.isoformat(),
            end=(end_date + datetime.timedelta(days=1)).isoformat(),
            progress=False,
            auto_adjust=False
        )
    except Exception as e:
        print(f"[ERROR] yfinance failed for {ticker}: {e}")
        return cache_path, None

    time.sleep(REQUEST_DELAY)  # rate limit

    if df is None or df.empty:
        print(f"[NODATA] {ticker}")
        return cache_path, None

    # normalize index
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    # Save cache
    _save_cache(df, cache_path)
    print(f"[SAVED] {ticker}: {len(df)} rows → {cache_path}")

    return cache_path, df


# ==========================
# BATCH DOWNLOADER (multi-threaded)
# ==========================
def batch_download_tickers(
    tickers: List[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    cache_dir: str = DEFAULT_CACHE_DIR,
    force: bool = False,
    workers: int = DEFAULT_WORKERS
) -> Dict[str, Dict]:
    """
    Download a list of tickers in parallel using a thread pool.

    Returns:
        dict[ticker] = {
            "cache_path": "...",
            "df": dataframe or None,
            "error": None or error string
        }
    """
    _ensure_dir(cache_dir)
    results = {}

    def _task(t):
        try:
            path, df = download_for_ticker(
                t, start=start, end=end, cache_dir=cache_dir, force=force
            )
            return t, path, df, None
        except Exception as e:
            return t, _cache_path(cache_dir, t), None, str(e)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_task, t): t for t in tickers}
        for f in as_completed(futures):
            t = futures[f]
            try:
                ticker, path, df, err = f.result()
                results[ticker] = {"cache_path": path, "df": df, "error": err}
            except Exception as e:
                results[t] = {"cache_path": _cache_path(cache_dir, t), "df": None, "error": str(e)}

    return results


# ==========================
# OPTIONAL CLI RUNNER
# ==========================
if __name__ == "__main__":
    import pandas as pd

    CSV_PATH = "/mnt/data/52WeekHigh.csv"  # your uploaded file

    df = pd.read_csv(CSV_PATH)
    col = df.columns[0]      # assume first column = symbol
    tickers = df[col].astype(str).str.strip().tolist()

    print(f"Downloading {len(tickers)} tickers...")

    res = batch_download_tickers(tickers, workers=6)

    print("Done. Summary:")
    ok = sum(1 for v in res.values() if v["df"] is not None)
    fail = sum(1 for v in res.values() if v["df"] is None)
    print(f"Success = {ok}, Failed = {fail}")
