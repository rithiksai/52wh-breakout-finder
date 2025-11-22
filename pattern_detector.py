"""
pattern_detector.py  (loader patched - robust CSV detection)

Only change from previous version is a much more robust _safe_load_csv() that:
 - searches the first few lines to find the header row (a row containing 'Date'),
 - falls back to heuristics and positional mapping when headers are missing,
 - avoids deprecated pandas args that cause FutureWarnings.

Everything else (detect_breakout_pattern, scan_cache_for_patterns) is unchanged
and will continue to work as before.
"""

from typing import Optional, Dict, List, Tuple
import os
import glob
import pandas as pd
import datetime
import traceback

REFERENCE_IMAGE_PATH = "/mnt/data/Screenshot 2025-11-22 at 12.35.00 PM.png"
REFERENCE_IMAGE_PATH_2 = "/mnt/data/Screenshot 2025-11-22 at 11.25.13 AM.png"


# -------------------------
# Robust CSV loader
# -------------------------
def _find_header_row(path: str, max_lines: int = 8) -> Optional[int]:
    """
    Inspect the first max_lines lines and return the index of the header row
    (the row that contains the string 'Date' in any column) if found.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            for i in range(max_lines):
                line = f.readline()
                if not line:
                    break
                if "Date" in line or "DATE" in line or "date" in line:
                    return i
    except Exception:
        return None
    return None


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Try to ensure df columns include High/Low/Close/Volume if possible.
    If header row got parsed into column names (like ticker symbols),
    detect numeric columns and map them by position to canonical names:
      ['Price','Adj Close','Close','High','Low','Open','Volume']
    """
    canonical = ["Price", "Adj Close", "Close", "High", "Low", "Open", "Volume"]

    # If obvious columns present, return unchanged
    lower_cols = [c.lower() for c in df.columns]
    if any(c in lower_cols for c in ("high", "close", "low")):
        return df

    # Otherwise attempt positional mapping:
    # pick numeric columns only (exclude non-numeric)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        # if none numeric, try coercion
        coerced = df.copy()
        for c in df.columns:
            coerced[c] = pd.to_numeric(coerced[c], errors="coerce")
        numeric_cols = [c for c in coerced.columns if coerced[c].notna().any()]
        df = coerced

    # map as many canonical names as possible using left-to-right order
    mapped = {}
    for i, col in enumerate(numeric_cols):
        if i < len(canonical):
            mapped[col] = canonical[i]

    if mapped:
        df = df.rename(columns=mapped)
    return df


def _safe_load_csv(path: str) -> Optional[pd.DataFrame]:
    """
    Robust loader that:
      - finds header row (first row that contains 'Date') and uses it as header
      - if not found, tries skiprows=2 (existing format)
      - if still not OK, loads without header and tries to coerce/rename columns by position
    Returns DataFrame indexed by Date, or None if unreadable.
    """
    if not os.path.exists(path):
        return None

    # 1) Try to find header row dynamically
    header_row = _find_header_row(path, max_lines=8)
    if header_row is not None:
        try:
            # Use header_row as header
            df = pd.read_csv(path, header=header_row, parse_dates=["Date"])
            if "Date" in df.columns:
                df = df.set_index("Date")
                df.index = pd.to_datetime(df.index, errors="coerce")
                df = df[~df.index.isna()]
                df.sort_index(inplace=True)
                df = _normalize_column_names(df)
                if not df.empty:
                    return df
        except Exception:
            # fall through to other strategies
            pass

    # 2) Try the original skiprows=2 approach
    try:
        df = pd.read_csv(path, skiprows=2)
        if "Date" in df.columns:
            # parse Date column
            try:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                df = df.loc[~df["Date"].isna()].set_index("Date")
                df.sort_index(inplace=True)
                df = _normalize_column_names(df)
                if not df.empty:
                    return df
            except Exception:
                pass
        else:
            # maybe pandas already made index the date
            # try reading with index_col=0
            df2 = pd.read_csv(path, skiprows=2, index_col=0, parse_dates=True)
            if isinstance(df2.index, pd.DatetimeIndex) and not df2.empty:
                df2.index = pd.to_datetime(df2.index, errors="coerce")
                df2 = df2[~df2.index.isna()]
                df2.sort_index(inplace=True)
                df2 = _normalize_column_names(df2)
                if not df2.empty:
                    return df2
    except Exception:
        pass

    # 3) Fallback: read full file without header and try to coerce first column to Date
    try:
        raw = pd.read_csv(path, header=None, dtype=str)
        if raw.shape[1] >= 2:
            # coerce first column
            raw.iloc[:, 0] = pd.to_datetime(raw.iloc[:, 0], errors="coerce")
            raw = raw.loc[~raw.iloc[:, 0].isna()]
            if not raw.empty:
                raw = raw.set_index(raw.columns[0])
                # convert remaining columns to numeric where possible
                for c in raw.columns:
                    raw[c] = pd.to_numeric(raw[c], errors="coerce")
                raw = _normalize_column_names(raw)
                raw.sort_index(inplace=True)
                if not raw.empty:
                    return raw
    except Exception:
        pass

    # 4) Final attempt: read with index_col=0, parse_dates True
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if isinstance(df.index, pd.DatetimeIndex) and not df.empty:
            df.index = pd.to_datetime(df.index, errors="coerce")
            df = df[~df.index.isna()]
            df.sort_index(inplace=True)
            df = _normalize_column_names(df)
            if not df.empty:
                return df
    except Exception:
        pass

    # Give up
    return None


# -------------------------
# The rest of the module: pattern detection functions
# (unchanged from prior working version)
# -------------------------
def load_ohlc_from_cache(cache_dir: str, ticker: str) -> Optional[pd.DataFrame]:
    path = os.path.join(cache_dir, f"{ticker}.csv")
    return _safe_load_csv(path)


def compute_emas(df: pd.DataFrame, short: int = 20, long: int = 50) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
    if "Close" not in df.columns and "Adj Close" in df.columns:
        price = df["Adj Close"]
    elif "Close" in df.columns:
        price = df["Close"]
    else:
        price = df.iloc[:, 0]
    if len(price) < max(short, long):
        return None, None
    ema_short = price.ewm(span=short, adjust=False).mean()
    ema_long = price.ewm(span=long, adjust=False).mean()
    return ema_short, ema_long


def avg_volume(df: pd.DataFrame, lookback: int = 10) -> Tuple[Optional[float], Optional[float]]:
    vcol = None
    for c in ("Volume", "volume", "VOL"):
        if c in df.columns:
            vcol = c
            break
    if vcol is None:
        return None, None
    all_avg = float(df[vcol].replace(0, float('nan')).dropna().mean()) if len(df) > 0 else None
    recent = df.tail(lookback)[vcol].replace(0, float('nan')).dropna()
    recent_avg = float(recent.mean()) if not recent.empty else None
    return all_avg, recent_avg


def detect_breakout_pattern(
    df: pd.DataFrame,
    min_distance: int = 60,
    min_trough_drop: float = 0.20,
    breakout_threshold: float = 0.00,
    require_ema_dir: bool = True,
    ema_short: int = 20,
    ema_long: int = 50,
    require_volume_spike: bool = False,
    volume_spike_factor: float = 1.2,
    verbose: bool = False
) -> Tuple[bool, Dict]:
    diag = {"reason": None}
    try:
        if df is None or df.empty:
            diag["reason"] = "no_data"
            return False, diag
        n = len(df)
        if n < 10:
            diag["reason"] = f"too_short_n={n}"
            return False, diag

        if "High" not in df.columns and "Close" not in df.columns:
            diag["reason"] = "no_price_columns"
            return False, diag

        recent_exclude = min_distance
        if n <= recent_exclude + 1:
            diag["reason"] = f"not_enough_history_for_min_distance (n={n} <= min_distance={min_distance})"
            return False, diag

        prior_region = df.iloc[: n - recent_exclude]
        if prior_region.empty:
            diag["reason"] = "no_prior_region"
            return False, diag

        high_col = "High" if "High" in prior_region.columns else ("Close" if "Close" in prior_region.columns else prior_region.columns[0])
        low_col = "Low" if "Low" in df.columns else ("Close" if "Close" in df.columns else df.columns[0])
        close_col = "Close" if "Close" in df.columns else ("Adj Close" if "Adj Close" in df.columns else df.columns[0])

        idx_prior = prior_region[high_col].idxmax()
        prior_high = float(prior_region.loc[idx_prior, high_col])
        prior_high_date = idx_prior.date()
        diag["prior_high"] = prior_high
        diag["prior_high_date"] = str(prior_high_date)

        start_idx = df.index.get_loc(idx_prior) + 1
        end_idx = n - recent_exclude - 1
        if start_idx > end_idx:
            search_region = df.iloc[: n - recent_exclude]
        else:
            search_region = df.iloc[start_idx: end_idx + 1]

        if search_region.empty:
            fallback_region = df.iloc[df.index.get_loc(idx_prior)+1 : n - recent_exclude]
            if fallback_region.empty:
                diag["reason"] = "no_trough_region"
                return False, diag
            trough_val = float(fallback_region[low_col].min())
            trough_idx = fallback_region[low_col].idxmin()
        else:
            trough_val = float(search_region[low_col].min())
            trough_idx = search_region[low_col].idxmin()

        diag["trough"] = trough_val
        diag["trough_date"] = str(trough_idx.date())

        trough_depth = (prior_high - trough_val) / prior_high if prior_high > 0 else 0.0
        diag["trough_depth"] = trough_depth

        if trough_depth < min_trough_drop:
            diag["reason"] = f"trough_too_shallow (depth={trough_depth:.3f} < min_trough_drop={min_trough_drop})"
            return False, diag

        last_close = float(df[close_col].iloc[-1])
        last_date = df.index[-1].date()
        diag["last_close"] = last_close
        diag["last_date"] = str(last_date)

        required = prior_high * (1.0 + breakout_threshold)
        diag["required_breakout_price"] = required
        raw_breakout = last_close > required
        diag["raw_breakout"] = bool(raw_breakout)

        if not raw_breakout:
            diag["reason"] = "no_breakout_now"
            return False, diag

        if require_ema_dir:
            ema_short_s, ema_long_s = compute_emas(df, short=ema_short, long=ema_long)
            diag["ema_short_latest"] = float(ema_short_s.iloc[-1]) if ema_short_s is not None else None
            diag["ema_long_latest"] = float(ema_long_s.iloc[-1]) if ema_long_s is not None else None
            if ema_short_s is None or ema_long_s is None:
                diag["reason"] = "ema_unavailable"
                return False, diag
            if ema_short_s.iloc[-1] <= ema_long_s.iloc[-1]:
                diag["reason"] = "ema_direction_not_ok"
                return False, diag

        if require_volume_spike:
            all_avg, recent_avg = avg_volume(df, lookback=10)
            diag["all_avg_vol"] = all_avg
            diag["recent_avg_vol"] = recent_avg
            if all_avg is None or recent_avg is None:
                diag["reason"] = "volume_unavailable"
                return False, diag
            if recent_avg < (all_avg * volume_spike_factor):
                diag["reason"] = f"volume_no_spike (recent {recent_avg:.0f} < {volume_spike_factor}*all_avg {all_avg:.0f})"
                return False, diag

        diag["accepted"] = True
        diag["reason"] = "accepted"
        return True, diag
    except Exception as e:
        diag["reason"] = f"error:{e}"
        diag["trace"] = traceback.format_exc()
        return False, diag


# -------------------------
# Scanning function (unchanged)
# -------------------------
def scan_cache_for_patterns(
    cache_dir: str = "data",
    min_distance: int = 60,
    min_trough_drop: float = 0.20,
    breakout_threshold: float = 0.00,
    require_ema_dir: bool = True,
    ema_short: int = 20,
    ema_long: int = 50,
    require_volume_spike: bool = False,
    volume_spike_factor: float = 1.2,
    min_history_days: int = 30,
    verbose: bool = True
) -> List[Dict]:
    out = []
    csv_paths = glob.glob(os.path.join(cache_dir, "*.csv"))
    tickers = [os.path.splitext(os.path.basename(p))[0] for p in csv_paths]
    if verbose:
        print(f"Scanning {len(tickers)} cached files in '{cache_dir}'")
    for t in sorted(tickers):
        if verbose:
            print(f" - {t} ...", end=" ")
        try:
            df = load_ohlc_from_cache(cache_dir, t)
            if df is None or df.empty:
                if verbose:
                    print("no data / unreadable")
                continue
            if len(df) < min_history_days:
                if verbose:
                    print(f"skip short ({len(df)} rows < min_history_days={min_history_days})")
                continue
            ok, diag = detect_breakout_pattern(
                df,
                min_distance=min_distance,
                min_trough_drop=min_trough_drop,
                breakout_threshold=breakout_threshold,
                require_ema_dir=require_ema_dir,
                ema_short=ema_short,
                ema_long=ema_long,
                require_volume_spike=require_volume_spike,
                volume_spike_factor=volume_spike_factor,
                verbose=False
            )
            if ok:
                rec = {"ticker": t}
                rec.update(diag)
                rec["cache_path"] = os.path.join(cache_dir, f"{t}.csv")
                rec["rows"] = len(df)
                out.append(rec)
                if verbose:
                    print("ACCEPT")
            else:
                if verbose:
                    print(f"REJECT ({diag.get('reason')})")
        except Exception as e:
            if verbose:
                print(f"ERROR ({e})")
            continue
    out.sort(key=lambda r: (r.get("prior_high", 0), r.get("trough_depth", 0)), reverse=True)
    return out


def get_reference_image_paths() -> Tuple[str, str]:
    return REFERENCE_IMAGE_PATH, REFERENCE_IMAGE_PATH_2


if __name__ == "__main__":
    res = scan_cache_for_patterns(cache_dir="data", min_distance=60, min_trough_drop=0.20, verbose=True)
    print("\nMatches:", len(res))
    if res:
        import pandas as pd, os
        os.makedirs("results", exist_ok=True)
        pd.DataFrame(res).to_csv("results/breakout_candidates.csv", index=False)
        print("Saved results to results/breakout_candidates.csv")
    else:
        print("No matches found with current settings.")
