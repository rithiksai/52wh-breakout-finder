"""
Module: load_52w_high_csv.py

Purpose:
    Load the NSE 52-week high CSV downloaded from the NSE website
    and extract the ticker list cleanly.

Usage:
    from load_52w_high_csv import load_52w_high_tickers

    tickers = load_52w_high_tickers("/path/to/52WeekHigh.csv")
    print(tickers)
"""

import pandas as pd
from typing import List


def load_52w_high_tickers(csv_path: str) -> List[str]:
    """
    Load NSE 52-week high CSV and return a list of ticker symbols.

    Assumes the CSV contains a column such as:
      - 'SYMBOL'
      - or 'Symbol'
      - or similar

    Automatically finds the correct column.
    """
    df = pd.read_csv(csv_path)

    # Normalize column names
    df.columns = [c.strip().upper() for c in df.columns]

    # Try to detect the symbol column
    possible_cols = ["SYMBOL", "TICKER", "SECURITY", "NAME"]

    symbol_col = None
    for col in possible_cols:
        if col in df.columns:
            symbol_col = col
            break

    if symbol_col is None:
        raise ValueError(
            f"Could not identify ticker column. Available columns: {df.columns}"
        )

    # Extract tickers
    tickers = (
        df[symbol_col]
        .astype(str)
        .str.strip()
        .str.upper()
        .tolist()
    )

    # Remove blanks / invalids
    tickers = [t for t in tickers if t not in ["", None] and t.isalnum()]

    return sorted(list(set(tickers)))  # dedup + sort
