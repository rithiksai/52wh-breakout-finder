# 📈 52-Week High Breakout Pattern Scanner

### *Automated Stock Pattern Detection System (NSE India)*

This project scans NSE India tickers that are near their **52-week highs**, downloads their historical data, and identifies a specific **breakout pattern** similar to the ones shown in stocks like **DYNAMATECH** and **NATIONALALUM**:

- A **prior high** that happened long back
- A **deep correction / trough** after that high
- A **strong recovery**
- A **clean breakout above the prior high**
- Optional: EMA trend confirmation + volume confirmation

The system is modular, easy to extend, and simple to maintain in the future.

---

## 📁 Project Structure

```
project-root/
│
├── main.py                      # Orchestrator (runs the full pipeline)
├── get_tickers.py               # Loads tickers from NSE 52-week high CSV
├── historical_downloader.py     # Downloads OHLC data for each ticker (yfinance)
├── pattern_detector.py          # Robust CSV loader + breakout pattern logic
│
├── 52WeekHigh.csv               # Latest NSE 52-week high CSV
│
├── data/                        # Cached OHLC CSVs (auto-generated)
│   └── <TICKER>.csv
│
├── results/                     # Output folder (auto-generated)
│   └── breakout_candidates.csv
│
└── charts/                      # (optional) If a chart generator is added later
```

---

## 🚀 High-Level Workflow

1. **Load tickers** from `52WeekHigh.csv`
2. **Ensure cached historical data** (download using yfinance if missing)
3. **Load OHLC data safely** using a robust CSV parser
4. **Run breakout pattern detection**
5. **Save the accepted candidates** into `results/breakout_candidates.csv`

---

## ▶️ How to Run

Inside the project folder:

```bash
python3 main.py
```

The script will:

- Load tickers
- Download missing historical data
- Detect breakout patterns
- Save results

---

## ⚙️ CLI Options

```bash
python3 main.py --help
```

### Available Options

| Argument | Description |
|----------|-------------|
| `--csv <path>` | Custom path to 52-week-high CSV |
| `--force` | Force re-download of all tickers |
| `--limit N` | Only process first N tickers (for testing) |
| `--min-history N` | Minimum rows required (default 30) |
| `--breakout-threshold X` | 0.01 = 1% above prior high |

### Examples

Process first 10 tickers:

```bash
python3 main.py --limit 10
```

Force re-download all data:

```bash
python3 main.py --force
```

Use a different CSV file:

```bash
python3 main.py --csv new.csv
```

---

## 📚 Pattern Detection (Short Summary)

The pattern detector checks for:

### 1️⃣ Prior High

Earlier than `min_distance` candles before the most recent data.

### 2️⃣ Trough (Correction)

Stock must drop by at least:

```python
min_trough_drop = 0.20  # 20% correction
```

### 3️⃣ Breakout

Current closing price > prior high × (1 + threshold)

### Optional Checks

- **EMA20 > EMA50** (trend confirmation)
- **Recent volume > average volume × factor** (optional)

These can be configured inside `main.py`:

```python
MIN_DISTANCE = 60
MIN_TROUGH_DROP = 0.20
BREAKOUT_THRESHOLD = 0.00
REQUIRE_EMA_DIR = True
REQUIRE_VOLUME_SPIKE = False
```

---

## 📤 Output

Results are saved to:

```
results/breakout_candidates.csv
```

Each row contains:

- ticker
- prior high + date
- trough + depth
- last close + date
- breakout confirmation
- EMA values
- reason (usually "accepted")
- number of rows of data

---

## 🧩 Updating the System Later

### Update tickers

Replace `52WeekHigh.csv` with a new one.

### Adjust pattern settings

Modify variables inside `main.py`:

```python
MIN_DISTANCE
MIN_TROUGH_DROP
BREAKOUT_THRESHOLD
REQUIRE_EMA_DIR
REQUIRE_VOLUME_SPIKE
```

### Change data source

Update `historical_downloader.py` (uses yfinance now).

### Modify pattern logic

All logic is inside `detect_breakout_pattern()`.

---

## 🌟 Suggested Future Extensions

- **Chart generator** (plot breakout regions)
- **Backtesting engine**
- **Parallel downloads** (multiprocessing)
- **Store data in SQLite/Postgres**
- **Telegram/Email alerts**

Just search for the section in `main.py`.

---

## 🙌 Final Notes

- The CSV loader is now extremely robust — handles all weird formats.
- The system detects the exact pattern used in your manual analysis.
- The orchestration is clean and easy to extend.

This README tells me everything I need in the future to resume work on the project.

---

## 💝 Credits

**Made with ❤️ for Finance by [Rithik Sai](https://github.com/rithiksai)**

*Empowering traders with automated pattern detection and data-driven insights.*