# New Features - Quick Reference

## 🆕 What's New

Six enhancements for better organization and control:

1. **Start Date Filter** - Focus charts on recent data
2. **Top-N Selection** - Limit individual charts
3. **CSV Directory** - Organized data files
4. **Plots Directory** - Organized dashboard files
5. **Charts Directory** - Organized individual charts
6. **Auto-Creation** - All directories automatic

---

## ⚡ Quick Commands

### Default (Recommended)
```bash
python stock_trend_analyzer.py
```
- Start: 2025-01-01
- Top-N: 10 stocks
- Directories: Auto-created

### Top 5 Best
```bash
python stock_trend_analyzer.py --top-n 5
```

### Recent 3 Months
```bash
python stock_trend_analyzer.py --start-date 2024-11-01
```

### All Trending Stocks
```bash
python stock_trend_analyzer.py --top-n 0
```

### Full Custom
```bash
python stock_trend_analyzer.py \
  --start-date 2024-09-01 \
  --top-n 15
```

---

## 📅 Start Date (--start-date)

**Default:** 2025-01-01  
**Format:** YYYY-MM-DD

**What it does:**
- Filters individual technical charts
- Shows data from date forward
- Focus on recent price action

**Examples:**
```bash
# Last 6 months
--start-date 2024-10-01

# This year
--start-date 2025-01-01

# Custom
--start-date 2024-07-15
```

**Affects:**
- ✅ Individual charts (visualization)
- ❌ Dashboard (uses all data)
- ❌ Analysis (uses all data)
- ❌ CSV output (all data)

---

## 🏆 Top-N (--top-n)

**Default:** 10  
**0 = All:** No limit

**What it does:**
- Limits individual chart generation
- Only trending stocks (≥4.0)
- Sorted by score (best first)

**Examples:**
```bash
# Top 3
--top-n 3

# Top 10 (default)
--top-n 10

# All trending
--top-n 0
```

**Smart:**
- Dashboard shows all stocks
- Charts only for top N trending
- Saves time, focuses on best

---

## 📁 Directory Structure

**Automatic:**
```
csv/                 # All data files
plots/               # All dashboards
trending_charts/     # Individual charts
```

**Created automatically** on first run

**Naming:**
```
csv/trending_stocks_20251104_123045.csv
plots/trending_stocks_20251104_123045_plot.png
trending_charts/AAPL_technical_analysis.png
```

---

## 💡 Common Patterns

### Daily Quick Check
```bash
python stock_trend_analyzer.py --top-n 5
```
Fast, focused on top 5

### Weekly Review
```bash
python stock_trend_analyzer.py \
  --start-date 2024-10-01 \
  --top-n 15
```
Recent 3 months, top 15

### Full Analysis
```bash
python stock_trend_analyzer.py \
  --start-date 2024-01-01 \
  --top-n 0
```
Full year, all trending

### Lightning Fast
```bash
python stock_trend_analyzer.py \
  --start-date 2025-03-01 \
  --top-n 3
```
2 months, just top 3

---

## 📊 What You Get

### csv/ Directory
- trending_stocks_TIMESTAMP.csv
- All analysis data
- Accumulates over time

### plots/ Directory
- trending_stocks_TIMESTAMP_plot.png
- Dashboard comparison
- Accumulates over time

### trending_charts/ Directory
- TICKER_technical_analysis.png
- One per trending stock
- Updated each run (top-n only)

---

## 🎯 Behavior Details

### Start Date
- Only visual (charts)
- Analysis uses all data
- Ensures accurate indicators
- Default: Jan 1, 2025

### Top-N
- Only trending stocks (≥4.0)
- Best scores first
- Dashboard unaffected
- Default: 10 stocks

### Directories
- Created once
- Persist forever
- No user action needed
- Clean organization

---

## ⚙️ Full Syntax

```bash
python stock_trend_analyzer.py \
  [--tickers X,Y,Z | --file path] \
  [--start-date YYYY-MM-DD] \
  [--top-n N] \
  [--interval 1d|5min|...] \
  [--period compact|full] \
  [--output filename.csv] \
  [--log-level LEVEL]
```

---

## 🔧 Examples by Use Case

### Screening
```bash
python stock_trend_analyzer.py --top-n 5
```

### Research
```bash
python stock_trend_analyzer.py --top-n 0
```

### Day Trading
```bash
python stock_trend_analyzer.py \
  --interval 5min \
  --start-date 2025-04-01 \
  --top-n 3
```

### Swing Trading
```bash
python stock_trend_analyzer.py \
  --start-date 2024-10-01 \
  --top-n 10
```

---

## ✅ Quick Checklist

**Before running:**
- [ ] API key set
- [ ] Ticker file ready (or use --tickers)

**Customize:**
- [ ] Start date (default: 2025-01-01)
- [ ] Top-N (default: 10)
- [ ] Output name (optional)

**After running:**
- [ ] Check csv/ for data
- [ ] Check plots/ for dashboard
- [ ] Check trending_charts/ for charts

---

## 🚀 TL;DR

**Simplest:**
```bash
python stock_trend_analyzer.py
```

**Most Common:**
```bash
python stock_trend_analyzer.py --top-n 5
```

**Full Control:**
```bash
python stock_trend_analyzer.py \
  --start-date 2024-09-01 \
  --top-n 15
```

**Everything organized automatically! 📁🎯**
