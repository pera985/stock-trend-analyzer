# Stock Trend Analyzer - New Features Summary

## 🎯 Latest Updates

Six major enhancements have been added to improve organization and flexibility:

### 1. Start Date Filter for Charts ✅
### 2. Top-N Stock Selection ✅
### 3. Organized CSV Directory ✅
### 4. Organized Plots Directory ✅
### 5. Organized Trending Charts Directory ✅
### 6. Automatic Directory Creation ✅

---

## 📅 1. Start Date Filter (--start-date)

### Purpose
Restrict individual stock charts to show data from a specific date forward, providing focused analysis on recent price action.

### Usage
```bash
# Use default (January 1, 2025)
python stock_trend_analyzer.py

# Specify custom start date
python stock_trend_analyzer.py --start-date 2024-06-01

# Different date format examples
python stock_trend_analyzer.py --start-date 2025-03-15
python stock_trend_analyzer.py --start-date 2024-01-01
```

### Default
**Date:** January 1, 2025 (2025-01-01)  
**Format:** YYYY-MM-DD

### Behavior
- Filters data in individual technical charts only
- Dashboard plot shows all data (unfiltered)
- Charts display actual date range in subtitle
- Empty data warning if no data after start date

### Examples

**Recent 6-month analysis:**
```bash
python stock_trend_analyzer.py --start-date 2024-10-01
```

**Year-to-date 2025:**
```bash
python stock_trend_analyzer.py --start-date 2025-01-01  # default
```

**Custom period:**
```bash
python stock_trend_analyzer.py --start-date 2024-07-15
```

---

## 🏆 2. Top-N Stock Selection (--top-n)

### Purpose
Generate individual technical charts for only the top N trending stocks (by score), reducing processing time and focusing on best opportunities.

### Usage
```bash
# Use default (top 10)
python stock_trend_analyzer.py

# Top 5 stocks only
python stock_trend_analyzer.py --top-n 5

# Top 20 stocks
python stock_trend_analyzer.py --top-n 20

# All trending stocks (no limit)
python stock_trend_analyzer.py --top-n 0
```

### Default
**Count:** 10 stocks  
**0 = All:** Use 0 for unlimited (all trending stocks)

### Behavior
- Only affects trending stocks (score ≥ 4.0)
- Sorted by score (highest first)
- Dashboard plot shows all stocks
- Individual charts generated for top N only

### Smart Filtering
```bash
# If you have 25 trending stocks and set --top-n 10:
# - All 25 shown in dashboard
# - Only top 10 get individual charts
# - Saves time and focuses on best opportunities
```

### Examples

**Quick analysis (top 3):**
```bash
python stock_trend_analyzer.py --top-n 3
```

**Standard analysis (top 10):**
```bash
python stock_trend_analyzer.py  # default
```

**Comprehensive (all trending):**
```bash
python stock_trend_analyzer.py --top-n 0
```

---

## 📁 3. Organized Directory Structure

### Purpose
Automatically organize outputs into separate directories for better file management.

### Directory Structure
```
your_project/
├── csv/                      # All CSV data files
│   └── trending_stocks_20251104_123045.csv
├── plots/                    # All dashboard PNG files
│   └── trending_stocks_20251104_123045_plot.png
├── trending_charts/          # All individual technical charts
│   ├── AAPL_technical_analysis.png
│   ├── MSFT_technical_analysis.png
│   └── GOOGL_technical_analysis.png
├── stock_trend_analyzer.py
└── plot_individual_tickers.py
```

### Automatic Creation
All three directories are **created automatically** if they don't exist:
- `csv/` - CSV data files
- `plots/` - Dashboard comparison plots
- `trending_charts/` - Individual technical charts

### Behavior
```bash
# First run
python stock_trend_analyzer.py
# Output:
# Created directory: csv/
# Created directory: plots/
# Created directory: trending_charts/

# Subsequent runs
python stock_trend_analyzer.py
# Directories already exist, files added to them
```

---

## 📊 Detailed Directory Descriptions

### csv/ Directory
**Contains:** All CSV data files  
**Naming:** `trending_stocks_YYYYMMDD_HHMMSS.csv` or custom name  
**Content:** Analysis results, scores, indicators

**Files include:**
- ticker
- is_trending
- score
- current_price
- All technical indicators
- Criteria pass/fail

### plots/ Directory
**Contains:** Dashboard comparison plots  
**Naming:** `trending_stocks_YYYYMMDD_HHMMSS_plot.png` or custom  
**Content:** 6-panel dashboard visualization

**Panels include:**
- Score comparison (all stocks)
- Distribution histogram
- Top 10 performers
- Statistics table
- Criteria breakdown
- Professional layout

### trending_charts/ Directory
**Contains:** Individual technical analysis charts  
**Naming:** `TICKER_technical_analysis.png`  
**Content:** 3-panel technical chart per stock

**Panels include:**
- Bollinger Bands with price
- Volume with moving averages
- Color-coded RSI

**Only includes:** Trending stocks (score ≥ 4.0), limited by --top-n

---

## 🎯 Combined Usage Examples

### Example 1: Default Behavior
```bash
python stock_trend_analyzer.py
```

**Result:**
- Start date: 2025-01-01
- Top-n: 10 stocks
- Directories: Auto-created (csv/, plots/, trending_charts/)
- Charts: Show data from Jan 1, 2025 onwards

### Example 2: Recent 3-Month Analysis, Top 5
```bash
python stock_trend_analyzer.py --start-date 2024-11-01 --top-n 5
```

**Result:**
- Start date: November 1, 2024
- Top-n: 5 stocks
- Charts: Show last 3 months of data
- Focus: Only top 5 trending stocks

### Example 3: Year-to-Date, All Trending
```bash
python stock_trend_analyzer.py --start-date 2025-01-01 --top-n 0
```

**Result:**
- Start date: January 1, 2025
- Top-n: All trending stocks (unlimited)
- Charts: Full 2025 data
- Complete: Every trending stock gets a chart

### Example 4: Custom Period, Specific Count
```bash
python stock_trend_analyzer.py \
  --start-date 2024-09-01 \
  --top-n 15 \
  --tickers AAPL,MSFT,GOOGL,NVDA,TSLA,AMD,INTC,META,AMZN,JPM,BAC,WMT,JNJ,PFE,XOM
```

**Result:**
- Start date: September 1, 2024
- Top-n: 15 stocks
- Specific tickers analyzed
- 4-month date range

### Example 5: With Custom Output Name
```bash
python stock_trend_analyzer.py \
  --start-date 2025-01-01 \
  --top-n 10 \
  --output weekly_analysis.csv
```

**Output:**
- `csv/weekly_analysis.csv`
- `plots/weekly_analysis_plot.png`
- `trending_charts/TICKER_technical_analysis.png`

---

## 📋 Complete Command Reference

### All New Arguments
```bash
--start-date YYYY-MM-DD    # Start date for charts (default: 2025-01-01)
--top-n N                  # Number of individual charts (default: 10, 0=all)
```

### Combined with Existing Arguments
```bash
python stock_trend_analyzer.py \
  [--tickers X,Y,Z | --file path] \
  [--start-date YYYY-MM-DD] \
  [--top-n N] \
  [--interval {1d,5min,15min,30min,60min}] \
  [--period {compact,full}] \
  [--output filename.csv] \
  [--log-level LEVEL] \
  [--log-file path]
```

---

## 🔍 Detailed Behavior

### Start Date Filtering

**What's Filtered:**
- ✅ Individual technical charts (Bollinger Bands, Volume, RSI)
- ❌ Dashboard plot (shows all data)
- ❌ Analysis calculations (uses all data)
- ❌ CSV output (includes all data)

**Why:**
- Analysis needs full history for accurate indicators
- Dashboard compares all stocks fairly
- Individual charts focus on recent action

### Top-N Selection

**Selection Criteria:**
1. Must be trending (score ≥ 4.0)
2. Sorted by score (highest first)
3. Take top N from sorted list

**Example:**
```
25 stocks analyzed:
- 8 trending (≥4.0)
- 17 not trending (<4.0)

With --top-n 5:
- Dashboard: Shows all 25
- Individual charts: Top 5 of the 8 trending
```

### Directory Organization

**Creation:**
- Automatic on first run
- Silent if already exist
- No user action needed

**Benefits:**
- Clean workspace
- Easy to find files
- Simple backups
- Clear organization

---

## 💡 Use Cases

### Daily Trading Routine
```bash
# Quick check of top opportunities
python stock_trend_analyzer.py \
  --start-date 2025-01-01 \
  --top-n 5
```

**Result:**
- Last 10 months of data
- Focus on 5 best stocks
- Quick analysis

### Weekly Review
```bash
# Comprehensive weekly analysis
python stock_trend_analyzer.py \
  --start-date 2024-10-01 \
  --top-n 15 \
  --output weekly_$(date +%Y%m%d).csv
```

**Result:**
- 3-month recent data
- Top 15 trending stocks
- Named for weekly review

### Research Deep Dive
```bash
# Full analysis of all trending stocks
python stock_trend_analyzer.py \
  --start-date 2024-01-01 \
  --top-n 0
```

**Result:**
- Full year of data
- All trending stocks
- Complete analysis

### Quick Screening
```bash
# Very fast scan
python stock_trend_analyzer.py \
  --start-date 2025-03-01 \
  --top-n 3 \
  --file shortlist.txt
```

**Result:**
- 2-month data only
- Just top 3 stocks
- Fastest analysis

---

## 📊 Output Structure Examples

### Default Run
```
csv/
  └── trending_stocks_20251104_123045.csv
plots/
  └── trending_stocks_20251104_123045_plot.png
trending_charts/
  ├── AAPL_technical_analysis.png
  ├── MSFT_technical_analysis.png
  ├── GOOGL_technical_analysis.png
  ├── NVDA_technical_analysis.png
  ├── TSLA_technical_analysis.png
  ├── AMD_technical_analysis.png
  ├── INTC_technical_analysis.png
  ├── META_technical_analysis.png
  ├── AMZN_technical_analysis.png
  └── JPM_technical_analysis.png  (10 total)
```

### With Custom Output
```
csv/
  └── tech_stocks.csv
plots/
  └── tech_stocks_plot.png
trending_charts/
  ├── AAPL_technical_analysis.png
  ├── MSFT_technical_analysis.png
  └── ... (top-n stocks)
```

### After Multiple Runs
```
csv/
  ├── trending_stocks_20251104_090000.csv
  ├── trending_stocks_20251104_120000.csv
  └── trending_stocks_20251104_150000.csv
plots/
  ├── trending_stocks_20251104_090000_plot.png
  ├── trending_stocks_20251104_120000_plot.png
  └── trending_stocks_20251104_150000_plot.png
trending_charts/
  ├── AAPL_technical_analysis.png
  ├── MSFT_technical_analysis.png
  └── ... (latest run's charts)
```

**Note:** trending_charts/ is overwritten each run with latest top-n stocks

---

## 🎓 Best Practices

### Start Date Selection

**Recent Action (2-3 months):**
```bash
--start-date 2024-10-01  # Last quarter
```

**Year-to-Date:**
```bash
--start-date 2025-01-01  # Default
```

**Longer Trend (6+ months):**
```bash
--start-date 2024-06-01  # Half year
```

### Top-N Selection

**Quick Screening:**
```bash
--top-n 3  # Just the best
```

**Standard Analysis:**
```bash
--top-n 10  # Default, balanced
```

**Thorough Review:**
```bash
--top-n 20  # More options
```

**Complete Analysis:**
```bash
--top-n 0  # All trending stocks
```

### Directory Management

**Regular Cleanup:**
```bash
# Archive old analyses
mkdir -p archive/$(date +%Y%m)
mv csv/*_202410* archive/202410/
mv plots/*_202410* archive/202410/
```

**Selective Backup:**
```bash
# Backup just CSV data
tar -czf backup_csv_$(date +%Y%m%d).tar.gz csv/
```

**Keep Charts Fresh:**
```bash
# trending_charts/ updates each run
# Old charts automatically replaced
```

---

## ⚠️ Important Notes

### Data Filtering
- Start date only affects **chart visualization**
- Analysis uses **full data history**
- Ensures accurate technical indicators

### Chart Limits
- Top-n applies only to **trending stocks** (≥4.0)
- Non-trending stocks never get individual charts
- Dashboard always shows all analyzed stocks

### Directory Persistence
- Directories created once, persist forever
- CSV and plots accumulate over time
- trending_charts/ overwritten each run

### Performance
- Lower top-n = faster execution
- Recent start-date = faster chart rendering
- Organized directories = easier management

---

## 🚀 Quick Start

### Absolute Simplest
```bash
python stock_trend_analyzer.py
```
Defaults: Start 2025-01-01, top 10, all directories auto-created

### Most Common
```bash
python stock_trend_analyzer.py --top-n 5
```
Quick analysis of 5 best opportunities

### Full Control
```bash
python stock_trend_analyzer.py \
  --start-date 2024-06-01 \
  --top-n 15 \
  --file watchlist.txt
```
Custom everything

---

## 📚 Related Documentation

- **FINAL_CONFIGURATION_SUMMARY.md** - All features overview
- **INDIVIDUAL_CHARTS_GUIDE.md** - Chart details
- **README_AlphaVantage.md** - Complete documentation

---

## ✅ Summary

Six powerful new features:

1. **--start-date** - Filter charts from specific date (default: 2025-01-01)
2. **--top-n** - Limit individual charts (default: 10)
3. **csv/** - Organized data directory
4. **plots/** - Organized dashboard directory
5. **trending_charts/** - Organized individual charts directory
6. **Auto-creation** - All directories created automatically

**Result:** More organized, flexible, and efficient analysis! 🎯📊
