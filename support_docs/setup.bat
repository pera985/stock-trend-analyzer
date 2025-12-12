@echo off
REM Stock Trend Analyzer - Windows Setup Script

echo ==========================================
echo Stock Trend Analyzer - Alpha Vantage
echo Setup Script (Windows)
echo ==========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo X Python is not installed. Please install Python 3.7 or higher.
    pause
    exit /b 1
)

echo Python found
python --version
echo.

REM Install requirements
echo Installing required packages...
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo X Failed to install dependencies
    pause
    exit /b 1
)

echo Dependencies installed successfully
echo.

echo ==========================================
echo API Key Setup
echo ==========================================
echo.

REM Check if API key is set
if defined ALPHAVANTAGE_API_KEY (
    echo API key already set in environment
) else (
    echo API key not found in environment
    echo.
    set /p "setup_key=Would you like to set it now? (y/n): "
    
    if /i "%setup_key%"=="y" (
        set /p "api_key=Enter your Alpha Vantage API key: "
        
        REM Set for current session
        set ALPHAVANTAGE_API_KEY=%api_key%
        
        echo.
        set /p "save_key=Save permanently? (y/n): "
        
        if /i "%save_key%"=="y" (
            setx ALPHAVANTAGE_API_KEY "%api_key%"
            echo API key saved permanently
            echo Please restart your command prompt for it to take effect
        ) else (
            echo API key set for current session only
        )
    ) else (
        echo You can set it later with:
        echo   set ALPHAVANTAGE_API_KEY=your_key_here
        echo   setx ALPHAVANTAGE_API_KEY your_key_here  ^(permanent^)
    )
)

echo.
echo ==========================================
echo Quick Test
echo ==========================================
echo.

if defined ALPHAVANTAGE_API_KEY (
    set /p "run_test=Run a quick test with AAPL and MSFT? (y/n): "
    
    if /i "%run_test%"=="y" (
        echo.
        echo Running test analysis...
        echo.
        python stock_trend_analyzer.py --tickers AAPL,MSFT
    )
) else (
    echo Skipping test ^(no API key set^)
)

echo.
echo ==========================================
echo Setup Complete!
echo ==========================================
echo.
echo Documentation:
echo   * START_HERE.md - Overview and file directory
echo   * QUICKSTART_AlphaVantage.md - Get started in 3 steps
echo   * README_AlphaVantage.md - Full documentation
echo   * API_SETUP_GUIDE.md - API key configuration
echo.
echo Quick Commands:
echo   REM Daily analysis
echo   python stock_trend_analyzer.py --tickers AAPL,MSFT,GOOGL
echo.
echo   REM 5-minute intraday
echo   python stock_trend_analyzer.py --tickers SPY,QQQ --interval 5min
echo.
echo   REM From file
echo   python stock_trend_analyzer.py --file sample_tickers.txt
echo.
echo   REM Full market scan
echo   python stock_trend_analyzer.py --scan-all
echo.
echo Happy trading!
echo.
pause
