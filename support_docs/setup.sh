#!/bin/bash

# Stock Trend Analyzer - Setup Script
# This script helps you get started quickly

echo "=========================================="
echo "Stock Trend Analyzer - Alpha Vantage"
echo "Setup Script"
echo "=========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.7 or higher."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"
echo ""

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip3."
    exit 1
fi

echo "✅ pip3 found"
echo ""

# Install requirements
echo "Installing required packages..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo ""
echo "=========================================="
echo "API Key Setup"
echo "=========================================="
echo ""

# Check if API key is already set
if [ -n "$ALPHAVANTAGE_API_KEY" ]; then
    echo "✅ API key already set in environment"
    echo "   Key: ${ALPHAVANTAGE_API_KEY:0:8}..."
else
    echo "⚠️  API key not found in environment"
    echo ""
    read -p "Would you like to set it now? (y/n): " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        read -p "Enter your Alpha Vantage API key: " api_key
        
        # Add to current session
        export ALPHAVANTAGE_API_KEY="$api_key"
        
        # Determine shell config file
        if [ -f "$HOME/.bashrc" ]; then
            config_file="$HOME/.bashrc"
        elif [ -f "$HOME/.zshrc" ]; then
            config_file="$HOME/.zshrc"
        else
            config_file="$HOME/.profile"
        fi
        
        # Ask if user wants to save permanently
        echo ""
        read -p "Save to $config_file for future sessions? (y/n): " -n 1 -r
        echo ""
        
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "" >> "$config_file"
            echo "# Alpha Vantage API Key" >> "$config_file"
            echo "export ALPHAVANTAGE_API_KEY=\"$api_key\"" >> "$config_file"
            echo "✅ API key saved to $config_file"
            echo "   Run: source $config_file"
        else
            echo "ℹ️  API key set for current session only"
        fi
    else
        echo "ℹ️  You can set it later with:"
        echo "   export ALPHAVANTAGE_API_KEY=your_key_here"
    fi
fi

echo ""
echo "=========================================="
echo "Quick Test"
echo "=========================================="
echo ""

if [ -n "$ALPHAVANTAGE_API_KEY" ]; then
    read -p "Run a quick test with AAPL and MSFT? (y/n): " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo ""
        echo "Running test analysis..."
        echo ""
        python3 stock_trend_analyzer.py --tickers AAPL,MSFT
    fi
else
    echo "⚠️  Skipping test (no API key set)"
fi

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "📚 Documentation:"
echo "   • START_HERE.md - Overview and file directory"
echo "   • QUICKSTART_AlphaVantage.md - Get started in 3 steps"
echo "   • README_AlphaVantage.md - Full documentation"
echo "   • API_SETUP_GUIDE.md - API key configuration"
echo ""
echo "🚀 Quick Commands:"
echo "   # Daily analysis"
echo "   python3 stock_trend_analyzer.py --tickers AAPL,MSFT,GOOGL"
echo ""
echo "   # 5-minute intraday"
echo "   python3 stock_trend_analyzer.py --tickers SPY,QQQ --interval 5min"
echo ""
echo "   # From file"
echo "   python3 stock_trend_analyzer.py --file sample_tickers.txt"
echo ""
echo "   # Full market scan"
echo "   python3 stock_trend_analyzer.py --scan-all"
echo ""
echo "Happy trading! 📈"
