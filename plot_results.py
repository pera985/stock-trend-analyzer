#!/usr/bin/env python3
"""
Stock Trend Analyzer - Visualization Module
Creates plots comparing trending vs non-trending stocks
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Color scheme
COLOR_PASS = '#4CAF50'  # Green for passing stocks
COLOR_FAIL = '#F44336'  # Red for failing stocks
COLOR_THRESHOLD = '#FF9800'  # Orange for threshold line
COLOR_GRID = '#E0E0E0'  # Light gray for grid


def plot_score_comparison(results_df, all_tickers, output_file='stock_comparison.png'):
    """
    Create comprehensive visualization of all stocks analyzed
    
    Args:
        results_df: DataFrame with trending stocks (score >= 4.0)
        all_tickers: List of all tickers analyzed
        output_file: Output filename for the plot
    """
    
    if results_df.empty and not all_tickers:
        print("No data to plot")
        return
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Prepare data - need to know scores for all stocks
    # For now, we'll work with what we have and simulate failed stocks
    passing_stocks = results_df if not results_df.empty else pd.DataFrame()
    
    # Create master list with all stocks
    all_scores = {}
    passing_tickers = set(passing_stocks['ticker'].tolist()) if not passing_stocks.empty else set()
    
    # Add passing stocks with their actual scores
    for _, row in passing_stocks.iterrows():
        all_scores[row['ticker']] = {
            'score': row['score'],
            'passed': True,
            'current_price': row['current_price'],
            'rsi': row['rsi'],
            'adx': row['adx']
        }
    
    # Note: Failed stocks (score < 4.0) aren't in results_df
    # We can only plot stocks that passed the threshold
    
    if len(all_scores) == 0:
        print("No stocks passed the trending threshold to plot")
        return
    
    # Sort by score
    sorted_stocks = sorted(all_scores.items(), key=lambda x: x[1]['score'], reverse=True)

    # Limit main plot to top 20 stocks
    top_n_main = min(20, len(sorted_stocks))
    sorted_stocks_main = sorted_stocks[:top_n_main]

    tickers = [s[0] for s in sorted_stocks_main]
    scores = [s[1]['score'] for s in sorted_stocks_main]
    colors = [COLOR_PASS if s[1]['passed'] else COLOR_FAIL for s in sorted_stocks_main]

    # Keep full list for statistics
    all_tickers_full = [s[0] for s in sorted_stocks]
    all_scores_full = [s[1]['score'] for s in sorted_stocks]

    # 1. MAIN PLOT: Score Bar Chart (Top 20)
    ax1 = fig.add_subplot(gs[0, :])
    bars = ax1.bar(range(len(tickers)), scores, color=colors, edgecolor='black', linewidth=1.5)
    ax1.axhline(y=4.0, color=COLOR_THRESHOLD, linestyle='--', linewidth=2, label='Threshold (4.0)')
    ax1.set_xlabel('Stock Ticker', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Trend Score (out of 6.0)', fontsize=12, fontweight='bold')
    title_suffix = f' (Top {len(tickers)})' if len(sorted_stocks) > 20 else ''
    ax1.set_title(f'Stock Trend Analysis: Score Comparison{title_suffix}', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xticks(range(len(tickers)))
    ax1.set_xticklabels(tickers, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3, color=COLOR_GRID)
    ax1.set_ylim(0, 6.5)
    
    # Add score labels on bars
    for i, (bar, score) in enumerate(zip(bars, scores)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{score:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Legend
    pass_patch = mpatches.Patch(color=COLOR_PASS, label=f'Trending ({len([s for s in scores if s >= 4.0])} stocks)')
    fail_patch = mpatches.Patch(color=COLOR_FAIL, label=f'Not Trending ({len([s for s in scores if s < 4.0])} stocks)')
    threshold_line = mpatches.Patch(color=COLOR_THRESHOLD, label='Threshold (4.0)')
    ax1.legend(handles=[pass_patch, fail_patch, threshold_line], loc='upper right', fontsize=10)
    
    # 2. SCORE DISTRIBUTION (using all stocks, not just top 20)
    ax2 = fig.add_subplot(gs[1, 0])
    score_ranges = ['0-1', '1-2', '2-3', '3-4', '4-5', '5-6']
    score_counts = [
        len([s for s in all_scores_full if 0 <= s < 1]),
        len([s for s in all_scores_full if 1 <= s < 2]),
        len([s for s in all_scores_full if 2 <= s < 3]),
        len([s for s in all_scores_full if 3 <= s < 4]),
        len([s for s in all_scores_full if 4 <= s < 5]),
        len([s for s in all_scores_full if 5 <= s <= 6])
    ]
    colors_dist = [COLOR_FAIL] * 4 + [COLOR_PASS] * 2
    ax2.bar(score_ranges, score_counts, color=colors_dist, edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('Score Range', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Number of Stocks', fontsize=11, fontweight='bold')
    ax2.set_title('Score Distribution', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, color=COLOR_GRID)
    
    # Add count labels
    for i, count in enumerate(score_counts):
        if count > 0:
            ax2.text(i, count + 0.1, str(count), ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 3. TOP PERFORMERS
    ax3 = fig.add_subplot(gs[1, 1])
    top_n = min(10, len(sorted_stocks))
    top_tickers = tickers[:top_n]
    top_scores = scores[:top_n]
    top_colors = colors[:top_n]
    
    y_pos = np.arange(len(top_tickers))
    ax3.barh(y_pos, top_scores, color=top_colors, edgecolor='black', linewidth=1.5)
    ax3.axvline(x=4.0, color=COLOR_THRESHOLD, linestyle='--', linewidth=2, alpha=0.7)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(top_tickers)
    ax3.set_xlabel('Trend Score', fontsize=11, fontweight='bold')
    ax3.set_title(f'Top {top_n} Stocks by Score', fontsize=12, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3, color=COLOR_GRID)
    ax3.set_xlim(0, 6.5)
    
    # Add score labels
    for i, score in enumerate(top_scores):
        ax3.text(score + 0.1, i, f'{score:.1f}', va='center', fontsize=9, fontweight='bold')
    
    # 4. STATISTICS TABLE (using all stocks, not just top 20)
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.axis('off')

    total_stocks = len(all_scores)
    passing_count = len([s for s in all_scores_full if s >= 4.0])
    failing_count = total_stocks - passing_count
    avg_score = np.mean(all_scores_full)
    median_score = np.median(all_scores_full)
    max_score = max(all_scores_full)
    min_score = min(all_scores_full)
    
    stats_text = f"""
    ANALYSIS STATISTICS
    {'='*40}
    
    Total Stocks Analyzed:     {total_stocks}
    Trending (≥4.0):           {passing_count} ({passing_count/total_stocks*100:.1f}%)
    Not Trending (<4.0):       {failing_count} ({failing_count/total_stocks*100:.1f}%)
    
    Average Score:             {avg_score:.2f}
    Median Score:              {median_score:.2f}
    Highest Score:             {max_score:.2f}
    Lowest Score:              {min_score:.2f}
    
    Threshold:                 4.0 out of 6.0
    """
    
    ax4.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', 
            facecolor='wheat', alpha=0.3))
    
    # 5. CRITERIA BREAKDOWN (for passing stocks)
    ax5 = fig.add_subplot(gs[2, 1])
    
    if not passing_stocks.empty:
        criteria = ['MA\nBullish', 'Momentum\nPositive', 'RSI\nFavorable', 
                   'MACD\nBullish', 'ADX\nStrong', 'Volume\nIncreasing']
        criteria_counts = [
            passing_stocks['ma_bullish'].sum(),
            passing_stocks['momentum_positive'].sum(),
            passing_stocks['rsi_favorable'].sum(),
            passing_stocks['macd_bullish'].sum(),
            passing_stocks['adx_strong'].sum(),
            passing_stocks['volume_increasing'].sum()
        ]
        
        bars = ax5.bar(range(len(criteria)), criteria_counts, 
                      color=COLOR_PASS, edgecolor='black', linewidth=1.5)
        ax5.set_xticks(range(len(criteria)))
        ax5.set_xticklabels(criteria, fontsize=9)
        ax5.set_ylabel('Number of Passing Stocks', fontsize=11, fontweight='bold')
        ax5.set_title('Criteria Success Rate (Trending Stocks Only)', fontsize=12, fontweight='bold')
        ax5.grid(axis='y', alpha=0.3, color=COLOR_GRID)
        
        # Add count labels
        for bar, count in zip(bars, criteria_counts):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                    f'{count}\n({count/len(passing_stocks)*100:.0f}%)',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    else:
        ax5.text(0.5, 0.5, 'No trending stocks to analyze', 
                ha='center', va='center', fontsize=12)
        ax5.axis('off')
    
    # Main title
    fig.suptitle('Stock Trend Analysis Dashboard', fontsize=16, fontweight='bold', y=0.995)
    
    # Add timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    fig.text(0.99, 0.01, f'Generated: {timestamp}', ha='right', fontsize=8, style='italic')
    
    # Save
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nVisualization saved to: {output_file}")
    
    return output_file


def create_simple_comparison_plot(passing_tickers, failing_tickers, 
                                  passing_scores, failing_scores,
                                  output_file='stock_comparison_simple.png'):
    """
    Create a simple comparison plot when we have both passing and failing stocks
    
    Args:
        passing_tickers: List of tickers that passed
        failing_tickers: List of tickers that failed
        passing_scores: List of scores for passing tickers
        failing_scores: List of scores for failing tickers
        output_file: Output filename
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Combine all stocks
    all_tickers = passing_tickers + failing_tickers
    all_scores = passing_scores + failing_scores
    colors = [COLOR_PASS] * len(passing_tickers) + [COLOR_FAIL] * len(failing_tickers)
    
    # Sort by score
    sorted_indices = np.argsort(all_scores)[::-1]
    all_tickers = [all_tickers[i] for i in sorted_indices]
    all_scores = [all_scores[i] for i in sorted_indices]
    colors = [colors[i] for i in sorted_indices]
    
    # Plot 1: All stocks
    bars = ax1.bar(range(len(all_tickers)), all_scores, color=colors, 
                   edgecolor='black', linewidth=1.5)
    ax1.axhline(y=4.0, color=COLOR_THRESHOLD, linestyle='--', linewidth=2, 
               label='Threshold (4.0)')
    ax1.set_xlabel('Stock Ticker', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Trend Score (out of 6.0)', fontsize=12, fontweight='bold')
    ax1.set_title('All Stocks: Trending vs Not Trending', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(all_tickers)))
    ax1.set_xticklabels(all_tickers, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 6.5)
    
    # Add score labels
    for bar, score in zip(bars, all_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{score:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Legend
    pass_patch = mpatches.Patch(color=COLOR_PASS, 
                                label=f'Trending ({len(passing_tickers)} stocks)')
    fail_patch = mpatches.Patch(color=COLOR_FAIL, 
                                label=f'Not Trending ({len(failing_tickers)} stocks)')
    ax1.legend(handles=[pass_patch, fail_patch], loc='upper right')
    
    # Plot 2: Side-by-side comparison
    x_pass = np.arange(len(passing_tickers))
    x_fail = np.arange(len(failing_tickers))
    
    ax2_left = ax2
    ax2_right = ax2.twiny()
    
    # Passing stocks on left
    bars_pass = ax2_left.barh(x_pass, passing_scores, color=COLOR_PASS, 
                              edgecolor='black', linewidth=1.5, label='Trending')
    ax2_left.set_yticks(x_pass)
    ax2_left.set_yticklabels(passing_tickers)
    ax2_left.set_xlabel('Trending Stocks →', fontsize=11, fontweight='bold', color=COLOR_PASS)
    ax2_left.tick_params(axis='x', labelcolor=COLOR_PASS)
    ax2_left.invert_xaxis()
    ax2_left.set_xlim(6.5, 0)
    
    # Failing stocks on right
    bars_fail = ax2_right.barh(x_fail, failing_scores, color=COLOR_FAIL, 
                               edgecolor='black', linewidth=1.5, label='Not Trending')
    ax2_right.set_yticks(x_fail)
    ax2_right.set_yticklabels(failing_tickers)
    ax2_right.set_xlabel('← Not Trending Stocks', fontsize=11, fontweight='bold', color=COLOR_FAIL)
    ax2_right.tick_params(axis='x', labelcolor=COLOR_FAIL)
    ax2_right.set_xlim(0, 6.5)
    
    # Threshold line
    ax2_left.axvline(x=4.0, color=COLOR_THRESHOLD, linestyle='--', linewidth=2, alpha=0.7)
    ax2_right.axvline(x=4.0, color=COLOR_THRESHOLD, linestyle='--', linewidth=2, alpha=0.7)
    
    ax2.set_title('Pass/Fail Comparison', fontsize=14, fontweight='bold', pad=40)
    ax2.grid(axis='x', alpha=0.3)
    
    plt.suptitle('Stock Trend Analysis: Pass vs Fail', fontsize=16, fontweight='bold')
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    fig.text(0.99, 0.01, f'Generated: {timestamp}', ha='right', fontsize=8, style='italic')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nVisualization saved to: {output_file}")
    
    return output_file


def main():
    """Main function for standalone usage"""
    if len(sys.argv) < 2:
        print("Usage: python plot_results.py <results_csv_file>")
        print("Example: python plot_results.py trending_stocks_20251104_123045.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    if not os.path.exists(csv_file):
        print(f"Error: File '{csv_file}' not found")
        sys.exit(1)
    
    # Read results
    try:
        results_df = pd.read_csv(csv_file)
        print(f"Loaded {len(results_df)} stocks from {csv_file}")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)
    
    # Get all tickers (in this case, we only have passing stocks in the CSV)
    all_tickers = results_df['ticker'].tolist()
    
    # Generate output filename
    base_name = os.path.splitext(csv_file)[0]
    output_file = f"{base_name}_plot.png"
    
    # Create plot
    plot_score_comparison(results_df, all_tickers, output_file)
    
    print(f"\n✓ Visualization complete!")
    print(f"  Input:  {csv_file}")
    print(f"  Output: {output_file}")


if __name__ == "__main__":
    main()
