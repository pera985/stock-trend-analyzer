#!/usr/bin/env python3
"""
Generate flowchart for Stock Trend Analyzer
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines

# Create figure
fig, ax = plt.subplots(1, 1, figsize=(14, 32))
ax.set_xlim(0, 10)
ax.set_ylim(0, 42)
ax.axis('off')

# Color scheme
color_start = '#4CAF50'  # Green
color_process = '#2196F3'  # Blue
color_decision = '#FF9800'  # Orange
color_data = '#9C27B0'  # Purple
color_output = '#4CAF50'  # Green
color_criteria = '#00BCD4'  # Cyan

def draw_box(ax, x, y, width, height, text, color, style='round'):
    """Draw a box with text"""
    if style == 'round':
        box = FancyBboxPatch((x, y), width, height,
                            boxstyle="round,pad=0.1",
                            facecolor=color, edgecolor='black', linewidth=2)
    elif style == 'diamond':
        # Draw diamond for decision
        points = [(x + width/2, y), (x + width, y + height/2), 
                 (x + width/2, y + height), (x, y + height/2)]
        box = patches.Polygon(points, facecolor=color, edgecolor='black', linewidth=2)
    else:
        box = FancyBboxPatch((x, y), width, height,
                            facecolor=color, edgecolor='black', linewidth=2)
    
    ax.add_patch(box)
    
    # Add text
    if style == 'diamond':
        ax.text(x + width/2, y + height/2, text, ha='center', va='center',
               fontsize=9, fontweight='bold', wrap=True)
    else:
        ax.text(x + width/2, y + height/2, text, ha='center', va='center',
               fontsize=10, fontweight='bold', wrap=True)

def draw_arrow(ax, x1, y1, x2, y2, label=''):
    """Draw an arrow between two points"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                          arrowstyle='->', mutation_scale=20, linewidth=2,
                          color='black')
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.3, mid_y, label, fontsize=8, style='italic',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Title
ax.text(5, 41.5, 'Stock Trend Analyzer Flowchart', 
       ha='center', fontsize=18, fontweight='bold')
ax.text(5, 41, 'Alpha Vantage Premium Edition (150 req/min)', 
       ha='center', fontsize=12, style='italic')

# START
y_pos = 40
draw_box(ax, 3.5, y_pos, 3, 0.6, 'START', color_start, 'round')
draw_arrow(ax, 5, y_pos, 5, y_pos - 0.8)

# Input
y_pos -= 1.8
draw_box(ax, 2, y_pos, 6, 0.8, 
         'INPUT\nAPI Key + Tickers + Interval + Period', 
         color_process)
draw_arrow(ax, 5, y_pos, 5, y_pos - 0.8)

# Initialize
y_pos -= 2.0
draw_box(ax, 2, y_pos, 6, 0.8,
         'Initialize AlphaVantageClient\n& RateLimiter (150 req/min)',
         color_process)
draw_arrow(ax, 5, y_pos, 5, y_pos - 0.8)

# Loop start
y_pos -= 2.0
draw_box(ax, 2, y_pos, 6, 0.6,
         'FOR EACH Stock in Tickers',
         color_process)
draw_arrow(ax, 5, y_pos, 5, y_pos - 0.8)

# Rate limiter check
y_pos -= 1.8
draw_box(ax, 2.5, y_pos, 5, 0.7,
         'Rate Limiter: Check API Limits',
         color_data, 'round')
draw_arrow(ax, 5, y_pos, 5, y_pos - 0.8)

# Fetch data decision
y_pos -= 1.8
draw_box(ax, 3, y_pos, 4, 0.8,
         'Interval\n= Daily?',
         color_decision, 'diamond')

# Daily branch
draw_arrow(ax, 3, y_pos + 0.4, 1.5, y_pos + 0.4, 'Yes')
draw_box(ax, 0.2, y_pos - 0.8, 2.5, 0.7,
         'Fetch Daily Data\n(20+ years)',
         color_data)
draw_arrow(ax, 1.45, y_pos - 0.8, 3, y_pos - 0.3)

# Intraday branch  
draw_arrow(ax, 7, y_pos + 0.4, 8.5, y_pos + 0.4, 'No')
draw_box(ax, 7.3, y_pos - 0.8, 2.5, 0.7,
         'Fetch Intraday Data\n(1/5/15/30/60 min)',
         color_data)
draw_arrow(ax, 8.55, y_pos - 0.8, 7, y_pos - 0.3)

# Continue
y_pos -= 2.0
draw_arrow(ax, 5, y_pos + 0.8, 5, y_pos + 0.3)

# Data validation
draw_box(ax, 3, y_pos, 4, 0.8,
         'Data Valid?\n(≥50 points)',
         color_decision, 'diamond')

# Invalid branch
draw_arrow(ax, 7, y_pos + 0.4, 8.8, y_pos + 0.4, 'No')
draw_box(ax, 8, y_pos - 0.2, 1.5, 0.6,
         'Skip Stock',
         '#F44336')
draw_arrow(ax, 8.75, y_pos - 0.2, 8.75, 10, '')
ax.text(9.2, 15, 'Continue', fontsize=8, rotation=90, va='center')

# Valid - continue
draw_arrow(ax, 5, y_pos, 5, y_pos - 0.8, 'Yes')

# Analysis section header
y_pos -= 2.0
draw_box(ax, 1.5, y_pos, 7, 0.5,
         '━━━━━━ TECHNICAL ANALYSIS (4 Criteria) ━━━━━━',
         '#E0E0E0', 'round')
draw_arrow(ax, 5, y_pos, 5, y_pos - 0.5)

# Criterion 1: Moving Averages
y_pos -= 1.5
draw_box(ax, 0.5, y_pos, 4, 0.7,
         '1. MOVING AVERAGES\nSMA 50, 200\nPrice > 50-day > 200-day?',
         color_criteria)
draw_box(ax, 5, y_pos, 1.2, 0.7,
         'Score\n+1.5',
         '#FFD700')

# Criterion 2: Momentum
y_pos -= 1.2
draw_box(ax, 0.5, y_pos, 4, 0.7,
         '2. MOMENTUM\n5-day, 10-day, 30-day\nAll Positive?',
         color_criteria)
draw_box(ax, 5, y_pos, 1.2, 0.7,
         'Score\n+1.5',
         '#FFD700')

# Criterion 3: Technical Indicators
y_pos -= 1.2
draw_box(ax, 0.5, y_pos, 4, 0.8,
         '3. INDICATORS\nRSI (50-70?) +1.0\nMACD Bullish? +1.0\nADX > 25? +0.5',
         color_criteria)
draw_box(ax, 5, y_pos, 1.2, 0.8,
         'Score\n+0-2.5',
         '#FFD700')

# Criterion 4: Volume
y_pos -= 1.3
draw_box(ax, 0.5, y_pos, 4, 0.7,
         '4. VOLUME\nRecent > Previous?\nIncreasing trend?',
         color_criteria)
draw_box(ax, 5, y_pos, 1.2, 0.7,
         'Score\n+0.5',
         '#FFD700')

# Aggregate scores
y_pos -= 1.5
draw_arrow(ax, 5, y_pos + 1.1, 5, y_pos + 0.3)
draw_box(ax, 2, y_pos, 6, 0.6,
         'Calculate Total Score (0-6 points)',
         color_process)
draw_arrow(ax, 5, y_pos, 5, y_pos - 0.6)

# Decision: Is trending?
y_pos -= 1.5
draw_box(ax, 3, y_pos, 4, 0.8,
         'Score ≥ 4.0?',
         color_decision, 'diamond')

# Not trending
draw_arrow(ax, 3, y_pos + 0.4, 1.2, y_pos + 0.4, 'No')
draw_box(ax, 0.2, y_pos - 0.1, 1.8, 0.5,
         'Discard',
         '#F44336')
draw_arrow(ax, 1.1, y_pos - 0.1, 1.1, 10, '')

# Trending - add to results
draw_arrow(ax, 7, y_pos + 0.4, 8.5, y_pos + 0.4, 'Yes')
draw_box(ax, 7.5, y_pos - 0.2, 2, 0.6,
         'Add to Results',
         color_output)
draw_arrow(ax, 8.5, y_pos - 0.2, 8.5, 10, '')

# Loop continuation point
y_pos = 10
draw_box(ax, 3.5, y_pos, 3, 0.5,
         'More Stocks?',
         color_decision, 'diamond')

# Yes - loop back
draw_arrow(ax, 3.5, y_pos + 0.25, 0.3, y_pos + 0.25, 'Yes')
ax.plot([0.3, 0.3, 5, 5], [y_pos + 0.25, 34, 34, 34], 
        'k-', linewidth=2)
arrow = FancyArrowPatch((5, 34), (5, 33.9),
                      arrowstyle='->', mutation_scale=20, linewidth=2, color='black')
ax.add_patch(arrow)

# No - continue to output
draw_arrow(ax, 5, y_pos, 5, y_pos - 0.6, 'No')

# Sort results
y_pos -= 1.5
draw_box(ax, 2.5, y_pos, 5, 0.6,
         'Sort Results by Score (Descending)',
         color_process)
draw_arrow(ax, 5, y_pos, 5, y_pos - 0.6)

# Display results
y_pos -= 1.5
draw_box(ax, 2, y_pos, 6, 0.7,
         'Display Results\n(Console + DataFrame)',
         color_output)
draw_arrow(ax, 5, y_pos, 5, y_pos - 0.6)

# Save to CSV
y_pos -= 1.5
draw_box(ax, 2, y_pos, 6, 0.7,
         'Save to CSV\n(trending_stocks_timestamp.csv)',
         color_output)
draw_arrow(ax, 5, y_pos, 5, y_pos - 0.6)

# END
y_pos -= 1.5
draw_box(ax, 3.5, y_pos, 3, 0.6,
         'END',
         color_start, 'round')

# Legend
legend_y = 1
legend_elements = [
    patches.Patch(facecolor=color_start, edgecolor='black', label='Start/End'),
    patches.Patch(facecolor=color_process, edgecolor='black', label='Process'),
    patches.Patch(facecolor=color_decision, edgecolor='black', label='Decision'),
    patches.Patch(facecolor=color_data, edgecolor='black', label='Data Fetch'),
    patches.Patch(facecolor=color_criteria, edgecolor='black', label='Analysis Criteria'),
    patches.Patch(facecolor=color_output, edgecolor='black', label='Output')
]
ax.legend(handles=legend_elements, loc='lower center', ncol=3, 
         frameon=True, fontsize=9, bbox_to_anchor=(0.5, -0.01))

# Add watermark
ax.text(5, 0.3, 'Alpha Vantage Premium - 150 req/min', 
       ha='center', fontsize=10, style='italic', alpha=0.6)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/stock_analyzer_flowchart.png', 
           dpi=300, bbox_inches='tight', facecolor='white')
print("Flowchart saved to: stock_analyzer_flowchart.png")
