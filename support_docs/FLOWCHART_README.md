# Flowchart Generation Script

## Overview

The `create_flowchart.py` script generates a comprehensive visual flowchart of the Stock Trend Analyzer's workflow using matplotlib.

## Usage

### Basic Usage

```bash
python3 create_flowchart.py
```

This will generate `stock_analyzer_flowchart.png` in the current directory.

### Requirements

```bash
pip install matplotlib
```

(Already included if you installed from `requirements.txt`)

## Customization

### Change Output Location

Edit line 285 in the script:

```python
plt.savefig('/path/to/your/output.png', 
           dpi=300, bbox_inches='tight', facecolor='white')
```

### Change Resolution

Modify the `dpi` parameter:
- `dpi=150` - Lower resolution (smaller file)
- `dpi=300` - High resolution (default, good for printing)
- `dpi=600` - Very high resolution (large file)

### Change Image Size

Edit line 14 to adjust figure dimensions:

```python
fig, ax = plt.subplots(1, 1, figsize=(14, 20))  # (width, height) in inches
```

### Modify Colors

The color scheme is defined at the top of the script:

```python
color_start = '#4CAF50'     # Green - Start/End
color_process = '#2196F3'   # Blue - Process steps
color_decision = '#FF9800'  # Orange - Decisions
color_data = '#9C27B0'      # Purple - Data operations
color_output = '#4CAF50'    # Green - Output
color_criteria = '#00BCD4'  # Cyan - Analysis criteria
```

Change these hex color codes to customize the appearance.

### Add or Modify Boxes

The script uses helper functions:

#### draw_box()
```python
draw_box(ax, x, y, width, height, text, color, style='round')
```

Parameters:
- `ax` - matplotlib axis object
- `x, y` - position (0-10 for x, 0-28 for y)
- `width, height` - box dimensions
- `text` - text to display
- `color` - hex color code
- `style` - 'round', 'diamond', or default square

Example:
```python
draw_box(ax, 2, 10, 6, 0.8, 'My New Step', '#FF5722')
```

#### draw_arrow()
```python
draw_arrow(ax, x1, y1, x2, y2, label='')
```

Parameters:
- `x1, y1` - start coordinates
- `x2, y2` - end coordinates
- `label` - optional text label

Example:
```python
draw_arrow(ax, 5, 10, 5, 9, 'Continue')
```

## Flowchart Structure

The flowchart is organized vertically with these sections:

1. **Header** (y: 27-26)
   - Title and subtitle

2. **Initialization** (y: 26-23)
   - Input parameters
   - Client setup
   - Rate limiter

3. **Main Loop** (y: 23-21)
   - Data fetching (daily vs intraday)
   - Data validation

4. **Analysis Section** (y: 21-17)
   - 4 criteria evaluation
   - Scoring system

5. **Decision & Results** (y: 17-16)
   - Score threshold check
   - Filter results

6. **Output** (y: 16-14)
   - Sort and display
   - CSV export

7. **Footer** (y: 2-0)
   - Legend
   - Watermark

## Tips for Modification

### Adding a New Analysis Step

1. Choose a y-position in the analysis section (around y: 17-19)
2. Add the box:
   ```python
   y_pos = 18
   draw_box(ax, 0.5, y_pos, 4, 0.7,
            '5. NEW CRITERION\nDescription here',
            color_criteria)
   ```
3. Add a score box if needed:
   ```python
   draw_box(ax, 5, y_pos, 1.2, 0.7,
            'Score\n+0.5',
            '#FFD700')
   ```
4. Connect with arrows

### Adjusting Layout

The coordinate system:
- X-axis: 0 to 10 (left to right)
- Y-axis: 0 to 28 (bottom to top)

Center alignment is at x=5.

### Changing Font Sizes

Global font sizes are set in the `draw_box()` function:
- Box labels: `fontsize=10`
- Diamond labels: `fontsize=9`
- Arrow labels: `fontsize=8`

Edit these values in the function definition.

## Output Formats

### PNG (Default)
```python
plt.savefig('output.png', dpi=300, bbox_inches='tight', facecolor='white')
```

### PDF (Vector)
```python
plt.savefig('output.pdf', bbox_inches='tight', facecolor='white')
```

### SVG (Scalable)
```python
plt.savefig('output.svg', bbox_inches='tight', facecolor='white')
```

### JPG
```python
plt.savefig('output.jpg', dpi=300, bbox_inches='tight', facecolor='white', quality=95)
```

## Troubleshooting

### "matplotlib not found"
```bash
pip install matplotlib
```

### Boxes overlapping
- Adjust y-positions
- Increase spacing between sections
- Reduce box heights

### Text too small/large
- Modify `fontsize` parameters in draw_box()
- Adjust `figsize` to make overall figure larger/smaller

### Colors not displaying correctly
- Ensure hex codes are valid
- Check color contrast for readability
- Use web-safe colors for compatibility

## Examples

### Create a Simplified Version
```python
# Minimal flowchart with just main steps
fig, ax = plt.subplots(1, 1, figsize=(10, 12))
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)

draw_box(ax, 3.5, 13, 3, 0.6, 'START', color_start)
draw_arrow(ax, 5, 13, 5, 12)
draw_box(ax, 2, 11.5, 6, 0.8, 'Fetch Data', color_data)
draw_arrow(ax, 5, 11.5, 5, 10.5)
draw_box(ax, 2, 10, 6, 0.8, 'Analyze', color_criteria)
draw_arrow(ax, 5, 10, 5, 9)
draw_box(ax, 2, 8.5, 6, 0.8, 'Output Results', color_output)
draw_arrow(ax, 5, 8.5, 5, 7.5)
draw_box(ax, 3.5, 7, 3, 0.6, 'END', color_start)

plt.savefig('simple_flowchart.png', dpi=300, bbox_inches='tight')
```

### Add a Timestamp
```python
from datetime import datetime

timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
ax.text(5, 0.2, f'Generated: {timestamp}', 
       ha='center', fontsize=8, style='italic')
```

### Add a Logo or Watermark
```python
from PIL import Image

# Add logo (requires logo.png file)
logo = Image.open('logo.png')
imagebox = OffsetImage(logo, zoom=0.1)
ab = AnnotationBbox(imagebox, (9, 27), frameon=False)
ax.add_artist(ab)
```

## Advanced Customization

### Gradient Backgrounds
```python
from matplotlib.colors import LinearSegmentedColormap

# Create gradient for a box
gradient = LinearSegmentedColormap.from_list('custom', ['#2196F3', '#64B5F6'])
```

### Curved Arrows
```python
arrow = FancyArrowPatch((x1, y1), (x2, y2),
                       connectionstyle="arc3,rad=.3",
                       arrowstyle='->', mutation_scale=20,
                       linewidth=2, color='black')
```

### Shadow Effects
```python
from matplotlib.patheffects import withSimplePatchShadow

box = FancyBboxPatch((x, y), width, height, ...)
box.set_path_effects([withSimplePatchShadow()])
```

## License

This script is part of the Stock Trend Analyzer package and follows the same MIT License.

## Support

For questions or issues with the flowchart script:
1. Check that matplotlib is properly installed
2. Verify Python version (3.7+)
3. Review the troubleshooting section above
4. Modify the script as needed for your use case
