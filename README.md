# Stair Wear Analysis Tool

A Python program to analyze 3D models (.obj files) of worn stairs to determine historical usage patterns based on the mathematical model from ModelG.md.

## Overview

This tool answers three key archaeological questions:
1. **How often were the stairs used?** (Traffic volume estimation)
2. **Was a certain direction of travel favored?** (Ascent vs descent analysis)
3. **How many people used the stairs simultaneously?** (Single file vs side-by-side)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python stair_analyzer.py path/to/stair_model.obj --material granite
```

### With Output Directory

```bash
python stair_analyzer.py stairs.obj --material marble --output results/
```

### Specify Stair Width

```bash
python stair_analyzer.py stairs.obj --material sandstone --width 1.5 --output results/
```

## Command Line Arguments

- `obj_file` (required): Path to .obj file containing 3D model of stairs
- `--material`: Material type (choices: granite, marble, sandstone, limestone, oak)
  - Default: granite
- `--width`: Stair width in meters (auto-detected if not specified)
- `--output` or `-o`: Output directory for results and visualizations

## Material Types

The tool supports different materials with varying wear rates:

| Material | Wear Rate | Best For |
|----------|-----------|----------|
| Granite | Very Low | Ancient structures, high-traffic areas |
| Marble | Medium | Classical architecture, moderate wear |
| Sandstone | High | Medieval structures, rapid wear detection |
| Limestone | Medium-High | Churches, temples, polishes easily |
| Oak | High | Wooden stairs, anisotropic wear |

## Output

The tool generates:

1. **Console Report**: Detailed analysis printed to terminal
2. **Text Report**: `report.txt` with summary findings
3. **Visualizations** (if output directory specified):
   - `wear_heatmap.png`: 2D heatmap of wear depth
   - `lateral_distribution.png`: Cross-sectional wear pattern
   - `longitudinal_profile.png`: Front-to-back wear profile
   - `summary_dashboard.png`: Combined visualization with all findings

## Example Output

```
STAIR WEAR ANALYSIS REPORT
======================================================================

--- QUESTION 1: HOW OFTEN WERE THE STAIRS USED? ---
Total estimated footsteps: 1.4e+08
Estimated daily traffic (500 years): 760 people/day

--- QUESTION 2: WAS A DIRECTION FAVORED? ---
Dominant direction: DESCENT
Confidence: 85%
Explanation: High nosing wear indicates heavy descent traffic

--- QUESTION 3: HOW MANY PEOPLE SIMULTANEOUSLY? ---
Usage pattern: BIDIRECTIONAL LANES
Number of lanes: 2
Description: Two distinct wear tracks indicate simultaneous bidirectional traffic
```

## How It Works

The tool implements the mathematical model from ModelG.md:

1. **Archard's Law**: Calculates traffic volume from wear volume
2. **Gaussian Mixture Models**: Detects lane formation patterns
3. **Nosing Wear Analysis**: Determines directional preferences
4. **Statistical Moments**: Analyzes kurtosis and skewness for traffic patterns

## Requirements

- Python 3.7+
- NumPy
- SciPy
- scikit-learn
- Matplotlib

## Limitations

- Requires non-destructive 3D scanning (photogrammetry, laser scanning)
- Assumes uniform material properties
- Age estimates require historical context
- Best results with clear wear patterns (>5mm depth)

## References

Based on "Mathematical Modelling of Staircase Wear: A Tribological and Pedestrian Dynamics Approach to Archaeological Reconstruction" (ModelG.md)
