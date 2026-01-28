"""
Wear Pattern Visualizer
Creates plots and visualizations of stair wear analysis results
Emphasizes stable conclusions (geometry, simultaneity, directionality)
and quarantines calibration-dependent traffic estimates.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path


class WearVisualizer:
    """Creates visualizations of wear analysis results"""

    def __init__(self, results, mesh):
        """
        Initialize visualizer

        Args:
            results: Dictionary containing all analysis results
            mesh: Original mesh data
        """
        self.results = results
        self.mesh = mesh

    def _draw_wear_heatmap(self, ax, include_colorbar=True, include_info_box=True):
        """Draw wear heatmap onto provided axis (internal helper)"""
        wear_profile = self.results['wear_profile']
        depth_grid = wear_profile['depth_grid']
        x_coords = wear_profile['x_coords']
        y_coords = wear_profile['y_coords']

        # Create heatmap
        im = ax.imshow(depth_grid, cmap='hot', aspect='auto',
                      extent=[x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]],
                      origin='lower')

        ax.set_xlabel('Width (m)')
        ax.set_ylabel('Depth (m)')
        ax.set_title('Wear Depth Heatmap (Geometry)')

        # Add colorbar if requested
        cbar = None
        if include_colorbar:
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Wear Depth (mm)')

        # Build info box if requested
        if include_info_box:
            method = wear_profile.get('method', 'legacy_grid')
            max_depth = wear_profile.get('max_depth', 0)
            volume = wear_profile.get('volume', 0)

            info_lines = [
                f"Method: {method}",
                f"Max depth: {max_depth:.1f} mm",
                f"Volume: {volume:.3g} m³"  # 3 sig figs
            ]

            # Add volume in liters for readability
            volume_liters = volume * 1000  # m³ to liters
            info_lines.append(f"Volume: {volume_liters:.1f} L")  # 1 decimal

            # Add RANSAC diagnostics if present
            ref_quality = wear_profile.get('ref_plane_quality', {})
            if ref_quality:
                inlier_frac = ref_quality.get('inlier_fraction', None)
                rms_error = ref_quality.get('rms_error_mm', None)
                if inlier_frac is not None:
                    info_lines.append(f"RANSAC inliers: {inlier_frac:.1%}")
                    # Warn if weak plane fit
                    if inlier_frac < 0.5:
                        info_lines.append("⚠ weak plane fit")
                if rms_error is not None:
                    info_lines.append(f"RMS error: {rms_error:.2f} mm")

            # Add segmentation diagnostics if present
            tread_seg = wear_profile.get('tread_seg', {})
            if tread_seg:
                excluded_frac = tread_seg.get('excluded_fraction', None)
                if excluded_frac is not None:
                    info_lines.append(f"Excluded: {excluded_frac:.1%}")

            info_text = '\n'.join(info_lines)
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                    fontsize=8, verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        return im, cbar

    def plot_wear_heatmap(self, output_file=None):
        """Plot 2D heatmap of wear depth with geometry info box"""
        fig, ax = plt.subplots(figsize=(10, 6))
        self._draw_wear_heatmap(ax, include_colorbar=True, include_info_box=True)
        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

        return fig, ax

    def _draw_lateral_distribution(self, ax, include_info_box=True, include_stat_note=True):
        """Draw lateral distribution onto provided axis (internal helper)"""
        lateral = self.results['lateral']

        if 'lateral_profile' not in lateral:
            return None

        x_coords = lateral['x_coords']
        lateral_profile = lateral['lateral_profile']

        # Plot observed profile
        ax.plot(x_coords, lateral_profile, 'b-', linewidth=2, label='Observed Wear')
        ax.fill_between(x_coords, 0, lateral_profile, alpha=0.3)

        # Get simultaneity data for labeling and notes
        sim = self.results.get('simultaneity', {})
        M_max = sim.get('M_max_two_abreast', 0)
        pattern_str = sim.get('pattern', '')

        # Plot GMM components if available
        # ALWAYS use "GMM Component" labels (never "Lane") for semantic honesty
        if 'means' in lateral and 'stds' in lateral:
            means = lateral['means']
            stds = lateral['stds']
            weights = lateral['weights']

            # Plot each Gaussian component
            x_fine = np.linspace(x_coords[0], x_coords[-1], 200)
            for i, (mean, std, weight) in enumerate(zip(means, stds, weights)):
                gaussian = weight * np.exp(-0.5 * ((x_fine - mean) / std) ** 2)
                gaussian = gaussian / np.max(gaussian) * np.max(lateral_profile)

                # Always use GMM Component labels (statistical interpretation)
                label = f'GMM Component {i+1}'
                ax.plot(x_fine, gaussian, '--', label=label)

        ax.set_xlabel('Lateral Position (m)')
        ax.set_ylabel('Wear Intensity (normalized)')

        # Title uses final classification from TrafficEstimator
        pattern = sim.get('pattern', 'Unknown')
        ax.set_title(f'Lateral Wear (Simultaneity Evidence): {pattern}')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

        # Add semantic note for SINGLE FILE patterns
        if include_stat_note and 'SINGLE FILE' in pattern_str:
            ax.text(0.5, 0.97, "GMM components are statistical modes, not physical lanes.",
                    transform=ax.transAxes, fontsize=8, ha='center', va='top',
                    style='italic', color='gray')

        # Build comprehensive simultaneity info box (reduced precision)
        if include_info_box:
            info_lines = [
                f"Final Pattern: {pattern}",
                f"W_nom: {sim.get('W_nom', 0):.2f}m  W_eff: {sim.get('W_eff', 0):.2f}m",
                f"shy_dist: {sim.get('shy_distance', 0.2):.2f}m  w_lane: {sim.get('w_lane', 0.55):.2f}m",
                f"M_max_two_abreast: {M_max}",
                f"Single file possible: {sim.get('single_file_possible', True)}"
            ]

            # Add temporal drift warning if detected
            if sim.get('temporal_drift_likely', False):
                info_lines.append("⚠ Temporal drift: YES")

            info_text = '\n'.join(info_lines)
            ax.text(0.98, 0.02, info_text, transform=ax.transAxes,
                    fontsize=8, verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def plot_lateral_distribution(self, output_file=None):
        """Plot lateral wear distribution with GMM fit (Simultaneity Evidence)"""
        lateral = self.results['lateral']

        if 'lateral_profile' not in lateral:
            print("No lateral profile data available")
            return None, None

        fig, ax = plt.subplots(figsize=(10, 6))
        self._draw_lateral_distribution(ax, include_info_box=True, include_stat_note=True)
        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

        return fig, ax

    def _draw_longitudinal_profile(self, ax, include_info_box=True):
        """Draw longitudinal profile onto provided axis (internal helper)"""
        longitudinal = self.results['longitudinal']

        if 'longitudinal_profile' not in longitudinal:
            return None

        y_coords = longitudinal['y_coords']
        profile = longitudinal['longitudinal_profile']

        # Plot profile
        ax.plot(y_coords, profile, 'g-', linewidth=2)
        ax.fill_between(y_coords, 0, profile, alpha=0.3, color='green')

        # Mark regions
        n = len(y_coords)
        ax.axvline(y_coords[n//4], color='r', linestyle='--', alpha=0.5, label='Front/Center')
        ax.axvline(y_coords[3*n//4], color='r', linestyle='--', alpha=0.5, label='Center/Back')

        # Get directionality data
        direction = self.results.get('directionality', {})
        nosing_ratio = longitudinal.get('nosing_wear_ratio', 1.0)
        dominant_dir = direction.get('dominant_direction', 'Unknown')
        confidence = direction.get('confidence', 0)

        # Build info box with heuristic annotation and caveat
        if include_info_box:
            info_lines = [
                f"Nosing Ratio: {nosing_ratio:.2f}",
                f"Direction: {dominant_dir}",
                f"Confidence: {confidence:.0f}%",
                "",
                "(Heuristic inference)",
                "Assumes depth axis orientation is correct; flip if reversed."
            ]

            info_text = '\n'.join(info_lines)
            ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
                    verticalalignment='top',
                    fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

        ax.set_xlabel('Longitudinal Position (m)')
        ax.set_ylabel('Wear Intensity')
        ax.set_title('Directionality Evidence (Heuristic)')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

    def plot_longitudinal_profile(self, output_file=None):
        """Plot longitudinal (front-to-back) wear profile (Directionality Evidence)"""
        longitudinal = self.results['longitudinal']

        if 'longitudinal_profile' not in longitudinal:
            print("No longitudinal profile data available")
            return None, None

        fig, ax = plt.subplots(figsize=(10, 6))
        self._draw_longitudinal_profile(ax, include_info_box=True)
        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

        return fig, ax

    def plot_summary_dashboard(self, output_file=None):
        """Create Stable Evidence Dashboard with key findings in 2x2 grid layout"""
        # Create 2x2 grid layout
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.subplots_adjust(hspace=0.3, wspace=0.3)

        fig.suptitle('Stair Wear Evidence Dashboard', fontsize=14, fontweight='bold')

        # Panel A (0,0): Wear Depth Heatmap
        ax_heatmap = axes[0, 0]
        self._draw_wear_heatmap(ax_heatmap, include_colorbar=True, include_info_box=True)

        # Panel B (0,1): Stable Conclusions (text)
        ax_conclusions = axes[0, 1]
        ax_conclusions.axis('off')

        # Build stable conclusions text
        wear_profile = self.results['wear_profile']
        method = wear_profile.get('method', 'legacy_grid')
        max_depth = wear_profile.get('max_depth', 0)
        volume = wear_profile.get('volume', 0)

        simultaneity = self.results.get('simultaneity', {})
        pattern = simultaneity.get('pattern', 'Unknown')
        W_eff = simultaneity.get('W_eff', 0)
        description = simultaneity.get('description', 'N/A')

        direction = self.results.get('directionality', {})
        dominant_dir = direction.get('dominant_direction', 'Unknown')
        confidence = direction.get('confidence', 0)
        explanation = direction.get('explanation', 'N/A')

        # Check for weak plane fit
        ref_quality = wear_profile.get('ref_plane_quality', {})
        weak_plane_note = ""
        if ref_quality.get('inlier_fraction', 1.0) < 0.5:
            weak_plane_note = " (⚠ weak plane fit)"

        conclusions_text = f"""STABLE CONCLUSIONS
━━━━━━━━━━━━━━━━━━━━━━━━━

✅ GEOMETRY:
  • Max depth: {max_depth:.1f} mm
  • Volume: {volume:.3g} m³ ({volume*1000:.1f} L)
  • Method: {method}{weak_plane_note}

✅ SIMULTANEITY (Physics-Constrained):
  • Pattern: {pattern}
  • W_eff: {W_eff:.2f} m
  • {description}

⚠ DIRECTION (Heuristic):
  • {dominant_dir}
  • Confidence: {confidence:.0f}%
  • {explanation}
  • Note: Assumes depth axis orientation
    is correct; flip if reversed.

⚠ TRAFFIC: CALIBRATION-LIMITED
  • Traffic estimates require validated
    scale/material parameters.
  • See traffic_assumptions.png for
    conditional estimates."""

        # Add temporal drift warning if detected
        if simultaneity.get('temporal_drift_likely', False):
            conclusions_text += "\n\n⚠ Temporal drift detected\n  (peaks shift over depth)"

        ax_conclusions.text(0.05, 0.95, conclusions_text, transform=ax_conclusions.transAxes,
                          fontsize=9, verticalalignment='top', fontfamily='monospace',
                          bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

        # Panel C (1,0): Lateral Wear Evidence
        ax_lateral = axes[1, 0]
        lateral = self.results['lateral']
        if 'lateral_profile' in lateral:
            self._draw_lateral_distribution(ax_lateral, include_info_box=False, include_stat_note=True)
            # Simplify title for dashboard
            ax_lateral.set_title(f'Lateral Wear: {pattern}', fontsize=10)

        # Panel D (1,1): Directionality Evidence
        ax_longitudinal = axes[1, 1]
        longitudinal = self.results['longitudinal']
        if 'longitudinal_profile' in longitudinal:
            self._draw_longitudinal_profile(ax_longitudinal, include_info_box=True)
            # Keep heuristic label in title
            ax_longitudinal.set_title(f'Direction (Heuristic): {dominant_dir}', fontsize=10)

        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

        return fig

    def plot_traffic_assumptions(self, output_file=None):
        """Create traffic assumptions plot with calibration warnings and checklist"""
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111)
        ax.axis('off')

        traffic = self.results.get('traffic_volume', {})
        wear_profile = self.results.get('wear_profile', {})
        warnings = traffic.get('warnings', [])

        # Determine calibration status
        daily_500y = traffic.get('daily_traffic_500y', 0)
        total_steps = traffic.get('total_steps', 0)
        is_limited = (
            daily_500y > 50000 or 
            daily_500y < 0.1 or 
            len(warnings) > 0
        )

        # Banner
        if is_limited:
            banner = "⚠ CALIBRATION LIMITED ⚠"
            banner_subtitle = "Verify scale/material before interpreting traffic estimates"
            banner_color = '#ffcccc'  # Light red
        else:
            banner = "Traffic Estimates (Conditional on Assumptions)"
            banner_subtitle = "These estimates depend on calibration parameters"
            banner_color = '#ccffcc'  # Light green

        # Draw banner
        ax.text(0.5, 0.95, banner, transform=ax.transAxes,
               fontsize=16, fontweight='bold', ha='center', va='top',
               bbox=dict(boxstyle='round,pad=0.5', facecolor=banner_color, alpha=0.8))
        ax.text(0.5, 0.88, banner_subtitle, transform=ax.transAxes,
               fontsize=10, ha='center', va='top', style='italic')

        # Assumptions block
        assumptions = traffic.get('assumptions', {})
        k_spec_value = traffic.get('k_spec_input_value', 0)
        k_spec_units = traffic.get('k_spec_input_units', 'unknown')
        k_spec_SI = traffic.get('k_spec_SI', 0)
        
        assumptions_text = f"""
ASSUMPTIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
k_spec value:    {k_spec_value:.2e}
k_spec units:    {k_spec_units}
k_spec SI:       {k_spec_SI:.2e} m²/N

Body weight:     {assumptions.get('avg_weight_N', 700)} N
GRF multiplier:  {assumptions.get('grf_multiplier', 1.7)}
Microslip:       {assumptions.get('microslip_m', 0.002)} m

Wear volume:     {traffic.get('wear_volume', 0):.3e} m³
"""
        ax.text(0.05, 0.78, assumptions_text, transform=ax.transAxes,
               fontsize=9, va='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

        # Traffic estimates table
        traffic_text = f"""
ESTIMATED TRAFFIC (footsteps/day)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total footsteps: {total_steps:.2e}

IF scale/material assumptions hold:

  Age         Daily Traffic
  ───────────────────────────
  50 years    {total_steps/(50*365) if total_steps > 0 else 0:,.1f} footsteps/day
  100 years   {traffic.get('daily_traffic_100y', 0):,.1f} footsteps/day
  200 years   {total_steps/(200*365) if total_steps > 0 else 0:,.1f} footsteps/day
  500 years   {traffic.get('daily_traffic_500y', 0):,.1f} footsteps/day
  1000 years  {traffic.get('daily_traffic_1000y', 0):,.1f} footsteps/day
"""
        ax.text(0.55, 0.78, traffic_text, transform=ax.transAxes,
               fontsize=9, va='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

        # Warnings block (if any)
        if warnings:
            warnings_text = "WARNINGS\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            for w in warnings:
                warnings_text += f"• {w}\n"
            ax.text(0.05, 0.38, warnings_text, transform=ax.transAxes,
                   fontsize=9, va='top', fontfamily='monospace', color='darkred',
                   bbox=dict(boxstyle='round', facecolor='#ffeeee', alpha=0.9))
            checklist_y = 0.22
        else:
            checklist_y = 0.38

        # Calibration checklist
        checklist_text = """
HOW TO CALIBRATE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. MEASURE REAL STAIR WIDTH
   Use: --measured-width-m <width_in_meters>

2. IF MESH UNITS ARE MILLIMETERS
   Use: --scale 0.001

3. IF MESH UNITS ARE CENTIMETERS
   Use: --scale 0.01

4. TRY DIFFERENT MATERIALS
   Use: --material sandstone
   Use: --material limestone  
   Use: --material marble

5. COMPARE ROBUST VS LEGACY METHODS
   Run: python validate_ql_comparison.py
"""
        ax.text(0.55, checklist_y, checklist_text, transform=ax.transAxes,
               fontsize=8, va='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        return fig

    def plot_all(self, output_dir=None, uncertainty_results=None):
        """Generate all plots including new traffic assumptions page
        
        Args:
            output_dir: Directory to save plots, or None to show
            uncertainty_results: Optional UncertaintyResults object for uncertainty plots
        """
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)

            self.plot_wear_heatmap(output_dir / 'wear_heatmap.png')
            self.plot_lateral_distribution(output_dir / 'lateral_distribution.png')
            self.plot_longitudinal_profile(output_dir / 'longitudinal_profile.png')
            self.plot_summary_dashboard(output_dir / 'summary_dashboard.png')
            self.plot_traffic_assumptions(output_dir / 'traffic_assumptions.png')
            
            # Generate uncertainty dashboard if uncertainty results provided
            if uncertainty_results:
                self.plot_uncertainty_dashboard(
                    output_dir / 'uncertainty_dashboard.png',
                    uncertainty_results
                )
        else:
            self.plot_wear_heatmap()
            self.plot_lateral_distribution()
            self.plot_longitudinal_profile()
            self.plot_summary_dashboard()
            self.plot_traffic_assumptions()
            if uncertainty_results:
                self.plot_uncertainty_dashboard(None, uncertainty_results)

    def plot_uncertainty_dashboard(self, output_file, uncertainty_results):
        """Create uncertainty visualization dashboard.
        
        Args:
            output_file: Output file path or None to show
            uncertainty_results: UncertaintyResults object from uncertainty.py
        """
        # Use GridSpec for flexible layout: 2 rows, 2 columns
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
        
        fig.suptitle('Uncertainty Analysis Dashboard', fontsize=14, fontweight='bold')
        
        # Panel A (0,0): Sensitivity Tornado Chart
        ax_sensitivity = fig.add_subplot(gs[0, 0])
        self._draw_sensitivity_tornado(ax_sensitivity, uncertainty_results)
        
        # Panel B (0,1): Traffic CI by Horizon
        ax_traffic = fig.add_subplot(gs[0, 1])
        self._draw_traffic_ci_comparison(ax_traffic, uncertainty_results)
        
        # Panel C (1,0): P(ascent) interpretation
        ax_direction = fig.add_subplot(gs[1, 0])
        self._draw_direction_interpretation(ax_direction, uncertainty_results)
        
        # Panel D (1,1): Summary text
        ax_summary = fig.add_subplot(gs[1, 1])
        self._draw_uncertainty_summary(ax_summary, uncertainty_results)
        
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        
        return fig
    
    def _draw_sensitivity_tornado(self, ax, unc):
        """Draw sensitivity tornado chart."""
        sensitivity = unc.sensitivity
        
        # Sort by importance
        sorted_items = sorted(sensitivity.items(), key=lambda x: x[1], reverse=True)
        names = [item[0] for item in sorted_items]
        values = [item[1] for item in sorted_items]
        
        # Create horizontal bar chart
        colors = ['#4CAF50' if v == max(values) else '#2196F3' for v in values]
        bars = ax.barh(names, values, color=colors, alpha=0.8)
        
        # Add value labels
        for bar, val in zip(bars, values):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{val:.0%}', va='center', fontsize=9)
        
        ax.set_xlabel('Variance Contribution')
        ax.set_title('Sensitivity Analysis: Uncertainty Sources')
        ax.set_xlim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Annotate dominant source
        dominant = unc.traffic_dominated_by
        ax.text(0.98, 0.02, f"Dominant: {dominant}",
               transform=ax.transAxes, fontsize=9, ha='right',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    def _draw_traffic_ci_comparison(self, ax, unc):
        """Draw traffic CI comparison across horizons."""
        horizons = ['50y', '100y', '200y', '500y', '1000y']
        traffic_dists = [unc.traffic_50y, unc.traffic_100y, unc.traffic_200y,
                        unc.traffic_500y, unc.traffic_1000y]
        
        medians = [d.median for d in traffic_dists]
        ci95_lows = [d.ci_95[0] for d in traffic_dists]
        ci95_highs = [d.ci_95[1] for d in traffic_dists]
        ci50_lows = [d.ci_50[0] for d in traffic_dists]
        ci50_highs = [d.ci_50[1] for d in traffic_dists]
        
        x = range(len(horizons))
        
        # Plot CI95 as error bars
        ax.fill_between(x, ci95_lows, ci95_highs, alpha=0.2, color='blue', label='95% CI')
        ax.fill_between(x, ci50_lows, ci50_highs, alpha=0.4, color='blue', label='50% CI')
        ax.plot(x, medians, 'bo-', linewidth=2, markersize=8, label='Median')
        
        ax.set_xticks(x)
        ax.set_xticklabels(horizons)
        ax.set_xlabel('Time Horizon')
        ax.set_ylabel('Daily Traffic (footsteps/day)')
        ax.set_title('Traffic Estimates by Age Assumption')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Use log scale if range is large
        if max(ci95_highs) / (min(ci95_lows) + 1e-20) > 100:
            ax.set_yscale('log')
    
    def _draw_direction_interpretation(self, ax, unc):
        """Draw direction interpretation with CI."""
        ax.axis('off')
        
        p = unc.prob_ascent.median
        ci_low, ci_high = unc.prob_ascent.ci_95
        classification = unc.get_direction_classification()
        label = unc.get_direction_label()
        
        # Color based on classification
        if classification == "ASCENT":
            color = '#4CAF50'  # Green
            icon = "↑"
        elif classification == "DESCENT":
            color = '#F44336'  # Red
            icon = "↓"
        else:
            color = '#FFC107'  # Amber
            icon = "↔"
        
        ax.text(0.5, 0.85, f"{icon} {classification}", 
               transform=ax.transAxes, fontsize=24, fontweight='bold',
               ha='center', color=color)
        
        ax.text(0.5, 0.65, label, 
               transform=ax.transAxes, fontsize=12, ha='center')
        
        # Draw simple CI visualization
        ax.axhline(0.4, xmin=0.1, xmax=0.9, color='gray', linewidth=2)
        ax.axvline(0.5, ymin=0.35, ymax=0.45, color='gray', linewidth=2, linestyle='--', label='0.5 threshold')
        
        # Show where the CI falls
        x_low = 0.1 + 0.8 * ci_low
        x_high = 0.1 + 0.8 * ci_high
        x_median = 0.1 + 0.8 * p
        
        ax.plot([x_low, x_high], [0.4, 0.4], color=color, linewidth=8, solid_capstyle='round', alpha=0.6)
        ax.plot([x_median], [0.4], 'o', color=color, markersize=12)
        
        # Labels
        ax.text(0.1, 0.32, '0%', transform=ax.transAxes, fontsize=9, ha='center')
        ax.text(0.5, 0.32, '50%', transform=ax.transAxes, fontsize=9, ha='center')
        ax.text(0.9, 0.32, '100%', transform=ax.transAxes, fontsize=9, ha='center')
        ax.text(0.5, 0.25, 'P(ascent)', transform=ax.transAxes, fontsize=10, ha='center')
        
        # Caveat
        if unc.axis_ambiguous:
            ax.text(0.5, 0.08, "⚠ Axis orientation ambiguous",
                   transform=ax.transAxes, fontsize=9, ha='center', color='orange')
        else:
            ax.text(0.5, 0.08, "Note: Assumes depth axis orientation is correct",
                   transform=ax.transAxes, fontsize=8, ha='center', style='italic', color='gray')
        
        ax.set_title('Directionality Assessment')
    
    def _draw_uncertainty_summary(self, ax, unc):
        """Draw uncertainty summary text panel."""
        ax.axis('off')
        
        # Build summary text
        summary = f"""UNCERTAINTY ANALYSIS SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Monte Carlo samples: {unc.n_samples}
Seed: {unc.seed}

GEOMETRY (Stable)
  Volume: {unc.volume.median:.2e} m³
     CI95: [{unc.volume.ci_95[0]:.2e}, {unc.volume.ci_95[1]:.2e}]
  Max depth: {unc.max_depth.median:.2f} mm
     CI95: [{unc.max_depth.ci_95[0]:.2f}, {unc.max_depth.ci_95[1]:.2f}]

TRAFFIC (Conditional, 500y)
  Daily: {unc.traffic_500y.median:.2e} steps/day
     CI95: [{unc.traffic_500y.ci_95[0]:.2e}, {unc.traffic_500y.ci_95[1]:.2e}]

FLAGS
  Scale ambiguous: {unc.scale_ambiguous}
  Weak plane fit: {unc.weak_plane_fit}
  Axis ambiguous: {unc.axis_ambiguous}
"""
        
        ax.text(0.05, 0.95, summary, transform=ax.transAxes,
               fontsize=9, va='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
        
        ax.set_title('Summary Statistics')

