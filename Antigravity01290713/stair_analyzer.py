 #!/usr/bin/env python3
"""
Stair Wear Analysis Tool
Analyzes 3D models (.obj files) of worn stairs to determine:
- Usage frequency (traffic volume)
- Directional preferences (ascent vs descent)
- Simultaneous usage patterns (single file vs side-by-side)

Based on the mathematical model from ModelG.md
"""

import numpy as np
import argparse
import sys
from pathlib import Path

# Import analysis modules
from obj_parser import OBJParser
from wear_analyzer import WearAnalyzer
from traffic_estimator import TrafficEstimator
from visualizer import WearVisualizer
from axis_config import AxisConfig



# ============================================================================
# Calibration Status Helpers
# ============================================================================

def detect_likely_units(mesh_width_raw, scale):
    """
    Heuristic detector for likely mesh units based on raw width
    
    Args:
        mesh_width_raw: Width in model units (before scaling)
        scale: Current scale factor
    
    Returns:
        dict with 'likely_units', 'suggested_scale', 'confidence'
    """
    # Plausible stair widths: 0.4m to 5m
    # If mesh is 500-1500 units -> likely mm
    # If mesh is 50-150 units -> likely cm
    # If mesh is 0.5-5 units -> likely m
    
    if scale == 1.0:
        if 500 <= mesh_width_raw <= 1500:
            return {
                'likely_units': 'mm',
                'suggested_scale': 0.001,
                'confidence': 'high',
                'reasoning': f'Width {mesh_width_raw:.1f} units suggests millimeters'
            }
        elif 50 <= mesh_width_raw <= 150:
            return {
                'likely_units': 'cm',
                'suggested_scale': 0.01,
                'confidence': 'medium',
                'reasoning': f'Width {mesh_width_raw:.1f} units suggests centimeters'
            }
        elif 0.4 <= mesh_width_raw <= 5.0:
            return {
                'likely_units': 'm',
                'suggested_scale': 1.0,
                'confidence': 'high',
                'reasoning': f'Width {mesh_width_raw:.2f} units suggests meters'
            }
        else:
            return {
                'likely_units': 'unknown',
                'suggested_scale': None,
                'confidence': 'low',
                'reasoning': f'Width {mesh_width_raw:.2f} units outside typical range'
            }
    else:
        # Scale already applied
        return {
            'likely_units': 'scaled',
            'suggested_scale': scale,
            'confidence': 'user_specified',
            'reasoning': f'User specified scale={scale}'
        }


def is_calibration_limited(traffic_volume_result, wear_profile, scale, width_m):
    """
    Detect if traffic volume estimates are calibration-limited
    
    Returns:
        dict with 'is_limited', 'reasons' (list of strings)
    """
    reasons = []
    
    # Check 1: Extremely high daily traffic
    daily_500y = traffic_volume_result.get('daily_traffic_500y', 0)
    if daily_500y > 50000:
        reasons.append(f"Extremely high traffic ({daily_500y:.0f} footsteps/day at 500y)")
    
    # Check 2: Scale == 1.0 with implausible width
    if scale == 1.0:
        if width_m < 0.4 or width_m > 5.0:
            reasons.append(f"Implausible stair width ({width_m:.2f}m) with scale=1.0 - likely unit mismatch")
    
    # Check 3: Weak reference plane fit (if robust method)
    if 'ref_plane_quality' in wear_profile:
        inlier_frac = wear_profile['ref_plane_quality'].get('inlier_fraction', 1.0)
        if inlier_frac < 0.5:
            reasons.append(f"Weak reference plane fit ({inlier_frac:.1%} inliers) - results less reliable")
    
    # Check 4: Extremely low traffic
    if daily_500y < 0.1 and traffic_volume_result.get('total_steps', 0) > 0:
        reasons.append(f"Extremely low traffic ({daily_500y:.3f} footsteps/day) - likely calibration issue")
    
    return {
        'is_limited': len(reasons) > 0,
        'reasons': reasons
    }


class StairAnalyzer:
    """Main class for analyzing stair wear patterns"""

    def __init__(self, obj_file, material_type='granite', stair_width=None,
                 scale=1.0, axis_config=None, use_legacy_wear_volume=False,
                 ransac_iters=500, ransac_thresh_mm=2.0, seg_angle_deg=25.0,
                 roi_quantiles=(0.01, 0.99)):
        """
        Initialize the stair analyzer

        Args:
            obj_file: Path to .obj file containing 3D model of stairs
            material_type: Type of material ('granite', 'marble', 'sandstone', 'limestone')
            stair_width: Width of stairs in meters (auto-detected if None)
            scale: Scale factor for mesh coordinates (default 1.0)
            axis_config: AxisConfig object for axis mapping (auto-detect if None)
            use_legacy_wear_volume: If True, use legacy grid method (default: False)
            ransac_iters: RANSAC iterations for plane fitting (default: 500)
            ransac_thresh_mm: RANSAC threshold in mm (default: 2.0)
            seg_angle_deg: Segmentation angle threshold (default: 25.0)
            roi_quantiles: ROI quantiles tuple (default: (0.01, 0.99))
        """
        self.obj_file = Path(obj_file)
        self.material_type = material_type
        self.stair_width = stair_width
        self.scale = scale
        self.axis_config = axis_config
        self.use_legacy_wear_volume = use_legacy_wear_volume
        self.ransac_iters = ransac_iters
        self.ransac_thresh_mm = ransac_thresh_mm
        self.seg_angle_deg = seg_angle_deg
        self.roi_quantiles = roi_quantiles

        # Material properties from ModelG.md Table 1
        # k_spec values with explicit units
        # Default units: mm³/(N·m) to match report labels
        self.material_properties = {
            'granite': {
                'k_spec_value': 1e-6,  # 1e-6 mm³/(N·m)
                'k_spec_units': 'mm3_per_Nm',
                'hardness': 7.5e9,  # Pa
                'name': 'Granite'
            },
            'marble': {
                'k_spec_value': 1e-5,  # 1e-5 mm³/(N·m)
                'k_spec_units': 'mm3_per_Nm',
                'hardness': 2.25e9,  # Pa
                'name': 'Marble'
            },
            'sandstone': {
                'k_spec_value': 5e-5,  # 5e-5 mm³/(N·m)
                'k_spec_units': 'mm3_per_Nm',
                'hardness': 2e9,  # Pa
                'name': 'Sandstone'
            },
            'limestone': {
                'k_spec_value': 3e-5,  # 3e-5 mm³/(N·m)
                'k_spec_units': 'mm3_per_Nm',
                'hardness': 2e9,  # Pa
                'name': 'Limestone'
            },
            'oak': {
                'k_spec_value': 1e-4,  # 1e-4 mm³/(N·m)
                'k_spec_units': 'mm3_per_Nm',
                'hardness': 5e8,  # Pa
                'name': 'Oak Wood'
            }
        }

        self.results = {}

    def load_model(self):
        """Load and parse the 3D model"""
        print(f"Loading 3D model from {self.obj_file}...")
        parser = OBJParser(self.obj_file, scale=self.scale)
        self.mesh = parser.parse()
        print(f"  Loaded {len(self.mesh['vertices'])} vertices, {len(self.mesh['faces'])} faces")
        
        # Display scale/units heuristics
        if hasattr(self, 'axis_config') and self.axis_config:
            width_raw = self.mesh['bounds']['size'][self.axis_config.width_axis]
            units_info = detect_likely_units(width_raw, self.scale)
            
            if units_info['likely_units'] != 'scaled':
                print(f"  Scale heuristic: {units_info['reasoning']}")
                if units_info['suggested_scale'] and units_info['suggested_scale'] != 1.0:
                    print(f"    Suggestion: Try --scale {units_info['suggested_scale']}")
        if self.scale != 1.0:
            print(f"  Applied scale factor: {self.scale}")

    def analyze_wear(self):
        """Analyze wear patterns on the stairs"""
        print("\nAnalyzing wear patterns...")
        # IMPORTANT: Use named arguments to avoid passing stair_width as axis_config
        analyzer = WearAnalyzer(self.mesh, axis_config=self.axis_config, stair_width=self.stair_width)

        # Extract wear depth profile with robust parameters
        self.wear_profile = analyzer.extract_wear_profile(
            use_triangle_method=not self.use_legacy_wear_volume,
            ransac_iters=self.ransac_iters,
            ransac_thresh_mm=self.ransac_thresh_mm,
            seg_angle_deg=self.seg_angle_deg,
            roi_quantiles=self.roi_quantiles
        )
        print(f"  Extracted wear profile: max depth = {self.wear_profile['max_depth']:.2f} mm")

        # Analyze lateral distribution
        self.lateral_analysis = analyzer.analyze_lateral_distribution()
        print(f"  Lateral analysis: {self.lateral_analysis['n_modes']} mode(s) detected")

        # Analyze longitudinal (front-to-back) wear
        self.longitudinal_analysis = analyzer.analyze_longitudinal_profile()
        print(f"  Nosing wear ratio: {self.longitudinal_analysis['nosing_wear_ratio']:.2f}")

        self.results['wear_profile'] = self.wear_profile
        self.results['lateral'] = self.lateral_analysis
        self.results['longitudinal'] = self.longitudinal_analysis

    def estimate_traffic(self):
        """Estimate traffic patterns and volume"""
        print("\nEstimating traffic patterns...")

        material_props = self.material_properties[self.material_type]
        estimator = TrafficEstimator(
            self.wear_profile,
            self.lateral_analysis,
            self.longitudinal_analysis,
            material_props
        )

        # Question 1: How often were the stairs used?
        traffic_volume = estimator.estimate_traffic_volume()
        self.results['traffic_volume'] = traffic_volume
        print(f"  Estimated total footsteps: {traffic_volume['total_steps']:.2e}")
        print(f"  Estimated daily traffic (500 years): {traffic_volume['daily_traffic_500y']:.3f} footsteps/day")

        # Display warnings if any
        if traffic_volume.get('warnings'):
            print(f"  WARNINGS:")
            for warning in traffic_volume['warnings']:
                print(f"    - {warning}")

        # Question 2: Was a certain direction favored?
        directionality = estimator.analyze_directionality()
        self.results['directionality'] = directionality
        print(f"  Direction bias: {directionality['dominant_direction']} ({directionality['confidence']:.1f}% confidence)")

        # Question 3: How many people used simultaneously?
        simultaneity = estimator.analyze_simultaneity()
        self.results['simultaneity'] = simultaneity
        print(f"  Usage pattern: {simultaneity['pattern']} ({simultaneity['description']})")

    def generate_report(self, output_file=None):
        """Generate detailed analysis report"""
        print("\n" + "="*70)
        print("STAIR WEAR ANALYSIS REPORT")
        print("="*70)

        print(f"\nInput File: {self.obj_file}")
        print(f"Material: {self.material_properties[self.material_type]['name']}")

        # Display wear method diagnostics
        method = self.wear_profile.get('method', 'unknown')
        print(f"Wear Volume Method: {method}")

        if method == 'robust_triangle' and 'ref_plane_quality' in self.wear_profile:
            print("\n--- ROBUST METHOD DIAGNOSTICS ---")
            ref = self.wear_profile['ref_plane_quality']
            print(f"Reference Plane:")
            print(f"  Method: {ref['method']}")
            print(f"  Inlier fraction: {ref['inlier_fraction']:.1%}")
            print(f"  RMS error: {ref['rms_error_mm']:.2f} mm")
            print(f"  Candidates used: {ref['candidates_used']}")

            if 'tread_seg' in self.wear_profile:
                seg = self.wear_profile['tread_seg']
                print(f"Tread Segmentation:")
                print(f"  Tread triangles: {seg['tread_count']}")
                print(f"  Excluded fraction: {seg['excluded_fraction']:.1%}")

            # Warnings
            if ref['inlier_fraction'] < 0.5:
                print(f"  WARNING: Weak reference plane fit (inliers < 50%)")
            if self.wear_profile.get('tread_seg', {}).get('excluded_fraction', 0) > 0.5:
                print(f"  WARNING: High triangle exclusion (> 50%)")
            if self.wear_profile.get('max_depth', 0) > 200:
                print(f"  WARNING: Implausible max depth (> 200mm)")

        print("\n--- QUESTION 1: HOW OFTEN WERE THE STAIRS USED? ---")
        tv = self.results['traffic_volume']
        print(f"Total estimated footsteps: {tv['total_steps']:.2e}")
        print(f"Wear volume: {tv['wear_volume']:.2e} m^3")
        print(f"\nEstimated daily traffic (by building age):")
        print(f"  If  50 years old:  {tv['total_steps']/(50*365):.3f} footsteps/day")
        print(f"  If 100 years old:  {tv['daily_traffic_100y']:.3f} footsteps/day")
        print(f"  If 200 years old:  {tv['total_steps']/(200*365):.3f} footsteps/day")
        print(f"  If 500 years old:  {tv['daily_traffic_500y']:.3f} footsteps/day")
        print(f"  If 1000 years old: {tv['daily_traffic_1000y']:.3f} footsteps/day")

        # Display warnings
        if tv.get('warnings'):
            print("\n  TRAFFIC VOLUME WARNINGS:")
            for warning in tv['warnings']:
                print(f"    - {warning}")

        print("\n--- QUESTION 2: WAS A DIRECTION FAVORED? ---")
        dir_result = self.results['directionality']
        print(f"Dominant direction: {dir_result['dominant_direction']}")
        print(f"Confidence: {dir_result['confidence']:.1f}%")
        print(f"Ascent/Descent ratio: {dir_result['ascent_descent_ratio']:.2f}")
        print(f"Explanation: {dir_result['explanation']}")

        print("\n--- QUESTION 3: HOW MANY PEOPLE SIMULTANEOUSLY? ---")
        sim = self.results['simultaneity']
        print(f"Usage pattern: {sim['pattern']}")
        print(f"Description: {sim['description']}")
        print(f"Number of lanes: {sim['n_lanes']}")
        print(f"Effective width: {sim['W_eff']:.2f} m")
        print(f"Single file possible: {sim.get('single_file_possible', True)}")
        print(f"Two-abreast lanes: {sim.get('M_max_two_abreast', sim.get('M_max', 0))}")
        
        # Display temporal drift
        if sim.get('temporal_drift_likely', False):
            print(f"Temporal drift detected: YES (peaks shift over depth)")
        else:
            print(f"Temporal drift detected: NO")

        print("\n" + "="*70)

        if output_file:
            with open(output_file, 'w') as f:
                # Write report to file (simplified version)
                f.write("STAIR WEAR ANALYSIS REPORT\n")
                f.write(f"Material: {self.material_type}\n")
                f.write(f"Traffic: {tv['total_steps']:.2e} steps\n")
                f.write(f"Direction: {dir_result['dominant_direction']}\n")
                f.write(f"Pattern: {sim['pattern']}\n")
            print(f"\nReport saved to {output_file}")

    def visualize(self, output_dir=None, uncertainty_results=None):
        """Generate visualization plots
        
        Args:
            output_dir: Output directory for plots
            uncertainty_results: Optional UncertaintyResults for uncertainty dashboard
        """
        print("\nGenerating visualizations...")
        viz = WearVisualizer(self.results, self.mesh)
        viz.plot_all(output_dir, uncertainty_results=uncertainty_results)
        if uncertainty_results:
            print("  Uncertainty dashboard generated")
        print("  Visualizations complete")

    def run_full_analysis(self, output_dir=None):
        """Run complete analysis pipeline"""
        self.load_model()
        self.analyze_wear()
        self.estimate_traffic()
        self.generate_report()

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
            self.visualize(output_dir)
            self.generate_report(output_dir / "report.txt")
    
    def run_uncertainty_analysis(self, output_dir=None, n_samples=500, seed=42):
        """
        Run analysis with uncertainty propagation.
        
        Args:
            output_dir: Output directory for results
            n_samples: Number of Monte Carlo samples
            seed: Random seed for reproducibility
            
        Returns:
            UncertaintyResults object
        """
        from uncertainty import (
            UncertaintyPropagator, ScaleUncertainty, 
            MaterialUncertainty, UncertaintyResults
        )
        
        print(f"\n{'='*70}")
        print("UNCERTAINTY PROPAGATION ANALYSIS")
        print(f"{'='*70}")
        print(f"Monte Carlo samples: {n_samples}")
        print(f"Seed: {seed}")
        
        # Load model
        self.load_model()
        
        # Create WearAnalyzer and run RANSAC
        from wear_analyzer import WearAnalyzer
        wear_analyzer = WearAnalyzer(self.mesh, axis_config=self.axis_config, stair_width=self.stair_width)
        
        # Get RANSAC result with inlier info
        ransac_result = wear_analyzer.fit_robust_reference_plane(
            ransac_iters=self.ransac_iters,
            ransac_thresh_mm=self.ransac_thresh_mm
        )
        print(f"  RANSAC: {ransac_result['inlier_fraction']:.1%} inliers, RMS {ransac_result['rms_error_mm']:.2f}mm")
        
        # Get bootstrap plane samples
        plane_unc = wear_analyzer.get_bootstrap_plane_uncertainty(
            ransac_result, n_bootstrap=n_samples, seed=seed
        )
        print(f"  Bootstrap samples: {plane_unc.n_samples}")
        print(f"  Weak fit: {plane_unc.weak_fit}")
        
        # Build uncertainty models - use correct width_axis from analyzer
        scale_unc = ScaleUncertainty.from_args(
            self.scale if self.scale != 1.0 else None,
            self.mesh,
            width_axis=wear_analyzer.width_axis
        )
        material_unc = MaterialUncertainty.from_name(self.material_type)
        
        print(f"  Scale: {'user-provided' if scale_unc.user_provided else 'inferred'}")
        print(f"  Material: {material_unc.name} (k_spec median: {material_unc.k_spec_median:.1e})")
        
        # Run Monte Carlo propagation
        propagator = UncertaintyPropagator(n_samples=n_samples, seed=seed)
        self.uncertainty_results = propagator.propagate(
            mesh=self.mesh,
            wear_analyzer=wear_analyzer,
            scale_unc=scale_unc,
            plane_unc=plane_unc,
            material_unc=material_unc
        )
        
        # Print results summary
        print(f"\n--- UNCERTAINTY RESULTS ---")
        print(f"Volume: {self.uncertainty_results.volume.format_with_ci(2, True)} m³")
        print(f"Max depth: {self.uncertainty_results.max_depth.format_compact(2)} mm")
        print(f"P(ascent): {self.uncertainty_results.prob_ascent.median:.1%} [{self.uncertainty_results.prob_ascent.ci_95[0]:.1%} - {self.uncertainty_results.prob_ascent.ci_95[1]:.1%}]")
        print(f"Direction: {self.uncertainty_results.get_direction_label()}")
        print(f"\nTraffic (conditional, 500y):")
        print(f"  {self.uncertainty_results.traffic_500y.format_with_ci(2)} footsteps/day")
        print(f"\nSensitivity:")
        for key, val in self.uncertainty_results.sensitivity.items():
            bar = '█' * int(val * 20)
            print(f"  {key:8s}: {bar} {val:.0%}")
        print(f"\nDominated by: {self.uncertainty_results.traffic_dominated_by}")
        
        # Save results if output_dir specified
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
            self._save_uncertainty_results(output_dir)
            
            # Run full analysis to populate self.results for visualization
            print("\n--- Generating Visualizations ---")
            self.analyze_wear()
            self.estimate_traffic()
            
            # Generate all visualizations including uncertainty dashboard
            self.visualize(output_dir, uncertainty_results=self.uncertainty_results)
        
        return self.uncertainty_results
    
    def _save_uncertainty_results(self, output_dir):
        """Save uncertainty results to JSON file."""
        import json
        
        results = self.uncertainty_results
        
        def dist_to_dict(d):
            """Convert DistributionSummary to JSON-serializable dict."""
            return {
                'median': float(d.median),
                'mean': float(d.mean),
                'ci_50': [float(d.ci_50[0]), float(d.ci_50[1])],
                'ci_95': [float(d.ci_95[0]), float(d.ci_95[1])]
            }
        
        output = {
            'metadata': {
                'n_samples': int(results.n_samples),
                'seed': int(results.seed),
                'material': self.material_type,
                'scale': float(self.scale),
                'scale_hypothesis_posterior': {k: float(v) for k, v in results.scale_hypothesis_posterior.items()} if results.scale_hypothesis_posterior else None
            },
            'geometry': {
                'volume_m3': dist_to_dict(results.volume),
                'max_depth_mm': dist_to_dict(results.max_depth),
                'mean_depth_mm': dist_to_dict(results.mean_depth)
            },
            'directionality': {
                'prob_ascent': dist_to_dict(results.prob_ascent),
                'nosing_ratio': dist_to_dict(results.nosing_ratio),
                'axis_ambiguous': bool(results.axis_ambiguous),
                'classification': results.get_direction_label(),
                'simple_classification': results.get_direction_classification()
            },
            'traffic_conditional': {
                'total_steps': dist_to_dict(results.total_steps),
                'daily_50y': dist_to_dict(results.traffic_50y),
                'daily_100y': dist_to_dict(results.traffic_100y),
                'daily_200y': dist_to_dict(results.traffic_200y),
                'daily_500y': dist_to_dict(results.traffic_500y),
                'daily_1000y': dist_to_dict(results.traffic_1000y),
                'sensitivity': {k: float(v) for k, v in results.sensitivity.items()},
                'dominant_source': str(results.traffic_dominated_by)
            },
            'flags': {
                'scale_ambiguous': bool(results.scale_ambiguous),
                'weak_plane_fit': bool(results.weak_plane_fit)
            }
        }
        
        output_file = output_dir / 'uncertainty_results.json'
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\nUncertainty results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze wear patterns on historic stairs from 3D models'
    )
    parser.add_argument('obj_file', help='Path to .obj file containing stair model')
    parser.add_argument('--material', default='granite',
                       choices=['granite', 'marble', 'sandstone', 'limestone', 'oak'],
                       help='Material type of stairs')
    parser.add_argument('--width', type=float, help='Stair width in meters (auto-detect if not specified)')
    parser.add_argument('--scale', type=float, default=1.0,
                       help='Scale factor for mesh coordinates (default 1.0)')
    parser.add_argument('--measured-width-m', type=float,
                       help='Measured real-world stair width in meters (auto-computes scale)')
    parser.add_argument('--output', '-o', help='Output directory for results')

    # Robust wear volume method parameters
    parser.add_argument('--use-legacy-wear-volume', action='store_true',
                       help='Use legacy grid method instead of robust triangle method')
    parser.add_argument('--ransac-iters', type=int, default=500,
                       help='RANSAC iterations for reference plane fitting (default: 500)')
    parser.add_argument('--ransac-thresh-mm', type=float, default=2.0,
                       help='RANSAC inlier threshold in mm (default: 2.0)')
    parser.add_argument('--seg-angle-deg', type=float, default=25.0,
                       help='Tread segmentation angle threshold in degrees (default: 25.0)')
    parser.add_argument('--roi-quantiles', type=float, nargs=2, default=[0.01, 0.99],
                       help='ROI quantiles for segmentation (default: 0.01 0.99)')

    # Uncertainty propagation parameters
    parser.add_argument('--uncertainty', action='store_true',
                       help='Enable uncertainty propagation analysis')
    parser.add_argument('--n-samples', type=int, default=500,
                       help='Number of Monte Carlo samples for uncertainty (default: 500)')
    parser.add_argument('--uncertainty-seed', type=int, default=42,
                       help='Random seed for uncertainty analysis (default: 42)')

    args = parser.parse_args()

    if not Path(args.obj_file).exists():
        print(f"Error: File not found: {args.obj_file}")
        sys.exit(1)

    # Compute scale from measured width if provided
    scale = args.scale
    if args.measured_width_m is not None:
        # Need to load mesh first to get its width
        print(f"Computing scale from measured width: {args.measured_width_m} m")
        temp_parser = OBJParser(args.obj_file, scale=1.0)
        temp_mesh = temp_parser.parse()

        # Infer axes to get width
        temp_axis = AxisConfig()
        temp_axis.infer_from_mesh(temp_mesh['vertices'], temp_mesh['faces'])
        mesh_width = temp_mesh['bounds']['size'][temp_axis.width_axis]

        scale = args.measured_width_m / mesh_width
        print(f"  Mesh width: {mesh_width:.4f} (units)")
        print(f"  Computed scale: {scale:.6f}")

    analyzer = StairAnalyzer(
        args.obj_file,
        args.material,
        args.width,
        scale=scale,
        use_legacy_wear_volume=args.use_legacy_wear_volume,
        ransac_iters=args.ransac_iters,
        ransac_thresh_mm=args.ransac_thresh_mm,
        seg_angle_deg=args.seg_angle_deg,
        roi_quantiles=tuple(args.roi_quantiles)
    )
    
    if args.uncertainty:
        # Run uncertainty propagation mode
        analyzer.run_uncertainty_analysis(
            output_dir=args.output,
            n_samples=args.n_samples,
            seed=args.uncertainty_seed
        )
    else:
        # Run standard deterministic analysis
        analyzer.run_full_analysis(args.output)


if __name__ == '__main__':
    main()
