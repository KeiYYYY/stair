"""
Uncertainty Propagation Module
Implements Monte Carlo uncertainty propagation for stair wear analysis.
Based on technical specification from implementation_plan.md.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


# =============================================================================
# DISTRIBUTION SUMMARY
# =============================================================================

@dataclass
class DistributionSummary:
    """Summary statistics for a scalar uncertainty distribution."""
    
    mean: float
    median: float
    std: float
    ci_50: Tuple[float, float]  # 25th-75th percentile
    ci_95: Tuple[float, float]  # 2.5th-97.5th percentile
    samples: np.ndarray = field(repr=False)  # Raw samples for plotting
    
    @classmethod
    def from_samples(cls, samples: np.ndarray) -> 'DistributionSummary':
        """Create summary from raw Monte Carlo samples."""
        samples = np.asarray(samples)
        return cls(
            mean=float(np.mean(samples)),
            median=float(np.median(samples)),
            std=float(np.std(samples)),
            ci_50=(float(np.percentile(samples, 25)), float(np.percentile(samples, 75))),
            ci_95=(float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))),
            samples=samples
        )
    
    def format_with_ci(self, precision: int = 2, use_scientific: bool = False) -> str:
        """Format as 'median [CI_low - CI_high]'."""
        fmt = f".{precision}e" if use_scientific else f".{precision}f"
        return f"{self.median:{fmt}} [{self.ci_95[0]:{fmt}} - {self.ci_95[1]:{fmt}}]"
    
    def format_compact(self, precision: int = 2) -> str:
        """Format as 'median [50% CI]' for dashboards."""
        fmt = f".{precision}f"
        return f"{self.median:{fmt}} [{self.ci_50[0]:{fmt}}-{self.ci_50[1]:{fmt}}]"


# =============================================================================
# INPUT UNCERTAINTY MODELS
# =============================================================================

@dataclass
class ScaleUncertainty:
    """Uncertainty model for mesh scale factor."""
    
    # Discrete scale hypotheses: (scale, prior_prob, label)
    hypotheses: List[Tuple[float, float, str]] = field(default_factory=lambda: [
        (0.001, 0.4, "mm"),
        (0.01, 0.3, "cm"),
        (1.0, 0.3, "m")
    ])
    
    # User-provided scale (if any)
    user_provided: bool = False
    user_scale: Optional[float] = None
    
    # Continuous perturbation (coefficient of variation)
    perturbation_cv: float = 0.05
    
    @classmethod
    def from_args(cls, scale: Optional[float], mesh: dict, width_axis: int = 0) -> 'ScaleUncertainty':
        """Create ScaleUncertainty from CLI arguments and mesh data.
        
        Args:
            scale: User-provided scale factor, or None to infer
            mesh: Mesh dictionary with bounds
            width_axis: Index of width axis (from axis_config, default 0)
        """
        if scale is not None:
            return cls(user_provided=True, user_scale=scale)
        
        # Infer likely scale from mesh dimensions using correct axis
        width = mesh['bounds']['size'][width_axis]
        
        # Adjust priors based on raw mesh width
        if width > 100:  # Likely mm
            hypotheses = [(0.001, 0.7, "mm"), (0.01, 0.2, "cm"), (1.0, 0.1, "m")]
        elif width > 1:  # Likely cm or m
            hypotheses = [(0.001, 0.1, "mm"), (0.01, 0.5, "cm"), (1.0, 0.4, "m")]
        else:  # Likely m
            hypotheses = [(0.001, 0.1, "mm"), (0.01, 0.2, "cm"), (1.0, 0.7, "m")]
        
        return cls(hypotheses=hypotheses, user_provided=False)
    
    def sample(self, rng: np.random.Generator) -> float:
        """Sample a scale factor."""
        scale, _ = self.sample_with_hypothesis(rng)
        return scale
    
    def sample_with_hypothesis(self, rng: np.random.Generator) -> Tuple[float, str]:
        """Sample a scale factor and return (scale, hypothesis_label)."""
        if self.user_provided and self.user_scale is not None:
            # User-provided: lognormal perturbation around user value
            return float(self.user_scale * rng.lognormal(0, self.perturbation_cv)), "user"
        else:
            # Sample from discrete hypotheses
            scales = [h[0] for h in self.hypotheses]
            probs = [h[1] for h in self.hypotheses]
            labels = [h[2] for h in self.hypotheses]
            probs = np.array(probs) / np.sum(probs)  # Normalize
            idx = rng.choice(len(scales), p=probs)
            base_scale = scales[idx]
            label = labels[idx]
            # Apply perturbation
            return float(base_scale * rng.lognormal(0, self.perturbation_cv)), label
    
    def get_hypothesis_labels(self) -> List[str]:
        """Get list of hypothesis labels."""
        if self.user_provided:
            return ["user"]
        return [h[2] for h in self.hypotheses]


@dataclass
class PlaneUncertainty:
    """Uncertainty model for reference plane fit."""
    
    # Bootstrap samples of plane parameters (n_samples, 3) for (a, b, c)
    plane_samples: np.ndarray
    
    # Quality metrics from RANSAC
    inlier_fraction: float
    rms_error_mm: float
    candidates_used: int
    
    # Flags
    weak_fit: bool = False
    
    def __post_init__(self):
        self.weak_fit = self.inlier_fraction < 0.6 or self.rms_error_mm > 5.0
    
    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """Sample a plane parameter set."""
        idx = rng.integers(len(self.plane_samples))
        return self.plane_samples[idx]
    
    @property
    def n_samples(self) -> int:
        return len(self.plane_samples)


@dataclass
class MaterialUncertainty:
    """Uncertainty model for material/wear parameters."""
    
    name: str
    
    # Lognormal parameters for k_spec (wear rate)
    k_spec_log_mean: float   # log(median)
    k_spec_log_std: float    # uncertainty factor
    
    # GRF multiplier: Normal, bounded [1.0, 2.5]
    grf_mean: float = 1.7
    grf_std: float = 0.2
    
    # Microslip: Lognormal, median 2mm
    microslip_log_mean: float = field(default_factory=lambda: np.log(0.002))
    microslip_log_std: float = 0.3
    
    # Body weight (N)
    body_weight: float = 700.0
    
    # Material database
    MATERIALS = {
        'granite':   {'k_spec_median': 1e-6,  'log_std': 0.7},
        'marble':    {'k_spec_median': 1e-5,  'log_std': 0.5},
        'limestone': {'k_spec_median': 3e-5,  'log_std': 1.0},
        'sandstone': {'k_spec_median': 5e-5,  'log_std': 1.2},
        'wood':      {'k_spec_median': 1e-4,  'log_std': 0.8},
    }
    
    @classmethod
    def from_name(cls, name: str) -> 'MaterialUncertainty':
        """Create MaterialUncertainty from material name."""
        name_lower = name.lower()
        if name_lower not in cls.MATERIALS:
            raise ValueError(f"Unknown material: {name}. Available: {list(cls.MATERIALS.keys())}")
        
        props = cls.MATERIALS[name_lower]
        return cls(
            name=name_lower,
            k_spec_log_mean=np.log(props['k_spec_median']),
            k_spec_log_std=props['log_std'],
            microslip_log_mean=np.log(0.002)
        )
    
    def sample(self, rng: np.random.Generator) -> Tuple[float, float, float]:
        """Sample (k_spec, grf_multiplier, microslip_distance)."""
        k_spec = rng.lognormal(self.k_spec_log_mean, self.k_spec_log_std)
        grf = np.clip(rng.normal(self.grf_mean, self.grf_std), 1.0, 2.5)
        microslip = rng.lognormal(self.microslip_log_mean, self.microslip_log_std)
        return k_spec, grf, microslip
    
    @property
    def k_spec_median(self) -> float:
        return np.exp(self.k_spec_log_mean)


# =============================================================================
# UNCERTAINTY RESULTS
# =============================================================================

@dataclass
class UncertaintyResults:
    """Complete uncertainty analysis results."""
    
    # Geometry (stable conclusions) - non-default fields first
    volume: DistributionSummary      # m³
    max_depth: DistributionSummary   # mm
    mean_depth: DistributionSummary  # mm
    
    # Directionality (probabilistic)
    prob_ascent: DistributionSummary  # P(ascent dominant)
    nosing_ratio: DistributionSummary
    
    # Traffic (conditional on assumptions) - all horizons
    total_steps: DistributionSummary
    traffic_50y: DistributionSummary
    traffic_100y: DistributionSummary
    traffic_200y: DistributionSummary
    traffic_500y: DistributionSummary
    traffic_1000y: DistributionSummary
    
    # Fields with defaults come last
    axis_ambiguous: bool = False
    sensitivity: Dict[str, float] = field(default_factory=dict)
    scale_ambiguous: bool = False
    scale_hypothesis_posterior: Dict[str, float] = field(default_factory=dict)
    weak_plane_fit: bool = False
    traffic_dominated_by: str = "unknown"
    n_samples: int = 0
    seed: int = 0
    
    def get_direction_label(self) -> str:
        """Get human-readable direction classification based on CI."""
        ci_low, ci_high = self.prob_ascent.ci_95
        p = self.prob_ascent.median
        
        # CI-based classification
        if ci_low > 0.5:  # CI95 entirely > 0.5
            return f"ASCENT ({p*100:.0f}% [{ci_low*100:.0f}-{ci_high*100:.0f}%])"
        elif ci_high < 0.5:  # CI95 entirely < 0.5
            return f"DESCENT ({(1-p)*100:.0f}% [{(1-ci_high)*100:.0f}-{(1-ci_low)*100:.0f}%])"
        else:  # CI95 straddles 0.5
            return f"AMBIGUOUS (P_ascent={p*100:.0f}% [{ci_low*100:.0f}-{ci_high*100:.0f}%])"
    
    def get_direction_classification(self) -> str:
        """Get simple direction classification: ASCENT, DESCENT, or AMBIGUOUS."""
        ci_low, ci_high = self.prob_ascent.ci_95
        if ci_low > 0.5:
            return "ASCENT"
        elif ci_high < 0.5:
            return "DESCENT"
        else:
            return "AMBIGUOUS"


# =============================================================================
# UNCERTAINTY PROPAGATOR
# =============================================================================

class UncertaintyPropagator:
    """Monte Carlo uncertainty propagation engine."""
    
    def __init__(self, n_samples: int = 500, seed: int = 42):
        self.n_samples = n_samples
        self.seed = seed
        self.rng = np.random.default_rng(seed)
    
    def propagate(self,
                  mesh: dict,
                  wear_analyzer,  # WearAnalyzer instance
                  scale_unc: ScaleUncertainty,
                  plane_unc: PlaneUncertainty,
                  material_unc: MaterialUncertainty) -> UncertaintyResults:
        """
        Run Monte Carlo propagation through the analysis pipeline.
        
        Args:
            mesh: Mesh dictionary with vertices, faces, bounds
            wear_analyzer: WearAnalyzer instance (for axis config)
            scale_unc: Scale uncertainty model
            plane_unc: Reference plane uncertainty model
            material_unc: Material parameter uncertainty model
            
        Returns:
            UncertaintyResults with all distributions
        """
        samples = []
        hypothesis_counts: Dict[str, int] = {}
        
        for _ in range(self.n_samples):
            # Sample inputs with hypothesis tracking
            scale, hypothesis = scale_unc.sample_with_hypothesis(self.rng)
            hypothesis_counts[hypothesis] = hypothesis_counts.get(hypothesis, 0) + 1
            
            plane = plane_unc.sample(self.rng)
            k_spec, grf, slip = material_unc.sample(self.rng)
            
            # Compute wear volume with sampled plane
            vol_result = self._compute_volume_for_plane(
                wear_analyzer, mesh, scale, plane
            )
            
            volume = vol_result['volume']
            max_depth = vol_result['max_depth']
            mean_depth = vol_result['mean_depth']
            
            # Compute nosing ratio
            nosing = self._compute_nosing_ratio(wear_analyzer, mesh, scale, plane)
            
            # Compute traffic
            W = material_unc.body_weight * grf
            if k_spec > 0 and slip > 0 and W > 0:
                traffic = volume / (k_spec * 1e-9 * W * slip)  # k_spec in mm³/Nm → m²/N
            else:
                traffic = 0
            
            samples.append({
                'scale': scale,
                'hypothesis': hypothesis,
                'volume': volume,
                'max_depth': max_depth,
                'mean_depth': mean_depth,
                'nosing_ratio': nosing,
                'traffic': traffic,
                'k_spec': k_spec,
                'grf': grf,
                'slip': slip
            })
        
        # Convert to arrays
        volumes = np.array([s['volume'] for s in samples])
        max_depths = np.array([s['max_depth'] for s in samples])
        mean_depths = np.array([s['mean_depth'] for s in samples])
        nosing_ratios = np.array([s['nosing_ratio'] for s in samples])
        traffics = np.array([s['traffic'] for s in samples])
        
        # Compute P(ascent) via logistic function with sampled parameters
        # Sample threshold and slope from reasonable priors for mapping uncertainty
        threshold = self.rng.normal(1.0, 0.1)  # Center at 1.0, ±0.1
        slope = self.rng.uniform(4.0, 6.0)      # Slope between 4-6
        prob_ascents = 1 / (1 + np.exp(slope * (nosing_ratios - threshold)))
        
        # Sensitivity analysis via variance decomposition (all 5 sources)
        sensitivity = self._compute_sensitivity(samples)
        
        # Compute scale hypothesis posterior
        scale_posterior = {k: float(v / self.n_samples) for k, v in hypothesis_counts.items()}
        
        # Check scale ambiguity (CV > 0.5 or multiple hypotheses with significant mass)
        # Use correct axis from analyzer
        scaled_widths = [s['scale'] * mesh['bounds']['size'][wear_analyzer.width_axis] for s in samples]
        cv_scale = float(np.std(scaled_widths) / np.mean(scaled_widths)) if np.mean(scaled_widths) > 0 else 0
        significant_hypotheses = sum(1 for p in scale_posterior.values() if p > 0.1)
        scale_ambiguous = cv_scale > 0.5 or significant_hypotheses > 1
        
        # Check axis ambiguity
        axis_ambiguous = self._check_axis_ambiguity(nosing_ratios)
        
        return UncertaintyResults(
            volume=DistributionSummary.from_samples(volumes),
            max_depth=DistributionSummary.from_samples(max_depths),
            mean_depth=DistributionSummary.from_samples(mean_depths),
            prob_ascent=DistributionSummary.from_samples(prob_ascents),
            nosing_ratio=DistributionSummary.from_samples(nosing_ratios),
            axis_ambiguous=bool(axis_ambiguous),
            total_steps=DistributionSummary.from_samples(traffics),
            traffic_50y=DistributionSummary.from_samples(traffics / (50 * 365)),
            traffic_100y=DistributionSummary.from_samples(traffics / (100 * 365)),
            traffic_200y=DistributionSummary.from_samples(traffics / (200 * 365)),
            traffic_500y=DistributionSummary.from_samples(traffics / (500 * 365)),
            traffic_1000y=DistributionSummary.from_samples(traffics / (1000 * 365)),
            sensitivity=sensitivity,
            scale_ambiguous=bool(scale_ambiguous),
            scale_hypothesis_posterior=scale_posterior,
            weak_plane_fit=bool(plane_unc.weak_fit),
            traffic_dominated_by=max(sensitivity, key=sensitivity.get) if sensitivity else "unknown",
            n_samples=int(self.n_samples),
            seed=int(self.seed)
        )
    
    def _compute_volume_for_plane(self, analyzer, mesh, scale, plane_params):
        """Compute wear volume for a specific plane using triangle-based method.
        
        Note: mesh['vertices'] are ALREADY scaled by OBJParser.
        The 'scale' parameter here is the MC sample scale factor, used only for
        perturbation tracking, NOT for re-scaling vertices.
        """
        a, b, c = plane_params
        
        # Use analyzer's triangle-based method for accurate volume
        # Get tread mask (cached in analyzer if available, else compute)
        if not hasattr(analyzer, '_cached_tread_mask'):
            seg_result = analyzer.segment_tread_triangles()
            analyzer._cached_tread_mask = seg_result['tread_mask']
        
        tread_mask = analyzer._cached_tread_mask
        
        # Compute triangle-based wear volume with the sampled plane
        vol_result = self._compute_triangle_wear_volume_for_plane(
            analyzer, plane_params, tread_mask
        )
        
        return {
            'volume': vol_result['volume'],
            'max_depth': vol_result['max_depth'],
            'mean_depth': vol_result['mean_depth']
        }
    
    def _compute_triangle_wear_volume_for_plane(self, analyzer, plane_params, tread_mask):
        """Compute wear volume using triangle integration for a specific plane.
        
        This is the MC-compatible version of WearAnalyzer.compute_triangle_wear_volume.
        """
        a, b, c = plane_params
        vertices = analyzer.vertices  # Already scaled
        faces = analyzer.faces
        
        width_axis = analyzer.width_axis
        depth_axis = analyzer.depth_axis
        up_axis = analyzer.up_axis
        
        total_vol = 0.0
        max_d = 0.0
        depth_sum = 0.0
        depth_cnt = 0
        
        for i, f in enumerate(faces):
            if not tread_mask[i]:
                continue
            
            v0, v1, v2 = vertices[f[0]], vertices[f[1]], vertices[f[2]]
            
            # Predicted height from reference plane
            h0_pred = a * v0[width_axis] + b * v0[depth_axis] + c
            h1_pred = a * v1[width_axis] + b * v1[depth_axis] + c
            h2_pred = a * v2[width_axis] + b * v2[depth_axis] + c
            
            # Wear depth = predicted - actual (positive = worn)
            d0 = max(0, h0_pred - v0[up_axis])
            d1 = max(0, h1_pred - v1[up_axis])
            d2 = max(0, h2_pred - v2[up_axis])
            
            mean_d = (d0 + d1 + d2) / 3.0
            max_d = max(max_d, d0, d1, d2)
            
            if mean_d > 0:
                depth_sum += mean_d
                depth_cnt += 1
            
            # 2D projected area
            p0 = [v0[width_axis], v0[depth_axis]]
            p1 = [v1[width_axis], v1[depth_axis]]
            p2 = [v2[width_axis], v2[depth_axis]]
            area = 0.5 * abs((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1]))
            
            total_vol += mean_d * area
        
        return {
            'volume': total_vol,
            'max_depth': max_d * 1000.0,  # Convert to mm
            'mean_depth': (depth_sum / depth_cnt * 1000.0) if depth_cnt > 0 else 0.0
        }
    
    def _compute_nosing_ratio(self, analyzer, mesh, scale, plane_params):
        """Compute nosing wear ratio for directionality."""
        a, b, c = plane_params
        vertices = mesh['vertices'] * scale
        
        up_axis = analyzer.up_axis
        depth_axis = analyzer.depth_axis
        
        # Get depth axis range
        d_min = np.min(vertices[:, depth_axis])
        d_max = np.max(vertices[:, depth_axis])
        d_range = d_max - d_min
        
        # Front 25% and center 50%
        front_mask = vertices[:, depth_axis] < d_min + 0.25 * d_range
        center_mask = ((vertices[:, depth_axis] >= d_min + 0.25 * d_range) & 
                      (vertices[:, depth_axis] <= d_max - 0.25 * d_range))
        
        # Predicted heights
        predicted_z = (a * vertices[:, analyzer.width_axis] + 
                      b * vertices[:, depth_axis] + c)
        wear_depths = np.maximum(predicted_z - vertices[:, up_axis], 0)
        
        front_wear = np.mean(wear_depths[front_mask]) if np.sum(front_mask) > 0 else 0
        center_wear = np.mean(wear_depths[center_mask]) if np.sum(center_mask) > 0 else 1e-10
        
        return front_wear / center_wear
    
    def _compute_sensitivity(self, samples: List[dict]) -> Dict[str, float]:
        """Compute variance contribution from each of 5 input sources.
        
        Uses Spearman rank correlation for robustness.
        Plane sensitivity is computed empirically from residual variance.
        """
        from scipy import stats as scipy_stats
        import warnings
        
        traffic = np.array([s['traffic'] for s in samples])
        k_specs = np.array([s['k_spec'] for s in samples])
        scales = np.array([s['scale'] for s in samples])
        grfs = np.array([s['grf'] for s in samples])
        slips = np.array([s['slip'] for s in samples])
        volumes = np.array([s['volume'] for s in samples])
        
        # Use Spearman rank correlation for robustness
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            log_traffic = np.log1p(traffic)
            
            def safe_spearman(x, y):
                """Compute squared Spearman correlation, handling edge cases."""
                if np.var(x) == 0 or np.var(y) == 0:
                    return 0.0
                try:
                    rho, _ = scipy_stats.spearmanr(x, y)
                    return float(rho ** 2) if not np.isnan(rho) else 0.0
                except:
                    return 0.0
            
            r_k = safe_spearman(np.log(k_specs + 1e-20), log_traffic)
            r_s = safe_spearman(np.log(scales + 1e-20), log_traffic)
            r_g = safe_spearman(grfs, log_traffic)
            r_slip = safe_spearman(slips, log_traffic)
            
            # Plane contribution: correlation between volume variance and traffic
            # (volume varies with plane; other inputs are independent of plane)
            log_volume = np.log1p(volumes)
            r_plane = safe_spearman(log_volume, log_traffic)
            # Scale down since volume is downstream of plane
            r_plane = max(0.02, r_plane * 0.3)  # At least 2%, capped at 30% of correlation
            
            # Normalize to sum to 1
            total_r = r_k + r_s + r_g + r_slip + r_plane
            if total_r < 1e-10:
                total_r = 1.0  # Avoid division by zero
            
        return {
            'k_spec': float(r_k / total_r),
            'scale': float(r_s / total_r),
            'plane': float(r_plane / total_r),
            'grf': float(r_g / total_r),
            'slip': float(r_slip / total_r)
        }
    
    def _check_axis_ambiguity(self, nosing_ratios: np.ndarray) -> bool:
        """Check if reversing depth axis would change inference."""
        # Reciprocal simulates axis flip
        flipped_ratios = 1 / (nosing_ratios + 1e-10)
        
        # Original inference
        orig_ascent = np.median(nosing_ratios) < 0.85
        flip_ascent = np.median(flipped_ratios) < 0.85
        
        return orig_ascent != flip_ascent


# =============================================================================
# BOOTSTRAP PLANE SAMPLING
# =============================================================================

def bootstrap_plane_samples(vertices: np.ndarray,
                           inlier_mask: np.ndarray,
                           plane_params: np.ndarray,
                           width_axis: int,
                           depth_axis: int,
                           up_axis: int,
                           n_bootstrap: int = 500,
                           seed: int = 42) -> PlaneUncertainty:
    """
    Generate bootstrap samples of reference plane parameters.
    
    Args:
        vertices: Mesh vertices (N, 3)
        inlier_mask: Boolean mask of RANSAC inliers
        plane_params: Initial plane parameters (a, b, c)
        width_axis, depth_axis, up_axis: Axis indices
        n_bootstrap: Number of bootstrap samples
        seed: Random seed for reproducibility
        
    Returns:
        PlaneUncertainty with bootstrap samples
    """
    rng = np.random.default_rng(seed)
    
    inlier_vertices = vertices[inlier_mask]
    n_inliers = len(inlier_vertices)
    
    if n_inliers < 10:
        # Too few inliers: return inflated uncertainty around initial fit
        samples = np.tile(plane_params, (n_bootstrap, 1))
        # Add noise proportional to initial plane params
        noise_scale = 0.1 * np.abs(plane_params) + 1e-6
        samples += rng.normal(0, noise_scale, samples.shape)
        
        return PlaneUncertainty(
            plane_samples=samples,
            inlier_fraction=n_inliers / len(vertices),
            rms_error_mm=0.0,
            candidates_used=n_inliers
        )
    
    samples = []
    
    for _ in range(n_bootstrap):
        # Bootstrap resample inliers
        idx = rng.choice(n_inliers, n_inliers, replace=True)
        boot_vertices = inlier_vertices[idx]
        
        # Fit plane via OLS: z = a*x + b*y + c
        X = np.column_stack([
            boot_vertices[:, width_axis],
            boot_vertices[:, depth_axis],
            np.ones(len(boot_vertices))
        ])
        z = boot_vertices[:, up_axis]
        
        try:
            params, _, _, _ = np.linalg.lstsq(X, z, rcond=None)
            samples.append(params)
        except:
            samples.append(plane_params)  # Fallback
    
    samples = np.array(samples)
    
    # Compute RMS error on original inliers
    X_orig = np.column_stack([
        inlier_vertices[:, width_axis],
        inlier_vertices[:, depth_axis],
        np.ones(n_inliers)
    ])
    residuals = np.abs(X_orig @ plane_params - inlier_vertices[:, up_axis])
    rms_mm = np.sqrt(np.mean(residuals ** 2)) * 1000
    
    return PlaneUncertainty(
        plane_samples=samples,
        inlier_fraction=n_inliers / len(vertices),
        rms_error_mm=rms_mm,
        candidates_used=n_inliers
    )
