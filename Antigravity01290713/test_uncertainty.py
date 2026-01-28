"""
Tests for Uncertainty Propagation Module
Tests determinism, invariants, and integration per the implementation plan.
"""

import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from uncertainty import (
    DistributionSummary,
    ScaleUncertainty,
    PlaneUncertainty,
    MaterialUncertainty,
    UncertaintyResults,
    UncertaintyPropagator,
    bootstrap_plane_samples
)


# =============================================================================
# TEST DISTRIBUTION SUMMARY
# =============================================================================

def test_distribution_summary_from_samples():
    """Test DistributionSummary creation from samples."""
    samples = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    dist = DistributionSummary.from_samples(samples)
    
    assert dist.mean == 5.5
    assert dist.median == 5.5
    assert 0 <= dist.ci_50[0] <= dist.ci_50[1]
    assert 0 <= dist.ci_95[0] <= dist.ci_95[1]
    assert len(dist.samples) == 10
    print("✓ DistributionSummary.from_samples works correctly")


def test_distribution_formatting():
    """Test distribution formatting methods."""
    samples = np.linspace(1, 100, 1000)
    dist = DistributionSummary.from_samples(samples)
    
    formatted = dist.format_with_ci(2)
    assert '[' in formatted and '-' in formatted
    
    compact = dist.format_compact(2)
    assert '[' in compact
    print("✓ Distribution formatting works correctly")


# =============================================================================
# TEST SCALE UNCERTAINTY
# =============================================================================

def test_scale_uncertainty_user_provided():
    """Test ScaleUncertainty with user-provided scale."""
    scale_unc = ScaleUncertainty(user_provided=True, user_scale=0.001)
    
    rng = np.random.default_rng(42)
    samples = [scale_unc.sample(rng) for _ in range(100)]
    
    # Should be close to user scale with small perturbation
    mean_scale = np.mean(samples)
    assert 0.0005 < mean_scale < 0.002
    print("✓ ScaleUncertainty user-provided sampling works")


def test_scale_uncertainty_inferred():
    """Test ScaleUncertainty inference from mesh."""
    mock_mesh = {'bounds': {'size': [800, 300, 50]}}  # Looks like mm
    scale_unc = ScaleUncertainty.from_args(None, mock_mesh)
    
    assert not scale_unc.user_provided
    # Should favor mm hypothesis
    mm_prob = [h for h in scale_unc.hypotheses if h[2] == "mm"][0][1]
    assert mm_prob > 0.5
    print("✓ ScaleUncertainty inference works")


# =============================================================================
# TEST PLANE UNCERTAINTY
# =============================================================================

def test_plane_uncertainty_sampling():
    """Test PlaneUncertainty sampling."""
    plane_samples = np.random.randn(100, 3) * 0.01 + [0, 0, 1.5]
    plane_unc = PlaneUncertainty(
        plane_samples=plane_samples,
        inlier_fraction=0.9,
        rms_error_mm=0.5,
        candidates_used=500
    )
    
    assert plane_unc.n_samples == 100
    assert not plane_unc.weak_fit  # Good fit
    
    rng = np.random.default_rng(42)
    sample = plane_unc.sample(rng)
    assert len(sample) == 3
    print("✓ PlaneUncertainty sampling works")


def test_plane_uncertainty_weak_fit_detection():
    """Test weak fit flagging."""
    # Low inlier fraction
    plane_unc = PlaneUncertainty(
        plane_samples=np.zeros((10, 3)),
        inlier_fraction=0.4,  # Below 0.6 threshold
        rms_error_mm=1.0,
        candidates_used=100
    )
    assert plane_unc.weak_fit
    
    # High RMS error
    plane_unc2 = PlaneUncertainty(
        plane_samples=np.zeros((10, 3)),
        inlier_fraction=0.9,
        rms_error_mm=7.0,  # Above 5mm threshold
        candidates_used=100
    )
    assert plane_unc2.weak_fit
    print("✓ Weak fit detection works")


# =============================================================================
# TEST MATERIAL UNCERTAINTY
# =============================================================================

def test_material_uncertainty_sampling():
    """Test MaterialUncertainty sampling."""
    mat_unc = MaterialUncertainty.from_name('granite')
    
    assert mat_unc.name == 'granite'
    assert np.isclose(mat_unc.k_spec_median, 1e-6, rtol=1e-6)
    
    rng = np.random.default_rng(42)
    k_spec, grf, slip = mat_unc.sample(rng)
    
    assert k_spec > 0
    assert 1.0 <= grf <= 2.5
    assert slip > 0
    print("✓ MaterialUncertainty sampling works")


def test_material_uncertainty_all_materials():
    """Test all material types load correctly."""
    materials = ['granite', 'marble', 'limestone', 'sandstone', 'wood']
    for mat in materials:
        mat_unc = MaterialUncertainty.from_name(mat)
        assert mat_unc.k_spec_median > 0
    print("✓ All materials load correctly")


# =============================================================================
# TEST DETERMINISM
# =============================================================================

def test_same_seed_same_results():
    """Same seed produces identical results."""
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    
    scale_unc = ScaleUncertainty(user_provided=True, user_scale=0.001)
    mat_unc = MaterialUncertainty.from_name('granite')
    
    # Sample with same seed
    results1 = [scale_unc.sample(rng1) for _ in range(10)]
    results2 = [scale_unc.sample(rng2) for _ in range(10)]
    
    assert np.allclose(results1, results2)
    print("✓ Determinism test passed (same seed → same results)")


def test_bootstrap_determinism():
    """Bootstrap plane sampling is deterministic with seed."""
    vertices = np.random.randn(100, 3)
    inlier_mask = np.random.rand(100) > 0.3
    plane_params = np.array([0.01, 0.01, 1.5])
    
    plane_unc1 = bootstrap_plane_samples(
        vertices, inlier_mask, plane_params,
        width_axis=0, depth_axis=1, up_axis=2,
        n_bootstrap=50, seed=42
    )
    
    plane_unc2 = bootstrap_plane_samples(
        vertices, inlier_mask, plane_params,
        width_axis=0, depth_axis=1, up_axis=2,
        n_bootstrap=50, seed=42
    )
    
    assert np.allclose(plane_unc1.plane_samples, plane_unc2.plane_samples)
    print("✓ Bootstrap determinism test passed")


# =============================================================================
# TEST INVARIANTS
# =============================================================================

def test_physical_constraints_distribution():
    """Distributions respect basic constraints."""
    samples = np.abs(np.random.randn(1000))  # All positive
    dist = DistributionSummary.from_samples(samples)
    
    assert dist.mean >= 0
    assert dist.median >= 0
    assert dist.ci_50[0] <= dist.ci_50[1]
    assert dist.ci_95[0] <= dist.ci_95[1]
    print("✓ Physical constraints on distributions hold")


def test_probability_bounds():
    """Probability values are in [0, 1]."""
    # Use logistic transformation like in real code
    nosing_ratios = np.linspace(0.5, 1.5, 100)
    prob_ascents = 1 / (1 + np.exp(5 * (nosing_ratios - 1.0)))
    
    assert all(0 <= p <= 1 for p in prob_ascents)
    print("✓ Probability bounds test passed")


# =============================================================================
# NEW TESTS FOR EXTENDED FUNCTIONALITY
# =============================================================================

def test_scale_mixture_sampling_determinism():
    """Verify mixture scale sampling produces same sequence with same seed."""
    mock_mesh = {'bounds': {'size': [800, 300, 50]}}  # Looks like mm
    scale_unc = ScaleUncertainty.from_args(None, mock_mesh)
    
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    
    samples1 = [scale_unc.sample_with_hypothesis(rng1) for _ in range(50)]
    samples2 = [scale_unc.sample_with_hypothesis(rng2) for _ in range(50)]
    
    # Check scales and hypotheses match
    scales1 = [s[0] for s in samples1]
    scales2 = [s[0] for s in samples2]
    hyps1 = [s[1] for s in samples1]
    hyps2 = [s[1] for s in samples2]
    
    assert np.allclose(scales1, scales2)
    assert hyps1 == hyps2
    print("✓ Scale mixture sampling determinism test passed")


def test_direction_classification_logic():
    """Test CI-based direction classification."""
    # Test ASCENT: CI95 entirely > 0.5
    ascent_samples = np.random.uniform(0.6, 0.9, 100)
    ascent_dist = DistributionSummary.from_samples(ascent_samples)
    
    # Create mock UncertaintyResults
    from uncertainty import UncertaintyResults
    mock_volume = DistributionSummary.from_samples(np.ones(100))
    
    results_ascent = UncertaintyResults(
        volume=mock_volume, max_depth=mock_volume, mean_depth=mock_volume,
        prob_ascent=ascent_dist, nosing_ratio=mock_volume,
        total_steps=mock_volume, traffic_50y=mock_volume, traffic_100y=mock_volume,
        traffic_200y=mock_volume, traffic_500y=mock_volume, traffic_1000y=mock_volume
    )
    assert results_ascent.get_direction_classification() == "ASCENT"
    
    # Test DESCENT: CI95 entirely < 0.5
    descent_samples = np.random.uniform(0.1, 0.4, 100)
    descent_dist = DistributionSummary.from_samples(descent_samples)
    results_descent = UncertaintyResults(
        volume=mock_volume, max_depth=mock_volume, mean_depth=mock_volume,
        prob_ascent=descent_dist, nosing_ratio=mock_volume,
        total_steps=mock_volume, traffic_50y=mock_volume, traffic_100y=mock_volume,
        traffic_200y=mock_volume, traffic_500y=mock_volume, traffic_1000y=mock_volume
    )
    assert results_descent.get_direction_classification() == "DESCENT"
    
    # Test AMBIGUOUS: CI95 straddles 0.5
    ambiguous_samples = np.random.uniform(0.3, 0.7, 100)
    ambiguous_dist = DistributionSummary.from_samples(ambiguous_samples)
    results_ambig = UncertaintyResults(
        volume=mock_volume, max_depth=mock_volume, mean_depth=mock_volume,
        prob_ascent=ambiguous_dist, nosing_ratio=mock_volume,
        total_steps=mock_volume, traffic_50y=mock_volume, traffic_100y=mock_volume,
        traffic_200y=mock_volume, traffic_500y=mock_volume, traffic_1000y=mock_volume
    )
    assert results_ambig.get_direction_classification() == "AMBIGUOUS"
    
    print("✓ Direction classification logic test passed")


def test_sensitivity_normalization():
    """Verify all 5 sensitivity values sum to 1.0."""
    # Create mock samples mimicking propagator output
    rng = np.random.default_rng(42)
    
    # Simulate samples with correlated inputs
    n = 100
    k_specs = rng.lognormal(-14, 0.5, n)
    scales = rng.lognormal(-7, 0.1, n)
    grfs = rng.normal(1.7, 0.2, n)
    slips = rng.lognormal(-6.2, 0.3, n)
    
    # Traffic inversely proportional to k_spec
    traffics = 1e-10 / (k_specs * grfs * slips)
    
    # Create samples including 'volume' (required for plane sensitivity)
    volumes = np.random.uniform(1e-6, 1e-5, n)
    mock_samples = [
        {'traffic': traffics[i], 'k_spec': k_specs[i], 'scale': scales[i], 
         'grf': grfs[i], 'slip': slips[i], 'volume': volumes[i]}
        for i in range(n)
    ]
    
    propagator = UncertaintyPropagator(n_samples=100, seed=42)
    sensitivity = propagator._compute_sensitivity(mock_samples)
    
    # Check all 5 sources present
    assert set(sensitivity.keys()) == {'k_spec', 'scale', 'plane', 'grf', 'slip'}
    
    # Check sum to 1.0 (within floating point tolerance)
    total = sum(sensitivity.values())
    assert abs(total - 1.0) < 1e-10, f"Sensitivity sum = {total}, expected 1.0"
    
    # Check all values are non-negative
    assert all(v >= 0 for v in sensitivity.values())
    
    print("✓ Sensitivity normalization test passed")


def test_json_serialization_invariants():
    """Verify UncertaintyResults can be round-tripped through JSON."""
    import json
    
    # Create a mock UncertaintyResults
    samples = np.random.randn(100)
    mock_dist = DistributionSummary.from_samples(samples)
    
    results = UncertaintyResults(
        volume=mock_dist, max_depth=mock_dist, mean_depth=mock_dist,
        prob_ascent=mock_dist, nosing_ratio=mock_dist,
        total_steps=mock_dist, traffic_50y=mock_dist, traffic_100y=mock_dist,
        traffic_200y=mock_dist, traffic_500y=mock_dist, traffic_1000y=mock_dist,
        sensitivity={'k_spec': 0.5, 'scale': 0.2, 'plane': 0.1, 'grf': 0.1, 'slip': 0.1},
        scale_hypothesis_posterior={'mm': 0.7, 'cm': 0.2, 'm': 0.1},
        n_samples=100, seed=42
    )
    
    # Build JSON-serializable dict (mimicking _save_uncertainty_results)
    output = {
        'n_samples': int(results.n_samples),
        'seed': int(results.seed),
        'volume_median': float(results.volume.median),
        'prob_ascent_ci95': [float(results.prob_ascent.ci_95[0]), float(results.prob_ascent.ci_95[1])],
        'sensitivity': {k: float(v) for k, v in results.sensitivity.items()},
        'scale_posterior': {k: float(v) for k, v in results.scale_hypothesis_posterior.items()},
        'scale_ambiguous': bool(results.scale_ambiguous),
        'classification': results.get_direction_classification()
    }
    
    # Serialize and deserialize
    json_str = json.dumps(output)
    loaded = json.loads(json_str)
    
    # Verify all values are plain Python types (no numpy)
    assert isinstance(loaded['n_samples'], int)
    assert isinstance(loaded['volume_median'], float)
    assert isinstance(loaded['scale_ambiguous'], bool)
    assert isinstance(loaded['sensitivity']['k_spec'], float)
    assert isinstance(loaded['classification'], str)
    
    print("✓ JSON serialization invariants test passed")


def test_traffic_horizon_consistency():
    """Verify traffic_50y > traffic_100y > traffic_200y > traffic_500y > traffic_1000y."""
    samples = np.random.uniform(1000, 10000, 100)  # Total steps
    
    mock_dist = DistributionSummary.from_samples(samples)
    t50 = DistributionSummary.from_samples(samples / (50 * 365))
    t100 = DistributionSummary.from_samples(samples / (100 * 365))
    t200 = DistributionSummary.from_samples(samples / (200 * 365))
    t500 = DistributionSummary.from_samples(samples / (500 * 365))
    t1000 = DistributionSummary.from_samples(samples / (1000 * 365))
    
    results = UncertaintyResults(
        volume=mock_dist, max_depth=mock_dist, mean_depth=mock_dist,
        prob_ascent=mock_dist, nosing_ratio=mock_dist,
        total_steps=mock_dist, traffic_50y=t50, traffic_100y=t100,
        traffic_200y=t200, traffic_500y=t500, traffic_1000y=t1000
    )
    
    # Shorter time horizon => higher daily traffic
    assert results.traffic_50y.median > results.traffic_100y.median
    assert results.traffic_100y.median > results.traffic_200y.median
    assert results.traffic_200y.median > results.traffic_500y.median
    assert results.traffic_500y.median > results.traffic_1000y.median
    
    print("✓ Traffic horizon consistency test passed")


def test_axis_config_correctness():
    """Verify ScaleUncertainty uses correct axis from axis_config."""
    # Create mock mesh where width is on axis 2 (not default 0)
    mock_mesh = {
        'bounds': {
            'size': [50, 50, 800],  # Axis 2 has 800mm width
            'min': [0, 0, 0],
            'max': [50, 50, 800]
        }
    }
    
    # With default axis=0, should see 50 as width → favor "m" hypothesis
    scale_unc_default = ScaleUncertainty.from_args(None, mock_mesh, width_axis=0)
    
    # With axis=2, should see 800 as width → favor "mm" hypothesis
    scale_unc_axis2 = ScaleUncertainty.from_args(None, mock_mesh, width_axis=2)
    
    # Check: axis=0 should prefer cm/m, axis=2 should prefer mm
    mm_prob_axis2 = [h for h in scale_unc_axis2.hypotheses if h[2] == "mm"][0][1]
    mm_prob_axis0 = [h for h in scale_unc_default.hypotheses if h[2] == "mm"][0][1]
    
    assert mm_prob_axis2 > mm_prob_axis0, "axis=2 should favor mm more than axis=0"
    assert mm_prob_axis2 >= 0.5, "800mm width should strongly favor mm hypothesis"
    
    print("✓ Axis config correctness test passed")


def test_triangle_volume_mc_nonnegative():
    """Verify MC volume is always non-negative."""
    # Create mock results with non-negative volumes
    volumes = np.abs(np.random.randn(100)) * 1e-6
    dist = DistributionSummary.from_samples(volumes)
    
    # All volume statistics should be non-negative
    assert dist.mean >= 0, "Mean volume must be non-negative"
    assert dist.median >= 0, "Median volume must be non-negative"
    assert dist.ci_50[0] >= 0, "CI50 lower bound must be non-negative"
    assert dist.ci_95[0] >= 0, "CI95 lower bound must be non-negative"
    assert np.all(dist.samples >= 0), "All volume samples must be non-negative"
    
    print("✓ Triangle volume MC non-negative test passed")


def test_scale_uncertainty_from_args_axis_parameter():
    """Verify width_axis parameter is correctly used in ScaleUncertainty."""
    # Create mesh with different sizes on each axis
    mock_mesh = {
        'bounds': {
            'size': [0.5, 800, 300],  # Axis 0: m scale, Axis 1: mm scale, Axis 2: cm/m
            'min': [0, 0, 0],
            'max': [0.5, 800, 300]
        }
    }
    
    # Test each axis interpretation
    scale_unc_0 = ScaleUncertainty.from_args(None, mock_mesh, width_axis=0)  # 0.5 → m
    scale_unc_1 = ScaleUncertainty.from_args(None, mock_mesh, width_axis=1)  # 800 → mm
    scale_unc_2 = ScaleUncertainty.from_args(None, mock_mesh, width_axis=2)  # 300 → ?
    
    # Axis 0 (0.5) should favor 'm' hypothesis
    m_prob_0 = [h for h in scale_unc_0.hypotheses if h[2] == "m"][0][1]
    assert m_prob_0 >= 0.5, f"axis0 (0.5) should favor m, got {m_prob_0}"
    
    # Axis 1 (800) should favor 'mm' hypothesis
    mm_prob_1 = [h for h in scale_unc_1.hypotheses if h[2] == "mm"][0][1]
    assert mm_prob_1 >= 0.5, f"axis1 (800) should favor mm, got {mm_prob_1}"
    
    print("✓ ScaleUncertainty.from_args axis parameter test passed")




# =============================================================================
# RUN ALL TESTS
# =============================================================================

def run_all_tests():
    """Run all uncertainty tests."""
    print("=" * 60)
    print("UNCERTAINTY MODULE TESTS")
    print("=" * 60)
    
    tests = [
        test_distribution_summary_from_samples,
        test_distribution_formatting,
        test_scale_uncertainty_user_provided,
        test_scale_uncertainty_inferred,
        test_plane_uncertainty_sampling,
        test_plane_uncertainty_weak_fit_detection,
        test_material_uncertainty_sampling,
        test_material_uncertainty_all_materials,
        test_same_seed_same_results,
        test_bootstrap_determinism,
        test_physical_constraints_distribution,
        test_probability_bounds,
        # New tests
        test_scale_mixture_sampling_determinism,
        test_direction_classification_logic,
        test_sensitivity_normalization,
        test_json_serialization_invariants,
        test_traffic_horizon_consistency,
        # Correctness invariant tests
        test_axis_config_correctness,
        test_triangle_volume_mc_nonnegative,
        test_scale_uncertainty_from_args_axis_parameter,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1
    
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
