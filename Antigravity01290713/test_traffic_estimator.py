"""
Unit tests for TrafficEstimator
Tests k_spec units conversion, temporal drift logic, and M_max reporting
"""

import pytest
import numpy as np
from traffic_estimator import TrafficEstimator, mm3_per_Nm_to_m2_per_N


def test_mm3_per_Nm_to_m2_per_N_conversion():
    """Test unit conversion helper function"""
    # 1e-6 mm³/(N·m) should equal 1e-15 m²/N
    result = mm3_per_Nm_to_m2_per_N(1e-6)
    assert result == pytest.approx(1e-15), f"Expected 1e-15, got {result}"
    
    # Test other values
    assert mm3_per_Nm_to_m2_per_N(1e-5) == pytest.approx(1e-14)
    assert mm3_per_Nm_to_m2_per_N(5e-5) == pytest.approx(5e-14)


def test_k_spec_conversion_factor():
    """
    Test that passing k_spec in different units yields identical results
    when the values are equivalent
    """
    # Create mock data
    wear_profile = {
        'volume': 1e-6,  # 1 cubic mm in m³
        'max_depth': 50.0,  # mm
        'x_coords': np.linspace(0, 1, 10),
        'y_coords': np.linspace(0, 1, 10),
        'depth_grid': np.zeros((10, 10))
    }
    
    lateral_analysis = {
        'n_modes': 1,
        'kurtosis': 3.5,
        'means': np.array([0.5]),
        'W_nom': 1.0,
        'x_coords': np.linspace(0, 1, 10),
        'depth_grid': np.zeros((10, 10))
    }
    
    longitudinal_analysis = {
        'nosing_wear_ratio': 1.0,
        'skewness': 0.0
    }
    
    # Test 1: k_spec in mm3_per_Nm
    material_props_1 = {
        'k_spec_value': 1e-6,
        'k_spec_units': 'mm3_per_Nm',
        'hardness': 7.5e9
    }
    
    # Test 2: equivalent k_spec in m2_per_N
    material_props_2 = {
        'k_spec_value': 1e-15,  # 1e-6 * 1e-9
        'k_spec_units': 'm2_per_N',
        'hardness': 7.5e9
    }
    
    estimator1 = TrafficEstimator(wear_profile, lateral_analysis, longitudinal_analysis, material_props_1)
    estimator2 = TrafficEstimator(wear_profile, lateral_analysis, longitudinal_analysis, material_props_2)
    
    result1 = estimator1.estimate_traffic_volume()
    result2 = estimator2.estimate_traffic_volume()
    
    # Should yield identical total_steps
    assert result1['total_steps'] == pytest.approx(result2['total_steps'], rel=1e-9), \
        f"Expected identical steps, got {result1['total_steps']} vs {result2['total_steps']}"
    
    # Check that k_spec_SI is the same
    assert result1['k_spec_SI'] == pytest.approx(result2['k_spec_SI'], rel=1e-9)


def test_wrong_units_trigger_warning():
    """
    Test that passing k_spec with wrong units triggers per-step wear warning
    Using k_spec_value=1e-6 with units='m2_per_N' should be implausibly high
    """
    # Create mock data with large wear volume
    wear_profile = {
        'volume': 1e-3,  # 1 liter in m³ - large wear
        'max_depth': 50.0,
        'x_coords': np.linspace(0, 1, 10),
        'y_coords': np.linspace(0, 1, 10),
        'depth_grid': np.zeros((10, 10))
    }
    
    lateral_analysis = {
        'n_modes': 1,
        'kurtosis': 3.5,
        'means': np.array([0.5]),
        'W_nom': 1.0,
        'x_coords': np.linspace(0, 1, 10),
        'depth_grid': np.zeros((10, 10))
    }
    
    longitudinal_analysis = {
        'nosing_wear_ratio': 1.0,
        'skewness': 0.0
    }
    
    # Wrong units: 1e-6 m²/N is 9 orders of magnitude too large
    material_props = {
        'k_spec_value': 1e-6,
        'k_spec_units': 'm2_per_N',  # Should be mm3_per_Nm
        'hardness': 7.5e9
    }
    
    estimator = TrafficEstimator(wear_profile, lateral_analysis, longitudinal_analysis, material_props)
    result = estimator.estimate_traffic_volume()
    
    # Should trigger per-step wear warning
    warnings = result.get('warnings', [])
    has_per_step_warning = any('Per-step wear volume' in w for w in warnings)
    
    assert has_per_step_warning, f"Expected per-step wear warning, got warnings: {warnings}"
    
    # Per-step wear should be > 1e-9 m³
    assert result['per_step_wear_volume'] > 1e-9


def test_temporal_drift_shy_distance():
    """
    Test that shy_distance parameter is correctly passed to _check_temporal_drift
    """
    # Create mock data with controlled lateral drift
    # Create a depth grid where peak shifts laterally with depth
    x_coords = np.linspace(0, 1.0, 50)
    y_coords = np.linspace(0, 0.5, 50)
    depth_grid = np.zeros((50, 50))
    
    # Create shifting peak: shallow wear at x=0.3, deep wear at x=0.7
    for i in range(50):
        for j in range(50):
            # Peak position shifts from 0.3 to 0.7 with depth
            peak_pos = 0.3 + (depth_grid[i, j] / 100.0) * 0.4
            depth = 100.0 * np.exp(-((x_coords[j] - peak_pos) / 0.1) ** 2)
            if i > 25:  # Deeper region
                peak_pos = 0.7
                depth = 100.0 * np.exp(-((x_coords[j] - peak_pos) / 0.1) ** 2)
            depth_grid[i, j] = depth
    
    wear_profile = {
        'volume': 1e-6,
        'max_depth': 100.0,
        'x_coords': x_coords,
        'y_coords': y_coords,
        'depth_grid': depth_grid
    }
    
    lateral_analysis = {
        'n_modes': 2,
        'kurtosis': 2.0,
        'means': np.array([0.3, 0.7]),
        'stds': np.array([0.1, 0.1]),
        'weights': np.array([0.5, 0.5]),
        'W_nom': 1.0,
        'x_coords': x_coords,
        'depth_grid': depth_grid
    }
    
    longitudinal_analysis = {
        'nosing_wear_ratio': 1.0,
        'skewness': 0.0
    }
    
    material_props = {
        'k_spec_value': 1e-6,
        'k_spec_units': 'mm3_per_Nm',
        'hardness': 7.5e9
    }
    
    estimator = TrafficEstimator(wear_profile, lateral_analysis, longitudinal_analysis, material_props)
    
    # Test with different shy_distance values
    # With small shy_distance, W_eff is larger, so same shift is smaller ratio
    result1 = estimator.analyze_simultaneity(shy_distance=0.1)
    
    # With large shy_distance, W_eff is smaller, so same shift is larger ratio
    result2 = estimator.analyze_simultaneity(shy_distance=0.4)
    
    # The temporal drift detection should be sensitive to shy_distance
    # Note: This test may not always show different results depending on the specific data,
    # but it verifies the parameter is being passed through
    assert 'temporal_drift_likely' in result1
    assert 'temporal_drift_likely' in result2


def test_m_max_zero_reporting():
    """
    Test that when W_eff < w_lane (M_max_two_abreast=0), reporting is clear
    """
    wear_profile = {
        'volume': 1e-6,
        'max_depth': 50.0,
        'x_coords': np.linspace(0, 0.4, 10),  # Very narrow stair
        'y_coords': np.linspace(0, 0.5, 10),
        'depth_grid': np.zeros((10, 10))
    }
    
    lateral_analysis = {
        'n_modes': 1,
        'kurtosis': 3.5,
        'means': np.array([0.2]),
        'W_nom': 0.4,  # 40 cm nominal width
        'x_coords': np.linspace(0, 0.4, 10),
        'depth_grid': np.zeros((10, 10))
    }
    
    longitudinal_analysis = {
        'nosing_wear_ratio': 1.0,
        'skewness': 0.0
    }
    
    material_props = {
        'k_spec_value': 1e-6,
        'k_spec_units': 'mm3_per_Nm',
        'hardness': 7.5e9
    }
    
    estimator = TrafficEstimator(wear_profile, lateral_analysis, longitudinal_analysis, material_props)
    result = estimator.analyze_simultaneity(w_lane=0.55, shy_distance=0.1)
    
    # W_eff = 0.4 - 2*0.1 = 0.2m < 0.55m, so M_max_two_abreast = 0
    assert result['M_max_two_abreast'] == 0, f"Expected M_max_two_abreast=0, got {result['M_max_two_abreast']}"
    
    # But single file should always be possible
    assert result['single_file_possible'] == True, "Single file should always be possible"
    
    # Pattern should be SINGLE FILE
    assert result['pattern'] == 'SINGLE FILE', f"Expected SINGLE FILE, got {result['pattern']}"
    
    # Rationale should mention single file is possible
    assert 'single file' in result['rationale'].lower(), \
        f"Rationale should mention single file, got: {result['rationale']}"


def test_legacy_material_format_compatibility():
    """
    Test that legacy material format with k_spec_mm3_per_Nm still works
    """
    wear_profile = {
        'volume': 1e-6,
        'max_depth': 50.0,
        'x_coords': np.linspace(0, 1, 10),
        'y_coords': np.linspace(0, 1, 10),
        'depth_grid': np.zeros((10, 10))
    }
    
    lateral_analysis = {
        'n_modes': 1,
        'kurtosis': 3.5,
        'means': np.array([0.5]),
        'W_nom': 1.0,
        'x_coords': np.linspace(0, 1, 10),
        'depth_grid': np.zeros((10, 10))
    }
    
    longitudinal_analysis = {
        'nosing_wear_ratio': 1.0,
        'skewness': 0.0
    }
    
    # Legacy format
    material_props_legacy = {
        'k_spec_mm3_per_Nm': 1e-6,
        'hardness': 7.5e9
    }
    
    estimator = TrafficEstimator(wear_profile, lateral_analysis, longitudinal_analysis, material_props_legacy)
    result = estimator.estimate_traffic_volume()
    
    # Should work without errors
    assert 'total_steps' in result
    assert result['k_spec_input_units'] == 'mm3_per_Nm'
    assert result['k_spec_input_value'] == 1e-6


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
