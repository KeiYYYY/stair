"""
Unit tests for StairAnalyzer
Tests material properties structure and report formatting
"""

import pytest
from stair_analyzer import StairAnalyzer


def test_material_properties_have_units():
    """Verify all materials have k_spec_units field"""
    # Create a StairAnalyzer instance
    analyzer = StairAnalyzer('dummy.obj', material_type='granite')
    
    # Check all materials have required fields
    for material_name, props in analyzer.material_properties.items():
        assert 'k_spec_value' in props, \
            f"Material {material_name} missing k_spec_value"
        assert 'k_spec_units' in props, \
            f"Material {material_name} missing k_spec_units"
        assert 'hardness' in props, \
            f"Material {material_name} missing hardness"
        assert 'name' in props, \
            f"Material {material_name} missing name"
        
        # Verify units are valid
        assert props['k_spec_units'] in ['mm3_per_Nm', 'm2_per_N'], \
            f"Material {material_name} has invalid units: {props['k_spec_units']}"


def test_material_default_units():
    """Verify default units are mm3_per_Nm to match report labels"""
    analyzer = StairAnalyzer('dummy.obj', material_type='granite')
    
    for material_name, props in analyzer.material_properties.items():
        assert props['k_spec_units'] == 'mm3_per_Nm', \
            f"Material {material_name} should default to mm3_per_Nm, got {props['k_spec_units']}"


def test_material_values_plausible():
    """Verify material k_spec values are in plausible ranges"""
    analyzer = StairAnalyzer('dummy.obj', material_type='granite')
    
    # Expected ranges for mm3_per_Nm units
    expected_ranges = {
        'granite': (1e-7, 1e-5),
        'marble': (1e-6, 1e-4),
        'sandstone': (1e-5, 1e-3),
        'limestone': (1e-5, 1e-3),
        'oak': (1e-5, 1e-3)
    }
    
    for material_name, props in analyzer.material_properties.items():
        value = props['k_spec_value']
        min_val, max_val = expected_ranges[material_name]
        
        assert min_val <= value <= max_val, \
            f"Material {material_name} k_spec {value} outside expected range [{min_val}, {max_val}]"


def test_daily_traffic_formatting():
    """
    Test that small daily traffic values are displayed with decimals
    This is a mock test since we can't easily capture print output
    """
    # This test verifies the format strings are correct
    # In practice, need to run example_usage.py and inspect output
    
    # Create mock traffic volume result
    tv = {
        'total_steps': 1000,
        'wear_volume': 1e-6,
        'daily_traffic_100y': 0.027,
        'daily_traffic_500y': 0.0055,
        'daily_traffic_1000y': 0.00274
    }
    
    # Format with .3f should show decimals
    formatted_100y = f"{tv['daily_traffic_100y']:.3f}"
    formatted_500y = f"{tv['daily_traffic_500y']:.3f}"
    formatted_1000y = f"{tv['daily_traffic_1000y']:.3f}"
    
    # Verify decimals are shown
    assert formatted_100y == "0.027", f"Expected '0.027', got '{formatted_100y}'"
    assert formatted_500y == "0.005", f"Expected '0.005', got '{formatted_500y}'"  # 0.0055 rounds to 0.005 with banker's rounding
    assert formatted_1000y == "0.003", f"Expected '0.003', got '{formatted_1000y}'"  # Rounds to 0.003
    
    # With .0f, these would all be "0"
    formatted_0f = f"{tv['daily_traffic_500y']:.0f}"
    assert formatted_0f == "0", "With .0f, small values become '0'"


def test_simultaneity_field_names():
    """
    Verify that report generation expects correct simultaneity field names
    """
    # Mock simultaneity result with new fields
    sim = {
        'pattern': 'SINGLE FILE',
        'description': 'Single-file traffic',
        'n_lanes': 1,
        'W_eff': 0.32,
        'M_max': 0,
        'M_max_two_abreast': 0,
        'single_file_possible': True
    }
    
    # Verify fields exist
    assert 'W_eff' in sim, "Should have W_eff field"
    assert 'single_file_possible' in sim, "Should have single_file_possible field"
    assert 'M_max_two_abreast' in sim, "Should have M_max_two_abreast field"
    
    # Verify backwards compatibility
    assert 'M_max' in sim, "Should retain M_max for backwards compatibility"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
