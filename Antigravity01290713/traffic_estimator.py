"""
Traffic Pattern Estimator
Estimates traffic volume, directionality, and simultaneity patterns
Implements Archard's Law and algorithms from ModelG.md
"""

import numpy as np


def mm3_per_Nm_to_m2_per_N(k_spec_mm3_per_Nm):
    """
    Convert k_spec from mm³/(N·m) to m²/N (SI units)
    
    Args:
        k_spec_mm3_per_Nm: k_spec value in mm³/(N·m)
    
    Returns:
        k_spec in m²/N
    """
    return k_spec_mm3_per_Nm * 1e-9


class TrafficEstimator:
    """Estimates traffic patterns from wear analysis results"""

    def __init__(self, wear_profile, lateral_analysis, longitudinal_analysis, material_props):
        """
        Initialize traffic estimator

        Args:
            wear_profile: Dictionary from WearAnalyzer.extract_wear_profile()
            lateral_analysis: Dictionary from WearAnalyzer.analyze_lateral_distribution()
            longitudinal_analysis: Dictionary from WearAnalyzer.analyze_longitudinal_profile()
            material_props: Dictionary with 'k_spec_value', 'k_spec_units' ('mm3_per_Nm' or 'm2_per_N'), 
                           and 'hardness' (Pa). Legacy format with 'k_spec_mm3_per_Nm' is also supported.
        """
        self.wear_profile = wear_profile
        self.lateral = lateral_analysis
        self.longitudinal = longitudinal_analysis
        
        # Handle legacy format for backwards compatibility
        if 'k_spec_mm3_per_Nm' in material_props:
            # Legacy format: assume units are mm3_per_Nm
            self.k_spec_value = material_props['k_spec_mm3_per_Nm']
            self.k_spec_units = 'mm3_per_Nm'
        else:
            # New format with explicit units
            self.k_spec_value = material_props.get('k_spec_value', 1e-6)
            self.k_spec_units = material_props.get('k_spec_units', 'mm3_per_Nm')
        
        self.material = material_props

        # Biomechanical constants (from ModelG.md Section 3.1)
        self.avg_body_weight = 700  # Newtons (70 kg person)
        self.grf_multiplier_ascent = 1.2  # Peak GRF during ascent
        self.grf_multiplier_descent = 1.7  # Peak GRF during descent
        self.microslip_distance = 0.002  # 2mm per step (typical)

    def estimate_traffic_volume(self):
        """
        QUESTION 1: How often were the stairs used?
        Uses Archard's Law: V = k_spec * W * L * N
        From ModelG.md Section 2.1

        Returns:
            dict: Contains 'total_steps', 'daily_traffic' estimates, warnings
        """
        # Get wear volume (m³)
        wear_volume = self.wear_profile['volume']

        # Convert k_spec to SI units (m²/N) based on input units
        if self.k_spec_units == 'mm3_per_Nm':
            k_spec_SI = mm3_per_Nm_to_m2_per_N(self.k_spec_value)
        elif self.k_spec_units == 'm2_per_N':
            k_spec_SI = self.k_spec_value
        else:
            raise ValueError(f"Unknown k_spec units: {self.k_spec_units}. Expected 'mm3_per_Nm' or 'm2_per_N'")

        # Archard's Law parameters
        W = self.avg_body_weight * self.grf_multiplier_descent  # Use descent (higher force), N
        L = self.microslip_distance  # Sliding distance per step, m

        # Calculate total number of steps
        # V = k_spec * W * L * N
        # N = V / (k_spec * W * L)
        if k_spec_SI > 0 and W > 0 and L > 0:
            total_steps = wear_volume / (k_spec_SI * W * L)
        else:
            total_steps = 0

        # Estimate daily traffic for different age scenarios
        # Assume 365 days/year
        daily_traffic_100y = total_steps / (100 * 365) if total_steps > 0 else 0
        daily_traffic_500y = total_steps / (500 * 365) if total_steps > 0 else 0
        daily_traffic_1000y = total_steps / (1000 * 365) if total_steps > 0 else 0

        # Calculate per-step wear volume for sanity check
        per_step_wear_volume = wear_volume / total_steps if total_steps > 0 else 0
        
        # Sanity warnings
        warnings = []
        max_depth_mm = self.wear_profile.get('max_depth', 0)

        if max_depth_mm > 200:
            warnings.append("Max wear depth > 200mm: likely axis/scale/segmentation issue")

        # Check for implausibly high per-step wear (> 1 cubic mm = 1e-9 m³)
        if per_step_wear_volume > 1e-9:
            warnings.append(f"Per-step wear volume {per_step_wear_volume:.2e} m³ is implausibly high for stone - check k_spec units (expected mm3_per_Nm but got m2_per_N?)")

        if daily_traffic_500y > 50000:
            warnings.append(f"Extremely high traffic ({daily_traffic_500y:.0f} footsteps/day): check k_spec and scale")

        if daily_traffic_500y < 0.1 and total_steps > 0:
            warnings.append(f"Extremely low traffic ({daily_traffic_500y:.3f} footsteps/day): likely k_spec units mismatch or wear volume under-estimated")

        return {
            'total_steps': total_steps,
            'wear_volume': wear_volume,  # m³
            'wear_volume_units': 'm³',
            'per_step_wear_volume': per_step_wear_volume,  # m³
            'daily_traffic_100y': daily_traffic_100y,
            'daily_traffic_500y': daily_traffic_500y,
            'daily_traffic_1000y': daily_traffic_1000y,
            'k_spec_SI': k_spec_SI,  # m²/N (Pa^-1)
            'k_spec_SI_units': 'm²/N',
            'k_spec_input_value': self.k_spec_value,
            'k_spec_input_units': self.k_spec_units,
            'assumptions': {
                'avg_weight_N': self.avg_body_weight,
                'grf_multiplier': self.grf_multiplier_descent,
                'microslip_m': self.microslip_distance
            },
            'warnings': warnings
        }

    def analyze_directionality(self):
        """
        QUESTION 2: Was a certain direction of travel favored?
        Uses nosing wear ratio and longitudinal skewness
        From ModelG.md Section 3.1

        Returns:
            dict: Contains 'dominant_direction', 'confidence', 'ratio'
        """
        nosing_ratio = self.longitudinal['nosing_wear_ratio']
        skewness = self.longitudinal['skewness']

        # Interpretation thresholds (from ModelG.md)
        # High nosing wear (>1.2) indicates descent dominance
        # Low nosing wear (<0.8) indicates ascent dominance
        # Positive skewness = front-heavy (descent)
        # Negative skewness = back-heavy (ascent)

        if nosing_ratio > 1.3:
            dominant_direction = "DESCENT"
            confidence = min(95, 50 + (nosing_ratio - 1.0) * 50)
            explanation = "High nosing wear indicates heavy descent traffic (braking forces at edge)"
        elif nosing_ratio < 0.7:
            dominant_direction = "ASCENT"
            confidence = min(95, 50 + (1.0 - nosing_ratio) * 50)
            explanation = "Low nosing wear with center concentration indicates ascent dominance"
        elif skewness > 0.5:
            dominant_direction = "DESCENT (moderate)"
            confidence = 60 + skewness * 20
            explanation = "Positive skewness suggests front-heavy wear pattern (descent bias)"
        elif skewness < -0.5:
            dominant_direction = "ASCENT (moderate)"
            confidence = 60 + abs(skewness) * 20
            explanation = "Negative skewness suggests back-heavy wear pattern (ascent bias)"
        else:
            dominant_direction = "BIDIRECTIONAL (balanced)"
            confidence = 70
            explanation = "Balanced wear pattern indicates equal ascent and descent traffic"

        # Calculate ascent/descent ratio estimate
        # Based on nosing wear ratio
        if nosing_ratio > 1.0:
            ascent_descent_ratio = 1.0 / nosing_ratio
        else:
            ascent_descent_ratio = 1.0 / (nosing_ratio + 0.1)

        return {
            'dominant_direction': dominant_direction,
            'confidence': confidence,
            'nosing_wear_ratio': nosing_ratio,
            'skewness': skewness,
            'ascent_descent_ratio': ascent_descent_ratio,
            'explanation': explanation
        }

    def analyze_simultaneity(self, w_lane=0.55, shy_distance=0.2):
        """
        QUESTION 3: How many people used the stairs simultaneously?
        Uses physics-constrained interpretation of GMM results
        From ModelG.md Section 4.1 and 5.2

        Args:
            w_lane: Minimum lane width in meters (default 0.55m)
            shy_distance: Distance from walls in meters (default 0.2m)

        Returns:
            dict: Contains 'pattern', 'n_lanes', 'description', geometric constraints
        """
        n_modes = self.lateral.get('n_modes', 0)
        kurtosis = self.lateral.get('kurtosis', 0)
        means = self.lateral.get('means', np.array([]))

        # Get nominal width from lateral analysis
        W_nom = self.lateral.get('W_nom', 0)
        if W_nom == 0:
            # Fallback to wear profile
            x_coords = self.wear_profile.get('x_coords', [])
            if len(x_coords) > 0:
                W_nom = x_coords[-1] - x_coords[0]

        # Calculate effective width with shy distance
        W_eff = W_nom - 2 * shy_distance

        # Hard geometric constraint: maximum physically possible lanes for two-abreast
        M_max_two_abreast = int(np.floor(W_eff / w_lane)) if W_eff > 0 else 0
        # Single file is always geometrically possible
        single_file_possible = True
        # Legacy M_max for backwards compatibility (keep old behavior for now)
        M_max = M_max_two_abreast

        # Check for temporal drift using depth quantiles
        temporal_drift_likely = self._check_temporal_drift(shy_distance=shy_distance)

        # Initialize result
        pattern = "unknown"
        n_lanes = 0
        rationale = ""
        separation = None  # Initialize for return dict
        sep_ratio = None   # Initialize for return dict

        # Apply geometric constraints
        if M_max_two_abreast < 2:
            # Physically impossible to have two-abreast, but single file is always possible
            pattern = "SINGLE FILE"
            n_lanes = 1
            if M_max_two_abreast == 0:
                rationale = f"Effective width {W_eff:.2f}m < {w_lane:.2f}m: physically constrained to single file (cannot fit standard lane width)"
            else:
                rationale = f"Effective width {W_eff:.2f}m < {2*w_lane:.2f}m: only single lane fits, single-file traffic"

        elif n_modes == 0:
            pattern = "INSUFFICIENT DATA"
            n_lanes = 0
            rationale = "No wear pattern detected"

        elif n_modes == 1:
            pattern = "SINGLE FILE"
            n_lanes = 1
            if kurtosis > 3:
                rationale = f"Unimodal peaked distribution (kurtosis={kurtosis:.2f}): single-file traffic"
            else:
                rationale = f"Unimodal dispersed distribution (kurtosis={kurtosis:.2f}): low-density single-file"

        elif n_modes == 2 and len(means) >= 2:
            # Calculate relative separation
            separation = abs(means[1] - means[0])
            sep_ratio = separation / W_eff if W_eff > 0 else 0

            if sep_ratio >= 0.45 and sep_ratio <= 0.75:
                # Well-separated peaks suggesting two lanes
                pattern = "TWO LANES POSSIBLE"
                n_lanes = 2
                rationale = f"Two peaks separated by {separation:.2f}m (sep_ratio={sep_ratio:.2f}): consistent with bi-directional lanes"
            elif sep_ratio < 0.35:
                # Close together - likely alternating feet or temporal variation
                pattern = "SINGLE FILE (alternating feet)"
                n_lanes = 1
                rationale = f"Two close peaks (sep={separation:.2f}m, ratio={sep_ratio:.2f}): likely alternating foot placement"
            else:
                # Intermediate case
                pattern = "UNCERTAIN (two modes)"
                n_lanes = 2
                rationale = f"Two peaks with intermediate separation (sep_ratio={sep_ratio:.2f}): ambiguous pattern"

        elif n_modes >= 3:
            # Multiple modes detected
            if M_max_two_abreast < n_modes:
                # More modes than physically possible lanes - force constraint
                pattern = "SINGLE FILE (complex)"
                n_lanes = 1
                rationale = f"GMM detected {n_modes} modes but W_eff={W_eff:.2f}m only allows {M_max_two_abreast} two-abreast lanes: likely temporal drift or artifacts"
            else:
                pattern = "COMPLEX MULTI-LANE"
                n_lanes = min(n_modes, M_max_two_abreast)
                rationale = f"{n_modes} modes detected within physical constraints (M_max_two_abreast={M_max_two_abreast})"

        # Add temporal drift flag to rationale
        if temporal_drift_likely:
            rationale += " | TEMPORAL DRIFT DETECTED: peaks may represent different usage periods"

        # Generate description
        description = self._generate_simultaneity_description(pattern, n_lanes, W_eff, M_max_two_abreast)

        return {
            'pattern': pattern,
            'n_lanes': n_lanes,
            'n_modes': n_modes,
            'description': description,
            'rationale': rationale,
            'W_nom': W_nom,
            'W_eff': W_eff,
            'M_max': M_max,  # Backwards compatibility
            'M_max_two_abreast': M_max_two_abreast,
            'single_file_possible': single_file_possible,
            'shy_distance': shy_distance,
            'w_lane': w_lane,
            'temporal_drift_likely': temporal_drift_likely,
            'kurtosis': kurtosis,
            'separation': separation if n_modes == 2 and len(means) >= 2 else None,
            'sep_ratio': sep_ratio if n_modes == 2 and len(means) >= 2 else None
        }

    def _check_temporal_drift(self, shy_distance=0.2, quantiles=[0.6, 0.75, 0.9]):
        """
        Check if lateral peak positions shift across depth quantiles
        Shifting peaks suggest temporal drift rather than true simultaneity

        Args:
            shy_distance: Distance from walls in meters (for W_eff calculation)
            quantiles: Depth quantiles to test

        Returns:
            bool: True if temporal drift is likely
        """
        depth_grid = self.lateral.get('depth_grid')
        x_coords = self.lateral.get('x_coords')

        if depth_grid is None or x_coords is None:
            return False

        # Get non-zero depths
        depths_flat = depth_grid[depth_grid > 0]
        if len(depths_flat) < 10:
            return False

        peak_positions = []

        for q in quantiles:
            threshold = np.quantile(depths_flat, q)

            # Get lateral profile for cells above threshold
            mask = depth_grid >= threshold
            if np.sum(mask) < 5:
                continue

            lateral_subset = np.sum(depth_grid * mask, axis=0)
            if np.sum(lateral_subset) == 0:
                continue

            # Find peak position (center of mass)
            peak_pos = np.sum(x_coords * lateral_subset) / np.sum(lateral_subset)
            peak_positions.append(peak_pos)

        if len(peak_positions) < 2:
            return False

        # Check if peak positions shift significantly
        W_nom = self.lateral.get('W_nom', 1.0)
        W_eff = W_nom - 2 * shy_distance
        peak_shift = np.max(peak_positions) - np.min(peak_positions)
        shift_ratio = peak_shift / W_eff if W_eff > 0 else 0

        # If peaks shift by more than 15% of effective width, flag as temporal drift
        return shift_ratio > 0.15

    def _generate_simultaneity_description(self, pattern, n_lanes, W_eff, M_max_two_abreast):
        """Generate human-readable description of simultaneity pattern"""
        if pattern == "SINGLE FILE":
            if M_max_two_abreast == 0:
                return f"Single-file traffic (W_eff={W_eff:.2f}m cannot fit standard lane, single file always possible)"
            else:
                return f"Single-file traffic (W_eff={W_eff:.2f}m allows max {M_max_two_abreast} two-abreast lanes)"
        elif pattern == "SINGLE FILE (alternating feet)":
            return f"Single-file with alternating foot placement pattern"
        elif pattern == "SINGLE FILE (complex)":
            return f"Single-file traffic with temporal variation (constrained by W_eff={W_eff:.2f}m)"
        elif pattern == "TWO LANES POSSIBLE":
            return f"Two distinct lanes detected (bi-directional or paired traffic)"
        elif pattern == "COMPLEX MULTI-LANE":
            return f"Multiple lanes detected ({n_lanes} lanes within {M_max_two_abreast} max two-abreast)"
        elif pattern == "UNCERTAIN (two modes)":
            return f"Two modes detected but separation is ambiguous"
        else:
            return f"Pattern: {pattern}"

