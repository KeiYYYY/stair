"""
Wear Pattern Analyzer
Extracts and analyzes wear depth profiles from 3D stair models
Implements algorithms from ModelG.md Section 5 and 6
"""

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.mixture import GaussianMixture
from axis_config import AxisConfig


class WearAnalyzer:
    """Analyzes wear patterns on stair treads"""

    def __init__(self, mesh, axis_config=None, stair_width=None):
        """
        Initialize wear analyzer

        Args:
            mesh: Dictionary containing 'vertices', 'faces', 'bounds'
            axis_config: AxisConfig object (auto-infer if None)
            stair_width: Width of stairs in meters (auto-detect if None)
        """
        self.mesh = mesh
        self.vertices = mesh['vertices']
        self.faces = mesh['faces']
        self.bounds = mesh['bounds']

        # Set up axis configuration
        if axis_config is None:
            axis_config = AxisConfig()

        if axis_config.is_auto():
            axis_config.infer_from_mesh(self.vertices, self.faces)
            print(f"  Auto-detected axes: {axis_config}")

        self.axis_config = axis_config

        # Extract axis indices
        self.width_axis = axis_config.width_axis
        self.depth_axis = axis_config.depth_axis
        self.up_axis = axis_config.up_axis

        # Auto-detect stair width if not provided
        if stair_width is None:
            self.stair_width = self.bounds['size'][self.width_axis]
        else:
            self.stair_width = stair_width

        # Stair depth (going)
        self.stair_depth = self.bounds['size'][self.depth_axis]

    def extract_wear_profile(self, grid_resolution=100, use_triangle_method=True,
                            ransac_iters=500, ransac_thresh_mm=2.0,
                            seg_angle_deg=25.0, roi_quantiles=(0.01, 0.99)):
        """
        Extract wear depth profile from 3D mesh
        
        Args:
            grid_resolution: Grid resolution for legacy method and visualization
            use_triangle_method: If True, use robust triangle-based method (default)
            ransac_iters: RANSAC iterations for plane fitting
            ransac_thresh_mm: RANSAC inlier threshold in mm
            seg_angle_deg: Tread segmentation angle threshold in degrees
            roi_quantiles: ROI bounding quantiles (min, max)
        
        Returns:
            dict: Contains 'depth_grid', 'max_depth', 'volume', diagnostics, etc.
        """
        if use_triangle_method:
            return self._extract_wear_profile_triangle(ransac_iters, ransac_thresh_mm,
                                                       seg_angle_deg, roi_quantiles, grid_resolution)
        else:
            return self._extract_wear_profile_legacy(grid_resolution)
    
    def _extract_wear_profile_legacy(self, grid_resolution=100):
        """
        Legacy grid-based wear profile extraction
        Creates a 2D grid and measures depth at each point

        Returns:
            dict: Contains 'depth_grid', 'max_depth', 'volume', 'x_coords', 'y_coords'
        """
        # Get bounds for width and depth axes
        width_min = self.bounds['min'][self.width_axis]
        width_max = self.bounds['max'][self.width_axis]
        depth_min = self.bounds['min'][self.depth_axis]
        depth_max = self.bounds['max'][self.depth_axis]
        up_min = self.bounds['min'][self.up_axis]
        up_max = self.bounds['max'][self.up_axis]

        # Create 2D grid over width and depth
        width_grid = np.linspace(width_min, width_max, grid_resolution)
        depth_grid = np.linspace(depth_min, depth_max, grid_resolution)

        # Initialize depth grid
        wear_depth_grid = np.zeros((len(depth_grid), len(width_grid)))

        # Find the reference plane (unworn surface)
        # Use the highest points near the edges as reference
        edge_margin = 0.1  # 10% from edges
        edge_vertices = self.vertices[
            (self.vertices[:, self.width_axis] < width_min + edge_margin * (width_max - width_min)) |
            (self.vertices[:, self.width_axis] > width_max - edge_margin * (width_max - width_min))
        ]

        if len(edge_vertices) > 0:
            reference_height = np.percentile(edge_vertices[:, self.up_axis], 95)
        else:
            reference_height = up_max

        # For each grid point, find the surface height
        for i, depth_val in enumerate(depth_grid):
            for j, width_val in enumerate(width_grid):
                # Find vertices near this grid point
                nearby = self.vertices[
                    (np.abs(self.vertices[:, self.width_axis] - width_val) < (width_max - width_min) / grid_resolution) &
                    (np.abs(self.vertices[:, self.depth_axis] - depth_val) < (depth_max - depth_min) / grid_resolution)
                ]

                if len(nearby) > 0:
                    # Surface height at this point
                    surface_height = np.max(nearby[:, self.up_axis])
                    # Wear depth is difference from reference
                    depth = reference_height - surface_height
                    # Clamp to non-negative (wear can't be negative)
                    wear_depth_grid[i, j] = max(0, depth)
                else:
                    wear_depth_grid[i, j] = 0

        # Convert to millimeters for easier interpretation
        depth_grid_mm = wear_depth_grid * 1000

        # Calculate wear volume (m³) - fix cell area calculation
        cell_width = (width_max - width_min) / (grid_resolution - 1)
        cell_depth = (depth_max - depth_min) / (grid_resolution - 1)
        cell_area = cell_width * cell_depth
        wear_volume = np.sum(wear_depth_grid) * cell_area

        max_depth_mm = np.max(depth_grid_mm)

        # Warning for implausible wear depth
        if max_depth_mm > 200:
            print(f"  WARNING: Max wear depth {max_depth_mm:.1f} mm is implausibly high (>200mm)")
            print(f"           This likely indicates axis mapping or scaling issues")

        return {
            'depth_grid': depth_grid_mm,
            'max_depth': max_depth_mm,
            'mean_depth': np.mean(depth_grid_mm[depth_grid_mm > 0]) if np.any(depth_grid_mm > 0) else 0,
            'volume': wear_volume,
            'x_coords': width_grid,  # Width axis
            'y_coords': depth_grid,  # Depth axis
            'reference_height': reference_height,
            'axis_config': self.axis_config,
            'method': 'legacy_grid'
        }


    # ========================================================================
    # ROBUST TRIANGLE-BASED WEAR VOLUME METHODS
    # ========================================================================
    
    def fit_robust_reference_plane(self, ransac_iters=500, ransac_thresh_mm=2.0):
        """Fit robust reference plane using RANSAC (deterministic with fixed seed)"""
        # Use local RNG for reproducible RANSAC results (cleaner than global seed)
        rng = np.random.default_rng(0)
        
        w_min, w_max = self.bounds['min'][self.width_axis], self.bounds['max'][self.width_axis]
        d_min, d_max = self.bounds['min'][self.depth_axis], self.bounds['max'][self.depth_axis]
        border = 0.1
        cand_mask = (
            ((self.vertices[:, self.width_axis] < w_min + border*(w_max-w_min)) |
             (self.vertices[:, self.width_axis] > w_max - border*(w_max-w_min))) |
            ((self.vertices[:, self.depth_axis] < d_min + border*(d_max-d_min)) |
             (self.vertices[:, self.depth_axis] > d_max - border*(d_max-d_min)))
        )
        candidates = self.vertices[cand_mask]
        if len(candidates) > 100:
            candidates = candidates[candidates[:, self.up_axis] >= np.percentile(candidates[:, self.up_axis], 75)]
        if len(candidates) < 10:
            return {'plane_params': [0, 0, np.median(self.vertices[:, self.up_axis])],
                   'inlier_fraction': 0.0, 'rms_error_mm': 0.0, 'method': 'fallback', 'candidates_used': len(candidates)}
        best_params, best_count, best_mask = None, 0, None
        thresh_m = ransac_thresh_mm / 1000.0
        for _ in range(ransac_iters):
            if len(candidates) < 3: break
            idx = rng.choice(len(candidates), 3, replace=False)
            X = np.column_stack([candidates[idx, self.width_axis], candidates[idx, self.depth_axis], np.ones(3)])
            try:
                params = np.linalg.solve(X, candidates[idx, self.up_axis])
            except:
                continue
            X_all = np.column_stack([candidates[:, self.width_axis], candidates[:, self.depth_axis], np.ones(len(candidates))])
            residuals = np.abs(X_all @ params - candidates[:, self.up_axis])
            inliers = residuals < thresh_m
            if np.sum(inliers) > best_count:
                best_count, best_params, best_mask = np.sum(inliers), params, inliers
        if best_params is None or best_count < 10:
            return {'plane_params': [0, 0, np.median(self.vertices[:, self.up_axis])],
                   'inlier_fraction': 0.0, 'rms_error_mm': 0.0, 'method': 'fallback', 'candidates_used': len(candidates)}
        inlier_pts = candidates[best_mask]
        X_in = np.column_stack([inlier_pts[:, self.width_axis], inlier_pts[:, self.depth_axis], np.ones(len(inlier_pts))])
        final_params = np.linalg.lstsq(X_in, inlier_pts[:, self.up_axis], rcond=None)[0]
        rms_mm = np.sqrt(np.mean((X_in @ final_params - inlier_pts[:, self.up_axis])**2)) * 1000.0
        return {'plane_params': final_params.tolist(), 'inlier_fraction': best_count/len(candidates),
                'rms_error_mm': rms_mm, 'method': 'RANSAC', 'candidates_used': len(candidates), 'inlier_count': best_count,
                'inlier_mask': best_mask, 'candidate_vertices': candidates}
    
    def get_bootstrap_plane_uncertainty(self, ransac_result: dict, n_bootstrap: int = 500, seed: int = 42):
        """
        Generate bootstrap samples of reference plane parameters for uncertainty propagation.
        
        Args:
            ransac_result: Result from fit_robust_reference_plane()
            n_bootstrap: Number of bootstrap samples
            seed: Random seed for reproducibility
            
        Returns:
            PlaneUncertainty object with bootstrap samples
        """
        from uncertainty import bootstrap_plane_samples
        
        if 'candidate_vertices' not in ransac_result or 'inlier_mask' not in ransac_result:
            # Fallback: re-run RANSAC to get inlier info
            ransac_result = self.fit_robust_reference_plane()
        
        candidates = ransac_result.get('candidate_vertices', self.vertices)
        inlier_mask = ransac_result.get('inlier_mask', np.ones(len(candidates), dtype=bool))
        plane_params = np.array(ransac_result['plane_params'])
        
        return bootstrap_plane_samples(
            vertices=candidates,
            inlier_mask=inlier_mask,
            plane_params=plane_params,
            width_axis=self.width_axis,
            depth_axis=self.depth_axis,
            up_axis=self.up_axis,
            n_bootstrap=n_bootstrap,
            seed=seed
        )
    
    def segment_tread_triangles(self, angle_thresh_deg=25, roi_quantiles=(0.01, 0.99)):
        """Segment tread triangles"""
        n_faces = len(self.faces)
        up_vec = np.zeros(3); up_vec[self.up_axis] = 1.0
        centroids = np.array([(self.vertices[f[0]]+self.vertices[f[1]]+self.vertices[f[2]])/3.0 for f in self.faces])
        normals = []
        for f in self.faces:
            v0, v1, v2 = self.vertices[f[0]], self.vertices[f[1]], self.vertices[f[2]]
            normal = np.cross(v1-v0, v2-v0)
            norm = np.linalg.norm(normal)
            normals.append(normal/norm if norm > 1e-10 else up_vec)
        normals = np.array(normals)
        cos_angles = np.abs(np.dot(normals, up_vec))
        angles_deg = np.arccos(np.clip(cos_angles, -1, 1)) * 180/np.pi
        angle_mask = angles_deg < angle_thresh_deg
        w_min = np.percentile(centroids[:, self.width_axis], roi_quantiles[0]*100)
        w_max = np.percentile(centroids[:, self.width_axis], roi_quantiles[1]*100)
        d_min = np.percentile(centroids[:, self.depth_axis], roi_quantiles[0]*100)
        d_max = np.percentile(centroids[:, self.depth_axis], roi_quantiles[1]*100)
        roi_mask = ((centroids[:, self.width_axis] >= w_min) & (centroids[:, self.width_axis] <= w_max) &
                   (centroids[:, self.depth_axis] >= d_min) & (centroids[:, self.depth_axis] <= d_max))
        tread_mask = angle_mask & roi_mask
        return {'tread_mask': tread_mask, 'tread_count': int(np.sum(tread_mask)),
                'excluded_count': int(n_faces-np.sum(tread_mask)), 'excluded_fraction': float(1-np.sum(tread_mask)/n_faces) if n_faces>0 else 0.0}
    
    def compute_triangle_wear_volume(self, ref_plane_params, tread_mask):
        """Compute wear volume via triangle integration"""
        a, b, c = ref_plane_params
        total_vol, max_d, depth_sum, depth_cnt = 0.0, 0.0, 0.0, 0
        for i, f in enumerate(self.faces):
            if not tread_mask[i]: continue
            v0, v1, v2 = self.vertices[f[0]], self.vertices[f[1]], self.vertices[f[2]]
            h0_pred = a*v0[self.width_axis] + b*v0[self.depth_axis] + c
            h1_pred = a*v1[self.width_axis] + b*v1[self.depth_axis] + c
            h2_pred = a*v2[self.width_axis] + b*v2[self.depth_axis] + c
            d0, d1, d2 = max(0, h0_pred-v0[self.up_axis]), max(0, h1_pred-v1[self.up_axis]), max(0, h2_pred-v2[self.up_axis])
            mean_d = (d0+d1+d2)/3.0
            max_d = max(max_d, d0, d1, d2)
            if mean_d > 0:
                depth_sum += mean_d; depth_cnt += 1
            p0 = [v0[self.width_axis], v0[self.depth_axis]]
            p1 = [v1[self.width_axis], v1[self.depth_axis]]
            p2 = [v2[self.width_axis], v2[self.depth_axis]]
            area = 0.5 * abs((p1[0]-p0[0])*(p2[1]-p0[1]) - (p2[0]-p0[0])*(p1[1]-p0[1]))
            total_vol += mean_d * area
        return {'volume': total_vol, 'max_depth': max_d*1000.0,
                'mean_depth': (depth_sum/depth_cnt*1000.0) if depth_cnt>0 else 0.0, 'triangles_used': int(np.sum(tread_mask))}
    
    def _create_depth_grid_from_triangles(self, ref_params, tread_mask, grid_res):
        """Create gridded depth map for visualization"""
        a, b, c = ref_params
        w_min, w_max = self.bounds['min'][self.width_axis], self.bounds['max'][self.width_axis]
        d_min, d_max = self.bounds['min'][self.depth_axis], self.bounds['max'][self.depth_axis]
        x_coords = np.linspace(w_min, w_max, grid_res)
        y_coords = np.linspace(d_min, d_max, grid_res)
        depth_grid_mm = np.zeros((grid_res, grid_res))
        cell_w, cell_d = (w_max-w_min)/grid_res, (d_max-d_min)/grid_res
        for i, y_val in enumerate(y_coords):
            for j, x_val in enumerate(x_coords):
                nearby = self.vertices[(np.abs(self.vertices[:, self.width_axis]-x_val) < cell_w) &
                                      (np.abs(self.vertices[:, self.depth_axis]-y_val) < cell_d)]
                if len(nearby) > 0:
                    depth_grid_mm[i,j] = max(0, a*x_val+b*y_val+c - np.max(nearby[:, self.up_axis])) * 1000.0
        return depth_grid_mm, x_coords, y_coords
    
    def _extract_wear_profile_triangle(self, ransac_iters, ransac_thresh_mm, seg_angle_deg, roi_quantiles, grid_res):
        """Extract wear profile using robust triangle method"""
        print(f"  Using robust triangle-based wear volume calculation")
        ref_plane = self.fit_robust_reference_plane(ransac_iters, ransac_thresh_mm)
        print(f"    Reference plane: {ref_plane['method']}, inliers={ref_plane['inlier_fraction']:.1%}, RMS={ref_plane['rms_error_mm']:.2f}mm")
        tread_seg = self.segment_tread_triangles(seg_angle_deg, roi_quantiles)
        print(f"    Tread segmentation: {tread_seg['tread_count']} triangles ({tread_seg['excluded_fraction']:.1%} excluded)")
        wear_vol = self.compute_triangle_wear_volume(ref_plane['plane_params'], tread_seg['tread_mask'])
        print(f"    Wear volume: {wear_vol['volume']:.2e} m^3, max_depth={wear_vol['max_depth']:.1f}mm")
        depth_grid_mm, x_coords, y_coords = self._create_depth_grid_from_triangles(ref_plane['plane_params'], tread_seg['tread_mask'], grid_res)
        if ref_plane['inlier_fraction'] < 0.5:
            print(f"  WARNING: Weak reference plane fit (inliers={ref_plane['inlier_fraction']:.1%})")
        if tread_seg['excluded_fraction'] > 0.5:
            print(f"  WARNING: High triangle exclusion ({tread_seg['excluded_fraction']:.1%})")
        if wear_vol['max_depth'] > 200:
            print(f"  WARNING: Max wear depth {wear_vol['max_depth']:.1f} mm implausibly high")
        return {'depth_grid': depth_grid_mm, 'max_depth': wear_vol['max_depth'], 'mean_depth': wear_vol['mean_depth'],
                'volume': wear_vol['volume'], 'x_coords': x_coords, 'y_coords': y_coords, 'axis_config': self.axis_config,
                'method': 'robust_triangle', 'ref_plane_quality': ref_plane, 'tread_seg': tread_seg, 'triangles_used': wear_vol['triangles_used']}

    def analyze_lateral_distribution(self):
        """
        Analyze lateral (x-axis) wear distribution using Gaussian Mixture Model
        Implements ModelG.md Section 5.2

        NOTE: This method only fits GMM and returns parameters.
        Interpretation with geometric constraints happens in TrafficEstimator.

        Returns:
            dict: Contains GMM parameters, NO pattern interpretation
        """
        # Get the wear profile
        if not hasattr(self, 'wear_profile'):
            self.wear_profile = self.extract_wear_profile()

        depth_grid = self.wear_profile['depth_grid']
        x_coords = self.wear_profile['x_coords']

        # Sum wear depth along y-axis to get lateral profile
        lateral_profile = np.sum(depth_grid, axis=0)

        # Normalize to probability distribution
        if np.sum(lateral_profile) > 0:
            lateral_profile = lateral_profile / np.sum(lateral_profile)
        else:
            return {
                'n_modes': 0,
                'pattern': 'gmm_uninterpreted',
                'lateral_profile': lateral_profile,
                'x_coords': x_coords
            }

        # Prepare data for GMM (need weighted samples)
        samples = []
        for i, weight in enumerate(lateral_profile):
            n_samples = int(weight * 1000)  # Scale up for GMM
            samples.extend([x_coords[i]] * n_samples)

        samples = np.array(samples).reshape(-1, 1)

        if len(samples) < 10:
            return {
                'n_modes': 0,
                'pattern': 'gmm_uninterpreted',
                'lateral_profile': lateral_profile,
                'x_coords': x_coords
            }

        # Fit GMM with 1, 2, and 3 components
        best_n = 1
        best_bic = np.inf
        best_gmm = None

        for n_components in [1, 2, 3]:
            try:
                gmm = GaussianMixture(n_components=n_components, random_state=42)
                gmm.fit(samples)
                bic = gmm.bic(samples)

                if bic < best_bic:
                    best_bic = bic
                    best_n = n_components
                    best_gmm = gmm
            except:
                continue

        if best_gmm is None:
            return {
                'n_modes': 0,
                'pattern': 'gmm_uninterpreted',
                'lateral_profile': lateral_profile,
                'x_coords': x_coords
            }

        # Extract parameters
        means = best_gmm.means_.flatten()
        stds = np.sqrt(best_gmm.covariances_.flatten())
        weights = best_gmm.weights_

        # Sort by position
        sort_idx = np.argsort(means)
        means = means[sort_idx]
        stds = stds[sort_idx]
        weights = weights[sort_idx]

        # Calculate statistical moments
        kurtosis = stats.kurtosis(samples.flatten())
        skewness = stats.skew(samples.flatten())

        # Compute W_nom for use in TrafficEstimator
        W_nom = x_coords[-1] - x_coords[0]

        return {
            'n_modes': best_n,
            'means': means,
            'stds': stds,
            'weights': weights,
            'pattern': 'gmm_uninterpreted',  # No interpretation here
            'kurtosis': kurtosis,
            'skewness': skewness,
            'lateral_profile': lateral_profile,
            'x_coords': x_coords,
            'W_nom': W_nom,
            'bic': best_bic,
            'depth_grid': depth_grid  # For quantile analysis
        }

    def analyze_longitudinal_profile(self):
        """
        Analyze longitudinal (y-axis, front-to-back) wear distribution
        Detects nosing wear to determine directionality
        Implements ModelG.md Section 3.1

        Returns:
            dict: Contains 'nosing_wear_ratio', 'center_wear', 'edge_profile'
        """
        if not hasattr(self, 'wear_profile'):
            self.wear_profile = self.extract_wear_profile()

        depth_grid = self.wear_profile['depth_grid']
        y_coords = self.wear_profile['y_coords']

        # Sum wear depth along x-axis to get longitudinal profile
        longitudinal_profile = np.sum(depth_grid, axis=1)

        # Divide into regions: front (nosing), center, back
        n_points = len(longitudinal_profile)
        front_region = longitudinal_profile[:n_points//4]
        center_region = longitudinal_profile[n_points//4:3*n_points//4]
        back_region = longitudinal_profile[3*n_points//4:]

        # Calculate wear in each region
        nosing_wear = np.mean(front_region) if len(front_region) > 0 else 0
        center_wear = np.mean(center_region) if len(center_region) > 0 else 0
        back_wear = np.mean(back_region) if len(back_region) > 0 else 0

        # Nosing wear ratio (higher means more descent traffic)
        if center_wear > 0:
            nosing_wear_ratio = nosing_wear / center_wear
        else:
            nosing_wear_ratio = 0

        # Calculate skewness (positive = front-heavy, negative = back-heavy)
        skewness = stats.skew(longitudinal_profile)

        return {
            'nosing_wear': nosing_wear,
            'center_wear': center_wear,
            'back_wear': back_wear,
            'nosing_wear_ratio': nosing_wear_ratio,
            'skewness': skewness,
            'longitudinal_profile': longitudinal_profile,
            'y_coords': y_coords
        }
