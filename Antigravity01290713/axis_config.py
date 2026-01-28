"""
Axis Configuration for Stair Models
Handles automatic axis inference and mapping for 3D stair models
"""

import numpy as np


class AxisConfig:
    """
    Configuration for axis mapping in 3D stair models

    Attributes:
        width_axis: Index (0,1,2) for width dimension
        depth_axis: Index (0,1,2) for depth/going dimension
        up_axis: Index (0,1,2) for vertical/height dimension
    """

    def __init__(self, width_axis=None, depth_axis=None, up_axis=None):
        """
        Initialize axis configuration

        Args:
            width_axis: Index for width (0=X, 1=Y, 2=Z), None for auto-detect
            depth_axis: Index for depth (0=X, 1=Y, 2=Z), None for auto-detect
            up_axis: Index for up (0=X, 1=Y, 2=Z), None for auto-detect
        """
        self.width_axis = width_axis
        self.depth_axis = depth_axis
        self.up_axis = up_axis

    def is_auto(self):
        """Check if any axis needs auto-detection"""
        return (self.width_axis is None or
                self.depth_axis is None or
                self.up_axis is None)

    def infer_from_mesh(self, vertices, faces):
        """
        Automatically infer axes from mesh geometry

        Strategy:
        1. Compute face normals
        2. Up axis = axis with largest mean absolute normal component
        3. Of remaining axes: width = smaller range, depth = larger range

        Args:
            vertices: Nx3 array of vertex coordinates
            faces: Mx3 array of face indices
        """
        # Compute face normals
        normals = self._compute_face_normals(vertices, faces)

        # Find up axis (largest mean absolute normal component)
        mean_abs_normals = np.mean(np.abs(normals), axis=0)
        self.up_axis = int(np.argmax(mean_abs_normals))

        # Remaining axes
        remaining = [i for i in [0, 1, 2] if i != self.up_axis]

        # Compute ranges for remaining axes
        ranges = vertices.max(axis=0) - vertices.min(axis=0)
        range_0 = ranges[remaining[0]]
        range_1 = ranges[remaining[1]]

        # Width = smaller range, depth = larger range
        if range_0 < range_1:
            self.width_axis = remaining[0]
            self.depth_axis = remaining[1]
        else:
            self.width_axis = remaining[1]
            self.depth_axis = remaining[0]

    def _compute_face_normals(self, vertices, faces):
        """
        Compute normal vectors for each face

        Args:
            vertices: Nx3 array of vertex coordinates
            faces: Mx3 array of face vertex indices

        Returns:
            Mx3 array of face normal vectors
        """
        # Get vertices for each face
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]

        # Compute edge vectors
        edge1 = v1 - v0
        edge2 = v2 - v0

        # Cross product gives normal
        normals = np.cross(edge1, edge2)

        # Normalize
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / (norms + 1e-10)  # Avoid division by zero

        return normals

    def get_axis_names(self):
        """Return human-readable axis names"""
        axis_labels = ['X', 'Y', 'Z']
        return {
            'width': axis_labels[self.width_axis],
            'depth': axis_labels[self.depth_axis],
            'up': axis_labels[self.up_axis]
        }

    def __repr__(self):
        names = self.get_axis_names()
        return f"AxisConfig(width={names['width']}, depth={names['depth']}, up={names['up']})"
