"""
OBJ File Parser for 3D Mesh Data
Parses Wavefront .obj files and extracts vertex and face information
"""

import numpy as np
from pathlib import Path


class OBJParser:
    """Parser for Wavefront .obj 3D model files"""

    def __init__(self, filepath, scale=1.0):
        """
        Initialize OBJ parser

        Args:
            filepath: Path to .obj file
            scale: Scale factor to apply to all coordinates (default 1.0)
        """
        self.filepath = Path(filepath)
        self.scale = scale
        self.vertices = []
        self.faces = []
        self.normals = []

    def parse(self):
        """
        Parse the .obj file and return mesh data

        Returns:
            dict: Contains 'vertices' (Nx3 array), 'faces' (Mx3 array),
                  'bounds' (min/max coordinates)
        """
        with open(self.filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split()
                if not parts:
                    continue

                # Vertex coordinates
                if parts[0] == 'v':
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    self.vertices.append([x, y, z])

                # Vertex normals
                elif parts[0] == 'vn':
                    nx, ny, nz = float(parts[1]), float(parts[2]), float(parts[3])
                    self.normals.append([nx, ny, nz])

                # Faces (triangles)
                elif parts[0] == 'f':
                    # Handle different face formats: f v1 v2 v3 or f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3
                    face_vertices = []
                    for vertex_data in parts[1:]:
                        # Extract vertex index (first number)
                        vertex_idx = int(vertex_data.split('/')[0]) - 1  # OBJ indices start at 1
                        face_vertices.append(vertex_idx)

                    # Triangulate if necessary (for quads or polygons)
                    if len(face_vertices) == 3:
                        self.faces.append(face_vertices)
                    elif len(face_vertices) == 4:
                        # Split quad into two triangles
                        self.faces.append([face_vertices[0], face_vertices[1], face_vertices[2]])
                        self.faces.append([face_vertices[0], face_vertices[2], face_vertices[3]])
                    else:
                        # Fan triangulation for polygons
                        for i in range(1, len(face_vertices) - 1):
                            self.faces.append([face_vertices[0], face_vertices[i], face_vertices[i+1]])

        # Convert to numpy arrays
        vertices = np.array(self.vertices)
        faces = np.array(self.faces)

        # Apply scaling
        if self.scale != 1.0:
            vertices = vertices * self.scale

        # Calculate bounds
        bounds = {
            'min': vertices.min(axis=0),
            'max': vertices.max(axis=0),
            'center': vertices.mean(axis=0),
            'size': vertices.max(axis=0) - vertices.min(axis=0)
        }

        return {
            'vertices': vertices,
            'faces': faces,
            'normals': np.array(self.normals) if self.normals else None,
            'bounds': bounds,
            'scale': self.scale
        }

    def compute_face_normals(self, vertices, faces):
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
