#!/usr/bin/env python3
import argparse
import json
import math
import os
import sys

import numpy as np
import trimesh


def _update_bbox(bbox, v):
    if bbox is None:
        return [[v[0], v[1], v[2]], [v[0], v[1], v[2]]]
    bbox[0][0] = min(bbox[0][0], v[0])
    bbox[0][1] = min(bbox[0][1], v[1])
    bbox[0][2] = min(bbox[0][2], v[2])
    bbox[1][0] = max(bbox[1][0], v[0])
    bbox[1][1] = max(bbox[1][1], v[1])
    bbox[1][2] = max(bbox[1][2], v[2])
    return bbox


def _bbox_extents(bbox):
    if bbox is None:
        return None
    return [
        bbox[1][0] - bbox[0][0],
        bbox[1][1] - bbox[0][1],
        bbox[1][2] - bbox[0][2],
    ]


def _parse_face_vertex_index(token, vertex_count):
    # Face tokens can be v, v/vt, v//vn, v/vt/vn
    if "/" in token:
        token = token.split("/")[0]
    if not token:
        return None
    idx = int(token)
    if idx < 0:
        idx = vertex_count + idx
    else:
        idx -= 1
    if idx < 0 or idx >= vertex_count:
        return None
    return idx


def parse_obj_stats(obj_path):
    vertices = []
    vt_count = 0
    vn_count = 0
    face_count = 0
    triangle_count = 0

    group_seen = False
    object_seen = False
    current_group = None
    current_object = None

    group_stats = {}
    object_stats = {}
    default_stats = {"triangles": 0, "bbox": None}

    def update_part(stats, name, indices, tri_count):
        entry = stats.get(name)
        if entry is None:
            entry = {"triangles": 0, "bbox": None}
            stats[name] = entry
        entry["triangles"] += tri_count
        bbox = entry["bbox"]
        for idx in indices:
            v = vertices[idx]
            bbox = _update_bbox(bbox, v)
        entry["bbox"] = bbox

    with open(obj_path, "r", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        vertices.append(
                            [float(parts[1]), float(parts[2]), float(parts[3])]
                        )
                    except ValueError:
                        continue
            elif line.startswith("vt "):
                vt_count += 1
            elif line.startswith("vn "):
                vn_count += 1
            elif line.startswith("f "):
                face_count += 1
                tokens = line.split()[1:]
                if len(tokens) < 3:
                    continue
                v_count = len(vertices)
                indices = []
                for token in tokens:
                    idx = _parse_face_vertex_index(token, v_count)
                    if idx is not None:
                        indices.append(idx)
                if len(indices) < 3:
                    continue
                tri_count = len(indices) - 2
                triangle_count += tri_count
                if current_group:
                    update_part(group_stats, current_group, indices, tri_count)
                if current_object:
                    update_part(object_stats, current_object, indices, tri_count)
                if not current_group and not current_object:
                    default_stats["triangles"] += tri_count
                    bbox = default_stats["bbox"]
                    for idx in indices:
                        bbox = _update_bbox(bbox, vertices[idx])
                    default_stats["bbox"] = bbox
            elif line.startswith("g "):
                group_seen = True
                parts = line.split()[1:]
                if not parts or parts[0].lower() == "off":
                    current_group = None
                else:
                    current_group = " ".join(parts)
            elif line.startswith("o "):
                object_seen = True
                parts = line.split()[1:]
                if not parts:
                    current_object = None
                else:
                    current_object = " ".join(parts)

    overall_bbox = None
    if vertices:
        verts_np = np.array(vertices, dtype=np.float64)
        overall_min = verts_np.min(axis=0).tolist()
        overall_max = verts_np.max(axis=0).tolist()
        overall_bbox = [overall_min, overall_max]

    return {
        "vertex_count": len(vertices),
        "vt_count": vt_count,
        "vn_count": vn_count,
        "face_count": face_count,
        "triangle_count": triangle_count,
        "group_seen": group_seen,
        "object_seen": object_seen,
        "group_stats": group_stats,
        "object_stats": object_stats,
        "default_stats": default_stats,
        "overall_bbox": overall_bbox,
    }


def print_obj_stats(stats):
    print("OBJ stats:")
    print(f"  vertices: {stats['vertex_count']}")
    print(f"  texcoords: {stats['vt_count']}")
    print(f"  normals: {stats['vn_count']}")
    print(f"  faces: {stats['face_count']}")
    print(f"  triangles: {stats['triangle_count']}")
    if stats["overall_bbox"] is not None:
        extents = _bbox_extents(stats["overall_bbox"])
        print("  overall bbox:")
        print(f"    min: {stats['overall_bbox'][0]}")
        print(f"    max: {stats['overall_bbox'][1]}")
        print(f"    extents: {extents}")
    if stats["object_seen"] and stats["object_stats"]:
        print("  objects:")
        for name, info in sorted(
            stats["object_stats"].items(),
            key=lambda kv: kv[1]["triangles"],
            reverse=True,
        ):
            extents = _bbox_extents(info["bbox"])
            print(f"    {name}: triangles={info['triangles']}")
            if info["bbox"] is not None:
                print(f"      bbox min: {info['bbox'][0]}")
                print(f"      bbox max: {info['bbox'][1]}")
                print(f"      extents: {extents}")
    if stats["group_seen"] and stats["group_stats"]:
        print("  groups:")
        for name, info in sorted(
            stats["group_stats"].items(),
            key=lambda kv: kv[1]["triangles"],
            reverse=True,
        ):
            extents = _bbox_extents(info["bbox"])
            print(f"    {name}: triangles={info['triangles']}")
            if info["bbox"] is not None:
                print(f"      bbox min: {info['bbox'][0]}")
                print(f"      bbox max: {info['bbox'][1]}")
                print(f"      extents: {extents}")
    if (
        not stats["object_seen"]
        and not stats["group_seen"]
        and stats["default_stats"]["triangles"] > 0
    ):
        info = stats["default_stats"]
        extents = _bbox_extents(info["bbox"])
        print("  default part:")
        print(f"    triangles={info['triangles']}")
        if info["bbox"] is not None:
            print(f"    bbox min: {info['bbox'][0]}")
            print(f"    bbox max: {info['bbox'][1]}")
            print(f"    extents: {extents}")


def load_mesh(obj_path, process):
    mesh = trimesh.load(obj_path, force="mesh", process=process)
    if isinstance(mesh, trimesh.Scene):
        geometries = []
        for geom in mesh.geometry.values():
            if isinstance(geom, trimesh.Trimesh):
                geometries.append(geom)
        if not geometries:
            raise RuntimeError("No mesh geometry found in OBJ.")
        mesh = trimesh.util.concatenate(geometries)
    return mesh


def normalize_mesh(mesh):
    bounds = mesh.bounds
    center = bounds.mean(axis=0)
    extents = bounds[1] - bounds[0]
    scale = float(extents.max())
    if scale <= 0:
        scale = 1.0
    normalized = mesh.copy()
    normalized.vertices = (mesh.vertices - center) / scale + 0.5
    return normalized, center, scale, bounds, extents


def _grid_coords(resolution):
    return (np.arange(resolution, dtype=np.float32) + 0.5) / float(resolution)


def compute_occupancy(mesh, resolution, chunk_size):
    grid = np.zeros((resolution, resolution, resolution), dtype=np.uint8)
    coords = _grid_coords(resolution)
    for z0 in range(0, resolution, chunk_size):
        z1 = min(resolution, z0 + chunk_size)
        zs = coords[z0:z1]
        xs, ys, zs = np.meshgrid(coords, coords, zs, indexing="ij")
        points = np.stack((xs, ys, zs), axis=-1).reshape(-1, 3)
        inside = mesh.contains(points)
        if inside is None:
            raise RuntimeError("mesh.contains returned None")
        grid[:, :, z0:z1] = inside.reshape(resolution, resolution, z1 - z0).astype(
            np.uint8
        )
    return grid


def compute_sdf(mesh, resolution, chunk_size):
    grid = np.zeros((resolution, resolution, resolution), dtype=np.float32)
    coords = _grid_coords(resolution)
    query = trimesh.proximity.ProximityQuery(mesh)
    for z0 in range(0, resolution, chunk_size):
        z1 = min(resolution, z0 + chunk_size)
        zs = coords[z0:z1]
        xs, ys, zs = np.meshgrid(coords, coords, zs, indexing="ij")
        points = np.stack((xs, ys, zs), axis=-1).reshape(-1, 3)
        signed = None
        try:
            inside = mesh.contains(points)
        except Exception:
            inside = None
        if inside is not None:
            _, distances, _ = query.on_surface(points)
            signed = distances.astype(np.float32)
            signed[inside] *= -1.0
        else:
            try:
                signed = query.signed_distance(points).astype(np.float32)
            except Exception:
                _, distances, _ = query.on_surface(points)
                signed = distances.astype(np.float32)
        grid[:, :, z0:z1] = signed.reshape(resolution, resolution, z1 - z0)
    return grid


def save_grid_dense(path, grid):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, grid)


def save_grid_sparse(path, grid):
    indices = np.argwhere(grid != 0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, indices=indices.astype(np.int32), shape=grid.shape)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze OBJ and voxelize into occupancy or SDF grids."
    )
    parser.add_argument("--obj", default="Stair.obj", help="Path to OBJ file.")
    parser.add_argument("--resolution", "-r", type=int, default=128)
    parser.add_argument("--out-dir", default="output")
    parser.add_argument(
        "--min-component-faces",
        type=int,
        default=200,
        help="Discard components with fewer faces than this threshold.",
    )
    parser.add_argument(
        "--keep-components",
        type=int,
        default=2,
        help="Number of largest components to keep after filtering.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=8,
        help="Number of Z slices per chunk for grid computations.",
    )
    parser.add_argument(
        "--sdf",
        action="store_true",
        help="Output signed distance field (float32) instead of occupancy.",
    )
    parser.add_argument(
        "--sparse",
        action="store_true",
        help="Additionally save a sparse NPZ for occupancy grids.",
    )
    parser.add_argument(
        "--process",
        action="store_true",
        help="Let trimesh process/clean the mesh on load.",
    )
    args = parser.parse_args()

    obj_path = args.obj
    if not os.path.exists(obj_path):
        print(f"OBJ not found: {obj_path}", file=sys.stderr)
        sys.exit(1)

    stats = parse_obj_stats(obj_path)
    print_obj_stats(stats)

    mesh = load_mesh(obj_path, process=args.process)
    components = mesh.split(only_watertight=False)
    if not components:
        components = [mesh]
    components = sorted(components, key=lambda m: len(m.faces), reverse=True)

    filtered = [c for c in components if len(c.faces) >= args.min_component_faces]
    if not filtered:
        filtered = components
    kept = filtered[: max(1, args.keep_components)]

    print(f"Found {len(components)} components; keeping {len(kept)}.")

    if args.resolution <= 0:
        print("Resolution must be positive.", file=sys.stderr)
        sys.exit(1)
    if args.chunk_size <= 0:
        print("Chunk size must be positive.", file=sys.stderr)
        sys.exit(1)

    for idx, comp in enumerate(kept, start=1):
        comp = comp.copy()
        comp.remove_unreferenced_vertices()
        normalized, center, scale, bounds, extents = normalize_mesh(comp)
        normalized_bounds = normalized.bounds

        pitch = 1.0 / float(args.resolution)
        print(
            f"Stair {idx}: faces={len(comp.faces)} vertices={len(comp.vertices)} "
            f"scale={scale:.6f} pitch={pitch:.6f}"
        )

        if args.sdf:
            grid = compute_sdf(normalized, args.resolution, args.chunk_size)
        else:
            try:
                grid = compute_occupancy(normalized, args.resolution, args.chunk_size)
            except Exception as exc:
                print(f"Occupancy via ray parity failed: {exc}")
                print("Falling back to surface voxelization.")
                vox = normalized.voxelized(pitch)
                grid = vox.matrix.astype(np.uint8)
                if grid.shape != (args.resolution, args.resolution, args.resolution):
                    print(
                        f"Warning: voxel grid shape {grid.shape} "
                        f"does not match requested resolution {args.resolution}."
                    )

        grid_path = os.path.join(args.out_dir, f"stair_{idx}_vox.npy")
        save_grid_dense(grid_path, grid)
        if args.sparse and not args.sdf:
            sparse_path = os.path.join(args.out_dir, f"stair_{idx}_vox_sparse.npz")
            save_grid_sparse(sparse_path, grid)

        meta = {
            "component_index": idx,
            "faces": int(len(comp.faces)),
            "vertices": int(len(comp.vertices)),
            "bbox_min": bounds[0].tolist(),
            "bbox_max": bounds[1].tolist(),
            "extents": extents.tolist(),
            "normalized_bbox_min": normalized_bounds[0].tolist(),
            "normalized_bbox_max": normalized_bounds[1].tolist(),
            "center": center.tolist(),
            "scale": float(scale),
            "resolution": int(args.resolution),
            "pitch": float(pitch),
            "pitch_world": float(pitch * scale),
            "grid_type": "sdf" if args.sdf else "occupancy",
            "assumptions": [
                "Inside/outside uses ray parity; non-watertight meshes can be inaccurate."
            ],
        }
        meta_path = os.path.join(args.out_dir, f"stair_{idx}_meta.json")
        os.makedirs(args.out_dir, exist_ok=True)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
