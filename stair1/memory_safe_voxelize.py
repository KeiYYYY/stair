#!/usr/bin/env python3
import argparse
import json
import math
import os
import struct
import sys
import time

import numpy as np
import trimesh

try:
    import zarr
except ImportError:
    zarr = None

try:
    import psutil
except ImportError:
    psutil = None


BYTES_IN_MB = 1024 * 1024
BITPACK_MAGIC = b"VPCK"
BITPACK_VERSION = 1
BITPACK_ORDER_XYZ = 0
BITPACK_BITORDER_BIG = 0


def _log(message):
    print(message, flush=True)


def _rss_mb():
    if psutil is None:
        return None
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / float(BYTES_IN_MB)


def _available_ram_mb():
    if psutil is not None:
        return psutil.virtual_memory().available / float(BYTES_IN_MB)
    if hasattr(os, "sysconf"):
        try:
            pages = os.sysconf("SC_AVPHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return (pages * page_size) / float(BYTES_IN_MB)
        except (ValueError, OSError):
            return None
    return None


def default_max_ram_mb():
    available = _available_ram_mb()
    if available is None:
        return 4096
    return max(256, int(available * 0.7))


def enforce_rss(max_mb, context):
    rss = _rss_mb()
    if rss is None:
        return
    if rss > max_mb:
        raise MemoryError(
            f"RSS {rss:.1f} MB exceeded cap {max_mb} MB during {context}"
        )


def estimate_stream_bytes(
    resolution,
    chunk_xy,
    dtype_bytes,
    use_contains,
    use_sdf,
    use_bitpack,
):
    n = resolution * chunk_xy
    points = n * 3 * 4
    inside = n * 1 if use_contains else 0
    sdf_tmp = n * 8 if use_sdf else 0
    output = n * dtype_bytes
    slice_buf = resolution * resolution if use_bitpack else 0
    return points + inside + sdf_tmp + output + slice_buf


def adjust_chunk_xy(
    resolution,
    chunk_xy,
    max_bytes,
    dtype_bytes,
    use_contains,
    use_sdf,
    use_bitpack,
):
    slice_buf = resolution * resolution if use_bitpack else 0
    per_row = resolution * (
        3 * 4
        + (1 if use_contains else 0)
        + (8 if use_sdf else 0)
        + dtype_bytes
    )
    if max_bytes <= slice_buf:
        raise MemoryError(
            "Memory cap too small to hold even a single slice buffer."
        )
    if per_row <= 0:
        return chunk_xy
    max_rows = int((max_bytes - slice_buf) // per_row)
    if max_rows < 1:
        raise MemoryError(
            "Memory cap too small for even one row of streaming buffers."
        )
    return max(1, min(chunk_xy, max_rows))


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
    normalized_bounds = normalized.bounds
    return normalized, center, scale, bounds, extents, normalized_bounds


def grid_coords(resolution):
    return (np.arange(resolution, dtype=np.float32) + 0.5) / float(resolution)


def surface_voxel_indices(mesh, resolution):
    pitch = 1.0 / float(resolution)
    vox = mesh.voxelized(pitch)
    if vox.points.size == 0:
        return np.zeros((0, 3), dtype=np.uint32)
    points = vox.points.astype(np.float32, copy=False)
    max_coord = np.nextafter(np.float32(1.0), np.float32(0.0))
    clipped = np.clip(points, 0.0, max_coord)
    indices = np.floor(clipped * float(resolution)).astype(np.int64, copy=False)
    indices = np.clip(indices, 0, resolution - 1).astype(np.uint32, copy=False)
    return indices


def open_output_array(out_path, shape, dtype, fmt, chunk_xy, chunk_z):
    if fmt == "zarr":
        if zarr is None:
            raise RuntimeError("zarr is not installed. Install it or choose --format memmap.")
        chunks = (
            min(chunk_xy, shape[0]),
            min(chunk_xy, shape[1]),
            min(chunk_z, shape[2]),
        )
        return zarr.open(out_path, mode="w", shape=shape, chunks=chunks, dtype=dtype)
    if fmt == "memmap":
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        return np.lib.format.open_memmap(out_path, mode="w+", dtype=dtype, shape=shape)
    raise ValueError(f"Unsupported dense format: {fmt}")


def write_sparse_npz(path, indices, shape):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        indices=indices.astype(np.uint32, copy=False),
        shape=np.array(shape, dtype=np.uint32),
    )


def write_dense_from_indices(arr, indices, resolution, log_every, max_ram_mb):
    slice_buf = np.zeros((resolution, resolution), dtype=np.uint8)
    if indices.size == 0:
        arr[:] = 0
        return
    order = np.argsort(indices[:, 2], kind="mergesort")
    indices = indices[order]
    pos = 0
    total = indices.shape[0]
    for z in range(resolution):
        slice_buf.fill(0)
        start = pos
        while pos < total and indices[pos, 2] == z:
            pos += 1
        if pos > start:
            coords = indices[start:pos].astype(np.intp, copy=False)
            slice_buf[coords[:, 0], coords[:, 1]] = 1
        arr[:, :, z] = slice_buf
        if (z % log_every) == 0:
            rss = _rss_mb()
            if rss is not None:
                _log(f"[z={z}] RSS={rss:.1f} MB")
            enforce_rss(max_ram_mb, f"surface dense write z={z}")


def write_bitpack_header(file_obj, resolution):
    bits_per_slice = resolution * resolution
    bytes_per_slice = (bits_per_slice + 7) // 8
    header = struct.pack(
        "<4sIIIII",
        BITPACK_MAGIC,
        BITPACK_VERSION,
        resolution,
        BITPACK_ORDER_XYZ,
        BITPACK_BITORDER_BIG,
        bytes_per_slice,
    )
    file_obj.write(header)
    return bytes_per_slice


def write_bitpack_from_indices(path, indices, resolution, log_every, max_ram_mb):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        bytes_per_slice = write_bitpack_header(f, resolution)
        slice_buf = np.zeros((resolution, resolution), dtype=np.uint8)
        if indices.size == 0:
            blank = bytes([0]) * bytes_per_slice
            for _ in range(resolution):
                f.write(blank)
            return
        order = np.argsort(indices[:, 2], kind="mergesort")
        indices = indices[order]
        pos = 0
        total = indices.shape[0]
        for z in range(resolution):
            slice_buf.fill(0)
            start = pos
            while pos < total and indices[pos, 2] == z:
                pos += 1
            if pos > start:
                coords = indices[start:pos].astype(np.intp, copy=False)
                slice_buf[coords[:, 0], coords[:, 1]] = 1
            packed = np.packbits(slice_buf.reshape(-1), bitorder="big")
            if packed.size != bytes_per_slice:
                raise RuntimeError("Bitpack size mismatch; check slice packing.")
            f.write(packed.tobytes())
            if (z % log_every) == 0:
                rss = _rss_mb()
                if rss is not None:
                    _log(f"[z={z}] RSS={rss:.1f} MB")
                enforce_rss(max_ram_mb, f"surface bitpack z={z}")


class SparseIndexWriter:
    def __init__(self, out_path):
        self.out_path = out_path
        self.tmp_path = out_path + ".tmp"
        self.file = open(self.tmp_path, "wb")
        self.count = 0

    def write(self, coords):
        if coords.size == 0:
            return
        coords = coords.astype(np.uint32, copy=False)
        coords.tofile(self.file)
        self.count += int(coords.shape[0])

    def finalize(self, shape):
        self.file.close()
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        if self.count == 0:
            indices = np.zeros((0, 3), dtype=np.uint32)
            write_sparse_npz(self.out_path, indices, shape)
        else:
            indices = np.memmap(
                self.tmp_path, dtype=np.uint32, mode="r", shape=(self.count, 3)
            )
            write_sparse_npz(self.out_path, indices, shape)
            del indices
        os.remove(self.tmp_path)
        return self.count


def fill_points(points_buf, x_coords, y_coords, z_coord):
    n = x_coords.shape[0] * y_coords.shape[0]
    idx = 0
    for y in y_coords:
        points_buf[idx : idx + x_coords.shape[0], 0] = x_coords
        points_buf[idx : idx + x_coords.shape[0], 1] = y
        idx += x_coords.shape[0]
    points_buf[:n, 2] = z_coord
    return n


def solid_occupancy_stream(
    mesh,
    resolution,
    chunk_z,
    chunk_xy,
    output,
    fmt,
    dense_grid,
    log_every,
    max_ram_mb,
):
    coords = grid_coords(resolution)
    x_coords = coords
    points_buf = np.empty((resolution * chunk_xy, 3), dtype=np.float32)
    voxel_count = 0
    if fmt == "bitpack":
        slice_buf = np.zeros((resolution, resolution), dtype=np.uint8)
        bytes_per_slice = write_bitpack_header(output, resolution)
    elif fmt == "npz_sparse":
        writer = output
    else:
        slice_buf = None
        writer = None

    for z0 in range(0, resolution, chunk_z):
        z1 = min(resolution, z0 + chunk_z)
        for z in range(z0, z1):
            z_coord = coords[z]
            if fmt == "bitpack":
                slice_buf.fill(0)
            for y0 in range(0, resolution, chunk_xy):
                y1 = min(resolution, y0 + chunk_xy)
                y_coords = coords[y0:y1]
                n = fill_points(points_buf, x_coords, y_coords, z_coord)
                inside = mesh.contains(points_buf[:n])
                if inside is None:
                    raise RuntimeError("mesh.contains returned None; solid mode failed.")
                inside = inside.reshape((y1 - y0, resolution))
                inside_t = inside.T
                if dense_grid is not None:
                    dense_grid[:, y0:y1, z] = inside_t.astype(np.uint8)
                elif fmt in ("zarr", "memmap"):
                    output[:, y0:y1, z] = inside_t.astype(np.uint8)
                elif fmt == "npz_sparse":
                    ys, xs = np.nonzero(inside)
                    if xs.size:
                        coords_out = np.stack(
                            (
                                xs.astype(np.uint32),
                                (ys + y0).astype(np.uint32),
                                np.full(xs.shape, z, dtype=np.uint32),
                            ),
                            axis=1,
                        )
                        writer.write(coords_out)
                        voxel_count += int(xs.shape[0])
                elif fmt == "bitpack":
                    slice_buf[:, y0:y1] = inside_t.astype(np.uint8)
                else:
                    raise ValueError(f"Unsupported format: {fmt}")
                if fmt != "npz_sparse":
                    voxel_count += int(inside.sum())
                enforce_rss(max_ram_mb, f"solid occupancy z={z} y0={y0}")
            if fmt == "bitpack":
                packed = np.packbits(slice_buf.reshape(-1), bitorder="big")
                if packed.size != bytes_per_slice:
                    raise RuntimeError("Bitpack size mismatch; check slice packing.")
                output.write(packed.tobytes())
            if (z % log_every) == 0:
                rss = _rss_mb()
                if rss is not None:
                    _log(f"[z={z}] RSS={rss:.1f} MB")
    return voxel_count


def sdf_stream(
    mesh,
    resolution,
    chunk_z,
    chunk_xy,
    output,
    dense_grid,
    log_every,
    max_ram_mb,
):
    coords = grid_coords(resolution)
    x_coords = coords
    points_buf = np.empty((resolution * chunk_xy, 3), dtype=np.float32)
    query = trimesh.proximity.ProximityQuery(mesh)
    contains_ok = True
    warned = False
    voxel_count = 0

    for z0 in range(0, resolution, chunk_z):
        z1 = min(resolution, z0 + chunk_z)
        for z in range(z0, z1):
            z_coord = coords[z]
            for y0 in range(0, resolution, chunk_xy):
                y1 = min(resolution, y0 + chunk_xy)
                y_coords = coords[y0:y1]
                n = fill_points(points_buf, x_coords, y_coords, z_coord)
                dist = query.signed_distance(points_buf[:n])
                unsigned = np.abs(dist).astype(np.float32, copy=False)
                if contains_ok:
                    try:
                        inside = mesh.contains(points_buf[:n])
                        if inside is None:
                            raise RuntimeError("mesh.contains returned None")
                    except Exception as exc:
                        contains_ok = False
                        inside = None
                        if not warned:
                            _log(
                                f"Warning: mesh.contains failed for SDF sign; "
                                f"writing unsigned distances. ({exc})"
                            )
                            warned = True
                else:
                    inside = None
                if inside is not None:
                    signed = unsigned.copy()
                    signed[inside] *= -1.0
                else:
                    signed = unsigned
                signed = signed.reshape((y1 - y0, resolution)).T
                if dense_grid is not None:
                    dense_grid[:, y0:y1, z] = signed
                else:
                    output[:, y0:y1, z] = signed
                voxel_count += int(signed.size)
                enforce_rss(max_ram_mb, f"SDF z={z} y0={y0}")
            if (z % log_every) == 0:
                rss = _rss_mb()
                if rss is not None:
                    _log(f"[z={z}] RSS={rss:.1f} MB")
    return voxel_count, contains_ok


def write_metadata(meta_path, meta):
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Memory-bounded voxelization for large OBJ meshes."
    )
    parser.add_argument("--obj", default="Stair.obj", help="Path to OBJ file.")
    parser.add_argument("--out-dir", default="output")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--keep-components", type=int, default=2)
    parser.add_argument("--min-component-faces", type=int, default=200)
    parser.add_argument("--mode", choices=("surface", "solid"), default="surface")
    parser.add_argument(
        "--format",
        choices=("zarr", "memmap", "npz_sparse", "bitpack"),
        default="npz_sparse",
    )
    parser.add_argument("--chunk-z", type=int, default=8)
    parser.add_argument("--chunk-xy", type=int, default=256)
    parser.add_argument("--max-ram-mb", type=int, default=None)
    parser.add_argument("--sdf", action="store_true")
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--process", action="store_true", help="Let trimesh process/clean on load."
    )
    args = parser.parse_args()

    if not os.path.exists(args.obj):
        print(f"OBJ not found: {args.obj}", file=sys.stderr)
        sys.exit(1)

    if args.resolution <= 0:
        print("Resolution must be positive.", file=sys.stderr)
        sys.exit(1)
    if args.chunk_z <= 0 or args.chunk_xy <= 0:
        print("Chunk sizes must be positive.", file=sys.stderr)
        sys.exit(1)

    if args.self_test:
        _log("Self-test mode: forcing resolution=128 and keep-components=1.")
        args.resolution = 128
        args.keep_components = 1

    max_ram_mb = args.max_ram_mb or default_max_ram_mb()
    max_ram_bytes = int(max_ram_mb * BYTES_IN_MB)
    _log(f"Max RAM cap: {max_ram_mb} MB")
    if psutil is None:
        _log("Note: psutil not installed; RSS enforcement is best-effort.")

    if args.sdf and args.format not in ("zarr", "memmap"):
        print("SDF output requires --format zarr or memmap.", file=sys.stderr)
        sys.exit(1)

    mesh = load_mesh(args.obj, process=args.process)
    components = mesh.split(only_watertight=False)
    if not components:
        components = [mesh]
    components = sorted(components, key=lambda m: len(m.faces), reverse=True)

    filtered = [c for c in components if len(c.faces) >= args.min_component_faces]
    if not filtered:
        filtered = components
    kept = filtered[: max(1, args.keep_components)]
    _log(f"Found {len(components)} components; keeping {len(kept)}.")

    log_every = max(1, args.resolution // 16)
    total_start = time.time()

    for idx, comp in enumerate(kept, start=1):
        comp = comp.copy()
        comp.remove_unreferenced_vertices()
        is_watertight = bool(comp.is_watertight)
        mode = args.mode
        if args.sdf:
            mode = "sdf"
        elif mode == "solid" and not is_watertight:
            _log(
                "Warning: mesh is not watertight; falling back to surface mode."
            )
            mode = "surface"

        normalized, center, scale, bounds, extents, norm_bounds = normalize_mesh(comp)
        resolution = int(args.resolution)
        pitch = 1.0 / float(resolution)

        dense_grid = None
        dtype_bytes = 4 if args.sdf else 1
        if args.dense:
            if resolution > 256:
                print(
                    "Dense mode only allowed for resolution<=256.",
                    file=sys.stderr,
                )
                sys.exit(1)
            dense_bytes = resolution ** 3 * dtype_bytes
            if dense_bytes > max_ram_bytes:
                print(
                    "Dense grid would exceed max RAM cap; reduce resolution or cap.",
                    file=sys.stderr,
                )
                sys.exit(1)
            dtype = np.float32 if args.sdf else np.uint8
            dense_grid = np.zeros((resolution, resolution, resolution), dtype=dtype)
            _log(f"Dense grid allocated ({dense_bytes / BYTES_IN_MB:.1f} MB).")

        if args.sdf:
            chunk_xy = adjust_chunk_xy(
                resolution,
                args.chunk_xy,
                max_ram_bytes,
                dtype_bytes=4,
                use_contains=True,
                use_sdf=True,
                use_bitpack=False,
            )
        else:
            chunk_xy = adjust_chunk_xy(
                resolution,
                args.chunk_xy,
                max_ram_bytes,
                dtype_bytes=1,
                use_contains=(mode == "solid"),
                use_sdf=False,
                use_bitpack=(args.format == "bitpack"),
            )
        if chunk_xy != args.chunk_xy:
            _log(f"Adjusted --chunk-xy to {chunk_xy} to fit RAM cap.")

        if args.sdf:
            est = estimate_stream_bytes(
                resolution,
                chunk_xy,
                dtype_bytes=4,
                use_contains=True,
                use_sdf=True,
                use_bitpack=False,
            )
        else:
            est = estimate_stream_bytes(
                resolution,
                chunk_xy,
                dtype_bytes=1,
                use_contains=(mode == "solid"),
                use_sdf=False,
                use_bitpack=(args.format == "bitpack"),
            )
        _log(f"Estimated streaming buffers: {est / BYTES_IN_MB:.1f} MB")

        out_base = os.path.join(args.out_dir, f"stair_{idx}")
        voxel_count = 0
        sdf_signed = True

        if args.sdf:
            out_path = (
                out_base + "_sdf.zarr"
                if args.format == "zarr"
                else out_base + "_sdf.npy"
            )
            output = (
                None
                if dense_grid is not None
                else open_output_array(
                    out_path,
                    (resolution, resolution, resolution),
                    np.float32,
                    args.format,
                    chunk_xy,
                    args.chunk_z,
                )
            )
            voxel_count, sdf_signed = sdf_stream(
                normalized,
                resolution,
                args.chunk_z,
                chunk_xy,
                output,
                dense_grid,
                log_every,
                max_ram_mb,
            )
            if dense_grid is not None:
                output = open_output_array(
                    out_path,
                    dense_grid.shape,
                    dense_grid.dtype,
                    args.format,
                    chunk_xy,
                    args.chunk_z,
                )
                output[:] = dense_grid
        else:
            if mode == "surface":
                indices = surface_voxel_indices(normalized, resolution)
                voxel_count = int(indices.shape[0])
                enforce_rss(max_ram_mb, "surface voxelization")
                if args.format == "npz_sparse":
                    write_sparse_npz(out_base + "_occ_sparse.npz", indices, (resolution, resolution, resolution))
                elif args.format == "bitpack":
                    write_bitpack_from_indices(
                        out_base + "_occ.bitpack",
                        indices,
                        resolution,
                        log_every,
                        max_ram_mb,
                    )
                elif args.format in ("zarr", "memmap"):
                    out_path = (
                        out_base + "_occ.zarr"
                        if args.format == "zarr"
                        else out_base + "_occ.npy"
                    )
                    output = (
                        None
                        if dense_grid is not None
                        else open_output_array(
                            out_path,
                            (resolution, resolution, resolution),
                            np.uint8,
                            args.format,
                            chunk_xy,
                            args.chunk_z,
                        )
                    )
                    if dense_grid is not None:
                        dense_grid.fill(0)
                        write_dense_from_indices(
                            dense_grid, indices, resolution, log_every, max_ram_mb
                        )
                        output = open_output_array(
                            out_path,
                            dense_grid.shape,
                            dense_grid.dtype,
                            args.format,
                            chunk_xy,
                            args.chunk_z,
                        )
                        output[:] = dense_grid
                    else:
                        write_dense_from_indices(
                            output, indices, resolution, log_every, max_ram_mb
                        )
                else:
                    raise ValueError(f"Unsupported format: {args.format}")
            else:
                if args.format == "npz_sparse":
                    writer = SparseIndexWriter(out_base + "_occ_sparse.npz")
                    output = writer
                elif args.format == "bitpack":
                    os.makedirs(args.out_dir, exist_ok=True)
                    output = open(out_base + "_occ.bitpack", "wb")
                elif args.format in ("zarr", "memmap"):
                    out_path = (
                        out_base + "_occ.zarr"
                        if args.format == "zarr"
                        else out_base + "_occ.npy"
                    )
                    output = (
                        None
                        if dense_grid is not None
                        else open_output_array(
                            out_path,
                            (resolution, resolution, resolution),
                            np.uint8,
                            args.format,
                            chunk_xy,
                            args.chunk_z,
                        )
                    )
                else:
                    raise ValueError(f"Unsupported format: {args.format}")
                voxel_count = solid_occupancy_stream(
                    normalized,
                    resolution,
                    args.chunk_z,
                    chunk_xy,
                    output,
                    args.format,
                    dense_grid,
                    log_every,
                    max_ram_mb,
                )
                if args.format == "npz_sparse":
                    voxel_count = output.finalize((resolution, resolution, resolution))
                if args.format == "bitpack":
                    output.close()
                if dense_grid is not None and args.format in ("zarr", "memmap"):
                    out_path = (
                        out_base + "_occ.zarr"
                        if args.format == "zarr"
                        else out_base + "_occ.npy"
                    )
                    output = open_output_array(
                        out_path,
                        dense_grid.shape,
                        dense_grid.dtype,
                        args.format,
                        chunk_xy,
                        args.chunk_z,
                    )
                    output[:] = dense_grid

        meta = {
            "component_index": idx,
            "faces": int(len(comp.faces)),
            "vertices": int(len(comp.vertices)),
            "bbox_min": bounds[0].tolist(),
            "bbox_max": bounds[1].tolist(),
            "extents": extents.tolist(),
            "normalized_bbox_min": norm_bounds[0].tolist(),
            "normalized_bbox_max": norm_bounds[1].tolist(),
            "center": center.tolist(),
            "scale": float(scale),
            "resolution": int(resolution),
            "pitch": float(pitch),
            "pitch_world": float(pitch * scale),
            "mode": mode,
            "format": args.format,
            "sdf": bool(args.sdf),
            "sdf_signed": bool(sdf_signed) if args.sdf else None,
            "watertight": is_watertight,
            "solid_requested": bool(args.mode == "solid"),
            "dense_requested": bool(args.dense),
            "voxels_written": int(voxel_count),
        }
        meta_path = out_base + "_meta.json"
        write_metadata(meta_path, meta)
        _log(f"Wrote {meta_path}")

    elapsed = time.time() - total_start
    _log(f"Done in {elapsed:.2f}s.")


if __name__ == "__main__":
    main()
