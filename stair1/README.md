# Stair Mesh Voxelization

Memory-bounded voxelization utilities for photogrammetry OBJ meshes (stairs).
Supports surface and solid occupancy as well as SDF output in streaming chunks.

Assumptions and caveats:
- Solid occupancy relies on ray parity; non-watertight meshes will auto-fallback to surface mode.
- SDF sign uses `mesh.contains`; if it fails, unsigned distances are written with a warning.

## Requirements
- Python 3.10+

Install dependencies:
```bash
python -m pip install -r requirements.txt
```

## Patch the MTL
Rewrites texture paths to local filenames.
```bash
python patch_mtl.py --in Stair.mtl
```
This updates `Stair.mtl` in-place. Use `--out` to write a new file:
```bash
python patch_mtl.py --in Stair.mtl --out Stair_patched.mtl
```

## Memory-Safe Voxelization
All outputs are stream-written to disk; no full `(r,r,r)` grids are allocated
unless you explicitly use `--dense` at `r<=256`.

Useful options:
- `--min-component-faces`: discard small fragments before keeping the largest.
- `--keep-components`: number of largest components to keep (default 2).
- `--chunk-z`, `--chunk-xy`: streaming chunk sizes.
- `--max-ram-mb`: RAM cap (default is ~70% of available RAM).
- `--process`: let trimesh clean the mesh on load.
- `--self-test`: run a quick 128^3 pipeline and report memory/runtime.

Surface occupancy -> sparse NPZ (recommended default):
```bash
python memory_safe_voxelize.py --obj Stair.obj --out-dir output --resolution 256 --mode surface --format npz_sparse
```

Surface occupancy -> bit-packed slices:
```bash
python memory_safe_voxelize.py --obj Stair.obj --out-dir output --resolution 256 --mode surface --format bitpack
```

Solid occupancy -> Zarr:
```bash
python memory_safe_voxelize.py --obj Stair.obj --out-dir output --resolution 256 --mode solid --format zarr
```

SDF -> Zarr (float32, chunked):
```bash
python memory_safe_voxelize.py --obj Stair.obj --out-dir output --resolution 256 --sdf --format zarr
```

Self-test (forces `resolution=128` and prints memory estimates):
```bash
python memory_safe_voxelize.py --obj Stair.obj --out-dir output --self-test
```

## Outputs
For each kept component (stair):
- `output/stair_i_meta.json`: metadata (bounds, scale, pitch, counts).
- One voxel output file in the selected format:
  - `stair_i_occ_sparse.npz` (COO indices + shape)
  - `stair_i_occ.bitpack` (packed occupancy slices)
  - `stair_i_occ.zarr` / `stair_i_occ.npy` (dense on disk, streamed)
  - `stair_i_sdf.zarr` / `stair_i_sdf.npy` (SDF)

`stair_i_meta.json` includes:
- `center`, `scale`: normalization parameters.
- `pitch`: voxel pitch in normalized space (unit cube).
- `pitch_world`: pitch in original mesh units.
- `mode`, `format`: chosen pipeline and storage.

Mapping between original and normalized coordinates:
- `p_norm = (p_world - center) / scale + 0.5`
- `p_world = (p_norm - 0.5) * scale + center`

## Notes on Memory
Dense grids explode quickly:
- `512^3` voxels ≈ 134M values
  - uint8 occupancy: ~128 MB
  - float32 SDF: ~512 MB
- `1024^3` voxels ≈ 1.07B values
  - uint8 occupancy: ~1.0 GB
  - float32 SDF: ~4.0 GB

This pipeline avoids full dense allocations by streaming per-slice and writing
directly to disk formats like Zarr, memmap, or sparse NPZ.
