#!/usr/bin/env python3
import argparse
import os
import sys


def rewrite_map_line(line, key, new_name):
    stripped = line.lstrip()
    prefix = line[: len(line) - len(stripped)]
    tokens = stripped.split()
    if not tokens or tokens[0] != key:
        return line
    if len(tokens) == 1:
        return f"{prefix}{key} {new_name}\n"
    # Preserve any map options (MTL options are typically before the filename).
    options = tokens[1:-1] if len(tokens) > 2 else []
    return f"{prefix}{' '.join([key] + options + [new_name])}\n"


def patch_mtl(in_path, out_path):
    with open(in_path, "r", errors="ignore") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        updated = line
        for key in ("map_Kd", "map_Ka"):
            updated = rewrite_map_line(updated, key, "Stair0CompleteMap.jpg")
        for key in ("bump", "map_bump", "map_Bump", "norm", "map_norm"):
            updated = rewrite_map_line(updated, key, "Stair0NormalsMap.jpg")
        new_lines.append(updated)

    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)


def main():
    parser = argparse.ArgumentParser(
        description="Rewrite MTL texture paths to local filenames."
    )
    parser.add_argument("--in", dest="in_path", default="Stair.mtl")
    parser.add_argument("--out", dest="out_path", default=None)
    args = parser.parse_args()

    if not os.path.exists(args.in_path):
        print(f"MTL not found: {args.in_path}", file=sys.stderr)
        sys.exit(1)

    out_path = args.out_path or args.in_path
    patch_mtl(args.in_path, out_path)
    print(f"Patched MTL written to: {out_path}")


if __name__ == "__main__":
    main()
