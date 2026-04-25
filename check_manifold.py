#!/usr/bin/env python3
"""
Analyse an OBJ file for non-manifold edges, reporting per-object statistics.

Vertices are welded by position (1 nm grid) before counting so the result
matches what a slicer sees after its own merge-duplicate-vertices step.

Usage:
  python check_manifold.py output.obj [--detail]
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path


def check_obj(path: str, detail: bool = False) -> int:
    """Return the total non-manifold edge count across all objects."""
    obj_path = Path(path)
    if not obj_path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return -1

    vertices: list = []
    current_object = "__unnamed__"
    object_faces: dict = defaultdict(list)
    object_vert_count: dict = defaultdict(int)
    object_tri_count: dict = defaultdict(int)

    with open(obj_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("o "):
                current_object = line[2:].strip()
            elif line.startswith("v "):
                parts = line.split()
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
                object_vert_count[current_object] += 1
            elif line.startswith("f "):
                parts = line.split()[1:]
                indices = [int(p.split("/")[0]) - 1 for p in parts]
                object_faces[current_object].append(indices)
                object_tri_count[current_object] += 1

    if not object_faces:
        print("No faces found in the file.")
        return 0

    verts_arr = vertices  # list of (x, y, z)

    total_nm = 0
    print(f"\nManifold analysis: {obj_path.name}  (position-welded)\n")
    print(f"  {'Object':<40} {'Verts':>8} {'Tris':>8} {'Non-manifold edges':>20}")
    print(f"  {'-'*40} {'-'*8} {'-'*8} {'-'*20}")

    for obj_name in sorted(object_faces):
        faces = object_faces[obj_name]

        # Collect all vertex indices used by this object and build position-weld map.
        used_raw = set()
        for face in faces:
            used_raw.update(face)

        epsilon = 1e-6  # 1 nm weld tolerance (mm)
        pos_to_welded: dict = {}
        raw_to_welded: dict = {}
        for raw_idx in sorted(used_raw):
            x, y, z = verts_arr[raw_idx]
            key = (round(x / epsilon), round(y / epsilon), round(z / epsilon))
            if key not in pos_to_welded:
                pos_to_welded[key] = raw_idx  # canonical index = first seen
            raw_to_welded[raw_idx] = pos_to_welded[key]

        # Build welded faces and count edges.
        edge_count: Counter = Counter()
        # Also track which original face each edge came from (for detail reporting)
        edge_faces: dict = defaultdict(list)

        for fi, face in enumerate(faces):
            welded = [raw_to_welded[v] for v in face]
            n = len(welded)
            # Skip degenerate faces (any two vertices the same after welding)
            if len(set(welded)) < n:
                continue
            for i in range(n):
                a = welded[i]
                b = welded[(i + 1) % n]
                edge = (min(a, b), max(a, b))
                edge_count[edge] += 1
                if detail:
                    edge_faces[edge].append(fi)

        nm_edges = {e: c for e, c in edge_count.items() if c != 2}
        n_open = sum(1 for c in nm_edges.values() if c == 1)
        n_multi = sum(1 for c in nm_edges.values() if c > 2)
        n_nm = len(nm_edges)

        t_count = object_tri_count[obj_name]
        w_count = len(pos_to_welded)  # unique positions after welding

        if n_nm == 0:
            status = "OK"
        else:
            parts_s = []
            if n_open:
                parts_s.append(f"{n_open} open")
            if n_multi:
                parts_s.append(f"{n_multi} T-junc")
            status = f"{n_nm} ({', '.join(parts_s)})"

        print(f"  {obj_name:<40} {w_count:>8,} {t_count:>8,} {status:>20}")
        total_nm += n_nm

        if detail and nm_edges:
            shown = 0
            for edge, count in sorted(nm_edges.items())[:20]:
                a, b = edge
                pa = verts_arr[a]
                pb = verts_arr[b]
                kind = "open" if count == 1 else f"count={count}"
                print(f"      {kind}  ({pa[0]:.4f},{pa[1]:.4f},{pa[2]:.4f})"
                      f" -- ({pb[0]:.4f},{pb[1]:.4f},{pb[2]:.4f})")
                shown += 1
            if len(nm_edges) > 20:
                print(f"      ... and {len(nm_edges) - 20} more")

    print()
    print(f"  Total non-manifold edges: {total_nm}")
    print()
    if total_nm > 0:
        print("Legend:")
        print("  open   = edge shared by only 1 triangle (missing neighbour)")
        print("  T-junc = edge shared by 3+ triangles (overlapping geometry)")

    return total_nm


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print("Usage: check_manifold.py <file.obj> [--detail]")
        sys.exit(1)
    detail_flag = "--detail" in args
    path_arg = next(a for a in args if not a.startswith("--"))
    result = check_obj(path_arg, detail=detail_flag)
    sys.exit(0 if result == 0 else 1)
