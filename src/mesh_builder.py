from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient as shapely_orient
from shapely.ops import unary_union

import mapbox_earcut as earcut


@dataclass
class ColorMesh:
    color_rgb: Tuple[int, int, int]
    vertices: np.ndarray   # shape (N, 3) float64, millimetres
    triangles: np.ndarray  # shape (T, 3) int32, vertex indices
    is_base: bool = False


# ---------------------------------------------------------------------------
# Geometry constants
# ---------------------------------------------------------------------------

# Holes whose area is below this threshold are dropped before triangulation.
# Sub-pixel holes are artefacts of floating-point geometry and confuse earcut,
# producing incomplete face triangulations and therefore open (non-manifold) edges.
_MIN_HOLE_AREA_MM2 = 0.01  # 0.1 mm × 0.1 mm — well below FDM resolution

# Sub-polygons (components of a MultiPolygon) whose area is below this threshold
# are dropped before extrusion.  Tiny slivers from the negative inset of thin
# color regions have near-zero width and earcut produces degenerate triangles for
# them, leaving open edges on the mesh.
_MIN_POLY_AREA_MM2 = 0.01  # same scale as the hole filter

# Additional "thin feature" filter: a sub-polygon is dropped if shrinking it by
# this extra amount (on top of the caller's inset) makes it disappear entirely.
# Such polygons are degenerate slivers whose width is ≤ 2 × _SLIVER_INSET_MM.
_SLIVER_INSET_MM = 0.005  # 5 µm — smaller than any FDM feature

# Tiny positive buffer applied to the base polygon union so that:
#   1. Near-touching sub-polygons are merged into one connected shape.
#   2. Sub-pixel holes from floating-point union artefacts are closed.
# Value is chosen to be smaller than the color inset (0.02 mm) so the base
# always covers the inset colour regions.
_BASE_CLEAN_BUFFER = 0.005  # mm

# Vertical gap between the top of color-layer meshes (Z = color_height) and
# the bottom of the base mesh.  Without this, both surfaces sit at exactly
# the same Z plane; slicers that merge geometry for repair find T-junction
# non-manifold edges where color-mesh boundary edges cut across base faces.
# 0.001 mm is far below any slicer weld tolerance and well below one layer
# height, so it creates no visible gap or weakness in the print.
_Z_GAP = 0.001  # mm


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _remove_collinear(
    ring: List[Tuple[float, float]], epsilon: float = 1e-9
) -> List[Tuple[float, float]]:
    """Remove vertices that lie exactly on the edge between their neighbours.

    When three or more consecutive ring vertices are collinear (cross-product ~0)
    earcut creates a diagonal that skips the intermediate points.  The skipped
    boundary sub-edges then appear in only one triangle → open (non-manifold) edges.
    Removing the redundant intermediate vertices gives earcut a clean ring.
    """
    if len(ring) < 3:
        return ring
    changed = True
    while changed:
        changed = False
        out: List[Tuple[float, float]] = []
        n = len(ring)
        for i in range(n):
            p0 = ring[(i - 1) % n]
            p1 = ring[i]
            p2 = ring[(i + 1) % n]
            ax, ay = p1[0] - p0[0], p1[1] - p0[1]
            bx, by = p2[0] - p0[0], p2[1] - p0[1]
            if abs(ax * by - ay * bx) > epsilon:
                out.append(p1)
            else:
                changed = True
        if len(out) >= 3:
            ring = out
        else:
            break  # degenerate — keep as-is and let the area check catch it
    return ring


def _ring_area(coords: List[Tuple[float, float]]) -> float:
    """Shoelace formula for signed ring area (positive = CCW)."""
    n = len(coords)
    total = 0.0
    for i in range(n):
        x0, y0 = coords[i]
        x1, y1 = coords[(i + 1) % n]
        total += x0 * y1 - x1 * y0
    return total * 0.5


def _triangulate_2d(
    exterior: List[Tuple[float, float]],
    interiors: List[List[Tuple[float, float]]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Triangulate a polygon (with optional holes) using the earcut algorithm.

    mapbox_earcut expects:
      arg0 – (N, 2) float64 array of all ring vertices concatenated
      arg1 – uint32 array of ring END indices (exclusive), last == N

    Returns (verts_2d, triangles) where:
      verts_2d  – (N, 2) float64 array
      triangles – (T, 3) int32 array of vertex indices into verts_2d
    """
    all_rings = [list(exterior)] + [list(h) for h in interiors]
    all_verts: List[Tuple[float, float]] = []
    ring_ends: List[int] = []
    for ring in all_rings:
        all_verts.extend(ring)
        ring_ends.append(len(all_verts))

    verts_np = np.array(all_verts, dtype=np.float64)        # shape (N, 2)
    ring_ends_np = np.array(ring_ends, dtype=np.uint32)     # last == N

    indices = earcut.triangulate_float64(verts_np, ring_ends_np)
    if len(indices) == 0:
        return verts_np, np.zeros((0, 3), dtype=np.int32)

    return verts_np, indices.reshape(-1, 3).astype(np.int32)


def _extrude_polygon(
    polygon: Polygon, z_bottom: float, z_top: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extrude a single shapely Polygon into a closed solid slab.

    Winding convention (right-hand rule, outward normals):
      Bottom face  → normal points -Z
      Top face     → normal points +Z
      Side walls   → normal points radially outward

    For both exterior (CCW) and interior/hole (CW in shapely) rings the same
    side-wall formula produces correct outward normals:
      - Exterior CCW edge A→B: right perpendicular points outside the solid ✓
      - Interior CW edge A→B: right perpendicular points into the hole ✓
    """
    # Normalise orientation: CCW exterior, CW interiors (shapely 2.x default,
    # but orient() makes it explicit and safe across versions).
    polygon = shapely_orient(polygon, sign=1.0)

    exterior = _remove_collinear(list(polygon.exterior.coords)[:-1])

    # Drop holes that are too small or degenerate.  Tiny holes (sub-pixel
    # artefacts from floating-point union operations) cause earcut to return
    # an incomplete triangulation, which leaves open edges on the mesh.
    interiors = [
        _remove_collinear(list(r.coords)[:-1])
        for r in polygon.interiors
        if len(list(r.coords)) >= 4                           # need ≥ 3 unique pts
        and abs(_ring_area(list(r.coords)[:-1])) >= _MIN_HOLE_AREA_MM2
    ]

    verts_2d, face_tris = _triangulate_2d(exterior, interiors)
    if len(face_tris) == 0:
        # Earcut failed to triangulate this polygon (degenerate or self-intersecting
        # ring after buffer/simplify).  Returning only side walls (no top/bottom cap)
        # would leave every side-wall boundary edge open.  Skip entirely.
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int32)
    n = len(verts_2d)

    # Build 3-D vertex arrays: first n = bottom layer, next n = top layer
    z_bot_col = np.full((n, 1), z_bottom)
    z_top_col = np.full((n, 1), z_top)
    verts_bottom = np.hstack([verts_2d, z_bot_col])
    verts_top = np.hstack([verts_2d, z_top_col])
    vertices = np.vstack([verts_bottom, verts_top])  # shape (2n, 3)

    tris: List[List[int]] = []

    # Bottom face: reverse winding so normal points -Z
    for tri in face_tris:
        tris.append([int(tri[2]), int(tri[1]), int(tri[0])])

    # Top face: keep winding so normal points +Z
    for tri in face_tris:
        tris.append([int(tri[0]) + n, int(tri[1]) + n, int(tri[2]) + n])

    # Side walls for every ring (exterior + kept holes)
    all_rings = [exterior] + interiors
    ring_starts: List[int] = []
    offset = 0
    for ring in all_rings:
        ring_starts.append(offset)
        offset += len(ring)

    for ring, rstart in zip(all_rings, ring_starts):
        ring_len = len(ring)
        for i in range(ring_len):
            j = (i + 1) % ring_len
            b0 = rstart + i       # bottom vertex i
            b1 = rstart + j       # bottom vertex j
            t0 = b0 + n           # top vertex i
            t1 = b1 + n           # top vertex j
            # Two triangles per quad wall panel; same formula for CCW exterior
            # and CW hole rings — the right-perpendicular of edge b0→b1 is
            # always the outward direction for the solid.
            tris.append([b0, b1, t1])
            tris.append([b0, t1, t0])

    return vertices, np.array(tris, dtype=np.int32)


def _weld_mesh(
    vertices: np.ndarray, triangles: np.ndarray, epsilon: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """Merge vertices at identical 3-D positions into a single vertex.

    After _extrude_geometry concatenates multiple sub-polygon meshes, boundary
    vertices that are shared between adjacent sub-polygons exist at the same 3-D
    coordinate but with different indices.  Welding them gives every shared edge
    exactly 2 incident triangles, fixing the open-edge non-manifold reports that
    slicers produce before they do their own vertex-merge step.

    Also removes degenerate triangles (two or more vertices coincident after
    welding) so they don't produce spurious single-occurrence edges.
    """
    if len(vertices) == 0 or len(triangles) == 0:
        return vertices, triangles

    # Snap vertices to a fine grid (1 nm) then find unique rows.
    rounded = np.round(vertices / epsilon).astype(np.int64)
    _, first, inverse = np.unique(rounded, axis=0, return_index=True, return_inverse=True)

    new_verts = vertices[first]
    new_tris = inverse.astype(np.int32)[triangles]

    # Drop triangles that collapsed to a line or point after welding.
    a, b, c = new_tris[:, 0], new_tris[:, 1], new_tris[:, 2]
    mask = (a != b) & (b != c) & (a != c)
    return new_verts, new_tris[mask]


def _extrude_geometry(
    geom: BaseGeometry, z_bottom: float, z_top: float, filter_slivers: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Extrude any Polygon or MultiPolygon geometry into a closed solid.

    filter_slivers: when True, drop sub-polygons that are too thin to
    triangulate reliably (area < _MIN_POLY_AREA_MM2, or ones that vanish after
    an extra _SLIVER_INSET_MM inset).  Use for color-layer geometry where thin
    features left by the negative inset produce degenerate earcut output.
    """
    polys = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]

    all_verts: List[np.ndarray] = []
    all_tris: List[np.ndarray] = []
    vert_offset = 0

    for poly in polys:
        if poly.is_empty or poly.area < 1e-6:
            continue
        if filter_slivers:
            if poly.area < _MIN_POLY_AREA_MM2:
                continue
            # A polygon that disappears after an extra tiny inset is a sliver
            # whose width is ≤ 2×_SLIVER_INSET_MM; earcut produces degenerate
            # triangles for it, leaving open edges on the mesh.
            if poly.buffer(-_SLIVER_INSET_MM).is_empty:
                continue
        v, t = _extrude_polygon(poly, z_bottom, z_top)
        if len(t) == 0:
            continue
        all_verts.append(v)
        all_tris.append(t + vert_offset)
        vert_offset += len(v)

    if not all_verts:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int32)

    return np.vstack(all_verts), np.vstack(all_tris)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_meshes(
    polygons: Dict[int, BaseGeometry],
    palette: List[Tuple[int, int, int]],
    color_height: float,
    base_height: float,
    base_color_index: int = 0,
    color_inset: float = 0.02,
) -> List[ColorMesh]:
    """
    Build ColorMesh objects for every color layer and the structural base.

    Layer layout (Z increases upward from build plate):
      Z = 0                        ... color_height       -> color layers
      Z = color_height + _Z_GAP   ... + base_height      -> solid base

    _Z_GAP (0.001 mm) separates the color-layer tops from the base bottom so
    they never share a coplanar surface.  Without it, slicers that merge
    objects for manifold repair find T-junction edges where the color mesh
    boundary edges cut across the base face triangulation.

    color_inset: each color polygon is shrunk inward before extrusion to
    prevent adjacent color meshes from sharing coincident side-wall edges.
    """
    meshes: List[ColorMesh] = []

    # --- Color layer meshes ---
    for color_idx, geom in sorted(polygons.items()):
        # Shrink inward to eliminate coincident edges with neighbouring colors.
        inset = geom.buffer(-color_inset, join_style='mitre', mitre_limit=5.0)
        # Fix any geometry invalidity introduced by the buffer.
        if not inset.is_valid:
            inset = inset.buffer(0)
        if inset is None or inset.is_empty:
            continue

        v, t = _extrude_geometry(inset, 0.0, color_height, filter_slivers=True)
        if len(t) == 0:
            continue
        v, t = _weld_mesh(v, t)
        if len(t) == 0:
            continue
        meshes.append(ColorMesh(
            color_rgb=palette[color_idx],
            vertices=v,
            triangles=t,
            is_base=False,
        ))

    # --- Structural base mesh ---
    if polygons and base_height > 0:
        base_geom = unary_union(list(polygons.values()))

        # Small positive buffer closes sub-pixel holes and merges near-touching
        # sub-polygon components that would otherwise create non-manifold
        # vertices (pinch points) in the extruded mesh.
        base_geom = base_geom.buffer(_BASE_CLEAN_BUFFER)
        if not base_geom.is_valid:
            base_geom = base_geom.buffer(0)

        # _Z_GAP lifts the base above the color-layer top plane so no two
        # objects share a coincident surface at Z = color_height.
        z_base_bottom = color_height + _Z_GAP
        z_base_top = z_base_bottom + base_height

        v, t = _extrude_geometry(base_geom, z_base_bottom, z_base_top)
        if len(t) > 0:
            v, t = _weld_mesh(v, t)
        if len(t) > 0:
            safe_idx = min(base_color_index, len(palette) - 1)
            meshes.append(ColorMesh(
                color_rgb=palette[safe_idx],
                vertices=v,
                triangles=t,
                is_base=True,
            ))

    return meshes
