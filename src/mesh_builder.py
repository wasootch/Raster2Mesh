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
# Internal helpers
# ---------------------------------------------------------------------------

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

    exterior = list(polygon.exterior.coords)[:-1]   # drop repeated closing pt
    interiors = [list(r.coords)[:-1] for r in polygon.interiors]

    verts_2d, face_tris = _triangulate_2d(exterior, interiors)
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

    # Side walls for every ring (exterior + all holes)
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


def _extrude_geometry(
    geom: BaseGeometry, z_bottom: float, z_top: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Extrude any Polygon or MultiPolygon geometry into a closed solid."""
    polys = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]

    all_verts: List[np.ndarray] = []
    all_tris: List[np.ndarray] = []
    vert_offset = 0

    for poly in polys:
        if poly.is_empty or poly.area == 0:
            continue
        v, t = _extrude_polygon(poly, z_bottom, z_top)
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
      Z = 0               ... color_height       -> colored image layers
      Z = color_height    ... color_height + base_height -> solid base layer

    Printing face-down means Z=0 hits the smooth build plate first, giving
    the best surface finish on the visible image face.

    color_inset: each color polygon is shrunk inward by this amount (mm)
    before extrusion.  This prevents adjacent color meshes from sharing
    coincident edges, which would create non-manifold geometry.  The gap is
    well below FDM nozzle diameter so it is invisible in the print.
    """
    meshes: List[ColorMesh] = []

    # --- Color layer meshes ---
    for color_idx, geom in sorted(polygons.items()):
        # Shrink inward to eliminate coincident edges with neighbouring colors.
        # join_style='mitre' preserves sharp logo corners.
        inset = geom.buffer(-color_inset, join_style='mitre', mitre_limit=5.0)
        if inset is None or inset.is_empty:
            continue

        v, t = _extrude_geometry(inset, 0.0, color_height)
        if len(t) == 0:
            continue
        meshes.append(ColorMesh(
            color_rgb=palette[color_idx],
            vertices=v,
            triangles=t,
            is_base=False,
        ))

    # --- Structural base mesh (union footprint of all colors, not inset) ---
    if polygons and base_height > 0:
        base_geom = unary_union(list(polygons.values()))
        z_base_top = color_height + base_height
        v, t = _extrude_geometry(base_geom, color_height, z_base_top)
        if len(t) > 0:
            safe_idx = min(base_color_index, len(palette) - 1)
            meshes.append(ColorMesh(
                color_rgb=palette[safe_idx],
                vertices=v,
                triangles=t,
                is_base=True,
            ))

    return meshes
