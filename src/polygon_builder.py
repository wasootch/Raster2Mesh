from typing import Dict, Optional

import numpy as np
from shapely.geometry import box as shapely_box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from .image_processor import ProcessedImage


def _mask_to_polygon(mask: np.ndarray, pixel_size: float) -> Optional[BaseGeometry]:
    """
    Convert a boolean mask to a Shapely polygon using row-run encoding.

    Each contiguous horizontal run of True pixels becomes one rectangle.
    Unioning row-runs (rather than individual pixel squares) keeps the box
    count proportional to image height × number of runs per row, which is
    orders of magnitude fewer objects than one box per pixel.

    Y coordinates are flipped so that the image origin is at the bottom-left
    in print-space (Y=0 at the bottom of the print, Y increases upward).
    """
    h, w = mask.shape
    boxes = []

    for r in range(h):
        row = mask[r]
        c = 0
        while c < w:
            if row[c]:
                c_start = c
                while c < w and row[c]:
                    c += 1
                # row r in image → y from (h-r-1)*ps to (h-r)*ps  (flip Y)
                y0 = (h - r - 1) * pixel_size
                y1 = (h - r) * pixel_size
                x0 = c_start * pixel_size
                x1 = c * pixel_size
                boxes.append(shapely_box(x0, y0, x1, y1))
            else:
                c += 1

    if not boxes:
        return None

    return unary_union(boxes)


def build_polygons(
    processed: ProcessedImage,
    pixel_size: float,
    simplify_tolerance: Optional[float] = None,
) -> Dict[int, BaseGeometry]:
    """
    Build a {color_index: shapely geometry} dict in millimetre coordinates.

    simplify_tolerance: if given, polygons are simplified with this tolerance
    (in mm) after union to reduce vertex count.  A good default is
    pixel_size * 0.5 which removes sub-pixel staircase detail.
    """
    result: Dict[int, BaseGeometry] = {}

    for color_idx in range(len(processed.palette)):
        mask = processed.color_indices == color_idx
        if not np.any(mask):
            continue

        poly = _mask_to_polygon(mask, pixel_size)
        if poly is None or poly.is_empty:
            continue

        if simplify_tolerance is not None and simplify_tolerance > 0:
            poly = poly.simplify(simplify_tolerance, preserve_topology=True)

        if not poly.is_empty:
            result[color_idx] = poly

    return result
