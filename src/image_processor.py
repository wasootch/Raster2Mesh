from collections import deque
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from PIL import Image


@dataclass
class ProcessedImage:
    color_indices: np.ndarray  # shape (H, W), dtype int16, -1 = background
    palette: List[Tuple[int, int, int]]  # RGB tuples, one per color index


def _edge_connected_white(near_white: np.ndarray) -> np.ndarray:
    """
    Return a mask of near-white pixels reachable from any image edge via
    4-connectivity.  Interior near-white regions (white text, white fills
    inside a logo) are NOT included because they are surrounded by darker
    pixels that block the flood fill.
    """
    h, w = near_white.shape
    bg = np.zeros((h, w), dtype=bool)
    queue: deque = deque()

    def _seed(r: int, c: int) -> None:
        if near_white[r, c] and not bg[r, c]:
            bg[r, c] = True
            queue.append((r, c))

    for c in range(w):
        _seed(0, c)
        _seed(h - 1, c)
    for r in range(1, h - 1):
        _seed(r, 0)
        _seed(r, w - 1)

    while queue:
        r, c = queue.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and near_white[nr, nc] and not bg[nr, nc]:
                bg[nr, nc] = True
                queue.append((nr, nc))

    return bg


def load_and_process(
    image_path: str,
    max_colors: int = 4,
    white_threshold: int = 240,
    alpha_threshold: int = 10,
    mirror: bool = True,
) -> ProcessedImage:
    """
    Load an image, remove the background, and quantize to at most max_colors colors.

    mirror=True flips the image horizontally so it reads correctly after the
    print is lifted off the build plate face-up (face-down printing reverses
    left and right).

    Background detection:
      - Transparent pixels (alpha < alpha_threshold) are always background.
      - Near-white pixels are background ONLY if reachable from the image
        border via flood fill.  Interior white (e.g. white text inside a
        dark logo shape) is preserved as a foreground color.

    Quantization runs on foreground pixels only, so the background never
    consumes a color slot and similar colors (e.g. shades of brown) are
    merged when max_colors is lower than the natural color count.
    """
    img = Image.open(image_path).convert("RGBA")

    if mirror:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    pixels = np.array(img, dtype=np.uint8)  # H x W x 4
    h, w = pixels.shape[:2]

    r, g, b, a = pixels[..., 0], pixels[..., 1], pixels[..., 2], pixels[..., 3]

    transparent = a < alpha_threshold
    near_white = (r >= white_threshold) & (g >= white_threshold) & (b >= white_threshold)

    # Only the border-connected portion of near_white is background.
    # Opaque near-white pixels inside the logo are treated as foreground.
    bg_mask = transparent | _edge_connected_white(near_white & ~transparent)

    fg_rows, fg_cols = np.where(~bg_mask)
    fg_pixels_rgb = pixels[fg_rows, fg_cols, :3]  # (N_fg, 3)

    if len(fg_pixels_rgb) == 0:
        return ProcessedImage(color_indices=np.full((h, w), -1, dtype=np.int16), palette=[])

    # Quantize foreground pixels only — presented as a 1-row image so PIL
    # treats every pixel equally regardless of spatial position.
    fg_img = Image.fromarray(fg_pixels_rgb.reshape(1, -1, 3), "RGB")
    fg_quantized = fg_img.quantize(colors=max_colors, method=Image.Quantize.MEDIANCUT)

    palette_flat = fg_quantized.getpalette()
    fg_q_flat = np.array(fg_quantized, dtype=np.int16).flatten()  # palette index per fg pixel

    # Build palette for the indices that were actually assigned
    n_colors = int(fg_q_flat.max()) + 1
    full_palette = [
        (palette_flat[i * 3], palette_flat[i * 3 + 1], palette_flat[i * 3 + 2])
        for i in range(n_colors)
    ]

    # Compact: squeeze out any unused palette slots
    unique_indices = np.unique(fg_q_flat)
    lut = np.zeros(n_colors, dtype=np.int16)
    for new_idx, old_idx in enumerate(unique_indices):
        lut[old_idx] = new_idx
    new_palette = [full_palette[i] for i in unique_indices]

    remapped_fg = lut[fg_q_flat]

    color_indices = np.full((h, w), -1, dtype=np.int16)
    color_indices[fg_rows, fg_cols] = remapped_fg

    return ProcessedImage(color_indices=color_indices, palette=new_palette)
