#!/usr/bin/env python3
"""
Raster2Mesh – Convert a raster image (PNG, JPG, …) to a multi-color flat
OBJ or 3MF file for FDM printing.

Print face-down for best surface quality.  Only the first few layers carry
color (one filament change per color); the remaining layers form a solid
structural base in a single color.

Usage examples
--------------
  python raster2mesh.py logo.png
  python raster2mesh.py logo.png -o logo.obj --width 80 --color-layers 3
  python raster2mesh.py logo.png --format 3mf --white-threshold 230 --colors 8
"""

import argparse
import sys
from pathlib import Path

from src.image_processor import load_and_process
from src.mesh_builder import build_meshes
from src.obj_writer import write_obj
from src.polygon_builder import build_polygons
from src.threemf_writer import write_3mf


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a raster image to a multi-color flat OBJ/3MF for FDM printing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", help="Input image file (PNG, JPG, ...)")
    parser.add_argument("-o", "--output", help="Output file path (default: <input>.<format>)")
    parser.add_argument("--format", choices=["obj", "3mf"], default="obj",
                        help="Output format")

    geo = parser.add_argument_group("geometry")
    geo.add_argument("--width", type=float, default=200.0,
                     help="Print width in mm")
    geo.add_argument("--layer-height", type=float, default=0.2,
                     help="Layer height in mm")
    geo.add_argument("--color-layers", type=int, default=2,
                     help="Number of layers that carry color (printed first, face-down)")
    geo.add_argument("--base-layers", type=int, default=15,
                     help="Number of solid base layers added for strength")
    geo.add_argument("--base-color-index", type=int, default=0,
                     help="Palette index of the color used for the base")

    img = parser.add_argument_group("image processing")
    img.add_argument("--colors", type=int, default=16,
                     help="Maximum number of colors after quantization")
    img.add_argument("--white-threshold", type=int, default=240,
                     help="Pixels with R, G, B all >= this value are treated as background")
    img.add_argument("--simplify", type=float, default=None,
                     help="Polygon simplification tolerance in mm "
                          "(default: pixel_size * 0.5).  Pass 0 to disable.")
    img.add_argument("--no-mirror", action="store_true",
                     help="Disable the default horizontal mirror "
                          "(mirror is on by default because face-down printing reverses left/right)")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    fmt = args.format
    output_path = Path(args.output) if args.output else input_path.with_suffix(f".{fmt}")

    # ------------------------------------------------------------------
    # 1. Load and process the image
    # ------------------------------------------------------------------
    mirror = not args.no_mirror
    print(f"Loading:  {input_path}  (mirror={'yes' if mirror else 'no'})")
    processed = load_and_process(
        str(input_path),
        max_colors=args.colors,
        white_threshold=args.white_threshold,
        mirror=mirror,
    )

    if not processed.palette:
        print("Error: no foreground pixels found after background removal.", file=sys.stderr)
        print("Try lowering --white-threshold.", file=sys.stderr)
        sys.exit(1)

    n_colors = len(processed.palette)
    print(f"Colors:   {n_colors}")
    for i, (r, g, b) in enumerate(processed.palette):
        print(f"  [{i}] #{r:02X}{g:02X}{b:02X}  rgb({r},{g},{b})")

    # ------------------------------------------------------------------
    # 2. Compute dimensions
    # ------------------------------------------------------------------
    h, w = processed.color_indices.shape
    pixel_size = args.width / w
    print_h = h * pixel_size

    if pixel_size < 0.1:
        print(
            f"Warning: pixel size is {pixel_size:.3f} mm which is below typical FDM "
            f"resolution (~0.1 mm). Consider resizing the image to ~{int(args.width / 0.1)}px wide.",
            file=sys.stderr,
        )

    print(f"Size:     {w}x{h} px -> {args.width:.1f}x{print_h:.1f} mm  ({pixel_size:.4f} mm/px)")

    color_height = args.color_layers * args.layer_height
    base_height  = args.base_layers  * args.layer_height
    print(f"Layers:   {args.color_layers} color ({color_height:.2f} mm)  +  "
          f"{args.base_layers} base ({base_height:.2f} mm)  =  "
          f"{color_height + base_height:.2f} mm total")

    # ------------------------------------------------------------------
    # 3. Build polygons from color masks
    # ------------------------------------------------------------------
    simplify_tol = args.simplify
    if simplify_tol is None:
        simplify_tol = pixel_size * 0.5  # sensible default: half a pixel

    print("Building polygons...")
    polygons = build_polygons(processed, pixel_size, simplify_tolerance=simplify_tol)
    print(f"  {len(polygons)} color region(s) vectorised")

    # ------------------------------------------------------------------
    # 4. Extrude polygons into 3-D meshes
    # ------------------------------------------------------------------
    print("Building meshes...")
    base_color_idx = min(args.base_color_index, n_colors - 1)
    meshes = build_meshes(
        polygons,
        processed.palette,
        color_height=color_height,
        base_height=base_height,
        base_color_index=base_color_idx,
    )

    for i, mesh in enumerate(meshes):
        r, g, b = mesh.color_rgb
        label = "base " if mesh.is_base else "color"
        print(f"  [{i}] {label} #{r:02X}{g:02X}{b:02X}  "
              f"{len(mesh.vertices):,} verts  {len(mesh.triangles):,} tris")

    # ------------------------------------------------------------------
    # 5. Write output
    # ------------------------------------------------------------------
    print(f"Writing:  {output_path}")
    if fmt == "obj":
        write_obj(str(output_path), meshes)
    else:
        write_3mf(str(output_path), meshes)
    print("Done.")

    print()
    print("Next steps:")
    if fmt == "obj":
        print("  1. Open the .obj in your slicer (Bambu Studio, PrusaSlicer, Orca).")
    else:
        print("  1. Open the .3mf in your slicer (Bambu Studio, PrusaSlicer, Orca).")
    print("  2. Assign each color object to the correct extruder / AMS slot.")
    print("  3. Print face-down -- the smooth build-plate side becomes the visible image.")


if __name__ == "__main__":
    main()
