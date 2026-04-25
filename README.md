# Raster2Mesh

Convert a raster image (PNG, JPG, etc.) into a multi-color flat OBJ or 3MF file for FDM 3D printing. Each color in the image becomes a separate mesh object, so you can assign filaments in Bambu Studio, PrusaSlicer, or Orca and print a full-color medallion or badge.

My kid's softball team came up with a logo and I couldn't find a tool that would convert it to a printable mesh while keeping the colors — existing tools produced reliefs or lost color information entirely. This tool does a flat depth extrusion of each color region and outputs a proper multi-body file ready for a multi-filament printer.

## How it works

1. **Background removal** — transparent pixels and near-white pixels reachable from the image border are treated as background and excluded. Interior white regions (e.g. white text inside a dark logo) are preserved as a foreground color.
2. **Color quantization** — foreground pixels are quantized to at most `--colors` colors (default 16) using median-cut.
3. **Polygon building** — each color region is vectorized into Shapely polygons, optionally simplified to reduce triangle count.
4. **Mesh extrusion** — each color polygon is inset slightly and extruded into a closed solid. A solid base slab is added underneath for strength.
5. **Output** — the mesh is written as OBJ (with MTL) or 3MF. Each color is a separate named object/component so your slicer can assign it to an AMS slot.

## Print orientation

Print **face-down**. The smooth build-plate surface becomes the visible front of the print. Because face-down printing mirrors left and right, the image is automatically flipped horizontally before processing (disable with `--no-mirror`).

## Layer layout

```
Z = 0 ... color_height          color layers  (one filament per color)
Z = color_height ... total      solid base    (single filament, for strength)
```

## Installation

```bash
pip install -r requirements.txt
```

Requirements: Python 3.10+, Pillow, NumPy, Shapely, mapbox-earcut.

## Usage

```bash
# Basic — 200 mm wide, up to 16 colors, OBJ output
python raster2mesh.py logo.png

# Custom width and color count
python raster2mesh.py logo.png --width 80 --colors 4

# 3MF output (better for Bambu Studio)
python raster2mesh.py logo.png --format 3mf

# Adjust layer counts and layer height
python raster2mesh.py logo.png --color-layers 3 --base-layers 20 --layer-height 0.2

# Tweak background detection (lower value = more aggressive white removal)
python raster2mesh.py logo.png --white-threshold 230
```

## Options

| Option | Default | Description |
|---|---|---|
| `--format` | `obj` | Output format: `obj` or `3mf` |
| `--width` | `200.0` | Print width in mm |
| `--layer-height` | `0.2` | Layer height in mm |
| `--color-layers` | `2` | Number of layers that carry color |
| `--base-layers` | `15` | Number of solid base layers |
| `--base-color-index` | `0` | Palette index of the base color |
| `--colors` | `16` | Maximum colors after quantization |
| `--white-threshold` | `240` | R, G, B all ≥ this → background candidate |
| `--simplify` | `0.5 × pixel_size` | Polygon simplification tolerance in mm |
| `--no-mirror` | off | Disable automatic horizontal flip |

## Workflow

1. Run `raster2mesh.py` to generate the OBJ or 3MF.
2. Open the file in your slicer.
3. Assign each color object to the correct AMS slot / extruder.
4. Slice with the print plate facing down.

## Manifold checking

`check_manifold.py` analyses an OBJ for non-manifold edges, matching what a slicer sees after its own vertex-merge step:

```bash
python check_manifold.py output.obj
python check_manifold.py output.obj --detail   # show coordinates of bad edges
```
