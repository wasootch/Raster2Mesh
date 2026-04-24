"""
Write a list of ColorMesh objects to an OBJ + MTL file pair.

OBJ and MTL are plain-text formats readable in any editor.  Each ColorMesh
becomes a named object (o) in the OBJ file with its own material entry in
the MTL file, so slicers can assign each object to a different extruder /
AMS slot independently.

OBJ uses 1-based vertex indices and accumulates vertex offsets across objects.
"""

from pathlib import Path
from typing import List

from .mesh_builder import ColorMesh


def write_obj(output_path: str, meshes: List[ColorMesh]) -> None:
    """Write meshes to <output_path>.obj and <output_path>.mtl."""
    obj_path = Path(output_path)
    mtl_path = obj_path.with_suffix(".mtl")

    # --- MTL file ---
    with open(mtl_path, "w", encoding="utf-8") as f:
        f.write("# Raster2Mesh material library\n\n")
        for i, mesh in enumerate(meshes):
            r, g, b = mesh.color_rgb
            label = "base" if mesh.is_base else f"color{i}"
            f.write(f"newmtl {label}\n")
            f.write(f"Kd {r/255:.6f} {g/255:.6f} {b/255:.6f}\n")
            f.write("Ka 0.000000 0.000000 0.000000\n")
            f.write("Ks 0.000000 0.000000 0.000000\n")
            f.write("\n")

    # --- OBJ file ---
    with open(obj_path, "w", encoding="utf-8") as f:
        f.write("# Raster2Mesh\n")
        f.write(f"mtllib {mtl_path.name}\n\n")

        vert_offset = 1  # OBJ vertex indices are 1-based and global

        for i, mesh in enumerate(meshes):
            label = "base" if mesh.is_base else f"color{i}"
            r, g, b = mesh.color_rgb
            f.write(f"o {label}_{r:02X}{g:02X}{b:02X}\n")
            f.write(f"usemtl {label}\n")

            for v in mesh.vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            for tri in mesh.triangles:
                v1 = int(tri[0]) + vert_offset
                v2 = int(tri[1]) + vert_offset
                v3 = int(tri[2]) + vert_offset
                f.write(f"f {v1} {v2} {v3}\n")

            f.write("\n")
            vert_offset += len(mesh.vertices)
