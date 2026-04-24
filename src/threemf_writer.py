"""
Write a list of ColorMesh objects to a 3MF file.

3MF is a ZIP archive containing:
  [Content_Types].xml   – MIME type declarations
  _rels/.rels           – package relationship to the model
  3D/3dmodel.model      – the actual XML geometry + materials

Each color becomes a <basematerials> entry; each mesh becomes a separate
<object> that references the matching material via pid/pindex.  Slicers
(Bambu Studio, PrusaSlicer, Orca) read this and let the user assign each
object to an extruder / AMS slot.
"""

import zipfile
from typing import List

from .mesh_builder import ColorMesh


_CONTENT_TYPES = """\
<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>
</Types>"""

_RELS = """\
<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rel0" Target="/3D/3dmodel.model" Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>
</Relationships>"""


def _rgb_to_hex(rgb: tuple) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def _build_model_xml(meshes: List[ColorMesh]) -> bytes:
    # Collect unique colors in encounter order
    color_order: List[tuple] = []
    color_index: dict = {}
    for mesh in meshes:
        if mesh.color_rgb not in color_index:
            color_index[mesh.color_rgb] = len(color_order)
            color_order.append(mesh.color_rgb)

    lines: List[str] = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<model unit="millimeter" xml:lang="en-US"')
    lines.append('  xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">')
    lines.append('  <metadata name="Application">Raster2Mesh</metadata>')
    lines.append('  <resources>')

    # Material definitions
    lines.append('    <basematerials id="1">')
    for i, rgb in enumerate(color_order):
        label = "Base" if i == color_index.get(meshes[-1].color_rgb if meshes else rgb) and meshes and meshes[-1].is_base else f"Color{i}"
        lines.append(f'      <base name="{label}" displaycolor="{_rgb_to_hex(rgb)}"/>')
    lines.append('    </basematerials>')

    # One <object> per mesh
    obj_ids: List[int] = []
    for mesh_idx, mesh in enumerate(meshes):
        obj_id = mesh_idx + 2   # resource IDs; 1 is taken by basematerials
        obj_ids.append(obj_id)
        pidx = color_index[mesh.color_rgb]

        lines.append(f'    <object id="{obj_id}" type="model" pid="1" pindex="{pidx}">')
        lines.append('      <mesh>')

        lines.append('        <vertices>')
        for v in mesh.vertices:
            lines.append(f'          <vertex x="{v[0]:.6f}" y="{v[1]:.6f}" z="{v[2]:.6f}"/>')
        lines.append('        </vertices>')

        lines.append('        <triangles>')
        for tri in mesh.triangles:
            lines.append(f'          <triangle v1="{tri[0]}" v2="{tri[1]}" v3="{tri[2]}"/>')
        lines.append('        </triangles>')

        lines.append('      </mesh>')
        lines.append('    </object>')

    lines.append('  </resources>')
    lines.append('  <build>')
    for obj_id in obj_ids:
        lines.append(f'    <item objectid="{obj_id}"/>')
    lines.append('  </build>')
    lines.append('</model>')

    return '\n'.join(lines).encode('utf-8')


def write_3mf(output_path: str, meshes: List[ColorMesh]) -> None:
    """Write meshes to a 3MF file at output_path."""
    model_bytes = _build_model_xml(meshes)

    with zipfile.ZipFile(output_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('[Content_Types].xml', _CONTENT_TYPES)
        zf.writestr('_rels/.rels', _RELS)
        zf.writestr('3D/3dmodel.model', model_bytes)
