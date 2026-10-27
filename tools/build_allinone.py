"""Build ``mesh_to_points_allinone.py``: the whole package merged into one standalone file.

    .venv/bin/python tools/build_allinone.py

The modules are concatenated in dependency order; relative imports are dropped (every
name is global in the single file), absolute imports are hoisted to the top, module
docstrings become section banners, and the two command line entry points are merged
(``python mesh_to_points_allinone.py shapes ...`` generates the test meshes).
"""
from __future__ import annotations

import ast
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
PKG = ROOT / "mesh_to_points"
OUT = ROOT / "mesh_to_points_allinone.py"
ORDER = ["geom", "mesh_io", "segment", "geodesic", "contours", "normals", "order",
         "sample", "pipeline", "viz", "shapes", "cli"]
RENAMES = {"shapes": {"main": "shapes_main"}}

HEADER = '''"""mesh_to_points - all-in-one single-file version.

Turn a container mesh (PLY, STL, OBJ, OFF, GLB, 3MF, ...) into an ordered, region-aware
point set for a wiping end-effector: every point has tool_pos, tool_dir (into the
material), norm_pos and norm_dir (outward normal), plus region / level / stroke ids.

Generated from the ``mesh_to_points`` package by ``tools/build_allinone.py`` - edit the
package and rebuild rather than editing this file.

Requirements (metres, Z-up by default):
    pip install numpy scipy trimesh shapely rtree networkx plotly matplotlib
    pip install embreex potpourri3d          # optional but recommended (fast rays, heat geodesics)
    pip install manifold3d                   # only for generating the rectangular tray test mesh

Usage:
    python mesh_to_points_allinone.py shapes --out test_meshes          # generate the test meshes
    python mesh_to_points_allinone.py test_meshes/dog_bowl.ply -o output --viz --png
    python mesh_to_points_allinone.py --help

    import mesh_to_points_allinone as m2p
    res = m2p.generate("bowl.ply", spacing=0.01, pattern="spiral")
    res.points.pos, res.points.tool_dir, res.points.normal, res.points.stroke
    res.save("output", "bowl"); m2p.plotly_html(res, "output/bowl_viz.html")
"""
'''


def strip_module_docstring(src: str) -> tuple[str, str]:
    tree = ast.parse(src)
    if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(getattr(tree.body[0], "value", None), ast.Constant) \
            and isinstance(tree.body[0].value.value, str):
        node = tree.body[0]
        lines = src.splitlines(keepends=True)
        doc = "".join(lines[node.lineno - 1:node.end_lineno])
        rest = "".join(lines[node.end_lineno:])
        return ast.get_docstring(tree) or "", rest
    return "", src


def merge_from_imports(lines: list[str]) -> list[str]:
    """Merge ``from M import a`` / ``from M import b`` into ``from M import a, b``."""
    out, seen, names = [], {}, {}
    for line in lines:
        m = re.match(r"^from (\S+) import (.+)$", line)
        if not m:
            out.append(line)
            continue
        mod, items = m.group(1), [x.strip() for x in m.group(2).split(",")]
        if mod not in seen:
            seen[mod] = len(out)
            names[mod] = []
            out.append(None)
        names[mod] += [x for x in items if x not in names[mod]]
    for mod, i in seen.items():
        out[i] = f"from {mod} import {', '.join(names[mod])}"
    return out


def main():
    imports: list[str] = []
    sections: list[str] = []
    for name in ORDER:
        src = (PKG / f"{name}.py").read_text()
        doc, body = strip_module_docstring(src)
        kept = []
        for line in body.splitlines():
            s = line.strip()
            if s == "from __future__ import annotations":
                continue
            if re.match(r"^\s*from \.", line):          # relative import (top level or local)
                continue
            if re.match(r"^(import |from )", line) and not line.startswith(" "):
                if line not in imports:
                    imports.append(line)
                continue
            kept.append(line)
        text = "\n".join(kept).strip("\n") + "\n"
        # drop per-module script entry points
        text = re.sub(r'\nif __name__ == "__main__":\n(    .*\n?)+', "\n", text)
        for old, new in RENAMES.get(name, {}).items():
            text = re.sub(rf"\bdef {old}\(", f"def {new}(", text)
        banner = f"# {'=' * 76}\n# {name}.py\n"
        if doc:
            banner += "".join(f"# {l}".rstrip() + "\n" for l in doc.splitlines())
        banner += f"# {'=' * 76}\n"
        sections.append(banner + "\n" + text)
    imports = merge_from_imports(imports)
    std = sorted(l for l in imports if re.match(r"^(import|from) (os|sys|re|json|time|math|argparse|warnings|dataclasses|typing|collections)\b", l))
    third = [l for l in imports if l not in std]
    out = HEADER + "from __future__ import annotations\n\n" + "\n".join(std) + "\n\n" + "\n".join(third) + "\n\n"
    out += '__version__ = "0.1.0"\n\n\n' + "\n\n".join(sections)
    out += '''

# ============================================================================
# entry point
# ============================================================================
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "shapes":
        sys.exit(shapes_main(sys.argv[2:]) or 0)
    sys.exit(main())
'''
    OUT.write_text(out)
    compile(out, str(OUT), "exec")
    print(f"wrote {OUT} ({len(out.splitlines())} lines)")


if __name__ == "__main__":
    main()
