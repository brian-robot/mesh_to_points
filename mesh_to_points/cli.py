"""Command line interface: ``mesh2points INPUT [options]``."""
from __future__ import annotations

import argparse
import os
import sys

from .pipeline import ORDERS, PATTERNS, generate
from .segment import DEFAULT_INCLUDE, REGION_NAMES


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="mesh2points",
        description="Generate ordered wiping points (tool_pos, tool_dir, norm_pos, norm_dir) on the "
                    "wipeable regions of a container mesh (PLY, STL, OBJ, OFF, GLB, 3MF, ...).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("input", help="mesh file")
    ap.add_argument("-o", "--out-dir", default="output", help="output directory")
    ap.add_argument("--name", default=None, help="base name of the output files (default: input stem)")
    g = ap.add_argument_group("density")
    g.add_argument("--spacing", type=float, default=0.01, help="distance between consecutive points [m]")
    g.add_argument("--row-spacing", type=float, default=None, help="distance between rings / lines [m] (default: spacing)")
    g.add_argument("--margin", type=float, default=None, help="inset from region edges [m] (default: row/2)")
    g = ap.add_argument_group("pattern & order")
    g.add_argument("--pattern", choices=PATTERNS, default="spiral")
    g.add_argument("--order", choices=ORDERS, default="auto",
                   help="auto = pattern's own order (tsp for poisson); nearest = greedy NN; tsp = NN + 2-opt/Or-opt")
    g.add_argument("--continuous", action="store_true", help="spiral: blend rings into one continuous helix")
    g.add_argument("--zigzag-angle", type=float, default=0.0, help="scan line direction on flat regions [deg]")
    g = ap.add_argument_group("region selection")
    g.add_argument("--include", default=",".join(DEFAULT_INCLUDE),
                   help=f"comma separated regions or 'all'; regions: {','.join(REGION_NAMES)}")
    g.add_argument("--exclude", default="", help="comma separated regions to drop")
    g.add_argument("--no-reachable-filter", action="store_true",
                   help="keep surfaces that are not visible from straight above")
    g = ap.add_argument_group("mesh handling")
    g.add_argument("--up", default="z", help="up axis: z, -z, x, -x, y, -y, auto, or 'x,y,z'")
    g.add_argument("--scale", type=float, default=1.0, help="multiply input coordinates (0.001 for mm meshes)")
    g.add_argument("--top-angle", type=float, default=45.0, help="max normal angle from up for top-facing [deg]")
    g.add_argument("--bottom-angle", type=float, default=45.0, help="max normal angle from down for down-facing [deg]")
    g.add_argument("--crease-angle", type=float, default=15.0, help="dihedral angle that separates top regions [deg]")
    g.add_argument("--normal-smoothing", type=float, default=30.0,
                   help="crease angle for smooth point normals [deg]; 0 = flat face normals")
    g.add_argument("--no-refine", action="store_true", help="do not subdivide coarse meshes")
    g.add_argument("--flip-normals", action="store_true", help="invert the mesh orientation after auto-fixing")
    g = ap.add_argument_group("outputs")
    g.add_argument("--viz", action="store_true", help="write an interactive HTML viewer")
    g.add_argument("--open", action="store_true", help="open the HTML viewer in the browser")
    g.add_argument("--png", action="store_true", help="write a static preview PNG")
    g.add_argument("--ply", action="store_true", help="also write the points as a PLY point cloud")
    g.add_argument("--seg-mesh", action="store_true", help="write the segmented mesh (face colours) as PLY")
    g.add_argument("--no-slider", action="store_true", help="HTML: skip the order replay slider (big point sets)")
    g.add_argument("--cdn", action="store_true", help="HTML: load plotly.js from the CDN (smaller file, needs internet)")
    g.add_argument("--arrow-scale", type=float, default=1.0)
    g.add_argument("-q", "--quiet", action="store_true")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    name = a.name or os.path.splitext(os.path.basename(a.input))[0]
    try:
        res = generate(a.input, spacing=a.spacing, row_spacing=a.row_spacing, pattern=a.pattern, order=a.order,
                       include=a.include, exclude=a.exclude or (), reachable_only=not a.no_reachable_filter,
                       up=a.up, scale=a.scale, margin=a.margin, continuous=a.continuous,
                       zigzag_angle=a.zigzag_angle, top_angle=a.top_angle, bottom_angle=a.bottom_angle,
                       crease_angle=a.crease_angle, normal_smoothing=a.normal_smoothing,
                       refine=not a.no_refine, flip_normals=a.flip_normals, verbose=not a.quiet)
    except Exception as e:
        print(f"error: {e}", file=sys.stderr)
        raise
    formats = ["csv", "npz", "json"] + (["ply"] if a.ply else [])
    paths = res.save(a.out_dir, name, formats, seg_mesh=a.seg_mesh)
    if a.viz or a.open:
        from .viz import plotly_html
        paths["html"] = plotly_html(res, os.path.join(a.out_dir, f"{name}_viz.html"), arrow_scale=a.arrow_scale,
                                    slider=not a.no_slider, embed_js=not a.cdn, open_browser=a.open)
    if a.png:
        from .viz import preview_png
        paths["png"] = preview_png(res, os.path.join(a.out_dir, f"{name}_preview.png"))
    if not a.quiet:
        print("[files]")
        for k, v in paths.items():
            print(f"  {k:8s} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
