"""Generate every test shape, run the patterns, and write all outputs + viewers to ./output.

    .venv/bin/python examples/run_all.py            # all shapes, spiral + zigzag (+ poisson for a few)
    .venv/bin/python examples/run_all.py --spacing 0.005 --shapes dog_bowl terraces --patterns spiral
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mesh_to_points import generate, shapes  # noqa: E402
from mesh_to_points.viz import plotly_html, preview_png  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meshes", default="test_meshes")
    ap.add_argument("--out", default="output")
    ap.add_argument("--spacing", type=float, default=0.01)
    ap.add_argument("--row-spacing", type=float, default=None)
    ap.add_argument("--shapes", nargs="*", default=list(shapes.SHAPES))
    ap.add_argument("--patterns", nargs="*", default=["spiral", "zigzag"])
    ap.add_argument("--poisson-for", nargs="*", default=["dog_bowl", "terraces"])
    ap.add_argument("--no-viz", action="store_true")
    a = ap.parse_args()

    missing = [s for s in a.shapes if not os.path.exists(os.path.join(a.meshes, f"{s}.ply"))]
    if missing:
        print(f"generating meshes {missing} -> {a.meshes}")
        shapes.generate_all(a.meshes, missing, formats=("ply", "stl"), verbose=True)
    os.makedirs(a.out, exist_ok=True)
    rows, links = [], []
    for name in a.shapes:
        pats = list(a.patterns) + (["poisson"] if name in a.poisson_for else [])
        for pattern in pats:
            t0 = time.time()
            res = generate(os.path.join(a.meshes, f"{name}.ply"), spacing=a.spacing, row_spacing=a.row_spacing,
                           pattern=pattern, verbose=False)
            tag = f"{name}_{pattern}"
            paths = res.save(a.out, tag, formats=("csv", "npz", "json"))
            if not a.no_viz:
                paths["html"] = plotly_html(res, os.path.join(a.out, f"{tag}_viz.html"))
                paths["png"] = preview_png(res, os.path.join(a.out, f"{tag}_preview.png"))
                links.append((tag, os.path.basename(paths["html"]), os.path.basename(paths["png"])))
            s = res.stats
            rows.append((name, pattern, s["n_points"], s["n_strokes"], s["path_length"], s["nn_median"] * 1000,
                         s["nn_min"] * 1000, time.time() - t0))
            print(f"{name:16s} {pattern:8s} points={s['n_points']:5d} strokes={s['n_strokes']:3d} "
                  f"path={s['path_length']:6.2f} m  nn median={s['nn_median']*1000:5.1f} mm  "
                  f"regions={s['points_per_region']}  ({time.time()-t0:.1f}s)")
    if links:
        with open(os.path.join(a.out, "index.html"), "w") as f:
            f.write("<html><body style='font-family:sans-serif'><h2>mesh_to_points results</h2>"
                    f"<p>spacing {a.spacing} m</p><table border=1 cellpadding=4>"
                    "<tr><th>shape / pattern</th><th>points</th><th>strokes</th><th>path [m]</th>"
                    "<th>nn median [mm]</th><th>viewer</th><th>preview</th></tr>")
            for (name, pattern, n, ns, pl, nnm, nnmin, dt), (tag, html, png) in zip(rows, links):
                f.write(f"<tr><td>{tag}</td><td>{n}</td><td>{ns}</td><td>{pl:.2f}</td><td>{nnm:.1f}</td>"
                        f"<td><a href='{html}'>interactive 3D</a></td><td><a href='{png}'>png</a></td></tr>")
            f.write("</table></body></html>")
        print(f"\nopen {os.path.join(a.out, 'index.html')} in a browser for all interactive viewers")


if __name__ == "__main__":
    main()
