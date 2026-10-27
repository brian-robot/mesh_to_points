"""Minimal Python API usage."""
import numpy as np

from mesh_to_points import generate

res = generate("test_meshes/dog_bowl.ply", spacing=0.01, row_spacing=0.01, pattern="spiral")
pts = res.points
print(len(pts), "points,", res.stats["n_strokes"], "strokes")
print("first point: pos", pts.pos[0], "tool_dir", pts.tool_dir[0], "norm_dir", pts.normal[0], "region", pts.region_names[0])

# iterate stroke by stroke (lift the tool between strokes)
for sid in np.unique(pts.stroke):
    m = pts.stroke == sid
    print(f"stroke {sid}: {m.sum():3d} points, regions {sorted(set(pts.region_names[m]))}")

res.save("output", "api_demo")               # csv + npz + json
from mesh_to_points.viz import plotly_html   # noqa: E402
plotly_html(res, "output/api_demo_viz.html")
