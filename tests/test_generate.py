import os

import numpy as np
import pytest
import trimesh

from conftest import mesh_path
from mesh_to_points import generate
from mesh_to_points.order import path_length

MAIN = ["dog_bowl", "vase", "bucket", "terraces"]
BONUS = ["hemisphere_bowl", "plate", "rect_tray", "vase_necked"]
SPACING = 0.01


def _check_common(res, spacing=SPACING):
    pts = res.points
    n = len(pts)
    assert n > 50
    # geometry: every point lies inside the plane of its own face (exact), unit normals,
    # tool_dir opposite to the normal
    tri = res.mesh.triangles[pts.face]
    fn = res.mesh.face_normals[pts.face]
    plane_dist = np.abs(np.einsum("ij,ij->i", pts.pos - tri[:, 0], fn))
    assert np.median(plane_dist) < 1e-9            # contour / scan-line points are exactly on their face
    assert plane_dist.max() < 2e-4                 # projected points may be tagged with a neighbouring face
    _, dist, _ = trimesh.proximity.closest_point(res.mesh, pts.pos)
    assert dist.max() < 1e-4
    assert np.allclose(np.linalg.norm(pts.normal, axis=1), 1.0, atol=1e-6)
    assert np.allclose(pts.tool_dir, -pts.normal)
    # the normal agrees with the face it lies on
    fn = res.mesh.face_normals[pts.face]
    assert np.all(np.einsum("ij,ij->i", fn, pts.normal) > np.cos(np.radians(35)))
    # only wipeable regions, all reachable
    names = set(pts.region_names)
    assert not names & {"outer_wall", "bottom", "inner_overhang"}
    assert pts.reachable.all()
    # spacing
    nn = pts.nn_distance()
    assert abs(np.median(nn) - spacing) < 0.25 * spacing
    step = np.linalg.norm(np.diff(pts.pos, axis=0), axis=1)
    same = pts.stroke[1:] == pts.stroke[:-1]
    assert step[same].max() < 1.75 * spacing       # never a jump inside a stroke
    if res.params["pattern"] == "spiral" and not res.params["continuous"]:
        assert step[same].max() < 1.15 * spacing   # rings are resampled uniformly
    # strokes are contiguous and start at 0
    assert pts.stroke[0] == 0
    d = np.diff(pts.stroke)
    assert np.all((d == 0) | (d == 1))


@pytest.mark.parametrize("name", MAIN + BONUS)
@pytest.mark.parametrize("pattern", ["spiral", "zigzag"])
def test_patterns(mesh_dir, name, pattern):
    res = generate(mesh_path(mesh_dir, name), pattern=pattern, verbose=False)
    _check_common(res)
    names = set(res.points.region_names)
    assert {"rim", "inner_floor"} <= names
    if name != "plate":
        assert "inner_wall" in names
    if name == "terraces":
        assert "tread" in names
        assert set(res.points.level) == {0, 1, 2, 3}


def test_spiral_order_is_outside_in_top_down(mesh_dir):
    res = generate(mesh_path(mesh_dir, "dog_bowl"), pattern="spiral", verbose=False)
    names = res.points.region_names
    first_wall = np.argmax(names == "inner_wall")
    first_floor = np.argmax(names == "inner_floor")
    assert np.all(names[:first_wall] == "rim")
    assert first_wall < first_floor
    # the first point is on the rim and far from the axis; the last point is near the centre
    p = res.points.pos
    assert np.linalg.norm(p[0, :2]) > 0.09
    assert np.linalg.norm(p[-1, :2]) < 0.012


def test_terraces_spiral_region_sequence(mesh_dir):
    res = generate(mesh_path(mesh_dir, "terraces"), pattern="spiral", verbose=False)
    lv = res.points.level
    # levels only ever increase (rim 0 -> riser/tread 1 -> ... -> floor 3)
    assert np.all(np.diff(lv) >= 0)


def test_inner_wall_normals_point_inward(mesh_dir):
    res = generate(mesh_path(mesh_dir, "bucket"), verbose=False)
    pts = res.points
    wall = pts.region_names == "inner_wall"
    radial = pts.pos[wall, :2] / np.linalg.norm(pts.pos[wall, :2], axis=1, keepdims=True)
    assert np.all(np.einsum("ij,ij->i", pts.normal[wall, :2], radial) < -0.9)
    top = pts.region_names != "inner_wall"
    assert np.all(pts.normal[top, 2] > 0.95)


def test_vase_necked_points_visible_from_above(mesh_dir):
    res = generate(mesh_path(mesh_dir, "vase_necked"), verbose=False)
    pts = res.points
    eps = 1e-4
    up = np.tile([0.0, 0.0, 1.0], (len(pts), 1))
    hit = res.mesh.ray.intersects_any(pts.pos + pts.normal * eps + up * eps, up)
    assert not hit.any()
    # wall points exist only where the wall is inside the neck's silhouette (r < neck inner radius)
    wall = pts.pos[pts.region_names == "inner_wall"]
    r = np.linalg.norm(wall[:, :2], axis=1)
    below_neck = wall[:, 2] < 0.19
    assert r[below_neck].max() < 0.0535             # nothing in the shadow of the neck (inner radius 0.052)
    assert (wall[:, 2] > 0.19).any() and (wall[:, 2] < 0.05).any()      # neck and lower body are both reachable
    full = generate(mesh_path(mesh_dir, "vase_necked"), reachable_only=False, verbose=False)
    assert len(full.points) > 2.5 * len(pts)
    assert not full.points.reachable.all()


def test_density_control(mesh_dir):
    a = generate(mesh_path(mesh_dir, "dog_bowl"), spacing=0.01, verbose=False)
    b = generate(mesh_path(mesh_dir, "dog_bowl"), spacing=0.005, verbose=False)
    ratio = len(b.points) / len(a.points)
    assert 3.2 < ratio < 4.8
    assert abs(np.median(b.points.nn_distance()) - 0.005) < 0.00125
    c = generate(mesh_path(mesh_dir, "dog_bowl"), spacing=0.01, row_spacing=0.02, verbose=False)
    assert c.stats["n_strokes"] < a.stats["n_strokes"]
    assert abs(np.median(c.points.nn_distance()) - 0.01) < 0.003


def test_continuous_helix(mesh_dir):
    res = generate(mesh_path(mesh_dir, "dog_bowl"), continuous=True, verbose=False)
    _check_common(res)
    assert res.stats["n_strokes"] <= 3
    nn = res.points.nn_distance()
    assert nn.min() > 0.15 * SPACING                 # only the very centre of the spiral tightens
    assert (nn < 0.5 * SPACING).sum() <= 12
    assert res.stats["step_min"] > 0.15 * SPACING    # the robot never has to make a tiny step


def test_poisson_and_orders(mesh_dir):
    base = generate(mesh_path(mesh_dir, "dog_bowl"), pattern="poisson", order="none", verbose=False)
    p = base.points.pos
    assert 0.7 < len(p) / generate(mesh_path(mesh_dir, "dog_bowl"), verbose=False).stats["n_points"] < 1.3
    nn = generate(mesh_path(mesh_dir, "dog_bowl"), pattern="poisson", order="nearest", verbose=False)
    tsp = generate(mesh_path(mesh_dir, "dog_bowl"), pattern="poisson", order="tsp", verbose=False)
    for r in (nn, tsp):
        assert len(r.points) == len(p)
        # same point set, permuted
        assert np.allclose(np.sort(r.points.pos, axis=0), np.sort(p, axis=0))
    assert path_length(tsp.points.pos) <= path_length(nn.points.pos)
    assert path_length(tsp.points.pos) < 0.95 * path_length(nn.points.pos)


def test_up_axis_invariance(mesh_dir):
    base = generate(mesh_path(mesh_dir, "bucket"), verbose=False)
    m = trimesh.load(mesh_path(mesh_dir, "bucket"), force="mesh")
    u = np.array([0.3, 0.2, 0.9])
    u /= np.linalg.norm(u)
    T = trimesh.geometry.align_vectors([0, 0, 1], u)
    m.apply_transform(T)
    m.apply_translation([0.3, -0.2, 0.1])
    res = generate(m, up=u, verbose=False)
    assert len(res.points) == len(base.points)
    assert res.stats["points_per_region"] == base.stats["points_per_region"]
    assert np.allclose(res.points.normal @ u, base.points.normal[:, 2], atol=1e-6)
    # results live in the input frame: the points lie on the rotated mesh
    _, dist, _ = trimesh.proximity.closest_point(m, res.points.pos)
    assert dist.max() < 1e-5
    auto = generate(m, up="auto", verbose=False)
    assert len(auto.points) == len(base.points)


def test_include_exclude(mesh_dir):
    res = generate(mesh_path(mesh_dir, "dog_bowl"), include="all", verbose=False)
    assert "outer_wall" in set(res.points.region_names)
    res = generate(mesh_path(mesh_dir, "dog_bowl"), include="rim,inner_floor", verbose=False)
    assert set(res.points.region_names) == {"rim", "inner_floor"}
    res = generate(mesh_path(mesh_dir, "dog_bowl"), exclude="rim", verbose=False)
    assert "rim" not in set(res.points.region_names)
    with pytest.raises(ValueError):
        generate(mesh_path(mesh_dir, "dog_bowl"), include="lid", verbose=False)


def test_scale_mm(mesh_dir):
    m = trimesh.load(mesh_path(mesh_dir, "dog_bowl"), force="mesh")
    m.apply_scale(1000.0)   # a millimetre mesh
    res = generate(m, scale=0.001, verbose=False)
    base = generate(mesh_path(mesh_dir, "dog_bowl"), verbose=False)
    assert abs(len(res.points) - len(base.points)) <= 2


def test_formats_and_export(mesh_dir, tmp_path):
    from mesh_to_points import shapes
    m = shapes.dog_bowl()
    for ext in ("stl", "obj", "off", "glb"):
        path = os.path.join(str(tmp_path), f"bowl.{ext}")
        m.export(path)
        res = generate(path, verbose=False)
        assert len(res.points) > 400
    paths = res.save(str(tmp_path), "bowl", formats=("csv", "npz", "json", "ply"), seg_mesh=True)
    for p in paths.values():
        assert os.path.exists(p) and os.path.getsize(p) > 0
    npz = np.load(paths["npz"])
    assert np.allclose(npz["tool_dir"], -npz["norm_dir"])
    assert np.allclose(npz["tool_pos"], npz["norm_pos"])
    with open(paths["csv"]) as f:
        lines = f.read().strip().splitlines()
    assert len(lines) == len(res.points) + 1
    assert lines[0].startswith("idx,stroke,region,level,reachable,tool_x")
    row = lines[1].split(",")
    assert len(row) == 17
    cloud = trimesh.load(paths["ply"])
    assert len(cloud.vertices) == len(res.points)


def test_cli(mesh_dir, tmp_path):
    from mesh_to_points.cli import main
    out = str(tmp_path / "out")
    rc = main([mesh_path(mesh_dir, "terraces"), "-o", out, "--pattern", "zigzag", "--viz", "--png", "--ply",
               "--seg-mesh", "--spacing", "0.015", "-q", "--no-slider"])
    assert rc == 0
    for suffix in ("points.csv", "points.npz", "meta.json", "viz.html", "preview.png", "points.ply", "segmented.ply"):
        assert os.path.exists(os.path.join(out, f"terraces_{suffix}"))
