"""Procedural test meshes (units: metres, Z-up).

Every shape is a watertight solid built as a surface of revolution from a closed
(r, z) profile, except the rectangular tray which is a boolean of boxes and the
"scan_bowl" which imitates a noisy, non-watertight scan of the dog bowl.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import trimesh
from scipy.interpolate import PchipInterpolator


def _densify(profile, max_len):
    P = np.asarray(profile, dtype=float)
    out = []
    n = len(P)
    for i in range(n):
        a, b = P[i], P[(i + 1) % n]
        L = np.linalg.norm(b - a)
        k = max(1, int(np.ceil(L / max_len)))
        for t in np.arange(k) / k:
            out.append(a + (b - a) * t)
    return np.array(out)


def revolve(profile, segments: int = 144, max_len: float = 0.004) -> trimesh.Trimesh:
    """Revolve a closed (r, z) polygon (r >= 0) about the Z axis into a watertight mesh.
    Profile points with r == 0 collapse onto the axis."""
    P = _densify(profile, max_len)
    keep = [0]
    for i in range(1, len(P)):
        if np.linalg.norm(P[i] - P[keep[-1]]) > 1e-9:
            keep.append(i)
    if np.linalg.norm(P[keep[-1]] - P[keep[0]]) < 1e-9:
        keep.pop()
    P = P[keep]
    n = len(P)
    theta = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    cos, sin = np.cos(theta), np.sin(theta)
    verts, rings = [], []
    for r, z in P:
        if r < 1e-9:
            rings.append((len(verts), 1))
            verts.append([0.0, 0.0, z])
        else:
            rings.append((len(verts), segments))
            verts.extend(np.c_[r * cos, r * sin, np.full(segments, z)].tolist())
    faces = []
    for i in range(n):
        j = (i + 1) % n
        s0, c0 = rings[i]
        s1, c1 = rings[j]
        if c0 == 1 and c1 == 1:
            continue
        for k in range(segments):
            k2 = (k + 1) % segments
            if c0 > 1 and c1 > 1:
                a, b, c, d = s0 + k, s0 + k2, s1 + k, s1 + k2
                faces.append([a, b, d])
                faces.append([a, d, c])
            elif c0 == 1:
                faces.append([s0, s1 + k, s1 + k2])
            else:
                faces.append([s0 + k, s0 + k2, s1])
    m = trimesh.Trimesh(np.array(verts), np.array(faces), process=True)
    m.merge_vertices()
    m.remove_unreferenced_vertices()
    m.fix_normals()
    assert m.is_watertight, "revolve produced a non-watertight mesh"
    return m


def _arc(center, radius, a0, a1, n=12):
    t = np.linspace(a0, a1, n)
    return np.c_[center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)].tolist()


def _spline_profile(z_ctrl, r_ctrl, n=80):
    f = PchipInterpolator(z_ctrl, r_ctrl)
    z = np.linspace(z_ctrl[0], z_ctrl[-1], n)
    return f, z, f(z)


# ----------------------------------------------------------------------------- shapes

def dog_bowl() -> trimesh.Trimesh:
    """Thick dog bowl: bulged outer wall, wide flat rim, slightly tapered inner wall,
    filleted inner edge and a flat inner floor.  Outer radius 0.106 m, height 0.07 m."""
    pts = [(0, 0), (0.085, 0),
           (0.097, 0.018), (0.106, 0.040), (0.102, 0.062), (0.100, 0.070),
           (0.070, 0.070),
           (0.066, 0.025)]
    pts += _arc((0.056, 0.025), 0.010, 0.0, -np.pi / 2, 10)[1:]
    pts += [(0, 0.015)]
    return revolve(pts)


def vase() -> trimesh.Trimesh:
    """Vase with a smoothly curved (S-shaped) body whose opening is its widest point,
    so the whole inner wall is reachable from above.  Height 0.22 m, wall 0.012 m."""
    H, t, zf = 0.22, 0.012, 0.015
    f, z, r = _spline_profile([0.0, 0.05, 0.11, 0.16, 0.20, H],
                              [0.045, 0.054, 0.072, 0.078, 0.080, 0.086])
    pts = [(0, 0)] + list(zip(r, z))                       # outer wall bottom -> top
    zi = np.linspace(H, zf, 70)
    pts += list(zip(f(zi) - t, zi))                        # rim + inner wall top -> bottom
    pts += [(0, zf)]
    return revolve(pts, max_len=0.003)


def vase_necked() -> trimesh.Trimesh:
    """Bulging vase with a narrower neck: part of the inner wall is NOT reachable from
    above (demonstrates the reachability filter).  Height 0.22 m, wall 0.012 m."""
    H, t, zf = 0.22, 0.012, 0.015
    f, z, r = _spline_profile([0.0, 0.05, 0.11, 0.16, 0.20, H],
                              [0.045, 0.064, 0.084, 0.078, 0.064, 0.070])
    pts = [(0, 0)] + list(zip(r, z))
    zi = np.linspace(H, zf, 70)
    pts += list(zip(f(zi) - t, zi))
    pts += [(0, zf)]
    return revolve(pts, max_len=0.003)


def bucket() -> trimesh.Trimesh:
    """Thick-walled tapered bucket: wall 0.02 m at the rim, height 0.20 m, flat floor."""
    pts = [(0, 0), (0.09, 0), (0.12, 0.20), (0.10, 0.20), (0.075, 0.02), (0, 0.02)]
    return revolve(pts)


def terraces() -> trimesh.Trimesh:
    """Circular terraces: three concentric steps descending inward to a floor.
    Each step has a flat annular tread and a vertical inner riser."""
    pts = [(0, 0), (0.15, 0), (0.15, 0.09), (0.12, 0.09), (0.12, 0.06), (0.09, 0.06),
           (0.09, 0.03), (0.06, 0.03), (0.06, 0.01), (0, 0.01)]
    return revolve(pts, segments=160)


def hemisphere_bowl() -> trimesh.Trimesh:
    """Thick hemispherical bowl: continuous curvature from rim to floor, no sharp
    wall/floor transition.  Outer radius 0.09 m, inner 0.08 m."""
    Ro, Ri, rb = 0.09, 0.08, 0.03
    zc = float(np.sqrt(Ro ** 2 - rb ** 2))
    a0 = -float(np.arcsin(zc / Ro))
    pts = [(0, 0), (rb, 0)]
    pts += _arc((0, zc), Ro, a0, 0.0, 40)[1:]
    pts += [(Ri, zc)]
    pts += _arc((0, zc), Ri, 0.0, -np.pi / 2, 40)[1:]
    return revolve(pts)


def plate() -> trimesh.Trimesh:
    """Shallow plate: wide, gently sloped inner surface (classified as floor, not wall)."""
    pts = [(0, 0), (0.06, 0), (0.125, 0.020), (0.130, 0.026), (0.122, 0.026),
           (0.070, 0.008), (0, 0.008)]
    return revolve(pts, segments=160)


def rect_tray() -> trimesh.Trimesh:
    """Rectangular tray with a central divider (two compartments), built with booleans.
    Outer 0.24 x 0.16 x 0.05 m, walls and floor 0.01 m."""
    outer = trimesh.creation.box(extents=(0.24, 0.16, 0.05))
    outer.apply_translation((0, 0, 0.025))
    cav_a = trimesh.creation.box(extents=(0.105, 0.14, 0.06))
    cav_a.apply_translation((-0.0575, 0, 0.04))
    cav_b = trimesh.creation.box(extents=(0.105, 0.14, 0.06))
    cav_b.apply_translation((0.0575, 0, 0.04))
    m = trimesh.boolean.difference([outer, cav_a, cav_b], engine="manifold")
    m = trimesh.Trimesh(m.vertices, m.faces, process=True)
    m.merge_vertices()
    m.fix_normals()
    return m


def scan_bowl(seed: int = 0) -> trimesh.Trimesh:
    """Imitation of a 3D scan of the dog bowl taken from above: only surfaces visible
    from the camera, vertex noise (0.3 mm), a few holes, not watertight."""
    rng = np.random.default_rng(seed)
    m = dog_bowl()
    cam = np.array([0.05, 0.03, 0.6])
    c = m.triangles_center
    n = m.face_normals
    to_cam = cam - c
    facing = np.einsum("ij,ij->i", n, to_cam) > 0
    eps = 1e-5
    origins = c + n * eps
    dirs = to_cam / np.linalg.norm(to_cam, axis=1, keepdims=True)
    blocked = m.ray.intersects_any(origins, dirs)
    keep = facing & ~blocked
    # punch a few holes
    for _ in range(4):
        fi = rng.choice(np.where(keep)[0])
        d = np.linalg.norm(c - c[fi], axis=1)
        keep &= d > rng.uniform(0.004, 0.009)
    s = m.submesh([np.where(keep)[0]], append=True)
    s.vertices = s.vertices + rng.normal(0, 0.0003, s.vertices.shape)
    s.remove_unreferenced_vertices()
    return s


SHAPES = {
    "dog_bowl": dog_bowl,
    "vase": vase,
    "bucket": bucket,
    "terraces": terraces,
    "hemisphere_bowl": hemisphere_bowl,
    "plate": plate,
    "rect_tray": rect_tray,
    "vase_necked": vase_necked,
    "scan_bowl": scan_bowl,
}


def generate_all(out_dir: str, names=None, formats=("ply", "stl"), verbose=True):
    os.makedirs(out_dir, exist_ok=True)
    names = names or list(SHAPES)
    written = []
    for name in names:
        m = SHAPES[name]()
        for fmt in formats:
            path = os.path.join(out_dir, f"{name}.{fmt}")
            m.export(path)
            written.append(path)
        if name == "dog_bowl":
            path = os.path.join(out_dir, f"{name}.obj")
            m.export(path)
            written.append(path)
        if verbose:
            print(f"{name:16s} faces={len(m.faces):6d} watertight={m.is_watertight!s:5s} "
                  f"extents={np.round(m.extents, 3).tolist()} -> {out_dir}/{name}.[{','.join(formats)}]")
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate the test meshes (metres, Z-up).")
    ap.add_argument("--out", default="test_meshes")
    ap.add_argument("--shapes", nargs="*", default=None, help=f"subset of {list(SHAPES)}")
    ap.add_argument("--formats", nargs="*", default=["ply", "stl"])
    a = ap.parse_args(argv)
    generate_all(a.out, a.shapes, tuple(a.formats))


if __name__ == "__main__":
    main()
