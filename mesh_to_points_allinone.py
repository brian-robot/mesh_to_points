"""mesh_to_points - all-in-one single-file version.

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
from __future__ import annotations

from dataclasses import dataclass, field
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import unary_union
from scipy.interpolate import PchipInterpolator

__version__ = "0.1.0"


# ============================================================================
# geom.py
# Small geometry helpers shared by the segmentation and sampling code.
# ============================================================================

def unit(v: np.ndarray, axis: int = -1) -> np.ndarray:
    """Normalise vectors, leaving zero-length vectors untouched."""
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    n[n == 0] = 1.0
    return v / n


def chain_segments(pairs: np.ndarray):
    """Link segments given as (S, 2) node-id pairs into open chains and closed loops.

    Every node is expected to have degree <= 2 (true for mesh boundary edges and for
    iso-contour segments, where nodes are mesh edges).  Returns a list of
    ``(segment_indices, node_sequence, closed)``.  ``segment_indices`` lists the
    segments in traversal order, ``node_sequence`` the nodes visited (for a closed
    loop the first node is not repeated at the end).
    """
    pairs = np.asarray(pairs)
    if len(pairs) == 0:
        return []
    nodes, inv = np.unique(pairs, return_inverse=True)
    inv = inv.reshape(-1, 2)
    n_nodes = len(nodes)
    # adjacency: for each node up to two incident segments
    inc = -np.ones((n_nodes, 2), dtype=np.int64)
    deg = np.zeros(n_nodes, dtype=np.int64)
    for s, (a, b) in enumerate(inv):
        for v in (a, b):
            if deg[v] < 2:
                inc[v, deg[v]] = s
            deg[v] += 1
    used = np.zeros(len(pairs), dtype=bool)
    chains = []

    def walk(start_node, start_seg):
        seq_nodes = [start_node]
        seq_segs = []
        node, seg = start_node, start_seg
        while seg >= 0 and not used[seg]:
            used[seg] = True
            seq_segs.append(seg)
            a, b = inv[seg]
            node = b if a == node else a
            seq_nodes.append(node)
            nxt = inc[node]
            seg = nxt[0] if nxt[0] != seg else nxt[1]
            if seg < 0 or used[seg]:
                break
        return seq_segs, seq_nodes

    # open chains first: start at nodes of degree 1
    for v in np.where(deg == 1)[0]:
        s = inc[v, 0]
        if s >= 0 and not used[s]:
            segs, seq = walk(v, s)
            chains.append((np.array(segs), nodes[np.array(seq)], False))
    # remaining segments form closed loops
    for s in np.where(~used)[0]:
        if used[s]:
            continue
        v = inv[s, 0]
        segs, seq = walk(v, s)
        closed = len(seq) > 1 and seq[0] == seq[-1]
        if closed:
            seq = seq[:-1]
        chains.append((np.array(segs), nodes[np.array(seq)], closed))
    return chains


def polyline_length(points: np.ndarray, closed: bool) -> float:
    p = np.asarray(points)
    if len(p) < 2:
        return 0.0
    d = np.linalg.norm(np.diff(p, axis=0), axis=1).sum()
    if closed:
        d += np.linalg.norm(p[0] - p[-1])
    return float(d)


def resample_polyline(points: np.ndarray, spacing: float, closed: bool,
                      attrs: dict | None = None):
    """Resample a polyline at (approximately) uniform arc-length spacing.

    Returns ``(samples (n,3), seg_index (n,), t (n,))`` where ``seg_index`` is the
    index of the source segment each sample lies on and ``t`` the parameter along it.
    Closed polylines get ``round(L / spacing)`` samples without a duplicated end
    point; open ones include both end points.
    """
    p = np.asarray(points, dtype=float)
    if closed:
        p = np.vstack([p, p[:1]])
    seg = np.diff(p, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    L = cum[-1]
    if L <= 1e-12 or len(p) < 2:
        return p[:1].copy(), np.zeros(1, dtype=int), np.zeros(1)
    if closed:
        n = max(3, int(round(L / spacing)))
        s = np.arange(n) * (L / n)
    else:
        n = max(2, int(round(L / spacing)) + 1)
        s = np.linspace(0.0, L, n)
    idx = np.searchsorted(cum, s, side="right") - 1
    idx = np.clip(idx, 0, len(seg_len) - 1)
    denom = np.where(seg_len[idx] > 0, seg_len[idx], 1.0)
    t = (s - cum[idx]) / denom
    samples = p[idx] + seg[idx] * t[:, None]
    return samples, idx, t


def signed_area_xy(points: np.ndarray) -> float:
    """Signed area of a closed polygon projected to XY (positive = CCW seen from +Z)."""
    p = np.asarray(points)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def boundary_edges(faces: np.ndarray):
    """Return (edges (B,2) sorted vertex pairs, face_index (B,)) for edges used by one face."""
    faces = np.asarray(faces)
    e = faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2)
    e = np.sort(e, axis=1)
    uniq, inv, counts = np.unique(e, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    is_b = counts[inv] == 1
    return e[is_b], np.where(is_b)[0] // 3


def rotation_to_z(up: np.ndarray) -> np.ndarray:
    """3x3 rotation that maps the unit vector ``up`` onto +Z."""
    up = unit(np.asarray(up, dtype=float))
    z = np.array([0.0, 0.0, 1.0])
    c = float(np.dot(up, z))
    if c > 1 - 1e-12:
        return np.eye(3)
    if c < -1 + 1e-12:
        return np.diag([1.0, -1.0, -1.0])
    v = np.cross(up, z)
    s = np.linalg.norm(v)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


# ============================================================================
# mesh_io.py
# Mesh loading, cleaning, orientation handling and conforming refinement.
# ============================================================================

UP_PRESETS = {
    "z": (0, 0, 1), "+z": (0, 0, 1), "-z": (0, 0, -1),
    "x": (1, 0, 0), "+x": (1, 0, 0), "-x": (-1, 0, 0),
    "y": (0, 1, 0), "+y": (0, 1, 0), "-y": (0, -1, 0),
}


def load_mesh(path: str, scale: float = 1.0) -> trimesh.Trimesh:
    """Load any trimesh-supported format (PLY, STL, OBJ, OFF, GLB/GLTF, 3MF, DAE ...)
    as a single cleaned triangle mesh."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.to_mesh()
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.faces) == 0:
        raise ValueError(f"{path}: no triangle faces found")
    mesh = trimesh.Trimesh(np.asarray(loaded.vertices, dtype=float),
                           np.asarray(loaded.faces, dtype=np.int64), process=True)
    if scale != 1.0:
        mesh.apply_scale(scale)
    clean_mesh(mesh)
    return mesh


def clean_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    return mesh


def parse_up(up, mesh: trimesh.Trimesh | None = None) -> np.ndarray:
    """Turn 'z', '-y', 'auto', '0,0,1' or an array into a unit up-vector."""
    if isinstance(up, str):
        key = up.strip().lower()
        if key in UP_PRESETS:
            return np.array(UP_PRESETS[key], dtype=float)
        if key == "auto":
            if mesh is None:
                raise ValueError("up='auto' needs a mesh")
            return estimate_up(mesh)
        parts = [float(x) for x in key.replace(";", ",").split(",")]
        if len(parts) != 3:
            raise ValueError(f"cannot parse up vector {up!r}")
        return unit(np.array(parts))
    v = np.asarray(up, dtype=float).reshape(3)
    return unit(v)


def estimate_up(mesh: trimesh.Trimesh) -> np.ndarray:
    """Guess the up direction: the direction whose opposite side carries the largest
    flat resting base (down-facing faces at the lowest height)."""
    cands = [np.array(v, dtype=float) for v in UP_PRESETS.values()]
    try:
        pcs = mesh.principal_inertia_vectors
        for v in pcs:
            cands.append(unit(v))
            cands.append(-unit(v))
    except Exception:  # pragma: no cover
        pass
    n = mesh.face_normals
    c = mesh.triangles_center
    a = mesh.area_faces
    best, best_score = cands[2], -1.0
    for u in cands:
        h = c @ u
        ext = h.max() - h.min()
        down = (n @ u) < -np.cos(np.radians(10))
        low = h < h.min() + 0.03 * ext
        score = a[down & low].sum()
        if score > best_score + 1e-12:
            best, best_score = u, score
    return unit(best)


class Frame:
    """Rigid transform between the input frame and the internal Z-up frame."""

    def __init__(self, up: np.ndarray):
        self.up = unit(np.asarray(up, dtype=float))
        self.R = rotation_to_z(self.up)  # x_canon = R @ x_input

    def to_canonical(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        T = np.eye(4)
        T[:3, :3] = self.R
        m = mesh.copy()
        m.apply_transform(T)
        return m

    def points_to_input(self, p: np.ndarray) -> np.ndarray:
        return np.asarray(p) @ self.R  # R^T applied row-wise

    def vectors_to_input(self, v: np.ndarray) -> np.ndarray:
        return np.asarray(v) @ self.R


def fix_orientation(mesh: trimesh.Trimesh, flip: bool = False, verbose: bool = False) -> trimesh.Trimesh:
    """Make face windings consistent and normals point out of the material.

    Watertight meshes use the signed volume.  Open meshes use a ray-parity vote
    (a correctly oriented face sees an even number of surfaces along +n and an odd
    number along -n); if that is inconclusive (thin open sheets) normals are made to
    face the open side, i.e. mostly upwards."""
    trimesh.repair.fix_winding(mesh)
    _refresh(mesh)
    if mesh.is_watertight:
        comps = trimesh.graph.connected_components(mesh.face_adjacency, nodes=np.arange(len(mesh.faces)), min_len=1)
        for comp in comps:
            comp = np.asarray(comp)
            sub = trimesh.Trimesh(mesh.vertices, mesh.faces[comp], process=False)
            if sub.volume < 0:
                f = mesh.faces.copy()
                f[comp] = f[comp][:, ::-1]
                mesh.faces = f
                _refresh(mesh)
                if verbose:
                    print(f"  flipped an inside-out component ({len(comp)} faces)")
    else:
        comps = trimesh.graph.connected_components(mesh.face_adjacency, nodes=np.arange(len(mesh.faces)), min_len=1)
        for comp in comps:
            comp = np.asarray(comp)
            flip_comp = _component_is_inverted(mesh, comp)
            if flip_comp:
                f = mesh.faces.copy()
                f[comp] = f[comp][:, ::-1]
                mesh.faces = f
                _refresh(mesh)
                if verbose:
                    print(f"  flipped orientation of a {len(comp)}-face component")
    if flip:
        mesh.faces = np.ascontiguousarray(mesh.faces[:, ::-1])
    _refresh(mesh)
    return mesh


def _refresh(mesh: trimesh.Trimesh):
    """Drop cached normals so they are recomputed from the current winding."""
    mesh._cache.clear()


def _component_is_inverted(mesh: trimesh.Trimesh, comp: np.ndarray) -> bool:
    # a component with many boundary edges is a sheet, not a solid: normals face the open side
    faces = mesh.faces[comp]
    e = np.sort(faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    if (counts == 1).sum() > 0.01 * len(counts):
        a = mesh.area_faces[comp]
        return float(np.sum(a * mesh.face_normals[comp][:, 2])) < 0
    rng = np.random.default_rng(0)
    sample = comp if len(comp) <= 400 else rng.choice(comp, 400, replace=False)
    n = mesh.face_normals[sample]
    c = mesh.triangles_center[sample]
    eps = 1e-4 * float(np.linalg.norm(mesh.extents))
    votes_ok = votes_flip = 0
    for sign in (1.0, -1.0):
        origins = c + n * (eps * sign)
        dirs = n * sign
        try:
            _, ray_idx, _ = mesh.ray.intersects_location(origins, dirs, multiple_hits=True)
        except Exception:  # pragma: no cover
            return False
        counts = np.bincount(ray_idx, minlength=len(sample))
        parity = counts % 2
        if sign > 0:
            plus_parity = parity
        else:
            minus_parity = parity
    ok = (plus_parity == 0) & (minus_parity == 1)
    bad = (plus_parity == 1) & (minus_parity == 0)
    votes_ok, votes_flip = int(ok.sum()), int(bad.sum())
    decided = votes_ok + votes_flip
    # the parity rule only holds for (nearly) closed solids: demand a clear verdict
    if decided >= 0.3 * len(sample) and max(votes_ok, votes_flip) >= 0.8 * decided:
        return votes_flip > votes_ok
    # inconclusive: thin open sheet -> normals should face the open (upper) side
    a = mesh.area_faces[comp]
    return float(np.sum(a * mesh.face_normals[comp][:, 2])) < 0


def refine_to_edge_length(mesh: trimesh.Trimesh, max_edge: float, max_faces: int = 2_000_000,
                          verbose: bool = False) -> trimesh.Trimesh:
    """Conforming midpoint refinement: split every edge longer than ``max_edge``
    (and re-triangulate the touched faces) until none remains.  Geometry is unchanged,
    the mesh stays manifold, and the orientation of every face is preserved."""
    V = np.asarray(mesh.vertices, dtype=float).copy()
    F = np.asarray(mesh.faces, dtype=np.int64).copy()
    n0 = len(F)
    for _ in range(24):
        E = np.sort(F[:, [[0, 1], [1, 2], [2, 0]]], axis=2).reshape(-1, 2)
        uniq, inv = np.unique(E, axis=0, return_inverse=True)
        inv = inv.reshape(-1, 3)
        L = np.linalg.norm(V[uniq[:, 0]] - V[uniq[:, 1]], axis=1)
        split = L > max_edge
        if not split.any():
            break
        S = split[inv]
        est = len(F) + int(S.sum(axis=1).sum())
        if est > max_faces:
            warnings.warn(f"refinement stopped at {len(F)} faces (limit {max_faces})")
            break
        mid = np.full(len(uniq), -1, dtype=np.int64)
        mid[split] = len(V) + np.arange(int(split.sum()))
        V = np.vstack([V, 0.5 * (V[uniq[split, 0]] + V[uniq[split, 1]])])
        M = mid[inv]
        code = S[:, 0] * 1 + S[:, 1] * 2 + S[:, 2] * 4
        new = [F[code == 0]]
        for c, rot in ((1, 0), (2, 1), (4, 2)):  # one edge split -> rotate it to edge 01
            idx = np.where(code == c)[0]
            if len(idx) == 0:
                continue
            f = np.roll(F[idx], -rot, axis=1)
            m = np.roll(M[idx], -rot, axis=1)
            m01 = m[:, 0]
            new.append(np.c_[f[:, 0], m01, f[:, 2]])
            new.append(np.c_[m01, f[:, 1], f[:, 2]])
        for c, rot in ((3, 0), (6, 1), (5, 2)):  # two edges split -> edges 01 and 12
            idx = np.where(code == c)[0]
            if len(idx) == 0:
                continue
            f = np.roll(F[idx], -rot, axis=1)
            m = np.roll(M[idx], -rot, axis=1)
            m01, m12 = m[:, 0], m[:, 1]
            new.append(np.c_[m01, f[:, 1], m12])
            d1 = np.linalg.norm(V[f[:, 0]] - V[m12], axis=1)
            d2 = np.linalg.norm(V[m01] - V[f[:, 2]], axis=1)
            use1 = (d1 <= d2)[:, None]
            new.append(np.where(use1, np.c_[f[:, 0], m01, m12], np.c_[f[:, 0], m01, f[:, 2]]))
            new.append(np.where(use1, np.c_[f[:, 0], m12, f[:, 2]], np.c_[m01, m12, f[:, 2]]))
        idx = np.where(code == 7)[0]
        if len(idx):
            f, m = F[idx], M[idx]
            m01, m12, m20 = m[:, 0], m[:, 1], m[:, 2]
            new += [np.c_[f[:, 0], m01, m20], np.c_[m01, f[:, 1], m12],
                    np.c_[m20, m12, f[:, 2]], np.c_[m01, m12, m20]]
        F = np.vstack(new)
    out = trimesh.Trimesh(V, F, process=False)
    if verbose and len(F) != n0:
        print(f"  refined mesh: {n0} -> {len(F)} faces (max edge {max_edge:.4g})")
    return out


# ============================================================================
# segment.py
# Face segmentation of a container mesh into semantic regions.
#
# Regions (face labels):
#   rim            top-facing surface connected to the exterior (the lip of the container)
#   inner_wall     side-facing surface that looks into a cavity
#   tread          top-facing surface inside the cavity above the floor (terrace steps)
#   inner_floor    lowest top-facing surface inside the cavity
#   inner_overhang down-facing surface inside the cavity (e.g. under a vase shoulder)
#   outer_wall     side-facing surface looking outwards
#   bottom         down-facing exterior surface (the base)
#
# The mesh is expected in the internal Z-up frame (see mesh_io.Frame).
# ============================================================================

RIM, INNER_WALL, TREAD, INNER_FLOOR, INNER_OVERHANG, OUTER_WALL, BOTTOM = range(7)
REGION_NAMES = ("rim", "inner_wall", "tread", "inner_floor", "inner_overhang", "outer_wall", "bottom")
REGION_IDS = {n: i for i, n in enumerate(REGION_NAMES)}
DEFAULT_INCLUDE = ("rim", "inner_wall", "tread", "inner_floor")
REGION_COLORS = {
    "rim": "#f4a261", "inner_wall": "#2a9d8f", "tread": "#e9c46a", "inner_floor": "#457b9d",
    "inner_overhang": "#9b5de5", "outer_wall": "#c9ccd1", "bottom": "#8d99ae",
}
UNREACHABLE_COLOR = "#e76f51"

_TOP, _SIDE, _DOWN = 0, 1, 2


@dataclass
class RegionComponent:
    id: int
    label: int
    level: int
    faces: np.ndarray
    area: float
    z_mean: float
    z_min: float
    z_max: float
    centroid: np.ndarray

    @property
    def name(self) -> str:
        return REGION_NAMES[self.label]


@dataclass
class Segmentation:
    labels: np.ndarray            # (F,) region id per face
    level: np.ndarray             # (F,) step level (0 = rim level, increasing downwards)
    component: np.ndarray         # (F,) id into ``components``
    reachable: np.ndarray         # (F,) face centroid visible from straight above
    enclosed: np.ndarray          # (F,) raw enclosure test result
    components: list = field(default_factory=list)
    exterior_edges: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.int64))
    params: dict = field(default_factory=dict)

    def names(self) -> np.ndarray:
        return np.array(REGION_NAMES)[self.labels]

    def select(self, include=DEFAULT_INCLUDE, exclude=(), reachable_only=True) -> np.ndarray:
        """Boolean face mask of the regions to generate points on."""
        if include is None or include == "all" or (isinstance(include, (list, tuple)) and "all" in include):
            inc = set(REGION_NAMES)
        else:
            inc = set([include] if isinstance(include, str) else include)
        exc = set([exclude] if isinstance(exclude, str) else (exclude or ()))
        unknown = (inc | exc) - set(REGION_NAMES)
        if unknown:
            raise ValueError(f"unknown region name(s) {sorted(unknown)}; valid: {list(REGION_NAMES)}")
        ids = [REGION_IDS[n] for n in inc - exc]
        mask = np.isin(self.labels, ids)
        if reachable_only:
            mask &= self.reachable
        return mask

    def summary(self, mesh: trimesh.Trimesh) -> list:
        rows = []
        area = mesh.area_faces
        for i, name in enumerate(REGION_NAMES):
            m = self.labels == i
            if not m.any():
                continue
            comps = sorted({int(c) for c in self.component[m]})
            rows.append({
                "region": name, "faces": int(m.sum()), "area": float(area[m].sum()),
                "components": len(comps), "levels": sorted({int(l) for l in self.level[m]}),
                "reachable_area": float(area[m & self.reachable].sum()),
            })
        return rows


def segment(mesh: trimesh.Trimesh, top_angle: float = 45.0, bottom_angle: float = 45.0,
            crease_angle: float = 15.0, min_region_frac: float = 0.002, min_region_faces: int = 6,
            verbose: bool = False) -> Segmentation:
    """Classify every face of a Z-up container mesh (see module docstring)."""
    F = len(mesh.faces)
    n = mesh.face_normals
    c = mesh.triangles_center
    area = mesh.area_faces
    total_area = float(area.sum())
    diag = float(np.linalg.norm(mesh.extents))
    eps = 1e-4 * diag
    height = float(mesh.extents[2])
    ztol = max(0.002, 0.05 * height)

    # 1. orientation class --------------------------------------------------------
    nz = n[:, 2]
    cls = np.full(F, _SIDE, dtype=np.int8)
    cls[nz > np.cos(np.radians(top_angle))] = _TOP
    cls[nz < -np.cos(np.radians(bottom_angle))] = _DOWN

    # 2. enclosure test: does the surface look at the mesh itself? -------------------
    d = n.copy()
    side = cls == _SIDE
    h = n[side].copy()
    h[:, 2] = 0.0
    hn = np.linalg.norm(h, axis=1)
    ok = hn > 1e-6
    h[ok] /= hn[ok][:, None]
    h[~ok] = n[side][~ok]
    d[side] = h
    enclosed = np.zeros(F, dtype=bool)
    test = cls != _TOP
    if test.any():
        enclosed[test] = mesh.ray.intersects_any(c[test] + n[test] * eps, d[test])

    # 3. initial labels + speckle removal ---------------------------------------------
    labels = np.full(F, -1, dtype=np.int64)        # -1 = top-facing, resolved later
    labels[side & enclosed] = INNER_WALL
    labels[side & ~enclosed] = OUTER_WALL
    down = cls == _DOWN
    labels[down & enclosed] = INNER_OVERHANG
    labels[down & ~enclosed] = BOTTOM
    adj = mesh.face_adjacency
    min_area = min_region_frac * total_area
    labels = _absorb_small_components(labels, adj, area, min_area, min_region_faces)

    # 4. top-facing components: rim vs. inner tops -------------------------------------
    adj_angles = mesh.face_adjacency_angles
    top = labels == -1
    top_adj = adj[top[adj[:, 0]] & top[adj[:, 1]] & (adj_angles < np.radians(crease_angle))]
    top_comps = [np.asarray(cc) for cc in
                 trimesh.graph.connected_components(top_adj, nodes=np.where(top)[0], min_len=1)]
    top_comps = _merge_small_top_components(top_comps, adj[top[adj[:, 0]] & top[adj[:, 1]]],
                                            area, min_area, min_region_faces, F)

    ext_face = np.zeros(F, dtype=bool)
    ext_lbl = np.isin(labels, [OUTER_WALL, BOTTOM])
    ext_face[adj[ext_lbl[adj[:, 1]], 0]] = True
    ext_face[adj[ext_lbl[adj[:, 0]], 1]] = True
    ext_faces, ext_edges = _exterior_boundary(mesh)
    ext_face |= ext_faces

    comp_info = []
    for cc in top_comps:
        comp_info.append({
            "faces": cc, "z_mean": float(np.average(c[cc, 2], weights=area[cc])),
            "ext": bool(ext_face[cc].any()), "area": float(area[cc].sum()),
        })
    is_rim = np.array([ci["ext"] for ci in comp_info], dtype=bool)
    if len(comp_info) and not is_rim.any():
        is_rim[int(np.argmax([ci["z_mean"] for ci in comp_info]))] = True
    # propagate rim across creases to neighbouring top components at the same height
    if len(comp_info):
        comp_of_face = np.full(F, -1)
        for k, ci in enumerate(comp_info):
            comp_of_face[ci["faces"]] = k
        tt = adj[(comp_of_face[adj[:, 0]] >= 0) & (comp_of_face[adj[:, 1]] >= 0)]
        pairs = {(int(a), int(b)) for a, b in comp_of_face[tt] if a != b}
        changed = True
        while changed:
            changed = False
            for a, b in pairs:
                for src, dst in ((a, b), (b, a)):
                    if is_rim[src] and not is_rim[dst] and comp_info[dst]["z_mean"] >= comp_info[src]["z_mean"] - ztol:
                        is_rim[dst] = True
                        changed = True

    level = np.zeros(F, dtype=np.int64)
    inner_idx = [k for k in range(len(comp_info)) if not is_rim[k]]
    inner_idx.sort(key=lambda k: -comp_info[k]["z_mean"])
    lvl, prev_z = 0, None
    comp_level = {}
    for k in inner_idx:
        z = comp_info[k]["z_mean"]
        if prev_z is None or z < prev_z - ztol:
            lvl += 1
            prev_z = z
        comp_level[k] = lvl
    max_lvl = lvl
    for k, ci in enumerate(comp_info):
        if is_rim[k]:
            labels[ci["faces"]] = RIM
            level[ci["faces"]] = 0
        else:
            labels[ci["faces"]] = INNER_FLOOR if comp_level[k] == max_lvl else TREAD
            level[ci["faces"]] = comp_level[k]
    if max_lvl == 0 and len(comp_info):
        pass  # no inner tops (e.g. a wall-only scan): walls get level 1 below

    # 5. build the component list ------------------------------------------------------
    components = []
    component = np.full(F, -1, dtype=np.int64)
    same = labels[adj[:, 0]] == labels[adj[:, 1]]
    crease_ok = adj_angles < np.radians(crease_angle)
    top_lbl = np.isin(labels, [RIM, TREAD, INNER_FLOOR])
    keep_edge = same & (~top_lbl[adj[:, 0]] | crease_ok)
    for cc in trimesh.graph.connected_components(adj[keep_edge], nodes=np.arange(F), min_len=1):
        cc = np.asarray(cc)
        cid = len(components)
        component[cc] = cid
        lab = int(labels[cc[0]])
        components.append(RegionComponent(
            id=cid, label=lab, level=int(level[cc[0]]), faces=cc, area=float(area[cc].sum()),
            z_mean=float(np.average(c[cc, 2], weights=area[cc])),
            z_min=float(mesh.vertices[mesh.faces[cc]][:, :, 2].min()),
            z_max=float(mesh.vertices[mesh.faces[cc]][:, :, 2].max()),
            centroid=np.average(c[cc], axis=0, weights=area[cc])))
    # wall levels: the deepest adjacent top level (bowl wall -> 1, terrace riser k -> k)
    comp_adj = adj[component[adj[:, 0]] != component[adj[:, 1]]]
    for comp in components:
        if comp.label in (INNER_WALL, INNER_OVERHANG):
            m = component[comp_adj] == comp.id
            others = np.concatenate([comp_adj[m[:, 0], 1], comp_adj[m[:, 1], 0]])
            lv = [int(level[o]) for o in np.unique(others) if top_lbl[o]]
            comp.level = max(lv) if lv else 1
            level[comp.faces] = comp.level

    # 6. reachability from straight above --------------------------------------------
    up = np.tile([0.0, 0.0, 1.0], (F, 1))
    origins = c + n * eps + up * eps        # off the surface so vertical walls do not graze themselves
    reachable = ~mesh.ray.intersects_any(origins, up)
    reachable &= nz > -0.01                 # down-facing surfaces cannot be wiped from above
    # tiny unreachable specks (noisy scans) are not real occlusions
    reachable = _absorb_small_components(reachable.astype(np.int64), adj, area, min_area, min_region_faces) == 1

    seg = Segmentation(labels=labels, level=level, component=component, reachable=reachable,
                       enclosed=enclosed, components=components, exterior_edges=ext_edges,
                       params=dict(top_angle=top_angle, bottom_angle=bottom_angle, crease_angle=crease_angle,
                                   min_region_frac=min_region_frac))
    if verbose:
        for row in seg.summary(mesh):
            print(f"  {row['region']:15s} faces={row['faces']:7d} area={row['area']:.5f} m² "
                  f"comps={row['components']} levels={row['levels']} reachable={row['reachable_area']:.5f}")
    return seg


def _absorb_small_components(labels, adj, area, min_area, min_faces, max_iter=6):
    labels = labels.copy()
    F = len(labels)
    for _ in range(max_iter):
        same = labels[adj[:, 0]] == labels[adj[:, 1]]
        comps = trimesh.graph.connected_components(adj[same], nodes=np.arange(F), min_len=1)
        comps = sorted((np.asarray(cc) for cc in comps), key=lambda cc: area[cc].sum())
        changed = False
        for cc in comps:
            if area[cc].sum() >= min_area and len(cc) >= min_faces:
                break
            mask = np.zeros(F, dtype=bool)
            mask[cc] = True
            cross = adj[mask[adj[:, 0]] != mask[adj[:, 1]]]
            if len(cross) == 0:
                continue
            neigh = np.where(mask[cross[:, 0]], labels[cross[:, 1]], labels[cross[:, 0]])
            vals, counts = np.unique(neigh, return_counts=True)
            new = int(vals[np.argmax(counts)])
            if new != labels[cc[0]]:
                labels[cc] = new
                changed = True
        if not changed:
            break
    return labels


def _exterior_boundary(mesh: trimesh.Trimesh):
    """Faces and edges on the largest boundary loop of an open mesh (holes are ignored).
    Returns ``(face_mask (F,), edges (B,2) sorted vertex pairs)``."""
    out = np.zeros(len(mesh.faces), dtype=bool)
    edges, face_idx = boundary_edges(mesh.faces)
    if len(edges) == 0:
        return out, np.zeros((0, 2), dtype=np.int64)
    chains = chain_segments(edges)
    best, best_area = None, -1.0
    for segs, nodes, closed in chains:
        a = abs(signed_area_xy(mesh.vertices[nodes])) if len(nodes) > 2 else 0.0
        if a > best_area:
            best, best_area = segs, a
    if best is None:
        return out, np.zeros((0, 2), dtype=np.int64)
    out[face_idx[best]] = True
    return out, edges[best]


def _merge_small_top_components(comps, top_adj, area, min_area, min_faces, F, max_iter=50):
    """Merge crease-split top components that are too small into the neighbouring
    top component they share the most edges with (noise robustness)."""
    comp_of = np.full(F, -1, dtype=np.int64)
    for k, cc in enumerate(comps):
        comp_of[cc] = k
    comps = [np.asarray(cc) for cc in comps]
    for _ in range(max_iter):
        sizes = np.array([area[cc].sum() for cc in comps])
        counts = np.array([len(cc) for cc in comps])
        small = np.where(((sizes < min_area) | (counts < min_faces)) & (counts > 0))[0]
        if len(small) == 0:
            break
        ca, cb = comp_of[top_adj[:, 0]], comp_of[top_adj[:, 1]]
        diff = (ca != cb) & (ca >= 0) & (cb >= 0)
        if not diff.any():
            break
        pairs = np.concatenate([np.c_[ca[diff], cb[diff]], np.c_[cb[diff], ca[diff]]])
        merged_any = False
        for k in small[np.argsort(sizes[small])]:
            if len(comps[k]) == 0:
                continue
            nb = pairs[pairs[:, 0] == k, 1]
            if len(nb) == 0:
                continue
            vals, cnt = np.unique(nb, return_counts=True)
            target = int(vals[np.argmax(cnt)])
            if target == k or len(comps[target]) == 0:
                continue
            comps[target] = np.concatenate([comps[target], comps[k]])
            comp_of[comps[k]] = target
            pairs[pairs == k] = target
            comps[k] = np.zeros(0, dtype=np.int64)
            merged_any = True
        if not merged_any:
            break
    return [cc for cc in comps if len(cc) > 0]


# ============================================================================
# geodesic.py
# Geodesic distance fields on triangle meshes (heat method with a Dijkstra fallback).
# ============================================================================

_HAS_PP3D = None


def has_heat_solver() -> bool:
    global _HAS_PP3D
    if _HAS_PP3D is None:
        try:
            import potpourri3d  # noqa: F401
            _HAS_PP3D = True
        except Exception:
            _HAS_PP3D = False
    return _HAS_PP3D


def geodesic_distance(V: np.ndarray, F: np.ndarray, sources, method: str = "auto") -> np.ndarray:
    """Per-vertex geodesic distance to the nearest of ``sources`` (vertex indices).

    ``method``: 'auto' (heat method if potpourri3d is installed, else Dijkstra),
    'heat' or 'dijkstra'.  The mesh should be a single connected component.
    """
    V = np.ascontiguousarray(V, dtype=np.float64)
    F = np.ascontiguousarray(F, dtype=np.int64)
    sources = np.unique(np.asarray(sources, dtype=np.int64))
    if len(sources) == 0:
        raise ValueError("geodesic_distance needs at least one source vertex")
    if method in ("auto", "heat") and has_heat_solver():
        try:
            import potpourri3d as pp3d
            solver = pp3d.MeshHeatMethodDistanceSolver(V, F)
            d = np.asarray(solver.compute_distance_multisource(sources.tolist()), dtype=float)
            if np.all(np.isfinite(d)):
                d = np.maximum(d, 0.0)
                d[sources] = 0.0
                return d
            warnings.warn("heat method returned non-finite distances; falling back to Dijkstra")
        except Exception as e:  # pragma: no cover
            if method == "heat":
                raise
            warnings.warn(f"heat method failed ({e}); falling back to Dijkstra")
    return dijkstra_distance(V, F, sources)


def dijkstra_distance(V: np.ndarray, F: np.ndarray, sources) -> np.ndarray:
    """Graph (edge-walking) geodesic distance.  Over-estimates by up to ~8% on a
    regular mesh; good enough as a fallback and for coarse ordering."""
    n = len(V)
    e = np.sort(F[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    e = np.unique(e, axis=0)
    w = np.linalg.norm(V[e[:, 0]] - V[e[:, 1]], axis=1)
    G = coo_matrix((np.r_[w, w], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)).tocsr()
    d = dijkstra(G, directed=False, indices=np.asarray(sources), min_only=True)
    return np.asarray(d, dtype=float)


# ============================================================================
# contours.py
# Iso-contour extraction of a per-vertex scalar field on a triangle mesh.
# ============================================================================

def iso_contours(V: np.ndarray, F: np.ndarray, f: np.ndarray, iso: float):
    """Polylines where the piecewise-linear field ``f`` equals ``iso``.

    Returns a list of ``(points (k,3), faces (s,), closed)`` where ``faces`` holds the
    triangle index of every polyline segment (``s = k`` for closed loops, ``k - 1``
    otherwise) in traversal order.
    """
    fv = f[F]
    rng = float(np.ptp(f)) if len(f) else 1.0
    if np.any(np.abs(fv - iso) < 1e-9 * max(rng, 1e-12)):
        iso = iso + 1e-7 * max(rng, 1e-12)          # avoid crossings exactly on vertices
    above = fv > iso
    n_above = above.sum(axis=1)
    active = np.where((n_above == 1) | (n_above == 2))[0]
    if len(active) == 0:
        return []
    a = above[active]
    cross = a[:, [0, 1, 2]] != a[:, [1, 2, 0]]           # (A,3): edges 01, 12, 20
    e_v0 = F[active][:, [0, 1, 2]]
    e_v1 = F[active][:, [1, 2, 0]]
    # exactly two crossing edges per active triangle
    order = np.argsort(~cross, axis=1, kind="stable")[:, :2]   # indices of the two crossing edges
    rows = np.arange(len(active))[:, None]
    v0 = e_v0[rows, order]
    v1 = e_v1[rows, order]
    f0 = f[v0]
    f1 = f[v1]
    t = (iso - f0) / np.where(f1 - f0 == 0, 1e-300, f1 - f0)
    t = np.clip(t, 0.0, 1.0)
    P = V[v0] + (V[v1] - V[v0]) * t[..., None]              # (A,2,3)
    nV = len(V)
    lo = np.minimum(v0, v1)
    hi = np.maximum(v0, v1)
    keys = lo.astype(np.int64) * nV + hi                     # (A,2) edge keys = chain nodes
    chains = chain_segments(keys)
    key_pos = {}
    for i in range(len(active)):
        key_pos[int(keys[i, 0])] = P[i, 0]
        key_pos[int(keys[i, 1])] = P[i, 1]
    out = []
    for segs, nodes, closed in chains:
        pts = np.array([key_pos[int(k)] for k in nodes])
        faces = active[segs]
        if closed and len(pts) < 3:
            continue
        if not closed and len(pts) < 2:
            continue
        out.append((pts, faces, closed))
    return out


# ============================================================================
# normals.py
# Crease-aware smooth normals for sample points lying on mesh faces.
# ============================================================================

def smooth_point_normals(mesh: trimesh.Trimesh, faces: np.ndarray, bary: np.ndarray,
                         crease_angle: float = 30.0) -> np.ndarray:
    """Normal at points given by (face index, barycentric coords).

    Per-corner normals average the area-weighted normals of the faces around the
    corner vertex whose orientation is within ``crease_angle`` of the point's own face
    (so curved surfaces get smooth normals while sharp edges stay sharp), then the
    corner normals are interpolated barycentrically.  ``crease_angle <= 0`` returns
    flat face normals.
    """
    faces = np.asarray(faces, dtype=np.int64)
    fn = mesh.face_normals
    if crease_angle <= 0 or len(faces) == 0:
        return fn[faces].copy()
    bary = np.asarray(bary, dtype=float)
    vf = mesh.vertex_faces                      # (V, maxdeg) padded with -1
    corner_v = mesh.faces[faces]                # (N,3)
    ring = vf[corner_v]                         # (N,3,D)
    valid = ring >= 0
    ring_safe = np.where(valid, ring, 0)
    rn = fn[ring_safe]                          # (N,3,D,3)
    ra = mesh.area_faces[ring_safe]             # (N,3,D)
    own = fn[faces][:, None, None, :]           # (N,1,1,3)
    cos = np.einsum("nkdj,nkdj->nkd", rn, np.broadcast_to(own, rn.shape))
    w = ra * valid * (cos > np.cos(np.radians(crease_angle)))
    corner = unit(np.einsum("nkd,nkdj->nkj", w, rn))      # (N,3,3)
    n = np.einsum("nk,nkj->nj", bary, corner)
    n = unit(n)
    bad = np.linalg.norm(n, axis=1) < 1e-9
    n[bad] = fn[faces[bad]]
    return n


# ============================================================================
# order.py
# Point ordering: greedy nearest neighbour, 2-opt/Or-opt improvement, stroke splitting.
# ============================================================================

def default_start(pos: np.ndarray) -> int:
    """Highest point, ties broken by distance from the XY centroid (= outer rim)."""
    z = pos[:, 2]
    top = z >= z.max() - 1e-3 * max(float(np.ptp(z)), 1e-9) - 1e-9
    idx = np.where(top)[0]
    c = pos[:, :2].mean(axis=0)
    r = np.linalg.norm(pos[idx, :2] - c, axis=1)
    return int(idx[np.argmax(r)])


def nearest_neighbor_order(pos: np.ndarray, start: int | None = None) -> np.ndarray:
    """Greedy nearest-neighbour path through all points."""
    pos = np.asarray(pos, dtype=float)
    n = len(pos)
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    if start is None:
        start = default_start(pos)
    tree = cKDTree(pos)
    visited = np.zeros(n, dtype=bool)
    order = np.empty(n, dtype=np.int64)
    cur = int(start)
    visited[cur] = True
    order[0] = cur
    for i in range(1, n):
        nxt = -1
        kk = 8
        while nxt < 0:
            kk = min(kk, n)
            _, idx = tree.query(pos[cur], k=kk)
            idx = np.atleast_1d(idx)
            cand = idx[~visited[idx]]
            if len(cand):
                nxt = int(cand[0])
            elif kk >= n:
                rem = np.where(~visited)[0]
                nxt = int(rem[np.argmin(np.linalg.norm(pos[rem] - pos[cur], axis=1))])
            else:
                kk *= 4
        visited[nxt] = True
        order[i] = nxt
        cur = nxt
    return order


def path_length(pos: np.ndarray, order: np.ndarray | None = None) -> float:
    p = pos if order is None else pos[order]
    if len(p) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def improve_path(pos: np.ndarray, order: np.ndarray, k: int = 10, max_passes: int = 50,
                 time_limit: float = 30.0, verbose: bool = False) -> np.ndarray:
    """Open-path local search (2-opt, then Or-opt) over k-nearest-neighbour candidate
    moves.  Removes most of the long jumps that a greedy nearest-neighbour order
    leaves behind."""
    pos = np.asarray(pos, dtype=float)
    order = np.asarray(order, dtype=np.int64).copy()
    n = len(order)
    if n < 4:
        return order
    kk = min(k + 1, n)
    _, nbrs = cKDTree(pos).query(pos, k=kk)
    nbrs = np.atleast_2d(nbrs)[:, 1:]
    t0 = time.time()
    L0 = path_length(pos, order)
    passes = 0
    for passes in range(1, max_passes + 1):
        improved = _two_opt_pass(pos, order, nbrs)
        if time.time() - t0 > time_limit:
            break
        order, imp2 = _or_opt_pass(pos, order, nbrs)
        improved |= imp2
        if not improved or time.time() - t0 > time_limit:
            break
    if verbose:
        print(f"  path improvement: {L0:.4f} -> {path_length(pos, order):.4f} m "
              f"({passes} passes, {time.time() - t0:.1f}s)")
    return order


def _two_opt_pass(pos, order, nbrs) -> bool:
    """One 2-opt sweep (in place). Open path: reversing order[i+1..j] replaces edges
    (a,b)=(order[i],order[i+1]) and (c,d)=(order[j],order[j+1]) by (a,c) and (b,d)."""
    n = len(order)
    position = np.empty(n, dtype=np.int64)
    position[order] = np.arange(n)
    improved = False
    for i in range(n - 1):
        a, b = order[i], order[i + 1]
        dab = float(np.linalg.norm(pos[a] - pos[b]))
        for c in nbrs[a]:
            j = position[c]
            if j > i + 1:
                if j == n - 1:
                    delta = float(np.linalg.norm(pos[a] - pos[c])) - dab
                else:
                    cn = order[j + 1]
                    delta = (float(np.linalg.norm(pos[a] - pos[c])) + float(np.linalg.norm(pos[b] - pos[cn]))
                             - dab - float(np.linalg.norm(pos[c] - pos[cn])))
                if delta < -1e-12:
                    order[i + 1:j + 1] = order[i + 1:j + 1][::-1].copy()
                    position[order] = np.arange(n)
                    improved = True
                    break
            elif j < i - 1:
                # c lies before a: reverse order[j+1..i] -> edges (c,c_next),(a,b) become (c,a),(c_next,b)
                cn = order[j + 1]
                delta = (float(np.linalg.norm(pos[c] - pos[a])) + float(np.linalg.norm(pos[cn] - pos[b]))
                         - float(np.linalg.norm(pos[c] - pos[cn])) - dab)
                if delta < -1e-12:
                    order[j + 1:i + 1] = order[j + 1:i + 1][::-1].copy()
                    position[order] = np.arange(n)
                    improved = True
                    break
    return improved


def _or_opt_pass(pos, order, nbrs, max_seg: int = 3):
    """One Or-opt sweep: relocate short segments (1..max_seg points) next to a
    neighbour of one of their end points, possibly reversed.  Returns (order, improved)."""
    improved = False
    order = order.copy()
    n = len(order)

    def d(a, b):
        return float(np.linalg.norm(pos[a] - pos[b]))

    for seg_len in range(1, max_seg + 1):
        i = 0
        while i + seg_len <= n:
            seg = order[i:i + seg_len]
            prev = order[i - 1] if i > 0 else -1
            nxt = order[i + seg_len] if i + seg_len < n else -1
            remove_gain = 0.0
            if prev >= 0:
                remove_gain += d(prev, seg[0])
            if nxt >= 0:
                remove_gain += d(seg[-1], nxt)
            if prev >= 0 and nxt >= 0:
                remove_gain -= d(prev, nxt)
            if remove_gain <= 1e-12:
                i += 1
                continue
            rest = np.concatenate([order[:i], order[i + seg_len:]])
            pos_rest = np.full(n, -1, dtype=np.int64)
            pos_rest[rest] = np.arange(len(rest))
            best_delta, best_j, best_rev = -1e-12, -1, False
            for end in (seg[0], seg[-1]):
                for c in nbrs[end]:
                    j = pos_rest[c]
                    if j < 0:
                        continue
                    for ins in (j, j + 1):          # insert before or after c
                        left = rest[ins - 1] if ins - 1 >= 0 else -1
                        right = rest[ins] if ins < len(rest) else -1
                        if left < 0 and right < 0:
                            continue
                        base = -d(left, right) if (left >= 0 and right >= 0) else 0.0
                        for rev in (False, True):
                            s0, s1 = (seg[-1], seg[0]) if rev else (seg[0], seg[-1])
                            add = base
                            if left >= 0:
                                add += d(left, s0)
                            if right >= 0:
                                add += d(s1, right)
                            delta = add - remove_gain
                            if delta < best_delta:
                                best_delta, best_j, best_rev = delta, ins, rev
            if best_j >= 0:
                s = seg[::-1] if best_rev else seg
                order = np.concatenate([rest[:best_j], s, rest[best_j:]])
                improved = True
            i += 1
    return order, improved


def strokes_from_jumps(pos: np.ndarray, jump: float) -> np.ndarray:
    """Stroke ids for an ordered point list: a new stroke starts wherever the step
    between consecutive points exceeds ``jump``."""
    if len(pos) == 0:
        return np.zeros(0, dtype=np.int64)
    step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    return np.concatenate([[0], np.cumsum(step > jump)]).astype(np.int64)


# ============================================================================
# sample.py
# Point pattern generators.
#
# spiral   contour-parallel: geodesic iso-distance rings from the rim's outer edge,
#          ordered outside-in / top-down (optionally blended into one continuous helix)
# zigzag   boustrophedon: scan lines on wide flat regions, serpentine rings on walls,
#          annular tops and narrow regions; regions chained rim -> wall -> ... -> floor
# poisson  blue-noise (evenly spaced random) samples; ordered afterwards (nearest/tsp)
#
# All functions work in the internal Z-up frame on the cleaned/refined mesh.
# ============================================================================

TOP_LABELS = (RIM, TREAD, INNER_FLOOR)


@dataclass
class Stroke:
    """One continuous run of points (a ring, a scan line, a chain or a single point)."""
    pos: np.ndarray        # (k,3)
    face: np.ndarray       # (k,) face index into the (canonical) mesh
    bary: np.ndarray       # (k,3) barycentric coordinates inside ``face``
    closed: bool = False
    tag: str = ""
    iso: float = float("nan")   # distance level the stroke was extracted at (rings)


# ----------------------------------------------------------------------------- helpers

class SubMesh:
    """A face subset of ``parent`` as its own Trimesh, with index maps back."""

    def __init__(self, parent: trimesh.Trimesh, face_idx):
        self.parent = parent
        self.face_map = np.asarray(face_idx, dtype=np.int64)
        fsel = parent.faces[self.face_map]
        uv, inv = np.unique(fsel, return_inverse=True)
        self.vert_map = uv
        self.mesh = trimesh.Trimesh(parent.vertices[uv], inv.reshape(-1, 3), process=False)

    def components(self):
        if len(self.face_map) == 0:
            return []
        adj = self.mesh.face_adjacency
        comps = trimesh.graph.connected_components(adj, nodes=np.arange(len(self.face_map)), min_len=1)
        return [SubMesh(self.parent, self.face_map[np.asarray(cc)]) for cc in comps]

    @property
    def area(self):
        return float(self.mesh.area)


class EdgeIndex:
    """Lookup from a sorted vertex pair to the parent's face-adjacency entry."""

    def __init__(self, mesh: trimesh.Trimesh):
        e = np.asarray(mesh.face_adjacency_edges, dtype=np.int64)
        self.nV = len(mesh.vertices)
        self.adj = np.asarray(mesh.face_adjacency, dtype=np.int64)
        keys = e[:, 0] * self.nV + e[:, 1]
        self.sorter = np.argsort(keys)
        self.sorted = keys[self.sorter]

    def lookup(self, pairs: np.ndarray) -> np.ndarray:
        pairs = np.sort(np.asarray(pairs, dtype=np.int64), axis=1)
        k = pairs[:, 0] * self.nV + pairs[:, 1]
        i = np.clip(np.searchsorted(self.sorted, k), 0, max(len(self.sorted) - 1, 0))
        if len(self.sorted) == 0:
            return np.full(len(k), -1, dtype=np.int64)
        found = self.sorted[i] == k
        return np.where(found, self.sorter[i], -1)


def _boundary_sources(sub: SubMesh, seg: Segmentation, edge_index: EdgeIndex, prefer: str = "exterior"):
    """Source vertices (sub-mesh ids) for the distance field and the remaining
    (non-source) boundary vertices.

    prefer='exterior': boundary edges adjacent to outer_wall/bottom faces or on the
    mesh's main exterior boundary loop; fallback = loop with the largest XY area.
    prefer='top': the boundary loop with the highest mean height (walls).
    prefer='area': the loop with the largest projected area (annular tops).
    """
    be, _ = boundary_edges(sub.mesh.faces)
    V = sub.mesh.vertices
    if len(be) == 0:
        return np.array([int(np.argmax(V[:, 2]))]), np.zeros(0, dtype=np.int64)
    src = np.zeros(0, dtype=np.int64)
    if prefer == "exterior":
        orig = sub.vert_map[be]
        ai = edge_index.lookup(orig)
        exterior = np.zeros(len(be), dtype=bool)
        has = ai >= 0
        if has.any():
            f2 = edge_index.adj[ai[has]]
            in_sub = np.isin(f2, sub.face_map)
            other = np.where(in_sub[:, 0], f2[:, 1], f2[:, 0])
            exterior[has] = np.isin(seg.labels[other], [OUTER_WALL, BOTTOM])
        if (~has).any() and len(seg.exterior_edges):
            nV = edge_index.nV
            k = orig[:, 0] * nV + orig[:, 1]
            ek = seg.exterior_edges[:, 0] * nV + seg.exterior_edges[:, 1]
            exterior[~has] = np.isin(k[~has], ek)
        src = np.unique(be[exterior])
    if len(src) == 0:
        chains = chain_segments(be)
        best, best_score = None, -np.inf
        for segs, nodes, closed in chains:
            pts = V[nodes]
            if prefer == "top":
                score = float(pts[:, 2].mean())
            else:
                score = abs(signed_area_xy(pts)) if len(pts) > 2 else float(polyline_length(pts, closed))
            if score > best_score:
                best, best_score = nodes, score
        src = np.unique(best)
    non_src = np.setdiff1d(np.unique(be), src)
    return src, non_src


def _bary(V, F, faces, pts):
    tri = V[F[faces]]
    b = trimesh.triangles.points_to_barycentric(tri, pts, method="cramer")
    b = np.clip(b, 0.0, 1.0)
    s = b.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return b / s


def _orient_ccw(pts: np.ndarray, closed: bool, center_xy: np.ndarray) -> bool:
    """True if the polyline runs counter-clockwise about +Z (closed: signed area,
    open: angular sweep about the component centre)."""
    if closed:
        return signed_area_xy(pts) >= 0
    ang = np.arctan2(pts[:, 1] - center_xy[1], pts[:, 0] - center_xy[0])
    sweep = np.unwrap(ang)
    return (sweep[-1] - sweep[0]) >= 0


def _rings_at(sub: SubMesh, dist: np.ndarray, dbound, isos, spacing: float, margin: float,
              center_xy: np.ndarray, want_ccw: bool | None = True):
    """Extract, filter, resample and orient the iso-rings at the given distances.
    Returns a list of (iso, Stroke)."""
    V, F = sub.mesh.vertices, sub.mesh.faces
    out = []
    for iso in isos:
        for pts, faces, closed in iso_contours(V, F, dist, float(iso)):
            L = polyline_length(pts, closed)
            if (closed and L < 2.5 * spacing) or (not closed and L < 0.5 * spacing):
                if closed:                                   # tiny loop at a centre -> one point
                    p, _, fsub = trimesh.proximity.closest_point(sub.mesh, pts.mean(axis=0, keepdims=True))
                    out.append((iso, Stroke(p, sub.face_map[fsub], _bary(V, F, fsub, p), False)))
                continue
            samples, si, _ = resample_polyline(pts, spacing, closed)
            fsub = faces[np.clip(si, 0, len(faces) - 1)]
            bary = _bary(V, F, fsub, samples)
            if dbound is not None:
                db = (dbound[F[fsub]] * bary).sum(axis=1)
                keep = db >= 0.5 * margin
                if not keep.all():
                    if keep.sum() < 2:
                        continue
                    # a filtered ring is no longer closed: split into runs of kept points
                    runs = np.split(np.arange(len(keep)), np.where(np.diff(keep.astype(int)) != 0)[0] + 1)
                    for run in runs:
                        if keep[run[0]] and len(run) >= 2:
                            out.append((iso, Stroke(samples[run], sub.face_map[fsub[run]], bary[run], False)))
                    continue
            if want_ccw is not None and _orient_ccw(samples, closed, center_xy) != want_ccw:
                samples, fsub, bary = samples[::-1], fsub[::-1], bary[::-1]
            out.append((iso, Stroke(samples, sub.face_map[fsub], bary, closed, iso=float(iso))))
    return out


def _sequence(ring_list, end, alternate: bool = False):
    """Order rings by a depth-first walk of the contour tree (a ring's parent is the
    nearest ring one level further out), so nested regions - e.g. two compartments
    of a tray - are finished one after the other instead of alternating.  Every ring
    starts nearest to where the previous one ended: closed rings are rotated (and,
    if ``alternate``, flipped every ring for a serpentine), open chains may be
    reversed.  Returns (strokes, new_end)."""
    if not ring_list:
        return [], end
    n = len(ring_list)
    isos = np.array([float(iso) for iso, _ in ring_list])
    levels = np.unique(isos)
    level_of = np.searchsorted(levels, isos)
    trees = [cKDTree(st.pos) for _, st in ring_list]
    children = [[] for _ in range(n)]
    roots = []
    for i in range(n):
        prev = np.where(level_of == level_of[i] - 1)[0] if level_of[i] > 0 else []
        if len(prev) == 0:
            roots.append(i)
            continue
        d = [trees[j].query(ring_list[i][1].pos)[0].min() for j in prev]
        children[int(prev[int(np.argmin(d))])].append(i)

    state = {"end": end, "flip": False}
    strokes = []

    def nearest(cands):
        if state["end"] is None:
            return cands[0]
        d = [trees[c].query(state["end"])[0] for c in cands]
        return cands[int(np.argmin(d))]

    def visit(i):
        s = ring_list[i][1]
        e = state["end"]
        if s.closed:
            if alternate and state["flip"]:
                s = Stroke(s.pos[::-1], s.face[::-1], s.bary[::-1], True, s.tag, s.iso)
            if e is not None:
                i0 = int(np.argmin(np.linalg.norm(s.pos - e, axis=1)))
                s = Stroke(np.roll(s.pos, -i0, axis=0), np.roll(s.face, -i0), np.roll(s.bary, -i0, axis=0), True, s.tag, s.iso)
        else:
            if e is not None and np.linalg.norm(s.pos[-1] - e) < np.linalg.norm(s.pos[0] - e):
                s = Stroke(s.pos[::-1], s.face[::-1], s.bary[::-1], False, s.tag, s.iso)
            elif e is None and alternate and state["flip"]:
                s = Stroke(s.pos[::-1], s.face[::-1], s.bary[::-1], False, s.tag, s.iso)
        strokes.append(s)
        state["end"] = s.pos[-1]
        state["flip"] = not state["flip"]
        kids = list(children[i])
        while kids:
            k = nearest(kids)
            kids.remove(k)
            visit(k)

    pending = list(roots)
    while pending:
        r = nearest(pending)
        pending.remove(r)
        visit(r)
    return strokes, state["end"]


def _point_at_fraction(loop: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Points on a closed polyline at normalised arc-length fractions ``u`` in [0,1)."""
    p = np.vstack([loop, loop[:1]])
    seg = np.diff(p, axis=0)
    L = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(L)])
    s_ = np.asarray(u) * cum[-1]
    idx = np.clip(np.searchsorted(cum, s_, side="right") - 1, 0, len(L) - 1)
    t = (s_ - cum[idx]) / np.where(L[idx] > 0, L[idx], 1.0)
    return p[idx] + seg[idx] * t[:, None]


def _make_continuous(strokes, sub: SubMesh, spacing: float, row: float):
    """Blend consecutive closed rings into one helix: ring k drifts towards the
    matching arc-length position on ring k+1 as it goes around (so it lands exactly
    where ring k+1 starts) and is re-projected onto the surface.  The innermost ring
    of a region spirals into its centre (absorbing the final centre point).  Rings
    that do not run alongside each other (another cavity, walls without a floor)
    are left as they are."""
    if len(strokes) < 2:
        return strokes, [False] * len(strokes)
    V, F = sub.mesh.vertices, sub.mesh.faces
    rings = list(strokes)
    out, joined = [], []
    prev_blended = False
    k = 0
    while k < len(rings):
        s = rings[k]
        nxt = rings[k + 1] if k + 1 < len(rings) else None
        target, consume_next = None, False
        if s.closed and len(s.pos) >= 6 and np.isfinite(s.iso):
            nested = (nxt is not None and nxt.closed and len(nxt.pos) >= 6 and np.isfinite(nxt.iso)
                      and nxt.iso > s.iso + 1e-9)
            if nested:
                j = int(np.argmin(np.linalg.norm(nxt.pos - s.pos[0], axis=1)))
                nxt = Stroke(np.roll(nxt.pos, -j, axis=0), np.roll(nxt.face, -j), np.roll(nxt.bary, -j, axis=0),
                             True, nxt.tag, nxt.iso)
                rings[k + 1] = nxt
                dn, _ = cKDTree(nxt.pos).query(s.pos)
                if float(dn.max()) < 2.5 * row:
                    target = _point_at_fraction(nxt.pos, np.arange(len(s.pos)) / len(s.pos))
            if target is None:
                # innermost ring: spiral into the centre if the centre lies on this surface
                if nxt is not None and len(nxt.pos) == 1 and np.linalg.norm(nxt.pos[0] - s.pos.mean(axis=0)) < row:
                    centre, consume_next = nxt.pos[0], True
                else:
                    c, dist_c, _ = trimesh.proximity.closest_point(sub.mesh, s.pos.mean(axis=0)[None])
                    centre = c[0] if float(dist_c[0]) < 0.5 * row else None
                if centre is not None and np.linalg.norm(s.pos - centre, axis=1).max() < 1.5 * row:
                    target = np.tile(centre, (len(s.pos), 1))
                else:
                    consume_next = False
        if target is not None:
            u = np.arange(len(s.pos)) / len(s.pos)
            p = s.pos + u[:, None] * (target - s.pos)
            if consume_next:
                p = np.vstack([p, target[:1]])
            # a spiral that converges keeps shrinking its steps: thin to >= spacing/2
            keep = [0]
            for i in range(1, len(p)):
                if np.linalg.norm(p[i] - p[keep[-1]]) >= 0.5 * spacing:
                    keep.append(i)
                elif i == len(p) - 1 and len(keep) > 1:
                    keep[-1] = i                      # keep the very last point (the centre) instead
            p = p[keep]
            closest, _, fsub = trimesh.proximity.closest_point(sub.mesh, p)
            out.append(Stroke(closest, sub.face_map[fsub], _bary(V, F, fsub, closest), False, s.tag))
            joined.append(prev_blended)
            prev_blended = True
            if consume_next:
                k += 1
                prev_blended = False
        else:
            out.append(Stroke(s.pos, s.face, s.bary, False, s.tag))
            joined.append(prev_blended)
            prev_blended = False
        k += 1
    return out, joined


def _iso_levels_symmetric(dmax: float, row: float, margin: float) -> np.ndarray:
    if dmax <= 2 * margin:
        return np.array([0.5 * dmax])
    n = int(np.floor((dmax - 2 * margin) / row + 1e-9)) + 1
    start = 0.5 * (dmax - (n - 1) * row)
    return start + row * np.arange(n)


# ----------------------------------------------------------------------------- patterns

def spiral_pattern(mesh: trimesh.Trimesh, seg: Segmentation, mask: np.ndarray, spacing: float,
                   row_spacing: float, margin: float, continuous: bool = False, verbose: bool = False):
    """Geodesic contour-parallel rings from the exterior edge of the selection inward."""
    fidx = np.where(mask)[0]
    if len(fidx) == 0:
        return [], []
    comps = SubMesh(mesh, fidx).components()
    comps.sort(key=lambda s: (-float(s.mesh.vertices[:, 2].max()), -s.area))
    edge_index = EdgeIndex(mesh)
    strokes, joined = [], []
    end = None
    for sub in comps:
        if sub.area < (0.5 * spacing) ** 2:
            continue
        V, F = sub.mesh.vertices, sub.mesh.faces
        src, non_src = _boundary_sources(sub, seg, edge_index, prefer="exterior")
        dist = geodesic_distance(V, F, src)
        dbound = geodesic_distance(V, F, non_src) if len(non_src) else None
        dmax = float(dist.max())
        isos = margin + row_spacing * np.arange(int(np.floor((dmax - margin) / row_spacing)) + 1) if dmax > margin \
            else np.array([0.5 * dmax])
        isos = isos[isos < dmax - 1e-9] if len(isos) > 1 else isos
        center_xy = V.mean(axis=0)[:2]
        rings = _rings_at(sub, dist, dbound, isos, spacing, margin, center_xy, want_ccw=True)
        # finish at the innermost point if the last ring leaves a gap
        imax = int(np.argmax(dist))
        if len(isos) and dmax - isos[-1] > 0.5 * row_spacing and (dbound is None or dbound[imax] >= margin):
            p = V[imax][None]
            last = [st for iso, st in rings if float(iso) == float(isos[-1])]
            near = min((np.linalg.norm(st.pos - p, axis=1).min() for st in last), default=np.inf)
            if near >= 0.5 * spacing:
                f0 = int(sub.mesh.vertex_faces[imax][0])
                rings.append((dmax, Stroke(p, sub.face_map[[f0]], _bary(V, F, [f0], p), False)))
        comp_strokes, end = _sequence(rings, end)
        if continuous:
            comp_strokes, comp_joined = _make_continuous(comp_strokes, sub, spacing, row_spacing)
        else:
            comp_joined = [False] * len(comp_strokes)
        strokes += comp_strokes
        joined += comp_joined
        if verbose:
            print(f"  spiral component: {len(sub.face_map)} faces, {len(src)} source vertices, "
                  f"{len(isos)} rings, max distance {dmax:.4f} m")
    return strokes, joined


def _projected_polygon(sub: SubMesh):
    be, _ = boundary_edges(sub.mesh.faces)
    loops = []
    for segs, nodes, closed in chain_segments(be):
        if closed and len(nodes) >= 3:
            loops.append(sub.mesh.vertices[nodes][:, :2])
    if not loops:
        tri = sub.mesh.vertices[sub.mesh.faces][:, :, :2]
        return unary_union([Polygon(t) for t in tri])
    loops.sort(key=lambda l: -abs(signed_area_xy(np.c_[l, np.zeros(len(l))])))
    shell = loops[0]
    holes = [l for l in loops[1:] if Polygon(shell).contains(Polygon(l).representative_point())]
    poly = Polygon(shell, holes)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def _lift(sub: SubMesh, xy: np.ndarray):
    """Project 2-D points vertically onto the piece's surface (ray cast from above)."""
    ztop = float(sub.mesh.vertices[:, 2].max()) + 1.0
    origins = np.c_[xy, np.full(len(xy), ztop)]
    dirs = np.tile([0.0, 0.0, -1.0], (len(xy), 1))
    loc, ray_idx, tri = sub.mesh.ray.intersects_location(origins, dirs, multiple_hits=False)
    pos = np.full((len(xy), 3), np.nan)
    face = np.full(len(xy), -1, dtype=np.int64)
    pos[ray_idx] = loc
    face[ray_idx] = tri
    return pos, face


def _scanline_strokes(sub: SubMesh, poly, spacing: float, row: float, margin: float,
                      angle_deg: float, end):
    p = poly.buffer(-margin)
    if p.is_empty:
        p = poly.buffer(-0.5 * margin)
    if p.is_empty:
        return None, end
    a = np.radians(angle_deg)
    d = np.array([np.cos(a), np.sin(a)])
    nrm = np.array([-np.sin(a), np.cos(a)])
    coords = np.array(p.exterior.coords) if isinstance(p, Polygon) else \
        np.vstack([np.array(g.exterior.coords) for g in p.geoms])
    s = coords @ nrm
    t = coords @ d
    offsets = _iso_levels_symmetric(float(s.max() - s.min()), row, 0.0) + s.min()
    c0 = 0.5 * (t.min() + t.max())
    half = 0.5 * (t.max() - t.min()) + 1.0
    lines = []
    for o in offsets:
        base = nrm * o + d * c0
        line = LineString([base - d * half, base + d * half])
        inter = line.intersection(p)
        pieces = []
        geoms = getattr(inter, "geoms", [inter])
        for g in geoms:
            if isinstance(g, LineString) and g.length > 1e-9:
                c = np.array(g.coords)
                if (c[-1] - c[0]) @ d < 0:
                    c = c[::-1]
                pieces.append(c)
        pieces.sort(key=lambda c: c[0] @ d)
        if pieces:
            lines.append(pieces)
    if not lines:
        return None, end
    # start with the line nearest to the current end point
    if end is not None:
        ends = [lines[0][0][0], lines[0][-1][-1], lines[-1][0][0], lines[-1][-1][-1]]
        dd = [np.linalg.norm(e[:2] - end[:2]) for e in ends]
        k = int(np.argmin(dd))
        if k >= 2:
            lines = lines[::-1]
        forward = k in (0, 2)
    else:
        forward = True
    strokes = []
    for pieces in lines:
        if not forward:
            pieces = [c[::-1] for c in pieces[::-1]]
        for c in pieces:
            samples, _, _ = resample_polyline(np.c_[c, np.zeros(len(c))], spacing, False)
            pos, face = _lift(sub, samples[:, :2])
            ok = face >= 0
            if ok.sum() == 0:
                continue
            pos, face = pos[ok], face[ok]
            bary = _bary(sub.mesh.vertices, sub.mesh.faces, face, pos)
            strokes.append(Stroke(pos, sub.face_map[face], bary, False))
            end = pos[-1]
        forward = not forward
    return strokes, end


def _thin_strokes(strokes, min_dist: float):
    """Drop points closer than ``min_dist`` to any point kept so far, across all
    strokes (collapses the two offset sides of a narrow region into one pass)."""
    kept = []
    out = []
    for st in strokes:
        keep = []
        for i, p in enumerate(st.pos):
            if kept and np.min(np.linalg.norm(np.array(kept) - p, axis=1)) < min_dist:
                continue
            kept.append(p)
            keep.append(i)
        if keep:
            keep = np.array(keep)
            out.append(Stroke(st.pos[keep], st.face[keep], st.bary[keep], False, st.tag))
    return out


def _ring_strokes(sub: SubMesh, seg, edge_index, spacing, row, margin, prefer, end, alternate=True):
    """Serpentine rings on one region piece.

    walls (prefer='top'): rings at uniform surface distance from the top edge.
    tops (prefer='area'): rings from the outer boundary inward; the region's real
    width is taken from the distance-to-any-boundary field so frames with ribs
    (e.g. a tray rim with a divider) are handled, and regions too narrow for two
    rings get a single centreline stroke instead.
    """
    V, F = sub.mesh.vertices, sub.mesh.faces
    src, non_src = _boundary_sources(sub, seg, edge_index, prefer=prefer)
    dist = geodesic_distance(V, F, src)
    dbound = geodesic_distance(V, F, non_src) if len(non_src) else None
    dmax = float(dist.max())
    center_xy = V.mean(axis=0)[:2]
    if prefer != "top" and len(non_src):
        dall = np.minimum(dist, dbound)
        fmax = float(dall.max())
        if fmax < row:                                     # narrow: one centreline stroke
            iso = min(margin, 0.8 * fmax)
            rings = _rings_at(sub, dall, None, [iso], spacing, margin, center_xy, want_ccw=True)
            ordered, _ = _sequence(rings, end, alternate=False)
            thinned = _thin_strokes(ordered, 0.8 * spacing)
            return _sequence([(0.0, st) for st in thinned], end, alternate=alternate)
        dmax = min(dmax, 2.0 * fmax)
    isos = _iso_levels_symmetric(dmax, row, margin)
    rings = _rings_at(sub, dist, dbound, isos, spacing, margin, center_xy, want_ccw=True)
    return _sequence(rings, end, alternate=alternate)


class _Region:
    def __init__(self, label: int, level: int):
        self.label, self.level = label, level

    @property
    def name(self):
        return REGION_NAMES[self.label]


def zigzag_pattern(mesh: trimesh.Trimesh, seg: Segmentation, mask: np.ndarray, spacing: float,
                   row_spacing: float, margin: float, angle_deg: float = 0.0, verbose: bool = False):
    """Boustrophedon per region piece, pieces chained rim -> nearest -> ... -> floor."""
    pieces = []          # (label, level, SubMesh)
    key = seg.labels * 1000 + seg.level
    for k in np.unique(key[mask]):
        f = np.where(mask & (key == k))[0]
        for piece in SubMesh(mesh, f).components():
            if piece.area >= (0.5 * spacing) ** 2:
                pieces.append((_Region(int(k // 1000), int(k % 1000)), piece))
    if not pieces:
        return [], []
    edge_index = EdgeIndex(mesh)
    # start from the rim (largest first), then always the nearest unvisited piece
    rims = sorted([i for i, (c, _) in enumerate(pieces) if c.label == RIM], key=lambda i: -pieces[i][1].area)
    start = rims[0] if rims else int(np.argmax([p.mesh.vertices[:, 2].max() for _, p in pieces]))
    remaining = set(range(len(pieces))) - {start}
    sequence = [start]
    strokes = []
    end = None

    def run_piece(i):
        nonlocal end
        comp, piece = pieces[i]
        if comp.label in TOP_LABELS or comp.label == BOTTOM:
            poly = _projected_polygon(piece)
            annular = isinstance(poly, Polygon) and len(poly.interiors) > 0
            narrow = poly.buffer(-row_spacing).is_empty
            if not annular and not narrow:
                s, end2 = _scanline_strokes(piece, poly, spacing, row_spacing, margin, angle_deg, end)
                if s:
                    for st in s:
                        st.tag = f"{comp.name}:scan"
                    end = end2
                    return s
            s, end = _ring_strokes(piece, seg, edge_index, spacing, row_spacing, margin, "area", end)
        else:
            s, end = _ring_strokes(piece, seg, edge_index, spacing, row_spacing, margin, "top", end)
        for st in s:
            st.tag = f"{comp.name}:ring"
        return s

    cur = start
    while True:
        strokes += run_piece(cur)
        if not remaining:
            break
        cands = list(remaining)
        if end is None:
            nxt = cands[0]
        else:
            dd = [np.min(np.linalg.norm(pieces[i][1].mesh.triangles_center - end, axis=1)) for i in cands]
            nxt = cands[int(np.argmin(dd))]
        remaining.discard(nxt)
        sequence.append(nxt)
        cur = nxt
    if verbose:
        names = [f"{pieces[i][0].name}(L{pieces[i][0].level})" for i in sequence]
        print("  zigzag region order: " + " -> ".join(names))
    return strokes, [False] * len(strokes)


def poisson_disk(mesh: trimesh.Trimesh, spacing: float, seed: int = 0, oversample: int = 30):
    """Blue-noise samples whose density matches a grid of pitch ``spacing``
    (about area / spacing**2 points): greedy dart thinning of a dense random set,
    with the exclusion radius calibrated by bisection."""
    rng = np.random.default_rng(seed)
    target = max(1, int(round(mesh.area / spacing ** 2)))
    cand, fidx = trimesh.sample.sample_surface(mesh, oversample * target, seed=seed)
    cand = np.asarray(cand)
    tree = cKDTree(cand)
    perm = rng.permutation(len(cand))

    def thin(r):
        blocked = np.zeros(len(cand), dtype=bool)
        accepted = []
        for i in perm:
            if blocked[i]:
                continue
            accepted.append(i)
            blocked[tree.query_ball_point(cand[i], r)] = True
        return np.array(accepted)

    lo, hi = 0.4 * spacing, 1.2 * spacing
    best = None
    for _ in range(10):
        r = 0.5 * (lo + hi)
        acc = thin(r)
        if best is None or abs(len(acc) - target) < abs(len(best) - target):
            best = acc
        if abs(len(acc) - target) <= max(2, 0.03 * target):
            break
        if len(acc) > target:
            lo = r
        else:
            hi = r
    return cand[best], np.asarray(fidx)[best]


def poisson_pattern(mesh: trimesh.Trimesh, mask: np.ndarray, spacing: float, seed: int = 0,
                    verbose: bool = False):
    """Evenly spaced random (blue-noise) samples at grid-equivalent density."""
    fidx = np.where(mask)[0]
    if len(fidx) == 0:
        return [], []
    sub = SubMesh(mesh, fidx)
    pts, face = poisson_disk(sub.mesh, spacing, seed)
    face = np.asarray(face, dtype=np.int64)
    bary = _bary(sub.mesh.vertices, sub.mesh.faces, face, pts)
    if verbose:
        print(f"  poisson: {len(pts)} samples on {sub.area:.4f} m² (target {sub.area / spacing ** 2:.0f})")
    return [Stroke(np.asarray(pts), sub.face_map[face], bary, False, "poisson")], [False]


def strokes_to_arrays(mesh: trimesh.Trimesh, strokes, joined, crease_angle: float):
    """Flatten strokes into (pos, normal, face, stroke_id)."""
    if not strokes:
        z = np.zeros((0, 3))
        return z, z, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    pos = np.vstack([s.pos for s in strokes])
    face = np.concatenate([s.face for s in strokes])
    bary = np.vstack([s.bary for s in strokes])
    sid, ids = 0, []
    for k, s in enumerate(strokes):
        if k > 0 and not joined[k]:
            sid += 1
        ids.append(np.full(len(s.pos), sid, dtype=np.int64))
    normal = smooth_point_normals(mesh, face, bary, crease_angle)
    return pos, normal, face, np.concatenate(ids)


# ============================================================================
# pipeline.py
# End-to-end API: mesh file -> ordered, region-aware wiping point set.
# ============================================================================

PATTERNS = ("spiral", "zigzag", "poisson")
ORDERS = ("auto", "none", "nearest", "tsp")

CSV_HEADER = ("idx,stroke,region,level,reachable,"
              "tool_x,tool_y,tool_z,tool_dx,tool_dy,tool_dz,"
              "norm_x,norm_y,norm_z,norm_dx,norm_dy,norm_dz")


@dataclass
class PointSet:
    """Ordered points in the input mesh frame.

    pos        (N,3) tool position == normal position (a point on the surface)
    normal     (N,3) unit outward surface normal (``norm_dir``)
    tool_dir   (N,3) == -normal, pointing into the material
    face       (N,)  index of the mesh face the point lies on
    stroke     (N,)  id of the continuous stroke the point belongs to (lift between ids)
    region     (N,)  region id (see REGION_NAMES)
    level      (N,)  step level (0 = rim, increasing downwards)
    reachable  (N,)  visible from straight above
    """
    pos: np.ndarray
    normal: np.ndarray
    face: np.ndarray
    stroke: np.ndarray
    region: np.ndarray
    level: np.ndarray
    reachable: np.ndarray

    @property
    def tool_dir(self) -> np.ndarray:
        return -self.normal

    @property
    def region_names(self) -> np.ndarray:
        return np.array(REGION_NAMES)[self.region]

    def __len__(self) -> int:
        return len(self.pos)

    def subset(self, idx) -> "PointSet":
        return PointSet(self.pos[idx], self.normal[idx], self.face[idx], self.stroke[idx],
                        self.region[idx], self.level[idx], self.reachable[idx])

    def nn_distance(self) -> np.ndarray:
        if len(self) < 2:
            return np.zeros(len(self))
        d, _ = cKDTree(self.pos).query(self.pos, k=2)
        return d[:, 1]

    def to_csv(self, path: str):
        n = len(self)
        cols = np.c_[self.pos, self.tool_dir, self.pos, self.normal]
        with open(path, "w") as f:
            f.write(CSV_HEADER + "\n")
            names = self.region_names
            for i in range(n):
                vals = ",".join(f"{v:.6f}" for v in cols[i])
                f.write(f"{i},{self.stroke[i]},{names[i]},{self.level[i]},{int(self.reachable[i])},{vals}\n")

    def to_npz(self, path: str):
        np.savez_compressed(path, tool_pos=self.pos, tool_dir=self.tool_dir, norm_pos=self.pos,
                            norm_dir=self.normal, stroke=self.stroke, region=self.region_names,
                            region_id=self.region, level=self.level, reachable=self.reachable, face=self.face)

    def to_ply(self, path: str):
        """Point cloud with normals (open in MeshLab/CloudCompare); ``order`` is stored as
        a scalar property."""
        n = len(self)
        with open(path, "w") as f:
            f.write("ply\nformat ascii 1.0\n")
            f.write(f"element vertex {n}\n")
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property float nx\nproperty float ny\nproperty float nz\n")
            f.write("property int order\nproperty int stroke\nproperty int region\n")
            f.write("end_header\n")
            for i in range(n):
                p, nn = self.pos[i], self.normal[i]
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {nn[0]:.5f} {nn[1]:.5f} {nn[2]:.5f} "
                        f"{i} {self.stroke[i]} {self.region[i]}\n")


@dataclass
class Result:
    points: PointSet
    mesh: trimesh.Trimesh          # cleaned + refined mesh in the input frame (faces match ``seg``)
    seg: Segmentation
    frame: Frame
    params: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)

    def save(self, out_dir: str, name: str, formats=("csv", "npz", "json"), seg_mesh: bool = False) -> dict:
        os.makedirs(out_dir, exist_ok=True)
        paths = {}
        if "csv" in formats:
            paths["csv"] = os.path.join(out_dir, f"{name}_points.csv")
            self.points.to_csv(paths["csv"])
        if "npz" in formats:
            paths["npz"] = os.path.join(out_dir, f"{name}_points.npz")
            self.points.to_npz(paths["npz"])
        if "ply" in formats:
            paths["ply"] = os.path.join(out_dir, f"{name}_points.ply")
            self.points.to_ply(paths["ply"])
        if "json" in formats:
            paths["json"] = os.path.join(out_dir, f"{name}_meta.json")
            with open(paths["json"], "w") as f:
                json.dump({"params": self.params, "stats": self.stats,
                           "regions": self.seg.summary(self.mesh)}, f, indent=2, default=_json_default)
        if seg_mesh:
            paths["seg_ply"] = os.path.join(out_dir, f"{name}_segmented.ply")
            m = self.mesh.copy()
            cols = np.array([trimesh.visual.color.hex_to_rgba(REGION_COLORS[n]) for n in REGION_NAMES])
            fc = cols[self.seg.labels]
            fc[~self.seg.reachable, :3] = (fc[~self.seg.reachable, :3] * 0.45).astype(np.uint8)
            m.visual.face_colors = fc
            m.export(paths["seg_ply"])
        return paths

    def summary_text(self) -> str:
        s = self.stats
        lines = [f"points: {s['n_points']}  strokes: {s['n_strokes']}  path length: {s['path_length']:.3f} m",
                 f"spacing target {self.params['spacing']:.4f} m | nearest-neighbour distance "
                 f"mean {s['nn_mean']:.4f} median {s['nn_median']:.4f} min {s['nn_min']:.4f} max {s['nn_max']:.4f}",
                 f"step within strokes mean {s['step_mean']:.4f}  min {s['step_min']:.4f}  max {s['step_max']:.4f}"]
        lines.append("points per region: " + ", ".join(f"{k}={v}" for k, v in s["points_per_region"].items()))
        lines.append(f"time: {s['time_total']:.1f}s")
        return "\n".join(lines)


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)


def _parse_names(v):
    if v is None:
        return None
    if isinstance(v, str):
        if v.lower() == "all":
            return "all"
        return tuple(x.strip() for x in v.split(",") if x.strip())
    return tuple(v)


def generate(mesh, spacing: float = 0.01, row_spacing: float | None = None, pattern: str = "spiral",
             order: str = "auto", include=DEFAULT_INCLUDE, exclude=(), reachable_only: bool = True,
             up="z", scale: float = 1.0, margin: float | None = None, continuous: bool = False,
             zigzag_angle: float = 0.0, top_angle: float = 45.0, bottom_angle: float = 45.0,
             crease_angle: float = 15.0, normal_smoothing: float = 30.0, refine: bool = True,
             max_edge: float | None = None, flip_normals: bool = False, seed: int = 0,
             verbose: bool = True) -> Result:
    """Generate an ordered wiping point set for a container mesh.

    mesh            path to a mesh file or a trimesh.Trimesh (any units; set ``scale``
                    so that the result is in metres, e.g. 0.001 for mm meshes)
    spacing         target distance between consecutive points along a stroke [m]
    row_spacing     distance between rings / scan lines [m] (default = spacing)
    pattern         'spiral' | 'zigzag' | 'poisson'
    order           'auto' (pattern order; 'tsp' for poisson) | 'none' | 'nearest' | 'tsp'
    include/exclude region names (or 'all'); default = rim, inner_wall, tread, inner_floor
    reachable_only  drop faces/points that are not visible from straight above
    up              'z' (default), '-z', 'x', ..., 'auto', or an (x,y,z) vector
    margin          inset of the first ring / scan line from region edges (default row/2)
    continuous      spiral only: blend rings into one continuous helix per region
    zigzag_angle    scan line direction on flat regions [deg, about the up axis]
    """
    t0 = time.time()
    if pattern not in PATTERNS:
        raise ValueError(f"pattern must be one of {PATTERNS}")
    if order not in ORDERS:
        raise ValueError(f"order must be one of {ORDERS}")
    if isinstance(mesh, (str, os.PathLike)):
        src_name = os.path.basename(str(mesh))
        mesh_in = load_mesh(str(mesh), scale)
    else:
        src_name = "mesh"
        mesh_in = trimesh.Trimesh(np.asarray(mesh.vertices, dtype=float), np.asarray(mesh.faces), process=True)
        if scale != 1.0:
            mesh_in.apply_scale(scale)
        clean_mesh(mesh_in)
    row_spacing = spacing if row_spacing is None else row_spacing
    margin = 0.5 * row_spacing if margin is None else margin
    include = _parse_names(include)
    exclude = _parse_names(exclude) or ()
    diag = float(np.linalg.norm(mesh_in.extents))
    if verbose:
        print(f"[mesh] {src_name}: {len(mesh_in.faces)} faces, extents {np.round(mesh_in.extents, 4).tolist()} "
              f"(diag {diag:.3f}); spacing {spacing} row {row_spacing} margin {margin}")
        if diag > 20:
            print("  warning: the mesh is very large for metres - is it in millimetres? (use scale=0.001)")

    up_vec = parse_up(up, mesh_in)
    frame = Frame(up_vec)
    mc = frame.to_canonical(mesh_in)
    if refine:
        mc = refine_to_edge_length(mc, max_edge or 0.6 * min(spacing, row_spacing), verbose=verbose)
    fix_orientation(mc, flip=flip_normals, verbose=verbose)
    t1 = time.time()
    seg = segment(mc, top_angle=top_angle, bottom_angle=bottom_angle, crease_angle=crease_angle, verbose=verbose)
    t2 = time.time()
    mask = seg.select(include, exclude, reachable_only)
    if verbose:
        print(f"[select] {int(mask.sum())}/{len(mask)} faces, area {float(mc.area_faces[mask].sum()):.5f} m² "
              f"(include={include}, exclude={exclude}, reachable_only={reachable_only})")

    if pattern == "spiral":
        strokes, joined = spiral_pattern(mc, seg, mask, spacing, row_spacing, margin, continuous, verbose)
    elif pattern == "zigzag":
        strokes, joined = zigzag_pattern(mc, seg, mask, spacing, row_spacing, margin, zigzag_angle, verbose)
    else:
        strokes, joined = poisson_pattern(mc, mask, spacing, seed, verbose)
    pos, normal, face, stroke = strokes_to_arrays(mc, strokes, joined, normal_smoothing)
    t3 = time.time()

    reach = seg.reachable[face] if len(face) else np.zeros(0, dtype=bool)
    region = seg.labels[face] if len(face) else np.zeros(0, dtype=np.int64)
    level = seg.level[face] if len(face) else np.zeros(0, dtype=np.int64)
    jump = 1.75 * max(spacing, row_spacing)
    if len(pos) > 1:   # safety net: a stroke never contains a step longer than ``jump``
        step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        brk = (stroke[1:] != stroke[:-1]) | (step > jump)
        stroke = np.concatenate([[0], np.cumsum(brk)]).astype(np.int64)

    # ordering ------------------------------------------------------------------------
    if order == "auto":
        order = "tsp" if pattern == "poisson" else "none"
    if order in ("nearest", "tsp") and len(pos) > 1:
        o = nearest_neighbor_order(pos)
        if order == "tsp":
            o = improve_path(pos, o, verbose=verbose)
        pos, normal, face, region, level, reach = pos[o], normal[o], face[o], region[o], level[o], reach[o]
        stroke = strokes_from_jumps(pos, jump)
    # renumber strokes consecutively
    if len(stroke):
        _, stroke = np.unique(stroke, return_inverse=True)
        stroke = stroke.astype(np.int64)
    t4 = time.time()

    # back to the input frame ---------------------------------------------------------------
    pts = PointSet(frame.points_to_input(pos), frame.vectors_to_input(normal), face, stroke, region, level, reach)
    mesh_out = trimesh.Trimesh(frame.points_to_input(mc.vertices), mc.faces, process=False)

    params = dict(source=src_name, spacing=spacing, row_spacing=row_spacing, margin=margin, pattern=pattern,
                  order=order, include=include, exclude=exclude, reachable_only=reachable_only,
                  up=np.round(up_vec, 6).tolist(), scale=scale, continuous=continuous, zigzag_angle=zigzag_angle,
                  top_angle=top_angle, bottom_angle=bottom_angle, crease_angle=crease_angle,
                  normal_smoothing=normal_smoothing, units="m")
    stats = _stats(pts, spacing, row_spacing)
    stats.update(time_prepare=t1 - t0, time_segment=t2 - t1, time_sample=t3 - t2, time_order=t4 - t3,
                 time_total=time.time() - t0, n_faces=len(mc.faces), selected_area=float(mc.area_faces[mask].sum()))
    res = Result(pts, mesh_out, seg, frame, params, stats)
    if verbose:
        print("[result] " + res.summary_text().replace("\n", "\n         "))
    return res


def _stats(pts: PointSet, spacing: float, row_spacing: float) -> dict:
    n = len(pts)
    out = dict(n_points=int(n), n_strokes=int(len(np.unique(pts.stroke))) if n else 0,
               path_length=path_length(pts.pos) if n else 0.0)
    if n >= 2:
        nn = pts.nn_distance()
        step = np.linalg.norm(np.diff(pts.pos, axis=0), axis=1)
        same = pts.stroke[1:] == pts.stroke[:-1]
        within = step[same] if same.any() else step
        out.update(nn_mean=float(nn.mean()), nn_median=float(np.median(nn)), nn_min=float(nn.min()),
                   nn_max=float(nn.max()), step_mean=float(within.mean()), step_min=float(within.min()),
                   step_max=float(within.max()), n_jumps=int((~same).sum()))
    else:
        out.update(nn_mean=0.0, nn_median=0.0, nn_min=0.0, nn_max=0.0, step_mean=0.0, step_min=0.0,
                   step_max=0.0, n_jumps=0)
    names = pts.region_names
    out["points_per_region"] = {r: int((names == r).sum()) for r in REGION_NAMES if (names == r).any()}
    return out


# ============================================================================
# viz.py
# Visualisation: interactive Plotly HTML and a static matplotlib preview.
# ============================================================================

def preview_png(result, path: str, max_arrows: int = 300, dpi: int = 110, title: str | None = None):
    """Static 4-panel preview (top / front / side / 3-D): mesh context in grey, points
    coloured by order, strokes as lines, transitions dashed, a subset of arrows."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import cm

    pts = result.points
    mesh = result.mesh
    n = len(pts)
    spacing = result.params["spacing"]
    fig = plt.figure(figsize=(15, 11))
    views = [("top (XY)", 0, 1), ("front (XZ)", 0, 2), ("side (YZ)", 1, 2)]
    colors = cm.viridis(np.linspace(0, 1, max(n, 2)))[:n]
    rng = np.random.default_rng(0)
    vsub = mesh.vertices[rng.choice(len(mesh.vertices), min(6000, len(mesh.vertices)), replace=False)]
    sel = result.seg.select(result.params["include"], result.params["exclude"], result.params["reachable_only"])
    csel = mesh.triangles_center[sel]
    csel = csel[rng.choice(len(csel), min(6000, len(csel)), replace=False)] if len(csel) else csel
    arrow_idx = np.linspace(0, n - 1, min(max_arrows, n)).astype(int) if n else np.zeros(0, int)
    for k, (name, a, b) in enumerate(views):
        ax = fig.add_subplot(2, 2, k + 1)
        ax.scatter(vsub[:, a], vsub[:, b], s=1, c="#d0d0d0", alpha=0.5, linewidths=0)
        if len(csel):
            ax.scatter(csel[:, a], csel[:, b], s=1, c="#9ecae1", alpha=0.5, linewidths=0)
        if n:
            _draw_path_2d(ax, pts, a, b)
            ax.scatter(pts.pos[:, a], pts.pos[:, b], s=6, c=colors, linewidths=0, zorder=3)
            ax.quiver(pts.pos[arrow_idx, a], pts.pos[arrow_idx, b], pts.tool_dir[arrow_idx, a] * spacing,
                      pts.tool_dir[arrow_idx, b] * spacing, color="#d62728", angles="xy", scale_units="xy",
                      scale=1, width=0.002, zorder=4)
            ax.quiver(pts.pos[arrow_idx, a], pts.pos[arrow_idx, b], pts.normal[arrow_idx, a] * spacing,
                      pts.normal[arrow_idx, b] * spacing, color="#2ca02c", angles="xy", scale_units="xy",
                      scale=1, width=0.002, zorder=4)
            ax.scatter(pts.pos[:1, a], pts.pos[:1, b], s=80, marker="*", c="k", zorder=5)
        ax.set_aspect("equal")
        ax.set_title(name)
        ax.grid(alpha=0.3)
    ax = fig.add_subplot(2, 2, 4, projection="3d")
    if n:
        for sid in np.unique(pts.stroke):
            m = pts.stroke == sid
            ax.plot(pts.pos[m, 0], pts.pos[m, 1], pts.pos[m, 2], "-", lw=0.6, color="#555555")
        ax.scatter(pts.pos[:, 0], pts.pos[:, 1], pts.pos[:, 2], s=5, c=colors, linewidths=0)
        ax.quiver(pts.pos[arrow_idx, 0], pts.pos[arrow_idx, 1], pts.pos[arrow_idx, 2],
                  pts.normal[arrow_idx, 0], pts.normal[arrow_idx, 1], pts.normal[arrow_idx, 2],
                  length=spacing, color="#2ca02c", linewidth=0.5)
    ax.scatter(vsub[::4, 0], vsub[::4, 1], vsub[::4, 2], s=1, c="#d0d0d0", alpha=0.3, linewidths=0)
    ext = mesh.extents.max()
    c = mesh.bounds.mean(axis=0)
    ax.set_xlim(c[0] - ext / 2, c[0] + ext / 2)
    ax.set_ylim(c[1] - ext / 2, c[1] + ext / 2)
    ax.set_zlim(c[2] - ext / 2, c[2] + ext / 2)
    ax.set_title("3D (colour = order, green = normal)")
    s = result.stats
    fig.suptitle(title or f"{result.params['source']} | {result.params['pattern']} | {n} points, "
                 f"{s.get('n_strokes', 0)} strokes, path {s.get('path_length', 0):.2f} m | "
                 f"red = tool_dir, green = norm_dir, star = start", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def _draw_path_2d(ax, pts, a, b):
    p = pts.pos
    same = pts.stroke[1:] == pts.stroke[:-1]
    for i in np.where(same)[0]:
        pass
    # vectorised line segments: within-stroke solid, jumps dashed
    from matplotlib.collections import LineCollection
    segs = np.stack([p[:-1][:, [a, b]], p[1:][:, [a, b]]], axis=1)
    ax.add_collection(LineCollection(segs[same], colors="#555555", linewidths=0.5, zorder=2))
    if (~same).any():
        ax.add_collection(LineCollection(segs[~same], colors="#e41a1c", linewidths=0.6, linestyles="dashed", zorder=2))


def plotly_html(result, path: str, arrow_scale: float = 1.0, slider: bool = True, slider_steps: int = 40,
                labels_every: int | None = None, embed_js: bool = True, title: str | None = None,
                open_browser: bool = False) -> str:
    """Interactive viewer (standalone HTML).

    Layers (toggle in the legend): mesh per region (unreachable parts darker), points
    coloured by order, path (solid within a stroke, dashed transitions), tool_dir
    cones (red, into the surface), norm_dir cones (green, out of the surface) and
    order labels.  Hover a point for index / region / level / stroke.  The slider
    replays the order.
    """
    import plotly.graph_objects as go


    pts = result.points
    mesh = result.mesh
    seg = result.seg
    n = len(pts)
    spacing = result.params["spacing"]
    selected = seg.select(result.params["include"], result.params["exclude"], result.params["reachable_only"])
    fig = go.Figure()
    r5 = lambda a: np.round(a, 5)

    # --- mesh, one trace per region (and unreachable part) ---------------------------
    for rid, name in enumerate(REGION_NAMES):
        for reach in (True, False):
            m = (seg.labels == rid) & (seg.reachable == reach)
            if not m.any():
                continue
            sub = SubMesh(mesh, np.where(m)[0]).mesh
            V, F = r5(sub.vertices), sub.faces
            color = REGION_COLORS[name] if reach else UNREACHABLE_COLOR
            is_sel = bool(selected[m].any())
            fig.add_trace(go.Mesh3d(
                x=V[:, 0], y=V[:, 1], z=V[:, 2], i=F[:, 0], j=F[:, 1], k=F[:, 2],
                color=color, opacity=0.5 if is_sel else 0.25, flatshading=True,
                lighting=dict(ambient=0.55, diffuse=0.7, specular=0.05, roughness=0.9),
                name=f"{name}{'' if reach else ' (unreachable)'}" + (" ✓" if is_sel else ""),
                hoverinfo="name", showlegend=True, legendgroup="mesh", legendgrouptitle_text="mesh regions"))

    if n == 0:
        fig.update_layout(title="no points generated")
        fig.write_html(path, include_plotlyjs=True if embed_js else "cdn")
        return path

    P = r5(pts.pos)
    names = pts.region_names
    hover = [f"#{i} | {names[i]} L{pts.level[i]} | stroke {pts.stroke[i]}" for i in range(n)]
    order = np.arange(n)

    # --- path (solid within strokes, dashed between) -------------------------------------
    same = pts.stroke[1:] == pts.stroke[:-1]
    px, py, pz = [], [], []
    for i in range(n - 1):
        if same[i]:
            px += [P[i, 0], P[i + 1, 0], None]
            py += [P[i, 1], P[i + 1, 1], None]
            pz += [P[i, 2], P[i + 1, 2], None]
    fig.add_trace(go.Scatter3d(x=px, y=py, z=pz, mode="lines", line=dict(color="#444444", width=3),
                               name="path (strokes)", hoverinfo="skip", legendgroup="path"))
    jx, jy, jz = [], [], []
    for i in np.where(~same)[0]:
        jx += [P[i, 0], P[i + 1, 0], None]
        jy += [P[i, 1], P[i + 1, 1], None]
        jz += [P[i, 2], P[i + 1, 2], None]
    fig.add_trace(go.Scatter3d(x=jx, y=jy, z=jz, mode="lines", line=dict(color="#e41a1c", width=3, dash="dash"),
                               name="transitions (lift)", hoverinfo="skip", legendgroup="path"))

    # --- points coloured by order ----------------------------------------------------------
    idx_points = len(fig.data)
    fig.add_trace(go.Scatter3d(
        x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers",
        marker=dict(size=3.5, color=order, colorscale="Viridis", colorbar=dict(title="order", thickness=12, len=0.5),
                    showscale=True),
        text=hover, hoverinfo="text", name=f"points ({n})", legendgroup="points"))
    fig.add_trace(go.Scatter3d(x=P[:1, 0], y=P[:1, 1], z=P[:1, 2], mode="markers",
                               marker=dict(size=9, color="#000000", symbol="diamond"),
                               name="start", hoverinfo="name", legendgroup="points"))

    # --- direction cones ---------------------------------------------------------------------
    L = spacing * 0.8 * arrow_scale
    cone_kw = dict(sizemode="raw", sizeref=1.0, anchor="tail", showscale=False, hoverinfo="skip")
    try:
        fig.add_trace(go.Cone(x=P[:, 0], y=P[:, 1], z=P[:, 2], u=pts.tool_dir[:, 0] * L, v=pts.tool_dir[:, 1] * L,
                              w=pts.tool_dir[:, 2] * L, colorscale=[[0, "#d62728"], [1, "#d62728"]],
                              name="tool_dir (into surface)", legendgroup="dirs", **cone_kw))
        fig.add_trace(go.Cone(x=P[:, 0], y=P[:, 1], z=P[:, 2], u=pts.normal[:, 0] * L, v=pts.normal[:, 1] * L,
                              w=pts.normal[:, 2] * L, colorscale=[[0, "#2ca02c"], [1, "#2ca02c"]],
                              name="norm_dir (out of surface)", legendgroup="dirs", **cone_kw))
    except ValueError:  # older plotly without sizemode='raw'
        cone_kw.update(sizemode="absolute", sizeref=L)
        fig.add_trace(go.Cone(x=P[:, 0], y=P[:, 1], z=P[:, 2], u=pts.tool_dir[:, 0], v=pts.tool_dir[:, 1],
                              w=pts.tool_dir[:, 2], colorscale=[[0, "#d62728"], [1, "#d62728"]],
                              name="tool_dir (into surface)", legendgroup="dirs", **cone_kw))
        fig.add_trace(go.Cone(x=P[:, 0], y=P[:, 1], z=P[:, 2], u=pts.normal[:, 0], v=pts.normal[:, 1],
                              w=pts.normal[:, 2], colorscale=[[0, "#2ca02c"], [1, "#2ca02c"]],
                              name="norm_dir (out of surface)", legendgroup="dirs", **cone_kw))

    # --- order labels ------------------------------------------------------------------------
    k = labels_every or max(1, int(round(n / 40)))
    li = np.arange(0, n, k)
    fig.add_trace(go.Scatter3d(x=P[li, 0], y=P[li, 1], z=P[li, 2], mode="text", text=[str(i) for i in li],
                               textfont=dict(size=9, color="#222222"), name=f"order labels (every {k})",
                               hoverinfo="skip", visible="legendonly" if n > 4000 else True, legendgroup="points"))

    # --- replay slider -------------------------------------------------------------------------
    if slider and n > 1:
        steps = np.unique(np.linspace(1, n, min(slider_steps, n)).astype(int))
        frames, slider_steps_cfg = [], []
        for kk in steps:
            fx, fy, fz = [], [], []
            for i in range(kk - 1):
                if same[i]:
                    fx += [P[i, 0], P[i + 1, 0], None]
                    fy += [P[i, 1], P[i + 1, 1], None]
                    fz += [P[i, 2], P[i + 1, 2], None]
            frames.append(go.Frame(
                name=str(kk),
                data=[go.Scatter3d(x=fx, y=fy, z=fz, mode="lines", line=dict(color="#444444", width=3)),
                      go.Scatter3d(x=P[:kk, 0], y=P[:kk, 1], z=P[:kk, 2], mode="markers",
                                   marker=dict(size=3.5, color=order[:kk], colorscale="Viridis",
                                               cmin=0, cmax=n - 1, showscale=True,
                                               colorbar=dict(title="order", thickness=12, len=0.5)),
                                   text=hover[:kk], hoverinfo="text")],
                traces=[idx_points - 2, idx_points]))
            slider_steps_cfg.append(dict(method="animate", label=str(kk),
                                         args=[[str(kk)], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                                               transition=dict(duration=0))]))
        fig.frames = frames
        fig.update_layout(sliders=[dict(active=len(steps) - 1, currentvalue=dict(prefix="points shown: "),
                                        pad=dict(t=30), steps=slider_steps_cfg)],
                          updatemenus=[dict(type="buttons", showactive=False, x=0.0, y=0.0, xanchor="left",
                                            buttons=[dict(label="▶ replay", method="animate",
                                                          args=[None, dict(frame=dict(duration=120, redraw=True),
                                                                           fromcurrent=False, mode="immediate",
                                                                           transition=dict(duration=0))])])])

    s = result.stats
    fig.update_layout(
        title=title or (f"{result.params['source']} | pattern={result.params['pattern']} order={result.params['order']} | "
                        f"{n} points, {s['n_strokes']} strokes, path {s['path_length']:.2f} m | "
                        f"spacing {spacing} m, row {result.params['row_spacing']} m"),
        scene=dict(aspectmode="data", xaxis_title="x [m]", yaxis_title="y [m]", zaxis_title="z [m]"),
        legend=dict(itemsizing="constant", groupclick="toggleitem"),
        margin=dict(l=0, r=0, t=50, b=0), height=850)
    fig.write_html(path, include_plotlyjs=True if embed_js else "cdn", auto_open=open_browser)
    return path


# ============================================================================
# shapes.py
# Procedural test meshes (units: metres, Z-up).
#
# Every shape is a watertight solid built as a surface of revolution from a closed
# (r, z) profile, except the rectangular tray which is a boolean of boxes and the
# "scan_bowl" which imitates a noisy, non-watertight scan of the dog bowl.
# ============================================================================

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


def shapes_main(argv=None):
    ap = argparse.ArgumentParser(description="Generate the test meshes (metres, Z-up).")
    ap.add_argument("--out", default="test_meshes")
    ap.add_argument("--shapes", nargs="*", default=None, help=f"subset of {list(SHAPES)}")
    ap.add_argument("--formats", nargs="*", default=["ply", "stl"])
    a = ap.parse_args(argv)
    generate_all(a.out, a.shapes, tuple(a.formats))




# ============================================================================
# cli.py
# Command line interface: ``mesh2points INPUT [options]``.
# ============================================================================

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
        paths["html"] = plotly_html(res, os.path.join(a.out_dir, f"{name}_viz.html"), arrow_scale=a.arrow_scale,
                                    slider=not a.no_slider, embed_js=not a.cdn, open_browser=a.open)
    if a.png:
        paths["png"] = preview_png(res, os.path.join(a.out_dir, f"{name}_preview.png"))
    if not a.quiet:
        print("[files]")
        for k, v in paths.items():
            print(f"  {k:8s} {v}")
    return 0




# ============================================================================
# entry point
# ============================================================================
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "shapes":
        sys.exit(shapes_main(sys.argv[2:]) or 0)
    sys.exit(main())
