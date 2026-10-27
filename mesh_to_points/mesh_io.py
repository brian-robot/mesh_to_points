"""Mesh loading, cleaning, orientation handling and conforming refinement."""
from __future__ import annotations

import os
import warnings

import numpy as np
import trimesh

from .geom import rotation_to_z, unit

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
