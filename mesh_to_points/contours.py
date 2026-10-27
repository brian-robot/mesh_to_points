"""Iso-contour extraction of a per-vertex scalar field on a triangle mesh."""
from __future__ import annotations

import numpy as np

from .geom import chain_segments


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
