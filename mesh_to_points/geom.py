"""Small geometry helpers shared by the segmentation and sampling code."""
from __future__ import annotations

import numpy as np


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
