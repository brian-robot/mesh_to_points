"""Point ordering: greedy nearest neighbour, 2-opt/Or-opt improvement, stroke splitting."""
from __future__ import annotations

import time

import numpy as np
from scipy.spatial import cKDTree


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
