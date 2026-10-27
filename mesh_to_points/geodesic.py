"""Geodesic distance fields on triangle meshes (heat method with a Dijkstra fallback)."""
from __future__ import annotations

import warnings

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

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
