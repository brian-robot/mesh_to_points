"""Crease-aware smooth normals for sample points lying on mesh faces."""
from __future__ import annotations

import numpy as np
import trimesh

from .geom import unit


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
