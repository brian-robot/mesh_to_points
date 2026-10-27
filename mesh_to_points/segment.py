"""Face segmentation of a container mesh into semantic regions.

Regions (face labels):
  rim            top-facing surface connected to the exterior (the lip of the container)
  inner_wall     side-facing surface that looks into a cavity
  tread          top-facing surface inside the cavity above the floor (terrace steps)
  inner_floor    lowest top-facing surface inside the cavity
  inner_overhang down-facing surface inside the cavity (e.g. under a vase shoulder)
  outer_wall     side-facing surface looking outwards
  bottom         down-facing exterior surface (the base)

The mesh is expected in the internal Z-up frame (see mesh_io.Frame).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import trimesh

from .geom import boundary_edges, chain_segments, signed_area_xy

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
