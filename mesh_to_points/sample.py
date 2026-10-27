"""Point pattern generators.

spiral   contour-parallel: geodesic iso-distance rings from the rim's outer edge,
         ordered outside-in / top-down (optionally blended into one continuous helix)
zigzag   boustrophedon: scan lines on wide flat regions, serpentine rings on walls,
         annular tops and narrow regions; regions chained rim -> wall -> ... -> floor
poisson  blue-noise (evenly spaced random) samples; ordered afterwards (nearest/tsp)

All functions work in the internal Z-up frame on the cleaned/refined mesh.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import unary_union

from .contours import iso_contours
from .geodesic import geodesic_distance
from .geom import boundary_edges, chain_segments, polyline_length, resample_polyline, signed_area_xy
from .normals import smooth_point_normals
from .segment import BOTTOM, INNER_FLOOR, OUTER_WALL, REGION_NAMES, RIM, TREAD, Segmentation

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
