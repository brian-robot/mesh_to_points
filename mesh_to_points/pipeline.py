"""End-to-end API: mesh file -> ordered, region-aware wiping point set."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .mesh_io import Frame, clean_mesh, fix_orientation, load_mesh, parse_up, refine_to_edge_length
from .order import improve_path, nearest_neighbor_order, path_length, strokes_from_jumps
from .sample import poisson_pattern, spiral_pattern, strokes_to_arrays, zigzag_pattern
from .segment import DEFAULT_INCLUDE, REGION_NAMES, Segmentation, segment

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
            from .segment import REGION_COLORS
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
