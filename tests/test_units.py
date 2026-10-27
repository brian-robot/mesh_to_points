"""Unit tests for the geometry building blocks."""
import numpy as np
import trimesh

from mesh_to_points.contours import iso_contours
from mesh_to_points.geodesic import dijkstra_distance, geodesic_distance
from mesh_to_points.geom import chain_segments, resample_polyline, signed_area_xy
from mesh_to_points.mesh_io import fix_orientation, refine_to_edge_length
from mesh_to_points.order import improve_path, nearest_neighbor_order, path_length, strokes_from_jumps


def test_chain_segments_open_and_closed():
    pairs = np.array([[0, 1], [1, 2], [2, 3], [10, 11], [11, 12], [12, 10]])
    chains = chain_segments(pairs)
    kinds = sorted((len(nodes), closed) for _, nodes, closed in chains)
    assert kinds == [(3, True), (4, False)]


def test_resample_polyline():
    sq = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
    s, idx, t = resample_polyline(sq, 0.25, closed=True)
    assert len(s) == 16
    d = np.linalg.norm(np.diff(np.vstack([s, s[:1]]), axis=0), axis=1)
    assert np.allclose(d, 0.25)
    s, _, _ = resample_polyline(sq[:2], 0.3, closed=False)
    assert len(s) == 4 and np.allclose(s[0], sq[0]) and np.allclose(s[-1], sq[1])
    assert signed_area_xy(sq) > 0 and signed_area_xy(sq[::-1]) < 0


def test_refine_is_conforming_and_preserves_geometry():
    m = trimesh.creation.box(extents=(0.2, 0.1, 0.05))
    r = refine_to_edge_length(m, 0.02)
    assert r.is_watertight
    assert r.edges_unique_length.max() <= 0.02 + 1e-12
    assert abs(r.volume - m.volume) < 1e-12
    assert abs(r.area - m.area) < 1e-12
    assert np.allclose(r.face_normals @ np.array([1, 1, 1]), r.face_normals @ np.array([1, 1, 1]))


def test_fix_orientation_closed_and_open():
    m = trimesh.creation.icosphere(subdivisions=3, radius=0.1)
    m = trimesh.Trimesh(m.vertices, m.faces[:, ::-1], process=False)     # inside out
    assert m.volume < 0
    fix_orientation(m)
    assert m.volume > 0
    assert np.all(np.einsum("ij,ij->i", m.face_normals, m.triangles_center) > 0)   # outward normals
    m2 = trimesh.creation.icosphere(subdivisions=3, radius=0.1)
    fix_orientation(m2, flip=True)
    assert m2.volume < 0 and np.all(np.einsum("ij,ij->i", m2.face_normals, m2.triangles_center) < 0)
    # open sheet: a bowl-like cap with normals pointing inwards -> should be flipped to face up/out
    cap = trimesh.creation.icosphere(subdivisions=3, radius=0.1)
    keep = cap.face_normals[:, 2] < -0.3
    cap = cap.submesh([np.where(keep)[0]], append=True)     # normals point down / outwards
    assert (cap.area_faces * cap.face_normals[:, 2]).sum() < 0
    fix_orientation(cap)                                     # a sheet: normals must face the open side (up)
    assert (cap.area_faces * cap.face_normals[:, 2]).sum() > 0
    flipped = trimesh.Trimesh(cap.vertices, cap.faces[:, ::-1], process=False)
    fix_orientation(flipped)
    assert (flipped.area_faces * flipped.face_normals[:, 2]).sum() > 0


def test_geodesic_on_plane():
    box = trimesh.creation.box(extents=(1.0, 1.0, 0.01))
    top = box.faces[box.face_normals[:, 2] > 0.9]
    plane = trimesh.Trimesh(box.vertices, top, process=True)
    plane.remove_unreferenced_vertices()
    plane = refine_to_edge_length(plane, 0.05)
    V, F = plane.vertices, plane.faces
    src = np.where(V[:, 0] < -0.499)[0]
    true = V[:, 0] + 0.5
    for d in (geodesic_distance(V, F, src), dijkstra_distance(V, F, src)):
        assert np.abs(d - true).max() < 0.03
    cs = iso_contours(V, F, true, 0.3)
    assert len(cs) == 1
    pts, faces, closed = cs[0]
    assert not closed
    assert np.allclose(pts[:, 0], -0.2, atol=1e-9)
    assert abs(np.ptp(pts[:, 1]) - 1.0) < 1e-9
    assert len(faces) == len(pts) - 1


def test_orders():
    rng = np.random.default_rng(1)
    p = rng.random((300, 3))
    o = nearest_neighbor_order(p)
    assert sorted(o.tolist()) == list(range(300))
    o2 = improve_path(p, o, time_limit=10)
    assert sorted(o2.tolist()) == list(range(300))
    assert path_length(p, o2) < path_length(p, o)
    s = strokes_from_jumps(np.array([[0, 0, 0], [0.01, 0, 0], [0.5, 0, 0], [0.51, 0, 0]]), 0.1)
    assert s.tolist() == [0, 0, 1, 1]
