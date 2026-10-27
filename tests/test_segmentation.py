import numpy as np
import pytest

from conftest import mesh_path
from mesh_to_points import generate

# expected region -> number of connected components
EXPECTED = {
    "dog_bowl": {"rim": 1, "inner_wall": 1, "inner_floor": 1, "outer_wall": 1, "bottom": 1},
    "vase": {"rim": 1, "inner_wall": 1, "inner_floor": 1, "outer_wall": 1, "bottom": 1},
    "bucket": {"rim": 1, "inner_wall": 1, "inner_floor": 1, "outer_wall": 1, "bottom": 1},
    "terraces": {"rim": 1, "inner_wall": 3, "tread": 2, "inner_floor": 1, "outer_wall": 1, "bottom": 1},
    "hemisphere_bowl": {"rim": 1, "inner_wall": 1, "inner_floor": 1, "outer_wall": 1, "bottom": 1},
    "plate": {"rim": 1, "tread": 1, "inner_floor": 1, "outer_wall": 1, "bottom": 1},
    "rect_tray": {"rim": 1, "inner_wall": 2, "inner_floor": 2, "outer_wall": 1, "bottom": 1},
    "vase_necked": {"rim": 1, "inner_wall": 1, "inner_floor": 1, "outer_wall": 1, "bottom": 1},
}


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_regions_and_components(mesh_dir, name):
    res = generate(mesh_path(mesh_dir, name), verbose=False)
    rows = {r["region"]: r for r in res.seg.summary(res.mesh)}
    for region, n_comp in EXPECTED[name].items():
        assert region in rows, f"{name}: region {region} missing (have {sorted(rows)})"
        assert rows[region]["components"] == n_comp, f"{name}: {region} components {rows[region]['components']} != {n_comp}"
    assert set(rows) == set(EXPECTED[name]), f"{name}: unexpected regions {set(rows) - set(EXPECTED[name])}"


def test_terrace_levels(mesh_dir):
    res = generate(mesh_path(mesh_dir, "terraces"), verbose=False)
    rows = {r["region"]: r for r in res.seg.summary(res.mesh)}
    assert rows["rim"]["levels"] == [0]
    assert rows["tread"]["levels"] == [1, 2]
    assert rows["inner_floor"]["levels"] == [3]
    assert rows["inner_wall"]["levels"] == [1, 2, 3]


def test_reachability_necked_vase(mesh_dir):
    res = generate(mesh_path(mesh_dir, "vase_necked"), verbose=False)
    rows = {r["region"]: r for r in res.seg.summary(res.mesh)}
    frac = rows["inner_wall"]["reachable_area"] / rows["inner_wall"]["area"]
    assert 0.05 < frac < 0.6, f"inner wall reachable fraction {frac}"
    assert rows["rim"]["reachable_area"] == pytest.approx(rows["rim"]["area"])
    assert rows["bottom"]["reachable_area"] == 0.0


def test_bottom_never_reachable(mesh_dir):
    res = generate(mesh_path(mesh_dir, "bucket"), verbose=False)
    rows = {r["region"]: r for r in res.seg.summary(res.mesh)}
    assert rows["bottom"]["reachable_area"] == 0.0
    assert rows["outer_wall"]["reachable_area"] == 0.0   # tapered bucket: the outer wall is under the rim


def test_scan_bowl_segments(mesh_dir):
    res = generate(mesh_path(mesh_dir, "scan_bowl"), verbose=False)
    rows = {r["region"]: r for r in res.seg.summary(res.mesh)}
    for region in ("rim", "inner_wall", "inner_floor"):
        assert region in rows
    assert rows["inner_floor"]["area"] > 0.008
    assert len(res.points) > 300
