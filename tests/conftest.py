import os

import pytest

from mesh_to_points import shapes

ALL_SHAPES = list(shapes.SHAPES)


@pytest.fixture(scope="session")
def mesh_dir(tmp_path_factory):
    """Generate the test meshes once per session (PLY only)."""
    d = tmp_path_factory.mktemp("meshes")
    shapes.generate_all(str(d), formats=("ply",), verbose=False)
    return str(d)


def mesh_path(mesh_dir, name, ext="ply"):
    return os.path.join(mesh_dir, f"{name}.{ext}")
