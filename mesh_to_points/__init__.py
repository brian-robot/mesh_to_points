"""mesh_to_points: region-aware, ordered wiping point generation from container meshes."""
__version__ = "0.1.0"

from .segment import REGION_NAMES, DEFAULT_INCLUDE  # noqa: F401,E402


def __getattr__(name):  # lazy imports keep `import mesh_to_points` light
    if name in ("generate", "Result", "PointSet"):
        from . import pipeline
        return getattr(pipeline, name)
    raise AttributeError(name)
