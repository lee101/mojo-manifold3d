"""Manifold triangle-mesh CSG with Mojo compute kernels."""

from .core import Error, Manifold, Mesh, OpType, set_csg_resolution

__version__ = "0.1.0"

__all__ = [
    "Error",
    "Manifold",
    "Mesh",
    "OpType",
    "set_csg_resolution",
]
