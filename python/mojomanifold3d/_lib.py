"""Load the Mojo shared library and declare its C ABI."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJOMANIFOLD3D_LIB") or os.path.join(
    ROOT, "dist", "libmojo-manifold3d.so"
)

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mm_signed_distance": ([I, I, I, F, F, F, F, I, I, I, I], None),
    "mm_signed_distance_gpu": ([I, I, I, I, F, F, F, F, I, I, I, I], I),
    "mm_signed_distance_pair_gpu": (
        [I, I, I, I, I, I, I, I, F, F, F, F, I, I, I, I, I],
        I,
    ),
    "mm_combine_fields": ([I, I, I, I, I], None),
    "mm_march_count": ([I, I, I, I], I),
    "mm_march_emit": ([I, I, I, I, I, I, F, F, F, F], I),
    "mm_transform": ([I, I, I, I], None),
    "mm_pairwise_sum": ([I, I, I, I, I, I], None),
    "mm_measure": ([I, I, I, I], None),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    source = os.path.join(ROOT, "src", "kernels.mojo")
    if not force and os.path.exists(LIB):
        if os.path.getmtime(LIB) >= os.path.getmtime(source):
            return LIB
    mojo = shutil.which("mojo")
    if mojo is None:
        raise BuildError("mojo not found; run inside `pixi run`")
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    process = subprocess.run(
        [mojo, "build", "--emit", "shared-lib", source, "-o", LIB],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if process.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((process.stderr or process.stdout).strip()[:4000])
    return LIB


_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def f64(value, *, copy: bool = False) -> np.ndarray:
    if copy:
        return np.array(value, dtype=np.float64, order="C", copy=True)
    return np.ascontiguousarray(value, dtype=np.float64)


def i64(value, *, copy: bool = False) -> np.ndarray:
    if copy:
        return np.array(value, dtype=np.int64, order="C", copy=True)
    return np.ascontiguousarray(value, dtype=np.int64)


def addr(array: np.ndarray, *, writable: bool = False) -> int:
    if not isinstance(array, np.ndarray):
        raise TypeError("FFI buffers must be numpy arrays")
    if array.dtype not in (np.dtype(np.float64), np.dtype(np.int64)):
        raise TypeError("FFI buffers must have dtype float64 or int64")
    if not array.flags.c_contiguous:
        raise TypeError("FFI arrays must be C-contiguous")
    if array.size == 0:
        raise ValueError("FFI buffers must not be empty")
    if writable and not array.flags.writeable:
        raise TypeError("FFI output buffers must be writable")
    address = int(array.ctypes.data)
    if address == 0:
        raise ValueError("FFI buffers must have a non-null address")
    if address % array.dtype.alignment:
        raise ValueError("FFI buffers must be naturally aligned")
    return address
