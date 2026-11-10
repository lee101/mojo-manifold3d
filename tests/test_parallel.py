"""Serial and parallel signed-distance output must be identical.

`mm_signed_distance` fans the grid out over `max.algorithm.parallelize` above
a measured size threshold and stays serial below it. The worker split is by
row range, so every grid point is computed by the same code in every
configuration. These tests pin that: the same input is evaluated at 1, 2, 4
and the worker cap, and the whole grid must match bit for bit.

`mm_signed_distance_data` is the test entry point. It bypasses the threshold
so the parallel path can be exercised on small grids, still caps the worker
count, and is the same `signed_distance_grid` the exported function calls.
"""

import numpy as np
import pytest

import mojomanifold3d as mm
from mojomanifold3d import core
from mojomanifold3d._lib import addr, lib

WORKER_CAP = 4
WORKER_COUNTS = [1, 2, 4]
# Below the threshold, just above it, and an odd shape with a ragged tail.
# 101 x 101 x 99 = 1,009,899 points, just past the 1,000,000 threshold.
SHAPES = [
    (24, 23, 22),      # 12,144 points: serial path of mm_signed_distance
    (101, 101, 99),    # 1,009,899 points: just over the threshold
    (61, 59, 57),      # 205,143 points: odd sizes, every row has a tail
]


def _solid():
    return mm.Manifold.sphere(1, 8)


def _grid(shape):
    nx, ny, nz = shape
    return (-1.5, -1.5, -1.5), 3.0 / (nx - 1)


def _run(shape, workers):
    solid = _solid()
    origin, spacing = _grid(shape)
    result = np.empty(shape, dtype=np.float64)
    lib().mm_signed_distance_data(
        addr(solid._vertices),
        addr(solid._faces),
        len(solid._faces),
        float(origin[0]),
        float(origin[1]),
        float(origin[2]),
        float(spacing),
        shape[0],
        shape[1],
        shape[2],
        addr(result, writable=True),
        workers,
    )
    return result


def _run_exported(shape, workers, monkeypatch):
    solid = _solid()
    origin, spacing = _grid(shape)
    monkeypatch.setattr(core, "_SIGNED_DISTANCE_WORKERS", workers)
    return core._field(solid, origin, spacing, shape)


def test_test_entry_point_signature_is_declared():
    # A signature mismatch here would surface as a confusing ctypes error.
    import ctypes

    function = lib().mm_signed_distance_data
    assert function.argtypes is not None
    assert function.restype is ctypes.c_int64


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("workers", WORKER_COUNTS)
def test_parallel_matches_serial_bitwise(shape, workers):
    reference = _run(shape, 1)
    parallel = _run(shape, workers)
    assert np.isfinite(reference).all()
    assert np.array_equal(reference, parallel)


@pytest.mark.parametrize("shape", SHAPES)
def test_worker_cap_does_not_change_output(shape):
    # Requesting more than the cap must not change the value, only clamp it.
    reference = _run(shape, 1)
    assert np.array_equal(reference, _run(shape, WORKER_CAP))
    assert np.array_equal(reference, _run(shape, 64))


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("workers", [1, 2, 4])
def test_exported_kernel_matches_serial_bitwise(shape, workers, monkeypatch):
    # The real entry point, threshold included: below it every worker count
    # takes the serial path, above it the parallel path must agree.
    reference = _run_exported(shape, 1, monkeypatch)
    assert np.array_equal(reference, _run_exported(shape, workers, monkeypatch))


def test_boolean_uses_parallel_field_and_matches_serial(monkeypatch):
    # End to end: the same boolean with the field pinned serial and pinned at
    # the cap must produce identical vertex and triangle buffers.
    first = mm.Manifold.cube((2, 2, 2), True)
    second = mm.Manifold.cube((2, 2, 2), True).translate((0.7, 0.2, 0))
    monkeypatch.setattr(core, "_CSG_RESOLUTION", 48)

    monkeypatch.setattr(core, "_SIGNED_DISTANCE_WORKERS", 1)
    serial = first + second
    monkeypatch.setattr(core, "_SIGNED_DISTANCE_WORKERS", WORKER_CAP)
    parallel = first + second

    assert np.array_equal(serial._vertices, parallel._vertices)
    assert np.array_equal(serial._faces, parallel._faces)
    assert serial.volume() == parallel.volume()


def _pair(first, second, shape, origin, spacing, workers):
    result = np.empty(shape, dtype=np.float64)
    other = np.empty(shape, dtype=np.float64)
    lib().mm_signed_distance_pair(
        addr(first._vertices),
        addr(first._faces),
        len(first._faces),
        addr(second._vertices),
        addr(second._faces),
        len(second._faces),
        float(origin[0]),
        float(origin[1]),
        float(origin[2]),
        float(spacing),
        shape[0],
        shape[1],
        shape[2],
        addr(result, writable=True),
        addr(other, writable=True),
        workers,
    )
    return result, other


@pytest.mark.parametrize("workers", WORKER_COUNTS)
def test_pair_form_matches_two_single_meshes(workers):
    # One launch evaluates both fields; the values must be the ones the
    # single-mesh kernel produces, for both distinct and shared geometry.
    first = mm.Manifold.sphere(1, 8)
    second = mm.Manifold.cube((1.2, 1.2, 1.2), True)
    shape = (61, 59, 57)
    # An arbitrary origin: no sample lands on a face of either mesh.
    spacing = 0.37
    origin = (-1.5, -1.5, -1.5)

    alone = core._field(first, origin, spacing, shape).reshape(shape)
    other_alone = core._field(second, origin, spacing, shape).reshape(shape)
    for second_mesh in (second, first):
        pair, other = _pair(first, second_mesh, shape, origin, spacing, workers)
        assert np.array_equal(pair, alone)
        expected = alone if second_mesh is first else other_alone
        assert np.array_equal(other, expected)
