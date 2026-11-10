"""Tail coverage for the SIMD kernels at every boundary length.

Each vectorised kernel is driven at 0, 1, W-1, W, W+1 and a large odd length
so the scalar remainder loop runs, and the result is compared against a numpy
reference computed from the same inputs. A length of 0 is driven straight
through the Python wrapper, because `addr()` rejects empty buffers.
"""

import numpy as np
import pytest

import mojomanifold3d as mm
from mojomanifold3d._lib import addr, lib

W = 4  # simd_width_of[DType.float64]() on the pinned toolchain
LENGTHS = [0, 1, W - 1, W, W + 1, 4 * W + 1, 10_001]


def _addr(array, *, writable=False):
    """1-element stand-in for an empty buffer, so length 0 still reaches Mojo."""
    if array.size == 0:
        return addr(np.zeros(1, dtype=array.dtype))
    return addr(array, writable=writable)


def _fields(count, seed):
    rng = np.random.default_rng(seed)
    first = rng.normal(size=count) * 3.0
    second = rng.normal(size=count) * 3.0
    if count:
        # Force both signs and a few exact ties so min/max matter.
        first[::3] = second[::3]
        second[::5] = -first[::5]
    return first, second


def _combine_reference(first, second, operation):
    if operation == 0:
        return np.maximum(first, second)
    if operation == 1:
        return np.minimum(first, -second)
    return np.minimum(first, second)


@pytest.mark.parametrize("length", LENGTHS)
@pytest.mark.parametrize("operation", [0, 1, 2])
def test_combine_fields_tail(length, operation):
    first, second = _fields(length, length + operation)
    result = np.full(length, np.nan, dtype=np.float64)
    lib().mm_combine_fields(
        _addr(first), _addr(second), _addr(result, writable=True), length,
        operation,
    )
    assert result == pytest.approx(_combine_reference(first, second, operation))


@pytest.mark.parametrize("length", LENGTHS)
def test_combine_fields_writes_only_count(length):
    first, second = _fields(length, 41)
    result = np.full(length + 2 * W, 7.0)
    lib().mm_combine_fields(
        _addr(first), _addr(second), _addr(result, writable=True), length, 0
    )
    assert np.all(result[length:] == 7.0)
    if length:
        assert result[:length] == pytest.approx(
            _combine_reference(first, second, 0)
        )


@pytest.mark.parametrize("length", LENGTHS)
def test_transform_vertices_tail(length):
    rng = np.random.default_rng(length)
    matrix = rng.normal(size=(3, 4))
    source = np.ascontiguousarray(rng.normal(size=(length, 3)))
    destination = np.zeros((length, 3), dtype=np.float64)
    if length == 0:
        empty = mm.Manifold()
        empty._apply_matrix(matrix)
        assert empty.is_empty()
        return
    lib().mm_transform(addr(source), addr(destination), addr(matrix), length)
    expected = source @ matrix[:, :3].T + matrix[:, 3]
    assert destination == pytest.approx(expected)


def _measure_reference(vertices, faces):
    a = vertices[faces[:, 0]]
    b = vertices[faces[:, 1]]
    c = vertices[faces[:, 2]]
    signed_volume = np.einsum("ij,ij->i", a, np.cross(b, c)).sum()
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1).sum() * 0.5
    return (abs(float(signed_volume)) / 6.0, float(area))


@pytest.mark.parametrize("length", LENGTHS)
def test_measure_tail(length):
    if length == 0:
        assert mm.Manifold().volume() == 0.0
        assert mm.Manifold().surface_area() == 0.0
        return
    rng = np.random.default_rng(100 + length)
    vertex_count = length * 3
    vertices = np.ascontiguousarray(rng.normal(size=(vertex_count, 3)))
    faces = rng.integers(0, vertex_count, size=(length, 3)).astype(np.int64)
    result = np.zeros(2, dtype=np.float64)
    lib().mm_measure(addr(vertices), addr(faces), len(faces), addr(result))
    assert result == pytest.approx(_measure_reference(vertices, faces))


@pytest.mark.parametrize("length", LENGTHS)
@pytest.mark.parametrize("negate", [0, 1])
def test_pairwise_sum_tail(length, negate):
    rng = np.random.default_rng(200 + length + negate)
    second = np.ascontiguousarray(rng.normal(size=(3, 3)))
    if length == 0:
        empty = mm.Manifold()
        assert empty.minkowski_sum(empty).is_empty()
        return
    first = np.ascontiguousarray(rng.normal(size=(length, 3)))
    destination = np.zeros((length * len(second), 3))
    lib().mm_pairwise_sum(
        addr(first), length, addr(second), len(second),
        addr(destination), negate,
    )
    sign = -1.0 if negate else 1.0
    expected = (first[:, None, :] + sign * second[None, :, :]).reshape(-1, 3)
    assert destination == pytest.approx(expected)
