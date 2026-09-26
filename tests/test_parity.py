import numpy as np
import pytest

import manifold3d as upstream
import mojomanifold3d as mm
import mojomanifold3d.core as core
from mojomanifold3d._lib import addr, lib


@pytest.fixture(scope="module", autouse=True)
def csg_resolution():
    mm.set_csg_resolution(28)
    yield
    mm.set_csg_resolution(40)


def assert_oriented_manifold(solid):
    assert solid.status() is mm.Error.NoError
    if solid.is_empty():
        return
    faces = solid.to_mesh().tri_verts.astype(np.int64)
    edges = np.concatenate(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])
    )
    undirected = np.sort(edges, axis=1)
    _, inverse, counts = np.unique(
        undirected, axis=0, return_inverse=True, return_counts=True
    )
    assert np.all(counts == 2)
    signs = np.where(edges[:, 0] < edges[:, 1], 1, -1)
    assert np.all(np.bincount(inverse, weights=signs) == 0)


def test_enum_values_match_upstream():
    assert [(item.name, item.value) for item in mm.OpType] == [
        (item.name, item.value) for item in upstream.OpType
    ]
    assert [(item.name, item.value) for item in mm.Error] == [
        (item.name, item.value) for item in upstream.Error
    ]


def test_mesh_roundtrip_contract_and_dtypes():
    original = mm.Manifold.cube()
    mesh = original.to_mesh()
    assert mesh.vert_properties.dtype == np.float32
    assert mesh.tri_verts.dtype == np.int32
    rebuilt = mm.Manifold(mesh)
    assert rebuilt.status() is mm.Error.NoError
    assert rebuilt.volume() == pytest.approx(1.0)
    assert rebuilt.num_vert() == 8
    assert rebuilt.num_tri() == 12
    assert rebuilt.num_prop_vert() == 8
    assert rebuilt.num_edge() == 18
    assert rebuilt.num_prop() == 0


def test_mesh_rejects_silent_narrowing():
    with pytest.raises(OverflowError):
        mm.Mesh(np.zeros((3, 3)), np.array([[0, 1, 2**32]], dtype=np.int64))
    with pytest.raises(OverflowError):
        mm.Mesh(
            np.array([[0, 0, 0], [1e40, 0, 0], [0, 1, 0]]),
            np.array([[0, 1, 2]]),
        )


def test_open_mesh_reports_not_manifold():
    mesh = mm.Mesh(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32),
        np.array([[0, 1, 2]], np.uint32),
    )
    solid = mm.Manifold(mesh)
    assert solid.status() is mm.Error.NotManifold
    assert solid.is_empty()


def test_nonfinite_mesh_reports_matching_error():
    mesh = mm.Mesh(
        np.array([[0, 0, 0], [1, 0, 0], [0, np.nan, 0]], np.float32),
        np.array([[0, 1, 2]], np.uint32),
    )
    assert mm.Manifold(mesh).status() is mm.Error.NonFiniteVertex


@pytest.mark.parametrize("center", [False, True])
def test_cube_matches_upstream(center):
    ours = mm.Manifold.cube((2.0, 3.0, 4.0), center=center)
    theirs = upstream.Manifold.cube((2.0, 3.0, 4.0), center=center)
    assert ours.bounding_box() == pytest.approx(theirs.bounding_box())
    assert ours.volume() == pytest.approx(theirs.volume())
    assert ours.surface_area() == pytest.approx(theirs.surface_area())
    assert (ours.num_vert(), ours.num_tri(), ours.genus()) == (
        theirs.num_vert(),
        theirs.num_tri(),
        theirs.genus(),
    )


def test_tetrahedron_matches_upstream():
    ours = mm.Manifold.tetrahedron()
    theirs = upstream.Manifold.tetrahedron()
    assert ours.bounding_box() == pytest.approx(theirs.bounding_box())
    assert ours.volume() == pytest.approx(theirs.volume())
    assert ours.surface_area() == pytest.approx(theirs.surface_area())
    assert (ours.num_vert(), ours.num_tri()) == (
        theirs.num_vert(),
        theirs.num_tri(),
    )


@pytest.mark.parametrize("center", [False, True])
def test_cylinder_matches_upstream(center):
    ours = mm.Manifold.cylinder(2.0, 1.0, 0.6, 16, center)
    theirs = upstream.Manifold.cylinder(2.0, 1.0, 0.6, 16, center)
    assert ours.bounding_box() == pytest.approx(theirs.bounding_box(), abs=1e-7)
    assert ours.volume() == pytest.approx(theirs.volume(), rel=2e-7)
    assert ours.surface_area() == pytest.approx(theirs.surface_area(), rel=2e-7)
    assert (ours.num_vert(), ours.num_tri()) == (
        theirs.num_vert(),
        theirs.num_tri(),
    )


def test_sphere_octahedral_refinement_matches_upstream():
    ours = mm.Manifold.sphere(1.0, 16)
    theirs = upstream.Manifold.sphere(1.0, 16)
    assert ours.bounding_box() == pytest.approx(theirs.bounding_box())
    assert (ours.num_vert(), ours.num_tri()) == (
        theirs.num_vert(),
        theirs.num_tri(),
    )
    assert ours.volume() == pytest.approx(theirs.volume(), rel=0.004)
    assert ours.surface_area() == pytest.approx(theirs.surface_area(), rel=0.004)


def test_transform_chain_matches_upstream_bounds_and_measure():
    ours = (
        mm.Manifold.cube((1.0, 2.0, 3.0), True)
        .rotate((20.0, -15.0, 35.0))
        .translate((4.0, -2.0, 1.0))
    )
    theirs = (
        upstream.Manifold.cube((1.0, 2.0, 3.0), True)
        .rotate((20.0, -15.0, 35.0))
        .translate((4.0, -2.0, 1.0))
    )
    assert ours.bounding_box() == pytest.approx(theirs.bounding_box(), abs=2e-7)
    assert ours.volume() == pytest.approx(theirs.volume())
    assert ours.surface_area() == pytest.approx(theirs.surface_area())


def test_scale_and_affine_transform_match_upstream():
    matrix = np.array(
        [[0.8, -0.2, 0.1, 3], [0.3, 1.1, 0, -2], [0, 0.2, 0.9, 1]]
    )
    ours = mm.Manifold.cube(center=True).scale((2, 3, 4)).transform(matrix)
    theirs = upstream.Manifold.cube(center=True).scale((2, 3, 4)).transform(matrix)
    assert ours.bounding_box() == pytest.approx(theirs.bounding_box())
    assert ours.volume() == pytest.approx(theirs.volume())


def test_warp_variants_apply_vertex_functions():
    solid = mm.Manifold.cube()
    scalar = solid.warp(lambda p: (p[0] + 2, p[1] - 1, p[2] * 3))
    batch = solid.warp_batch(
        lambda points: points * np.array([1, 2, 1]) + np.array([2, -1, 0])
    )
    assert scalar.bounding_box() == pytest.approx((2, -1, 0, 3, 0, 3))
    assert batch.bounding_box() == pytest.approx((2, -1, 0, 3, 1, 1))
    assert_oriented_manifold(scalar)
    assert_oriented_manifold(batch)


def test_mirror_preserves_oriented_manifold_and_measure():
    original = mm.Manifold.cylinder(2, 1, 0.5, 12)
    mirrored = original.mirror((1, 2, 0))
    assert_oriented_manifold(mirrored)
    assert mirrored.volume() == pytest.approx(original.volume())
    assert mirrored.surface_area() == pytest.approx(original.surface_area())


def test_hull_points_matches_upstream_on_random_cloud():
    points = np.random.default_rng(4).normal(size=(200, 3))
    ours = mm.Manifold.hull_points(points)
    theirs = upstream.Manifold.hull_points(points)
    assert_oriented_manifold(ours)
    assert ours.volume() == pytest.approx(theirs.volume(), rel=2e-7)
    assert ours.surface_area() == pytest.approx(theirs.surface_area(), rel=2e-7)
    assert ours.bounding_box() == pytest.approx(theirs.bounding_box(), abs=2e-7)


def test_degenerate_hull_is_empty_like_upstream():
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    assert mm.Manifold.hull_points(points).is_empty()
    # Upstream 3.5.2 currently returns a zero-volume tetrahedral shell here,
    # despite its documented empty-result contract for coplanar points.
    assert upstream.Manifold.hull_points(points).volume() == 0.0


def test_batch_hull_matches_upstream():
    ours_inputs = [
        mm.Manifold.cube().translate((-2, 0, 0)),
        mm.Manifold.tetrahedron().translate((2, 0, 0)),
    ]
    upstream_inputs = [
        upstream.Manifold.cube().translate((-2, 0, 0)),
        upstream.Manifold.tetrahedron().translate((2, 0, 0)),
    ]
    ours = mm.Manifold.batch_hull(ours_inputs)
    theirs = upstream.Manifold.batch_hull(upstream_inputs)
    assert ours.volume() == pytest.approx(theirs.volume(), rel=2e-7)
    assert ours.bounding_box() == pytest.approx(theirs.bounding_box())


def test_instance_hull_and_compose_are_covered():
    parts = [
        mm.Manifold.cube().translate((-0.2, 0, 0)),
        mm.Manifold.cube().translate((0.2, 0, 0)),
    ]
    composed = mm.Manifold.compose(parts)
    assert_oriented_manifold(composed)
    assert composed.volume() == pytest.approx(1.4, rel=0.03)
    hull = composed.hull()
    assert_oriented_manifold(hull)
    assert hull.volume() >= composed.volume()


def test_convex_minkowski_sum_matches_upstream():
    ours = mm.Manifold.cube((2, 3, 4)).minkowski_sum(
        mm.Manifold.cube((1, 2, 0.5))
    )
    theirs = upstream.Manifold.cube((2, 3, 4)).minkowski_sum(
        upstream.Manifold.cube((1, 2, 0.5))
    )
    assert_oriented_manifold(ours)
    assert ours.bounding_box() == pytest.approx(theirs.bounding_box())
    assert ours.volume() == pytest.approx(theirs.volume())


def test_convex_minkowski_difference_matches_upstream():
    ours = mm.Manifold.cube((3, 4, 5)).minkowski_difference(
        mm.Manifold.cube((1, 1, 2))
    )
    theirs = upstream.Manifold.cube((3, 4, 5)).minkowski_difference(
        upstream.Manifold.cube((1, 1, 2))
    )
    assert_oriented_manifold(ours)
    assert ours.bounding_box() == pytest.approx(theirs.bounding_box())
    assert ours.volume() == pytest.approx(theirs.volume())


def overlapping_boxes(module):
    return (
        module.Manifold.cube((2, 2, 2), True),
        module.Manifold.cube((2, 2, 2), True).translate((0.7, 0.2, 0.0)),
    )


@pytest.mark.parametrize(
    ("operation", "op"),
    [
        ("union", lambda a, b: a + b),
        ("difference", lambda a, b: a - b),
        ("intersection", lambda a, b: a ^ b),
    ],
)
def test_boolean_volume_and_bounds_parity(operation, op):
    ours = op(*overlapping_boxes(mm))
    theirs = op(*overlapping_boxes(upstream))
    assert_oriented_manifold(ours)
    assert ours.volume() == pytest.approx(theirs.volume(), rel=0.012)
    assert ours.bounding_box() == pytest.approx(theirs.bounding_box(), abs=0.08)
    assert ours.genus() == theirs.genus() == 0


def test_curved_boolean_parity_and_manifold_guarantee():
    ours = mm.Manifold.sphere(1, 12) - mm.Manifold.cube(
        (1.2, 1.2, 1.2), True
    ).translate((0.45, 0, 0))
    theirs = upstream.Manifold.sphere(1, 12) - upstream.Manifold.cube(
        (1.2, 1.2, 1.2), True
    ).translate((0.45, 0, 0))
    assert_oriented_manifold(ours)
    assert ours.volume() == pytest.approx(theirs.volume(), rel=0.035)


def test_disjoint_intersection_is_empty():
    ours = mm.Manifold.cube() ^ mm.Manifold.cube().translate((3, 0, 0))
    assert ours.status() is mm.Error.NoError
    assert ours.is_empty()


def test_batch_boolean_matches_operator_result():
    solids = [mm.Manifold.cube().translate((x, 0, 0)) for x in (0, 0.4, 0.8)]
    batched = mm.Manifold.batch_boolean(solids, mm.OpType.Add)
    pairwise = solids[0] + solids[1] + solids[2]
    assert_oriented_manifold(batched)
    assert batched.volume() == pytest.approx(pairwise.volume(), rel=0.02)


def test_resolution_rejects_toy_grids():
    with pytest.raises(ValueError):
        mm.set_csg_resolution(8)


@pytest.mark.parametrize("operation", [0, 1, 2])
def test_simd_field_combine_handles_scalar_tail(operation):
    first = np.linspace(-2.0, 3.0, 19)
    second = np.linspace(4.0, -1.0, 19)
    result = np.empty_like(first)
    lib().mm_combine_fields(
        addr(first), addr(second), addr(result), len(result), operation
    )
    expected = (
        np.maximum(first, second)
        if operation == 0
        else np.minimum(first, -second)
        if operation == 1
        else np.minimum(first, second)
    )
    assert result == pytest.approx(expected)


def test_parallel_field_combine_handles_task_and_simd_tails():
    count = 262_147
    first = np.linspace(-3.0, 5.0, count)
    second = np.linspace(2.0, -4.0, count)
    result = np.empty_like(first)
    lib().mm_combine_fields(addr(first), addr(second), addr(result), count, 0)
    assert result == pytest.approx(np.maximum(first, second))


def test_ffi_address_validation():
    with pytest.raises(TypeError):
        addr(np.ones(3, dtype=np.float32))
    with pytest.raises(TypeError):
        addr(np.ones(6, dtype=np.float64)[::2])
    with pytest.raises(ValueError):
        addr(np.empty(0, dtype=np.float64))
    readonly = np.ones(3, dtype=np.float64)
    readonly.flags.writeable = False
    with pytest.raises(TypeError):
        addr(readonly, writable=True)


def test_measurement_matches_numpy_for_non_multiple_triangle_count():
    solid = mm.Manifold.cube()
    vertices = solid._vertices
    faces = solid._faces[:5].copy()
    result = np.empty(2, dtype=np.float64)
    lib().mm_measure(addr(vertices), addr(faces), len(faces), addr(result))
    a = vertices[faces[:, 0]]
    b = vertices[faces[:, 1]]
    c = vertices[faces[:, 2]]
    expected_volume = abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6
    expected_area = np.linalg.norm(np.cross(b - a, c - a), axis=1).sum() / 2
    assert result == pytest.approx((expected_volume, expected_area))


def test_explicit_gpu_boolean_matches_cpu_or_falls_back():
    first, second = overlapping_boxes(mm)
    cpu = first.boolean(second, mm.OpType.Add)
    requested_gpu = first.boolean(second, mm.OpType.Add, device="gpu")
    assert_oriented_manifold(requested_gpu)
    assert requested_gpu.volume() == pytest.approx(cpu.volume(), rel=1e-12)


def test_gpu_memory_gate_falls_back_to_cpu(monkeypatch):
    first, second = overlapping_boxes(mm)
    monkeypatch.setattr(core, "_gpu_memory_available", lambda required_bytes: False)
    expected = first.boolean(second, mm.OpType.Add)
    with pytest.warns(RuntimeWarning, match="using the CPU"):
        fallback = first.boolean(second, mm.OpType.Add, device="gpu")
    assert fallback.volume() == pytest.approx(expected.volume(), rel=1e-12)


def test_gpu_allocation_cap_is_enforced():
    assert not core._gpu_memory_available(2_000_000_000)


def test_boolean_rejects_unknown_device():
    first, second = overlapping_boxes(mm)
    with pytest.raises(ValueError):
        first.boolean(second, mm.OpType.Add, device="tpu")


RAY_DIRECTION = (1.0, 0.3713906763541037, 0.6947465906068658)


def reference_signed_distance(solid, origin, spacing, shape):
    """Plain-Python point-triangle distance and ray parity, for kernel tests."""
    vertices = solid._vertices
    faces = solid._faces
    nx, ny, nz = shape
    values = np.empty(nx * ny * nz)
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                point = origin + spacing * np.array([x, y, z], dtype=np.float64)
                best2 = np.inf
                hits = 0
                for face in faces:
                    a, b, c = vertices[face]
                    ab = b - a
                    ac = c - a
                    ap = point - a
                    d1 = ab @ ap
                    d2 = ac @ ap
                    d3 = ab @ (point - b)
                    d4 = ac @ (point - b)
                    d5 = ab @ (point - c)
                    d6 = ac @ (point - c)
                    vc = d1 * d4 - d3 * d2
                    vb = d5 * d2 - d1 * d6
                    va = d3 * d6 - d5 * d4
                    if d1 <= 0.0 and d2 <= 0.0:
                        best2 = min(best2, ap @ ap)
                    elif d3 >= 0.0 and d4 <= d3:
                        best2 = min(best2, (point - b) @ (point - b))
                    elif vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
                        v = d1 / (d1 - d3)
                        delta = ap - v * ab
                        best2 = min(best2, delta @ delta)
                    elif d6 >= 0.0 and d5 <= d6:
                        best2 = min(best2, (point - c) @ (point - c))
                    elif vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
                        w = d2 / (d2 - d6)
                        delta = ap - w * ac
                        best2 = min(best2, delta @ delta)
                    elif va <= 0.0 and d4 - d3 >= 0.0 and d5 - d6 >= 0.0:
                        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
                        delta = (point - b) - w * (c - b)
                        best2 = min(best2, delta @ delta)
                    else:
                        denom = 1.0 / (va + vb + vc)
                        delta = ap - ab * (vb * denom) - ac * (vc * denom)
                        best2 = min(best2, delta @ delta)
                    direction = np.array(RAY_DIRECTION)
                    h = np.cross(direction, ac)
                    det = ab @ h
                    if abs(det) > 1.0e-14:
                        inv = 1.0 / det
                        u = (ap @ h) * inv
                        q = np.cross(ap, ab)
                        v = (direction @ q) * inv
                        t = (ac @ q) * inv
                        if 0.0 <= u <= 1.0 and v >= 0.0 and u + v <= 1.0:
                            if t > 1.0e-12:
                                hits += 1
                distance = max(np.sqrt(max(best2, 0.0)), spacing * 1.0e-9)
                values[z * nx * ny + y * nx + x] = (
                    distance if hits % 2 else -distance
                )
    return values


@pytest.mark.parametrize("shape", [(2, 2, 2), (5, 3, 2), (7, 2, 2), (8, 1, 2)])
def test_signed_distance_matches_reference_for_both_widths(shape):
    # nx below the vector width takes the scalar fallback, and nx above it
    # leaves a short row that the overlapping final chunk has to cover.
    solid = mm.Manifold.cube()
    spacing = 0.37
    origin = np.array([-0.6, -0.4, -0.2])
    expected = reference_signed_distance(solid, origin, spacing, shape)
    result = np.empty(int(np.prod(shape)))
    lib().mm_signed_distance(
        addr(solid._vertices),
        addr(solid._faces),
        len(solid._faces),
        float(origin[0]),
        float(origin[1]),
        float(origin[2]),
        spacing,
        shape[0],
        shape[1],
        shape[2],
        addr(result, writable=True),
    )
    assert result == pytest.approx(expected, abs=1e-12)


def test_weld_crossings_dense_path_matches_sorted_path():
    rng = np.random.default_rng(11)
    nx = ny = nz = 6
    node_count = nx * ny * nz
    cells = rng.integers(0, nx - 1, size=(60, 3))
    # The 13 cell-internal edges a six-tetrahedron cell can cross.
    corner_pairs = [
        (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7),
        (1, 3), (1, 5), (2, 3), (2, 6), (4, 5), (4, 6),
    ]
    corners = [
        rng.integers(0, len(corner_pairs), size=len(cells)) for _ in range(3)
    ]
    keys = np.empty((len(cells), 3), dtype=np.int64)
    for slot, chosen in enumerate(corners):
        first = np.array([corner_pairs[c][0] for c in chosen])
        second = np.array([corner_pairs[c][1] for c in chosen])
        base = (cells[:, 2] * ny + cells[:, 1]) * nx + cells[:, 0]
        start = base + (first & 1) + ((first >> 1) & 1) * nx + (first >> 2) * nx * ny
        stop = base + (second & 1) + ((second >> 1) & 1) * nx + (second >> 2) * nx * ny
        keys[:, slot] = np.minimum(start, stop) * node_count + np.maximum(
            start, stop
        )
    # A crossing point only depends on its edge, so every occurrence of an edge
    # carries the same coordinates, as the marching-tetrahedra kernel emits.
    triangles = np.repeat(keys.astype(np.float64)[:, :, None], 3, axis=2)
    dense_vertices, dense_faces = core._weld_crossings(
        triangles, keys, node_count, nx, ny
    )
    sorted_vertices, sorted_faces = core._weld_crossings(
        triangles, keys, node_count, 1, ny
    )
    assert len(dense_vertices) == len(sorted_vertices)
    assert np.array_equal(
        np.sort(dense_vertices, axis=0), np.sort(sorted_vertices, axis=0)
    )
    assert dense_faces.shape == sorted_faces.shape
    assert len(np.unique(dense_faces)) == len(np.unique(sorted_faces))


def test_weld_crossings_merges_every_repeated_edge():
    # Two triangles whose corners sit on the cell edges {3,4} and {0,3}, and on
    # {0,7} and {0,5}, each edge repeated by both incident triangles.
    triangles = np.arange(2 * 3 * 3, dtype=np.float64).reshape((2, 3, 3))
    node_count = 8
    keys = np.array(
        [[3 * node_count + 4, 3, 3 * node_count + 4], [7, 5, 7]],
        dtype=np.int64,
    )
    vertices, faces = core._weld_crossings(
        triangles, keys, node_count, 2, 2
    )
    assert len(vertices) == 4
    assert faces[0, 0] == faces[0, 2]
    assert faces[1, 0] == faces[1, 2]
    assert len(np.unique(faces)) == 4
    fallback_vertices, fallback_faces = core._weld_crossings(
        triangles, keys, node_count, 1, 2
    )
    assert len(fallback_vertices) == len(vertices)
    assert len(np.unique(fallback_faces)) == 4


def test_edge_status_accepts_two_closed_meshes_and_rejects_a_flap():
    cube = mm.Manifold.cube()
    vertices, faces = cube._vertices, cube._faces
    assert core._edge_status(vertices, faces) is mm.Error.NoError
    doubled = np.concatenate((faces, faces + len(vertices)))
    doubled_vertices = np.concatenate((vertices, vertices + 0.5))
    assert core._edge_status(doubled_vertices, doubled) is mm.Error.NoError
    flipped = faces.copy()
    flipped[0] = flipped[0][[0, 2, 1]]
    assert core._edge_status(vertices, flipped) is mm.Error.NotManifold
    opened = np.delete(faces, 0, axis=0)
    assert core._edge_status(vertices, opened) is mm.Error.NotManifold
    assert core._edge_status(vertices, faces[:0]) is mm.Error.NoError


def test_march_count_and_emit_agree_on_saturated_and_straddling_fields():
    nx = ny = nz = 7
    field = np.ones(nx * ny * nz)
    assert lib().mm_march_count(addr(field), nx, ny, nz) == 0
    field = -np.ones(nx * ny * nz)
    assert lib().mm_march_count(addr(field), nx, ny, nz) == 0
    rng = np.random.default_rng(5)
    field = rng.normal(size=nx * ny * nz)
    count = int(lib().mm_march_count(addr(field), nx, ny, nz))
    assert count > 0
    triangles = np.empty((count, 3, 3))
    keys = np.empty((count, 3), dtype=np.int64)
    emitted = lib().mm_march_emit(
        addr(field),
        addr(triangles, writable=True),
        addr(keys, writable=True),
        nx,
        ny,
        nz,
        -1.0,
        0.5,
        0.25,
        0.5,
    )
    assert emitted == count
    assert np.isfinite(triangles).all()
    assert (keys >= 0).all()
    assert (keys < nx * ny * nz * nx * ny * nz).all()
    assert 0 < np.unique(keys).size <= 3 * count
