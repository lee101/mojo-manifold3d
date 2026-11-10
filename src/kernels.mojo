"""Geometry kernels exported through a stable C ABI.

Python owns every buffer. Addresses cross the ABI as Int so exported
functions stay non-parametric. CPU kernels are serial below their measured
crossover and fan out through `max.algorithm.parallelize` above it.
"""

from max.algorithm import parallelize
from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.host import DeviceContext
from std.math import sqrt
from std.sys import has_accelerator, simd_width_of

comptime FPtr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int64, AnyOrigin[mut=True]]
comptime GPUFPtr = Pointer[Float64, AnyOrigin[mut=True]]
comptime GPUIPtr = Pointer[Int64, AnyOrigin[mut=True]]

comptime W = simd_width_of[DType.float64]()
comptime FD = SIMD[DType.float64, W]
comptime BD = SIMD[DType.bool, W]
comptime F_ZERO = FD(0.0)
comptime F_ONE = FD(1.0)
comptime F_EPS = FD(1.0e-12)
# A fixed non-axis-aligned ray direction, shared by the CPU and device kernels.
comptime RAY_X = 1.0
comptime RAY_Y = 0.3713906763541037
comptime RAY_Z = 0.6947465906068658
comptime F_RAY_X = FD(RAY_X)
comptime F_RAY_Y = FD(RAY_Y)
comptime F_RAY_Z = FD(RAY_Z)



def fp(address: Int) -> FPtr:
    return FPtr(unsafe_from_address=address)


def ip(address: Int) -> IPtr:
    return IPtr(unsafe_from_address=address)


@always_inline
def point_triangle_distance2(
    px: Float64,
    py: Float64,
    pz: Float64,
    ax: Float64,
    ay: Float64,
    az: Float64,
    bx: Float64,
    by: Float64,
    bz: Float64,
    cx: Float64,
    cy: Float64,
    cz: Float64,
) -> Float64:
    var abx = bx - ax
    var aby = by - ay
    var abz = bz - az
    var acx = cx - ax
    var acy = cy - ay
    var acz = cz - az
    var apx = px - ax
    var apy = py - ay
    var apz = pz - az
    var d1 = abx * apx + aby * apy + abz * apz
    var d2 = acx * apx + acy * apy + acz * apz
    if d1 <= 0.0 and d2 <= 0.0:
        return apx * apx + apy * apy + apz * apz

    var bpx = px - bx
    var bpy = py - by
    var bpz = pz - bz
    var d3 = abx * bpx + aby * bpy + abz * bpz
    var d4 = acx * bpx + acy * bpy + acz * bpz
    if d3 >= 0.0 and d4 <= d3:
        return bpx * bpx + bpy * bpy + bpz * bpz

    var vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        var v = d1 / (d1 - d3)
        var dx = apx - v * abx
        var dy = apy - v * aby
        var dz = apz - v * abz
        return dx * dx + dy * dy + dz * dz

    var cpx = px - cx
    var cpy = py - cy
    var cpz = pz - cz
    var d5 = abx * cpx + aby * cpy + abz * cpz
    var d6 = acx * cpx + acy * cpy + acz * cpz
    if d6 >= 0.0 and d5 <= d6:
        return cpx * cpx + cpy * cpy + cpz * cpz

    var vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        var w = d2 / (d2 - d6)
        var dx = apx - w * acx
        var dy = apy - w * acy
        var dz = apz - w * acz
        return dx * dx + dy * dy + dz * dz

    var va = d3 * d6 - d5 * d4
    if va <= 0.0 and d4 - d3 >= 0.0 and d5 - d6 >= 0.0:
        var bcx = cx - bx
        var bcy = cy - by
        var bcz = cz - bz
        var w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        var dx = bpx - w * bcx
        var dy = bpy - w * bcy
        var dz = bpz - w * bcz
        return dx * dx + dy * dy + dz * dz

    var denom = 1.0 / (va + vb + vc)
    var v = vb * denom
    var w = vc * denom
    var dx = apx - abx * v - acx * w
    var dy = apy - aby * v - acy * w
    var dz = apz - abz * v - acz * w
    return dx * dx + dy * dy + dz * dz


@always_inline
def ray_intersects(
    px: Float64,
    py: Float64,
    pz: Float64,
    ax: Float64,
    ay: Float64,
    az: Float64,
    bx: Float64,
    by: Float64,
    bz: Float64,
    cx: Float64,
    cy: Float64,
    cz: Float64,
) -> Bool:
    # A non-axis-aligned direction avoids the systematic edge hits produced by
    # CAD meshes whose faces lie on coordinate planes.
    var dx = 1.0
    var dy = 0.3713906763541037
    var dz = 0.6947465906068658
    var e1x = bx - ax
    var e1y = by - ay
    var e1z = bz - az
    var e2x = cx - ax
    var e2y = cy - ay
    var e2z = cz - az
    var hx = dy * e2z - dz * e2y
    var hy = dz * e2x - dx * e2z
    var hz = dx * e2y - dy * e2x
    var det = e1x * hx + e1y * hy + e1z * hz
    if abs(det) <= 1.0e-14:
        return False
    var inv_det = 1.0 / det
    var sx = px - ax
    var sy = py - ay
    var sz = pz - az
    var u = (sx * hx + sy * hy + sz * hz) * inv_det
    if u < 0.0 or u > 1.0:
        return False
    var qx = sy * e1z - sz * e1y
    var qy = sz * e1x - sx * e1z
    var qz = sx * e1y - sy * e1x
    var v = (dx * qx + dy * qy + dz * qz) * inv_det
    if v < 0.0 or u + v > 1.0:
        return False
    return (e2x * qx + e2y * qy + e2z * qz) * inv_det > 1.0e-12


@always_inline
def tri_distance2_simd(
    px: FD,
    py: FD,
    pz: FD,
    ax: Float64,
    ay: Float64,
    az: Float64,
    abx: Float64,
    aby: Float64,
    abz: Float64,
    acx: Float64,
    acy: Float64,
    acz: Float64,
    ab2: Float64,
    ac2: Float64,
    bc2: Float64,
    ab_dot_ac: Float64,
    inv_ab2: Float64,
    inv_ac2: Float64,
    inv_bc2: Float64,
) -> FD:
    # Branch-free restatement of point_triangle_distance2 for W points at once.
    # The seven Voronoi regions of a triangle are disjoint, so selecting them in
    # the reverse of the scalar early-exit order reproduces it. Every lane
    # evaluates every region, which is why the three edge regions divide by
    # per-triangle constants: d1 - d3 is |ab|^2, d2 - d6 is |ac|^2 and
    # (d4 - d3) + (d5 - d6) is |bc|^2, so all three become multiplies by a
    # reciprocal hoisted out of the point loop.
    var apx = px - ax
    var apy = py - ay
    var apz = pz - az
    var d1 = abx * apx + aby * apy + abz * apz
    var d2 = acx * apx + acy * apy + acz * apz
    var d3 = d1 - ab2
    var d4 = d2 - ab_dot_ac
    var d5 = d1 - ab_dot_ac
    var d6 = d2 - ac2
    var vc = d1 * d4 - d3 * d2
    var vb = d5 * d2 - d1 * d6
    var va = d3 * d6 - d5 * d4
    var edge_x = d4 - d3
    var edge_y = d5 - d6
    var r1 = apx * apx + apy * apy + apz * apz

    var c1: BD = SIMD.le(d1, F_ZERO) & SIMD.le(d2, F_ZERO)
    var c2: BD = SIMD.ge(d3, F_ZERO) & SIMD.le(d4, d3)
    var c3: BD = SIMD.le(vc, F_ZERO) & SIMD.ge(d1, F_ZERO) & SIMD.le(d3, F_ZERO)
    var c4: BD = SIMD.ge(d6, F_ZERO) & SIMD.le(d5, d6)
    var c5: BD = SIMD.le(vb, F_ZERO) & SIMD.ge(d2, F_ZERO) & SIMD.le(d6, F_ZERO)
    var c6: BD = SIMD.le(va, F_ZERO) & SIMD.ge(edge_x, F_ZERO) & SIMD.ge(edge_y, F_ZERO)

    var v3 = d1 * inv_ab2
    var e3x = apx - v3 * abx
    var e3y = apy - v3 * aby
    var e3z = apz - v3 * abz
    var r3 = e3x * e3x + e3y * e3y + e3z * e3z

    var w5 = d2 * inv_ac2
    var e5x = apx - w5 * acx
    var e5y = apy - w5 * acy
    var e5z = apz - w5 * acz
    var r5 = e5x * e5x + e5y * e5y + e5z * e5z

    var w6 = edge_x * inv_bc2
    var b6x = abx + w6 * (acx - abx)
    var b6y = aby + w6 * (acy - aby)
    var b6z = abz + w6 * (acz - abz)
    var e6x = apx - b6x
    var e6y = apy - b6y
    var e6z = apz - b6z
    var r6 = e6x * e6x + e6y * e6y + e6z * e6z

    var denom = 1.0 / (va + vb + vc)
    var v7 = vb * denom
    var w7 = vc * denom
    var e7x = apx - abx * v7 - acx * w7
    var e7y = apy - aby * v7 - acy * w7
    var e7z = apz - abz * v7 - acz * w7
    var r7 = e7x * e7x + e7y * e7y + e7z * e7z

    var r2 = r1 - 2.0 * d1 + ab2
    var r4 = r1 - 2.0 * d2 + ac2
    var result = r7
    result = c6.select(r6, result)
    result = c5.select(r5, result)
    result = c4.select(r4, result)
    result = c3.select(r3, result)
    result = c2.select(r2, result)
    result = c1.select(r1, result)
    return result


@always_inline
def ray_hit_simd(
    px: FD,
    py: FD,
    pz: FD,
    ax: Float64,
    ay: Float64,
    az: Float64,
    e1x: Float64,
    e1y: Float64,
    e1z: Float64,
    e2x: Float64,
    e2y: Float64,
    e2z: Float64,
    hx: Float64,
    hy: Float64,
    hz: Float64,
    inv_det: Float64,
) -> BD:
    # Branch-free restatement of ray_intersects: Moller-Trumbore without the
    # early returns, so every lane tests the same conjunction of bounds.
    var sx = px - ax
    var sy = py - ay
    var sz = pz - az
    var u = (sx * hx + sy * hy + sz * hz) * inv_det
    var qx = sy * e1z - sz * e1y
    var qy = sz * e1x - sx * e1z
    var qz = sx * e1y - sy * e1x
    var v = (F_RAY_X * qx + F_RAY_Y * qy + F_RAY_Z * qz) * inv_det
    var t = (e2x * qx + e2y * qy + e2z * qz) * inv_det
    var hit: BD = SIMD.ge(u, F_ZERO) & SIMD.le(u, F_ONE) & SIMD.ge(v, F_ZERO)
    hit = hit & SIMD.le(u + v, F_ONE) & SIMD.gt(t, F_EPS)
    return hit


@always_inline
def sdf_chunk(
    vertices: FPtr,
    faces: IPtr,
    triangle_count: Int,
    ox: Float64,
    spacing: Float64,
    row: Int,
    py: Float64,
    pz: Float64,
    x: Int,
    result: FPtr,
):
    # One W-wide chunk of consecutive x samples. The triangle data is loaded and
    # reduced once per triangle instead of once per (point, triangle) pair.
    var lane = FD()
    var i = 0
    while i < W:
        lane[i] = Float64(i)
        i += 1
    var px = FD(ox) + spacing * (lane + Float64(x))
    var best2 = FD(1.7976931348623157e308)
    var odd = BD(fill=False)
    for triangle in range(triangle_count):
        var ia = Int(faces.unsafe_load(triangle * 3)) * 3
        var ib = Int(faces.unsafe_load(triangle * 3 + 1)) * 3
        var ic = Int(faces.unsafe_load(triangle * 3 + 2)) * 3
        var ax = vertices.unsafe_load(ia)
        var ay = vertices.unsafe_load(ia + 1)
        var az = vertices.unsafe_load(ia + 2)
        var bx = vertices.unsafe_load(ib)
        var by = vertices.unsafe_load(ib + 1)
        var bz = vertices.unsafe_load(ib + 2)
        var cx = vertices.unsafe_load(ic)
        var cy = vertices.unsafe_load(ic + 1)
        var cz = vertices.unsafe_load(ic + 2)
        var abx = bx - ax
        var aby = by - ay
        var abz = bz - az
        var acx = cx - ax
        var acy = cy - ay
        var acz = cz - az
        var bcx = acx - abx
        var bcy = acy - aby
        var bcz = acz - abz
        var ab2 = abx * abx + aby * aby + abz * abz
        var ac2 = acx * acx + acy * acy + acz * acz
        var bc2 = bcx * bcx + bcy * bcy + bcz * bcz
        best2 = min(
            best2,
            tri_distance2_simd(
                px, py, pz, ax, ay, az, abx, aby, abz, acx, acy, acz,
                ab2, ac2, bc2, abx * acx + aby * acy + abz * acz,
                1.0 / ab2, 1.0 / ac2, 1.0 / bc2,
            ),
        )
        var hx = RAY_Y * acz - RAY_Z * acy
        var hy = RAY_Z * acx - RAY_X * acz
        var hz = RAY_X * acy - RAY_Y * acx
        var det = abx * hx + aby * hy + abz * hz
        if abs(det) > 1.0e-14:
            odd = odd ^ ray_hit_simd(
                px, py, pz, ax, ay, az, abx, aby, abz, acx, acy, acz,
                hx, hy, hz, 1.0 / det,
            )
    var distance = max(sqrt(max(best2, F_ZERO)), FD(spacing * 1.0e-9))
    result.unsafe_store(row + x, odd.select(distance, -distance))



@always_inline
def sdf_point(
    vertices: FPtr,
    faces: IPtr,
    triangle_count: Int,
    px: Float64,
    py: Float64,
    pz: Float64,
    spacing: Float64,
    result: FPtr,
    point: Int,
):
    # Scalar fallback for rows narrower than a single vector.
    var best2 = 1.7976931348623157e308
    var hits = 0
    for triangle in range(triangle_count):
        var ia = Int(faces.unsafe_load(triangle * 3)) * 3
        var ib = Int(faces.unsafe_load(triangle * 3 + 1)) * 3
        var ic = Int(faces.unsafe_load(triangle * 3 + 2)) * 3
        var ax = vertices.unsafe_load(ia)
        var ay = vertices.unsafe_load(ia + 1)
        var az = vertices.unsafe_load(ia + 2)
        var bx = vertices.unsafe_load(ib)
        var by = vertices.unsafe_load(ib + 1)
        var bz = vertices.unsafe_load(ib + 2)
        var cx = vertices.unsafe_load(ic)
        var cy = vertices.unsafe_load(ic + 1)
        var cz = vertices.unsafe_load(ic + 2)
        best2 = min(
            best2,
            point_triangle_distance2(px, py, pz, ax, ay, az, bx, by, bz, cx, cy, cz),
        )
        if ray_intersects(px, py, pz, ax, ay, az, bx, by, bz, cx, cy, cz):
            hits += 1
    var distance = max(sqrt(max(best2, 0.0)), spacing * 1.0e-9)
    result.unsafe_store(point, distance if hits % 2 == 1 else -distance)


comptime PARALLEL_POINTS_THRESHOLD = 1_000_000
# Measured on the shared host: 2 and 4 workers both remove the pool start-up
# cost that makes the first parallel boolean slow; 4 is the cap this port takes.
comptime PARALLEL_WORKERS = 4
# Fixed task count keeps the output independent of the worker count: each task
# owns one fixed row range, so the split never depends on how many workers run.
comptime PARALLEL_TASKS = 16
comptime MAX_ROW_TASKS = 4096


@always_inline
def row_tasks(rows: Int) -> Int:
    # Fixed split: one task per row down to PARALLEL_TASKS, capped at
    # MAX_ROW_TASKS. The count depends only on the grid, never on the worker
    # count, so every worker configuration produces an identical grid.
    if rows > MAX_ROW_TASKS:
        return MAX_ROW_TASKS
    if rows > PARALLEL_TASKS:
        return rows
    return PARALLEL_TASKS


@export("mm_signed_distance_data")
def mm_signed_distance_data(
    vertices_address: Int,
    faces_address: Int,
    triangle_count: Int,
    ox: Float64,
    oy: Float64,
    oz: Float64,
    spacing: Float64,
    nx: Int,
    ny: Int,
    nz: Int,
    result_address: Int,
    workers: Int,
) abi("C") -> Int:
    # Test entry point: any worker count, threshold bypassed, still capped.
    # Lets the parity test compare worker counts on the same input.
    signed_distance_grid(
        fp(vertices_address), ip(faces_address), triangle_count,
        ox, oy, oz, spacing, nx, ny, nz, fp(result_address),
        min(max(workers, 1), PARALLEL_WORKERS),
    )
    return 1


def signed_distance_grid(
    vertices: FPtr,
    faces: IPtr,
    triangle_count: Int,
    ox: Float64,
    oy: Float64,
    oz: Float64,
    spacing: Float64,
    nx: Int,
    ny: Int,
    nz: Int,
    result: FPtr,
    workers: Int,
):
    # `workers == 1` is the serial path: the same task bodies, no launch.
    # Every row writes only its own samples and each row's value is computed by
    # the same code, so every worker count produces a bit-identical grid.
    var row_stride = nx * ny
    if nx < W:
        for z in range(nz):
            var pz = oz + spacing * Float64(z)
            for y in range(ny):
                var py = oy + spacing * Float64(y)
                for x in range(nx):
                    sdf_point(
                        vertices, faces, triangle_count,
                        ox + spacing * Float64(x), py, pz, spacing,
                        result, z * row_stride + y * nx + x,
                    )
        return
    var vector_end = nx // W * W
    var rows = ny * nz
    var tasks = row_tasks(rows)
    def work(task: Int) {imm}:
        var first_row = task * rows // tasks
        var last_row = (task + 1) * rows // tasks
        for flat_row in range(first_row, last_row):
            var z = flat_row // ny
            var y = flat_row - z * ny
            var pz = oz + spacing * Float64(z)
            var row = z * row_stride + y * nx
            var py = oy + spacing * Float64(y)
            var x = 0
            while x < vector_end:
                sdf_chunk(
                    vertices, faces, triangle_count, ox, spacing,
                    row, py, pz, x, result,
                )
                x += W
            if vector_end < nx:
                # The row's short tail is covered by re-running the final
                # full-width chunk, which recomputes and rewrites the same values.
                sdf_chunk(
                    vertices, faces, triangle_count, ox, spacing,
                    row, py, pz, nx - W, result,
                )

    # The serial path runs the same task bodies on the caller's thread. The
    # parallel path walks the same fixed ranges, so the output is identical.
    if workers <= 1:
        for task in range(tasks):
            work(task)
    else:
        parallelize(work, tasks, workers)


def signed_distance_pair(
    vertices: FPtr,
    faces: IPtr,
    triangle_count: Int,
    other_vertices: FPtr,
    other_faces: IPtr,
    other_triangle_count: Int,
    ox: Float64,
    oy: Float64,
    oz: Float64,
    spacing: Float64,
    nx: Int,
    ny: Int,
    nz: Int,
    result: FPtr,
    other_result: FPtr,
    workers: Int,
):
    # One launch evaluates both meshes: each task owns a fixed row range and
    # runs the same row body for the first mesh and then the second. Worker
    # count does not change the split, so the output is bit-identical.
    var row_stride = nx * ny
    if nx < W:
        # Scalar fallback, the same one the single-mesh kernel uses.
        for z in range(nz):
            var pz = oz + spacing * Float64(z)
            for y in range(ny):
                var py = oy + spacing * Float64(y)
                for x in range(nx):
                    var point = z * row_stride + y * nx + x
                    var px = ox + spacing * Float64(x)
                    sdf_point(
                        vertices, faces, triangle_count,
                        px, py, pz, spacing, result, point,
                    )
                    sdf_point(
                        other_vertices, other_faces, other_triangle_count,
                        px, py, pz, spacing, other_result, point,
                    )
        return
    var vector_end = nx // W * W
    var rows = ny * nz
    var tasks = row_tasks(rows)
    def work(task: Int) {imm}:
        var first_row = task * rows // tasks
        var last_row = (task + 1) * rows // tasks
        for mesh in range(2):
            var mesh_triangles = (
                triangle_count if mesh == 0 else other_triangle_count
            )
            if mesh_triangles == 0:
                continue
            var mesh_vertices = (
                vertices if mesh == 0 else other_vertices
            )
            var mesh_faces = faces if mesh == 0 else other_faces
            var mesh_result = result if mesh == 0 else other_result
            for flat_row in range(first_row, last_row):
                var z = flat_row // ny
                var y = flat_row - z * ny
                var pz = oz + spacing * Float64(z)
                var row = z * row_stride + y * nx
                var py = oy + spacing * Float64(y)
                var x = 0
                while x < vector_end:
                    sdf_chunk(
                        mesh_vertices, mesh_faces, mesh_triangles,
                        ox, spacing, row, py, pz, x, mesh_result,
                    )
                    x += W
                if vector_end < nx:
                    sdf_chunk(
                        mesh_vertices, mesh_faces, mesh_triangles,
                        ox, spacing, row, py, pz, nx - W, mesh_result,
                    )

    # The serial path runs the same task bodies on the caller's thread. The
    # parallel path walks the same fixed ranges, so the output is identical.
    if workers <= 1:
        for task in range(tasks):
            work(task)
    else:
        parallelize(work, tasks, workers)


@export("mm_signed_distance_pair")
def mm_signed_distance_pair(
    first_vertices_address: Int,
    first_faces_address: Int,
    first_triangle_count: Int,
    second_vertices_address: Int,
    second_faces_address: Int,
    second_triangle_count: Int,
    ox: Float64,
    oy: Float64,
    oz: Float64,
    spacing: Float64,
    nx: Int,
    ny: Int,
    nz: Int,
    first_result_address: Int,
    second_result_address: Int,
    workers: Int,
) abi("C"):
    var first_vertices = fp(first_vertices_address)
    var second_vertices = fp(second_vertices_address)
    var first_result = fp(first_result_address)
    var second_result = fp(second_result_address)
    # Identical geometry (a solid unioned with itself, or two handles to one
    # mesh) needs one grid, not two: both fields hold the same values, so the
    # second result is a copy of the first.
    if (
        first_vertices_address == second_vertices_address
        and first_faces_address == second_faces_address
        and first_triangle_count == second_triangle_count
        and first_result_address != second_result_address
    ):
        signed_distance_grid(
            first_vertices, ip(first_faces_address), first_triangle_count,
            ox, oy, oz, spacing, nx, ny, nz, first_result,
            effective_workers(workers, nx * ny * nz),
        )
        var count = nx * ny * nz
        for i in range(count):
            second_result.unsafe_store(i, first_result.unsafe_load(i))
        return
    # Below the vector width the pair body falls back to the scalar path, the
    # same as the single-mesh kernel.
    var active = effective_workers(workers, nx * ny * nz)
    if nx < W:
        active = 1
    signed_distance_pair(
        first_vertices, ip(first_faces_address), first_triangle_count,
        second_vertices, ip(second_faces_address), second_triangle_count,
        ox, oy, oz, spacing, nx, ny, nz, first_result, second_result, active,
    )


@always_inline
def effective_workers(workers: Int, point_count: Int) -> Int:
    # -1 asks for the compiled-in count, 0 by mistake is still one worker.
    # Below the measured threshold any request above one stays serial.
    var active = workers
    if active == 0:
        active = 1
    elif active == -1:
        active = PARALLEL_WORKERS
    if active > 1 and point_count < PARALLEL_POINTS_THRESHOLD:
        active = 1
    return active


@export("mm_signed_distance")
def mm_signed_distance(
    vertices_address: Int,
    faces_address: Int,
    triangle_count: Int,
    ox: Float64,
    oy: Float64,
    oz: Float64,
    spacing: Float64,
    nx: Int,
    ny: Int,
    nz: Int,
    result_address: Int,
    workers: Int,
) abi("C"):
    var active = effective_workers(workers, nx * ny * nz)
    signed_distance_grid(
        fp(vertices_address), ip(faces_address), triangle_count,
        ox, oy, oz, spacing, nx, ny, nz, fp(result_address), active,
    )



def signed_distance_gpu_kernel(
    vertices: GPUFPtr,
    faces: GPUIPtr,
    triangle_count: Int32,
    ox: Float64,
    oy: Float64,
    oz: Float64,
    spacing: Float64,
    nx: Int32,
    ny: Int32,
    count: Int32,
    result: GPUFPtr,
):
    var point = block_idx.x * block_dim.x + thread_idx.x
    if point >= Int(count):
        return
    var x_index = point % Int(nx)
    var yz = point // Int(nx)
    var y_index = yz % Int(ny)
    var z_index = yz // Int(ny)
    var px = ox + spacing * Float64(x_index)
    var py = oy + spacing * Float64(y_index)
    var pz = oz + spacing * Float64(z_index)
    var best2 = 1.7976931348623157e308
    var hits = 0
    for triangle in range(Int(triangle_count)):
        var ia = Int(faces.unsafe_load(Int(triangle) * 3)) * 3
        var ib = Int(faces.unsafe_load(Int(triangle) * 3 + 1)) * 3
        var ic = Int(faces.unsafe_load(Int(triangle) * 3 + 2)) * 3
        var ax = vertices.unsafe_load(ia)
        var ay = vertices.unsafe_load(ia + 1)
        var az = vertices.unsafe_load(ia + 2)
        var bx = vertices.unsafe_load(ib)
        var by = vertices.unsafe_load(ib + 1)
        var bz = vertices.unsafe_load(ib + 2)
        var cx = vertices.unsafe_load(ic)
        var cy = vertices.unsafe_load(ic + 1)
        var cz = vertices.unsafe_load(ic + 2)
        best2 = min(
            best2,
            point_triangle_distance2(
                px, py, pz, ax, ay, az, bx, by, bz, cx, cy, cz
            ),
        )
        if ray_intersects(
            px, py, pz, ax, ay, az, bx, by, bz, cx, cy, cz
        ):
            hits += 1
    var distance = max(sqrt(max(best2, 0.0)), spacing * 1.0e-9)
    result.unsafe_store(
        Int(point), (distance if hits % 2 == 1 else -distance)
    )


@export("mm_signed_distance_gpu")
def mm_signed_distance_gpu(
    vertices_address: Int,
    faces_address: Int,
    vertex_count: Int,
    triangle_count: Int,
    ox: Float64,
    oy: Float64,
    oz: Float64,
    spacing: Float64,
    nx: Int,
    ny: Int,
    nz: Int,
    result_address: Int,
) abi("C") -> Int:
    comptime if not has_accelerator():
        return 0
    else:
        try:
            var count = nx * ny * nz
            var device_bytes = (
                vertex_count * 24 + triangle_count * 24 + count * 8
            )
            if device_bytes >= 2_000_000_000:
                return 0
            with DeviceContext() as ctx:
                var vertices_device = ctx.enqueue_create_buffer[DType.float64](
                    vertex_count * 3
                )
                var faces_device = ctx.enqueue_create_buffer[DType.int64](
                    triangle_count * 3
                )
                var result_device = ctx.enqueue_create_buffer[DType.float64](
                    count
                )
                ctx.enqueue_copy(vertices_device, fp(vertices_address))
                ctx.enqueue_copy(faces_device, ip(faces_address))
                ctx.enqueue_function[signed_distance_gpu_kernel](
                    vertices_device,
                    faces_device,
                    Int32(triangle_count),
                    ox,
                    oy,
                    oz,
                    spacing,
                    Int32(nx),
                    Int32(ny),
                    Int32(count),
                    result_device,
                    grid_dim=(count + 255) // 256,
                    block_dim=256,
                )
                ctx.enqueue_copy(fp(result_address), result_device)
                ctx.synchronize()
            return 1
        except:
            return 0


@export("mm_signed_distance_pair_gpu")
def mm_signed_distance_pair_gpu(
    first_vertices_address: Int,
    first_faces_address: Int,
    first_vertex_count: Int,
    first_triangle_count: Int,
    second_vertices_address: Int,
    second_faces_address: Int,
    second_vertex_count: Int,
    second_triangle_count: Int,
    ox: Float64,
    oy: Float64,
    oz: Float64,
    spacing: Float64,
    nx: Int,
    ny: Int,
    nz: Int,
    first_result_address: Int,
    second_result_address: Int,
) abi("C") -> Int:
    comptime if not has_accelerator():
        return 0
    else:
        try:
            var count = nx * ny * nz
            var device_bytes = (
                first_vertex_count * 24
                + first_triangle_count * 24
                + second_vertex_count * 24
                + second_triangle_count * 24
                + count * 16
            )
            if device_bytes >= 2_000_000_000:
                return 0
            with DeviceContext() as ctx:
                var first_vertices = ctx.enqueue_create_buffer[DType.float64](
                    first_vertex_count * 3
                )
                var first_faces = ctx.enqueue_create_buffer[DType.int64](
                    first_triangle_count * 3
                )
                var second_vertices = ctx.enqueue_create_buffer[DType.float64](
                    second_vertex_count * 3
                )
                var second_faces = ctx.enqueue_create_buffer[DType.int64](
                    second_triangle_count * 3
                )
                var first_result = ctx.enqueue_create_buffer[DType.float64](
                    count
                )
                var second_result = ctx.enqueue_create_buffer[DType.float64](
                    count
                )
                ctx.enqueue_copy(first_vertices, fp(first_vertices_address))
                ctx.enqueue_copy(first_faces, ip(first_faces_address))
                ctx.enqueue_copy(second_vertices, fp(second_vertices_address))
                ctx.enqueue_copy(second_faces, ip(second_faces_address))
                var kernel = ctx.compile_function[signed_distance_gpu_kernel]()
                ctx.enqueue_function(
                    kernel,
                    first_vertices,
                    first_faces,
                    Int32(first_triangle_count),
                    ox,
                    oy,
                    oz,
                    spacing,
                    Int32(nx),
                    Int32(ny),
                    Int32(count),
                    first_result,
                    grid_dim=(count + 255) // 256,
                    block_dim=256,
                )
                ctx.enqueue_function(
                    kernel,
                    second_vertices,
                    second_faces,
                    Int32(second_triangle_count),
                    ox,
                    oy,
                    oz,
                    spacing,
                    Int32(nx),
                    Int32(ny),
                    Int32(count),
                    second_result,
                    grid_dim=(count + 255) // 256,
                    block_dim=256,
                )
                ctx.enqueue_copy(fp(first_result_address), first_result)
                ctx.enqueue_copy(fp(second_result_address), second_result)
                ctx.synchronize()
            return 1
        except:
            return 0


@export("mm_combine_fields")
def mm_combine_fields(
    first_address: Int,
    second_address: Int,
    result_address: Int,
    count: Int,
    operation: Int,
) abi("C"):
    var first = fp(first_address)
    var second = fp(second_address)
    var result = fp(result_address)
    comptime W = simd_width_of[DType.float64]()
    var vector_end = count // W * W
    for i in range(0, vector_end, W):
        var a = first.unsafe_load[width=W](i)
        var b = second.unsafe_load[width=W](i)
        if operation == 0:
            result.unsafe_store(i, max(a, b))
        elif operation == 1:
            result.unsafe_store(i, min(a, -b))
        else:
            result.unsafe_store(i, min(a, b))
    for i in range(vector_end, count):
        if operation == 0:
            result.unsafe_store(
                i, max(first.unsafe_load(i), second.unsafe_load(i))
            )
        elif operation == 1:
            result.unsafe_store(
                i, min(first.unsafe_load(i), -second.unsafe_load(i))
            )
        else:
            result.unsafe_store(
                i, min(first.unsafe_load(i), second.unsafe_load(i))
            )


@always_inline
def corner_offset(corner: Int, x_stride: Int, plane_stride: Int) -> Int:
    # Corner 0..7 of a cell: bit 0 steps x, bit 1 steps y, bit 2 steps z.
    return (
        (corner & 1)
        + ((corner >> 1) & 1) * x_stride
        + (corner >> 2) * plane_stride
    )


@always_inline
def tet_triangle_count(
    first: Float64, second: Float64, third: Float64, fourth: Float64
) -> Int:
    var inside = 0
    if first > 0.0:
        inside += 1
    if second > 0.0:
        inside += 1
    if third > 0.0:
        inside += 1
    if fourth > 0.0:
        inside += 1
    if inside == 2:
        return 2
    if inside == 1 or inside == 3:
        return 1
    return 0


@export("mm_march_count")
def mm_march_count(
    field_address: Int, nx: Int, ny: Int, nz: Int
) abi("C") -> Int:
    # The six tetrahedra of a cell all share the corner 0-7 diagonal, so the
    # eight field samples are loaded once per cell rather than four times per
    # tetrahedron, and the corner-to-tetrahedron table is a compile-time list
    # of call sites instead of a branchy lookup.
    var field = fp(field_address)
    var plane_stride = nx * ny
    var count = 0
    for z in range(nz - 1):
        for y in range(ny - 1):
            var row = z * plane_stride + y * nx
            for x in range(nx - 1):
                var base = row + x
                var v0 = field.unsafe_load(base)
                var v1 = field.unsafe_load(base + 1)
                var v2 = field.unsafe_load(base + nx)
                var v3 = field.unsafe_load(base + nx + 1)
                var v4 = field.unsafe_load(base + plane_stride)
                var v5 = field.unsafe_load(base + plane_stride + 1)
                var v6 = field.unsafe_load(base + plane_stride + nx)
                var v7 = field.unsafe_load(base + plane_stride + nx + 1)
                count += tet_triangle_count(v0, v1, v3, v7)
                count += tet_triangle_count(v0, v3, v2, v7)
                count += tet_triangle_count(v0, v2, v6, v7)
                count += tet_triangle_count(v0, v6, v4, v7)
                count += tet_triangle_count(v0, v4, v5, v7)
                count += tet_triangle_count(v0, v5, v1, v7)
    return count


@always_inline
def crossing(
    values: Array[Float64, 4],
    coordinates: Array[Float64, 12],
    node_ids: Array[Int, 4],
    first: Int,
    second: Int,
    mut points: Array[Float64, 12],
    mut edge_keys: Array[Int64, 4],
    slot: Int,
    grid_node_count: Int,
):
    var a = first
    var b = second
    if node_ids[a] > node_ids[b]:
        a = second
        b = first
    var denominator = values[a] - values[b]
    var amount = values[a] / denominator
    edge_keys[slot] = (
        Int64(node_ids[a]) * Int64(grid_node_count) + Int64(node_ids[b])
    )
    for axis in range(3):
        points[slot * 3 + axis] = (
            coordinates[a * 3 + axis]
            + amount
            * (coordinates[b * 3 + axis] - coordinates[a * 3 + axis])
        )


@always_inline
def write_triangle(
    destination: FPtr,
    destination_keys: IPtr,
    triangle: Int,
    points: Array[Float64, 12],
    edge_keys: Array[Int64, 4],
    a: Int,
    b: Int,
    c: Int,
    direction_x: Float64,
    direction_y: Float64,
    direction_z: Float64,
):
    var abx = points[b * 3] - points[a * 3]
    var aby = points[b * 3 + 1] - points[a * 3 + 1]
    var abz = points[b * 3 + 2] - points[a * 3 + 2]
    var acx = points[c * 3] - points[a * 3]
    var acy = points[c * 3 + 1] - points[a * 3 + 1]
    var acz = points[c * 3 + 2] - points[a * 3 + 2]
    var nx = aby * acz - abz * acy
    var ny = abz * acx - abx * acz
    var nz = abx * acy - aby * acx
    var flip = nx * direction_x + ny * direction_y + nz * direction_z < 0.0
    var second = c if flip else b
    var third = b if flip else c
    for axis in range(3):
        destination.unsafe_store(triangle * 9 + axis, points[a * 3 + axis])
        destination.unsafe_store(
            triangle * 9 + 3 + axis, points[second * 3 + axis]
        )
        destination.unsafe_store(
            triangle * 9 + 6 + axis, points[third * 3 + axis]
        )
    destination_keys.unsafe_store(triangle * 3, edge_keys[a])
    destination_keys.unsafe_store(triangle * 3 + 1, edge_keys[second])
    destination_keys.unsafe_store(triangle * 3 + 2, edge_keys[third])


@always_inline
def emit_tetrahedron(
    destination: FPtr,
    destination_keys: IPtr,
    triangle: Int,
    base: Int,
    x_stride: Int,
    plane_stride: Int,
    x: Int,
    y: Int,
    z: Int,
    ox: Float64,
    oy: Float64,
    oz: Float64,
    spacing: Float64,
    grid_node_count: Int,
    first_corner: Int,
    second_corner: Int,
    third_corner: Int,
    fourth_corner: Int,
    first_value: Float64,
    second_value: Float64,
    third_value: Float64,
    fourth_value: Float64,
) -> Int:
    var values = Array[Float64, 4](
        first_value, second_value, third_value, fourth_value,
        __list_literal__=None,
    )
    var corners = Array[Int, 4](
        first_corner, second_corner, third_corner, fourth_corner,
        __list_literal__=None,
    )
    var node_ids = Array[Int, 4](0, 0, 0, 0, __list_literal__=None)
    var inside = Array[Int, 4](0, 0, 0, 0, __list_literal__=None)
    var outside = Array[Int, 4](0, 0, 0, 0, __list_literal__=None)
    var inside_count = 0
    var outside_count = 0
    for corner in range(4):
        if values[corner] > 0.0:
            inside[inside_count] = corner
            inside_count += 1
        else:
            outside[outside_count] = corner
            outside_count += 1
    if inside_count == 0 or inside_count == 4:
        return triangle

    # Only the straddling tetrahedra build node ids and coordinates, so the
    # interior and exterior cells of the grid cost four comparisons each.
    var coordinates = Array[Float64, 12](fill=0.0)
    for corner in range(4):
        var node = corners[corner]
        node_ids[corner] = base + corner_offset(node, x_stride, plane_stride)
        coordinates[corner * 3] = ox + spacing * Float64(x + (node & 1))
        coordinates[corner * 3 + 1] = oy + spacing * Float64(
            y + ((node >> 1) & 1)
        )
        coordinates[corner * 3 + 2] = oz + spacing * Float64(z + (node >> 2))

    var direction_x = 0.0
    var direction_y = 0.0
    var direction_z = 0.0
    for i in range(outside_count):
        direction_x += coordinates[outside[i] * 3] / Float64(outside_count)
        direction_y += coordinates[outside[i] * 3 + 1] / Float64(outside_count)
        direction_z += coordinates[outside[i] * 3 + 2] / Float64(outside_count)
    for i in range(inside_count):
        direction_x -= coordinates[inside[i] * 3] / Float64(inside_count)
        direction_y -= coordinates[inside[i] * 3 + 1] / Float64(inside_count)
        direction_z -= coordinates[inside[i] * 3 + 2] / Float64(inside_count)

    var points = Array[Float64, 12](fill=0.0)
    var edge_keys = Array[Int64, 4](fill=0)
    if inside_count == 1:
        crossing(values, coordinates, node_ids, inside[0], outside[0], points, edge_keys, 0, grid_node_count)
        crossing(values, coordinates, node_ids, inside[0], outside[1], points, edge_keys, 1, grid_node_count)
        crossing(values, coordinates, node_ids, inside[0], outside[2], points, edge_keys, 2, grid_node_count)
        write_triangle(
            destination, destination_keys, triangle, points, edge_keys, 0, 1, 2,
            direction_x, direction_y, direction_z
        )
        return triangle + 1
    if inside_count == 3:
        crossing(values, coordinates, node_ids, outside[0], inside[0], points, edge_keys, 0, grid_node_count)
        crossing(values, coordinates, node_ids, outside[0], inside[1], points, edge_keys, 1, grid_node_count)
        crossing(values, coordinates, node_ids, outside[0], inside[2], points, edge_keys, 2, grid_node_count)
        write_triangle(
            destination, destination_keys, triangle, points, edge_keys, 0, 1, 2,
            direction_x, direction_y, direction_z
        )
        return triangle + 1

    crossing(values, coordinates, node_ids, inside[0], outside[0], points, edge_keys, 0, grid_node_count)
    crossing(values, coordinates, node_ids, inside[0], outside[1], points, edge_keys, 1, grid_node_count)
    crossing(values, coordinates, node_ids, inside[1], outside[1], points, edge_keys, 2, grid_node_count)
    crossing(values, coordinates, node_ids, inside[1], outside[0], points, edge_keys, 3, grid_node_count)
    write_triangle(
        destination, destination_keys, triangle, points, edge_keys, 0, 1, 2,
        direction_x, direction_y, direction_z
    )
    write_triangle(
        destination, destination_keys, triangle + 1, points, edge_keys, 0, 2, 3,
        direction_x, direction_y, direction_z
    )
    return triangle + 2


@export("mm_march_emit")
def mm_march_emit(
    field_address: Int,
    destination_address: Int,
    destination_keys_address: Int,
    nx: Int,
    ny: Int,
    nz: Int,
    ox: Float64,
    oy: Float64,
    oz: Float64,
    spacing: Float64,
) abi("C") -> Int:
    var field = fp(field_address)
    var destination = fp(destination_address)
    var destination_keys = ip(destination_keys_address)
    var plane_stride = nx * ny
    var grid_node_count = plane_stride * nz
    var triangle = 0
    for z in range(nz - 1):
        for y in range(ny - 1):
            var row = z * plane_stride + y * nx
            for x in range(nx - 1):
                var base = row + x
                var v0 = field.unsafe_load(base)
                var v1 = field.unsafe_load(base + 1)
                var v2 = field.unsafe_load(base + nx)
                var v3 = field.unsafe_load(base + nx + 1)
                var v4 = field.unsafe_load(base + plane_stride)
                var v5 = field.unsafe_load(base + plane_stride + 1)
                var v6 = field.unsafe_load(base + plane_stride + nx)
                var v7 = field.unsafe_load(base + plane_stride + nx + 1)
                triangle = emit_tetrahedron(
                    destination, destination_keys, triangle, base, nx,
                    plane_stride, x, y, z, ox, oy, oz, spacing,
                    grid_node_count, 0, 1, 3, 7, v0, v1, v3, v7,
                )
                triangle = emit_tetrahedron(
                    destination, destination_keys, triangle, base, nx,
                    plane_stride, x, y, z, ox, oy, oz, spacing,
                    grid_node_count, 0, 3, 2, 7, v0, v3, v2, v7,
                )
                triangle = emit_tetrahedron(
                    destination, destination_keys, triangle, base, nx,
                    plane_stride, x, y, z, ox, oy, oz, spacing,
                    grid_node_count, 0, 2, 6, 7, v0, v2, v6, v7,
                )
                triangle = emit_tetrahedron(
                    destination, destination_keys, triangle, base, nx,
                    plane_stride, x, y, z, ox, oy, oz, spacing,
                    grid_node_count, 0, 6, 4, 7, v0, v6, v4, v7,
                )
                triangle = emit_tetrahedron(
                    destination, destination_keys, triangle, base, nx,
                    plane_stride, x, y, z, ox, oy, oz, spacing,
                    grid_node_count, 0, 4, 5, 7, v0, v4, v5, v7,
                )
                triangle = emit_tetrahedron(
                    destination, destination_keys, triangle, base, nx,
                    plane_stride, x, y, z, ox, oy, oz, spacing,
                    grid_node_count, 0, 5, 1, 7, v0, v5, v1, v7,
                )
    return triangle


@export("mm_transform")
def mm_transform(
    source_address: Int,
    destination_address: Int,
    matrix_address: Int,
    vertex_count: Int,
) abi("C"):
    var source = fp(source_address)
    var destination = fp(destination_address)
    var matrix = fp(matrix_address)
    for i in range(vertex_count):
        var x = source.unsafe_load(i * 3)
        var y = source.unsafe_load(i * 3 + 1)
        var z = source.unsafe_load(i * 3 + 2)
        destination.unsafe_store(
            i * 3,
            matrix.unsafe_load(0) * x
            + matrix.unsafe_load(1) * y
            + matrix.unsafe_load(2) * z
            + matrix.unsafe_load(3),
        )
        destination.unsafe_store(
            i * 3 + 1,
            matrix.unsafe_load(4) * x
            + matrix.unsafe_load(5) * y
            + matrix.unsafe_load(6) * z
            + matrix.unsafe_load(7),
        )
        destination.unsafe_store(
            i * 3 + 2,
            matrix.unsafe_load(8) * x
            + matrix.unsafe_load(9) * y
            + matrix.unsafe_load(10) * z
            + matrix.unsafe_load(11),
        )


@export("mm_pairwise_sum")
def mm_pairwise_sum(
    first_address: Int,
    first_count: Int,
    second_address: Int,
    second_count: Int,
    destination_address: Int,
    negate_second: Int,
) abi("C"):
    var first = fp(first_address)
    var second = fp(second_address)
    var destination = fp(destination_address)
    var sign = -1.0 if negate_second != 0 else 1.0
    for i in range(first_count):
        for j in range(second_count):
            var base = (i * second_count + j) * 3
            destination.unsafe_store(
                base, first.unsafe_load(i * 3) + sign * second.unsafe_load(j * 3)
            )
            destination.unsafe_store(
                base + 1,
                first.unsafe_load(i * 3 + 1)
                + sign * second.unsafe_load(j * 3 + 1),
            )
            destination.unsafe_store(
                base + 2,
                first.unsafe_load(i * 3 + 2)
                + sign * second.unsafe_load(j * 3 + 2),
            )


@export("mm_measure")
def mm_measure(
    vertices_address: Int,
    faces_address: Int,
    triangle_count: Int,
    result_address: Int,
) abi("C"):
    var vertices = fp(vertices_address)
    var faces = ip(faces_address)
    var result = fp(result_address)
    var signed_volume = 0.0
    var area = 0.0
    for triangle in range(triangle_count):
        var ia = Int(faces.unsafe_load(triangle * 3)) * 3
        var ib = Int(faces.unsafe_load(triangle * 3 + 1)) * 3
        var ic = Int(faces.unsafe_load(triangle * 3 + 2)) * 3
        var ax = vertices.unsafe_load(ia)
        var ay = vertices.unsafe_load(ia + 1)
        var az = vertices.unsafe_load(ia + 2)
        var bx = vertices.unsafe_load(ib)
        var by = vertices.unsafe_load(ib + 1)
        var bz = vertices.unsafe_load(ib + 2)
        var cx = vertices.unsafe_load(ic)
        var cy = vertices.unsafe_load(ic + 1)
        var cz = vertices.unsafe_load(ic + 2)
        var cross_x = by * cz - bz * cy
        var cross_y = bz * cx - bx * cz
        var cross_z = bx * cy - by * cx
        signed_volume += ax * cross_x + ay * cross_y + az * cross_z
        var abx = bx - ax
        var aby = by - ay
        var abz = bz - az
        var acx = cx - ax
        var acy = cy - ay
        var acz = cz - az
        var normal_x = aby * acz - abz * acy
        var normal_y = abz * acx - abx * acz
        var normal_z = abx * acy - aby * acx
        area += 0.5 * sqrt(
            normal_x * normal_x
            + normal_y * normal_y
            + normal_z * normal_z
        )
    result.unsafe_store(0, abs(signed_volume) / 6.0)
    result.unsafe_store(1, area)



