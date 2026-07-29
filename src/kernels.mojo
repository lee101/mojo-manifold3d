"""Geometry kernels exported through a stable C ABI.

Python owns every buffer. Addresses cross the ABI as Int so exported
functions are non-parametric under the Mojo 1.0 nightly compiler.
"""

from std.algorithm import parallelize
from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext
from std.math import sqrt
from std.sys import has_accelerator
from std.sys.info import num_physical_cores, simd_width_of as simdwidthof

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime GPUFPtr = UnsafePointer[Float64, MutAnyOrigin]
comptime GPUIPtr = UnsafePointer[Int64, MutAnyOrigin]


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
def signed_distance_chunk(
    vertices: FPtr,
    faces: IPtr,
    triangle_count: Int,
    ox: Float64,
    oy: Float64,
    oz: Float64,
    spacing: Float64,
    nx: Int,
    ny: Int,
    begin: Int,
    end: Int,
    result: FPtr,
):
    for point in range(begin, end):
        var x_index = point % nx
        var yz = point // nx
        var y_index = yz % ny
        var z_index = yz // ny
        var px = ox + spacing * Float64(x_index)
        var py = oy + spacing * Float64(y_index)
        var pz = oz + spacing * Float64(z_index)
        var best2 = 1.7976931348623157e308
        var hits = 0
        for triangle in range(triangle_count):
            var ia = Int(faces[triangle * 3]) * 3
            var ib = Int(faces[triangle * 3 + 1]) * 3
            var ic = Int(faces[triangle * 3 + 2]) * 3
            var ax = vertices[ia]
            var ay = vertices[ia + 1]
            var az = vertices[ia + 2]
            var bx = vertices[ib]
            var by = vertices[ib + 1]
            var bz = vertices[ib + 2]
            var cx = vertices[ic]
            var cy = vertices[ic + 1]
            var cz = vertices[ic + 2]
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
        result[point] = distance if hits % 2 == 1 else -distance


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
) abi("C"):
    var vertices = fp(vertices_address)
    var faces = ip(faces_address)
    var result = fp(result_address)
    var count = nx * ny * nz
    var tasks = min(num_physical_cores(), 16)
    if count < 8_192:
        tasks = 1

    @parameter
    @__copy_capture(
        vertices, faces, triangle_count, ox, oy, oz, spacing, nx, ny,
        count, result, tasks
    )
    @always_inline
    def process(task: Int):
        signed_distance_chunk(
            vertices,
            faces,
            triangle_count,
            ox,
            oy,
            oz,
            spacing,
            nx,
            ny,
            count * task // tasks,
            count * (task + 1) // tasks,
            result,
        )

    if tasks == 1:
        process(0)
    else:
        parallelize[process](tasks, tasks)


def signed_distance_gpu_kernel(
    vertices: GPUFPtr,
    faces: GPUIPtr,
    triangle_count: Int,
    ox: Float64,
    oy: Float64,
    oz: Float64,
    spacing: Float64,
    nx: Int,
    ny: Int,
    count: Int,
    result: GPUFPtr,
):
    var point = block_idx.x * block_dim.x + thread_idx.x
    if point >= count:
        return
    var x_index = point % nx
    var yz = point // nx
    var y_index = yz % ny
    var z_index = yz // ny
    var px = ox + spacing * Float64(x_index)
    var py = oy + spacing * Float64(y_index)
    var pz = oz + spacing * Float64(z_index)
    var best2 = 1.7976931348623157e308
    var hits = 0
    for triangle in range(triangle_count):
        var ia = Int(faces[triangle * 3]) * 3
        var ib = Int(faces[triangle * 3 + 1]) * 3
        var ic = Int(faces[triangle * 3 + 2]) * 3
        var ax = vertices[ia]
        var ay = vertices[ia + 1]
        var az = vertices[ia + 2]
        var bx = vertices[ib]
        var by = vertices[ib + 1]
        var bz = vertices[ib + 2]
        var cx = vertices[ic]
        var cy = vertices[ic + 1]
        var cz = vertices[ic + 2]
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
    result[point] = distance if hits % 2 == 1 else -distance


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
                    triangle_count,
                    ox,
                    oy,
                    oz,
                    spacing,
                    nx,
                    ny,
                    count,
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
                    first_triangle_count,
                    ox,
                    oy,
                    oz,
                    spacing,
                    nx,
                    ny,
                    count,
                    first_result,
                    grid_dim=(count + 255) // 256,
                    block_dim=256,
                )
                ctx.enqueue_function(
                    kernel,
                    second_vertices,
                    second_faces,
                    second_triangle_count,
                    ox,
                    oy,
                    oz,
                    spacing,
                    nx,
                    ny,
                    count,
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
    comptime W = simdwidthof[DType.float64]()

    @parameter
    @__copy_capture(first, second, result, count, operation)
    @always_inline
    def process(task: Int):
        var tasks = min(num_physical_cores(), 16) if count >= 262_144 else 1
        var begin = count * task // tasks
        var end = count * (task + 1) // tasks
        var vector_end = begin + (end - begin) // W * W
        for i in range(begin, vector_end, W):
            var a = first.load[width=W](i)
            var b = second.load[width=W](i)
            if operation == 0:
                result.store(i, max(a, b))
            elif operation == 1:
                result.store(i, min(a, -b))
            else:
                result.store(i, min(a, b))
        for i in range(vector_end, end):
            if operation == 0:
                result[i] = max(first[i], second[i])
            elif operation == 1:
                result[i] = min(first[i], -second[i])
            else:
                result[i] = min(first[i], second[i])

    var tasks = min(num_physical_cores(), 16) if count >= 262_144 else 1
    if tasks == 1:
        process(0)
    else:
        parallelize[process](tasks, tasks)


@always_inline
def tet_node(tetrahedron: Int, corner: Int) -> Int:
    if tetrahedron == 0:
        if corner == 0:
            return 0
        if corner == 1:
            return 1
        return 3 if corner == 2 else 7
    if tetrahedron == 1:
        if corner == 0:
            return 0
        if corner == 1:
            return 3
        return 2 if corner == 2 else 7
    if tetrahedron == 2:
        if corner == 0:
            return 0
        if corner == 1:
            return 2
        return 6 if corner == 2 else 7
    if tetrahedron == 3:
        if corner == 0:
            return 0
        if corner == 1:
            return 6
        return 4 if corner == 2 else 7
    if tetrahedron == 4:
        if corner == 0:
            return 0
        if corner == 1:
            return 4
        return 5 if corner == 2 else 7
    if corner == 0:
        return 0
    if corner == 1:
        return 5
    return 1 if corner == 2 else 7


@always_inline
def node_x(node: Int) -> Int:
    return node % 2


@always_inline
def node_y(node: Int) -> Int:
    return (node // 2) % 2


@always_inline
def node_z(node: Int) -> Int:
    return node // 4


@always_inline
def grid_index(x: Int, y: Int, z: Int, nx: Int, ny: Int) -> Int:
    return (z * ny + y) * nx + x


@always_inline
def tetrahedron_triangle_count(
    field: FPtr,
    x: Int,
    y: Int,
    z: Int,
    nx: Int,
    ny: Int,
    tetrahedron: Int,
) -> Int:
    var inside = 0
    for corner in range(4):
        var node = tet_node(tetrahedron, corner)
        var index = grid_index(
            x + node_x(node), y + node_y(node), z + node_z(node), nx, ny
        )
        if field[index] > 0.0:
            inside += 1
    if inside == 0 or inside == 4:
        return 0
    return 2 if inside == 2 else 1


@export("mm_march_count")
def mm_march_count(
    field_address: Int, nx: Int, ny: Int, nz: Int
) abi("C") -> Int:
    var field = fp(field_address)
    var count = 0
    for z in range(nz - 1):
        for y in range(ny - 1):
            for x in range(nx - 1):
                for tetrahedron in range(6):
                    count += tetrahedron_triangle_count(
                        field, x, y, z, nx, ny, tetrahedron
                    )
    return count


@always_inline
def crossing(
    values: InlineArray[Float64, 4],
    coordinates: InlineArray[Float64, 12],
    node_ids: InlineArray[Int, 4],
    first: Int,
    second: Int,
    mut points: InlineArray[Float64, 12],
    mut edge_keys: InlineArray[Int64, 4],
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
    points: InlineArray[Float64, 12],
    edge_keys: InlineArray[Int64, 4],
    a: Int,
    b: Int,
    c: Int,
    direction_x: Float64,
    direction_y: Float64,
    direction_z: Float64,
    reverse: Bool,
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
    if reverse:
        flip = not flip
    var second = c if flip else b
    var third = b if flip else c
    for axis in range(3):
        destination[triangle * 9 + axis] = points[a * 3 + axis]
        destination[triangle * 9 + 3 + axis] = points[second * 3 + axis]
        destination[triangle * 9 + 6 + axis] = points[third * 3 + axis]
    destination_keys[triangle * 3] = edge_keys[a]
    destination_keys[triangle * 3 + 1] = edge_keys[second]
    destination_keys[triangle * 3 + 2] = edge_keys[third]


@always_inline
def emit_tetrahedron(
    field: FPtr,
    destination: FPtr,
    destination_keys: IPtr,
    triangle: Int,
    x: Int,
    y: Int,
    z: Int,
    nx: Int,
    ny: Int,
    ox: Float64,
    oy: Float64,
    oz: Float64,
    spacing: Float64,
    tetrahedron: Int,
    grid_node_count: Int,
) -> Int:
    var values = InlineArray[Float64, 4](fill=0.0)
    var coordinates = InlineArray[Float64, 12](fill=0.0)
    var node_ids = InlineArray[Int, 4](fill=0)
    var inside = InlineArray[Int, 4](fill=0)
    var outside = InlineArray[Int, 4](fill=0)
    var inside_count = 0
    var outside_count = 0
    for corner in range(4):
        var node = tet_node(tetrahedron, corner)
        var gx = x + node_x(node)
        var gy = y + node_y(node)
        var gz = z + node_z(node)
        var index = grid_index(gx, gy, gz, nx, ny)
        node_ids[corner] = index
        values[corner] = field[index]
        coordinates[corner * 3] = ox + spacing * Float64(gx)
        coordinates[corner * 3 + 1] = oy + spacing * Float64(gy)
        coordinates[corner * 3 + 2] = oz + spacing * Float64(gz)
        if values[corner] > 0.0:
            inside[inside_count] = corner
            inside_count += 1
        else:
            outside[outside_count] = corner
            outside_count += 1
    if inside_count == 0 or inside_count == 4:
        return triangle

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

    var points = InlineArray[Float64, 12](fill=0.0)
    var edge_keys = InlineArray[Int64, 4](fill=0)
    if inside_count == 1:
        crossing(values, coordinates, node_ids, inside[0], outside[0], points, edge_keys, 0, grid_node_count)
        crossing(values, coordinates, node_ids, inside[0], outside[1], points, edge_keys, 1, grid_node_count)
        crossing(values, coordinates, node_ids, inside[0], outside[2], points, edge_keys, 2, grid_node_count)
        write_triangle(
            destination, destination_keys, triangle, points, edge_keys, 0, 1, 2,
            direction_x, direction_y, direction_z, False
        )
        return triangle + 1
    if inside_count == 3:
        crossing(values, coordinates, node_ids, outside[0], inside[0], points, edge_keys, 0, grid_node_count)
        crossing(values, coordinates, node_ids, outside[0], inside[1], points, edge_keys, 1, grid_node_count)
        crossing(values, coordinates, node_ids, outside[0], inside[2], points, edge_keys, 2, grid_node_count)
        write_triangle(
            destination, destination_keys, triangle, points, edge_keys, 0, 1, 2,
            direction_x, direction_y, direction_z, False
        )
        return triangle + 1

    crossing(values, coordinates, node_ids, inside[0], outside[0], points, edge_keys, 0, grid_node_count)
    crossing(values, coordinates, node_ids, inside[0], outside[1], points, edge_keys, 1, grid_node_count)
    crossing(values, coordinates, node_ids, inside[1], outside[1], points, edge_keys, 2, grid_node_count)
    crossing(values, coordinates, node_ids, inside[1], outside[0], points, edge_keys, 3, grid_node_count)
    write_triangle(
        destination, destination_keys, triangle, points, edge_keys, 0, 1, 2,
        direction_x, direction_y, direction_z, False
    )
    write_triangle(
        destination, destination_keys, triangle + 1, points, edge_keys, 0, 2, 3,
        direction_x, direction_y, direction_z, False
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
    var triangle = 0
    for z in range(nz - 1):
        for y in range(ny - 1):
            for x in range(nx - 1):
                for tetrahedron in range(6):
                    triangle = emit_tetrahedron(
                        field, destination, destination_keys, triangle, x, y, z, nx, ny,
                        ox, oy, oz, spacing, tetrahedron, nx * ny * nz
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
        var x = source[i * 3]
        var y = source[i * 3 + 1]
        var z = source[i * 3 + 2]
        destination[i * 3] = (
            matrix[0] * x + matrix[1] * y + matrix[2] * z + matrix[3]
        )
        destination[i * 3 + 1] = (
            matrix[4] * x + matrix[5] * y + matrix[6] * z + matrix[7]
        )
        destination[i * 3 + 2] = (
            matrix[8] * x + matrix[9] * y + matrix[10] * z + matrix[11]
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
            destination[base] = first[i * 3] + sign * second[j * 3]
            destination[base + 1] = first[i * 3 + 1] + sign * second[j * 3 + 1]
            destination[base + 2] = first[i * 3 + 2] + sign * second[j * 3 + 2]


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
        var ia = Int(faces[triangle * 3]) * 3
        var ib = Int(faces[triangle * 3 + 1]) * 3
        var ic = Int(faces[triangle * 3 + 2]) * 3
        var ax = vertices[ia]
        var ay = vertices[ia + 1]
        var az = vertices[ia + 2]
        var bx = vertices[ib]
        var by = vertices[ib + 1]
        var bz = vertices[ib + 2]
        var cx = vertices[ic]
        var cy = vertices[ic + 1]
        var cz = vertices[ic + 2]
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
    result[0] = abs(signed_volume) / 6.0
    result[1] = area
