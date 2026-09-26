# mojo-manifold3d

`mojo-manifold3d` is a standalone mesh-solid library with its compute-heavy
CSG kernels written in [Mojo](https://www.modular.com/mojo). Its Python API
mirrors the covered part of the production
[`manifold3d`](https://pypi.org/project/manifold3d/) package: `Mesh`,
`Manifold`, `Error`, `OpType`, primitive constructors, boolean operators,
convex hulls, and Minkowski operations.

The design goal is a useful correctness-first port, not a thin wrapper around
the upstream C++ extension. The test environment installs `manifold3d` 3.5.2
only as a parity and benchmark reference. Runtime calls made through
`mojomanifold3d` do not call it.

## Coverage

Implemented:

| area | covered API |
|---|---|
| mesh input | positional `Mesh`, position/index validation, oriented 2-manifold validation, matching `Error` enum values |
| primitives | `cube`, `tetrahedron`, `cylinder`, `sphere` |
| boolean CSG | `+` union, `-` difference, `^` intersection, `batch_boolean`, `compose` |
| convex geometry | `hull`, `hull_points`, `batch_hull` |
| Minkowski | `minkowski_sum` and `minkowski_difference` for convex solids |
| transforms | `translate`, `rotate`, `scale`, `transform`, `mirror`, `warp`, `warp_batch` |
| queries | `status`, `is_empty`, `to_mesh`, `num_vert`, `num_prop_vert`, `num_tri`, `num_edge`, `num_prop`, `bounding_box`, `volume`, `surface_area`, `genus` |

Boolean CSG accepts arbitrary closed, consistently wound triangle meshes. It
samples signed distance on an adaptive grid and extracts the zero set with a
consistent six-tetrahedra subdivision. The result is geometrically
resolution-dependent, unlike upstream's surface-preserving exact boolean.
The default is 40 cells across the longest result axis; call
`set_csg_resolution(n)` to trade memory and time for accuracy. Every result is
welded and checked for the oriented 2-manifold invariant: each undirected edge
must occur exactly twice, once in each direction.

Not covered are arbitrary vertex-property propagation (extra `Mesh` metadata
is retained but not used by geometry operations), `Mesh64`,
`CrossSection`, extrusion/revolution, smoothing/refinement, slicing,
curvature, ray casting, and non-convex Minkowski operations. Convex hull
topology uses SciPy's Qhull implementation; Mojo supplies the pairwise
Minkowski expansion, transforms, measurements, signed-distance evaluation,
field algebra, and marching tetrahedra. Coplanar `hull_points` returns empty,
following upstream's documented contract; upstream 3.5.2 currently returns a
zero-volume shell in that case.

## Install

```bash
pixi install
pixi run build
pixi run test
```

The build produces `dist/libmojo-manifold3d.so`. Pixi adds `python/` to
`PYTHONPATH`; outside Pixi, install the Python package and set
`MOJOMANIFOLD3D_LIB` to the shared-library path.

## Usage

```python
import mojomanifold3d as m3d

body = m3d.Manifold.cube((3, 2, 1), center=True)
tool = m3d.Manifold.cylinder(
    height=2,
    radius_low=0.45,
    circular_segments=32,
    center=True,
).rotate((90, 0, 0))

cut = body - tool
assert cut.status() is m3d.Error.NoError
assert cut.genus() == 1

# Boolean operators use the CPU. Request the optional GPU path explicitly.
cut_gpu = body.boolean(tool, m3d.OpType.Subtract, device="gpu")

mesh = cut.to_mesh()
print(mesh.vert_properties.shape, mesh.tri_verts.shape, cut.volume())
```

For the API listed in the coverage table, migration is normally an import
change from `manifold3d` to `mojomanifold3d`. Code using other upstream API
will require changes.

## How it works

Python validates shapes and owns every allocation. C-contiguous `float64`
vertex buffers and `int64` triangle-index buffers cross one `ctypes` call as
integer addresses. The exported Mojo functions reconstruct
`Pointer[..., AnyOrigin[mut=True]]` values inside non-parametric
`@export` functions. Before each synchronous call, the Python layer checks
dtype, contiguity, alignment, non-empty/non-null buffers, and output
writability. Python keeps references to all NumPy buffers for the duration of
the call; the shared library never retains a pointer and never allocates
Python-visible memory.

For boolean CSG, Mojo computes unsigned point-to-triangle distance and
inside/outside ray parity at every grid point. Union, difference, and
intersection are `max(a, b)`, `min(a, -b)`, and `min(a, b)` on positive-inside
fields. A count pass sizes the output exactly, then a second pass emits
consistently oriented marching-tetrahedra triangles. Python welds crossings by
exact grid-edge ID and validates topology before constructing the result.
Primitive and hull meshes remain `float64` internally; `Manifold.to_mesh()`
uses upstream-compatible interleaved `float32` vertex properties and `int32`
triangle indices.

### Single-threaded CPU kernels

The CPU kernels run on one thread. They previously fanned the grid out over
`std.algorithm.parallelize`, but that function no longer exists in the
pinned toolchain: importing it fails with
`package 'algorithm' does not contain 'parallelize'`, and the toolchain
ships no replacement — there is no `std.parallelism` or `std.threading`
module. Rather than invent a task API that the runtime does not provide, the
signed-distance and field-algebra kernels walk their whole range in a single
loop, and `mm_combine_fields` keeps its vectorised body and scalar tail.

The practical cost is on the boolean path, which evaluates a dense grid: the
measured boolean cases below are about 3x slower than the previous parallel
build, and the tables in this section reflect the serial build. The GPU path
is unaffected and remains available through `device="gpu"`, where the grid
is genuinely spread across device threads.

The signed-distance field is the only kernel with enough arithmetic intensity
to justify GPU execution. `Manifold.boolean(..., device="gpu")` checks for at
least 4,000 MiB of free NVIDIA memory before every request, caps total device
buffers below 2 GB, and uses the CPU with a `RuntimeWarning` if the GPU is
absent, busy, or fails. Device buffers are released when the call completes.
Boolean operators remain CPU-only so GPU use is always explicit.

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux 6.8.0-139-generic. Each number is the best complete Python API call from
the benchmark's repeat count. Transform is now ahead of upstream; the
remaining cases are slower. Upstream is a mature parallel C++ exact-mesh
engine, while this port's boolean path evaluates a dense signed-distance grid
on a single thread to obtain a simple, explicit manifold guarantee.

| case | mojo-manifold3d | manifold3d 3.5.2 | relative |
|---|---:|---:|---:|
| volume + surface area (2,048 tri) | 0.101 ms | 0.073 ms | 0.721x slower |
| affine transform + mesh (1,026 vert) | 0.159 ms | 0.465 ms | 2.921x faster |
| convex hull (10,000 points) | 5.066 ms | 0.753 ms | 0.149x slower |
| Minkowski sum (66 x 8 vertices) | 6.844 ms | 3.207 ms | 0.469x slower |
| box union (resolution 28) | 82.957 ms | 0.356 ms | 0.004x slower |
| sphere-box difference (resolution 28) | 368.646 ms | 0.781 ms | 0.002x slower |

The one GPU benchmark is deliberately limited to a small, fixed input rather
than a sweep. The GPU win is larger than before because the serial CPU
baseline got slower while the device path kept its parallelism.

| GPU case | CPU | GPU | acceleration |
|---|---:|---:|---:|
| sphere-box difference (resolution 40) | 1060.512 ms | 569.211 ms | 1.863x |

Run `pixi run bench` to reproduce the table under the repository's
machine-wide benchmark lock.

## License

MIT
