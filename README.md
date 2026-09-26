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
consistently oriented marching-tetrahedra triangles. Primitive and hull meshes
remain `float64` internally; `Manifold.to_mesh()` uses upstream-compatible
interleaved `float32` vertex properties and `int32` triangle indices.

### Vectorised signed-distance kernel

`mm_signed_distance` walks one grid row at a time and, for each chunk of
`W = simd_width_of[DType.float64]()` consecutive x samples, loops over the
triangles once. Three properties of that loop order carry the speed:

- the triangle is loaded and reduced once per chunk instead of once per
  (point, triangle) pair, so the vertex fetches and edge setup are amortised
  over `W` points;
- `tri_distance2_simd` is a branch-free restatement of the scalar
  point-triangle distance. The seven Voronoi regions are disjoint, so selecting
  them in the reverse of the scalar early-exit order reproduces the scalar
  result exactly, and the ray test loses its early returns the same way;
- the three edge regions of a triangle divide by constants, not by
  point-dependent values: `d1 - d3` is `|ab|^2`, `d2 - d6` is `|ac|^2` and
  `(d4 - d3) + (d5 - d6)` is `|bc|^2`. Their reciprocals are hoisted into the
  triangle setup, which leaves one division per point-triangle pair in the face
  region instead of four.

The same algebra replaces the four remaining dot products per pair
(`d3 = d1 - |ab|^2`, `d4 = d2 - ab.ac`, `d5 = d1 - ab.ac`, `d6 = d2 - |ac|^2`),
and the second and fourth vertex-region distances are recovered from the
first. On the benchmark's sphere grid the largest absolute difference against
the previous scalar kernel is 3.7e-16, against field values up to 1.4, which
is at the level the parity tests already tolerate.

A row whose length is not a multiple of `W` is covered by re-running the final
full-width chunk, which recomputes and rewrites the same values, and a grid
narrower than one vector falls back to the scalar `sdf_point` path. Both are
covered by `test_signed_distance_matches_reference_for_both_widths`, which
checks the kernel against an independent point-triangle and ray-parity
reference for grids of 2, 5, 7 and 8 samples across.

### Weld and topology check

Crossings are welded by cell edge. A marching-tetrahedra cell can only cross
13 distinct edges, all inside one cell, so the pair of grid nodes that encodes
an edge collapses to a dense `14 * low + gap_class` slot and the weld is a
scatter into a table rather than a sort of three times the triangle count.
Grids too small to keep those 13 node gaps distinct fall back to `np.unique`.

The oriented 2-manifold test packs each directed edge as
`2 * undirected_key + orientation` and sorts once. A closed oriented manifold
then has exactly two entries per group, differing only in the low bit, which
one pass over the sorted array checks without building edge keys, counts, or a
weighted `bincount`.

### Single-threaded CPU kernels

The CPU kernels run on one thread. They previously fanned the grid out over
`std.algorithm.parallelize`, but that function no longer exists in the
pinned toolchain: importing it fails with
`package 'algorithm' does not contain 'parallelize'`, and the toolchain
ships no replacement — there is no `std.parallelism` or `std.threading`
module. No replacement was found in the `max` package, so the grid is walked in
a single loop. This port is therefore CPU-serial by toolchain, not by choice,
and the boolean cases below pay for it against upstream's thread pool.

The signed-distance field is the only kernel with enough arithmetic intensity
to justify GPU execution. `Manifold.boolean(..., device="gpu")` checks for at
least 4,000 MiB of free NVIDIA memory before every request, caps total device
buffers below 2 GB, and uses the CPU with a `RuntimeWarning` if the GPU is
absent, busy, or fails. Device buffers are released when the call completes.
Boolean operators remain CPU-only so GPU use is always explicit. One kernel
launch covers the whole grid, the mesh and the field are each copied once, and
the tail is covered by the same per-thread bounds check as the body.

### Alternatives that did not ship

Measured on the interleaved kernel harness and reverted:

- vectors of 8 and 16 `float64` lanes: the wider chunk needs just as many live
  vectors, and the overlap the 33-sample rows need costs more than the wider
  triangle amortisation returns (56 ms at `W = 4`, 60 ms at 8, 77 ms at 16);
- splitting the ray-parity pass into its own triangle loop, to relieve register
  pressure: 61 ms against 58 ms for the fused loop.

Rejected without shipping, because the numbers would not have been honest:

- a packed per-triangle constant block in a scratch buffer: the width
  experiment above already showed triangle setup is not the binding
  constraint, so it was not worth the added state;
- replacing the divisions in the marching-tetrahedra outward normal with
  multiplication by a reciprocal: `1/3` is inexact, so it would change a
  triangle's winding decision in the last bit, and the topology guarantee is
  worth more than the divisions.

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux 6.8.0-139-generic. Each number is the best complete Python API call from
the benchmark's repeat count. The "before" column is the same table measured on
the same idle host from the pre-optimisation tree, rebuilt from source and run
back to back with the "now" column. This is a shared machine and repeated runs
move by up to 20% in both directions, so the table is one run rather than a
best-of, and the ratios below are from those two runs.

| case | before | mojo-manifold3d | manifold3d 3.5.2 | relative |
|---|---:|---:|---:|---:|
| volume + surface area (2,048 tri) | 0.074 ms | 0.001 ms | 0.073 ms | 73.861x faster |
| affine transform + mesh (1,026 vert) | 0.152 ms | 0.132 ms | 0.294 ms | 2.234x faster |
| convex hull (10,000 points) | 4.318 ms | 4.720 ms | 0.704 ms | 0.149x slower |
| Minkowski sum (66 x 8 vertices) | 6.239 ms | 3.427 ms | 2.974 ms | 0.868x slower |
| box union (resolution 28) | 83.907 ms | 42.040 ms | 0.330 ms | 0.008x slower |
| sphere-box difference (resolution 28) | 203.509 ms | 95.817 ms | 0.632 ms | 0.007x slower |

- The two boolean rows are 2.0x and 2.1x faster than this port's own previous
  build. They stay far behind upstream because upstream is a parallel C++
  exact-mesh engine and this port evaluates a dense signed-distance grid on one
  thread to get a simple, explicit manifold guarantee.
- The convex hull row is Qhull. `scipy.spatial.ConvexHull` accounts for
  essentially all of it, nothing in this repository runs on that path, and the
  two columns differ only by run-to-run noise: replacing Qhull with a Mojo
  quickhull is a different project, not an optimisation.
- The Minkowski row is 1.8x faster than before and now at parity with upstream,
  where it straddles between runs (0.8x slower to 1.4x faster). It is
  dominated by two Qhull calls, one for the sphere and one for the expanded
  point cloud.
- The affine transform row is untouched by this work and its two columns
  differ only by noise; it was already ahead of upstream.
- `volume()` and `surface_area()` on the same solid share one measurement pass.
  The buffers they read never change after construction, so the result is
  cached on the instance. A single cold `volume()` call still costs about
  0.05 ms; the benchmark measures the pair, which is what a caller asks for.

The one GPU benchmark is deliberately limited to a small, fixed input rather
than a sweep. The device path is unchanged by this work, so the smaller
acceleration is the CPU side catching up: the same boolean that took 508.8 ms
on the CPU before now takes 236.9 ms.

| GPU case | CPU before | CPU now | GPU | acceleration |
|---|---:|---:|---:|---:|
| sphere-box difference (resolution 40) | 508.768 ms | 236.943 ms | 128.463 ms | 1.844x |

Run `pixi run bench` to reproduce the table under the repository's
machine-wide benchmark lock.

## License

MIT
