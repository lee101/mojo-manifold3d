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

The compute-heavy signed-distance kernel runs in Mojo, is vectorised, and fans
its grid out over up to 4 workers above a measured size threshold. Below that
threshold it is serial, because the worker start-up costs more than the work.
The threshold and the measurements behind it are in
[CPU parallelism behind a measured threshold](#cpu-parallelism-behind-a-measured-threshold).

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

# Boolean operators use the CPU. The optional GPU field path is opt-in and only
# engages at 1,000,000 grid points or more; below that it warns and uses the CPU.
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

### CPU parallelism behind a measured threshold

`mm_signed_distance` and the pair form `mm_signed_distance_pair` fan the grid
out over `max.algorithm.parallelize` above a size threshold and stay serial
below it. The row loop is the same either way: on the serial path the caller's
thread runs the same task bodies, so no worker is launched at all.

The toolchain's `parallelize` lives in `max.algorithm`; the older
`std.algorithm` and `std.threading` forms do not compile on the pinned 1.2.0
toolchain. The worker closure must be a plain nested `def` with an `{imm}`
capture convention, which is what `parallelize` accepts as
`def(Int) -> None`.

**Threshold.** Measured, not guessed. The cost that decides it is the worker
pool start-up: for grids that are pure start-up (512 to 32,768 points) the
parallel cells in this session's sweep took 4.66-7.38 ms, so the start-up is
about 4.6-4.8 ms, and it is independent of the grid. Serial therefore wins
up to roughly the point where one thread's work is itself on the order of
several milliseconds. On this host that is 50,000-110,592 points for this
kernel (table below). With a 16x margin for a busier host and a
smaller-mesh caller, the threshold is **1,000,000 grid points**. Below it the
call is serial; every boolean at the shipped default resolution is below it.

**Workers.** Capped at 4. On the raw kernel, 4 workers give 2.6-3.9x over one
thread from 128,000 points upward. Eight workers measured 1.5-2x faster again
at 1-4 million points, but the host is shared, the gain per added worker
flattens, and the cap is 4. The column is recorded here rather than taken.

| points | serial | 2 workers | 4 workers | 4 / 1 |
|---:|---:|---:|---:|---:|
| 512 | 0.029 ms | 4.742 ms | 4.659 ms | 0.01x |
| 8,000 | 0.451 ms | 5.241 ms | 4.949 ms | 0.09x |
| 32,768 | 1.851 ms | 6.965 ms | 7.379 ms | 0.25x |
| 64,000 | 3.708 ms | 7.188 ms | 5.879 ms | 0.63x |
| 110,592 | 6.196 ms | 7.917 ms | 5.975 ms | 1.04x |
| 262,144 | 15.583 ms | 13.546 ms | 8.955 ms | 1.74x |
| 884,736 | 50.587 ms | 37.629 ms | 25.132 ms | 2.01x |
| 2,097,152 | 119.459 ms | 71.172 ms | 38.665 ms | 3.09x |
| 7,077,888 | 419.804 ms | 255.658 ms | 136.094 ms | 3.08x |

`bench/parallel_crossover.cc` reproduces the table: it calls the exported
kernel directly from C++, so no interpreter runs between launches. Every cell
is the best of 3-15 calls. The crossover is where the 4-worker column overtakes
the serial column, 64,000-110,592 points; this run's 32,768-point 4-worker cell
(7.379 ms) was slower than its 2-worker cell (6.965 ms), so that row is noise in
the start-up cost rather than a trend. Two workers only match serial near the
crossover because they pay the same start-up.


The grid is split into a task count that depends only on the grid (`row_tasks`:
one task per row down to 16, capped at 4096), never on the worker count. Each
task owns a fixed row range and writes only its own samples, so every worker
configuration produces a bit-identical grid; `tests/test_parallel.py` checks
that at 1, 2 and 4 workers on several shapes and end to end on a boolean.

The boolean path uses the pair form so one launch evaluates both meshes rather
than two, and a solid unioned with itself (or two handles to one mesh) is
detected and evaluated once.

**What it does for real callers.** Every boolean at the default resolution 40
sits in the 36,000-91,000 point range, below the threshold, so it takes the
serial path and its timing is unchanged. The parallel path only engages for
callers that raise the resolution, and those are the ones where it pays. Same
build, same host, measured with `mm_signed_distance_data` at 126 samples across
(2,000,376 points), best of 3: 896.4 ms with one worker to 253.3 ms at 4, a
3.54x gain. The table row at the same size in this session's C++ sweep was
3.09x; the two harnesses share a start-up cost of the same order, so the gap is
the shared host.

### GPU path behind a measured size gate

The signed-distance field is the only kernel in this port with enough arithmetic
intensity to make a device transfer worth considering; every other kernel is
memory-bound or far too small, so it is **CPU only: memory-bound**. The GPU path
is one kernel launch per call over the whole grid, with the mesh and the field
copied in once and the field copied out once.

It is also the one path here whose advantage depends on the toolchain's device
start-up. Measured on the host below with the exported pair kernel called
directly (no interpreter between launches), same geometry as the benchmark's
sphere-box difference, best of three calls after one warm-up call:

| resolution | grid | points | CPU | GPU | GPU / CPU |
|---:|---|---:|---:|---:|---:|
| 28 | 33 x 33 x 33 | 35,937 | 44.4 ms | 2.9 ms | 0.06x |
| 40 | 45 x 45 x 45 | 91,125 | 131.1 ms | 3.8 ms | 0.03x |
| 60 | 65 x 64 x 64 | 266,240 | 315.7 ms | 4.1 ms | 0.01x |
| 80 | 85 x 84 x 84 | 599,760 | 706.3 ms | 16.7 ms | 0.02x |

(GPU / CPU is GPU time over CPU time, so a smaller figure is a larger GPU
gain.) Arrays agreed with the CPU path to 7.0e-15 and less on every row. The
one slow GPU measurement in this session was the first device call in a fresh
process, 337.1 ms at 33^3 before any warm-up; every later call at that size was
3 ms or less, so that is device start-up, not a per-call cost.

The shipped boolean resolutions are 28 and 40, where the kernel is 0.06x and
0.03x of the serial CPU time. Even so the explicit GPU request is left opt-in
and gated at **`GPU_MIN_POINTS = 1_000_000`**, because the gate is set from
the serial/parallel crossover rather than from this table, it predates these
measurements, and the whole-boolean cost is dominated by passes the device
does not touch (see the benchmarks section). Below the gate
`Manifold.boolean(..., device="gpu")` runs the CPU path and warns why. Plain
boolean operators never request the device, so the default path is unchanged.
Whether the gate should be removed is a decision for the maintainers, not a
measurement; this section records what the kernel does.

The GPU path keeps the toolchain's `max.gpu.host.DeviceContext` with
`block_idx`/`thread_idx` from `max.gpu`. The grid is `(count + 255) // 256`
blocks of 256 threads, so the last block is partly out of range and the
kernel's own `point >= count` bounds check masks those threads; grids of 3 x 5
x 7, 17 x 13 x 11 and 33 x 33 x 33 (105, 2,431 and 35,937 points, none a
multiple of 256) are checked against the CPU field in
`test_gpu_field_matches_cpu_including_tail`. A request also needs at least
4,000 MiB free, caps total device buffers below 2 GB, falls back to the CPU
with a `RuntimeWarning` when either check fails, and releases its buffers when
the call completes.

**Skipped on this host.** This machine's GPU is shared and was at 27.5 of
32.6 GiB in use throughout, with one foreign process holding 15.9 GiB and only
3391-5353 MiB free. Tests that need the device skip cleanly when there is no
CUDA device or when free memory is under the 5,000 MiB they need for headroom,
and the skip message reports the number it saw. On this host that was the usual
outcome for a full-suite run; with the floor lowered for a single run, all 48
tests in `tests/test_parity.py` passed, GPU tests included. The README does not
count a skipped GPU test as evidence the GPU path works.

Two measurements matter for anyone running these trails on a shared device:

- one device call lowers free memory by 759-1553 MiB for the lifetime of the
  process, so a sequence of calls walks a shared GPU down towards the
  4,000 MiB production gate; that is why the tests probe before each call;
- an explicit GPU request allocates a fresh `DeviceContext` per call, and the
  toolchain's allocator does not return all of it while the process lives.

The GPU tests assert the device path actually ran (a fallback raises under
`warnings.simplefilter("error")`), so a silent CPU fallback fails the test
rather than passing it, which is what a volume-only comparison would do.


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
  worth more than the divisions;
- one sign pass per cell in `mm_march_count` instead of a four-way comparison
  per tetrahedron: the count pass is 0.3-1.0 ms out of a 24-160 ms boolean, so
  the rewrite had nothing to win and it was reverted rather than kept. The
  emit, weld and edge-check passes were not touched for the same reason: the
  measured bottleneck is the signed-distance field at every resolution.

The raw kernel below the threshold is the only parallel path that loses, and it
is the reason for the threshold rather than a path that ships. For the other
kernels, the question is whether anything would win: `mm_transform` (0.013 ms
at 1,026 vertices) and `mm_pairwise_sum` (0.011 ms at 66 x 8) are far below the
4.3 ms pool start-up, `mm_combine_fields` is 0.23 ms at 250,000 samples,
`mm_march_count` is 0.3-1.0 ms, and `mm_march_emit` is 12.7 ms on the largest
benchmark boolean. Only the last is within an order of magnitude of
break-even, and it was not threaded: no measured caller would gain, so there
was nothing to verify.

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux 6.8.0-147-generic. Each number is the best complete Python API call from
the benchmark's repeat count. The "before" column is the same table measured on
the same host from the pre-optimisation tree (`5272194`), rebuilt from source
and run interleaved with the "now" column over five rounds. This is a shared
machine and repeated runs move by up to 30% in both directions, so the table is
one run rather than a best-of, and the ratios below are from that one run. The
table was measured before the parallel path landed. The boolean rows sit below
the parallelism threshold, so the path they take did not change; the parallel
path's own numbers, measured separately on the same host, are in
[CPU parallelism behind a measured threshold](#cpu-parallelism-behind-a-measured-threshold).

| case | before | mojo-manifold3d | manifold3d 3.5.2 | relative |
|---|---:|---:|---:|---:|
| volume + surface area (2,048 tri) | 0.058 ms | 0.001 ms | 0.089 ms | 63.490x faster |
| affine transform + mesh (1,026 vert) | 0.096 ms | 0.104 ms | 0.370 ms | 3.544x faster |
| convex hull (10,000 points) | 2.947 ms | 3.190 ms | 0.488 ms | 0.153x slower |
| Minkowski sum (66 x 8 vertices) | 4.031 ms | 2.521 ms | 2.570 ms | 1.020x faster |
| box union (resolution 28) | 53.625 ms | 26.988 ms | 0.390 ms | 0.014x slower |
| sphere-box difference (resolution 28) | 77.664 ms | 62.097 ms | 0.783 ms | 0.013x slower |

- The two boolean rows are 2.0x and 1.25x faster than this port's own previous
  build. They stay far behind upstream because upstream is a parallel C++
  exact-mesh engine and this port evaluates a dense signed-distance grid below
  the parallelism threshold to get a simple, explicit manifold guarantee. The
  threading of that grid does not move these rows, and the reason is measured
  above: they pass about 36,000 points, and the 4-worker path only overtakes
  serial near 50,000-64,000.
- The convex hull row is Qhull. `scipy.spatial.ConvexHull` accounts for
  essentially all of it, nothing in this repository runs on that path, and the
  two columns differ only by run-to-run noise: replacing Qhull with a Mojo
  quickhull is a different project, not an optimisation.
- The Minkowski row is 1.6x faster than before and at parity with upstream. It
  is dominated by two Qhull calls, one for the sphere and one for the expanded
  point cloud, which is why the two columns differ only by noise.
- The affine transform row is untouched by this work and its two columns
  differ only by noise; it was already ahead of upstream.
- `volume()` and `surface_area()` on the same solid share one measurement pass.
  The buffers they read never change after construction, so the result is
  cached on the instance. The benchmark measures the pair on an already-warm
  solid, so its 0.001 ms is a cache hit; a cold `volume()` on a new solid still
  costs about 0.055 ms, which is what the "before" column shows.


The GPU row below is not reproducible from `pixi run bench` on the shipped
gate: at resolution 40 the grid is 91,125 points, below `GPU_MIN_POINTS`, so
`device="gpu"` takes the CPU path and the two columns would be the same call.
`bench/bench.py` refuses to print a GPU row unless the device path actually ran,
so a fresh run prints no GPU table. Measured in a separate process with the
gate bypassed (`GPU_MIN_POINTS = 0` and the memory probe stubbed True, which is
what the tests do so the device path is exercised), the first pass starts the
device and later passes do not, so both columns are shown:

| GPU case | CPU (first call) | CPU (later) | GPU (first call) | GPU (later) |
|---|---:|---:|---:|---:|
| sphere-box difference (resolution 40) | 161.7 ms | 185.1 ms | 39.3 ms | 42.4 ms |

Both paths return the same volume, 2.182260707. The GPU column here is the
whole boolean, not only the field kernel, so it includes the per-call copy-in
and copy-out; the kernel-only numbers that decided the gate are in
[GPU path behind a measured size gate](#gpu-path-behind-a-measured-size-gate),
and they are restated there because the first-call cost they were measured
with is a device start-up, not a per-call cost.

Run `pixi run bench` to reproduce the table under the repository's
machine-wide benchmark lock. `pixi run python bench/parallel.py` compares the
field pinned serial against pinned at the worker cap on the same booleans, and
`bench/parallel_crossover.cc` reproduces the crossover table.

## License

MIT
