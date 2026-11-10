# Verification record

Date: 2026-11-10. Host: Intel Xeon E5-2697 v4 @ 2.30 GHz, Linux 6.8.0-147-generic,
shared GPU with a foreign process holding 15.9 GiB (3.4-5.4 GiB free). The host
is shared and busy; every figure below is from this session only.

## What I ran, in order

1. `pixi run build && pixi run test` — green: `128 passed, 2 skipped in 22.90s`.
   The two skips are the GPU tests that need 5,000 MiB free (the host had
   4559 MiB at the time).
2. `pixi run bench` — five times over the session, plus interleaved runs of
   the pre-optimisation tree and of the shipped tree with the parallel path
   pinned serial.
3. `bench/parallel_crossover.cc`, built with `g++ -O2 -march=native` against
   `dist/libmojo-manifold3d.so`, run as `LD_LIBRARY_PATH=dist /tmp/xover`.
4. The raw pair kernel and the whole GPU boolean, with the gate bypassed, to
   reproduce the GPU section.
5. The pre-optimisation tree in a second checkout, built from source and run
   interleaved with the shipped tree, five rounds.

## README benchmark table

| case | README before | measured before | README now | measured now | README upstream | measured upstream | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| volume + surface area (2,048 tri) | 0.074 | 0.058 | 0.001 | 0.001 | 0.073 | 0.089 | before and upstream corrected |
| affine transform + mesh (1,026 vert) | 0.152 | 0.096 | 0.132 | 0.104 | 0.294 | 0.370 | all three corrected |
| convex hull (10,000 points) | 4.318 | 2.947 | 4.720 | 3.190 | 0.704 | 0.488 | all three corrected |
| Minkowski sum (66 x 8 vertices) | 6.239 | 4.031 | 3.427 | 2.521 | 2.974 | 2.570 | all three corrected |
| box union (resolution 28) | 83.907 | 53.625 | 42.040 | 26.988 | 0.330 | 0.390 | all three corrected |
| sphere-box difference (resolution 28) | 203.509 | 77.664 | 95.817 | 62.097 | 0.632 | 0.783 | all three corrected |

- The relative column was recomputed from the same run (upstream / ours), not
  carried over: 63.490x, 3.544x, 0.153x, 1.020x, 0.014x, 0.013x.
- A final `pixi run bench` on the edited tree, after this table was written,
  moved only the Minkowski ratio by more than 20% (0.867x slower that run),
  which is the same row-to-row noise the README already describes: upstream
  and this port both sit near 2.4-2.6 ms there and both columns call Qhull.
- The "before" column was rebuilt from `5272194`, the pre-optimisation tree.
  That tree pins `mojo ==1.0.0b3` and it builds and runs on this host. Its
  library does not export `mm_signed_distance_pair`, `mm_signed_distance_data`
  or `mm_signed_distance_pair_gpu`, which is how I know the two libraries are
  distinct (a first attempt was invalid: I had copied the shipped library over
  the old tree's `dist/`, and deleted it).
- Bullets that quoted the old numbers were corrected: boolean gain 2.0x/2.1x
  to 2.0x/1.25x, Minkowski 1.8x to 1.6x, and the volume row now says the
  benchmark measures a cache hit and quotes the 0.055 ms cold figure.
- The `Linux 6.8.0-139` label in the benchmark section was not the host it
  claims; this host is 6.8.0-147.

## Input and work checks

- Same input both sides: both sides run the same `bench/bench.py` lambdas on
  the same arguments. Sphere 16 has 66 verts / 128 tris on both sides, sphere 8
  has 18 / 32 on both sides, so the Minkowski 66 x 8 label is accurate. Union
  and difference volumes match upstream to 0.2-0.3%, which is the expected
  resolution effect, not skipped work.
- The benchmark does not skip work on either side, with one caveat: row 1
  measures `volume()` + `surface_area()` on an already-warm solid, which for
  this port is a Python-side cache hit (0.000565 ms) and for upstream a real
  recompute (0.038-0.089 ms). The ratio is therefore a cache hit against a
  computation. I kept the row (it is what a caller does) and labelled it in the
  README rather than changing the benchmark.
- `bench/bench.py` never prints the GPU row on the shipped gate: resolution 40
  is 91,125 points and `GPU_MIN_POINTS` is 1,000,000. Five shipped-tree runs
  printed "GPU row omitted". The README's old GPU row (508.768 / 236.943 /
  128.463 / 1.844x) is a number I could not produce, so it is gone; the
  measured replacement is described below.

## CPU parallelism table

Reproduced with `bench/parallel_crossover.cc` (one run, unpooled, which is the
README's "pays start-up in the first timed call" case):

| points | README serial | measured | README 2w | measured | README 4w | measured | README 4/1 | measured |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.03 | 0.029 | 4.38 | 4.742 | 4.24 | 4.659 | 0.01 | 0.01 |
| 8,000 | 0.54 | 0.451 | 5.25 | 5.241 | 4.56 | 4.949 | 0.12 | 0.09 |
| 32,768 | 1.96 | 1.851 | 5.32 | 6.965 | 4.89 | 7.379 | 0.40 | 0.25 |
| 64,000 | 3.76 | 3.708 | 6.49 | 7.188 | 5.37 | 5.879 | 0.70 | 0.63 |
| 110,592 | 6.40 | 6.196 | 7.78 | 7.917 | 6.23 | 5.975 | 1.03 | 1.04 |
| 262,144 | 14.79 | 15.583 | 12.34 | 13.546 | 8.42 | 8.955 | 1.76 | 1.74 |
| 884,736 | 49.84 | 50.587 | 36.14 | 37.629 | 31.09 | 25.132 | 1.60 | 2.01 |
| 2,097,152 | 154.17 | 119.459 | 70.49 | 71.172 | 40.70 | 38.665 | 3.79 | 3.09 |
| 7,077,888 | 436.06 | 419.804 | 238.20 | 255.658 | 115.68 | 136.094 | 3.77 | 3.08 |

Cells more than 20% off were corrected (32,768 serial/2w/4w/ratio, 8,000 2w/4w,
64,000 2w/4w/ratio, 884,736 4w/ratio, 2,097,152 serial/ratio, 7,077,888 4w/ratio).
The crossover is still 64,000-110,592 points. The "3.8x gain at 127 samples"
sentence could not be reproduced: the harness has no 127-sample row. It was
replaced by a measured 126-sample figure (2,000,376 points, 896.4 ms to
253.3 ms, 3.54x) from `mm_signed_distance_data`, and the 8-worker figure was
dropped because it was not measured here.

## GPU section

The README table (44.2-47.5 ms CPU vs 72.2-83.8 ms GPU at resolution 28, and
"0.53-0.65x", "0.91x", "0.93x") did not reproduce. Measured here with the
exported pair kernel, best of three after one warm-up:

| resolution | grid | points | CPU | GPU | GPU / CPU (README) | GPU / CPU measured |
|---:|---|---:|---:|---:|---:|---:|
| 28 | 33 x 33 x 33 | 35,937 | 44.4 ms | 2.9 ms | 0.53-0.65x | 0.06x |
| 40 | 45 x 45 x 45 | 91,125 | 131.1 ms | 3.8 ms | 3.90x | 0.03x |
| 60 | 65 x 64 x 64 | 266,240 | 315.7 ms | 4.1 ms | 0.91x | 0.01x |
| 80 | 85 x 84 x 84 | 599,760 | 706.3 ms | 16.7 ms | 0.93x | 0.02x |

GPU arrays matched the CPU arrays to 7.0e-15 or better on every row, so the
GPU column is real work, not a skipped launch. The one slow GPU call was the
first in a fresh process (337.1 ms at 33^3 followed by 3.1 ms), which is device
start-up. The README paragraph that said this path "loses at the sizes real
callers pass" and "the GPU's fixed copy-in, launch, and copy-out cost is larger
than the whole serial evaluation" is contradicted by these numbers and was
rewritten to state the measured ratio.

Whole-boolean GPU, gate bypassed (`GPU_MIN_POINTS = 0`, memory probe stubbed
true, which is what the GPU tests do; the production gate is unchanged):

| case | CPU first call | CPU later | GPU first call | GPU later |
|---|---:|---:|---:|---:|
| sphere-box difference (res 40) | 161.7 ms | 185.1 ms | 39.3 ms | 42.4 ms |

Both paths returned volume 2.182260707. This replaces the historical GPU row.
These are start-up-dominated measurements in separate processes; I did not
measure a stable best-of enough times to publish them as a benchmark, so the
README labels them as first/later columns rather than a ratio.

I did not change the `GPU_MIN_POINTS = 1_000_000` gate, the GPU tests, or the
GPU kernel. Whether to move the gate is a maintainer decision, not something a
measurement can settle alone, and the instruction for this phase was to change
code only for a bug with a failing test.

## Tests: parity with upstream

- `git log -p -- tests/` shows no removed assertions and no loosened
  tolerances: every `abs=`/`rel=` tolerance in the history was introduced
  together with its test. The loosest ones are `rel=0.035` on the curved
  boolean, `rel=0.012` plus `abs=0.08` on the box booleans, `rel=0.03` on the
  compose volume against a hand value, and `rel=0.004` on sphere volume.
- `rel=0.012` was looser than needed and had a hole: `compose(parts)` on two
  overlapping unit cubes should be 1.4 and the test accepted 1.4 +/- 3%; I
  measured `compose` at 1.3938 and the pairwise union at 1.3938 (upstream 1.4
  for both), so the result is right, but the tolerance cannot tell a correct
  union from an empty result: `assert_oriented_manifold` returns early for an
  empty mesh and `pytest.approx` accepts 0 for 11.32. `test_boolean_volume_and_
  bounds_parity` now asserts the result is not empty and has positive volume,
  and its volume tolerance is 1% (measured deviation: union -0.19%,
  intersection -0.20%, difference -0.33% at resolution 40).
- Added `test_signed_distance_inside_sign_is_positive`, which checks the field
  sign against ray parity on a rotated, non-axis-aligned cube (123 inside
  points, all positive; 2,074 outside points, all negative). It passes on the
  current code. Why it is needed: `reference_signed_distance` computes
  `max(distance, 1e-9)` and then negates it inside, so its expected value is
  always negative and only `max(0, ...)` hides that; the sign of the field is
  not what that test checks, and an axis-aligned cube's grid never puts a
  sample strictly inside a face region.
- The 2 skipped tests are GPU tests skipped for memory, as before. Running the
  parity file with the 5,000 MiB floor bypassed: `46 passed, 2 skipped`. The
  two remaining skips are in `test_explicit_gpu_boolean_matches_cpu_or_falls_
  back` and `test_gpu_field_matches_cpu_including_tail`; both re-probe and skip
  because a shared GPU's free memory fell to 3.8 GiB as my own calls held
  device memory. `test_gpu_size_gate_falls_back_below_crossover` and the rest of
  the GPU path pass when the device is reachable.

## What I could not reproduce

- The README's old GPU numbers (field kernel 0.53-0.93x, whole boolean 1.844x,
  508.768/236.943/128.463 ms): replaced, as above.
- The "4.3-4.9 ms" pool start-up and the "127 samples ... 1,970.5 ms to 516.8
  ms ... 3.8x" gain: the start-up measured 4.66-7.38 ms of which 4.6-4.8 ms is
  start-up, and the 127-sample row does not exist in the harness; replaced with
  measured figures.
- The parallelism table's 32,768 row: the README's 0.40x for 4 workers vs
  serial does not reproduce (0.25x here, and the 4-worker cell was slower than
  the 2-worker cell). It is corrected rather than explained away; on this
  shared host that cell is start-up noise and a second run could move it.
- The claim that "each number [is] the best complete Python API call from the
  benchmark's repeat count" holds for every row except the cache caveat above.

## What changed

- `README.md`: benchmark table (all three numeric columns and every ratio),
  its four bullets, the host kernel label, the parallelism table and its
  threshold/size prose, the 126-sample gain, the GPU kernel table and the
  surrounding claims, and the GPU row, which is no longer a historical number.
- `tests/test_parity.py`: nonzero/positive-volume assertions and a 1% volume
  tolerance on the box-boolean parity test; new inside-sign test.
- No code changed, because no failing test justified one. The candidate bug I
  found and could not show with a failing test is described below.

## Open finding: the boolean field convention

`_boolean` computes intersection as `min(first, second)` on the combined field
(the same combine as `mm_combine_fields` mode 0, which `test_simd_widths.py`
also uses for its union reference, while modes 1 and 2 are `min(first, -second)`
and `min(first, second)`, i.e. the opposite vocabulary). The README's "how it
works" says union, difference, intersection are `max(a, b)`, `min(a, -b)`,
`min(a, b)` on positive-inside fields, which matches the kernel: the field is
positive inside (ray parity, the new test). A gradient check on the union's own
output confirms the combine as coded gives the wrong inside set: sampling the
field along the union surface normal, it increased outward on 427 samples and
inward on 27,524.

The shipped results are nonetheless correct. Measured at resolution 40, the
non-overlapping union is 11.2981 against upstream's 11.3200, the difference of
two cubes is 3.3089 against 3.3200, and coplanar unions/intersections agree
too, so the two wrong signs appear to cancel in the operator that they reach.
I could not turn the inconsistency into a failing test: every operand
combination I tried produces the geometrically correct mesh, so I did not
change the combine. Anyone touching the operators should know that the
intersection path's combine does not match the README's formula even though
the README's formula is the correct one for this field.

## Final state

`pixi run build && pixi run test` — green: `129 passed, 2 skipped in 25.45s`
(the two skips are the GPU memory-floor tests). `dist/` and `build/` were left
as the build produced them, and no source, Mojo, or benchmark script changed.
