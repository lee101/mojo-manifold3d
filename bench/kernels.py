"""Time individual exported kernels at the sizes the boolean path actually uses.

Usage: python bench/kernels.py [label]

This is diagnostic support for `bench/bench.py`; the README numbers come from
`pixi run bench`. It does not take the machine-wide lock, so do not run it
while a locked benchmark is in flight on the same host.
"""

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import mojomanifold3d as mm  # noqa: E402
from mojomanifold3d._lib import addr, lib  # noqa: E402


def best_ms(function, repeat=15, inner=20):
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        for _ in range(inner):
            function()
        best = min(best, (time.perf_counter() - start) / inner)
    return best * 1e3


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "run"
    rng = np.random.default_rng(11)

    # Real caller sizes from bench/bench.py: cube/box boolean is resolution 28,
    # sphere-box difference is resolution 28, GPU case resolution 40.
    cases = []
    for name, solid in (
        ("union res28", mm.Manifold.cube((2, 2, 2), True)
         + mm.Manifold.cube((2, 2, 2), True).translate((0.7, 0.2, 0))),
        ("diff res28", mm.Manifold.sphere(1, 12)
         - mm.Manifold.cube((1.2, 1.2, 1.2), True).translate((0.45, 0, 0))),
    ):
        cases.append((name, solid))

    rows = []

    # mm_measure at the triangle counts of the benchmarked solids.
    sphere = mm.Manifold.sphere(1, 64)
    tri = len(sphere._faces)
    vertices = np.ascontiguousarray(sphere._vertices)
    faces = np.ascontiguousarray(sphere._faces)
    result = np.zeros(2)
    ms = best_ms(lambda: lib().mm_measure(
        addr(vertices), addr(faces), tri, addr(result, writable=True)))
    rows.append(("mm_measure", tri, "tri", ms))

    # mm_transform at the vertex counts of the benchmarked solids.
    vcount = len(sphere._vertices)
    matrix = rng.normal(size=(3, 4))
    dest = np.empty_like(vertices)
    ms = best_ms(lambda: lib().mm_transform(
        addr(vertices), addr(dest, writable=True), addr(matrix), vcount))
    rows.append(("mm_transform", vcount, "vert", ms))

    # mm_pairwise_sum at the Minkowski sizes (66 x 8).
    first = np.ascontiguousarray(sphere._vertices[:66])
    second = np.ascontiguousarray(mm.Manifold.cube()._vertices[:8])
    out = np.empty((len(first) * len(second), 3))
    ms = best_ms(lambda: lib().mm_pairwise_sum(
        addr(first), len(first), addr(second), len(second),
        addr(out, writable=True), 0))
    rows.append(("mm_pairwise_sum", len(first) * len(second), "prod", ms))

    # mm_combine_fields and the two march passes at a real grid size.
    for name, solid in cases:
        low = solid._vertices.min(axis=0)
        high = solid._vertices.max(axis=0)
        spacing = float((high - low).max()) / 28
        origin = low - 2 * spacing
        upper = high + 2 * spacing
        shape = np.ceil((upper - origin) / spacing).astype(int) + 1
        nx, ny, nz = (int(v) for v in shape)
        count = nx * ny * nz
        field = np.ascontiguousarray(-rng.random(count) + 0.5)
        other = np.ascontiguousarray(-rng.random(count) + 0.5)
        combined = np.empty(count)
        ms = best_ms(lambda: lib().mm_combine_fields(
            addr(field), addr(other), addr(combined, writable=True),
            count, 0), repeat=7, inner=10)
        rows.append((f"mm_combine_fields [{name}]", count, "samples", ms))

        ms = best_ms(lambda: lib().mm_march_count(
            addr(combined), nx, ny, nz), repeat=7, inner=5)
        rows.append((f"mm_march_count [{name}]", count, "cells", ms))

        tc = int(lib().mm_march_count(addr(combined), nx, ny, nz))
        if tc == 0:
            continue
        triangles = np.empty((tc, 3, 3))
        keys = np.empty((tc, 3), dtype=np.int64)
        ox, oy, oz = (float(v) for v in origin)
        ms = best_ms(lambda: lib().mm_march_emit(
            addr(combined), addr(triangles, writable=True),
            addr(keys, writable=True), nx, ny, nz, ox, oy, oz,
            float(spacing)), repeat=7, inner=5)
        rows.append((f"mm_march_emit [{name}]", tc, "tri", ms))

    print(f"[{label}]")
    print("| kernel | size | unit | best ms |")
    print("|---|---:|---|---:|")
    for name, size, unit, ms in rows:
        print(f"| {name} | {size:,} | {unit} | {ms:.4f} |")


if __name__ == "__main__":
    main()
