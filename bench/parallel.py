"""Before/after for the parallel signed-distance path, same host, back to back.

Runs the real boolean cases from bench/bench.py and the raw signed-distance
kernel at benchmark sizes, with the field pinned serial and pinned parallel.

Usage: pixi run python bench/parallel.py [label]
"""

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import mojomanifold3d as mm  # noqa: E402
from mojomanifold3d import core  # noqa: E402


def best(function, repeat=3):
    function()
    result = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        result = min(result, time.perf_counter() - start)
    return result * 1e3


def with_workers(workers, function, repeat=3):
    previous = core._SIGNED_DISTANCE_WORKERS
    core._SIGNED_DISTANCE_WORKERS = workers
    try:
        return best(function, repeat)
    finally:
        core._SIGNED_DISTANCE_WORKERS = previous


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "run"
    mm.set_csg_resolution(28)
    rng = np.random.default_rng(7)

    print(f"[{label}]")

    # The same cases bench.py measures, run end to end.
    cases = [
        (
            "box union (resolution 28)",
            lambda: (
                mm.Manifold.cube((2, 2, 2), True)
                + mm.Manifold.cube((2, 2, 2), True).translate((0.7, 0.2, 0))
            ).volume(),
        ),
        (
            "sphere-box difference (resolution 28)",
            lambda: (
                mm.Manifold.sphere(1, 12)
                - mm.Manifold.cube((1.2, 1.2, 1.2), True).translate((0.45, 0, 0))
            ).volume(),
        ),
        (
            "sphere-box difference (resolution 40)",
            lambda: (
                mm.Manifold.sphere(1, 12).boolean(
                    mm.Manifold.cube((1.2, 1.2, 1.2), True).translate((0.45, 0, 0)),
                    mm.OpType.Subtract,
                )
            ).volume(),
        ),
        (
            "disjoint boxes difference (resolution 28)",
            lambda: (
                mm.Manifold.cube((2, 2, 2), True)
                - mm.Manifold.cube((0.5, 0.5, 0.5), True).translate((3.0, 0, 0))
            ).volume(),
        ),
    ]
    # The resolution-40 case only runs at 40.
    mm.set_csg_resolution(28)
    for name, function in cases:
        if "resolution 40" in name:
            mm.set_csg_resolution(40)
        serial = with_workers(1, function)
        parallel = with_workers(-1, function, repeat=5)
        print(
            f"| {name} | {serial:.3f} ms | {parallel:.3f} ms | "
            f"{serial / parallel:.2f}x |"
        )
        mm.set_csg_resolution(28)

    # Raw kernel at the sizes the boolean path actually passes.
    solid = mm.Manifold.sphere(1, 12)
    samples = np.ascontiguousarray(rng.random(2_000_000)) - 0.5
    shape = (128, 128, 122)
    origin = (-1.5, -1.5, -1.5)
    spacing = 3.0 / 127

    def field():
        return core._field(solid, origin, spacing, shape)

    serial = with_workers(1, field, repeat=3)
    row = [f"{serial:.1f}"]
    for workers in (2, 4, 8):
        row.append(f"{with_workers(workers, field, repeat=3):.1f}")
    print(
        "| mm_signed_distance, 1,998,592 points (w1 / w2 / w4 / w8) | "
        + " / ".join(row)
        + " ms |"
    )


if __name__ == "__main__":
    main()
