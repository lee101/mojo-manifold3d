"""Reproduces the crossover table in the README's parallelism section.

The measured numbers come from `bench/parallel_crossover.cc`, which calls the
exported kernel straight from C++ so the interpreter is off the timed path.
This driver is the same sweep through ctypes. It is diagnostic: it does not
take the machine-wide benchmark lock, so do not run it while a locked
benchmark is in flight on the same host.

Usage: pixi run python bench/parallel_crossover.py [pooled]
"""

import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import mojomanifold3d as mm  # noqa: E402
from mojomanifold3d import core  # noqa: E402

SIZES = [8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 96, 128, 160, 192]
WORKERS = [1, 2, 4]


def timed(solid, shape, workers, repeat):
    nx, ny, nz = shape
    origin = (-1.5, -1.5, -1.5)
    spacing = 3.0 / (nx - 1)
    previous = core._SIGNED_DISTANCE_WORKERS
    core._SIGNED_DISTANCE_WORKERS = workers
    try:
        core._field(solid, origin, spacing, shape)
        best = float("inf")
        for _ in range(repeat):
            start = time.perf_counter()
            core._field(solid, origin, spacing, shape)
            best = min(best, time.perf_counter() - start)
        return best * 1e3
    finally:
        core._SIGNED_DISTANCE_WORKERS = previous


def main():
    solid = mm.Manifold.sphere(1, 8)
    # A warm-up launch starts the pool when it is needed; skip it with no
    # argument to pay the start-up in the first timed call.
    if len(sys.argv) > 1:
        timed(solid, (8, 8, 8), 4, 1)
    print(f"{'nx':>5} {'points':>11} {'w1 ms':>9} {'w2 ms':>9} {'w4 ms':>9} {'w1/w4':>6}")
    for nx in SIZES:
        shape = (nx, nx, nx)
        points = nx**3
        repeat = 3 if points > 2_000_000 else 15
        times = [timed(solid, shape, workers, repeat) for workers in WORKERS]
        print(
            f"{nx:>5} {points:>11,} {times[0]:9.3f} {times[1]:9.3f} "
            f"{times[2]:9.3f} {times[0] / times[2]:6.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
