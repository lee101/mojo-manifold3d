"""Benchmark the covered API against manifold3d on identical geometry."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import manifold3d as upstream  # noqa: E402
import mojomanifold3d as mojo  # noqa: E402
from mojomanifold3d.core import _gpu_memory_available  # noqa: E402


def timeit(function, repeat=3):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def main():
    mojo.set_csg_resolution(28)
    rng = np.random.default_rng(7)
    cloud = rng.normal(size=(10_000, 3))

    ours_sphere = mojo.Manifold.sphere(1, 64)
    upstream_sphere = upstream.Manifold.sphere(1, 64)
    transform = np.array(
        [[0.8, -0.2, 0.1, 3.0], [0.3, 1.1, 0.0, -2.0], [0.0, 0.2, 0.9, 1.0]]
    )

    cases = [
        (
            "volume + surface area (2,048 tri)",
            lambda: (ours_sphere.volume(), ours_sphere.surface_area()),
            lambda: (upstream_sphere.volume(), upstream_sphere.surface_area()),
            7,
        ),
        (
            "affine transform + mesh (1,026 vert)",
            lambda: ours_sphere.transform(transform).to_mesh(),
            lambda: upstream_sphere.transform(transform).to_mesh(),
            7,
        ),
        (
            "convex hull (10,000 points)",
            lambda: mojo.Manifold.hull_points(cloud).volume(),
            lambda: upstream.Manifold.hull_points(cloud).volume(),
            5,
        ),
        (
            "Minkowski sum (66 x 8 vertices)",
            lambda: mojo.Manifold.sphere(1, 16)
            .minkowski_sum(mojo.Manifold.cube())
            .volume(),
            lambda: upstream.Manifold.sphere(1, 16)
            .minkowski_sum(upstream.Manifold.cube())
            .volume(),
            5,
        ),
        (
            "box union (resolution 28)",
            lambda: (
                mojo.Manifold.cube((2, 2, 2), True)
                + mojo.Manifold.cube((2, 2, 2), True).translate((0.7, 0.2, 0))
            ).volume(),
            lambda: (
                upstream.Manifold.cube((2, 2, 2), True)
                + upstream.Manifold.cube((2, 2, 2), True).translate(
                    (0.7, 0.2, 0)
                )
            ).volume(),
            3,
        ),
        (
            "sphere-box difference (resolution 28)",
            lambda: (
                mojo.Manifold.sphere(1, 12)
                - mojo.Manifold.cube((1.2, 1.2, 1.2), True).translate(
                    (0.45, 0, 0)
                )
            ).volume(),
            lambda: (
                upstream.Manifold.sphere(1, 12)
                - upstream.Manifold.cube((1.2, 1.2, 1.2), True).translate(
                    (0.45, 0, 0)
                )
            ).volume(),
            3,
        ),
    ]
    for _, ours, theirs, _ in cases:
        ours()
        theirs()

    print(f"Machine: {cpu_name()}; {platform.system()} {platform.release()}")
    print()
    print("| case | mojo-manifold3d | manifold3d 3.5.2 | relative |")
    print("|---|---:|---:|---:|")
    for name, ours, theirs, repeat in cases:
        mojo_time = timeit(ours, repeat)
        upstream_time = timeit(theirs, repeat)
        ratio = upstream_time / mojo_time
        label = "faster" if ratio >= 1 else "slower"
        print(
            f"| {name} | {mojo_time * 1e3:.3f} ms | "
            f"{upstream_time * 1e3:.3f} ms | {ratio:.3f}x {label} |"
        )

    if _gpu_memory_available(32 * 1024 * 1024):
        def boolean_at_resolution(device):
            mojo.set_csg_resolution(40)
            try:
                return mojo.Manifold.sphere(1, 12).boolean(
                    mojo.Manifold.cube((1.2, 1.2, 1.2), True).translate(
                        (0.45, 0, 0)
                    ),
                    mojo.OpType.Subtract,
                    device=device,
                ).volume()
            finally:
                mojo.set_csg_resolution(28)

        cpu = lambda: boolean_at_resolution("cpu")
        gpu = lambda: boolean_at_resolution("gpu")
        cpu()
        gpu()
        cpu_time = timeit(cpu, 3)
        gpu_time = timeit(gpu, 3)
        print()
        print("| GPU case | CPU | GPU | acceleration |")
        print("|---|---:|---:|---:|")
        print(
            "| sphere-box difference (resolution 40) | "
            f"{cpu_time * 1e3:.3f} ms | {gpu_time * 1e3:.3f} ms | "
            f"{cpu_time / gpu_time:.3f}x |"
        )


if __name__ == "__main__":
    main()
