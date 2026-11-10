// Two-sided crossover sweep, timed in C++ so no Python runs between launches.
//
// Build with
//   g++ -O2 -march=native -o /tmp/xover bench/parallel_crossover.cc \
//       -L dist -lmojo-manifold3d
// then run it with LD_LIBRARY_PATH=dist. Pass `pooled` to start the worker
// pool first; the default pays the pool start-up inside the first timed call,
// which is the case that decides the threshold. The output is the table quoted
// in the README's parallelism section.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

extern "C" void mm_signed_distance_data(
    long, long, long, double, double, double, double,
    long, long, long, long, long);

// A four-triangle tetrahedron. The kernel's cost is dominated by the grid for
// meshes this small, which is the worst case for paying a launch cost.
static double kVertices[12] = {0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1};
static long kFaces[12] = {0, 1, 2, 0, 1, 3, 0, 2, 3, 1, 2, 3};

static double run(int nx, int workers, int repeat) {
    std::vector<double> out((size_t)nx * nx * nx);
    double spacing = 3.0 / (nx - 1);
    double best = std::numeric_limits<double>::infinity();
    for (int r = 0; r < repeat; r++) {
        auto t0 = std::chrono::steady_clock::now();
        mm_signed_distance_data((long)kVertices, (long)kFaces, 4, 0.0, -1.5,
                                -0.5, spacing, nx, nx, nx, (long)out.data(),
                                workers);
        auto t1 = std::chrono::steady_clock::now();
        best = std::min(
            best, std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    return best;
}

int main(int argc, char** argv) {
    // `pooled` runs one warm-up at nx=8 with the requested worker count first,
    // so the worker pool is already started. `cold` skips that: the first
    // timed call pays the pool start-up.
    bool pooled = argc > 1 && std::string(argv[1]) == "pooled";
    if (pooled) {
        for (int workers : {2, 4}) run(8, workers, 1);
    }
    std::printf("%5s %11s %9s %9s %9s %6s\n", "nx", "points", "w1 ms",
                "w2 ms", "w4 ms", "w1/w4");
    for (int nx : {8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 96, 128, 160, 192}) {
        long n = (long)nx * nx * nx;
        int repeat = n > 2000000 ? 3 : 15;
        double t1 = run(nx, 1, repeat);
        double t2 = run(nx, 2, repeat);
        double t4 = run(nx, 4, repeat);
        std::printf("%5d %11ld %9.3f %9.3f %9.3f %6.2f\n", nx, n, t1, t2, t4,
                    t1 / t4);
        std::fflush(stdout);
    }
    return 0;
}
