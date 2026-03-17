"""Profile the VSD solver — run with: conda run -n vsd python profile_vsd.py"""
import cProfile
import io
import os
import pstats
import time

# Limit BLAS to 1 thread per IPOPT instance — prevents thread over-subscription
# when running multiple concurrent IPOPT solvers.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from vsd.problem import VSDProblem  # noqa: E402
from vsd.solver import VSDSolver    # noqa: E402

if __name__ == "__main__":
    prob = VSDProblem(0, 0, 0, 5, 3, 1.0, v_max=2.0, v_min=0.5, omega_max=1.0)

    # ---- Serial baseline ----
    solver1 = VSDSolver(n_workers=1)
    print("Warming up (serial)...")
    _ = solver1.solve(prob)

    print("Profiling serial solve...")
    pr = cProfile.Profile()
    pr.enable()
    t0 = time.perf_counter()
    paths = solver1.solve(prob)
    t1 = time.perf_counter()
    pr.disable()
    print(f"Serial solve: {t1 - t0:.3f}s, {len(paths)} feasible paths\n")

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(15)
    print(s.getvalue())

    # ---- Threaded ----
    for nw in [2, 4, 8]:
        solver_t = VSDSolver(n_workers=nw, use_threads=True)
        t0 = time.perf_counter()
        paths_t = solver_t.solve(prob)
        t1 = time.perf_counter()
        print(f"Threaded  ({nw:2d} workers): {t1 - t0:.3f}s, {len(paths_t)} feasible paths")
