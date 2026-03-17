"""VSDSolver: parallel solve over all 84 candidates (76 VSD + 8 Dubins)."""

import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Optional

try:
    from threadpoolctl import threadpool_limits as _threadpool_limits
    _HAS_THREADPOOLCTL = True
except ImportError:  # pragma: no cover
    _HAS_THREADPOOLCTL = False

import numpy as np

from vsd.candidates import candidate_list as _make_candidate_list
from vsd.dubins import solve_dubins
from vsd.nlp import solve_candidate
from vsd.path import VSDPath
from vsd.problem import VSDProblem

# Cache the candidate list at import time — it is constant for the lifetime
# of the process and avoids rebuilding 76 dicts on every call to solve().
_ALL_VSD_CANDIDATES = _make_candidate_list()


_INF = float("inf")


class _NullCtx:
    """No-op context manager used when threadpoolctl is unavailable."""

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


# ---------------------------------------------------------------------------
# Module-level worker functions (must be picklable)
# ---------------------------------------------------------------------------

def _solve_vsd_worker(args):
    """Worker for a single VSD candidate — module-level for pickling."""
    candidate, R, r, xf, yf, hf, L_max, n_samples, tol, seed = args
    return solve_candidate(candidate, R, r, xf, yf, hf,
                           L_max=L_max, n_samples=n_samples, tol=tol, seed=seed)


def _solve_dubins_worker(args):
    """Worker for Dubins candidates at a given radius — module-level."""
    xf, yf, hf, R = args
    return solve_dubins(xf, yf, hf, R)


# ---------------------------------------------------------------------------
# VSDSolver
# ---------------------------------------------------------------------------

class VSDSolver:
    """Solve a VSD problem by trying all 84 candidates in parallel.

    Args:
        L_max: maximum straight-segment length (default 10.0)
        n_samples: number of samples for initial guess (default 250)
        tol: IPOPT convergence tolerance (default 1e-7)
        n_workers: number of parallel workers (default = CPU count)
        use_threads: use ThreadPoolExecutor instead of ProcessPoolExecutor
            (default True).  Threads avoid process-spawn overhead on Windows
            and work well because CasADi/IPOPT releases the GIL during solving.
    """

    def __init__(
        self,
        L_max: float = 10.0,
        n_samples: int = 250,
        tol: float = 1e-7,
        n_workers: Optional[int] = None,
        use_threads: bool = True,
    ):
        self.L_max       = L_max
        self.n_samples   = n_samples
        self.tol         = tol
        self.n_workers   = n_workers or os.cpu_count() or 1
        self.use_threads = use_threads

    def solve(self, problem: VSDProblem) -> list[VSDPath]:
        """Solve the problem and return all feasible, non-suboptimal paths.

        The returned list is sorted by cost (ascending).  Returns empty list
        if no feasible solution is found.
        """
        # Normalise: translate + rotate so start is at origin with h=0
        norm = problem.normalized()
        xf, yf, hf = norm.xf, norm.yf, norm.hf
        R, r = problem.R, problem.r

        # Prepare VSD candidate args
        cands = _ALL_VSD_CANDIDATES
        vsd_args = [
            (c, R, r, xf, yf, hf, self.L_max, self.n_samples, self.tol, i)
            for i, c in enumerate(cands)
        ]

        # Solve VSD candidates in parallel
        vsd_results = self._parallel_vsd(vsd_args)

        # Solve Dubins candidates (two radii: R and r)
        dub_R = solve_dubins(xf, yf, hf, R)
        dub_r_paths = solve_dubins(xf, yf, hf, r)
        # Only LRL and RLR at radius r (indices 4,5)
        dub_r = [dub_r_paths[4], dub_r_paths[5]]

        # Collect all paths
        paths: list[VSDPath] = []

        # Process VSD results — store original initial state so path_history()
        # and endpoint() return global-frame coordinates.
        for res in vsd_results:
            if not res["feasible"] or res["suboptimal"]:
                continue
            vp = VSDPath(
                candidate=res["candidate"],
                params=res["params"],
                cost=res["cost"],
                R=R, r=r,
                feasible=True,
                suboptimal=False,
                x0=problem.x0, y0=problem.y0, h0=problem.h0,
            )
            paths.append(vp)

        # Process Dubins results (6 at R + 2 at r)
        for dp in dub_R + dub_r:
            if not dp.feasible:
                continue
            # Build a dummy candidate dict for Dubins paths
            cand = {"path_class": "Dubins", "path_type": dp.path_type,
                    "orientation": dp.path_type}
            vp = VSDPath(
                candidate=cand,
                params=np.array([dp.a, dp.b, dp.c, 0.0]),
                cost=dp.cost,
                R=R, r=r,
                feasible=True,
                suboptimal=False,
                is_dubins=True,
                dubins_path=dp,
                x0=problem.x0, y0=problem.y0, h0=problem.h0,
            )
            paths.append(vp)

        paths.sort(key=lambda p: p.cost)
        return paths

    def _parallel_vsd(self, args_list):
        if self.n_workers == 1:
            return [_solve_vsd_worker(a) for a in args_list]
        Executor = ThreadPoolExecutor if self.use_threads else ProcessPoolExecutor
        results = [None] * len(args_list)
        # Limit BLAS to 1 thread per worker so that n_workers concurrent IPOPT
        # instances don't over-subscribe the CPU with their own thread pools.
        ctx = (
            _threadpool_limits(limits=1)
            if _HAS_THREADPOOLCTL and self.use_threads
            else _NullCtx()
        )
        with ctx, Executor(max_workers=self.n_workers) as pool:
            futures = {pool.submit(_solve_vsd_worker, a): i
                       for i, a in enumerate(args_list)}
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    results[i] = fut.result()
                except Exception:
                    results[i] = dict(feasible=False, suboptimal=False,
                                      params=None, cost=_INF,
                                      candidate=args_list[i][0])
        return results
