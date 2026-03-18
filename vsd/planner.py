"""Multi-waypoint greedy planner — solves all legs in parallel."""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

from vsd.path import VSDPath
from vsd.problem import VSDProblem
from vsd.solver import VSDSolver


@dataclass
class Waypoint:
    """A waypoint with required heading.

    Args:
        x, y: position
        h: heading in radians (required at this waypoint)
        frame: ``'global'`` or ``'relative'``
    """
    x: float
    y: float
    h: float
    frame: str = "global"


class InfeasibleLegError(Exception):
    """Raised when no feasible path exists for a particular leg."""

    def __init__(self, leg_index: int, message: str = ""):
        self.leg_index = leg_index
        super().__init__(
            f"No feasible solution for leg {leg_index}. {message}"
        )


def _solve_leg(args):
    """Module-level worker: solve one leg."""
    (leg_idx, x0, y0, h0, xf, yf, hf,
     v_max, v_min, omega_max, L_max, n_samples, tol, n_workers) = args
    prob = VSDProblem(x0, y0, h0, xf, yf, hf, v_max, v_min, omega_max)
    solver = VSDSolver(L_max=L_max, n_samples=n_samples, tol=tol,
                       n_workers=n_workers)
    paths = solver.solve(prob)
    return leg_idx, paths


class WaypointPlanner:
    """Plan a sequence of legs through a list of waypoints.

    Each waypoint requires an explicit heading.  All legs are solved in
    parallel using ProcessPoolExecutor.

    Args:
        v_max, v_min, omega_max: vehicle dynamics
        L_max: max straight-segment length per leg
        n_samples: initial-guess samples per candidate
        tol: solver tolerance
        n_workers: process workers (default = CPU count)
    """

    def __init__(
        self,
        v_max: float,
        v_min: float,
        omega_max: float,
        L_max: float = 10.0,
        n_samples: int = 250,
        tol: float = 1e-7,
        n_workers: Optional[int] = None,
    ):
        self.v_max     = v_max
        self.v_min     = v_min
        self.omega_max = omega_max
        self.L_max     = L_max
        self.n_samples = n_samples
        self.tol       = tol
        self.n_workers = n_workers or os.cpu_count() or 1

    def plan(
        self,
        waypoints: list[Waypoint],
        initial_state: Waypoint,
    ) -> tuple[list[VSDPath], float]:
        """Plan all legs in parallel.

        Args:
            waypoints: sequence of goal waypoints (each with required heading)
            initial_state: starting state

        Returns:
            (list of VSDPath for each leg, total cost)

        Raises:
            InfeasibleLegError: if any leg has no feasible solution
        """
        # Build full sequence: initial_state + waypoints
        all_pts = [initial_state] + list(waypoints)
        n_legs  = len(all_pts) - 1

        if n_legs < 1:
            return [], 0.0

        # Relative-frame waypoints are not supported in multi-waypoint planning
        # because each leg's vehicle state is needed for the coordinate transform,
        # which is only known after solving the preceding leg.
        for wp in all_pts:
            if wp.frame == "relative":
                raise ValueError(
                    "Relative-frame waypoints are not supported by WaypointPlanner. "
                    "Convert all waypoints to global frame before calling plan()."
                )
        converted = list(all_pts)

        # Build leg arguments — use 1 worker per leg (legs parallelised here)
        leg_args = []
        for i in range(n_legs):
            s = converted[i]
            g = converted[i + 1]
            leg_args.append((
                i,
                s.x, s.y, s.h,
                g.x, g.y, g.h,
                self.v_max, self.v_min, self.omega_max,
                self.L_max, self.n_samples, self.tol,
                1,  # inner n_workers = 1 (outer pool handles parallelism)
            ))

        # Solve all legs
        leg_results: dict[int, list[VSDPath]] = {}
        if self.n_workers == 1 or n_legs == 1:
            for args in leg_args:
                idx, paths = _solve_leg(args)
                leg_results[idx] = paths
        else:
            with ProcessPoolExecutor(max_workers=min(self.n_workers, n_legs)) as pool:
                futures = {pool.submit(_solve_leg, a): a[0] for a in leg_args}
                for fut in as_completed(futures):
                    idx, paths = fut.result()
                    leg_results[idx] = paths

        # Check feasibility and collect best path per leg
        best_paths: list[VSDPath] = []
        for i in range(n_legs):
            paths = leg_results.get(i, [])
            if not paths:
                raise InfeasibleLegError(i)
            best_paths.append(paths[0])

        total_cost = sum(p.cost for p in best_paths)
        return best_paths, total_cost
