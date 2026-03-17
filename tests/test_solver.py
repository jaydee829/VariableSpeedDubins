"""Tests for vsd.solver.VSDSolver — integration tests."""

import math

import pytest

from vsd.kinematics import path_endpoint
from vsd.problem import VSDProblem
from vsd.solver import VSDSolver


@pytest.mark.slow
def test_solver_returns_list():
    """Solver always returns a list (possibly empty)."""
    prob = VSDProblem(0, 0, 0, 3, 2, 1.0, v_max=1.0, v_min=0.3, omega_max=1.0)
    solver = VSDSolver(L_max=10.0, n_samples=50, tol=1e-5, n_workers=1)
    paths = solver.solve(prob)
    assert isinstance(paths, list)


@pytest.mark.slow
def test_solver_paths_sorted_by_cost():
    prob = VSDProblem(0, 0, 0, 3, 0, 0, v_max=1.0, v_min=0.3, omega_max=1.0)
    solver = VSDSolver(L_max=10.0, n_samples=50, tol=1e-5, n_workers=1)
    paths = solver.solve(prob)
    costs = [p.cost for p in paths]
    assert costs == sorted(costs)


@pytest.mark.slow
def test_solver_endpoints_correct():
    """Every returned path should reach the target endpoint."""
    prob = VSDProblem(0, 0, 0, 3, 2, 1.0, v_max=1.0, v_min=0.3, omega_max=1.0)
    solver = VSDSolver(L_max=10.0, n_samples=50, tol=1e-5, n_workers=1)
    paths = solver.solve(prob)
    for vp in paths[:5]:  # check top 5
        if vp.is_dubins:
            continue
        ep = path_endpoint(vp.params, vp.candidate, vp.R, vp.r)
        # endpoint in normalised frame should match normalised target
        norm = prob.normalized()
        dist = math.hypot(ep[0] - norm.xf, ep[1] - norm.yf)
        dh   = abs((ep[2] - norm.hf + math.pi) % (2 * math.pi) - math.pi)
        assert dist < 5e-3, f"Endpoint x/y error {dist:.6f} for {vp.candidate}"
        assert dh   < 5e-3, f"Endpoint heading error {dh:.6f} for {vp.candidate}"


@pytest.mark.slow
def test_solver_all_feasible_flag():
    """All returned paths have feasible=True."""
    prob = VSDProblem(0, 0, 0, 2, 1, 0.5, v_max=1.0, v_min=0.3, omega_max=1.0)
    solver = VSDSolver(L_max=10.0, n_samples=50, tol=1e-5, n_workers=1)
    paths = solver.solve(prob)
    for p in paths:
        assert p.feasible


@pytest.mark.slow
def test_solver_none_suboptimal():
    """All returned paths have suboptimal=False."""
    prob = VSDProblem(0, 0, 0, 2, 1, 0.5, v_max=1.0, v_min=0.3, omega_max=1.0)
    solver = VSDSolver(L_max=10.0, n_samples=50, tol=1e-5, n_workers=1)
    paths = solver.solve(prob)
    for p in paths:
        assert not p.suboptimal


@pytest.mark.slow
def test_solver_dubins_endpoints_correct():
    """T-5: Dubins paths returned by solver reach the normalized-frame target."""
    prob = VSDProblem(0, 0, 0, 3, 2, 1.0, v_max=1.0, v_min=0.3, omega_max=1.0)
    solver = VSDSolver(L_max=10.0, n_samples=50, tol=1e-5, n_workers=1)
    paths = solver.solve(prob)
    norm = prob.normalized()
    dubins_paths = [vp for vp in paths if vp.is_dubins]
    for vp in dubins_paths:
        ep = vp.dubins_path.endpoint()
        dist = math.hypot(ep[0] - norm.xf, ep[1] - norm.yf)
        dh   = abs((ep[2] - norm.hf + math.pi) % (2 * math.pi) - math.pi)
        assert dist < 5e-3, (
            f"Dubins {vp.candidate['path_type']} x/y error {dist:.6f}"
        )
        assert dh < 5e-3, (
            f"Dubins {vp.candidate['path_type']} heading error {dh:.6f}"
        )
