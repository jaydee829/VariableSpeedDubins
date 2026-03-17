"""Tests for vsd.dubins — classical Dubins solver."""

import math

from vsd.dubins import DubinsPath, solve_dubins


def _check_endpoint(dp: DubinsPath, xf, yf, hf, tol=1e-3):
    ep = dp.endpoint()
    dist = math.hypot(ep[0] - xf, ep[1] - yf)
    dh   = abs((ep[2] - hf + math.pi) % (2 * math.pi) - math.pi)
    return dist <= tol and dh <= tol


def test_straight_line():
    """Start and end on same heading, only straight segment."""
    R = 1.0
    paths = solve_dubins(5.0, 0.0, 0.0, R)
    feasible = [p for p in paths if p.feasible]
    assert len(feasible) > 0
    # best should be straight (LSL or RSR with b ≈ 5, a≈c≈0)
    best = min(feasible, key=lambda p: p.cost)
    assert abs(best.cost - 5.0) < 0.1


def test_returns_6_paths():
    R = 1.0
    paths = solve_dubins(3.0, 2.0, 1.5, R)
    assert len(paths) == 6


def test_path_types():
    paths = solve_dubins(2.0, 1.0, 1.0, 1.0)
    types = [p.path_type for p in paths]
    assert "LSL" in types
    assert "LRL" in types
    assert "RLR" in types


def test_feasible_endpoint_satisfied():
    """All feasible Dubins paths should reach the specified endpoint."""
    R  = 1.5
    xf, yf, hf = 3.0, -2.0, math.pi / 3
    paths = solve_dubins(xf, yf, hf, R)
    for p in paths:
        if p.feasible:
            assert _check_endpoint(p, xf, yf, hf), (
                f"{p.path_type} endpoint check failed"
            )


def test_scale_invariance():
    """Scaling R by k should scale the cost by k."""
    xf, yf, hf = 4.0, 2.0, 0.8
    paths1 = solve_dubins(xf, yf, hf, R=1.0)
    paths2 = solve_dubins(xf * 2, yf * 2, hf, R=2.0)
    for p1, p2 in zip(paths1, paths2):
        if p1.feasible and p2.feasible:
            assert abs(p1.cost * 2 - p2.cost) < 1e-6


def test_degenerate_same_state():
    """Start == end: some feasible paths should exist with cost ≈ 0 or wraparound."""
    paths = solve_dubins(0.0, 0.0, 0.0, 1.0)
    # All BSB should be feasible (a=b=c=0 for LSL at least)
    lsl = next(p for p in paths if p.path_type == "LSL")
    assert lsl.feasible
    assert lsl.cost < 1e-3


# ---------------------------------------------------------------------------
# T-4: BBB endpoint correctness
# ---------------------------------------------------------------------------

def test_bbb_endpoint_satisfied():
    """T-4: LRL and RLR (BBB) feasible paths must reach the specified endpoint."""
    R = 0.5
    xf, yf, hf = 0.0, 1.5, 0.0
    paths = solve_dubins(xf, yf, hf, R)
    bbb_paths = [p for p in paths if p.path_type in ("LRL", "RLR") and p.feasible]
    for dp in bbb_paths:
        ep = dp.endpoint()
        dist = math.hypot(ep[0] - xf, ep[1] - yf)
        dh   = abs((ep[2] - hf + math.pi) % (2 * math.pi) - math.pi)
        assert dist < 1e-3, f"{dp.path_type} BBB endpoint x/y error {dist:.6f}"
        assert dh   < 1e-3, f"{dp.path_type} BBB endpoint heading error {dh:.6f}"


def test_bbb_feasibility():
    """T-4b: For a tight geometry (small R), BBB paths should be feasible."""
    R = 0.5
    # A point reachable mainly via turning (x=0, y=1 with h=0)
    paths = solve_dubins(0.0, 1.5, 0.0, R)
    types_feasible = [p.path_type for p in paths if p.feasible]
    # At least one of LRL/RLR should be feasible for this geometry
    assert any(t in types_feasible for t in ("LRL", "RLR")), (
        f"No BBB path feasible; feasible types: {types_feasible}"
    )
