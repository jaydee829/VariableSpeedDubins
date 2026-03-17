"""Tests for vsd.planner.WaypointPlanner."""

import pytest

from vsd.planner import InfeasibleLegError, Waypoint, WaypointPlanner


@pytest.mark.slow
def test_two_waypoint_plan():
    planner = WaypointPlanner(v_max=1.0, v_min=0.3, omega_max=1.0,
                              L_max=10.0, n_samples=50, tol=1e-5, n_workers=1)
    start = Waypoint(0.0, 0.0, 0.0)
    goal  = Waypoint(3.0, 2.0, 1.0)
    paths, total = planner.plan([goal], start)
    assert len(paths) == 1
    assert total > 0
    assert paths[0].feasible


@pytest.mark.slow
def test_three_waypoint_plan():
    planner = WaypointPlanner(v_max=1.0, v_min=0.3, omega_max=1.0,
                              L_max=10.0, n_samples=50, tol=1e-5, n_workers=1)
    start = Waypoint(0.0, 0.0, 0.0)
    wp1   = Waypoint(3.0, 0.0, 0.0)
    wp2   = Waypoint(6.0, 0.0, 0.0)
    paths, total = planner.plan([wp1, wp2], start)
    assert len(paths) == 2
    assert total > 0


@pytest.mark.slow
def test_total_cost_sum():
    planner = WaypointPlanner(v_max=1.0, v_min=0.3, omega_max=1.0,
                              L_max=10.0, n_samples=50, tol=1e-5, n_workers=1)
    start = Waypoint(0.0, 0.0, 0.0)
    wp1   = Waypoint(3.0, 0.0, 0.0)
    wp2   = Waypoint(6.0, 0.0, 0.0)
    paths, total = planner.plan([wp1, wp2], start)
    assert abs(total - sum(p.cost for p in paths)) < 1e-10


@pytest.mark.slow
def test_empty_waypoints():
    planner = WaypointPlanner(v_max=1.0, v_min=0.3, omega_max=1.0,
                              n_workers=1)
    start = Waypoint(0.0, 0.0, 0.0)
    paths, total = planner.plan([], start)
    assert paths == []
    assert total == 0.0


# ---------------------------------------------------------------------------
# T-6: InfeasibleLegError
# ---------------------------------------------------------------------------

def test_infeasible_leg_error_attributes():
    """T-6a: InfeasibleLegError carries the correct leg_index."""
    err = InfeasibleLegError(3, "no path found")
    assert err.leg_index == 3
    assert "leg 3" in str(err).lower()


def test_relative_frame_raises():
    """T-6b: Relative-frame waypoints raise ValueError (not silently no-op)."""
    planner = WaypointPlanner(v_max=1.0, v_min=0.3, omega_max=1.0, n_workers=1)
    start = Waypoint(0.0, 0.0, 0.0)
    goal  = Waypoint(1.0, 0.0, 0.0, frame="relative")
    with pytest.raises(ValueError, match="[Rr]elative"):
        planner.plan([goal], start)
