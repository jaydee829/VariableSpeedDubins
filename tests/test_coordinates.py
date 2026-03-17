"""Tests for vsd.coordinates."""

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from vsd.coordinates import global_to_relative, relative_to_global

# ---------------------------------------------------------------------------
# Round-trip properties
# ---------------------------------------------------------------------------

@given(
    x=st.floats(-100, 100, allow_nan=False, allow_infinity=False),
    y=st.floats(-100, 100, allow_nan=False, allow_infinity=False),
    h=st.floats(0, 2 * math.pi, allow_nan=False, allow_infinity=False),
    x0=st.floats(-100, 100, allow_nan=False, allow_infinity=False),
    y0=st.floats(-100, 100, allow_nan=False, allow_infinity=False),
    h0=st.floats(0, 2 * math.pi, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=200)
def test_round_trip_global_to_relative(x, y, h, x0, y0, h0):
    """g2r then r2g should recover the original state."""
    state = (x, y, h)
    vehicle = (x0, y0, h0)
    rel  = global_to_relative(state, vehicle)
    back = relative_to_global(rel, vehicle)
    assert abs(back[0] - x) < 1e-10
    assert abs(back[1] - y) < 1e-10
    # heading comparison using sin/cos to avoid modulo issues
    assert abs(math.sin(back[2]) - math.sin(h)) < 1e-10
    assert abs(math.cos(back[2]) - math.cos(h)) < 1e-10


def test_vehicle_at_origin_identity():
    """With vehicle at origin h=0, relative == global."""
    state   = (3.0, 4.0, 1.2)
    vehicle = (0.0, 0.0, 0.0)
    rel = global_to_relative(state, vehicle)
    assert abs(rel[0] - 3.0) < 1e-12
    assert abs(rel[1] - 4.0) < 1e-12
    assert abs(rel[2] - 1.2) < 1e-12


def test_vehicle_frame_origin():
    """Vehicle position maps to origin in relative frame."""
    vehicle = (5.0, -3.0, 0.5)
    rel = global_to_relative(vehicle, vehicle)
    assert abs(rel[0]) < 1e-12
    assert abs(rel[1]) < 1e-12
    assert abs(rel[2]) < 1e-12 or abs(rel[2] - 2 * math.pi) < 1e-12


def test_rotation_90_deg():
    """Point directly ahead maps to (distance, 0) in vehicle frame."""
    dist = 5.0
    h0 = math.pi / 4  # vehicle heading 45 deg
    # point in global that is directly ahead of vehicle
    xg = dist * math.cos(h0)
    yg = dist * math.sin(h0)
    vehicle = (0.0, 0.0, h0)
    rel = global_to_relative((xg, yg, h0), vehicle)
    assert abs(rel[0] - dist) < 1e-10
    assert abs(rel[1]) < 1e-10
