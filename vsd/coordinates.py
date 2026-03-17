"""Coordinate frame transforms: global <-> vehicle-relative."""

import numpy as np


def global_to_relative(state: tuple, vehicle_state: tuple) -> tuple:
    """Transform a state from global frame to vehicle-relative frame.

    The vehicle frame has origin at vehicle_state (x0,y0) and x-axis aligned
    with heading h0. The returned state is expressed in this body frame.

    Args:
        state: (x, y, h) in global frame
        vehicle_state: (x0, y0, h0) — vehicle pose in global frame

    Returns:
        (x_rel, y_rel, h_rel) in vehicle-relative frame
    """
    x, y, h = state
    x0, y0, h0 = vehicle_state
    # translate
    dx = x - x0
    dy = y - y0
    # rotate by -h0
    c, s = np.cos(-h0), np.sin(-h0)
    x_rel = c * dx - s * dy
    y_rel = s * dx + c * dy
    h_rel = (h - h0) % (2 * np.pi)
    return (x_rel, y_rel, h_rel)


def relative_to_global(state: tuple, vehicle_state: tuple) -> tuple:
    """Transform a state from vehicle-relative frame to global frame.

    Args:
        state: (x_rel, y_rel, h_rel) in vehicle-relative frame
        vehicle_state: (x0, y0, h0) — vehicle pose in global frame

    Returns:
        (x, y, h) in global frame
    """
    x_rel, y_rel, h_rel = state
    x0, y0, h0 = vehicle_state
    # rotate by +h0
    c, s = np.cos(h0), np.sin(h0)
    x = c * x_rel - s * y_rel + x0
    y = s * x_rel + c * y_rel + y0
    h = (h_rel + h0) % (2 * np.pi)
    return (x, y, h)
