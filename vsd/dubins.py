"""Classical Dubins solver — exact port of RobustDubins_Solver.cpp.

Solves LSL, LSR, RSL, RSR, LRL, RLR at a given turning radius.
Returns a list of DubinsPath objects (feasible or infeasible).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_TOL_DIST  = 1e-3
_TOL_THETA = 1e-3
_INF       = float("inf")


@dataclass
class DubinsPath:
    """Result for a single Dubins path type."""
    path_type: str        # LSL, LSR, RSL, RSR, LRL, RLR
    a: float = 0.0        # unsigned arc length (in radians * R if scaled)
    b: float = 0.0
    c: float = 0.0
    radius: float = 1.0
    feasible: bool = False
    cost: float = _INF    # arc-length cost = (a+b+c)*R (BSB) or (a+b+c)*R (BBB)

    # initial state (set when solving with non-unit radius)
    x0: float = 0.0
    y0: float = 0.0
    h0: float = 0.0

    def endpoint(self):
        """Return the (x, y, h) endpoint of this path from (x0, y0, h0)."""
        return _dubins_endpoint(
            self.path_type, self.a, self.b, self.c,
            self.x0, self.y0, self.h0, self.radius,
        )


def _mod(x, m=2 * math.pi):
    return x % m


def _dubins_endpoint(path_type, a, b, c, x0, y0, h0, R):
    """Compute endpoint for a Dubins path starting from (x0, y0, h0).

    a, b, c are stored as physical quantities (arc_length = angle * R for
    turning segments; straight distance for BSB middle segment).
    """
    from vsd.kinematics import bcb_displacement, s_displacement
    if path_type in ("LSL", "LSR", "RSL", "RSR"):
        # BSB: a/c are arc lengths (divide by R for angle), b is straight distance
        k1 = 1.0 if path_type.startswith("L") else -1.0
        k2 = 1.0 if path_type.endswith("L") else -1.0
        ang_a = a / R
        ang_c = c / R
        d1   = bcb_displacement(ang_a, 0.0, 0.0, h0, k1, R, R)
        psi1 = h0 + k1 * ang_a
        s    = s_displacement(psi1, b)
        d2   = bcb_displacement(ang_c, 0.0, 0.0, psi1, k2, R, R)
        ep   = np.array([x0, y0, h0]) + d1 + s + d2
        ep[2] = (h0 + d1[2] + d2[2]) % (2 * math.pi)
        return ep
    else:
        # BBB: all arcs
        k1   = 1.0 if path_type == "LRL" else -1.0
        ang_a = a / R
        ang_b = b / R
        ang_c = c / R
        d1   = bcb_displacement(ang_a, 0.0, 0.0, h0, k1, R, R)
        psi1 = h0 + k1 * ang_a
        d2   = bcb_displacement(ang_b, 0.0, 0.0, psi1, -k1, R, R)
        psi2 = psi1 - k1 * ang_b
        d3   = bcb_displacement(ang_c, 0.0, 0.0, psi2, k1, R, R)
        ep   = np.array([x0, y0, h0]) + d1 + d2 + d3
        ep[2] = (h0 + d1[2] + d2[2] + d3[2]) % (2 * math.pi)
        return ep


def _compare_endpoints(test, true, tol_d=_TOL_DIST, tol_h=_TOL_THETA):
    dist = math.hypot(test[0] - true[0], test[1] - true[1])
    dh   = abs(_mod(test[2]) - _mod(true[2]))
    dh   = min(dh, 2 * math.pi - dh)
    return dist <= tol_d and dh <= tol_h


def _solve_bsb(path_type, xf, yf, hf):
    """Solve one BSB path type.  Problem normalised: x0=y0=h0=0, R=1."""
    cth = math.cos(hf)
    sth = math.sin(hf)
    h   = _mod(hf)

    if path_type == "LSL":
        b_val = math.sqrt((xf - sth) ** 2 + (yf + cth - 1.0) ** 2)
        ay = yf + cth - 1.0
        ax = xf - sth
    elif path_type == "LSR":
        sq = (xf + sth) ** 2 + (yf - cth - 1.0) ** 2 - 4.0
        if sq < 0:
            return None
        b_val = math.sqrt(sq)
        ay = 2.0 * (xf + sth) + b_val * (yf - cth - 1.0)
        ax = b_val * (xf + sth) - 2.0 * (yf - cth - 1.0)
    elif path_type == "RSL":
        sq = (xf - sth) ** 2 + (yf + cth + 1.0) ** 2 - 4.0
        if sq < 0:
            return None
        b_val = math.sqrt(sq)
        ay = 2.0 * (xf - sth) - b_val * (yf + cth + 1.0)
        ax = b_val * (xf - sth) + 2.0 * (yf + cth + 1.0)
    elif path_type == "RSR":
        b_val = math.sqrt((xf + sth) ** 2 + (yf - cth + 1.0) ** 2)
        ay = -(yf - cth + 1.0)
        ax = xf + sth

    if math.isnan(b_val):
        return None

    a_raw = math.atan2(ay, ax)
    while a_raw < 0:
        a_raw += math.pi
    candidates_a = [a_raw, a_raw + math.pi]

    for a_val in candidates_a:
        if path_type == "LSL":
            c_val = _mod(h - a_val)
        elif path_type == "LSR":
            c_val = _mod(-h + a_val)
        elif path_type == "RSL":
            c_val = _mod(h + a_val)
        elif path_type == "RSR":
            if a_val == 0.0:
                c_val = 0.0 if h == 0.0 else 2.0 * math.pi - h
            else:
                unsignedCW = 2.0 * math.pi - _mod(h)
                if a_val >= unsignedCW:
                    c_val = unsignedCW + 2.0 * math.pi - a_val
                else:
                    c_val = 2.0 * math.pi - h - a_val

        # verify endpoint (R=1 normalised)
        test = _dubins_endpoint_norm(path_type, a_val, b_val, c_val)
        if _compare_endpoints(test, [xf, yf, h]):
            return (abs(a_val), abs(b_val), abs(c_val))
    return None


def _dubins_endpoint_norm(path_type, a, b, c):
    """Endpoint for normalised Dubins (R=1, x0=y0=h0=0)."""
    return _dubins_endpoint(path_type, a, b, c, 0.0, 0.0, 0.0, 1.0)


def _check_bbb_conditions(a, b, c):
    return max(a, c) < b and min(a, c) < b + math.pi


def _solve_bbb(path_type, xf, yf, hf):
    """Solve one BBB path type.  Problem normalised: R=1, x0=y0=h0=0."""
    h = _mod(hf)
    if path_type == "RLR":
        v = (xf + math.sin(h)) / 2.0
        w = (-yf - 1.0 + math.cos(h)) / 2.0
    else:  # LRL
        v = (xf - math.sin(h)) / 2.0
        w = (yf - 1.0 + math.cos(h)) / 2.0

    inner = 1.0 - (v * v + w * w) / 2.0
    if inner < -1.0 or inner > 1.0:
        return None
    b0 = math.acos(max(-1.0, min(1.0, inner)))
    if math.isnan(b0):
        return None
    b0 = abs(b0)
    b_cands = [b0, 2.0 * math.pi - b0]

    for b_val in b_cands:
        if 1.0 - math.cos(b_val) < 1e-15:
            continue
        denom = 1.0 - math.cos(b_val)
        A = (v * v - w * w) / (2.0 * denom)
        B = v * w / denom
        ay = B * math.cos(b_val) + A * math.sin(b_val)
        ax = A * math.cos(b_val) - B * math.sin(b_val)
        a_raw = 0.5 * math.atan2(ay, ax)
        if math.isnan(a_raw):
            continue
        while a_raw < 0:
            a_raw += math.pi / 2.0
        a_cands = [_mod(a_raw + k * math.pi / 2.0) for k in range(4)]
        for a_val in a_cands:
            if path_type == "RLR":
                c_val = _mod(-h - a_val + b_val)
            else:
                c_val = _mod(h - a_val + b_val)
            test = _dubins_endpoint_norm(path_type, abs(a_val), abs(b_val), abs(c_val))
            if (_compare_endpoints(test, [xf, yf, h]) and
                    _check_bbb_conditions(abs(a_val), abs(b_val), abs(c_val))):
                return (abs(a_val), abs(b_val), abs(c_val))
    return None


def solve_dubins(xf: float, yf: float, hf: float, R: float = 1.0) -> list[DubinsPath]:
    """Solve all 6 Dubins path types for a problem normalised at origin.

    The problem is normalised: start at (0, 0, 0), end at (xf, yf, hf).
    ``R`` is the turning radius.  Costs are arc-length (distance).

    Returns a list of 6 DubinsPath objects in order LSL, LSR, RSL, RSR, LRL, RLR.
    """
    # Normalise: translate + rotate + scale by 1/R
    xf_n = xf / R
    yf_n = yf / R
    # hf is already an angle, no scaling

    results = []
    for pt in ("LSL", "LSR", "RSL", "RSR"):
        sol = _solve_bsb(pt, xf_n, yf_n, hf)
        if sol is not None:
            a, b, c = sol
            # scale back: arc lengths in distance = angle * R
            dp = DubinsPath(
                path_type=pt, a=a * R, b=b * R, c=c * R,
                radius=R, feasible=True,
                cost=(a * R + b * R + c * R),
            )
        else:
            dp = DubinsPath(path_type=pt, radius=R, feasible=False, cost=_INF)
        results.append(dp)

    for pt in ("LRL", "RLR"):
        sol = _solve_bbb(pt, xf_n, yf_n, hf)
        if sol is not None:
            a, b, c = sol
            dp = DubinsPath(
                path_type=pt, a=a * R, b=b * R, c=c * R,
                radius=R, feasible=True,
                cost=(a + b + c) * R,
            )
        else:
            dp = DubinsPath(path_type=pt, radius=R, feasible=False, cost=_INF)
        results.append(dp)

    return results
