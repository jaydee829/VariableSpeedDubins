"""CasADi NLP builder per VSD candidate — port of VSDNLP.cpp.

Each NLP has:
  - 4 decision variables (arc-length params)
  - Linear inequality constraints (from linearConstraints)
  - 3 nonlinear equality constraints (endpoint boundary conditions)
  - Linear cost function (travel time)
"""

from __future__ import annotations

import logging
import math
import threading

import casadi as ca
import numpy as np

from vsd.candidates import determine_curvature_params
from vsd.kinematics import expand_params_short, path_endpoint, path_endpoint_unwrapped

_log = logging.getLogger(__name__)

_INF = float("inf")

# ---------------------------------------------------------------------------
# NLP solver cache — compiled once per (candidate, R, r, tol), reused across
# calls with different bounds / initial guesses.
# ---------------------------------------------------------------------------

_NLP_CACHE: dict = {}
_NLP_LOCK = threading.Lock()


def _build_nlpsol(candidate: dict, R: float, r: float, tol: float) -> tuple:
    """Build and return a compiled ca.nlpsol for *candidate*.

    Returns ``(solver, n_lin)`` where ``n_lin`` is the number of linear
    constraint rows (so the caller can split lbg/ubg correctly).

    The symbolic NLP is: minimise  grad·x
                         subject to  lbx <= x <= ubx
                                     lbg <= [A@x; ep(x)] <= ubg
    All numerical bounds are passed at call time, so R, r, and the candidate
    topology are the only things baked in.
    """
    # A is purely integer coefficients — independent of R, r, L_max.
    # We call linear_constraints with dummy L_max=10 to extract A only.
    a_sub_dummy = alpha_subopt(math.pi, R, r)
    _, _, g_l_dummy, _, A = linear_constraints(candidate, R, r, a_sub_dummy, 10.0)
    n_lin = len(g_l_dummy)

    grad = cost_gradient(candidate, R)
    x = ca.MX.sym("x", 4)
    f = (
        float(grad[0]) * x[0]
        + float(grad[1]) * x[1]
        + float(grad[2]) * x[2]
        + float(grad[3]) * x[3]
    )
    ep = ca_path_endpoint(x, candidate, R, r, h0=0.0)

    if n_lin > 0:
        A_ca = ca.MX(np.array(A, dtype=float))
        g = ca.vertcat(A_ca @ x, ep)
    else:
        g = ep

    opts = {
        "ipopt.print_level": 0,
        "print_time": 0,
        "ipopt.tol": tol,
        "ipopt.acceptable_tol": tol * 100,
        "ipopt.max_iter": 500,
        "ipopt.mu_strategy": "adaptive",
        "ipopt.hessian_approximation": "limited-memory",
        "error_on_fail": False,   # return without raising; we check residual
    }
    solver = ca.nlpsol("S", "ipopt", {"x": x, "f": f, "g": g}, opts)
    return solver, n_lin


def _get_cached_nlp(candidate: dict, R: float, r: float, tol: float) -> tuple:
    """Return (solver, n_lin), building and caching on first call per key."""
    key = (
        candidate["path_class"],
        candidate["path_type"],
        candidate["orientation"],
        R, r, tol,
    )
    # Fast path — no lock needed for pure reads in CPython
    if key in _NLP_CACHE:
        return _NLP_CACHE[key]
    # Slow path — double-checked locking prevents duplicate builds
    with _NLP_LOCK:
        if key not in _NLP_CACHE:
            _NLP_CACHE[key] = _build_nlpsol(candidate, R, r, tol)
    return _NLP_CACHE[key]


# ---------------------------------------------------------------------------
# Linear constraints — exact port of VSDUtils_linearConstraints.cpp
# ---------------------------------------------------------------------------

def linear_constraints(
    candidate: dict, R: float, r: float, a_subopt_pi: float, L_max: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (x_l, x_u, g_l, g_u, A) for linear constraints.

    The linear constraints are: g_l <= A @ p <= g_u.
    """
    pi = math.pi
    pc = candidate["path_class"]
    pt = candidate["path_type"]

    if pc == "TST":
        if pt == "BCB-S-BCB":
            x_l = [0, 0, 0, 0]
            x_u = [a_subopt_pi, pi / 2, L_max, a_subopt_pi]
            A = [[1, -1, 0, 0], [0, -1, 0, 1]]
            g_l = [-pi, -pi]
            g_u = [0, 0]
        elif pt == "BCB-S-BC":
            x_l = [0, 0, 0, 0]
            x_u = [a_subopt_pi, pi / 2, L_max, pi]
            A = [[1, -1, 0, 0]]
            g_l = [-pi]
            g_u = [0]
        elif pt == "BCB-S-B":
            x_l = [0, 0, 0, 0]
            x_u = [a_subopt_pi, pi / 2, L_max, pi]
            A = [[1, -1, 0, 0], [0, -1, 0, 1]]
            g_l = [-pi, -pi]
            g_u = [0, 0]
        elif pt == "CB-S-BCB":
            x_l = [0, 0, 0, 0]
            x_u = [pi, pi / 2, L_max, a_subopt_pi]
            A = [[0, -1, 0, 1]]
            g_l = [-pi]
            g_u = [0]
        elif pt == "CB-S-BC":
            x_l = [0, 0, 0, 0]
            x_u = [pi, pi, L_max, pi]
            A = [[1, 2, 0, 1], [0, 2, 0, 1]]
            g_l = [0, 0]
            g_u = [2 * pi, 2 * pi]
        elif pt == "CB-S-B":
            x_l = [0, 0, 0, 0]
            x_u = [pi, pi, L_max, pi]
            A = [[1, 2, 0, 1], [0, -1, 0, 1]]
            g_l = [0, -pi]
            g_u = [2 * pi, 0]
        elif pt == "B-S-BCB":
            x_l = [0, 0, 0, 0]
            x_u = [pi, L_max, pi, a_subopt_pi]
            A = [[1, 0, -1, 0], [0, 0, -1, 1]]
            g_l = [-pi, -pi]
            g_u = [0, 0]
        elif pt == "B-S-BC":
            x_l = [0, 0, 0, 0]
            x_u = [pi, L_max, pi, pi]
            A = [[1, 0, -1, 0], [0, 0, 2, 1]]
            g_l = [-pi, 0]
            g_u = [0, 2 * pi]
        else:
            raise ValueError(f"Unknown TST path_type {pt!r}")

    elif pc == "TT":
        if pt == "BCB-BCB":
            x_l = [0, 0, 0, 0]
            x_u = [pi, 2 * pi, pi, pi]
            A = [[1, 0, -1, 0], [0, 0, -1, 1], [0, 1, 2, 0]]
            g_l = [-pi, -pi, 0]
            g_u = [0, 0, 2 * pi]
        elif pt == "BCB-BC":
            x_l = [0, 0, 0, 0]
            x_u = [pi, 2 * pi, pi, 2 * pi]
            A = [[1, 0, -1, 0], [0, -1, 0, 1], [0, 1, 2, 0]]
            g_l = [-pi, -2 * pi, 0]
            g_u = [0, 0, 2 * pi]
        elif pt == "BCB-B":
            x_l = [0, 0, 0, 0]
            x_u = [pi, 2 * pi, pi, pi]
            A = [[1, 0, -1, 0], [0, 0, -1, 1], [0, 1, 2, 0]]
            g_l = [-pi, -pi, 0]
            g_u = [0, 0, 2 * pi]
        elif pt == "CB-BCB":
            x_l = [0, 0, 0, 0]
            x_u = [2 * pi, pi, 2 * pi, pi]
            A = [[1, 0, -1, 0], [0, -1, 0, 1], [0, 2, 1, 0]]
            g_l = [-2 * pi, -pi, 0]
            g_u = [0, 0, 2 * pi]
        elif pt == "B-BCB":
            x_l = [0, 0, 0, 0]
            x_u = [pi, L_max, pi, pi]
            A = [[1, -1, 0, 0], [0, -1, 0, 1], [0, 2, 1, 0]]
            g_l = [-pi, -pi, 0]
            g_u = [0, 0, 2 * pi]
        else:
            raise ValueError(f"Unknown TT path_type {pt!r}")

    elif pc == "TTTT":
        if pt == "CB-TT-BC":
            x_l = [0, 0, 0, 0]
            x_u = [2 * pi, pi, 2 * pi, 2 * pi]
            A = [[1, 0, -1, 0], [0, 0, -1, 1], [0, 2, 1, 0]]
            g_l = [-2 * pi, -2 * pi, 0]
            g_u = [0, 0, 2 * pi]
        elif pt == "CB-TT-B":
            x_l = [0, 0, 0, 0]
            x_u = [2 * pi, pi, 2 * pi, pi]
            A = [[1, 0, -1, 0], [0, -1, 0, 1], [0, 2, 1, 0]]
            g_l = [-2 * pi, -pi, 0]
            g_u = [0, 0, 2 * pi]
        elif pt == "B-TT-BC":
            x_l = [0, 0, 0, 0]
            x_u = [pi, pi, 2 * pi, 2 * pi]
            A = [[1, -1, 0, 0], [0, 0, -1, 1], [0, 2, 1, 0]]
            g_l = [-pi, -2 * pi, 0]
            g_u = [0, 0, 2 * pi]
        elif pt == "B-TT-B":
            x_l = [0, 0, 0, 0]
            x_u = [pi, pi, 2 * pi, pi]
            A = [[1, -1, 0, 0], [0, -1, 0, 1], [0, 2, 1, 0]]
            g_l = [-pi, -pi, 0]
            g_u = [0, 0, 2 * pi]
        else:
            raise ValueError(f"Unknown TTTT path_type {pt!r}")

    elif pc == "TTT":
        if pt == "CB-T-BCB":
            x_l = [0, 0, 0, 0]
            x_u = [2 * pi, pi, 2 * pi, pi]
            A = [[1, 0, -1, 0], [0, -1, 0, 1], [0, 2, 1, 0]]
            g_l = [-2 * pi, -pi, 0]
            g_u = [0, 0, 2 * pi]
        elif pt == "CB-T-BC":
            x_l = [0, 0, 0, 0]
            x_u = [2 * pi, pi, 2 * pi, 2 * pi]
            A = [[1, 0, -1, 0], [0, 0, -1, 1], [0, 2, 1, 0]]
            g_l = [-2 * pi, -2 * pi, 0]
            g_u = [0, 0, 2 * pi]
        elif pt == "CB-T-B":
            x_l = [0, 0, 0, 0]
            x_u = [2 * pi, pi, 2 * pi, pi]
            A = [[1, 0, -1, 0], [0, -1, 0, 1], [0, 2, 1, 0]]
            g_l = [-2 * pi, -pi, 0]
            g_u = [0, 0, 2 * pi]
        elif pt == "B-T-BCB":
            x_l = [0, 0, 0, 0]
            x_u = [pi, pi, 2 * pi, pi]
            A = [[1, -1, 0, 0], [0, -1, 0, 1], [0, 2, 1, 0]]
            g_l = [-pi, -pi, 0]
            g_u = [0, 0, 2 * pi]
        elif pt == "B-T-BC":
            x_l = [0, 0, 0, 0]
            x_u = [pi, pi, 2 * pi, 2 * pi]
            A = [[1, -1, 0, 0], [0, 0, -1, 1], [0, 2, 1, 0]]
            g_l = [-pi, -2 * pi, 0]
            g_u = [0, 0, 2 * pi]
        elif pt == "B-T-B":
            x_l = [0, 0, 0, 0]
            x_u = [pi, pi, 2 * pi, pi]
            A = [[1, -1, 0, 0], [0, -1, 0, 1], [0, 2, 1, 0]]
            g_l = [-pi, -pi, 0]
            g_u = [0, 0, 2 * pi]
        elif pt == "BCB-T-BC":
            x_l = [0, 0, 0, 0]
            x_u = [pi, 2 * pi, pi, 2 * pi]
            A = [[1, -1, 0, 0], [0, -1, 0, 1], [0, 2, 1, 0]]
            g_l = [-pi, -2 * pi, 0]
            g_u = [0, 0, 2 * pi]
        elif pt == "BCB-T-B":
            x_l = [0, 0, 0, 0]
            x_u = [pi, 2 * pi, pi, pi]
            A = [[1, 0, -1, 0], [0, 0, -1, 1], [0, 2, 1, 0]]
            g_l = [-pi, -pi, 0]
            g_u = [0, 0, 2 * pi]
        else:
            raise ValueError(f"Unknown TTT path_type {pt!r}")
    else:
        raise ValueError(f"Unknown path_class {pc!r}")

    return (np.array(x_l), np.array(x_u),
            np.array(g_l), np.array(g_u),
            np.array(A, dtype=float))


# ---------------------------------------------------------------------------
# Cost function gradient — port of costFunctionGradient()
# ---------------------------------------------------------------------------

def cost_gradient(candidate: dict, R: float) -> np.ndarray:
    """Return 4-element gradient of the linear cost function."""
    pc, pt = candidate["path_class"], candidate["path_type"]
    if pc == "TST":
        g = {
            "BCB-S-BCB": [R, 2*R, 1.0, R],
            "BCB-S-BC":  [R, 2*R, 1.0, R],
            "BCB-S-B":   [R, R,   1.0, R],
            "CB-S-BCB":  [R, 2*R, 1.0, R],
            "CB-S-BC":   [R, 2*R, 1.0, R],
            "CB-S-B":    [R, R,   1.0, R],
            "B-S-BCB":   [R, 1.0, R,   R],
            "B-S-BC":    [R, 1.0, R,   R],
        }[pt]
    elif pc == "TT":
        g = {
            "BCB-BCB": [R, 2*R, 2*R, R],
            "BCB-BC":  [R, R,   2*R, R],
            "BCB-B":   [R, R,   R,   R],
            "CB-BCB":  [R, 2*R, R,   R],
            "B-BCB":   [R, R,   R,   R],
        }[pt]
    elif pc == "TTT":
        g = {
            # "BCB-T-BCB" is not in candidate_list() for TTT; kept for reference
            "BCB-T-BCB": [R, 3*R, 4*R, R],
            "BCB-T-BC":  [R, 2*R, 4*R, R],
            "BCB-T-B":   [R, 2*R, 3*R, R],
            "CB-T-BCB":  [R, 4*R, 2*R, R],
            "CB-T-BC":   [R, 4*R, R,   R],
            "CB-T-B":    [R, 3*R, R,   R],
            "B-T-BCB":   [R, 3*R, 2*R, R],
            "B-T-BC":    [R, 3*R, R,   R],
            "B-T-B":     [R, 2*R, R,   R],
        }[pt]
    elif pc == "TTTT":
        g = {
            "CB-TT-BC": [R, 6*R, 2*R, R],
            "CB-TT-B":  [R, 5*R, 2*R, R],
            "B-TT-BC":  [R, 5*R, 2*R, R],
            "B-TT-B":   [R, 4*R, 2*R, R],
        }[pt]
    else:
        raise ValueError(f"Unknown path_class {pc!r}")
    return np.array(g)


# ---------------------------------------------------------------------------
# Alpha suboptimality threshold — port of alphaSubopt()
# ---------------------------------------------------------------------------

def alpha_subopt(beta: float, R: float, r: float) -> float:
    return math.pi - beta / 2.0 - math.asin(math.sin(beta / 2.0) * (R - r) / (R + r))


# ---------------------------------------------------------------------------
# Suboptimality check — port of checkSuboptimality() / checkAlphaSuboptimality()
# ---------------------------------------------------------------------------

def check_suboptimality(
    params_short: np.ndarray,
    candidate: dict,
    R: float,
    r: float,
) -> bool:
    """Return True if the solution is known to be suboptimal.

    Intentional divergence from C++ checkAlphaSuboptimality: the C++ code
    has a variable-reuse bug where it overwrites alpha1/as1 before using them
    in the final comparison for TT and TST classes.  This Python version
    correctly evaluates both BCB segments independently.
    """
    pc = candidate["path_class"]
    pt = candidate["path_type"]
    p = expand_params_short(params_short, pc, pt)
    a1, b1, g1 = p[0], p[1], p[2]
    a2, b2, g2 = p[4], p[5], p[6]

    if pc == "TT":
        alpha1 = min(a1, g1)
        as1    = alpha_subopt(b1, R, r)
        alpha2 = min(a2, g2)
        as2    = alpha_subopt(b2, R, r)
        return (alpha1 >= as1) or (alpha2 >= as2)
    elif pc in ("TTT", "TTTT"):
        # Middle BCB only (outer segments are determined by inner)
        beta  = b2
        alpha = a2
        return alpha >= alpha_subopt(beta, R, r)
    elif pc == "TST":
        alpha1 = min(a1, g1)
        as1    = alpha_subopt(b1, R, r)
        alpha2 = min(a2, g2)
        as2    = alpha_subopt(b2, R, r)
        return (alpha1 >= as1) or (alpha2 >= as2)
    return False


# ---------------------------------------------------------------------------
# Initial guess — port of guessStartPoint() (barycentric sampling)
# ---------------------------------------------------------------------------

def _barycentric_sample(
    x_l: np.ndarray,
    x_u: np.ndarray,
    A: np.ndarray,
    g_l: np.ndarray,
    g_u: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Vectorised rejection sampler — batched numpy ops, no Python loop.

    Draws a large batch of uniform samples at once, filters with A @ x bounds
    in a single matrix multiply + boolean mask, and returns up to n_samples
    survivors.  This replaces O(n_samples) per-sample numpy API calls with
    three bulk operations.
    """
    x_l = np.asarray(x_l, dtype=float)
    x_u = np.asarray(x_u, dtype=float)
    n = len(x_l)

    # Draw enough extras to survive rejection (heuristic: 20× + floor)
    n_draw = max(n_samples * 20, 2000)
    raw = rng.uniform(x_l, x_u, size=(n_draw, n))

    if len(A) > 0:
        A_arr = np.asarray(A, dtype=float)
        g_l_arr = np.asarray(g_l, dtype=float)
        g_u_arr = np.asarray(g_u, dtype=float)
        Ax = raw @ A_arr.T                        # (n_draw, n_constraints)
        mask = np.all((Ax >= g_l_arr - 1e-8) & (Ax <= g_u_arr + 1e-8), axis=1)
        raw = raw[mask]

    if len(raw) == 0:
        return (x_l + x_u) / 2.0         # fallback: midpoint
    return raw[:n_samples]


def _batch_endpoints(
    samples: np.ndarray,
    candidate: dict,
    R: float,
    r: float,
) -> np.ndarray:
    """Evaluate path endpoint for every row in *samples* (shape N×4).

    Returns an N×3 array of [x, y, h] endpoints, using vectorised numpy
    trig instead of calling path_endpoint() N times in a Python loop.
    The heading returned is the scalar unwrapped value (no modulo).
    """
    pc = candidate["path_class"]
    pt = candidate["path_type"]
    k1, k2 = determine_curvature_params(candidate)
    N = len(samples)
    pi = math.pi
    s = samples  # (N, 4)

    # ------------------------------------------------------------------
    # Inline vectorised expand_params_short for each path class / type.
    # All outputs are 1-D arrays of length N.
    # ------------------------------------------------------------------
    z = np.zeros(N)
    one_pi = np.full(N, pi)

    if pc == "TST":
        if pt == "BCB-S-BCB":
            a1, b1, g1, L = s[:, 0], one_pi, s[:, 1], s[:, 2]
            a2, b2, g2    = g1, one_pi, s[:, 3]
        elif pt == "BCB-S-BC":
            a1, b1, g1, L = s[:, 0], one_pi, s[:, 1], s[:, 2]
            a2, b2, g2    = g1, s[:, 3], z
        elif pt == "BCB-S-B":
            a1, b1, g1, L = s[:, 0], one_pi, s[:, 1], s[:, 2]
            a2, b2, g2    = s[:, 3], z, z
        elif pt == "CB-S-BCB":
            a1, b1, g1, L = z, s[:, 0], s[:, 1], s[:, 2]
            a2, b2, g2    = g1, one_pi, s[:, 3]
        elif pt == "CB-S-BC":
            a1, b1, g1, L = z, s[:, 0], s[:, 1], s[:, 2]
            a2, b2, g2    = g1, s[:, 3], z
        elif pt == "CB-S-B":
            a1, b1, g1, L = z, s[:, 0], s[:, 1], s[:, 2]
            a2, b2, g2    = s[:, 3], z, z
        elif pt == "B-S-BCB":
            a1, b1, g1, L = z, z, s[:, 0], s[:, 1]
            a2, b2, g2    = g1, one_pi, s[:, 3]
        elif pt == "B-S-BC":
            a1, b1, g1, L = z, z, s[:, 0], s[:, 1]
            a2, b2, g2    = g1, s[:, 3], z
        elif pt == "B-S-B":
            a1, b1, g1, L = z, z, s[:, 0], s[:, 1]
            a2, b2, g2    = s[:, 3], z, z
        elif pt == "BCB-S-CB":
            a1, b1, g1, L = s[:, 0], one_pi, s[:, 1], s[:, 2]
            a2, b2, g2    = z, s[:, 3], s[:, 1]
        elif pt == "CB-S-CB":
            a1, b1, g1, L = z, s[:, 0], s[:, 1], s[:, 2]
            a2, b2, g2    = z, s[:, 3], s[:, 1]
        elif pt == "B-S-CB":
            a1, b1, g1, L = z, z, s[:, 0], s[:, 1]
            a2, b2, g2    = z, s[:, 3], s[:, 0]
        elif pt == "BCB-S-BCB-r":
            a1, b1, g1, L = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
            a2, b2, g2    = z, z, z
            b2 = z
        else:
            # Fallback for any unrecognised TST type
            return np.column_stack([
                np.array([path_endpoint(s[i], candidate, R, r) for i in range(N)])
            ])
        # BCB1 → S → BCB2
        psi0 = np.zeros(N)
        t1dx, t1dy, t1dh = _bcb_disp_v(a1, b1, g1, psi0, k1, R, r)
        psi1 = psi0 + t1dh
        sdx  = L * np.cos(psi1)
        sdy  = L * np.sin(psi1)
        t2dx, t2dy, t2dh = _bcb_disp_v(a2, b2, g2, psi1, k2, R, r)
        x = t1dx + sdx + t2dx
        y = t1dy + sdy + t2dy
        h = psi1 + t2dh
        return np.column_stack([x, y, h])

    elif pc == "TT":
        if pt == "BCB-BCB":
            a1, b1, g1 = s[:, 0], one_pi, s[:, 1]
            a2, b2, g2 = s[:, 2], one_pi, s[:, 3]
        elif pt == "BCB-BC":
            a1, b1, g1 = s[:, 0], one_pi, s[:, 1]
            a2, b2, g2 = s[:, 2], s[:, 3], z
        elif pt == "BCB-B":
            a1, b1, g1 = s[:, 0], one_pi, s[:, 1]
            a2, b2, g2 = s[:, 2], z, z
        elif pt == "CB-BCB":
            a1, b1, g1 = z, s[:, 0], s[:, 1]
            a2, b2, g2 = s[:, 2], one_pi, s[:, 3]
        elif pt == "CB-BC":
            a1, b1, g1 = z, s[:, 0], s[:, 1]
            a2, b2, g2 = s[:, 2], s[:, 3], z
        elif pt == "CB-B":
            a1, b1, g1 = z, s[:, 0], s[:, 1]
            a2, b2, g2 = s[:, 2], z, z
        elif pt == "B-BCB":
            a1, b1, g1 = z, z, s[:, 0]
            a2, b2, g2 = s[:, 1], one_pi, s[:, 3]
        elif pt == "B-BC":
            a1, b1, g1 = z, z, s[:, 0]
            a2, b2, g2 = s[:, 1], s[:, 3], z
        elif pt == "B-B":
            a1, b1, g1 = z, z, s[:, 0]
            a2, b2, g2 = s[:, 1], z, z
        else:
            return np.column_stack([
                np.array([path_endpoint(s[i], candidate, R, r) for i in range(N)])
            ])
        psi0 = np.zeros(N)
        t1dx, t1dy, t1dh = _bcb_disp_v(a1, b1, g1, psi0, k1, R, r)
        psi1 = psi0 + t1dh
        t2dx, t2dy, t2dh = _bcb_disp_v(a2, b2, g2, psi1, k2, R, r)
        x = t1dx + t2dx
        y = t1dy + t2dy
        h = psi1 + t2dh
        return np.column_stack([x, y, h])

    elif pc == "TTT":
        # Three BCBs: k1, -k1, k1
        if pt == "CB-T-BCB":
            a1, b1, g1 = z, s[:, 0], s[:, 1]
            g1_ = g1
            a2, b2, g2 = g1_, s[:, 2], g1_
            a3, b3, g3 = g1_, b2,     s[:, 3]
        elif pt == "CB-T-BC":
            a1, b1, g1 = z, s[:, 0], s[:, 1]
            g1_ = g1
            a2, b2, g2 = g1_, s[:, 2], g1_
            a3, b3, g3 = g1_, s[:, 3], z
        elif pt == "CB-T-B":
            a1, b1, g1 = z, s[:, 0], s[:, 1]
            g1_ = g1
            a2, b2, g2 = g1_, s[:, 2], g1_
            a3, b3, g3 = s[:, 3], z, z
        elif pt == "B-T-BCB":
            a1, b1, g1 = z, z, s[:, 0]
            g1_ = g1
            a2, b2, g2 = s[:, 1], s[:, 2], s[:, 1]
            a3, b3, g3 = s[:, 1], s[:, 2], s[:, 3]
        elif pt == "B-T-BC":
            a1, b1, g1 = z, z, s[:, 0]
            a2, b2, g2 = s[:, 1], s[:, 2], s[:, 1]
            a3, b3, g3 = s[:, 1], s[:, 3], z
        elif pt == "B-T-B":
            a1, b1, g1 = z, z, s[:, 0]
            a2, b2, g2 = s[:, 1], s[:, 2], s[:, 1]
            a3, b3, g3 = s[:, 3], z, z
        elif pt == "BCB-T-BCB":
            a1, b1, g1 = s[:, 0], s[:, 1], s[:, 2]
            a2, b2, g2 = g1, b1, g1
            a3, b3, g3 = g1, b1, s[:, 3]
        elif pt == "BCB-T-BC":
            a1, b1, g1 = s[:, 0], s[:, 1], s[:, 2]
            a2, b2, g2 = g1, b1, g1
            a3, b3, g3 = g1, s[:, 3], z
        elif pt == "BCB-T-B":
            a1, b1, g1 = s[:, 0], s[:, 1], s[:, 2]
            a2, b2, g2 = g1, b1, g1
            a3, b3, g3 = s[:, 3], z, z
        else:
            return np.array([path_endpoint(s[i], candidate, R, r) for i in range(N)])
        psi0 = np.zeros(N)
        t1dx, t1dy, t1dh = _bcb_disp_v(a1, b1, g1, psi0, k1, R, r)
        psi1 = psi0 + t1dh
        t2dx, t2dy, t2dh = _bcb_disp_v(a2, b2, g2, psi1, -k1, R, r)
        psi2 = psi1 + t2dh
        t3dx, t3dy, t3dh = _bcb_disp_v(a3, b3, g3, psi2, k1, R, r)
        x = t1dx + t2dx + t3dx
        y = t1dy + t2dy + t3dy
        h = psi2 + t3dh
        return np.column_stack([x, y, h])

    elif pc == "TTTT":
        # Four BCBs: k1, -k1, k1, -k1; middle pair always a3=a2, b3=b2, g3=a2
        if pt == "CB-TT-BC":
            a1, b1, g1 = z, s[:, 0], s[:, 1]
            g1_ = g1
            a2, b2, g2 = g1_, s[:, 2], g1_
            a3, b3, g3 = g1_, s[:, 2], g1_
            a4, b4, g4 = g1_, s[:, 3], z
        elif pt == "CB-TT-B":
            a1, b1, g1 = z, s[:, 0], s[:, 1]
            g1_ = g1
            a2, b2, g2 = g1_, s[:, 2], g1_
            a3, b3, g3 = g1_, s[:, 2], g1_
            a4, b4, g4 = s[:, 3], z, z
        elif pt == "B-TT-BCB":
            a1, b1, g1 = z, z, s[:, 0]
            a2_, b2_ = s[:, 1], s[:, 2]
            a2, b2, g2 = a2_, b2_, a2_
            a3, b3, g3 = a2_, b2_, a2_
            a4, b4, g4 = a2_, b2_, s[:, 3]
        elif pt == "B-TT-BC":
            a1, b1, g1 = z, z, s[:, 0]
            a2_, b2_ = s[:, 1], s[:, 2]
            a2, b2, g2 = a2_, b2_, a2_
            a3, b3, g3 = a2_, b2_, a2_
            a4, b4, g4 = a2_, s[:, 3], z
        elif pt == "B-TT-B":
            a1, b1, g1 = z, z, s[:, 0]
            a2_, b2_ = s[:, 1], s[:, 2]
            a2, b2, g2 = a2_, b2_, a2_
            a3, b3, g3 = a2_, b2_, a2_
            a4, b4, g4 = s[:, 3], z, z
        elif pt == "BCB-TT-BCB":
            a1, b1, g1 = s[:, 0], s[:, 1], s[:, 2]
            a2, b2, g2 = g1, b1, g1
            a3, b3, g3 = g1, b1, g1
            a4, b4, g4 = g1, b1, s[:, 3]
        elif pt == "BCB-TT-BC":
            a1, b1, g1 = s[:, 0], s[:, 1], s[:, 2]
            a2, b2, g2 = g1, b1, g1
            a3, b3, g3 = g1, b1, g1
            a4, b4, g4 = g1, s[:, 3], z
        elif pt == "BCB-TT-B":
            a1, b1, g1 = s[:, 0], s[:, 1], s[:, 2]
            a2, b2, g2 = g1, b1, g1
            a3, b3, g3 = g1, b1, g1
            a4, b4, g4 = s[:, 3], z, z
        else:
            return np.array([path_endpoint(s[i], candidate, R, r) for i in range(N)])
        psi0 = np.zeros(N)
        t1dx, t1dy, t1dh = _bcb_disp_v(a1, b1, g1, psi0, k1, R, r)
        psi1 = psi0 + t1dh
        t2dx, t2dy, t2dh = _bcb_disp_v(a2, b2, g2, psi1, -k1, R, r)
        psi2 = psi1 + t2dh
        t3dx, t3dy, t3dh = _bcb_disp_v(a3, b3, g3, psi2, k1, R, r)
        psi3 = psi2 + t3dh
        t4dx, t4dy, t4dh = _bcb_disp_v(a4, b4, g4, psi3, -k1, R, r)
        x = t1dx + t2dx + t3dx + t4dx
        y = t1dy + t2dy + t3dy + t4dy
        h = psi3 + t4dh
        return np.column_stack([x, y, h])

    # Unrecognised path class — scalar fallback (should not be reached)
    return np.array([path_endpoint(s[i], candidate, R, r) for i in range(N)])


def _bcb_disp_v(a_u, b_u, g_u, psi0, sgn_k, R, r):
    """Vectorised BCB displacement.  All args are arrays of shape (N,).

    Returns (dx, dy, dpsi) each of shape (N,).
    """
    a = a_u * sgn_k
    b = b_u * sgn_k
    g = g_u * sgn_k
    dx = sgn_k * (
        R * (np.sin(a + b + g + psi0) - np.sin(psi0))
        + (R - r) * (np.sin(a + psi0) - np.sin(psi0 + a + b))
    )
    dy = sgn_k * (
        R * (np.cos(psi0) - np.cos(a + b + g + psi0))
        + (R - r) * (np.cos(psi0 + a + b) - np.cos(a + psi0))
    )
    dpsi = a + b + g
    return dx, dy, dpsi


def guess_start_point(
    candidate: dict,
    R: float,
    r: float,
    x_l: np.ndarray,
    x_u: np.ndarray,
    g_l: np.ndarray,
    g_u: np.ndarray,
    A: np.ndarray,
    state_final: np.ndarray,
    n_samples: int = 250,
    seed: int | None = None,
) -> np.ndarray:
    """Find best initial guess via barycentric sampling."""
    rng = np.random.default_rng(seed)
    samples = _barycentric_sample(x_l, x_u, A, g_l, g_u, n_samples, rng)
    if samples.ndim == 1:
        return samples                   # midpoint fallback (single row)

    R_scale = math.sqrt(state_final[0] ** 2 + state_final[1] ** 2)
    if R_scale < 1e-10:
        R_scale = 1.0

    # Evaluate all sample endpoints in one vectorised call, then pick the
    # sample whose endpoint is closest to the target state.
    eps = _batch_endpoints(samples, candidate, R, r)   # (N, 3)
    xe  = (eps[:, 0] - state_final[0]) ** 2
    ye  = (eps[:, 1] - state_final[1]) ** 2
    # Circular heading distance (sin/cos, avoids modulo discontinuity)
    he  = np.sqrt(
        (np.sin(eps[:, 2]) - math.sin(state_final[2])) ** 2
        + (np.cos(eps[:, 2]) - math.cos(state_final[2])) ** 2
    )
    err = np.sqrt(xe + ye) / R_scale + he / 2.0
    return samples[np.argmin(err)]


# ---------------------------------------------------------------------------
# CasADi symbolic endpoint — for NLP constraints
# ---------------------------------------------------------------------------

def _ca_bcb(a_u, b_u, g_u, psi0, sgn_k, R, r):
    """CasADi BCB displacement."""
    a = a_u * sgn_k
    b = b_u * sgn_k
    g = g_u * sgn_k
    dx = sgn_k * (R * (ca.sin(a + b + g + psi0) - ca.sin(psi0))
                  + (R - r) * (ca.sin(a + psi0) - ca.sin(psi0 + a + b)))
    dy = sgn_k * (R * (ca.cos(psi0) - ca.cos(a + b + g + psi0))
                  + (R - r) * (ca.cos(psi0 + a + b) - ca.cos(a + psi0)))
    dpsi = a + b + g
    return ca.vertcat(dx, dy, dpsi)


def _ca_s(psi, L):
    return ca.vertcat(L * ca.cos(psi), L * ca.sin(psi), 0.0)


def _ca_expand(p, path_class, path_type):
    """CasADi symbolic version of expand_params_short."""
    pi = math.pi
    z  = ca.MX(0.0)
    a1 = b1 = g1 = L = a2 = b2 = g2 = a3 = b3 = g3 = a4 = b4 = g4 = z

    pc, pt = path_class, path_type

    if pc == "TST":
        if pt == "BCB-S-BCB":
            a1 = p[0]
            b1 = pi
            g1 = p[1]
            L = p[2]
            a2 = g1
            b2 = pi
            g2 = p[3]
        elif pt == "BCB-S-BC":
            a1 = p[0]
            b1 = pi
            g1 = p[1]
            L = p[2]
            a2 = g1
            b2 = p[3]
            g2 = z
        elif pt == "BCB-S-B":
            a1 = p[0]
            b1 = pi
            g1 = p[1]
            L = p[2]
            a2 = p[3]
            b2 = z
            g2 = z
        elif pt == "CB-S-BCB":
            a1 = z
            b1 = p[0]
            g1 = p[1]
            L = p[2]
            a2 = g1
            b2 = pi
            g2 = p[3]
        elif pt == "CB-S-BC":
            a1 = z
            b1 = p[0]
            g1 = p[1]
            L = p[2]
            a2 = g1
            b2 = p[3]
            g2 = z
        elif pt == "CB-S-B":
            a1 = z
            b1 = p[0]
            g1 = p[1]
            L = p[2]
            a2 = p[3]
            b2 = z
            g2 = z
        elif pt == "B-S-BCB":
            a1 = z
            b1 = z
            g1 = p[0]
            L = p[1]
            a2 = p[2]
            b2 = pi
            g2 = p[3]
        elif pt == "B-S-BC":
            a1 = z
            b1 = z
            g1 = p[0]
            L = p[1]
            a2 = p[2]
            b2 = p[3]
            g2 = z

    elif pc == "TT":
        if pt == "BCB-BCB":
            a1 = p[0]
            b1 = p[1]
            g1 = p[2]
            a2 = g1
            b2 = b1
            g2 = p[3]
        elif pt == "BCB-BC":
            a1 = p[0]
            b1 = p[1]
            g1 = p[2]
            a2 = g1
            b2 = p[3]
            g2 = z
        elif pt == "BCB-B":
            a1 = p[0]
            b1 = p[1]
            g1 = p[2]
            a2 = p[3]
            b2 = z
            g2 = z
        elif pt == "CB-BCB":
            a1 = z
            b1 = p[0]
            g1 = p[1]
            a2 = g1
            b2 = p[2]
            g2 = p[3]
        elif pt == "B-BCB":
            a1 = z
            b1 = z
            g1 = p[0]
            a2 = p[1]
            b2 = p[2]
            g2 = p[3]

    elif pc == "TTT":
        if pt in ("BCB-T-BCB", "BCB-T-BC", "BCB-T-B"):
            a1 = p[0]
            b1 = p[1]
            g1 = p[2]
            a2 = g1
            b2 = b1
            g2 = a2
            a3 = a2
            if pt == "BCB-T-BCB":
                b3 = b1
                g3 = p[3]
            elif pt == "BCB-T-BC":
                b3 = p[3]
                g3 = z
            elif pt == "BCB-T-B":
                a3 = p[3]
                b3 = z
                g3 = z
        elif pt in ("CB-T-BCB", "CB-T-BC", "CB-T-B"):
            a1 = z
            b1 = p[0]
            g1 = p[1]
            a2 = g1
            b2 = p[2]
            g2 = a2
            a3 = a2
            if pt == "CB-T-BCB":
                b3 = b2
                g3 = p[3]
            elif pt == "CB-T-BC":
                b3 = p[3]
                g3 = z
            elif pt == "CB-T-B":
                a3 = p[3]
                b3 = z
                g3 = z
        elif pt in ("B-T-BCB", "B-T-BC", "B-T-B"):
            a1 = z
            b1 = z
            g1 = p[0]
            a2 = p[1]
            b2 = p[2]
            g2 = a2
            a3 = a2
            if pt == "B-T-BCB":
                b3 = b2
                g3 = p[3]
            elif pt == "B-T-BC":
                b3 = p[3]
                g3 = z
            elif pt == "B-T-B":
                a3 = p[3]
                b3 = z
                g3 = z
        # "BCB-T-BC" and "BCB-T-B" are unreachable here — already handled by
        # the `if pt in ("BCB-T-BCB", "BCB-T-BC", "BCB-T-B"):` branch above.
        # elif pt == "BCB-T-BC": ...
        # elif pt == "BCB-T-B":  ...

    elif pc == "TTTT":
        if pt in ("BCB-TT-BCB", "BCB-TT-BC", "BCB-TT-B"):
            a1 = p[0]
            b1 = p[1]
            g1 = p[2]
            a2 = g1
            b2 = b1
            g2 = a2
            if pt == "BCB-TT-BCB":
                a4 = g1
                b4 = b1
                g4 = p[3]
            elif pt == "BCB-TT-BC":
                a4 = g1
                b4 = p[3]
                g4 = z
            elif pt == "BCB-TT-B":
                a4 = p[3]
                b4 = z
                g4 = z
        elif pt in ("CB-TT-BCB", "CB-TT-BC", "CB-TT-B"):
            a1 = z
            b1 = p[0]
            g1 = p[1]
            a2 = g1
            b2 = p[2]
            g2 = a2
            if pt == "CB-TT-BCB":
                a4 = g1
                b4 = b2
                g4 = p[3]
            elif pt == "CB-TT-BC":
                a4 = g1
                b4 = p[3]
                g4 = z
            elif pt == "CB-TT-B":
                a4 = p[3]
                b4 = z
                g4 = z
        elif pt in ("B-TT-BCB", "B-TT-BC", "B-TT-B"):
            a1 = z
            b1 = z
            g1 = p[0]
            a2 = p[1]
            b2 = p[2]
            g2 = a2
            if pt == "B-TT-BCB":
                a4 = a2
                b4 = b2
                g4 = p[3]
            elif pt == "B-TT-BC":
                a4 = a2
                b4 = p[3]
                g4 = z
            elif pt == "B-TT-B":
                a4 = p[3]
                b4 = z
                g4 = z
        a3 = a2
        b3 = b2
        g3 = a2

    return a1, b1, g1, L, a2, b2, g2, a3, b3, g3, a4, b4, g4


def ca_path_endpoint(p, candidate: dict, R: float, r: float, h0: float = 0.0):
    """Build CasADi symbolic endpoint expression (no angle wrapping).

    The heading returned is the raw (unwrapped) accumulated heading.
    """
    k1, k2 = determine_curvature_params(candidate)
    pc = candidate["path_class"]
    (a1, b1, g1, L,
     a2, b2, g2,
     a3, b3, g3,
     a4, b4, g4) = _ca_expand(p, pc, candidate["path_type"])

    origin = ca.vertcat(0.0, 0.0, h0)

    if pc == "TST":
        t1   = _ca_bcb(a1, b1, g1, h0, k1, R, r)
        psi1 = h0 + k1 * (a1 + b1 + g1)
        s    = _ca_s(psi1, L)
        t2   = _ca_bcb(a2, b2, g2, psi1, k2, R, r)
        ep   = origin + t1 + s + t2

    elif pc == "TT":
        t1   = _ca_bcb(a1, b1, g1, h0, k1, R, r)
        psi1 = h0 + k1 * (a1 + b1 + g1)
        t2   = _ca_bcb(a2, b2, g2, psi1, k2, R, r)
        ep   = origin + t1 + t2

    elif pc == "TTT":
        t1   = _ca_bcb(a1, b1, g1, h0, k1, R, r)
        psi1 = h0 + k1 * (a1 + b1 + g1)
        t2   = _ca_bcb(a2, b2, g2, psi1, -k1, R, r)
        psi2 = psi1 - k1 * (a2 + b2 + g2)
        t3   = _ca_bcb(a3, b3, g3, psi2, k1, R, r)
        ep   = origin + t1 + t2 + t3

    elif pc == "TTTT":
        t1   = _ca_bcb(a1, b1, g1, h0, k1, R, r)
        psi1 = h0 + k1 * (a1 + b1 + g1)
        t2   = _ca_bcb(a2, b2, g2, psi1, -k1, R, r)
        psi2 = psi1 - k1 * (a2 + b2 + g2)
        t3   = _ca_bcb(a3, b3, g3, psi2, k1, R, r)
        psi3 = psi2 + k1 * (a3 + b3 + g3)
        t4   = _ca_bcb(a4, b4, g4, psi3, -k1, R, r)
        ep   = origin + t1 + t2 + t3 + t4

    else:
        raise ValueError(f"Unknown path_class {pc!r}")
    return ep


# ---------------------------------------------------------------------------
# Build and solve NLP for a single candidate
# ---------------------------------------------------------------------------

def solve_candidate(
    candidate: dict,
    R: float,
    r: float,
    xf: float,
    yf: float,
    hf: float,
    L_max: float = 10.0,
    n_samples: int = 250,
    tol: float = 1e-7,
    seed: int | None = None,
) -> dict:
    """Build and solve the NLP for a single VSD candidate.

    Returns a dict with keys:
        feasible (bool), suboptimal (bool), params (ndarray),
        cost (float), candidate (dict).
    """
    _FAIL = dict(feasible=False, suboptimal=False, params=None,
                 cost=_INF, candidate=candidate)

    # suboptimality threshold
    a_sub = alpha_subopt(math.pi, R, r)

    try:
        x_l, x_u, g_l, g_u, A = linear_constraints(candidate, R, r, a_sub, L_max)
    except (ValueError, KeyError):
        return _FAIL

    # initial guess
    x0 = guess_start_point(
        candidate, R, r, x_l, x_u, g_l, g_u, A,
        np.array([xf, yf, hf]), n_samples=n_samples, seed=seed,
    )

    # estimate heading integer k so constraint matches the unwrapped CasADi expression
    # Use unwrapped endpoint (no % 2pi) to determine the integer winding number
    ep_guess_unwrapped = path_endpoint_unwrapped(x0, candidate, R, r)
    k_h = round((ep_guess_unwrapped[2] - hf) / (2 * math.pi))
    hf_adj = hf + 2 * math.pi * k_h

    # Build a fresh ca.Opti NLP for this call.  A new Opti object is needed
    # per call because ca.Opti is not thread-safe for concurrent use; building
    # it here ensures each thread/invocation has its own independent IPOPT
    # workspace and releases the GIL cleanly during solving.
    try:
        opti = ca.Opti()
        p    = opti.variable(4)
        grad = cost_gradient(candidate, R)

        opti.minimize(
            float(grad[0]) * p[0] + float(grad[1]) * p[1]
            + float(grad[2]) * p[2] + float(grad[3]) * p[3]
        )

        for i in range(4):
            opti.subject_to(p[i] >= float(x_l[i]))
            opti.subject_to(p[i] <= float(x_u[i]))

        if len(A) > 0:
            Aca  = ca.MX(A)
            glin = Aca @ p
            for i in range(len(g_l)):
                opti.subject_to(glin[i] >= float(g_l[i]))
                opti.subject_to(glin[i] <= float(g_u[i]))

        ep = ca_path_endpoint(p, candidate, R, r, h0=0.0)
        opti.subject_to(ep[0] == xf)
        opti.subject_to(ep[1] == yf)
        opti.subject_to(ep[2] == hf_adj)

        opti.set_initial(p, x0)
        opti.solver("ipopt", {
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.tol": tol,
            "ipopt.acceptable_tol": tol * 100,
            "ipopt.max_iter": 500,
            "ipopt.mu_strategy": "adaptive",
            "ipopt.hessian_approximation": "limited-memory",
        })

        sol  = opti.solve()
        popt = np.array(sol.value(p)).ravel()

        grad_v = cost_gradient(candidate, R)
        cost   = float(np.dot(grad_v, popt))

        ep_final = path_endpoint(popt, candidate, R, r)
        res = math.sqrt((ep_final[0] - xf) ** 2 + (ep_final[1] - yf) ** 2)
        dh  = abs((ep_final[2] - hf + math.pi) % (2 * math.pi) - math.pi)
        if res > 1e-4 or dh > 1e-4:
            return _FAIL

        subopt = check_suboptimality(popt, candidate, R, r)
        return dict(feasible=True, suboptimal=subopt,
                    params=popt, cost=cost, candidate=candidate)

    except Exception as exc:
        _log.debug(
            "solve_candidate failed for %s/%s: %s",
            candidate.get("path_class"), candidate.get("path_type"), exc,
        )
        return _FAIL
