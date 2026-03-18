"""Tests for vsd.nlp — NLP building and suboptimality checks."""

import math

import numpy as np
import pytest

from vsd.candidates import candidate_list
from vsd.nlp import (
    alpha_subopt,
    check_suboptimality,
    cost_gradient,
    linear_constraints,
    solve_candidate,
)

R = 1.0
r = 0.3

# Computed once at module level so parametrize decorators can reference it.
_A_SUB = alpha_subopt(math.pi, R, r)


# ---------------------------------------------------------------------------
# alpha_subopt
# ---------------------------------------------------------------------------

def test_alpha_subopt_pi():
    """alphaSubopt(pi, R, r) matches C++ formula."""
    a = alpha_subopt(math.pi, R, r)
    expected = math.pi - math.pi / 2.0 - math.asin(math.sin(math.pi / 2.0) * (R - r) / (R + r))
    assert abs(a - expected) < 1e-12


@pytest.mark.parametrize("beta", [0.1, 0.5, 1.0, math.pi])
def test_alpha_subopt_positive(beta):
    assert alpha_subopt(beta, R, r) >= 0


# ---------------------------------------------------------------------------
# linear_constraints: shapes
# ---------------------------------------------------------------------------

def test_linear_constraints_shapes():
    for cand in candidate_list():
        x_l, x_u, g_l, g_u, A = linear_constraints(cand, R, r, _A_SUB, 10.0)
        n_lin = len(g_l)
        assert A.shape == (n_lin, 4)
        assert len(x_l) == 4
        assert len(x_u) == 4
        assert np.all(x_l <= x_u + 1e-12)
        assert np.all(np.array(g_l) <= np.array(g_u) + 1e-12)


# ---------------------------------------------------------------------------
# cost_gradient: all candidates return length-4 vector
# ---------------------------------------------------------------------------

def test_cost_gradient_shape():
    for cand in candidate_list():
        g = cost_gradient(cand, R)
        assert len(g) == 4
        assert np.all(g >= 0)


# ---------------------------------------------------------------------------
# check_suboptimality
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cand,p,expected", [
    # TST: a1=g1=a_sub+0.5 → both BCB segments flagged suboptimal
    ({"path_class": "TST", "path_type": "BCB-S-BCB", "orientation": "LSL"},
     [_A_SUB + 0.5, _A_SUB + 0.5, 1.0, 0.1], True),
    # TST: alpha=0 → never suboptimal
    ({"path_class": "TST", "path_type": "BCB-S-BCB", "orientation": "LSL"},
     [0.0, 0.0, 1.0, 0.0], False),
    # TTT (CB-T-BCB): a2=g1=p[1], b2=p[2]; large a2 → suboptimal
    ({"path_class": "TTT", "path_type": "CB-T-BCB", "orientation": "LRL"},
     [0.5, _A_SUB + 0.5, math.pi, 0.5], True),
    # TTT: alpha2=0 → not suboptimal
    ({"path_class": "TTT", "path_type": "CB-T-BCB", "orientation": "LRL"},
     [0.5, 0.0, math.pi, 0.5], False),
    # TTTT (CB-TT-BC): a2=g1=p[1], b2=p[2]; large a2 → suboptimal
    ({"path_class": "TTTT", "path_type": "CB-TT-BC", "orientation": "LRRL"},
     [0.5, _A_SUB + 0.5, math.pi, 0.5], True),
], ids=["TST-subopt", "TST-ok", "TTT-subopt", "TTT-ok", "TTTT-subopt"])
def test_check_suboptimality(cand, p, expected):
    assert check_suboptimality(np.array(p), cand, R, r) == expected


# ---------------------------------------------------------------------------
# solve_candidate: quick smoke test on a known-feasible case
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_solve_simple_candidate():
    """Solve TST/BCB-S-BCB/LSL with a known reachable endpoint."""
    cand = {"path_class": "TST", "path_type": "BCB-S-BCB", "orientation": "LSL"}
    # Use the reference test parameters from C++: R=1.0, r=0.3
    # p=[0.2, 0.5, 0.7, 0.3] → compute endpoint as the target
    from vsd.kinematics import path_endpoint
    p_ref = [0.2, 0.5, 0.7, 0.3]
    ep = path_endpoint(p_ref, cand, R, r)
    res = solve_candidate(cand, R, r, ep[0], ep[1], ep[2],
                          L_max=10.0, n_samples=50, tol=1e-6, seed=42)
    assert res["feasible"], "Expected feasible solution"
    assert res["cost"] > 0
    # optimal cost should be <= the reference feasible cost
    grad = cost_gradient(cand, R)
    ref_cost = float(np.dot(grad, p_ref))
    assert res["cost"] <= ref_cost + 0.05
