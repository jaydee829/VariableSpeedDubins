"""Tests for vsd.kinematics."""

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from vsd.kinematics import bcb_displacement, expand_params_short, path_endpoint, s_displacement

R = 1.0
r = 0.3


# ---------------------------------------------------------------------------
# Basic displacement tests
# ---------------------------------------------------------------------------

def test_s_displacement_zero():
    d = s_displacement(0.0, 0.0)
    assert np.allclose(d, [0, 0, 0])


def test_s_displacement_forward():
    d = s_displacement(0.0, 5.0)
    assert abs(d[0] - 5.0) < 1e-12
    assert abs(d[1]) < 1e-12
    assert abs(d[2]) < 1e-12


def test_bcb_zero_arcs():
    """Zero arcs → zero displacement."""
    d = bcb_displacement(0, 0, 0, 0.0, 1.0, R, r)
    assert np.allclose(d, [0, 0, 0])


@pytest.mark.parametrize("sgn_k", [1.0, -1.0])
def test_bcb_heading_change(sgn_k):
    """dpsi = sgn_k * (a + b + g) for any valid triple and either turn sign."""
    a, b, g = 0.2, math.pi, 0.4
    d = bcb_displacement(a, b, g, 0.0, sgn_k, R, r)
    assert abs(d[2] - sgn_k * (a + b + g)) < 1e-12


def test_bcb_equals_rr_circle_when_r_equals_R():
    """With r=R BCB is just an arc of a single circle."""
    a, b, g, psi0, sgn_k = 0.5, math.pi, 0.3, 0.1, 1.0
    d = bcb_displacement(a, b, g, psi0, sgn_k, R, R)
    # arc of radius R through total angle (a+b+g)*sgn_k
    total = a + b + g
    dx_exp = R * (math.sin(total + psi0) - math.sin(psi0))
    dy_exp = R * (math.cos(psi0) - math.cos(total + psi0))
    assert abs(d[0] - dx_exp) < 1e-12
    assert abs(d[1] - dy_exp) < 1e-12


# ---------------------------------------------------------------------------
# expand_params_short
# ---------------------------------------------------------------------------

def test_expand_bcb_s_bcb():
    p = [0.2, 0.3, 1.5, 0.4]
    pl = expand_params_short(p, "TST", "BCB-S-BCB")
    assert abs(pl[1] - math.pi) < 1e-12   # b1 == pi
    assert abs(pl[5] - math.pi) < 1e-12   # b2 == pi
    assert abs(pl[4] - pl[2]) < 1e-12     # a2 == g1


def test_expand_tst_b_s_b():
    p = [0.5, 1.2, 0.3, 0.7]
    pl = expand_params_short(p, "TST", "BCB-S-B")
    assert abs(pl[0] - p[0]) < 1e-12     # a1
    assert abs(pl[1] - math.pi) < 1e-12  # b1 = pi
    assert abs(pl[7]) < 1e-12            # a3 = 0


def test_expand_tt_bcb_bcb():
    p = [0.3, 0.8, 0.5, 0.2]
    pl = expand_params_short(p, "TT", "BCB-BCB")
    assert abs(pl[4] - pl[2]) < 1e-12   # a2 == g1
    assert abs(pl[5] - pl[1]) < 1e-12   # b2 == b1


# ---------------------------------------------------------------------------
# path_endpoint — symmetry and self-consistency
# ---------------------------------------------------------------------------

def test_endpoint_zero_params():
    """All-zero params → endpoint at origin."""
    cand = {"path_class": "TST", "path_type": "BCB-S-BCB", "orientation": "LSL"}
    ep = path_endpoint([0, 0, 0, 0], cand, R, r)
    assert abs(ep[0]) < 1e-12
    assert abs(ep[1]) < 1e-12


def test_endpoint_lsl_vs_rsr_symmetry():
    """LSL and RSR should produce mirror-image paths."""
    p = [0.2, 0.3, 1.0, 0.2]
    c_lsl = {"path_class": "TST", "path_type": "BCB-S-BCB", "orientation": "LSL"}
    c_rsr = {"path_class": "TST", "path_type": "BCB-S-BCB", "orientation": "RSR"}
    ep_lsl = path_endpoint(p, c_lsl, R, r)
    ep_rsr = path_endpoint(p, c_rsr, R, r)
    # x same, y negated (y reflection), heading negated mod 2pi
    assert abs(ep_lsl[0] - ep_rsr[0]) < 1e-10
    assert abs(ep_lsl[1] + ep_rsr[1]) < 1e-10


@given(
    a=st.floats(0.01, 0.5), g=st.floats(0.01, 0.5),
    L=st.floats(0.0, 5.0), g2=st.floats(0.01, 0.5),
)
@settings(max_examples=100)
def test_endpoint_cost_positive(a, g, L, g2):
    """Cost (path length) should always be positive for non-zero params."""
    from vsd.nlp import cost_gradient
    p = [a, g, L, g2]
    cand = {"path_class": "TST", "path_type": "BCB-S-BCB", "orientation": "LSL"}
    grad = cost_gradient(cand, R)
    cost = float(np.dot(grad, p))
    assert cost >= 0


# ---------------------------------------------------------------------------
# T-1: Numerical reference values — C++ testSolveSinglePath params
# ---------------------------------------------------------------------------

def test_path_endpoint_reference_values():
    """Regression: path_endpoint for C++ reference params {0.2, 0.5, 0.7, 0.3}.

    Ground-truth values computed from the BCB/S kinematics formulas directly.
    R=1.0, r=0.3, TST/BCB-S-BCB/LSL.
    """
    cand = {"path_class": "TST", "path_type": "BCB-S-BCB", "orientation": "LSL"}
    p = [0.2, 0.5, 0.7, 0.3]
    ep = path_endpoint(p, cand, R, r)

    # Endpoint must be finite and heading in [0, 2π)
    assert np.all(np.isfinite(ep))
    assert 0 <= ep[2] < 2 * math.pi

    # Reference values derived from the closed-form BCB formula (see plan doc).
    # Verified once and locked in as a regression guard.
    assert abs(ep[0] - (-0.5646)) < 1e-3, f"x endpoint {ep[0]:.4f} differs from reference"
    assert abs(ep[1] - (-0.3863)) < 1e-3, f"y endpoint {ep[1]:.4f} differs from reference"
    assert abs(ep[2] - 1.5) < 1e-3,       f"h endpoint {ep[2]:.4f} differs from reference"
