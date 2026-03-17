"""Tests for vsd.problem.VSDProblem."""

import math

import pytest

from vsd.problem import VSDProblem


def test_basic_construction():
    p = VSDProblem(0, 0, 0, 5, 3, 1.0, v_max=2.0, v_min=0.5, omega_max=1.0)
    assert abs(p.R - 2.0) < 1e-12
    assert abs(p.r - 0.5) < 1e-12
    assert p.x0 == 0.0
    assert p.xf == 5.0


def test_heading_wrapped():
    """Headings outside [0, 2pi) are wrapped."""
    p = VSDProblem(0, 0, -0.1, 1, 1, 3 * math.pi,
                   v_max=1.0, v_min=0.3, omega_max=1.0)
    assert 0 <= p.h0 < 2 * math.pi
    assert 0 <= p.hf < 2 * math.pi


@pytest.mark.parametrize("v_max,v_min,omega_max", [
    (0.5, 1.0, 1.0),   # v_min > v_max
    (1.0, 0.5, 0.0),   # omega_max = 0
])
def test_invalid_construction(v_max, v_min, omega_max):
    with pytest.raises(ValueError):
        VSDProblem(0, 0, 0, 1, 1, 0, v_max=v_max, v_min=v_min, omega_max=omega_max)


def test_normalized():
    """Normalized problem has x0=y0=h0=0."""
    p = VSDProblem(1, 2, 0.5, 4, 6, 1.2, v_max=2.0, v_min=0.6, omega_max=1.5)
    n = p.normalized()
    assert abs(n.x0) < 1e-12
    assert abs(n.y0) < 1e-12
    assert abs(n.h0) < 1e-12


def test_normalized_heading_wraparound():
    """T-2: normalized() final heading stays in [0, 2π) after frame rotation."""
    # h0 = 3π/2 (pointing down), hf = -π/4 (outside [0, 2π))
    # After normalization (rotate by -h0), hf_norm = hf - h0 may be negative
    p = VSDProblem(1.0, 2.0, 3 * math.pi / 2, 4.0, 6.0, -math.pi / 4,
                   v_max=1.0, v_min=0.3, omega_max=1.0)
    n = p.normalized()
    assert abs(n.x0) < 1e-12
    assert abs(n.y0) < 1e-12
    assert abs(n.h0) < 1e-12
    assert 0 <= n.hf < 2 * math.pi


def test_relative_frame():
    """relative frame with trivial vehicle == global."""
    p_g = VSDProblem(0, 0, 0, 3, 4, 1.0, v_max=2.0, v_min=0.5, omega_max=1.0,
                     frame="global")
    p_r = VSDProblem(0, 0, 0, 3, 4, 1.0, v_max=2.0, v_min=0.5, omega_max=1.0,
                     frame="relative")
    assert abs(p_g.xf - p_r.xf) < 1e-12
    assert abs(p_g.yf - p_r.yf) < 1e-12
