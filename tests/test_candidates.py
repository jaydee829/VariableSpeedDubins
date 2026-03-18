"""Tests for vsd.candidates."""

import pytest

from vsd.candidates import candidate_list, determine_curvature_params, nlp_dimensions


def test_candidate_list_length():
    assert len(candidate_list()) == 76


def test_candidate_list_has_required_keys():
    for c in candidate_list():
        assert "path_class" in c
        assert "path_type"  in c
        assert "orientation" in c


@pytest.mark.parametrize("path_class,expected_count", [
    ("TST",  32),
    ("TT",   20),
    ("TTT",  16),
    ("TTTT",  8),
])
def test_class_count(path_class, expected_count):
    cl = candidate_list()
    assert len([c for c in cl if c["path_class"] == path_class]) == expected_count


def test_nlp_dimensions_n_always_4():
    for c in candidate_list():
        n, m, nm = nlp_dimensions(c)
        assert n == 4
        assert nm == n * m


@pytest.mark.parametrize("candidate,expected_k1,expected_k2", [
    ({"path_class": "TST", "path_type": "BCB-S-BCB", "orientation": "LSL"},  1.0,  1.0),
    ({"path_class": "TST", "path_type": "BCB-S-BCB", "orientation": "RSR"}, -1.0, -1.0),
    ({"path_class": "TTT", "path_type": "CB-T-B",    "orientation": "LRL"},  1.0, -1.0),
])
def test_curvature_params(candidate, expected_k1, expected_k2):
    k1, k2 = determine_curvature_params(candidate)
    assert k1 == expected_k1 and k2 == expected_k2
