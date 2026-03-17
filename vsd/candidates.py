"""Enumerate all 76 VSD candidate path specs — exact port of candidateList().

Each candidate is a dict with keys: 'path_class', 'path_type', 'orientation'.
Indices 0-75 are the 76 VSD candidates (matching C++ indices 0-75).
The 8 Dubins candidates (indices 76-83) are provided separately by dubins.py.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Candidate list
# ---------------------------------------------------------------------------

def candidate_list() -> list[dict]:
    """Return the 76 VSD candidate path specs in C++ order."""
    cl = [None] * 76

    def c(path_class, path_type, orientation):
        return {"path_class": path_class, "path_type": path_type,
                "orientation": orientation}

    # TST — BCB-S-BCB: 0-3
    cl[0]  = c("TST", "BCB-S-BCB", "LSL")
    cl[1]  = c("TST", "BCB-S-BCB", "LSR")
    cl[2]  = c("TST", "BCB-S-BCB", "RSL")
    cl[3]  = c("TST", "BCB-S-BCB", "RSR")
    # TST — BCB-S-BC: 4-7
    cl[4]  = c("TST", "BCB-S-BC", "LSL")
    cl[5]  = c("TST", "BCB-S-BC", "LSR")
    cl[6]  = c("TST", "BCB-S-BC", "RSL")
    cl[7]  = c("TST", "BCB-S-BC", "RSR")
    # TST — BCB-S-B: 8-11
    cl[8]  = c("TST", "BCB-S-B", "LSL")
    cl[9]  = c("TST", "BCB-S-B", "LSR")
    cl[10] = c("TST", "BCB-S-B", "RSL")
    cl[11] = c("TST", "BCB-S-B", "RSR")
    # TST — CB-S-BCB: 12-15
    cl[12] = c("TST", "CB-S-BCB", "LSL")
    cl[13] = c("TST", "CB-S-BCB", "LSR")
    cl[14] = c("TST", "CB-S-BCB", "RSL")
    cl[15] = c("TST", "CB-S-BCB", "RSR")
    # TST — CB-S-BC: 16-19
    cl[16] = c("TST", "CB-S-BC", "LSL")
    cl[17] = c("TST", "CB-S-BC", "LSR")
    cl[18] = c("TST", "CB-S-BC", "RSL")
    cl[19] = c("TST", "CB-S-BC", "RSR")
    # TST — CB-S-B: 20-23
    cl[20] = c("TST", "CB-S-B", "LSL")
    cl[21] = c("TST", "CB-S-B", "LSR")
    cl[22] = c("TST", "CB-S-B", "RSL")
    cl[23] = c("TST", "CB-S-B", "RSR")
    # TST — B-S-BCB: 24-27
    cl[24] = c("TST", "B-S-BCB", "LSL")
    cl[25] = c("TST", "B-S-BCB", "LSR")
    cl[26] = c("TST", "B-S-BCB", "RSL")
    cl[27] = c("TST", "B-S-BCB", "RSR")
    # TST — B-S-BC: 28-31
    cl[28] = c("TST", "B-S-BC", "LSL")
    cl[29] = c("TST", "B-S-BC", "LSR")
    cl[30] = c("TST", "B-S-BC", "RSL")
    cl[31] = c("TST", "B-S-BC", "RSR")

    # TT — BCB-BCB: 32-35
    cl[32] = c("TT", "BCB-BCB", "LL")
    cl[33] = c("TT", "BCB-BCB", "LR")
    cl[34] = c("TT", "BCB-BCB", "RL")
    cl[35] = c("TT", "BCB-BCB", "RR")
    # TT — BCB-BC: 36-39
    cl[36] = c("TT", "BCB-BC", "LL")
    cl[37] = c("TT", "BCB-BC", "LR")
    cl[38] = c("TT", "BCB-BC", "RL")
    cl[39] = c("TT", "BCB-BC", "RR")
    # TT — BCB-B: 40-43
    cl[40] = c("TT", "BCB-B", "LL")
    cl[41] = c("TT", "BCB-B", "LR")
    cl[42] = c("TT", "BCB-B", "RL")
    cl[43] = c("TT", "BCB-B", "RR")
    # TT — CB-BCB: 44-47
    cl[44] = c("TT", "CB-BCB", "LL")
    cl[45] = c("TT", "CB-BCB", "LR")
    cl[46] = c("TT", "CB-BCB", "RL")
    cl[47] = c("TT", "CB-BCB", "RR")
    # TT — B-BCB: 48-51
    cl[48] = c("TT", "B-BCB", "LL")
    cl[49] = c("TT", "B-BCB", "LR")
    cl[50] = c("TT", "B-BCB", "RL")
    cl[51] = c("TT", "B-BCB", "RR")

    # TTTT — CB-TT-BC: 52-53
    cl[52] = c("TTTT", "CB-TT-BC", "LRLR")
    cl[53] = c("TTTT", "CB-TT-BC", "RLRL")
    # TTTT — CB-TT-B: 54-55
    cl[54] = c("TTTT", "CB-TT-B", "LRLR")
    cl[55] = c("TTTT", "CB-TT-B", "RLRL")
    # TTT — CB-T-BCB: 56-57
    cl[56] = c("TTT", "CB-T-BCB", "LRL")
    cl[57] = c("TTT", "CB-T-BCB", "RLR")
    # TTT — CB-T-BC: 58-59
    cl[58] = c("TTT", "CB-T-BC", "LRL")
    cl[59] = c("TTT", "CB-T-BC", "RLR")
    # TTT — CB-T-B: 60-61
    cl[60] = c("TTT", "CB-T-B", "LRL")
    cl[61] = c("TTT", "CB-T-B", "RLR")
    # TTTT — B-TT-BC: 62-63
    cl[62] = c("TTTT", "B-TT-BC", "LRLR")
    cl[63] = c("TTTT", "B-TT-BC", "RLRL")
    # TTTT — B-TT-B: 64-65
    cl[64] = c("TTTT", "B-TT-B", "LRLR")
    cl[65] = c("TTTT", "B-TT-B", "RLRL")
    # TTT — B-T-BCB: 66-67
    cl[66] = c("TTT", "B-T-BCB", "LRL")
    cl[67] = c("TTT", "B-T-BCB", "RLR")
    # TTT — B-T-BC: 68-69
    cl[68] = c("TTT", "B-T-BC", "LRL")
    cl[69] = c("TTT", "B-T-BC", "RLR")
    # TTT — B-T-B: 70-71
    cl[70] = c("TTT", "B-T-B", "LRL")
    cl[71] = c("TTT", "B-T-B", "RLR")
    # TTT — BCB-T-BC: 72-73
    cl[72] = c("TTT", "BCB-T-BC", "LRL")
    cl[73] = c("TTT", "BCB-T-BC", "RLR")
    # TTT — BCB-T-B: 74-75
    cl[74] = c("TTT", "BCB-T-B", "LRL")
    cl[75] = c("TTT", "BCB-T-B", "RLR")

    return cl


# ---------------------------------------------------------------------------
# NLP dimensions — exact port of NLPdimensions()
# ---------------------------------------------------------------------------

def nlp_dimensions(candidate: dict) -> tuple[int, int, int]:
    """Return (n_params, n_constraints, n_jacobian_elements).

    All VSD candidates have n_params=4. n_constraints includes both linear
    inequality constraints and 3 nonlinear equality boundary conditions.
    """
    path_class = candidate["path_class"]
    path_type  = candidate["path_type"]
    n = 4  # always 4 free parameters for VSD

    tst_map = {
        "BCB-S-BCB": 5, "BCB-S-BC": 4, "BCB-S-B": 5,
        "CB-S-BCB":  4, "CB-S-BC":  5, "CB-S-B":  5,
        "B-S-BCB":   5, "B-S-BC":   5,
    }
    tt_map = {
        "BCB-BCB": 6, "BCB-BC": 6, "BCB-B": 6,
        "CB-BCB":  6, "B-BCB":  6,
    }
    ttt_map = {
        "CB-T-BCB": 6, "CB-T-BC": 6, "CB-T-B": 6,
        "B-T-BCB":  6, "B-T-BC":  6, "B-T-B":  6,
        "BCB-T-BC": 6, "BCB-T-B": 6,
    }
    tttt_map = {
        "CB-TT-BC": 6, "CB-TT-B": 6,
        "B-TT-BC":  6, "B-TT-B":  6,
    }

    lookup = {"TST": tst_map, "TT": tt_map, "TTT": ttt_map, "TTTT": tttt_map}
    m = lookup[path_class][path_type]
    return n, m, n * m


# ---------------------------------------------------------------------------
# Curvature sign parameters — port of determineCurvatureParams()
# ---------------------------------------------------------------------------

def determine_curvature_params(candidate: dict) -> tuple[float, float]:
    """Return (k1, k2) signs for the given candidate."""
    path_class  = candidate["path_class"]
    orientation = candidate["orientation"]

    if path_class == "TST":
        return {
            "LSL": (1.0, 1.0), "LSR": (1.0, -1.0),
            "RSL": (-1.0, 1.0), "RSR": (-1.0, -1.0),
        }[orientation]
    elif path_class == "TT":
        return {
            "LL": (1.0, 1.0), "LR": (1.0, -1.0),
            "RL": (-1.0, 1.0), "RR": (-1.0, -1.0),
        }[orientation]
    elif path_class == "TTT":
        return {"LRL": (1.0, -1.0), "RLR": (-1.0, 1.0)}[orientation]
    elif path_class == "TTTT":
        return {"LRLR": (1.0, -1.0), "RLRL": (-1.0, 1.0)}[orientation]
    raise ValueError(f"Unknown path_class {path_class!r}")
