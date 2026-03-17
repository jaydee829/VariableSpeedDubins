"""BCB/S kinematic displacement math — pure numpy, no solver dependencies.

Exact port of VSDUtils_pathGeneration.cpp and VSDUtils_expandParamsShort.cpp.
"""

from __future__ import annotations

import math

import numpy as np

from vsd.candidates import determine_curvature_params

# ---------------------------------------------------------------------------
# Elementary displacement functions
# ---------------------------------------------------------------------------

def s_displacement(psi: float, L: float) -> np.ndarray:
    """Straight segment displacement: [dx, dy, dpsi]."""
    return np.array([L * np.cos(psi), L * np.sin(psi), 0.0])


def bcb_displacement(
    a_unsigned: float,
    b_unsigned: float,
    g_unsigned: float,
    psi0: float,
    sgn_k: float,
    R: float,
    r: float,
) -> np.ndarray:
    """BCB arc displacement: [dx, dy, dpsi].

    Exact port of VarSpeedDubins::BCB().
    a/b/g are unsigned arc-angle parameters; sgn_k is ±1.
    """
    a = a_unsigned * sgn_k
    b = b_unsigned * sgn_k
    g = g_unsigned * sgn_k
    dx = sgn_k * (
        R * (np.sin(a + b + g + psi0) - np.sin(psi0))
        + (R - r) * (np.sin(a + psi0) - np.sin(psi0 + a + b))
    )
    dy = sgn_k * (
        R * (np.cos(psi0) - np.cos(a + b + g + psi0))
        + (R - r) * (np.cos(psi0 + a + b) - np.cos(a + psi0))
    )
    dpsi = a + b + g  # = (a_u + b_u + g_u) * sgn_k
    return np.array([dx, dy, dpsi])


# ---------------------------------------------------------------------------
# expand_params_short — exact port of expandParamsShort()
# ---------------------------------------------------------------------------

def expand_params_short(
    p: np.ndarray | list,
    path_class: str,
    path_type: str,
) -> np.ndarray:
    """Expand 4-element short param vector to 13-element long form.

    Long form: [a1, b1, g1, L, a2, b2, g2, a3, b3, g3, a4, b4, g4]
    """
    p = np.asarray(p, dtype=float)
    pi = math.pi
    a1 = b1 = g1 = L = a2 = b2 = g2 = a3 = b3 = g3 = a4 = b4 = g4 = 0.0

    if path_class == "TST":
        if path_type == "BCB-S-BCB":
            a1 = p[0]
            b1 = pi
            g1 = p[1]
            L = p[2]
            a2 = g1
            b2 = pi
            g2 = p[3]
        elif path_type == "BCB-S-BC":
            a1 = p[0]
            b1 = pi
            g1 = p[1]
            L = p[2]
            a2 = g1
            b2 = p[3]
            g2 = 0.0
        elif path_type == "BCB-S-B":
            a1 = p[0]
            b1 = pi
            g1 = p[1]
            L = p[2]
            a2 = p[3]
            b2 = 0.0
            g2 = 0.0
        elif path_type == "CB-S-BCB":
            a1 = 0.0
            b1 = p[0]
            g1 = p[1]
            L = p[2]
            a2 = g1
            b2 = pi
            g2 = p[3]
        elif path_type == "CB-S-BC":
            a1 = 0.0
            b1 = p[0]
            g1 = p[1]
            L = p[2]
            a2 = g1
            b2 = p[3]
            g2 = 0.0
        elif path_type == "CB-S-B":
            a1 = 0.0
            b1 = p[0]
            g1 = p[1]
            L = p[2]
            a2 = p[3]
            b2 = 0.0
            g2 = 0.0
        elif path_type == "B-S-BCB":
            a1 = 0.0
            b1 = 0.0
            g1 = p[0]
            L = p[1]
            a2 = p[2]
            b2 = pi
            g2 = p[3]
        elif path_type == "B-S-BC":
            a1 = 0.0
            b1 = 0.0
            g1 = p[0]
            L = p[1]
            a2 = p[2]
            b2 = p[3]
            g2 = 0.0

    elif path_class == "TT":
        if path_type == "BCB-BCB":
            a1 = p[0]
            b1 = p[1]
            g1 = p[2]
            a2 = g1
            b2 = b1
            g2 = p[3]
        elif path_type == "BCB-BC":
            a1 = p[0]
            b1 = p[1]
            g1 = p[2]
            a2 = g1
            b2 = p[3]
            g2 = 0.0
        elif path_type == "BCB-B":
            a1 = p[0]
            b1 = p[1]
            g1 = p[2]
            a2 = p[3]
            b2 = 0.0
            g2 = 0.0
        elif path_type == "CB-BCB":
            a1 = 0.0
            b1 = p[0]
            g1 = p[1]
            a2 = g1
            b2 = p[2]
            g2 = p[3]
        elif path_type == "CB-BC":
            a1 = 0.0
            b1 = p[0]
            g1 = p[1]
            a2 = g1
            b2 = p[2]
            g2 = 0.0
        elif path_type == "CB-B":
            a1 = 0.0
            b1 = p[0]
            g1 = p[1]
            a2 = p[2]
            b2 = 0.0
            g2 = 0.0
        elif path_type == "B-BCB":
            a1 = 0.0
            b1 = 0.0
            g1 = p[0]
            a2 = p[1]
            b2 = p[2]
            g2 = p[3]
        elif path_type == "B-BC":
            a1 = 0.0
            b1 = 0.0
            g1 = p[0]
            a2 = p[1]
            b2 = p[2]
            g2 = 0.0

    elif path_class == "TTT":
        if path_type == "BCB-T-BCB":
            a1 = p[0]
            b1 = p[1]
            g1 = p[2]
            a2 = g1
            b2 = b1
            g2 = a2
            a3 = a2
            b3 = b1
            g3 = p[3]
        elif path_type == "BCB-T-BC":
            a1 = p[0]
            b1 = p[1]
            g1 = p[2]
            a2 = g1
            b2 = b1
            g2 = a2
            a3 = a2
            b3 = p[3]
            g3 = 0.0
        elif path_type == "BCB-T-B":
            a1 = p[0]
            b1 = p[1]
            g1 = p[2]
            a2 = g1
            b2 = b1
            g2 = a2
            a3 = p[3]
            b3 = 0.0
            g3 = 0.0
        elif path_type == "CB-T-BCB":
            a1 = 0.0
            b1 = p[0]
            g1 = p[1]
            a2 = g1
            b2 = p[2]
            g2 = a2
            a3 = a2
            b3 = b2
            g3 = p[3]
        elif path_type == "CB-T-BC":
            a1 = 0.0
            b1 = p[0]
            g1 = p[1]
            a2 = g1
            b2 = p[2]
            g2 = a2
            a3 = a2
            b3 = p[3]
            g3 = 0.0
        elif path_type == "CB-T-B":
            a1 = 0.0
            b1 = p[0]
            g1 = p[1]
            a2 = g1
            b2 = p[2]
            g2 = a2
            a3 = p[3]
            b3 = 0.0
            g3 = 0.0
        elif path_type == "B-T-BCB":
            a1 = 0.0
            b1 = 0.0
            g1 = p[0]
            a2 = p[1]
            b2 = p[2]
            g2 = a2
            a3 = a2
            b3 = b2
            g3 = p[3]
        elif path_type == "B-T-BC":
            a1 = 0.0
            b1 = 0.0
            g1 = p[0]
            a2 = p[1]
            b2 = p[2]
            g2 = a2
            a3 = a2
            b3 = p[3]
            g3 = 0.0
        elif path_type == "B-T-B":
            a1 = 0.0
            b1 = 0.0
            g1 = p[0]
            a2 = p[1]
            b2 = p[2]
            g2 = a2
            a3 = p[3]
            b3 = 0.0
            g3 = 0.0
        # NOTE: "BCB-T-BC" and "BCB-T-B" are not listed in candidate_list()
        # for the TTT class; the branches below are unreachable dead code
        # retained only as documentation of the C++ structure.
        # elif path_type == "BCB-T-BC":
        #     a1 = p[0]; b1 = p[1]; g1 = p[2]
        #     a2 = g1;  b2 = b1;  g2 = a2; a3 = a2; b3 = p[3]; g3 = 0.0
        # elif path_type == "BCB-T-B":
        #     a1 = p[0]; b1 = p[1]; g1 = p[2]
        #     a2 = g1;  b2 = b1;  g2 = a2; a3 = p[3]; b3 = 0.0; g3 = 0.0

    elif path_class == "TTTT":
        if path_type == "BCB-TT-BCB":
            a1 = p[0]
            b1 = p[1]
            g1 = p[2]
            a2 = g1
            b2 = b1
            g2 = g1
            a4 = g1
            b4 = b1
            g4 = p[3]
        elif path_type == "BCB-TT-BC":
            a1 = p[0]
            b1 = p[1]
            g1 = p[2]
            a2 = g1
            b2 = b1
            g2 = a2
            a4 = g1
            b4 = p[3]
            g4 = 0.0
        elif path_type == "BCB-TT-B":
            a1 = p[0]
            b1 = p[1]
            g1 = p[2]
            a2 = g1
            b2 = b1
            g2 = a2
            a4 = p[3]
            b4 = 0.0
            g4 = 0.0
        elif path_type == "CB-TT-BCB":
            a1 = 0.0
            b1 = p[0]
            g1 = p[1]
            a2 = g1
            b2 = p[2]
            g2 = a2
            a4 = g1
            b4 = b2
            g4 = p[3]
        elif path_type == "CB-TT-BC":
            a1 = 0.0
            b1 = p[0]
            g1 = p[1]
            a2 = g1
            b2 = p[2]
            g2 = a2
            a4 = g1
            b4 = p[3]
            g4 = 0.0
        elif path_type == "CB-TT-B":
            a1 = 0.0
            b1 = p[0]
            g1 = p[1]
            a2 = g1
            b2 = p[2]
            g2 = a2
            a4 = p[3]
            b4 = 0.0
            g4 = 0.0
        elif path_type == "B-TT-BCB":
            a1 = 0.0
            b1 = 0.0
            g1 = p[0]
            a2 = p[1]
            b2 = p[2]
            g2 = a2
            a4 = a2
            b4 = b2
            g4 = p[3]
        elif path_type == "B-TT-BC":
            a1 = 0.0
            b1 = 0.0
            g1 = p[0]
            a2 = p[1]
            b2 = p[2]
            g2 = a2
            a4 = a2
            b4 = p[3]
            g4 = 0.0
        elif path_type == "B-TT-B":
            a1 = 0.0
            b1 = 0.0
            g1 = p[0]
            a2 = p[1]
            b2 = p[2]
            g2 = a2
            a4 = p[3]
            b4 = 0.0
            g4 = 0.0
        # TTTT always sets a3, b3, g3 from a2, b2, a2
        a3 = a2
        b3 = b2
        g3 = a2

    return np.array([a1, b1, g1, L, a2, b2, g2, a3, b3, g3, a4, b4, g4])


# ---------------------------------------------------------------------------
# path_endpoint — exact port of VarSpeedDubins::pathEndpoint()
# ---------------------------------------------------------------------------

def path_endpoint(
    params_short: np.ndarray | list,
    candidate: dict,
    R: float,
    r: float,
    x0: float = 0.0,
    y0: float = 0.0,
    h0: float = 0.0,
) -> np.ndarray:
    """Compute path endpoint given short params and candidate spec.

    Returns [x, y, h] in the frame where the path starts at (x0, y0, h0).
    The heading h is wrapped to [0, 2π).
    """
    k1, k2 = determine_curvature_params(candidate)
    p = expand_params_short(params_short, candidate["path_class"], candidate["path_type"])
    a1, b1, g1 = p[0], p[1], p[2]
    L          = p[3]
    a2, b2, g2 = p[4], p[5], p[6]
    a3, b3, g3 = p[7], p[8], p[9]
    a4, b4, g4 = p[10], p[11], p[12]

    pc = candidate["path_class"]

    if pc == "TST":
        t1  = bcb_displacement(a1, b1, g1, h0, k1, R, r)
        psi1 = h0 + k1 * (a1 + b1 + g1)
        s   = s_displacement(psi1, L)
        t2  = bcb_displacement(a2, b2, g2, psi1, k2, R, r)
        ep  = np.array([x0, y0, h0]) + t1 + s + t2

    elif pc == "TT":
        t1  = bcb_displacement(a1, b1, g1, h0, k1, R, r)
        psi1 = h0 + k1 * (a1 + b1 + g1)
        t2  = bcb_displacement(a2, b2, g2, psi1, k2, R, r)
        ep  = np.array([x0, y0, h0]) + t1 + t2

    elif pc == "TTT":
        t1   = bcb_displacement(a1, b1, g1, h0, k1, R, r)
        psi1 = h0 + k1 * (a1 + b1 + g1)
        t2   = bcb_displacement(a2, b2, g2, psi1, -k1, R, r)
        psi2 = psi1 - k1 * (a2 + b2 + g2)
        t3   = bcb_displacement(a3, b3, g3, psi2, k1, R, r)
        ep   = np.array([x0, y0, h0]) + t1 + t2 + t3

    elif pc == "TTTT":
        t1   = bcb_displacement(a1, b1, g1, h0, k1, R, r)
        psi1 = h0 + k1 * (a1 + b1 + g1)
        t2   = bcb_displacement(a2, b2, g2, psi1, -k1, R, r)
        psi2 = psi1 - k1 * (a2 + b2 + g2)
        t3   = bcb_displacement(a3, b3, g3, psi2, k1, R, r)
        psi3 = psi2 + k1 * (a3 + b3 + g3)
        t4   = bcb_displacement(a4, b4, g4, psi3, -k1, R, r)
        ep   = np.array([x0, y0, h0]) + t1 + t2 + t3 + t4
    else:
        raise ValueError(f"Unknown path_class {pc!r}")

    # wrap heading
    ep[2] = ep[2] % (2 * math.pi)
    return ep


def path_endpoint_unwrapped(
    params_short: np.ndarray | list,
    candidate: dict,
    R: float,
    r: float,
    x0: float = 0.0,
    y0: float = 0.0,
    h0: float = 0.0,
) -> np.ndarray:
    """Same as path_endpoint but with NO heading wrapping — matches CasADi."""
    k1, k2 = determine_curvature_params(candidate)
    p = expand_params_short(params_short, candidate["path_class"], candidate["path_type"])
    a1, b1, g1 = p[0], p[1], p[2]
    L          = p[3]
    a2, b2, g2 = p[4], p[5], p[6]
    a3, b3, g3 = p[7], p[8], p[9]
    a4, b4, g4 = p[10], p[11], p[12]

    pc = candidate["path_class"]

    if pc == "TST":
        t1  = bcb_displacement(a1, b1, g1, h0, k1, R, r)
        psi1 = h0 + k1 * (a1 + b1 + g1)
        s   = s_displacement(psi1, L)
        t2  = bcb_displacement(a2, b2, g2, psi1, k2, R, r)
        ep  = np.array([x0, y0, h0]) + t1 + s + t2
    elif pc == "TT":
        t1  = bcb_displacement(a1, b1, g1, h0, k1, R, r)
        psi1 = h0 + k1 * (a1 + b1 + g1)
        t2  = bcb_displacement(a2, b2, g2, psi1, k2, R, r)
        ep  = np.array([x0, y0, h0]) + t1 + t2
    elif pc == "TTT":
        t1   = bcb_displacement(a1, b1, g1, h0, k1, R, r)
        psi1 = h0 + k1 * (a1 + b1 + g1)
        t2   = bcb_displacement(a2, b2, g2, psi1, -k1, R, r)
        psi2 = psi1 - k1 * (a2 + b2 + g2)
        t3   = bcb_displacement(a3, b3, g3, psi2, k1, R, r)
        ep   = np.array([x0, y0, h0]) + t1 + t2 + t3
    elif pc == "TTTT":
        t1   = bcb_displacement(a1, b1, g1, h0, k1, R, r)
        psi1 = h0 + k1 * (a1 + b1 + g1)
        t2   = bcb_displacement(a2, b2, g2, psi1, -k1, R, r)
        psi2 = psi1 - k1 * (a2 + b2 + g2)
        t3   = bcb_displacement(a3, b3, g3, psi2, k1, R, r)
        psi3 = psi2 + k1 * (a3 + b3 + g3)
        t4   = bcb_displacement(a4, b4, g4, psi3, -k1, R, r)
        ep   = np.array([x0, y0, h0]) + t1 + t2 + t3 + t4
    else:
        raise ValueError(f"Unknown path_class {pc!r}")

    # NO wrapping — matches CasADi symbolic expression
    return ep


# ---------------------------------------------------------------------------
# Path history (for visualization)
# ---------------------------------------------------------------------------

def _bcb_path(a_u, b_u, g_u, psi0, sgn_k, R, r, nom_spacing=0.05):
    """Generate path history for a BCB segment."""
    a = a_u * sgn_k
    b = b_u * sgn_k
    g = g_u * sgn_k
    pts = []
    # alpha arc
    if a_u > 1e-10:
        n = max(2, int(a_u / nom_spacing) + 1)
        for t in np.linspace(0, a, n):
            x = sgn_k * R * (np.sin(t + psi0) - np.sin(psi0))
            y = sgn_k * R * (np.cos(psi0) - np.cos(t + psi0))
            pts.append((x, y, t))
    # beta arc
    psi_a = psi0 + a
    if b_u > 1e-10:
        n = max(2, int(b_u / nom_spacing) + 1)
        prev_x = sgn_k * R * (np.sin(a + psi0) - np.sin(psi0))
        prev_y = sgn_k * R * (np.cos(psi0) - np.cos(a + psi0))
        for t in np.linspace(0, b, n):
            x = prev_x + sgn_k * r * (np.sin(t + psi_a) - np.sin(psi_a))
            y = prev_y + sgn_k * r * (np.cos(psi_a) - np.cos(t + psi_a))
            pts.append((x, y, a + t))
    # gamma arc
    psi_ab = psi_a + b
    if g_u > 1e-10:
        disp_ab = bcb_displacement(a_u, b_u, 0.0, psi0, sgn_k, R, r)
        base_x, base_y = disp_ab[0], disp_ab[1]
        n = max(2, int(g_u / nom_spacing) + 1)
        for t in np.linspace(0, g, n):
            x = base_x + sgn_k * R * (np.sin(t + psi_ab) - np.sin(psi_ab))
            y = base_y + sgn_k * R * (np.cos(psi_ab) - np.cos(t + psi_ab))
            pts.append((x, y, a + b + t))
    if not pts:
        pts.append((0.0, 0.0, 0.0))
    return np.array(pts)


def path_history(
    params_short: np.ndarray | list,
    candidate: dict,
    R: float,
    r: float,
    x0: float = 0.0,
    y0: float = 0.0,
    h0: float = 0.0,
    nom_spacing: float = 0.05,
) -> np.ndarray:
    """Compute dense (x, y, h) path history.  Returns Nx3 array."""
    k1, k2 = determine_curvature_params(candidate)
    p = expand_params_short(params_short, candidate["path_class"], candidate["path_type"])
    a1, b1, g1 = p[0], p[1], p[2]
    L          = p[3]
    a2, b2, g2 = p[4], p[5], p[6]
    a3, b3, g3 = p[7], p[8], p[9]
    a4, b4, g4 = p[10], p[11], p[12]
    pc = candidate["path_class"]

    def accumulate(segments):
        """Stack segments into absolute (x, y, h) coordinates."""
        all_pts = []
        origin = np.array([x0, y0, h0])
        for seg in segments:
            abs_pts = seg + origin
            all_pts.append(abs_pts)
            origin = abs_pts[-1]
        if all_pts:
            return np.vstack(all_pts)
        return np.array([[x0, y0, h0]])

    if pc == "TST":
        psi1 = h0 + k1 * (a1 + b1 + g1)
        n_s = max(2, int(L / nom_spacing) + 1)
        s_pts = np.column_stack([
            np.linspace(0, L * np.cos(psi1), n_s),
            np.linspace(0, L * np.sin(psi1), n_s),
            np.zeros(n_s),
        ])
        t1_raw = _bcb_path(a1, b1, g1, h0, k1, R, r, nom_spacing)
        t2_raw = _bcb_path(a2, b2, g2, psi1, k2, R, r, nom_spacing)
        # absolute positions
        t1_abs = t1_raw + np.array([x0, y0, h0])
        t1_end = t1_abs[-1]
        s_abs = s_pts + t1_end
        s_end = s_abs[-1]
        t2_abs = t2_raw + s_end
        return np.vstack([t1_abs, s_abs[1:], t2_abs[1:]])

    elif pc == "TT":
        psi1 = h0 + k1 * (a1 + b1 + g1)
        t1_raw = _bcb_path(a1, b1, g1, h0, k1, R, r, nom_spacing)
        t2_raw = _bcb_path(a2, b2, g2, psi1, k2, R, r, nom_spacing)
        t1_abs = t1_raw + np.array([x0, y0, h0])
        t2_abs = t2_raw + t1_abs[-1]
        return np.vstack([t1_abs, t2_abs[1:]])

    elif pc == "TTT":
        psi1 = h0 + k1 * (a1 + b1 + g1)
        psi2 = psi1 - k1 * (a2 + b2 + g2)
        t1_raw = _bcb_path(a1, b1, g1, h0, k1, R, r, nom_spacing)
        t2_raw = _bcb_path(a2, b2, g2, psi1, -k1, R, r, nom_spacing)
        t3_raw = _bcb_path(a3, b3, g3, psi2, k1, R, r, nom_spacing)
        t1_abs = t1_raw + np.array([x0, y0, h0])
        t2_abs = t2_raw + t1_abs[-1]
        t3_abs = t3_raw + t2_abs[-1]
        return np.vstack([t1_abs, t2_abs[1:], t3_abs[1:]])

    elif pc == "TTTT":
        psi1 = h0 + k1 * (a1 + b1 + g1)
        psi2 = psi1 - k1 * (a2 + b2 + g2)
        psi3 = psi2 + k1 * (a3 + b3 + g3)
        t1_raw = _bcb_path(a1, b1, g1, h0, k1, R, r, nom_spacing)
        t2_raw = _bcb_path(a2, b2, g2, psi1, -k1, R, r, nom_spacing)
        t3_raw = _bcb_path(a3, b3, g3, psi2, k1, R, r, nom_spacing)
        t4_raw = _bcb_path(a4, b4, g4, psi3, -k1, R, r, nom_spacing)
        t1_abs = t1_raw + np.array([x0, y0, h0])
        t2_abs = t2_raw + t1_abs[-1]
        t3_abs = t3_raw + t2_abs[-1]
        t4_abs = t4_raw + t3_abs[-1]
        return np.vstack([t1_abs, t2_abs[1:], t3_abs[1:], t4_abs[1:]])

    raise ValueError(f"Unknown path_class {pc!r}")
