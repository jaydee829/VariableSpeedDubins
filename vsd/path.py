"""VSDPath: holds a solved path with cost, params, and history generation."""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class VSDPath:
    """Holds one solved path candidate.

    Attributes:
        candidate: dict with path_class, path_type, orientation
        params: 4-element short-form parameter vector (or 3-element for Dubins)
        cost: travel-time cost (arc length)
        R: bang radius
        r: corner radius
        feasible: True if the NLP converged and endpoint was satisfied
        suboptimal: True if alpha-suboptimality check flagged it
        is_dubins: True for classical Dubins solutions
        dubins_path: underlying DubinsPath object (if is_dubins=True)
    """
    candidate: dict
    params: Optional[np.ndarray]
    cost: float
    R: float
    r: float
    feasible: bool = True
    suboptimal: bool = False
    is_dubins: bool = False
    dubins_path: object = None  # DubinsPath | None

    # initial state (for history computation)
    x0: float = 0.0
    y0: float = 0.0
    h0: float = 0.0

    def path_history(self, nom_spacing: float = 0.05) -> np.ndarray:
        """Compute dense (x, y, h) path history.  Returns Nx3 array."""
        if self.is_dubins:
            return self._dubins_history(nom_spacing)
        from vsd.kinematics import path_history as kin_history
        return kin_history(
            self.params, self.candidate, self.R, self.r,
            self.x0, self.y0, self.h0, nom_spacing,
        )

    def endpoint(self) -> np.ndarray:
        """Return the path endpoint [x, y, h]."""
        if self.is_dubins and self.dubins_path is not None:
            ep = self.dubins_path.endpoint()
            ep[0] += self.x0
            ep[1] += self.y0
            ep[2]  = (ep[2] + self.h0) % (2 * math.pi)
            return ep
        from vsd.kinematics import path_endpoint
        return path_endpoint(self.params, self.candidate, self.R, self.r,
                              self.x0, self.y0, self.h0)

    def _dubins_history(self, nom_spacing: float) -> np.ndarray:
        """Path history for a Dubins (BSB/BBB) path."""
        dp = self.dubins_path
        if dp is None:
            return np.array([[self.x0, self.y0, self.h0]])
        R = dp.radius
        a, b, c = dp.a, dp.b, dp.c
        pt = dp.path_type
        if pt in ("LSL", "LSR", "RSL", "RSR"):
            k1 = 1.0 if pt.startswith("L") else -1.0
            k2 = 1.0 if pt.endswith("L") else -1.0
            # Point counts based on arc length (a, c) and straight length (b)
            # so that density is consistent across all segment types.
            n1 = max(2, int(a / nom_spacing) + 1) if a > 0 else 2
            n2 = max(2, int(c / nom_spacing) + 1) if c > 0 else 2
            ns = max(2, int(b / nom_spacing) + 1) if b > 0 else 2
            a_ang = a / R
            c_ang = c / R
            pts = []
            h = self.h0
            x, y = self.x0, self.y0
            for t in np.linspace(0, a_ang * k1, n1):
                pts.append((x + R * (np.sin(t + h) - np.sin(h)) * (1 if k1 > 0 else -1),
                             y + R * (np.cos(h) - np.cos(t + h)) * (1 if k1 > 0 else -1),
                             h + t))
            psi1 = h + k1 * a_ang
            x1 = x + k1 * R * (np.sin(k1 * a_ang + h) - np.sin(h))
            y1 = y + k1 * R * (np.cos(h) - np.cos(k1 * a_ang + h))
            for t in np.linspace(0, b, ns):
                pts.append((x1 + t * np.cos(psi1), y1 + t * np.sin(psi1), psi1))
            x2 = x1 + b * np.cos(psi1)
            y2 = y1 + b * np.sin(psi1)
            for t in np.linspace(0, c_ang * k2, n2):
                pts.append((x2 + R * (np.sin(t + psi1) - np.sin(psi1)) * (1 if k2 > 0 else -1),
                             y2 + R * (np.cos(psi1) - np.cos(t + psi1)) * (1 if k2 > 0 else -1),
                             psi1 + t))
            return np.array(pts)
        else:
            return np.array([[self.x0, self.y0, self.h0]])

    def __repr__(self) -> str:
        tag = "Dubins" if self.is_dubins else (
            f"{self.candidate['path_class']}/{self.candidate['path_type']}")
        return (f"VSDPath({tag}, cost={self.cost:.4f}, "
                f"feasible={self.feasible}, suboptimal={self.suboptimal})")
