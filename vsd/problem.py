"""VSDProblem: problem statement for Variable Speed Dubins solver."""

import numpy as np

from vsd.coordinates import global_to_relative


class VSDProblem:
    """Defines a single VSD planning problem.

    The problem is specified by initial and final states plus vehicle dynamics.
    If ``frame='relative'``, the states are interpreted as being in the vehicle
    body frame (origin at vehicle, heading = 0) and are automatically
    transformed to a global representation before solving.

    Args:
        x0, y0, h0: initial state (position + heading in radians)
        xf, yf, hf: final state
        v_max: maximum speed (m/s or normalized)
        v_min: minimum speed (cornering speed)
        omega_max: maximum turn rate (rad/s)
        frame: ``'global'`` (default) or ``'relative'``
    """

    def __init__(
        self,
        x0: float,
        y0: float,
        h0: float,
        xf: float,
        yf: float,
        hf: float,
        v_max: float,
        v_min: float,
        omega_max: float,
        frame: str = "global",
    ):
        if frame not in ("global", "relative"):
            raise ValueError(f"frame must be 'global' or 'relative', got {frame!r}")
        if v_max <= 0 or v_min <= 0:
            raise ValueError("speeds must be positive")
        if v_min >= v_max:
            raise ValueError("v_min must be less than v_max")
        if omega_max <= 0:
            raise ValueError("omega_max must be positive")

        self.v_max = v_max
        self.v_min = v_min
        self.omega_max = omega_max
        # derived radii
        self.R = v_max / omega_max
        self.r = v_min / omega_max

        if frame == "relative":
            # vehicle is at origin with h=0 in global frame; transform inputs
            vehicle = (0.0, 0.0, 0.0)
            x0_g, y0_g, h0_g = relative_to_global_state((x0, y0, h0), vehicle)
            xf_g, yf_g, hf_g = relative_to_global_state((xf, yf, hf), vehicle)
        else:
            x0_g, y0_g, h0_g = x0, y0, h0
            xf_g, yf_g, hf_g = xf, yf, hf

        self.x0 = x0_g
        self.y0 = y0_g
        self.h0 = float(h0_g) % (2 * np.pi)
        self.xf = xf_g
        self.yf = yf_g
        self.hf = float(hf_g) % (2 * np.pi)

    def normalized(self) -> "VSDProblem":
        """Return a new problem normalized so initial state is at origin with h=0."""
        vehicle_state = (self.x0, self.y0, self.h0)
        xf_n, yf_n, hf_n = global_to_relative(
            (self.xf, self.yf, self.hf), vehicle_state
        )
        return VSDProblem(
            0.0, 0.0, 0.0,
            xf_n, yf_n, hf_n,
            self.v_max, self.v_min, self.omega_max,
        )

    def __repr__(self) -> str:
        return (
            f"VSDProblem(x0={self.x0}, y0={self.y0}, h0={self.h0:.4f}, "
            f"xf={self.xf}, yf={self.yf}, hf={self.hf:.4f}, "
            f"R={self.R}, r={self.r})"
        )


def relative_to_global_state(state, vehicle):
    """Simple helper — same as coordinates.relative_to_global but local."""
    from vsd.coordinates import relative_to_global
    return relative_to_global(state, vehicle)
