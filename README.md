# Variable Speed Dubins

Minimum-time path planning for a kinematic vehicle with variable speed and symmetric turn-rate limits. Based on:

> Wolek, A., Cliff, E. M., & Woolsey, C. A. (2016). *Time-optimal path planning for a kinematic car with variable speed controls.* Journal of Guidance, Control, and Dynamics, 39(10), 2374–2390. [doi:10.2514/1.G001317](https://doi.org/10.2514/1.G001317)

<p align="center">
<img src="https://raw.githubusercontent.com/robotics-uncc/VariableSpeedDubins/master/VariableSpeedDubins.png" width="320">
</p>

The solver searches 84 candidate path types (76 VSD + 8 classical Dubins) and returns the globally minimum-time feasible path. Each candidate is a **BCB** sequence — bang-corner-bang arcs — solved as a small nonlinear program via CasADi/IPOPT, run in parallel across all CPU cores.

## Installation

Requires Python ≥ 3.10 and a working [CasADi](https://web.casadi.org/) installation with IPOPT.

```bash
pip install -e ".[dev]"
```

## Quick start

```python
from vsd import VSDProblem, VSDSolver

problem = VSDProblem(
    x0=0, y0=0, h0=0,      # initial state (x, y, heading in radians)
    xf=5, yf=3, hf=1.0,    # final state
    v_max=2.0, v_min=0.5, omega_max=1.0,
)

paths = VSDSolver().solve(problem)   # sorted by travel-time cost

best = paths[0]
print(best.cost)                          # scalar travel time
print(best.endpoint())                    # [x, y, heading]
history = best.path_history()             # Nx3 array of [x, y, heading]
```

## Documentation

| Document | Description |
|----------|-------------|
| [docs/usage.md](docs/usage.md) | Usage guide — single solve, multi-waypoint, plotting, performance tips |
| [docs/api.md](docs/api.md) | Full API reference for all public classes and methods |
| [docs/architecture.md](docs/architecture.md) | Solver internals — path classes, NLP formulation, parallelism |

## Running tests

```bash
pytest -m "not slow"   # fast unit tests (~2s)
pytest                 # full suite including NLP solves (~90s)
```

## C++ library

The original C++ implementation is in `src/`. See [CLAUDE.md](CLAUDE.md) for build instructions.

## References

Wolek, A., Cliff, E. M., & Woolsey, C. A. (2016). Time-optimal path planning for a kinematic car with variable speed controls. *Journal of Guidance, Control, and Dynamics*, 39(10), 2374–2390. [doi:10.2514/1.G001317](https://doi.org/10.2514/1.G001317)

## Contact

Artur Wolek — awolek@uncc.edu
