# Usage Guide

## Installation

Requires Python ≥ 3.10 and [CasADi](https://web.casadi.org/) with IPOPT.

```bash
pip install -e ".[dev]"
```

Runtime dependencies: `casadi>=3.6`, `numpy>=1.24`, `threadpoolctl>=3.0`.

---

## Single point-to-point solve

```python
from vsd import VSDProblem, VSDSolver

problem = VSDProblem(
    x0=0, y0=0, h0=0,          # initial (x, y, heading in radians)
    xf=5, yf=3, hf=1.0,        # final state
    v_max=2.0,                  # maximum speed
    v_min=0.5,                  # minimum (cornering) speed
    omega_max=1.0,              # maximum turn rate (rad/s)
)

solver = VSDSolver()
paths = solver.solve(problem)   # returns list[VSDPath], sorted by cost

if not paths:
    print("No feasible solution found")
else:
    best = paths[0]
    print(f"Cost:      {best.cost:.4f}")
    print(f"Path type: {best.candidate['path_class']} / {best.candidate['path_type']}")
    print(f"Endpoint:  {best.endpoint()}")
```

`VSDSolver.solve()` returns an empty list when no candidate converges. This is rare for well-formed problems but can happen near degenerate configurations (e.g. zero-length paths, extreme aspect ratios).

---

## Path history and plotting

`path_history()` returns a dense `(N, 3)` array of `[x, y, heading]` samples spaced approximately `nom_spacing` apart in arc length.

```python
import matplotlib.pyplot as plt

history = best.path_history(nom_spacing=0.05)
plt.plot(history[:, 0], history[:, 1])
plt.axis("equal")
plt.xlabel("x"); plt.ylabel("y")
plt.title(f"VSD path — cost {best.cost:.3f}")
plt.show()
```

---

## Multi-waypoint planning

`WaypointPlanner` chains `VSDSolver` calls across a sequence of `Waypoint` objects. Each waypoint requires an explicit heading. All legs are solved in parallel.

```python
import math
from vsd import WaypointPlanner, Waypoint

planner = WaypointPlanner(v_max=2.0, v_min=0.5, omega_max=1.0)

waypoints = [
    Waypoint(x=0,  y=0,  h=0),
    Waypoint(x=5,  y=3,  h=math.pi / 4),
    Waypoint(x=10, y=0,  h=0),
    Waypoint(x=15, y=3,  h=math.pi / 2),
]

legs = planner.plan(waypoints)
total_cost = sum(leg.cost for leg in legs)

for i, leg in enumerate(legs):
    print(f"Leg {i}: {leg.candidate['path_type']}, cost={leg.cost:.4f}")
```

To supply a separate departure state (e.g. the vehicle's current pose) without treating it as a waypoint to visit:

```python
start = Waypoint(x=0.5, y=-0.2, h=0.1)
legs = planner.plan(waypoints, start=start)
```

Raises `InfeasibleLegError(leg_index)` if any leg has no feasible solution.

---

## Relative-frame problems

Both `VSDProblem` and `Waypoint` accept `frame='relative'`. In relative frame the vehicle is placed at the origin with heading 0, and all states are expressed in that body frame.

```python
from vsd import VSDProblem, VSDSolver

# Target is 5 m ahead and 2 m to the left of the vehicle, facing 45°
problem = VSDProblem(
    x0=0, y0=0, h0=0,
    xf=5, yf=2, hf=math.pi / 4,
    v_max=2.0, v_min=0.5, omega_max=1.0,
    frame="relative",
)
```

---

## Controlling parallelism

```python
# All CPU cores (default)
solver = VSDSolver()

# Fixed worker count — good for benchmarking or nested parallelism
solver = VSDSolver(n_workers=4)

# Serial — useful for profiling or when calling from a thread pool yourself
solver = VSDSolver(n_workers=1)
```

The solver uses `ThreadPoolExecutor` by default. CasADi/IPOPT releases the GIL during solving so threads achieve real parallelism. `threadpoolctl` automatically limits each worker's BLAS threads to 1 to prevent CPU over-subscription when many IPOPT instances run concurrently.

To use process-based parallelism (e.g. on Linux where `fork` is available):

```python
solver = VSDSolver(use_threads=False)
```

---

## Solver tuning

```python
solver = VSDSolver(
    L_max=20.0,      # increase for longer straight segments (default 10.0)
    n_samples=100,   # fewer initial-guess samples → faster but may miss solutions
    tol=1e-6,        # looser tolerance → faster convergence
)
```

`n_samples` controls how many parameter-space samples are evaluated to pick the best initial guess for each IPOPT call. The default of 250 is conservative; 50–100 usually works well for typical geometries.

---

## Inspecting all candidate solutions

`solver.solve()` filters out infeasible and suboptimal candidates. To inspect the raw output of a single candidate directly:

```python
from vsd.nlp import solve_candidate
from vsd.candidates import candidate_list

cands = candidate_list()
tst_lsl = next(c for c in cands
               if c["path_class"] == "TST" and c["orientation"] == "LSL")

result = solve_candidate(tst_lsl, R=2.0, r=0.5, xf=5, yf=3, hf=1.0)
print(result["feasible"], result["cost"], result["params"])
```
