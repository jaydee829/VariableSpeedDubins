# API Reference

All public classes and functions are exported from the top-level `vsd` package:

```python
from vsd import VSDProblem, VSDSolver, VSDPath, WaypointPlanner, Waypoint, InfeasibleLegError
```

---

## VSDProblem

```python
class VSDProblem(x0, y0, h0, xf, yf, hf, v_max, v_min, omega_max, frame='global')
```

Defines a single point-to-point planning problem.

**Parameters**

| Name | Type | Description |
|------|------|-------------|
| `x0, y0, h0` | `float` | Initial position and heading (radians) |
| `xf, yf, hf` | `float` | Final position and heading |
| `v_max` | `float` | Maximum speed |
| `v_min` | `float` | Minimum (cornering) speed — `0 < v_min < v_max` |
| `omega_max` | `float` | Maximum turn rate (rad/s) — must be positive |
| `frame` | `str` | `'global'` (default) or `'relative'` |

Raises `ValueError` if `v_min >= v_max` or `omega_max <= 0`.

**Attributes**

| Name | Description |
|------|-------------|
| `R` | Bang radius = `v_max / omega_max` |
| `r` | Corner radius = `v_min / omega_max` |
| `x0, y0, h0` | Stored initial state (global frame) |
| `xf, yf, hf` | Stored final state (global frame) |

**Methods**

| Method | Returns | Description |
|--------|---------|-------------|
| `normalized()` | `VSDProblem` | Returns a copy translated and rotated so the start is at the origin with `h=0` |

---

## VSDSolver

```python
class VSDSolver(L_max=10.0, n_samples=250, tol=1e-7, n_workers=None, use_threads=True)
```

Solves a `VSDProblem` by running all 84 candidates in parallel.

**Parameters**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `L_max` | `float` | `10.0` | Maximum straight-segment length |
| `n_samples` | `int` | `250` | Barycentric samples for initial-guess selection |
| `tol` | `float` | `1e-7` | IPOPT convergence tolerance |
| `n_workers` | `int \| None` | `None` | Worker count; `None` → `os.cpu_count()` |
| `use_threads` | `bool` | `True` | `True` → `ThreadPoolExecutor`; `False` → `ProcessPoolExecutor` |

**Methods**

### `solve(problem) → list[VSDPath]`

Run the full 84-candidate search.

- Returns a list of `VSDPath` objects sorted by `cost` ascending.
- Returns `[]` if no feasible path is found.
- Only includes paths that are both feasible and not flagged as suboptimal by the KKT check.

---

## VSDPath

```python
@dataclass
class VSDPath
```

Holds one solved path candidate. Returned by `VSDSolver.solve()`.

**Attributes**

| Name | Type | Description |
|------|------|-------------|
| `cost` | `float` | Travel-time cost |
| `feasible` | `bool` | NLP converged and endpoint residual within tolerance |
| `suboptimal` | `bool` | Alpha-suboptimality condition flagged this solution |
| `candidate` | `dict` | `{'path_class', 'path_type', 'orientation'}` |
| `params` | `ndarray` | 4-element arc-length parameter vector (short form) |
| `R` | `float` | Bang radius |
| `r` | `float` | Corner radius |
| `is_dubins` | `bool` | `True` for classical Dubins solutions |
| `x0, y0, h0` | `float` | Initial state used for global-frame output |

**Methods**

### `endpoint() → ndarray`

Returns `[x, y, heading]` of the path end in global frame. Heading is wrapped to `[0, 2π)`.

### `path_history(nom_spacing=0.05) → ndarray`

Returns an `(N, 3)` array of `[x, y, heading]` samples along the path, with consecutive points separated by approximately `nom_spacing` in arc length.

---

## WaypointPlanner

```python
class WaypointPlanner(v_max, v_min, omega_max, L_max=10.0, n_samples=250, tol=1e-7, n_workers=None)
```

Plans a sequence of legs through a list of `Waypoint` objects.

**Parameters** — same vehicle dynamics as `VSDProblem`; `n_workers` controls per-leg parallelism.

**Methods**

### `plan(waypoints, start=None) → list[VSDPath]`

| Parameter | Type | Description |
|-----------|------|-------------|
| `waypoints` | `list[Waypoint]` | Ordered sequence of states to pass through |
| `start` | `Waypoint \| None` | Departure state; if `None`, the first waypoint is used as start and no path is generated for it |

Returns one `VSDPath` per leg (i.e. `len(waypoints) - 1` legs when `start=None`, or `len(waypoints)` legs when `start` is provided).

Raises `InfeasibleLegError(leg_index)` if any leg has no feasible solution. All legs that complete successfully before the failure are returned in the exception's context.

---

## Waypoint

```python
@dataclass
class Waypoint(x, y, h, frame='global')
```

A state waypoint with required heading.

| Attribute | Type | Description |
|-----------|------|-------------|
| `x, y` | `float` | Position |
| `h` | `float` | Heading in radians |
| `frame` | `str` | `'global'` or `'relative'` |

---

## InfeasibleLegError

```python
class InfeasibleLegError(leg_index, message='')
```

Raised by `WaypointPlanner.plan()` when a leg has no feasible solution.

| Attribute | Description |
|-----------|-------------|
| `leg_index` | Zero-based index of the infeasible leg |

---

## Low-level API

These are internal modules that can be used directly when you need finer control.

### `vsd.nlp.solve_candidate`

```python
solve_candidate(candidate, R, r, xf, yf, hf,
                L_max=10.0, n_samples=250, tol=1e-7, seed=None) → dict
```

Solve a single VSD candidate NLP. Returns a dict with keys `feasible`, `suboptimal`, `params`, `cost`, `candidate`.

### `vsd.candidates.candidate_list`

```python
candidate_list() → list[dict]
```

Returns the 76 VSD candidate dicts, each with `path_class`, `path_type`, `orientation`.

### `vsd.kinematics.path_endpoint`

```python
path_endpoint(params_short, candidate, R, r, x0=0, y0=0, h0=0) → ndarray
```

Compute `[x, y, heading]` endpoint for given short-form parameters and candidate.

### `vsd.kinematics.path_history`

```python
path_history(params_short, candidate, R, r, x0=0, y0=0, h0=0, nom_spacing=0.05) → ndarray
```

Return dense `(N, 3)` path history.

### `vsd.dubins.solve_dubins`

```python
solve_dubins(xf, yf, hf, R) → list[DubinsPath]
```

Solve classical Dubins problem at radius `R`. Returns 6 paths (LSL, LSR, RSL, RSR, LRL, RLR); infeasible ones are flagged with `feasible=False`.
