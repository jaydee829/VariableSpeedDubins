# Architecture

This document describes the internal structure of the Python VSD solver for contributors and AI coding agents working in this codebase.

---

## Algorithm overview

The Variable Speed Dubins problem finds the minimum-time path between two states `(x0, y0, h0)` and `(xf, yf, hf)` for a vehicle with:
- Maximum speed `v_max`, minimum speed `v_min`
- Maximum turn rate `omega_max`
- Derived bang radius `R = v_max / omega_max`
- Derived corner radius `r = v_min / omega_max`

Optimal controls consist of three extremals:
- **B (bang)**: max turn rate + max speed → arc radius R
- **C (corner)**: max turn rate + min speed → arc radius r
- **S (straight)**: no turn + max speed

An optimal *turn* is a BCB sequence (bang-corner-bang). The globally optimal path is one of 84 path-type candidates:
- 76 VSD candidates in four structural classes: TST, TT, TTT, TTTT
- 8 classical Dubins candidates at radii R and r (only LRL/RLR at r)

---

## Module map

```
VSDProblem                  problem definition + frame normalisation
    ↓
VSDSolver                   orchestrates 84 candidates in parallel
    ├── VSD candidates       vsd/nlp.py — one IPOPT NLP per candidate
    │       ├── guess_start_point  → _barycentric_sample + _batch_endpoints
    │       └── ca.Opti solve
    └── Dubins candidates    vsd/dubins.py — closed-form geometric solution
    ↓
VSDPath                     result container; lazy path history
```

### Key source files

| File | Responsibility |
|------|---------------|
| `vsd/problem.py` | `VSDProblem` — stores states, derives R/r, normalises frame |
| `vsd/solver.py` | `VSDSolver` — dispatches 76 VSD + 8 Dubins candidates via thread pool |
| `vsd/nlp.py` | CasADi/IPOPT NLP per candidate — sampling, NLP build, solve |
| `vsd/path.py` | `VSDPath` dataclass — endpoint and dense history |
| `vsd/kinematics.py` | Pure-numpy BCB/S displacement math; `expand_params_short` |
| `vsd/candidates.py` | 76 VSD candidate dicts; `determine_curvature_params` |
| `vsd/dubins.py` | Classical Dubins solver (LSL/LSR/RSL/RSR/LRL/RLR) |
| `vsd/coordinates.py` | Frame conversion utilities |
| `vsd/planner.py` | `WaypointPlanner` — multi-leg planning via `ProcessPoolExecutor` |

---

## Solver workflow (`VSDSolver.solve`)

1. **Normalise** the problem so the start is at `(0, 0, 0)` via `problem.normalized()`.
2. **Dispatch VSD candidates** — 76 candidates are solved in parallel via `ThreadPoolExecutor`. Each worker calls `nlp.solve_candidate`.
3. **Solve Dubins candidates** — `dubins.solve_dubins` is called for radius R (6 types) and r (LRL/RLR only, 2 types).
4. **Filter** — only paths that are `feasible=True` and `suboptimal=False` are kept.
5. **Restore frame** — set `x0, y0, h0` on each `VSDPath` so `endpoint()` and `path_history()` return global-frame coordinates.
6. **Sort** by cost ascending.

---

## NLP formulation (`nlp.solve_candidate`)

Each of the 76 VSD candidates is solved as a 4-variable NLP:

```
minimise   grad · p            (linear travel-time cost)
subject to  x_l ≤ p ≤ x_u     (parameter bounds)
            g_l ≤ A p ≤ g_u   (linear inequality constraints)
            endpoint(p) = [xf, yf, hf_adj]   (nonlinear equality)
```

where `p` is the 4-element short-form arc-length parameter vector.

**Steps inside `solve_candidate`:**

1. `linear_constraints(candidate, R, r, a_sub, L_max)` → `x_l, x_u, g_l, g_u, A`
2. `guess_start_point(...)` → vectorised barycentric sample + endpoint ranking
3. `path_endpoint_unwrapped(x0)` → determines heading winding number `k_h` → `hf_adj`
4. Build `ca.Opti` NLP with `ca_path_endpoint` for the symbolic endpoint expression
5. `opti.solve()` via IPOPT
6. Feasibility check: `|endpoint(popt) - target| < 1e-4`
7. `check_suboptimality(popt, ...)` — KKT alpha-suboptimality test

A fresh `ca.Opti` object is built per call so each thread has its own independent IPOPT workspace, which is necessary for thread safety.

---

## Parameter representation

Short form: `p = [p0, p1, p2, p3]` — at most 4 free parameters.

Long form: `[a1, b1, g1, L, a2, b2, g2, a3, b3, g3, a4, b4, g4]` — 13 arc-length values (α, β, γ for each BCB segment plus straight length L).

`expand_params_short(p, path_class, path_type)` converts short → long by filling in the fixed/constrained components (e.g. `b = π` for a full corner, `a = 0` for a half-arc).

The same expansion is done symbolically in `_ca_expand` for the CasADi NLP, and in batch-numpy form in `_batch_endpoints` for the initial-guess selection.

---

## Path classes

| Class | Structure | Free params | Count |
|-------|-----------|-------------|-------|
| TST | BCB – S – BCB | 4 | 32 |
| TT | BCB – BCB | 4 | 20 |
| TTT | BCB – BCB – BCB | 4 | 16 |
| TTTT | BCB – BCB – BCB – BCB | 4 | 8 |
| **Total VSD** | | | **76** |

Within each class, variants encode which BCB segments are present (full BCB vs BC vs B) and the turn orientation (L/R). `candidates.py` encodes all 76 combinations.

---

## Initial guess (`guess_start_point`)

The barycentric sampler draws `n_samples` candidate parameter vectors uniformly in the feasible box `[x_l, x_u]`, filters against the linear constraints `A p ∈ [g_l, g_u]`, then picks the sample whose path endpoint is closest to the target `(xf, yf, hf)`.

Both steps are fully vectorised:
- `_barycentric_sample`: one `rng.uniform(size=(N, 4))` call + matrix multiply + boolean mask — replaces an O(n_samples) Python loop.
- `_batch_endpoints` + `_bcb_disp_v`: vectorised BCB kinematics for all four path classes — replaces an O(n_samples) loop over `path_endpoint`.

---

## Threading design

`VSDSolver` uses `ThreadPoolExecutor` by default. This works because:

1. CasADi/IPOPT is a C++ library that releases the Python GIL during the solve.
2. Each `solve_candidate` call builds its own `ca.Opti` — no shared CasADi state across threads.
3. `threadpoolctl` limits each worker's BLAS threads to 1, preventing CPU over-subscription when `n_workers` IPOPT instances run concurrently.

**Why not `ProcessPoolExecutor`?** On Windows, `multiprocessing` uses `spawn` which re-imports Python and CasADi in every worker process (~1–3 s overhead per worker), making it slower than serial for small-to-medium problems.

**Why not `ca.nlpsol` caching?** Pre-compiling the NLP with `ca.nlpsol` and calling it from multiple threads causes internal serialisation (the IPOPT workspace is shared per function-object). The `ca.Opti` approach is therefore faster for concurrent use.

---

## Suboptimality check (`check_suboptimality`)

After IPOPT converges, the KKT alpha-suboptimality condition is evaluated. If either BCB segment's α parameter exceeds the threshold `alpha_subopt(β, R, r)`, the path is provably sub-optimal and is discarded.

**Important divergence from C++**: The C++ implementation has a variable-reuse bug where `alpha1/as1` are overwritten before the final comparison for TT and TST classes. The Python implementation correctly evaluates both segments independently.

---

## Coordinate conventions

- Headings are in radians, zero = positive-x direction, increasing counter-clockwise.
- `path_endpoint` returns heading wrapped to `[0, 2π)`.
- `path_endpoint_unwrapped` returns the raw accumulated heading (no modulo) — used inside the NLP to avoid discontinuities.
- The NLP constraint `ep[2] == hf_adj` uses an adjusted `hf_adj = hf + 2π·k_h` where `k_h` is the winding number estimated from the initial guess.
