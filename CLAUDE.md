# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

C++ library implementing minimum-time path planning for kinematic vehicles with variable speed and symmetric turn rate limits. Based on: Wolek, Cliff & Woolsey (2016), "Time-optimal path planning for a kinematic car with variable speed controls," *JGCD* 39(10).

The key insight is that optimal paths consist of **Bang-Corner-Bang (BCB)** segments where:
- **B (bang)** = max turn rate + max speed → radius R
- **C (corner)** = max turn rate + min speed → radius r < R
- **S (straight)** = no turn + max speed

## Build Commands

```bash
# Ubuntu 16.04 only — install all dependencies (IPOPT, Armadillo, cddlib, OpenBLAS)
./install_deps.sh

# Build library and all test executables
./build.sh
# Outputs: lib/libVarSpeedDubins.a and bin/ executables

# Clean build artifacts and generated files (*.m, *.py, *.lp, *.out)
./clean.sh
```

CMake requires: IPOPT (Coin-OR), Armadillo, BLAS, LAPACK, cddlib, GMP.

## Running Test Programs

After building, executables are in `bin/`:

```bash
./bin/testSolveSinglePath     # Solve one path type with known params, compare to reference
./bin/testSolveAllPaths       # Iterate all 76 VSD candidate paths
./bin/testEndpointAllPaths    # Test endpoint calculations for all paths
./bin/testPlotPath            # Generate Octave plot for a single path
./bin/testPlotAllPathTypes    # Generate Octave plots for all path types
```

Plotting programs emit `.m` scripts; run with Octave to visualize.

## Architecture

### Core Flow

```
VSDProblem (problem definition)
    → VSDSolver (coordinates all candidates)
        → VSDNLP (IPOPT interface per candidate)
            ← VSDUtils (constraints, cost, gradients, sampling)
        → RobustDubins (classical Dubins as 8 additional candidates)
    → VSDPath (holds solution, computes path history and cost)
```

### Solver Workflow

1. **84 candidates** generated: 76 VSD path types + 8 Dubins (R and r radii)
2. **Initial guess** via `VSDUtils::guessStartPoint()` using barycentric sampling
3. **IPOPT optimization** minimizes travel time with endpoint equality constraints
4. **Suboptimality check** via geometric KKT conditions post-optimization
5. **Global minimum** = lowest-cost feasible solution across all candidates

### Path Parameterization

- **Long format**: (a₁,b₁,g₁, a₂,b₂,g₂, a₃,b₃,g₃, a₄,b₄,g₄, L) — arc lengths for each BCB segment + straight length
- **Short format**: ≤4 independent parameters (path class determines which are free)
- 76 candidate path classes encode which segments are present and their turn orientations

### Key Source Files

| File | Role |
|------|------|
| `src/VSDSolver.h/cpp` | Top-level solver; orchestrates all candidates |
| `src/VSDPath.h/cpp` | Path representation, endpoint computation, cost |
| `src/VSDProblem.h/cpp` | Problem statement (states + radii R, r) |
| `src/VSDNLP.h/cpp` | IPOPT TNLP interface |
| `src/VSDUtils.h` + `src/VSDUtils_*.cpp` | Constraints, cost function, sampling, path generation |
| `src/RobustDubins*.h/cpp` | Classical Dubins solver (LSL, LSR, RSL, RSR, LRL, RLR) |
| `src/MathTools*.h/cpp` | Linear algebra (Armadillo wrappers), geometry, polytopes |
| `programs/` | Standalone test/demo executables |

### VSDUtils Split

The `VSDUtils` utilities are split across files by function:
- `_pathGeneration` — `S()`, `BCB()`, `Spath()`, `BCBpath()`
- `_guessStartPoint` — initial guess sampling
- `_linearConstraints` — parameter bounds
- `_costFunction` — travel time + gradients
- `_suboptimalityConditions` — post-solve KKT checks
- `_pathProperties` — dimensions, curvature signs, candidate list
