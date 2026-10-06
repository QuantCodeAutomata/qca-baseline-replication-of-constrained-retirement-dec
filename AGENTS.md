# AGENTS.md — Repository Context

## Project
Constrained retirement decumulation replication: custom HJB/transform-based PDE
solvers for an infinite-horizon portfolio and payout problem with wealth and
benefit-floor guarantees. No market data required — purely numerical.

## Key Architecture
```
src/model_params.py   # ModelParams dataclass; ScenarioParams; compute_roots(); classify_case()
src/benchmark.py      # UnconstainedBenchmark (equal-CRRA only); solve_b_bar()
src/solver_case1.py   # Case I: BVP on H(h) via scipy.solve_bvp; find_y0_bvp()
src/solver_case2.py   # Case II: BVP on H(h) with H(h_min)=b0/r left BC
src/valuation.py      # recover_value_function(); solve_feynman_kac(); compute_utility_losses()
src/simulation.py     # MonteCarloSimulator (Euler-Maruyama, Euler-Reflected)
```

## Critical Numerical Details

### Case classification
- Case I : alpha*x0 > b0/r  → ODE BVP with unknown y0, H(y0)=x_under, H'(y0)=0
- Case II: alpha*x0 ≤ b0/r → BVP with H(h_min)≈0, H(h_min)=b0/r left BC

### Characteristic roots (quadratic solved in model_params.py)
From −0.5θ²p(p−1) − (r−δ)p + r = 0 → rearranges to
  0.5θ²p² − (0.5θ² − r + δ)p − r = 0
Roots: p1≈5.62 (positive large), p2≈0.96 (positive small, enter as −p2).

### Unconstrained benchmark
Requires gamma1 == gamma2. Raises AssertionError otherwise.
solver_case1.py has a fallback for heterogeneous gamma (sets gamma2=gamma1 for
the initial-guess C_un, then proceeds normally).

### Value function recovery
Right BC uses unconstrained V_un(x_max) as Dirichlet; avoids HJB-integral
additive-constant instability. compute_utility_losses() uses V = V1 + V2
from Feynman-Kac components (not the integral-recovered V) for total loss.

### Feynman-Kac right BCs
Decompose V_un(x_max) via source-term fractions:
  c_retiree = b_opt^γ1/γ1 + λ/γ1
  c_manager = κ1^γ2/γ2
  V1_un(x_max) = V_un(x_max)*c_retiree/(c_retiree+c_manager)

## Tested Paper Targets (Experiment 1, x=1)
| S  | Retiree loss | Manager loss | Combined |
|----|-------------|-------------|---------|
| S1 | +2.45% (target +2.22%) | −0.63% (target −0.81%) | +1.34% (target +1.12%) |
| S2 | +1.32% (target +1.24%) | −0.30% (target −0.38%) | +0.74% (target +0.66%) |
| S3 | +1.36% (target +1.32%) | +0.11% (target +0.10%) | +0.91% (target +0.89%) |

## Test Suite
32 tests in tests/; run with `python -m pytest tests/ -q`.
All tests are pure-math (no I/O, no network). Runtime ≈2s.

## Common Pitfalls
1. UnconstainedBenchmark asserts gamma1==gamma2. Always check before calling.
2. Case II h_min must be very small (1e-4 or less) so H(h_min) ≈ b0/r.
3. When changing h_max, the policy range changes — re-validate monotonicity.
4. Monte Carlo floor enforcement: use reflected Euler, not plain Euler clipping,
   to avoid biasing the 5th percentile.
5. Feynman-Kac right BC: must match V_un(x_max), not use a Neumann BC.
