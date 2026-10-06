# Constrained Retirement Decumulation: Baseline Replication

This repository replicates the main numerical results from the paper on constrained retirement decumulation under constant and time-varying mortality.

## Overview

The paper studies an infinite-horizon retirement withdrawal problem where a fund manager controls the risky investment fraction and benefit payout, subject to wealth and benefit guarantee constraints. The model uses a CRRA utility framework with constant or Makeham mortality.

## Repository Structure

```
qca-baseline-replication/
├── data/               # No data files needed (scalar-input calibration)
├── src/                # Core mathematical modules
│   ├── model_params.py      # ModelParams and ScenarioParams dataclasses
│   ├── benchmark.py         # Unconstrained equal-CRRA benchmark
│   ├── solver_case1.py      # Case I ODE-based constrained solver
│   ├── solver_case2.py      # Case II integral-equation fixed-point solver
│   ├── valuation.py         # Feynman-Kac value function decomposition
│   ├── simulation.py        # Euler-Maruyama Monte Carlo paths
│   └── reporting.py         # Plot and table generation
├── exp/                # Experiment scripts
│   ├── exp1_baseline.py     # Figures 6, 7, 8 and Table 1
│   ├── exp2_sensitivity.py  # Table 2 parameter sweeps
│   ├── exp3_heterogeneous.py# Figure 9 heterogeneous preferences
│   └── exp4_makeham.py      # Figure 10 Makeham mortality
├── tests/              # pytest test suite
├── results/            # Output figures, tables, metrics
│   └── RESULTS.md      # Summary of findings
└── README.md
```

## Key Model Parameters (Baseline)

| Parameter | Value | Description |
|-----------|-------|-------------|
| r1 | 0.053 | Gross risk-free rate |
| kappa1 | 0.004 | Fee rate |
| mu | 0.08 | Risky asset drift |
| sigma | 0.2 | Risky asset volatility |
| theta | 0.135 | Market price of risk |
| lambda_ | 1/23.6 | Constant hazard rate |
| rho | 0.04 | Time preference |
| gamma1=gamma2 | 0.2 | CRRA exponents (baseline) |

## Three Guarantee Scenarios

| Scenario | b0 | alpha | x_under | Case |
|----------|-----|-------|---------|------|
| S1 | 0.035 | 0.95 | 0.9500 | I |
| S2 | 0.035 | 0.80 | 0.8000 | I |
| S3 | 0.040 | 0.80 | 0.8163 | II |

## Running Experiments

```bash
# Install dependencies
pip install numpy scipy matplotlib numba pandas

# Run all experiments
python exp/exp1_baseline.py
python exp/exp2_sensitivity.py
python exp/exp3_heterogeneous.py
python exp/exp4_makeham.py

# Run tests
pytest tests/ -v
```

## Results

See `results/RESULTS.md` for a summary of findings and `results/` for figures and tables.

## License

Research replication code. All rights reserved.
