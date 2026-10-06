"""Experiment 1: Baseline replication of constrained retirement decumulation.

Reproduces Figures 6, 7, 8 and Table 1 from the paper.
All parameters from the paper's stated baseline calibration.
"""
import sys
sys.path.insert(0, '/workspace/project')

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from src.model_params import ModelParams, ScenarioParams, get_baseline_scenarios
from src.benchmark import UnconstainedBenchmark
from src.solver_case1 import Case1Solver, U2_value
from src.solver_case2 import Case2Solver
from src.valuation import recover_value_function, solve_feynman_kac, compute_utility_losses
from src.simulation import simulate_paths
from src.reporting import (
    plot_policy_functions, plot_value_functions,
    plot_percentile_paths, format_table1, RESULTS_DIR
)


def run_exp1() -> dict:
    """Run the full baseline replication experiment."""
    print("=" * 70)
    print("EXPERIMENT 1: Baseline Replication")
    print("=" * 70)

    # --- Model setup ---
    params = ModelParams()
    print(f"Model: r={params.r:.4f}, theta={params.theta:.4f}, delta={params.delta:.6f}")
    print(f"Admissible: {params.check_admissibility()}")
    p1, p2 = params.compute_roots()
    print(f"Roots: p1={p1:.4f}, p2={p2:.4f}")

    scenarios = get_baseline_scenarios(params)
    bench = UnconstainedBenchmark(params)
    print(f"Benchmark: b_bar={bench.b_bar:.4f}, pi*={bench.pi_star:.4f}, b_opt={bench.b_bar_opt:.4f}")

    # --- Solve each scenario ---
    solvers = []
    for i, sc in enumerate(scenarios):
        label = f'S{i+1} (b0={sc.b0}, alpha={sc.alpha})'
        print(f"\nSolving Scenario {i+1}: {label} (Case {'I' if sc.is_case1 else 'II'})")
        print(f"  x_under = {sc.x_under:.6f}")
        if sc.is_case1:
            solver = Case1Solver(params, sc, h_max=300.0, n_h=3000, x_grid_size=500)
        else:
            solver = Case2Solver(params, sc, h_min=1e-5, h_max=200.0, n_h=2000,
                                 omega=0.4, max_iter=600, tol=1e-7, x_grid_size=500)
        solver.solve()
        solvers.append(solver)
        print(f"  Policy range: pi* in [{solver.pi_star.min():.3f}, {solver.pi_star.max():.3f}]")
        print(f"  b_star range: [{solver.b_star.min():.4f}, {solver.b_star.max():.4f}]")

    # --- Figure 6: Policy functions ---
    print("\nGenerating Figure 6...")
    scenario_dicts = []
    for i, (sc, sv) in enumerate(zip(scenarios, solvers)):
        scenario_dicts.append({
            'x_grid': sv.x_grid,
            'pi_star': sv.pi_star,
            'payout_ratio': sv.payout_ratio,
            'b_star': sv.b_star,
            'x_under': sc.x_under,
            'b0': sc.b0,
            'alpha': sc.alpha,
            'label': f'S{i+1}: b0={sc.b0}, α={sc.alpha}',
        })
    plot_policy_functions(scenario_dicts, bench, x_range=(0.75, 3.0))

    # --- Recover value functions ---
    print("\nRecovering value functions...")
    for i, (sc, sv) in enumerate(zip(scenarios, solvers)):
        marginal_v = 1.0 / sv.h_of_x
        V = recover_value_function(sv.x_grid, marginal_v, params, sc)
        # Feynman-Kac decomposition
        V1 = solve_feynman_kac(sv.x_grid, sv.pi_star, sv.b_star, params, sc, 'retiree')
        V2 = solve_feynman_kac(sv.x_grid, sv.pi_star, sv.b_star, params, sc, 'manager')
        scenario_dicts[i]['V'] = V
        scenario_dicts[i]['V1'] = V1
        scenario_dicts[i]['V2'] = V2
        print(f"  S{i+1}: V(1)={np.interp(1.0, sv.x_grid, V):.4f}, "
              f"V_un(1)={bench.value(np.array([1.0]))[0]:.4f}")

    # --- Figure 7: Value functions and losses ---
    print("\nGenerating Figure 7...")
    plot_value_functions(scenario_dicts, bench, x_range=(0.75, 3.0))

    # --- Table 1: Utility loss decomposition ---
    print("\nComputing Table 1 utility losses at x=1...")
    losses = []
    for i, (sc, sv) in enumerate(zip(scenarios, solvers)):
        loss = compute_utility_losses(
            x_eval=1.0,
            x_grid=sv.x_grid,
            V_constrained=scenario_dicts[i]['V'],
            V1_constrained=scenario_dicts[i]['V1'],
            V2_constrained=scenario_dicts[i]['V2'],
            bench=bench,
            params=params,
        )
        losses.append(loss)
        print(f"  S{i+1}: total={loss['total']*100:+.2f}%, "
              f"retiree={loss['retiree']*100:+.2f}%, "
              f"manager={loss['manager']*100:+.2f}%")
    table_str = format_table1(scenario_dicts, losses)
    print(table_str)

    # --- Figure 8: Percentile paths ---
    print("\nRunning Monte Carlo simulations for Figure 8 (10,000 paths, 30 years)...")
    sim_results = []
    for i, (sc, sv) in enumerate(zip(scenarios, solvers)):
        print(f"  Simulating S{i+1}...")
        res = simulate_paths(
            x_grid=sv.x_grid,
            pi_star=sv.pi_star,
            b_star=sv.b_star,
            params=params,
            x0=1.0, T=30.0, dt=1.0/52, n_paths=10000,
            seed=42 + i, x_floor=sc.x_under,
        )
        sim_results.append(res)
        print(f"    X_pct5(30yr)={res['X_pct5'][-1]:.3f}, B_pct5(30yr)={res['B_pct5'][-1]:.4f}")
        # Validate floors
        assert np.all(res['X_pct5'] >= sc.x_under - 0.01), f"S{i+1}: Wealth floor violated!"
        print(f"    Floor validation passed (X >= {sc.x_under:.3f})")

    plot_percentile_paths(
        sim_results,
        labels=[f'S{i+1}: b0={s.b0}, α={s.alpha}' for i, s in enumerate(scenarios)],
        x_floor_vals=[s.x_under for s in scenarios],
        b_floor_vals=[s.b0 for s in scenarios],
    )

    # --- Save metrics ---
    metrics = {
        'benchmark': {
            'b_bar': float(bench.b_bar),
            'pi_star': float(bench.pi_star),
            'b_bar_opt': float(bench.b_bar_opt),
            'V_un_at_x1': float(bench.value(np.array([1.0]))[0]),
        },
        'roots': {'p1': float(p1), 'p2': float(p2)},
        'scenarios': [],
    }
    for i, (sc, sv, loss) in enumerate(zip(scenarios, solvers, losses)):
        metrics['scenarios'].append({
            'id': f'S{i+1}',
            'b0': sc.b0, 'alpha': sc.alpha,
            'x_under': sc.x_under,
            'is_case1': sc.is_case1,
            'pi_star_at_xunder': float(sv.pi_star[0]),
            'b_star_at_x1': float(np.interp(1.0, sv.x_grid, sv.b_star)),
            'pi_star_at_x1': float(np.interp(1.0, sv.x_grid, sv.pi_star)),
            'V_at_x1': float(np.interp(1.0, sv.x_grid, scenario_dicts[i]['V'])),
            'loss_total': float(loss['total']),
            'loss_retiree': float(loss['retiree']),
            'loss_manager': float(loss['manager']),
            'target_loss_total': [1.12, 0.66, 0.89][i],
            'target_loss_retiree': [2.22, 1.24, 1.32][i],
            'target_loss_manager': [-0.81, -0.38, 0.10][i],
        })

    metrics_path = RESULTS_DIR / 'exp1_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    print("\nEXPERIMENT 1 COMPLETE")
    return metrics


if __name__ == '__main__':
    run_exp1()
