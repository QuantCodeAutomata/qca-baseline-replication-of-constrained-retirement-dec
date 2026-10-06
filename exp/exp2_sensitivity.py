"""Experiment 2: Sensitivity analysis of maximum utility losses.

Reproduces Table 2 by sweeping alpha, b0, gamma, theta one at a time
and jointly, measuring maximum utility losses at x=1.
# Custom -- no library covers this parameter-sweep HJB problem.
"""
import sys
sys.path.insert(0, '/workspace/project')

import json
import itertools
import numpy as np
from pathlib import Path
from src.model_params import ModelParams, ScenarioParams
from src.benchmark import UnconstainedBenchmark
from src.solver_case1 import Case1Solver
from src.solver_case2 import Case2Solver
from src.valuation import recover_value_function, solve_feynman_kac, compute_utility_losses
from src.reporting import RESULTS_DIR


def solve_one_scenario(
    params: ModelParams, sc: ScenarioParams
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve a single scenario; returns (x_grid, pi_star, b_star, h_of_x)."""
    try:
        if sc.is_case1:
            solver = Case1Solver(params, sc, h_max=200.0, n_h=1500, x_grid_size=300)
        else:
            solver = Case2Solver(params, sc, h_min=1e-5, h_max=150.0, n_h=1000,
                                 omega=0.4, max_iter=400, tol=1e-6, x_grid_size=300)
        solver.solve()
        return solver.x_grid, solver.pi_star, solver.b_star, solver.h_of_x
    except Exception as e:
        print(f"  [WARN] Solver failed: {e}")
        return None, None, None, None


def evaluate_losses(
    params: ModelParams, b0: float, alpha: float
) -> dict[str, float]:
    """Solve and evaluate utility losses for a single (params, b0, alpha) triple."""
    sc = ScenarioParams(b0=b0, alpha=alpha, model=params)
    x_grid, pi_star, b_star, h_of_x = solve_one_scenario(params, sc)
    if x_grid is None:
        return {'total': 0.0, 'retiree': 0.0, 'manager': 0.0}

    bench = UnconstainedBenchmark(params)
    marginal_v = 1.0 / h_of_x
    V = recover_value_function(x_grid, marginal_v, params, sc)
    V1 = solve_feynman_kac(x_grid, pi_star, b_star, params, sc, 'retiree')
    V2 = solve_feynman_kac(x_grid, pi_star, b_star, params, sc, 'manager')
    loss = compute_utility_losses(1.0, x_grid, V, V1, V2, bench, params)
    return loss


def sweep_parameter(
    param_name: str,
    values: np.ndarray,
    baseline_params: ModelParams,
    baseline_scenarios: list[tuple[float, float]],
) -> dict[str, float]:
    """One-at-a-time sweep over a parameter, returning maximum losses."""
    max_total = max_retiree = max_manager = 0.0
    records = []

    for val in values:
        for (b0, alpha) in baseline_scenarios:
            # Build params with varied parameter
            p = ModelParams(
                r1=baseline_params.r1,
                kappa1=baseline_params.kappa1,
                kappa2=baseline_params.kappa2,
                mu=baseline_params.mu,
                sigma=baseline_params.sigma,
                lambda_=baseline_params.lambda_,
                rho=baseline_params.rho,
                gamma1=baseline_params.gamma1,
                gamma2=baseline_params.gamma2,
            )
            b0_use = b0
            alpha_use = alpha

            if param_name == 'alpha':
                alpha_use = val
            elif param_name == 'b0':
                b0_use = val
            elif param_name == 'gamma':
                p.gamma1 = val
                p.gamma2 = val
                p.__post_init__()
            elif param_name == 'theta':
                p.mu = p.r1 + p.sigma * val
                p.__post_init__()

            if not p.check_admissibility():
                continue

            loss = evaluate_losses(p, b0_use, alpha_use)
            max_total = max(max_total, abs(loss.get('total', 0.0)))
            max_retiree = max(max_retiree, abs(loss.get('retiree', 0.0)))
            max_manager = max(max_manager, abs(loss.get('manager', 0.0)))
            records.append({
                'param': param_name, 'val': float(val),
                'b0': b0_use, 'alpha': alpha_use,
                **{k: float(v) for k, v in loss.items()}
            })
            print(f"  {param_name}={val:.3f}, b0={b0_use}, alpha={alpha_use}: "
                  f"total={loss.get('total',0)*100:.2f}%")

    return {
        'max_total': max_total,
        'max_retiree': max_retiree,
        'max_manager': max_manager,
        'records': records,
    }


def run_exp2() -> dict:
    """Run sensitivity analysis sweeps."""
    print("=" * 70)
    print("EXPERIMENT 2: Sensitivity Analysis (Table 2)")
    print("=" * 70)

    baseline = ModelParams()
    scen_pairs = [(0.035, 0.95), (0.035, 0.80), (0.040, 0.80)]

    results = {}

    # --- alpha sweep ---
    print("\nalpha sweep...")
    alpha_vals = np.linspace(0.70, 0.98, 15)
    results['alpha'] = sweep_parameter('alpha', alpha_vals, baseline, scen_pairs)
    print(f"  Max losses: total={results['alpha']['max_total']*100:.2f}%, "
          f"retiree={results['alpha']['max_retiree']*100:.2f}%, "
          f"manager={results['alpha']['max_manager']*100:.2f}%")

    # --- b0 sweep ---
    print("\nb0 sweep...")
    b0_vals = np.linspace(0.025, 0.040, 10)
    results['b0'] = sweep_parameter('b0', b0_vals, baseline, scen_pairs)
    print(f"  Max losses: total={results['b0']['max_total']*100:.2f}%, "
          f"retiree={results['b0']['max_retiree']*100:.2f}%, "
          f"manager={results['b0']['max_manager']*100:.2f}%")

    # --- gamma sweep ---
    print("\ngamma sweep...")
    gamma_vals = np.linspace(0.10, 0.30, 10)
    results['gamma'] = sweep_parameter('gamma', gamma_vals, baseline, scen_pairs)
    print(f"  Max losses: total={results['gamma']['max_total']*100:.2f}%, "
          f"retiree={results['gamma']['max_retiree']*100:.2f}%, "
          f"manager={results['gamma']['max_manager']*100:.2f}%")

    # --- theta sweep ---
    print("\ntheta sweep...")
    theta_vals = np.linspace(0.085, 0.235, 10)
    results['theta'] = sweep_parameter('theta', theta_vals, baseline, scen_pairs)
    print(f"  Max losses: total={results['theta']['max_total']*100:.2f}%, "
          f"retiree={results['theta']['max_retiree']*100:.2f}%, "
          f"manager={results['theta']['max_manager']*100:.2f}%")

    # --- Joint sweep (coarse grid) ---
    print("\nJoint sweep (coarse grid)...")
    alpha_j = [0.75, 0.85, 0.95]
    b0_j = [0.030, 0.035, 0.040]
    gamma_j = [0.12, 0.20, 0.28]
    theta_j = [0.10, 0.135, 0.20]
    joint_max_total = joint_max_retiree = joint_max_manager = 0.0
    joint_records = []
    for alpha_v, b0_v, gamma_v, theta_v in itertools.product(alpha_j, b0_j, gamma_j, theta_j):
        p = ModelParams()
        p.gamma1 = gamma_v
        p.gamma2 = gamma_v
        p.mu = p.r1 + p.sigma * theta_v
        p.__post_init__()
        if not p.check_admissibility():
            continue
        for (b0, alpha) in [(b0_v, alpha_v)]:
            loss = evaluate_losses(p, b0, alpha)
            joint_max_total = max(joint_max_total, abs(loss.get('total', 0.0)))
            joint_max_retiree = max(joint_max_retiree, abs(loss.get('retiree', 0.0)))
            joint_max_manager = max(joint_max_manager, abs(loss.get('manager', 0.0)))
            joint_records.append({'alpha': alpha, 'b0': b0, 'gamma': gamma_v, 'theta': theta_v,
                                   **{k: float(v) for k, v in loss.items()}})

    results['joint'] = {
        'max_total': joint_max_total,
        'max_retiree': joint_max_retiree,
        'max_manager': joint_max_manager,
        'records': joint_records,
    }
    print(f"  Joint max losses: total={joint_max_total*100:.2f}%, "
          f"retiree={joint_max_retiree*100:.2f}%, "
          f"manager={joint_max_manager*100:.2f}%")

    # --- Format Table 2 ---
    target_rows = [
        ('alpha', 2.51, 0.10, 1.40),
        ('b0', 2.33, 0.10, 1.24),
        ('gamma', 3.87, 0.20, 2.20),
        ('theta', 4.73, 2.56, 3.62),
        ('joint', 9.01, 6.42, 8.21),
    ]
    lines = ["## Table 2: Maximum Utility Losses under Parameter Sweeps\n"]
    lines.append("| Parameter | Retiree Max (%) | Manager Max (%) | Combined Max (%) | "
                 "Target Retiree | Target Manager | Target Combined |")
    lines.append("|-----------|-----------------|-----------------|-----------------|"
                 "----------------|----------------|----------------|")
    for (pname, tr, tm, tc) in target_rows:
        r = results[pname]
        lines.append(f"| {pname:10s} | {r['max_retiree']*100:+.2f} | {r['max_manager']*100:+.2f} | "
                     f"{r['max_total']*100:+.2f} | {tr:+.2f} | {tm:+.2f} | {tc:+.2f} |")
    table2_str = '\n'.join(lines)
    (RESULTS_DIR / 'table2_sensitivity.md').write_text(table2_str)
    print("\n" + table2_str)

    # Save metrics
    metrics = {k: {'max_total': v['max_total'], 'max_retiree': v['max_retiree'],
                   'max_manager': v['max_manager']}
               for k, v in results.items()}
    with open(RESULTS_DIR / 'exp2_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("\nEXPERIMENT 2 COMPLETE")
    return metrics


if __name__ == '__main__':
    run_exp2()
