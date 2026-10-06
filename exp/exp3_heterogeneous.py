"""Experiment 3: Heterogeneous utility exponent analysis (Figure 9).

Solves the constrained problem for (gamma1, gamma2) pairs:
  (0.1, 0.3), (0.1, 0.1), (0.3, 0.3)
under scenario (b0, alpha) = (0.035, 0.95).
"""
import sys
sys.path.insert(0, '/workspace/project')

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.model_params import ModelParams, ScenarioParams
from src.solver_case1 import Case1Solver
from src.reporting import RESULTS_DIR


def run_exp3() -> dict:
    """Run heterogeneous exponent analysis."""
    print("=" * 70)
    print("EXPERIMENT 3: Heterogeneous Utility Exponents (Figure 9)")
    print("=" * 70)

    # Fixed scenario: Case I with (b0, alpha) = (0.035, 0.95)
    b0 = 0.035
    alpha = 0.95
    exponent_pairs = [(0.1, 0.3), (0.1, 0.1), (0.3, 0.3)]
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    styles = ['-', '--', '-.']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].set_xlabel('Wealth x', fontsize=12)
    axes[0].set_ylabel('Risky fraction π*(x)', fontsize=12)
    axes[0].set_title('Optimal Risky Allocation (Heterogeneous Preferences)', fontsize=13)
    axes[1].set_xlabel('Wealth x', fontsize=12)
    axes[1].set_ylabel('Payout ratio b*(x)/x', fontsize=12)
    axes[1].set_title('Optimal Payout Ratio (Heterogeneous Preferences)', fontsize=13)

    results = {'exponent_pairs': [], 'metrics': []}

    for (gamma1, gamma2), color, style in zip(exponent_pairs, colors, styles):
        print(f"\nSolving (gamma1={gamma1}, gamma2={gamma2})...")
        p = ModelParams(gamma1=gamma1, gamma2=gamma2)
        sc = ScenarioParams(b0=b0, alpha=alpha, model=p)
        assert sc.is_case1, f"Expected Case I but got Case II: x_under={sc.x_under:.4f}"
        print(f"  x_under={sc.x_under:.4f}, case={'I' if sc.is_case1 else 'II'}")

        solver = Case1Solver(p, sc, h_max=250.0, n_h=2500, x_grid_size=500)
        solver.solve()

        xg = solver.x_grid
        mask = xg <= 3.0
        label = f'(γ₁={gamma1}, γ₂={gamma2})'
        axes[0].plot(xg[mask], solver.pi_star[mask], color=color, ls=style, lw=2, label=label)
        axes[1].plot(xg[mask], solver.payout_ratio[mask], color=color, ls=style, lw=2, label=label)

        results['exponent_pairs'].append((gamma1, gamma2))
        results['metrics'].append({
            'gamma1': gamma1, 'gamma2': gamma2,
            'pi_at_x_under': float(solver.pi_star[0]),
            'pi_at_x1': float(np.interp(1.0, xg, solver.pi_star)),
            'b_ratio_at_x1': float(np.interp(1.0, xg, solver.payout_ratio)),
        })
        print(f"  pi*(x_under)={solver.pi_star[0]:.3f}, pi*(1)={np.interp(1.0, xg, solver.pi_star):.3f}")

    for ax in axes:
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.92, 3.0)
        ax.axvline(0.95, color='gray', ls=':', lw=1, alpha=0.5, label='x_under=0.95')

    plt.tight_layout()
    fname = RESULTS_DIR / 'fig9_heterogeneous_preferences.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved {fname}")

    with open(RESULTS_DIR / 'exp3_metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("EXPERIMENT 3 COMPLETE")
    return results


if __name__ == '__main__':
    run_exp3()
