"""Reporting utilities: plots and table generation."""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path('/workspace/project/results')
RESULTS_DIR.mkdir(exist_ok=True)


def plot_policy_functions(
    scenarios: list[dict],
    bench,
    x_range: tuple[float, float] = (None, 3.0),
    fname: str = 'fig6_policy_functions.png',
) -> None:
    """Reproduce Figure 6: policy functions across scenarios and benchmark."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    styles = ['-', '--', '-.']
    
    for ax in axes:
        ax.set_xlabel('Wealth x', fontsize=12)
    axes[0].set_ylabel('Risky fraction π*(x)', fontsize=12)
    axes[1].set_ylabel('Payout ratio b*(x)/x', fontsize=12)
    axes[0].set_title('Optimal Risky Allocation', fontsize=13)
    axes[1].set_title('Optimal Payout Ratio', fontsize=13)
    
    x_max = x_range[1]
    # Benchmark
    x_bench = np.linspace(0.5, x_max, 500)
    axes[0].axhline(bench.pi_star, color='black', ls=':', lw=2, label='Unconstrained')
    axes[1].axhline(bench.b_bar_opt, color='black', ls=':', lw=2, label='Unconstrained')
    
    for i, scen in enumerate(scenarios):
        xg = scen['x_grid']
        mask = (xg <= x_max) & (xg >= (x_range[0] or xg[0]))
        label = scen.get('label', f'Scenario {i+1}')
        axes[0].plot(xg[mask], scen['pi_star'][mask], color=colors[i], ls=styles[i],
                     lw=2, label=label)
        axes[1].plot(xg[mask], scen['payout_ratio'][mask], color=colors[i], ls=styles[i],
                     lw=2, label=label)
        # Mark lower boundary
        axes[0].axvline(scen['x_under'], color=colors[i], ls=':', alpha=0.4, lw=1)
        axes[1].axvline(scen['x_under'], color=colors[i], ls=':', alpha=0.4, lw=1)
    
    for ax in axes:
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x_range[0] or 0.7, x_max)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {fname}")


def plot_value_functions(
    scenarios: list[dict],
    bench,
    x_range: tuple[float, float] = (0.7, 3.0),
    fname: str = 'fig7_value_functions.png',
) -> None:
    """Reproduce Figure 7: value functions and relative utility losses."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = ['tab:blue', 'tab:orange', 'tab:green']
    styles = ['-', '--', '-.']
    x_lo, x_hi = x_range
    
    x_bench = np.linspace(x_lo, x_hi, 500)
    V_un = bench.value(x_bench)
    axes[0].plot(x_bench, V_un, 'k:', lw=2, label='Unconstrained')
    axes[0].set_ylabel('Value function V(x)', fontsize=12)
    axes[0].set_xlabel('Wealth x', fontsize=12)
    axes[0].set_title('Value Functions', fontsize=13)
    axes[1].set_ylabel('Relative loss (V_un - V) / V_un', fontsize=12)
    axes[1].set_xlabel('Wealth x', fontsize=12)
    axes[1].set_title('Relative Utility Loss', fontsize=13)
    
    for i, scen in enumerate(scenarios):
        xg = scen['x_grid']
        Vc = scen['V']
        mask = (xg >= x_lo) & (xg <= x_hi)
        label = scen.get('label', f'Scenario {i+1}')
        axes[0].plot(xg[mask], Vc[mask], color=colors[i], ls=styles[i], lw=2, label=label)
        V_un_i = bench.value(xg[mask])
        loss = (V_un_i - Vc[mask]) / np.abs(V_un_i)
        axes[1].plot(xg[mask], loss * 100, color=colors[i], ls=styles[i], lw=2, label=label)
    
    for ax in axes:
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x_lo, x_hi)
    axes[1].set_ylabel('Relative loss (%)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {fname}")


def plot_percentile_paths(
    sim_results: list[dict],
    labels: list[str],
    x_floor_vals: list[float],
    b_floor_vals: list[float],
    fname: str = 'fig8_percentile_paths.png',
) -> None:
    """Reproduce Figure 8: 5th and 20th percentile wealth and benefit paths."""
    n = len(sim_results)
    fig, axes = plt.subplots(2, n, figsize=(5*n, 8), sharex=True)
    if n == 1:
        axes = axes.reshape(2, 1)
    colors_pct = {'5th': 'tab:red', '20th': 'tab:orange'}
    
    for col, (res, lbl, xf, bf) in enumerate(zip(sim_results, labels, x_floor_vals, b_floor_vals)):
        t = res['time']
        # Wealth panel
        ax_w = axes[0, col]
        ax_w.plot(t, res['X_pct5'], color='tab:red', lw=2, label='5th pct')
        ax_w.plot(t, res['X_pct20'], color='tab:orange', lw=2, label='20th pct')
        ax_w.axhline(xf, color='gray', ls='--', lw=1.5, label=f'Floor {xf:.2f}')
        ax_w.set_title(f'{lbl}\nWealth Percentiles', fontsize=11)
        ax_w.set_ylabel('Wealth X(t)', fontsize=10)
        ax_w.legend(fontsize=9)
        ax_w.grid(True, alpha=0.3)
        
        # Benefit panel
        ax_b = axes[1, col]
        ax_b.plot(t, res['B_pct5'], color='tab:red', lw=2, label='5th pct')
        ax_b.plot(t, res['B_pct20'], color='tab:orange', lw=2, label='20th pct')
        ax_b.axhline(bf, color='gray', ls='--', lw=1.5, label=f'Floor b0={bf}')
        ax_b.set_title('Benefit Percentiles', fontsize=11)
        ax_b.set_ylabel('Benefit b(t)', fontsize=10)
        ax_b.set_xlabel('Time (years)', fontsize=10)
        ax_b.legend(fontsize=9)
        ax_b.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved {fname}")


def format_table1(
    scenarios: list[dict],
    losses: list[dict],
    fname: str = 'table1_utility_losses.md',
) -> str:
    """Format Table 1: utility loss decomposition."""
    lines = []
    lines.append("## Table 1: Utility Loss Decomposition at x=1\n")
    lines.append("| Scenario | b0 | alpha | Retiree Loss (%) | Manager Loss (%) | Combined Loss (%) |")
    lines.append("|----------|----|-------|-----------------|-----------------|-------------------|")
    target_retiree = [2.22, 1.24, 1.32]
    target_manager = [-0.81, -0.38, 0.10]
    target_combined = [1.12, 0.66, 0.89]
    for i, (scen, loss) in enumerate(zip(scenarios, losses)):
        b0 = scen.get('b0', '?')
        alpha = scen.get('alpha', '?')
        r_loss = loss.get('retiree', 0.0) * 100
        m_loss = loss.get('manager', 0.0) * 100
        c_loss = loss.get('total', 0.0) * 100
        lines.append(f"| S{i+1} | {b0} | {alpha} | {r_loss:+.2f} (target {target_retiree[i]:+.2f}) | "
                     f"{m_loss:+.2f} (target {target_manager[i]:+.2f}) | "
                     f"{c_loss:+.2f} (target {target_combined[i]:+.2f}) |")
    table_str = '\n'.join(lines)
    (RESULTS_DIR / fname).write_text(table_str)
    print(f"  Saved {fname}")
    return table_str
