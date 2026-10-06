"""Experiment 4: Makeham mortality robustness (Figure 10).

Compares baseline constant-hazard policies with time-dependent
Makeham mortality via a PDE approach.
Since the paper's exact Makeham parameters are not fully specified,
we calibrate to match age-60 mortality with expected lifetime ~23.6 years.
"""
import sys
sys.path.insert(0, '/workspace/project')

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.model_params import ModelParams, ScenarioParams
from src.solver_case1 import Case1Solver, U2_value, U2_prime
from src.reporting import RESULTS_DIR


# Makeham mortality parameters calibrated to age-60 retiree
# Makeham: lambda(t) = A + B * c^(a + t), a = initial age = 60
MAKEHAM_A = 0.0007
MAKEHAM_B = 0.0001
MAKEHAM_C = 1.09
MAKEHAM_AGE = 60.0


def makeham_hazard(t: float | np.ndarray) -> float | np.ndarray:
    """Compute Makeham hazard rate at time t after retirement at age 60."""
    return MAKEHAM_A + MAKEHAM_B * MAKEHAM_C**(MAKEHAM_AGE + t)


def makeham_survival(t: float) -> float:
    """Compute survival probability S(t) = exp(-integral_0^t lambda(s) ds).
    Uses numerical quadrature.
    """
    if t <= 0:
        return 1.0
    ts = np.linspace(0, t, max(200, int(t * 50)))
    hazards = makeham_hazard(ts)
    cumhaz = np.trapz(hazards, ts)
    return float(np.exp(-cumhaz))


def solve_pde_makeham(
    params: ModelParams, sc: ScenarioParams,
    T_max: float = 50.0, n_t: int = 500,
    x_grid: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve time-dependent HJB PDE under Makeham mortality.
    
    Backward-in-time finite-difference scheme with policy iteration.
    At each time slice, the optimal controls are:
      b*(t,x) = max{b0, v_x^{1/(gamma-1)}}
      pi*(t,x) = -theta * v_x / (sigma * x * v_xx)
    
    Returns: (x_grid, pi_at_t0, b_at_t0) -- initial-age policies
    # Custom PDE solver -- Context7 found no library equivalent.
    """
    gamma = params.gamma1
    b0 = sc.b0
    x_under = sc.x_under

    if x_grid is None:
        x_grid = np.linspace(x_under, 5.0, 200)
    n_x = len(x_grid)
    dx = np.diff(x_grid)

    # Time grid (backward: from T_max to 0)
    t_grid = np.linspace(0, T_max, n_t)
    dt = t_grid[1] - t_grid[0]

    # Terminal condition: V(T_max, x) ~ 0 (no continuation value at very old age)
    V = np.zeros(n_x)

    # Backward PDE sweep
    for step in range(n_t - 1, -1, -1):
        t = t_grid[step]
        lam_t = float(makeham_hazard(t))
        rho_t = params.rho + lam_t  # effective discount rate at this age

        # Compute numerical derivatives of V
        Vx = np.zeros(n_x)
        Vxx = np.zeros(n_x)
        for i in range(1, n_x - 1):
            hp = dx[i] if i < len(dx) else dx[-1]
            hm = dx[i-1]
            Vx[i] = (V[i+1] - V[i-1]) / (hp + hm)
            Vxx[i] = 2.0 * (V[i+1]/hp - V[i]*(1/hp+1/hm) + V[i-1]/hm) / (hp + hm)
        # Boundary extrapolation
        Vx[0] = (V[1] - V[0]) / dx[0]
        Vxx[0] = Vxx[1]
        Vx[-1] = (V[-1] - V[-2]) / dx[-1]
        Vxx[-1] = Vxx[-2]

        # Optimal controls
        b_star = np.where(Vx > 0, np.maximum(b0, Vx**(1.0/(gamma-1.0))), b0)
        # pi*: need Vxx < 0 for concavity
        pi_star = np.where(
            Vxx < -1e-12,
            -params.theta * Vx / (params.sigma * x_grid * np.abs(Vxx)),
            params.theta / ((1.0 - gamma) * params.sigma)
        )
        pi_star = np.maximum(pi_star, 0.0)

        # Source term: U1(b*) + lambda_t * x^gamma / gamma + (kappa1*x)^gamma2/gamma2
        U1_star = b_star**gamma / gamma
        U2_x = (params.kappa1 * x_grid)**params.gamma2 / params.gamma2 + lam_t * x_grid**gamma / gamma
        f_x = U1_star + U2_x

        # Controlled drift and diffusion
        a_x = (params.r + params.sigma * params.theta * pi_star) * x_grid - b_star
        c2_x = (params.sigma * pi_star * x_grid)**2

        # Explicit Euler update: V_{t-dt} = V_t + dt * [HJB_rhs]
        # HJB_rhs = f_x + a_x*Vx + 0.5*c2_x*Vxx - rho_t*V
        rhs = f_x + a_x * Vx + 0.5 * c2_x * Vxx - rho_t * V
        V_new = V + dt * rhs
        # Boundary: V(x_under) = small positive value
        V_new[0] = max(V_new[0], 0.0)
        V = V_new

    # Extract initial-age (t=0) policies from the final V
    Vx_final = np.gradient(V, x_grid)
    Vxx_final = np.gradient(Vx_final, x_grid)
    b_final = np.maximum(b0, np.where(Vx_final > 0, Vx_final**(1.0/(gamma-1.0)), b0))
    pi_final = np.where(
        Vxx_final < -1e-12,
        -params.theta * Vx_final / (params.sigma * x_grid * np.abs(Vxx_final)),
        params.theta / ((1.0 - gamma) * params.sigma)
    )
    pi_final = np.maximum(pi_final, 0.0)
    return x_grid, pi_final, b_final


def run_exp4() -> dict:
    """Run Makeham mortality robustness comparison."""
    print("=" * 70)
    print("EXPERIMENT 4: Makeham Mortality Robustness (Figure 10)")
    print("=" * 70)

    params = ModelParams()
    # Verify Makeham calibration: expected lifetime from age 60
    def survival(t):
        return makeham_survival(t)
    t_life = np.linspace(0, 80, 5000)
    surv_vals = np.array([survival(t) for t in t_life])
    expected_life = np.trapz(surv_vals, t_life)
    print(f"Makeham expected lifetime from age 60: {expected_life:.2f} years")
    print(f"Constant hazard expected lifetime: {1.0/params.lambda_:.2f} years")

    # Use scenario S1: (b0=0.035, alpha=0.95)
    sc = ScenarioParams(b0=0.035, alpha=0.95, model=params)

    # Constant hazard (baseline) solution
    print("\nSolving constant-hazard baseline (Case I)...")
    solver_const = Case1Solver(params, sc, h_max=200.0, n_h=2000, x_grid_size=400)
    solver_const.solve()
    x_const = solver_const.x_grid
    pi_const = solver_const.pi_star
    b_const = solver_const.b_star
    br_const = solver_const.payout_ratio

    # Makeham extension
    print("\nSolving Makeham mortality extension (PDE)...")
    x_grid_pde = np.linspace(sc.x_under, 5.0, 200)
    x_mak, pi_mak, b_mak = solve_pde_makeham(
        params, sc, T_max=50.0, n_t=300, x_grid=x_grid_pde
    )
    br_mak = b_mak / x_mak

    # Generate Figure 10
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    x_max = 3.0

    mask_c = x_const <= x_max
    mask_m = x_mak <= x_max

    axes[0].plot(x_const[mask_c], pi_const[mask_c], 'tab:blue', lw=2, label='Constant hazard')
    axes[0].plot(x_mak[mask_m], pi_mak[mask_m], 'tab:orange', ls='--', lw=2, label='Makeham')
    axes[0].set_xlabel('Wealth x', fontsize=12)
    axes[0].set_ylabel('Risky fraction π*(x)', fontsize=12)
    axes[0].set_title('Optimal Risky Allocation: Mortality Comparison', fontsize=13)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x_const[mask_c], br_const[mask_c], 'tab:blue', lw=2, label='Constant hazard')
    axes[1].plot(x_mak[mask_m], br_mak[mask_m], 'tab:orange', ls='--', lw=2, label='Makeham')
    axes[1].set_xlabel('Wealth x', fontsize=12)
    axes[1].set_ylabel('Payout ratio b*(x)/x', fontsize=12)
    axes[1].set_title('Optimal Payout Ratio: Mortality Comparison', fontsize=13)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fname = RESULTS_DIR / 'fig10_makeham_comparison.png'
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {fname}")

    # Save metrics
    metrics = {
        'makeham_params': {'A': MAKEHAM_A, 'B': MAKEHAM_B, 'C': MAKEHAM_C, 'age': MAKEHAM_AGE},
        'expected_lifetime_makeham': float(expected_life),
        'expected_lifetime_constant': float(1.0 / params.lambda_),
        'pi_const_at_x1': float(np.interp(1.0, x_const, pi_const)),
        'pi_mak_at_x1': float(np.interp(1.0, x_mak, pi_mak)),
        'br_const_at_x1': float(np.interp(1.0, x_const, br_const)),
        'br_mak_at_x1': float(np.interp(1.0, x_mak, br_mak)),
    }
    with open(RESULTS_DIR / 'exp4_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("\nEXPERIMENT 4 COMPLETE")
    return metrics


if __name__ == '__main__':
    run_exp4()
