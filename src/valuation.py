"""Value function recovery and Feynman-Kac decomposition.

Steps:
1. Recover V(x) from v'(x) = 1/h(x) by numerical integration
2. Solve linear Feynman-Kac ODEs for V1 (retiree) and V2 (manager)
3. Compute relative utility losses
# Custom -- Context7 confirmed no library equivalent for this specific Feynman-Kac problem.
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import solve_banded
from src.model_params import ModelParams, ScenarioParams
from src.solver_case1 import U2_value


def recover_value_function(x_grid: np.ndarray, marginal_v: np.ndarray,
                           params: ModelParams, sc: ScenarioParams,
                           anchor_idx: int = -1) -> np.ndarray:
    """Recover V(x) by integrating v'(x) = marginal_v from a reference point.

    The additive constant is fixed by matching the asymptotic unconstrained
    value function at the right boundary x_max, where constraints are non-binding
    and V(x) → V_un(x) = b_bar * x^γ / γ.  This avoids the numerically unstable
    HJB-midpoint approach whose pi²-term blows up when v'' ≈ 0.

    Parameters
    ----------
    x_grid : wealth grid
    marginal_v : array of v'(x) values (= 1/h(x))
    params : model parameters
    sc : scenario parameters
    anchor_idx : unused, kept for API compatibility

    Returns
    -------
    V : value function array on x_grid
    """
    # Integrate v'(x) from x_grid[0] upward; result is V up to an additive constant
    V_relative = cumulative_trapezoid(marginal_v, x_grid, initial=0.0)

    # Right-boundary asymptotic match: for large x constraints vanish, V → V_un
    from src.benchmark import UnconstainedBenchmark
    bench = UnconstainedBenchmark(params)
    V_right = float(bench.value(np.array([x_grid[-1]]))[0])
    C = V_right - V_relative[-1]
    return V_relative + C


def solve_feynman_kac(
    x_grid: np.ndarray,
    pi_star: np.ndarray,
    b_star: np.ndarray,
    params: ModelParams,
    sc: ScenarioParams,
    component: str = 'total'
) -> np.ndarray:
    """Solve the linear Feynman-Kac ODE for value function components.
    
    For component='retiree' (V1):
        0.5*c^2*V1'' + a*V1' - delta*V1 + [b*(x)^gamma1/gamma1 + lambda*x^gamma1/gamma1] = 0
    For component='manager' (V2):
        0.5*c^2*V2'' + a*V2' - delta*V2 + (kappa1*x)^gamma2/gamma2 = 0
    For component='total':
        Sum: 0.5*c^2*V'' + a*V' - delta*V + U2(x) + U1(b*(x)) = 0
    
    a(x) = [r + sigma*theta*pi*(x)]*x - b*(x)
    c(x) = sigma*pi*(x)*x
    
    Solved by finite differences with boundary conditions:
    - At x_under: Neumann or Dirichlet from economics
    - At x_max: asymptotic power-law match
    
    Parameters
    ----------
    x_grid : wealth grid
    pi_star : optimal risky allocation
    b_star : optimal benefit
    params : model parameters
    sc : scenario parameters
    component : 'retiree', 'manager', or 'total'
    
    Returns
    -------
    V_comp : value function component on x_grid
    """
    p = params
    n = len(x_grid)
    gamma1, gamma2 = p.gamma1, p.gamma2
    
    # Controlled drift and diffusion
    a_x = (p.r + p.sigma * p.theta * pi_star) * x_grid - b_star
    c_x = p.sigma * pi_star * x_grid
    
    # Source term depends on component
    if component == 'retiree':
        # U1(b*(x)) + lambda*x^gamma1/gamma1
        f_x = b_star**gamma1 / gamma1 + p.lambda_ * x_grid**gamma1 / gamma1
    elif component == 'manager':
        # (kappa1*x)^gamma2 / gamma2
        f_x = (p.kappa1 * x_grid)**gamma2 / gamma2
    else:  # total
        f_x = (b_star**gamma1 / gamma1 + p.lambda_ * x_grid**gamma1 / gamma1
               + (p.kappa1 * x_grid)**gamma2 / gamma2)
    
    # Finite difference discretization on x_grid (non-uniform)
    # ODE: 0.5*c^2*V'' + a*V' - delta*V + f = 0
    # => A*V = -f
    # Use central differences: V'_i ~ (V_{i+1}-V_{i-1})/(x_{i+1}-x_{i-1})
    #                          V''_i ~ 2*(V_{i+1}/(hp*hm+hp^2) - V_i/(hp*hm) + V_{i-1}/(hm^2+hp*hm))
    # where hp = x_{i+1}-x_i, hm = x_i-x_{i-1}
    
    # Build tridiagonal system
    lower = np.zeros(n)
    main = np.zeros(n)
    upper = np.zeros(n)
    rhs = np.zeros(n)
    
    for i in range(1, n-1):
        hp = x_grid[i+1] - x_grid[i]
        hm = x_grid[i] - x_grid[i-1]
        c2 = 0.5 * c_x[i]**2
        a_i = a_x[i]
        
        # Second derivative coefficients
        coeff_pp = c2 * 2.0 / (hp * (hp + hm))
        coeff_mm = c2 * 2.0 / (hm * (hp + hm))
        coeff_0 = -c2 * 2.0 / (hp * hm)
        
        # First derivative coefficients
        coeff_p1 = a_i / (hp + hm)
        coeff_m1 = -a_i / (hp + hm)
        
        lower[i] = coeff_mm + coeff_m1
        main[i] = coeff_0 - p.delta
        upper[i] = coeff_pp + coeff_p1
        rhs[i] = -f_x[i]
    
    # Boundary conditions
    # Left: x = x_under (reflecting-like; use Neumann V'=0 or small slope)
    # Use forward difference for derivative at left boundary
    hp0 = x_grid[1] - x_grid[0]
    c2_0 = 0.5 * c_x[0]**2
    a_0 = a_x[0]
    main[0] = -p.delta + c2_0 * (-2.0 / hp0**2) + a_0 * (-1.0 / hp0)
    upper[0] = c2_0 * 2.0 / hp0**2 + a_0 / hp0
    lower[0] = 0.0
    rhs[0] = -f_x[0]
    
    # Right: asymptotic Dirichlet at x_max using unconstrained decomposition.
    # For large x, constraints vanish → V_j(x) → V_un_j(x).
    # Since all source terms are proportional to x^γ1, the unconstrained
    # retiree and manager value functions are fixed fractions of V_un(x).
    from src.benchmark import UnconstainedBenchmark
    bench = UnconstainedBenchmark(params)
    x_max = x_grid[-1]
    V_un_xmax = float(bench.value(np.array([x_max]))[0])
    # Decomposition fractions (source terms both ∝ x^γ1, ratio is constant)
    b_opt = bench.b_bar_opt   # b_un*(x) = b_opt * x
    c_retiree = b_opt**gamma1 / gamma1 + p.lambda_ / gamma1
    c_manager = p.kappa1**gamma2 / gamma2
    c_total = c_retiree + c_manager
    if component == 'retiree':
        V_right = V_un_xmax * c_retiree / c_total
    elif component == 'manager':
        V_right = V_un_xmax * c_manager / c_total
    else:
        V_right = V_un_xmax
    main[-1] = 1.0
    lower[-1] = 0.0
    upper[-1] = 0.0
    rhs[-1] = V_right
    
    # Solve banded system
    ab = np.zeros((3, n))
    ab[0, 1:] = upper[:-1]   # superdiagonal
    ab[1, :] = main           # diagonal
    ab[2, :-1] = lower[1:]   # subdiagonal
    
    try:
        V_comp = solve_banded((1, 1), ab, rhs)
    except Exception as e:
        print(f"  Warning: Feynman-Kac solve failed ({e}), using approximation")
        V_comp = bench.value(x_grid)
    
    return V_comp


def compute_utility_losses(
    x_eval: float,
    x_grid: np.ndarray,
    V_constrained: np.ndarray,
    V1_constrained: np.ndarray,
    V2_constrained: np.ndarray,
    bench,
    params: ModelParams,
) -> dict[str, float]:
    """Compute relative utility losses at x_eval.

    The total constrained V is taken as V1 + V2 from the Feynman-Kac ODEs,
    which avoids the additive-constant sensitivity of the HJB-integral approach.
    The unconstrained benchmark is decomposed proportionally to source-term
    weights, which are all ∝ x^γ1 (exact when gamma1 = gamma2 = γ).

    Returns dict with 'total', 'retiree', 'manager' relative losses.
    """
    # Interpolate constrained component values at x_eval
    V1_c = float(np.interp(x_eval, x_grid, V1_constrained))
    V2_c = float(np.interp(x_eval, x_grid, V2_constrained))
    V_c = V1_c + V2_c  # total constrained value from FK components

    # Cross-check against integral-recovered V (informational only)
    V_c_integral = float(np.interp(x_eval, x_grid, V_constrained))

    # Unconstrained benchmark at x_eval
    V_un = float(bench.value(np.array([x_eval]))[0])

    # Unconstrained decomposition fractions: source terms ∝ x^γ1, so
    # V_un_j / V_un = c_j / c_total (exact for equal-CRRA; approximation otherwise)
    b_opt = bench.b_bar_opt
    gamma1, gamma2 = params.gamma1, params.gamma2
    c_retiree = b_opt**gamma1 / gamma1 + params.lambda_ / gamma1
    c_manager = params.kappa1**gamma2 / gamma2
    c_total = c_retiree + c_manager
    V1_un = V_un * c_retiree / c_total
    V2_un = V_un * c_manager / c_total

    def safe_loss(v_un: float, v_c: float) -> float:
        if abs(v_un) < 1e-15:
            return 0.0
        return (v_un - v_c) / v_un

    return {
        'total': safe_loss(V_un, V_c),
        'retiree': safe_loss(V1_un, V1_c),
        'manager': safe_loss(V2_un, V2_c),
        'V_un': V_un, 'V_c': V_c, 'V_c_integral': V_c_integral,
        'V1_un': V1_un, 'V1_c': V1_c,
        'V2_un': V2_un, 'V2_c': V2_c,
    }
