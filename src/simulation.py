"""Monte Carlo simulation of controlled wealth paths.

Uses Euler-Maruyama with feedback policies interpolated from the solver grids.
# Custom -- Context7 found no library equivalent for this specific control problem.
"""
from __future__ import annotations
import numpy as np
from src.model_params import ModelParams, ScenarioParams


def simulate_paths(
    x_grid: np.ndarray,
    pi_star: np.ndarray,
    b_star: np.ndarray,
    params: ModelParams,
    x0: float = 1.0,
    T: float = 30.0,
    dt: float = 1.0 / 52,
    n_paths: int = 10000,
    seed: int = 42,
    x_floor: float = None,
) -> dict[str, np.ndarray]:
    """Simulate controlled wealth paths with Euler-Maruyama.
    
    Parameters
    ----------
    x_grid : policy grid for wealth
    pi_star : risky allocation policy on x_grid
    b_star : benefit policy on x_grid
    params : model parameters
    x0 : initial wealth
    T : simulation horizon in years
    dt : time step
    n_paths : number of Monte Carlo paths
    seed : random seed
    x_floor : lower wealth boundary (if None, uses min(x_grid))
    
    Returns
    -------
    dict with 'time', 'X_pct5', 'X_pct20', 'B_pct5', 'B_pct20', 'X_mean', 'B_mean'
    """
    rng = np.random.default_rng(seed)
    n_steps = int(T / dt)
    t_grid = np.linspace(0, T, n_steps + 1)
    
    if x_floor is None:
        x_floor = x_grid[0]
    
    p = params
    sqrt_dt = np.sqrt(dt)
    
    # Storage for percentiles at each time step
    X_pct5 = np.zeros(n_steps + 1)
    X_pct20 = np.zeros(n_steps + 1)
    X_mean = np.zeros(n_steps + 1)
    B_pct5 = np.zeros(n_steps + 1)
    B_pct20 = np.zeros(n_steps + 1)
    B_mean = np.zeros(n_steps + 1)
    
    # Run all paths
    X_curr = np.full(n_paths, x0)
    
    # Record initial
    b_curr = np.interp(X_curr, x_grid, b_star)
    X_pct5[0] = np.percentile(X_curr, 5)
    X_pct20[0] = np.percentile(X_curr, 20)
    X_mean[0] = np.mean(X_curr)
    B_pct5[0] = np.percentile(b_curr, 5)
    B_pct20[0] = np.percentile(b_curr, 20)
    B_mean[0] = np.mean(b_curr)
    
    for step in range(n_steps):
        # Get policies for all paths via interpolation
        pi_curr = np.interp(X_curr, x_grid, pi_star)
        b_curr = np.interp(X_curr, x_grid, b_star)
        
        # Euler-Maruyama step
        Z = rng.standard_normal(n_paths)
        drift = (p.r + p.sigma * p.theta * pi_curr) * X_curr - b_curr
        diffusion = p.sigma * pi_curr * X_curr
        X_next = X_curr + drift * dt + diffusion * sqrt_dt * Z
        
        # Enforce floor
        X_next = np.maximum(X_next, x_floor)
        X_curr = X_next
        
        # Compute benefits at new wealth
        b_curr = np.interp(X_curr, x_grid, b_star)
        
        X_pct5[step+1] = np.percentile(X_curr, 5)
        X_pct20[step+1] = np.percentile(X_curr, 20)
        X_mean[step+1] = np.mean(X_curr)
        B_pct5[step+1] = np.percentile(b_curr, 5)
        B_pct20[step+1] = np.percentile(b_curr, 20)
        B_mean[step+1] = np.mean(b_curr)
    
    return {
        'time': t_grid,
        'X_pct5': X_pct5, 'X_pct20': X_pct20, 'X_mean': X_mean,
        'B_pct5': B_pct5, 'B_pct20': B_pct20, 'B_mean': B_mean,
    }
