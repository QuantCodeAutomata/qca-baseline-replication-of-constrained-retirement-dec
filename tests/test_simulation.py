"""Tests for Monte Carlo simulation."""
import sys
sys.path.insert(0, '/workspace/project')
import pytest
import numpy as np
from src.model_params import ModelParams, ScenarioParams
from src.simulation import simulate_paths


def make_constant_policy(x_grid, pi_val, b_val):
    """Create constant policy arrays for testing."""
    pi_star = np.full_like(x_grid, pi_val)
    b_star = np.full_like(x_grid, b_val)
    return pi_star, b_star


def test_simulation_returns_correct_shapes():
    """Simulation output arrays should have correct shapes."""
    p = ModelParams()
    x_grid = np.linspace(0.8, 5.0, 100)
    pi_star, b_star = make_constant_policy(x_grid, 0.5, 0.04)
    T, dt = 1.0, 1.0/12
    result = simulate_paths(x_grid, pi_star, b_star, p, x0=1.0, T=T, dt=dt,
                            n_paths=100, seed=0)
    n_steps = int(T/dt)
    assert result['time'].shape == (n_steps+1,)
    assert result['X_pct5'].shape == (n_steps+1,)
    assert result['B_pct20'].shape == (n_steps+1,)


def test_simulation_initial_wealth():
    """Initial wealth percentiles should equal x0."""
    p = ModelParams()
    x_grid = np.linspace(0.8, 5.0, 100)
    pi_star, b_star = make_constant_policy(x_grid, 0.5, 0.04)
    result = simulate_paths(x_grid, pi_star, b_star, p, x0=1.0, T=1.0, dt=1.0/12,
                            n_paths=200, seed=1)
    assert abs(result['X_pct5'][0] - 1.0) < 1e-10
    assert abs(result['X_pct20'][0] - 1.0) < 1e-10
    assert abs(result['X_mean'][0] - 1.0) < 1e-10


def test_simulation_floor_enforced():
    """Wealth floor should be enforced during simulation."""
    p = ModelParams()
    x_grid = np.linspace(0.5, 5.0, 100)
    # Very high benefit to force wealth down
    b_star = np.full_like(x_grid, 0.2)
    pi_star = np.zeros_like(x_grid)
    floor = 0.5
    result = simulate_paths(x_grid, pi_star, b_star, p, x0=1.0, T=5.0, dt=1.0/52,
                            n_paths=200, seed=2, x_floor=floor)
    assert np.all(result['X_pct5'] >= floor - 1e-10)


def test_simulation_deterministic_seed():
    """Same seed should give same results."""
    p = ModelParams()
    x_grid = np.linspace(0.8, 5.0, 100)
    pi_star, b_star = make_constant_policy(x_grid, 0.5, 0.04)
    r1 = simulate_paths(x_grid, pi_star, b_star, p, x0=1.0, T=1.0, dt=1.0/12,
                        n_paths=100, seed=42)
    r2 = simulate_paths(x_grid, pi_star, b_star, p, x0=1.0, T=1.0, dt=1.0/12,
                        n_paths=100, seed=42)
    np.testing.assert_array_equal(r1['X_pct5'], r2['X_pct5'])


def test_simulation_pct5_le_pct20():
    """5th percentile must be <= 20th percentile at all times."""
    p = ModelParams()
    x_grid = np.linspace(0.8, 5.0, 100)
    pi_star, b_star = make_constant_policy(x_grid, 0.3, 0.04)
    result = simulate_paths(x_grid, pi_star, b_star, p, x0=1.0, T=5.0, dt=1.0/12,
                            n_paths=500, seed=10)
    assert np.all(result['X_pct5'] <= result['X_pct20'] + 1e-10)
    assert np.all(result['B_pct5'] <= result['B_pct20'] + 1e-10)
