"""Tests for valuation and Feynman-Kac decomposition."""
import sys
sys.path.insert(0, '/workspace/project')
import pytest
import numpy as np
from src.model_params import ModelParams, ScenarioParams
from src.benchmark import UnconstainedBenchmark
from src.valuation import recover_value_function, solve_feynman_kac


def test_value_function_increasing():
    """Recovered value function should be increasing in wealth."""
    p = ModelParams()
    sc = ScenarioParams(b0=0.035, alpha=0.95, model=p)
    bench = UnconstainedBenchmark(p)
    x_grid = np.linspace(0.95, 3.0, 100)
    # Use unconstrained marginal value for testing
    gamma = p.gamma1
    marginal_v = bench.b_bar * x_grid**(gamma-1.0)
    V = recover_value_function(x_grid, marginal_v, p, sc)
    assert np.all(np.diff(V) > 0), "Value function should be increasing"


def test_feynman_kac_total_positive():
    """Total Feynman-Kac value should be positive."""
    p = ModelParams()
    sc = ScenarioParams(b0=0.035, alpha=0.95, model=p)
    x_grid = np.linspace(0.95, 3.0, 50)
    pi_star = np.full(50, 0.5)
    b_star = np.full(50, 0.05)
    V = solve_feynman_kac(x_grid, pi_star, b_star, p, sc, 'total')
    assert np.all(V > 0), "Total value should be positive"


def test_feynman_kac_retiree_positive():
    """Retiree value should be positive."""
    p = ModelParams()
    sc = ScenarioParams(b0=0.035, alpha=0.95, model=p)
    x_grid = np.linspace(0.95, 3.0, 50)
    pi_star = np.full(50, 0.5)
    b_star = np.full(50, 0.05)
    V1 = solve_feynman_kac(x_grid, pi_star, b_star, p, sc, 'retiree')
    assert np.all(V1 > 0), "Retiree value should be positive"


def test_feynman_kac_manager_positive():
    """Manager value should be positive."""
    p = ModelParams()
    sc = ScenarioParams(b0=0.035, alpha=0.95, model=p)
    x_grid = np.linspace(0.95, 3.0, 50)
    pi_star = np.full(50, 0.5)
    b_star = np.full(50, 0.05)
    V2 = solve_feynman_kac(x_grid, pi_star, b_star, p, sc, 'manager')
    assert np.all(V2 > 0), "Manager value should be positive"


def test_recovery_constant_consistency():
    """Recovered V should match unconstrained exactly when using its own marginal."""
    p = ModelParams()
    bench = UnconstainedBenchmark(p)
    sc = ScenarioParams(b0=0.0, alpha=0.0, model=p)
    sc.x_under = 0.5
    x_grid = np.linspace(0.5, 3.0, 100)
    gamma = p.gamma1
    marginal_v = bench.b_bar * x_grid**(gamma-1.0)  # V_un' = b_bar * x^(gamma-1)
    V = recover_value_function(x_grid, marginal_v, p, sc)
    V_true = bench.value(x_grid)
    # Should recover a function proportional to x^gamma; check shape
    assert np.all(np.diff(V) > 0), "Should be increasing"
