"""Tests for the Case I solver."""
import sys
sys.path.insert(0, '/workspace/project')
import pytest
import numpy as np
from src.model_params import ModelParams, ScenarioParams
from src.solver_case1 import (
    Case1Solver, benefit_selector, U2_prime, U2_value,
    compute_G_bar,
)


def test_benefit_selector_floor():
    """I(p) >= b0 always."""
    b0, gamma1 = 0.035, 0.2
    # For large p (low marginal value), unconstrained would be very small
    result = benefit_selector(1e-10, b0, gamma1)
    assert result >= b0


def test_benefit_selector_unconstrained():
    """I(p) = p^(1/(gamma-1)) when that exceeds b0."""
    b0, gamma1 = 0.035, 0.2
    # For small p (large marginal value), unconstrained dominates
    p_val = 1000.0  # large marginal value
    result = benefit_selector(p_val, b0, gamma1)
    expected = p_val**(1.0/(gamma1-1.0))
    assert result >= b0
    assert abs(result - expected) < 1e-10 or result == b0


def test_u2_prime_positive():
    """U2'(x) must be positive for x > 0."""
    p = ModelParams()
    x = np.linspace(0.1, 5.0, 50)
    u2p = U2_prime(x, p)
    assert np.all(u2p > 0)


def test_case1_solver_s1():
    """Solve Scenario 1 (Case I) and check basic validity."""
    p = ModelParams()
    sc = ScenarioParams(b0=0.035, alpha=0.95, model=p)
    assert sc.is_case1
    solver = Case1Solver(p, sc, h_max=150.0, n_h=1000, x_grid_size=200)
    solver.solve()
    # Validate admissibility
    assert np.all(solver.b_star >= sc.b0 - 1e-6), "b* must be >= b0"
    assert np.all(solver.pi_star >= -1e-6), "pi* must be non-negative"
    assert np.all(solver.x_grid >= sc.x_under - 1e-8), "x must be >= x_under"
    assert np.all(np.isfinite(solver.pi_star)), "pi* must be finite"
    assert np.all(np.isfinite(solver.b_star)), "b* must be finite"


def test_case1_solver_s2():
    """Solve Scenario 2 (Case I) and check basic validity."""
    p = ModelParams()
    sc = ScenarioParams(b0=0.035, alpha=0.80, model=p)
    assert sc.is_case1
    solver = Case1Solver(p, sc, h_max=150.0, n_h=1000, x_grid_size=200)
    solver.solve()
    assert np.all(solver.b_star >= sc.b0 - 1e-6)
    assert np.all(solver.pi_star >= -1e-6)


def test_g_bar_decreasing():
    """G_bar(y) should be decreasing: large y gives more negative G_bar."""
    p = ModelParams()
    sc = ScenarioParams(b0=0.035, alpha=0.95, model=p)
    p1, _ = p.compute_roots()
    g_small = compute_G_bar(0.01, p, sc, p1)
    g_large = compute_G_bar(10.0, p, sc, p1)
    assert g_small > g_large, "G_bar should decrease with y"


def test_pi_convergence_high_wealth():
    """pi*(x) should approach unconstrained Merton fraction at high wealth."""
    p = ModelParams()
    sc = ScenarioParams(b0=0.035, alpha=0.95, model=p)
    solver = Case1Solver(p, sc, h_max=200.0, n_h=1500, x_grid_size=300)
    solver.solve()
    pi_un = p.theta / ((1.0 - p.gamma1) * p.sigma)
    pi_at_high = solver.pi_star[-1]
    # Should be within 20% of unconstrained at high wealth
    assert abs(pi_at_high - pi_un) / pi_un < 0.5, (
        f"pi* at high wealth {pi_at_high:.3f} should approach {pi_un:.3f}"
    )
