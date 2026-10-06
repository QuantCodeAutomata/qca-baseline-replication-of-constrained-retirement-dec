"""Tests for the unconstrained benchmark."""
import sys
sys.path.insert(0, '/workspace/project')
import pytest
import numpy as np
from src.model_params import ModelParams
from src.benchmark import UnconstainedBenchmark, solve_b_bar, compute_eta


def test_b_bar_positive():
    """b_bar must be positive."""
    p = ModelParams()
    b_bar = solve_b_bar(p)
    assert b_bar > 0, f"b_bar={b_bar} must be positive"


def test_pi_star_positive():
    """Unconstrained risky fraction must be positive."""
    p = ModelParams()
    bench = UnconstainedBenchmark(p)
    assert bench.pi_star > 0


def test_value_function_increasing():
    """V_un(x) must be increasing in x (monotone)."""
    p = ModelParams()
    bench = UnconstainedBenchmark(p)
    x = np.linspace(0.5, 5.0, 50)
    V = bench.value(x)
    assert np.all(np.diff(V) > 0), "V_un should be increasing"


def test_value_function_concave():
    """V_un(x) must be concave (CRRA with gamma < 1)."""
    p = ModelParams()
    bench = UnconstainedBenchmark(p)
    x = np.linspace(0.5, 5.0, 100)
    V = bench.value(x)
    d2V = np.diff(np.diff(V))
    assert np.all(d2V < 1e-8), "V_un should be concave"


def test_payout_ratio_constant():
    """Unconstrained payout ratio b*/x should be constant."""
    p = ModelParams()
    bench = UnconstainedBenchmark(p)
    x = np.linspace(0.5, 5.0, 20)
    ratio = bench.payout_ratio(x)
    assert np.allclose(ratio, ratio[0], rtol=1e-10), "Payout ratio should be constant"


def test_value_power_law():
    """Value function should follow power law V(x) = b_bar * x^gamma / gamma."""
    p = ModelParams()
    bench = UnconstainedBenchmark(p)
    x = np.array([1.0, 2.0, 3.0])
    V = bench.value(x)
    gamma = p.gamma1
    expected = bench.b_bar * x**gamma / gamma
    np.testing.assert_allclose(V, expected, rtol=1e-10)


def test_eta_positive():
    """eta must be positive."""
    p = ModelParams()
    eta = compute_eta(p)
    assert eta > 0
