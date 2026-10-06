"""Tests for model_params module."""
import sys
sys.path.insert(0, '/workspace/project')
import pytest
import numpy as np
from src.model_params import ModelParams, ScenarioParams, get_baseline_scenarios


def test_model_params_derived():
    """Test that derived parameters are computed correctly."""
    p = ModelParams()
    assert abs(p.r - 0.049) < 1e-10, f"r should be 0.049, got {p.r}"
    assert abs(p.theta - 0.135) < 1e-10, f"theta should be 0.135, got {p.theta}"
    assert abs(p.lambda_ - 1.0/23.6) < 1e-10
    assert p.delta == p.rho + p.lambda_


def test_admissibility():
    """Test that baseline params satisfy admissibility condition."""
    p = ModelParams()
    assert p.check_admissibility(), "Baseline should satisfy admissibility"


def test_admissibility_violation():
    """Test that admissibility fails when delta is very small."""
    p = ModelParams(rho=0.0, lambda_=0.001)  # Very small delta
    assert not p.check_admissibility(), "Should fail admissibility with tiny delta"


def test_roots_positive():
    """Test that characteristic roots are positive."""
    p = ModelParams()
    p1, p2 = p.compute_roots()
    assert p1 > 0 and p2 > 0, f"Both roots must be positive: p1={p1}, p2={p2}"


def test_roots_satisfy_quadratic():
    """Test that roots satisfy the characteristic equation."""
    p = ModelParams()
    p1, p2 = p.compute_roots()
    def quad(val):
        return (-0.5*p.theta**2*val*(val-1) - (p.r-p.delta)*val + p.r)
    assert abs(quad(p1)) < 1e-10, f"p1 doesn't satisfy quadratic: {quad(p1)}"
    assert abs(quad(-p2)) < 1e-10, f"-p2 doesn't satisfy quadratic: {quad(-p2)}"


def test_scenario_classification():
    """Test Case I / Case II classification."""
    p = ModelParams()
    scenarios = get_baseline_scenarios(p)
    # S1: alpha*x0=0.95 > b0/r=0.035/0.049 => Case I
    assert scenarios[0].is_case1
    # S2: alpha*x0=0.80 > 0.7143 => Case I
    assert scenarios[1].is_case1
    # S3: alpha*x0=0.80 < 0.04/0.049=0.8163 => Case II
    assert not scenarios[2].is_case1


def test_x_under_values():
    """Test x_under computation for each scenario."""
    p = ModelParams()
    scenarios = get_baseline_scenarios(p)
    assert abs(scenarios[0].x_under - 0.95) < 1e-10
    assert abs(scenarios[1].x_under - 0.80) < 1e-10
    expected_s3 = max(0.80, 0.04/0.049)
    assert abs(scenarios[2].x_under - expected_s3) < 1e-8


def test_p1_gt_unconstrained_threshold():
    """Test that p1 > 1/(1-gamma) as required by the paper."""
    p = ModelParams()
    p1, p2 = p.compute_roots()
    gamma = p.gamma1
    threshold = 1.0 / (1.0 - gamma)
    assert p1 > threshold, f"p1={p1} should be > 1/(1-gamma)={threshold}"
