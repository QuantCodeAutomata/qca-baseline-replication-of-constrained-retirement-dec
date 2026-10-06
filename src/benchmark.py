"""Unconstrained retirement decumulation benchmark.

For equal CRRA (gamma1=gamma2=gamma), the value function is explicit:
    V_un(x) = b_bar * x^gamma / gamma

where b_bar solves a scalar nonlinear equation.
# Custom implementation -- Context7 found no library equivalent for this
# transform-based HJB closed-form benchmark.
"""
import numpy as np
from scipy.optimize import brentq
from src.model_params import ModelParams


def compute_eta(params: ModelParams) -> float:
    """Compute the scalar eta = kappa1^gamma2 + lambda_ used in V_un.
    
    When kappa2=0, U_m,2 = 0 so U2(x) = (kappa1*x)^gamma2/gamma2 + lambda*x^gamma1/gamma1.
    For equal gamma, eta = kappa1^gamma + lambda_.
    """
    gamma = params.gamma1  # equal CRRA
    return params.kappa1**gamma + params.lambda_


def solve_b_bar(params: ModelParams) -> float:
    """Solve the scalar equation for b_bar in the unconstrained benchmark.
    
    Equation: eta + (1-gamma)*b_bar^(gamma/(gamma-1)) - b_bar*C = 0
    where C = delta - r*gamma - gamma*theta^2 / (2*(1-gamma))
    
    Uses brentq root finding on the transformed equation.
    """
    assert params.gamma1 == params.gamma2, "Benchmark formula requires equal CRRA"
    gamma = params.gamma1
    eta = compute_eta(params)
    C = params.delta - params.r * gamma - gamma * params.theta**2 / (2.0 * (1.0 - gamma))
    assert C > 0, f"Admissibility violated: C={C} must be positive"

    # g(b_bar) = eta + (1-gamma)*b_bar^(gamma/(gamma-1)) - b_bar * C
    exp = gamma / (gamma - 1.0)  # negative exponent

    def g(b_bar: float) -> float:
        return eta + (1.0 - gamma) * b_bar**exp - b_bar * C

    # b_bar should be positive; find bracket
    # For small b_bar, b_bar^exp -> +inf (negative exponent)
    # For large b_bar, dominated by -b_bar*C
    b_lo = 1e-8
    b_hi = 1e4
    # Ensure sign change
    assert g(b_lo) > 0 and g(b_hi) < 0, "Cannot bracket root for b_bar"
    return brentq(g, b_lo, b_hi, xtol=1e-12, rtol=1e-12)


class UnconstainedBenchmark:
    """Closed-form unconstrained benchmark for equal-CRRA retirement decumulation."""

    def __init__(self, params: ModelParams) -> None:
        self.params = params
        assert params.check_admissibility(), "Admissibility condition not satisfied"
        self.b_bar = solve_b_bar(params)
        gamma = params.gamma1
        # Unconstrained optimal payout level scalar: b_un*(x) = b_bar_opt * x
        self.b_bar_opt: float = self.b_bar**(1.0 / (gamma - 1.0))
        # Unconstrained risky fraction (constant)
        self.pi_star: float = params.theta / ((1.0 - gamma) * params.sigma)

    def value(self, x: np.ndarray) -> np.ndarray:
        """Compute unconstrained value function V_un(x) = b_bar * x^gamma / gamma."""
        gamma = self.params.gamma1
        return self.b_bar * x**gamma / gamma

    def benefit_policy(self, x: np.ndarray) -> np.ndarray:
        """Compute unconstrained optimal benefit b_un*(x) = b_bar_opt * x."""
        return self.b_bar_opt * x

    def pi_policy(self, x: np.ndarray) -> np.ndarray:
        """Compute unconstrained optimal risky fraction pi_un* (constant)."""
        return np.full_like(np.asarray(x, dtype=float), self.pi_star)

    def payout_ratio(self, x: np.ndarray) -> np.ndarray:
        """Compute unconstrained optimal payout ratio b*(x)/x = b_bar_opt."""
        return np.full_like(np.asarray(x, dtype=float), self.b_bar_opt)
