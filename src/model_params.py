"""Model parameters for constrained retirement decumulation.

All calibration values follow the paper's stated baseline.
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class ModelParams:
    """Container for all model calibration parameters."""
    # Market parameters
    r1: float = 0.053          # Gross risk-free rate
    kappa1: float = 0.004      # Fee rate (proportional to wealth)
    kappa2: float = 0.0        # Terminal fee coefficient (zero in baseline)
    mu: float = 0.08           # Risky asset drift
    sigma: float = 0.2         # Risky asset volatility
    # Mortality and time preference
    lambda_: float = 1.0 / 23.6  # Constant hazard rate (~0.042372...)
    rho: float = 0.04          # Time preference rate
    # Utility exponents
    gamma1: float = 0.2        # Retiree CRRA exponent
    gamma2: float = 0.2        # Manager CRRA exponent
    # Initial wealth
    x0: float = 1.0

    def __post_init__(self) -> None:
        """Compute derived quantities."""
        self.r: float = self.r1 - self.kappa1          # Net risk-free rate: 0.049
        self.theta: float = (self.mu - self.r1) / self.sigma  # Market price of risk: 0.135
        self.delta: float = self.rho + self.lambda_    # Effective discount rate

    def check_admissibility(self) -> bool:
        """Check the unconstrained-benchmark admissibility condition.
        
        Requires delta > r*gamma + gamma*theta^2 / (2*(1-gamma))
        using the common gamma = gamma1 = gamma2.
        """
        gamma = self.gamma1
        lhs = self.delta
        rhs = self.r * gamma + gamma * self.theta**2 / (2.0 * (1.0 - gamma))
        return lhs > rhs

    def compute_roots(self) -> tuple[float, float]:
        """Compute characteristic roots p1 (>0) and p2 (>0) of
        
        -0.5*theta^2 * p*(p-1) - (r-delta)*p + r = 0
        
        Equivalently: 0.5*theta^2*p^2 - (0.5*theta^2 - (delta-r))*p - r = 0
        
        Returns: (p1, p2) where the roots of the quadratic are p1 and -p2.
        """
        # Quadratic: 0.5*theta^2*p^2 - [0.5*theta^2 - r + delta]*p - r = 0
        # Derived from: -0.5*theta^2*p*(p-1) - (r-delta)*p + r = 0
        a = 0.5 * self.theta**2
        b = -(0.5 * self.theta**2 + (self.delta - self.r))  # = r - delta - 0.5*theta^2
        c = -self.r
        disc = b**2 - 4.0 * a * c
        assert disc >= 0, f"Negative discriminant: {disc}"
        sqrt_disc = np.sqrt(disc)
        p1 = (-b + sqrt_disc) / (2.0 * a)
        neg_p2 = (-b - sqrt_disc) / (2.0 * a)
        p2 = -neg_p2
        assert p1 > 0 and p2 > 0, f"Roots must be positive: p1={p1}, p2={p2}"
        return p1, p2


@dataclass
class ScenarioParams:
    """Parameters for a single guarantee scenario."""
    b0: float          # Minimum benefit floor
    alpha: float       # Wealth floor fraction of initial wealth
    model: ModelParams = field(default_factory=ModelParams)

    def __post_init__(self) -> None:
        """Compute scenario-specific derived quantities."""
        self.x_under: float = max(self.alpha * self.model.x0,
                                   self.b0 / self.model.r)
        # Case classification
        self.is_case1: bool = (self.alpha * self.model.x0 > self.b0 / self.model.r)


# Baseline scenarios from the paper
def get_baseline_scenarios(model: ModelParams = None) -> list[ScenarioParams]:
    """Return the three baseline guarantee scenarios from the paper."""
    if model is None:
        model = ModelParams()
    s1 = ScenarioParams(b0=0.035, alpha=0.95, model=model)
    s2 = ScenarioParams(b0=0.035, alpha=0.80, model=model)
    s3 = ScenarioParams(b0=0.04,  alpha=0.80, model=model)
    return [s1, s2, s3]
