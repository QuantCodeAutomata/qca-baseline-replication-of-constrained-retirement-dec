"""Case II constrained HJB solver via BVP in h-space.

Case II applies when alpha*x0 <= b0/r, so x_under = b0/r.

The same ODE as Case I holds:
  -0.5*theta^2*h^2*H'' - (r-delta)*h*H' - U2'(H)*h^2*H' + r*H - I(1/h) = 0

but with different boundary conditions reflecting H(0) = b0/r:
  Left  (h = h_min):  H(h_min) = b0/r        [Dirichlet — from H(0) = b0/r]
  Right (h = h_max):  H'(h_max)*h_max = q*H(h_max)  [power-law asymptotics]

We use scipy.integrate.solve_bvp for robustness — the fixed-point integral-equation
approach is numerically divergent because I(1/z)~z^{-q} near z=0 is not integrable
for the lower integral (exponent -q + p2 - 1 < -1 when p2 < 1/(1-γ)).
# Custom — Context7 confirmed no library solves this transform-based HJB.
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import solve_bvp
from scipy.interpolate import PchipInterpolator
from src.model_params import ModelParams, ScenarioParams
from src.solver_case1 import (
    benefit_selector, U2_value, U2_prime,
    I_of_inv_h, _make_ode_rhs,
)


def _make_bcs_case2(h_min: float, h_max: float, q: float, b0r: float):
    """BCs for Case II: Dirichlet at left, power-law at right.

    Left  (h_min): H - b0r = 0
    Right (h_max): H'*h_max - q*H = 0
    """
    def bcs(ya: np.ndarray, yb: np.ndarray) -> np.ndarray:
        return np.array([
            ya[0] - b0r,               # H(h_min) = b0/r
            yb[1] * h_max - q * yb[0], # power-law slope at h_max
        ])
    return bcs


def solve_bvp_case2(
    params: ModelParams,
    sc: ScenarioParams,
    h_min: float = 0.01,
    h_max: float = 80.0,
    n_init: int = 300,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Solve Case II BVP on [h_min, h_max].

    Returns (h_grid, H_grid, dH_grid, success).
    """
    q = 1.0 / (1.0 - params.gamma1)
    b0r = sc.b0 / params.r   # = x_under for Case II

    # Asymptotic scale from unconstrained benchmark: H(h) ~ C_un * h^q for large h
    from src.benchmark import UnconstainedBenchmark
    bm = UnconstainedBenchmark(params)
    C_un = bm.b_bar ** q  # matches unconstrained v'(x) = b_bar * x^{γ-1}

    h_mesh = np.logspace(np.log10(h_min), np.log10(h_max), n_init)

    # Initial guess: smoothly blends the boundary value b0r at h_min to
    # unconstrained power-law at h_max
    blend = (h_mesh - h_min) / (h_max - h_min)
    H_guess = b0r + blend * C_un * h_mesh ** q
    dH_guess = (1.0 / (h_max - h_min)) * C_un * h_mesh ** q + blend * q * C_un * h_mesh ** (q - 1.0)
    y_init = np.vstack([H_guess, dH_guess])

    ode_rhs = _make_ode_rhs(params, sc)
    bcs_fn = _make_bcs_case2(h_min, h_max, q, b0r)

    sol = solve_bvp(
        fun=ode_rhs,
        bc=bcs_fn,
        x=h_mesh,
        y=y_init,
        max_nodes=10000,
        tol=tol,
        verbose=0,
    )
    if not sol.success:
        # Try with relaxed tolerance
        sol = solve_bvp(
            fun=ode_rhs,
            bc=bcs_fn,
            x=h_mesh,
            y=y_init,
            max_nodes=15000,
            tol=tol * 100,
            verbose=0,
        )
    return sol.x, sol.y[0], sol.y[1], sol.success


class Case2Solver:
    """Solve the Case II constrained decumulation problem via BVP.

    Uses scipy.integrate.solve_bvp for numerical stability — the ODE is the same
    as Case I but with a Dirichlet left BC H(h_min) = b0/r instead of H'(y0) = 0.
    # Custom — Context7 confirmed no library solves this transform-based HJB.
    """

    def __init__(
        self,
        params: ModelParams,
        sc: ScenarioParams,
        h_min: float = 0.01,
        h_max: float = 80.0,
        n_h: int = 300,
        # legacy keyword arguments kept for backward compatibility (ignored)
        omega: float = 0.4,
        max_iter: int = 600,
        tol: float = 1e-6,
        x_grid_size: int = 500,
    ) -> None:
        assert not sc.is_case1, "Case2Solver requires Case II (alpha*x0 <= b0/r)."
        self.params = params
        self.sc = sc
        self.h_min = h_min
        self.h_max = h_max
        self.n_h = n_h
        self.tol = tol
        self.x_grid_size = x_grid_size
        self._solved = False

    def solve(self) -> None:
        """Full Case II solution: BVP on [h_min, h_max], then build policy maps."""
        self.p1, self.p2 = self.params.compute_roots()
        b0r = self.sc.b0 / self.params.r

        print(f"  Case II BVP solver (x_under={self.sc.x_under:.4f}, b0/r={b0r:.4f})")
        h_grid, H_grid, dH_grid, ok = solve_bvp_case2(
            self.params, self.sc,
            h_min=self.h_min,
            h_max=self.h_max,
            n_init=self.n_h,
            tol=self.tol,
        )
        if not ok:
            print("  [Case2] BVP did not reach full convergence; using best available solution.")

        # Enforce monotonicity and lower bound
        H_grid = np.maximum(H_grid, b0r)
        # Ensure H is strictly increasing by clipping any backslides
        for i in range(1, len(H_grid)):
            if H_grid[i] <= H_grid[i - 1]:
                H_grid[i] = H_grid[i - 1] + 1e-8

        self.h_grid = h_grid
        self.H_grid = H_grid
        self.dH_grid = np.maximum(dH_grid, 0.0)

        self.H_interp = PchipInterpolator(h_grid, H_grid)
        self.dH_interp = PchipInterpolator(h_grid, self.dH_grid)

        # Build wealth grid and policy maps
        x_min = self.sc.x_under
        x_max = float(np.clip(np.max(H_grid), x_min + 0.01, 25.0))
        self.x_grid = np.linspace(x_min, x_max, self.x_grid_size)
        self._build_policy_maps()
        self._solved = True

        # Diagnostics
        print(f"  [Case2] H range: [{H_grid[0]:.4f}, {H_grid[-1]:.4f}]  "
              f"h range: [{h_grid[0]:.4f}, {h_grid[-1]:.4f}]")
        print(f"  [Case2] H(h_min) = {H_grid[0]:.6f} (target b0/r = {b0r:.6f})")

    def _build_policy_maps(self) -> None:
        """Invert monotone H to get h(x); compute b*, pi*, payout ratio."""
        h_vals = np.zeros(len(self.x_grid))
        for i, x in enumerate(self.x_grid):
            if x <= self.H_grid[0]:
                h_vals[i] = self.h_grid[0]
            elif x >= self.H_grid[-1]:
                h_vals[i] = self.h_grid[-1]
            else:
                lo_h, hi_h = self.h_grid[0], self.h_grid[-1]
                for _ in range(60):
                    mid_h = 0.5 * (lo_h + hi_h)
                    if float(self.H_interp(mid_h)) < x:
                        lo_h = mid_h
                    else:
                        hi_h = mid_h
                h_vals[i] = 0.5 * (lo_h + hi_h)

        self.h_of_x = h_vals
        dH_vals = np.array([float(self.dH_interp(h)) for h in h_vals])
        dH_vals = np.maximum(dH_vals, 0.0)

        # b*(x) = I(1/h(x)) = max{b0, h^{1/(1-γ)}}
        self.b_star = np.maximum(
            I_of_inv_h(h_vals, self.sc.b0, self.params.gamma1),
            self.sc.b0,
        )

        # pi*(x) = theta * h(x) * H'(h(x)) / (sigma * x)
        with np.errstate(divide='ignore', invalid='ignore'):
            self.pi_star = np.where(
                self.x_grid > 0,
                self.params.theta * h_vals * dH_vals / (self.params.sigma * self.x_grid),
                0.0,
            )
        self.pi_star = np.maximum(self.pi_star, 0.0)
        self.payout_ratio = self.b_star / self.x_grid

    def marginal_value(self, x: np.ndarray) -> np.ndarray:
        """v'(x) = 1/h(x) via interpolation."""
        h_interp_x = np.interp(x, self.x_grid, self.h_of_x)
        return 1.0 / np.maximum(h_interp_x, 1e-12)
