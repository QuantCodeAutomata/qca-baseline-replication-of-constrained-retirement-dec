"""Case I constrained HJB solver via BVP in h-space.

Case I applies when alpha*x0 > b0/r.  The solution H(h) satisfies:

  -0.5*theta^2*h^2*H'' - (r-delta)*h*H' - U2'(H)*h^2*H' + r*H - I(1/h) = 0

on h in [y0, h_max], with BCs:
  H'(y0) = 0  (smooth pasting)
  H'(h_max)*h_max = q*H(h_max)  (power-law asymptotics, q=1/(1-gamma))

y0 is determined by H(y0) = x_under (wealth floor).

We use scipy.integrate.solve_bvp (two-point BVP) instead of forward ODE shooting,
which was found to be numerically unstable due to exponential growth of the wrong
fundamental solution in the forward direction.
# Custom -- Context7 confirmed no library solves this transform-based HJB.
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import solve_bvp
from scipy.optimize import brentq
from scipy.interpolate import PchipInterpolator
from src.model_params import ModelParams, ScenarioParams


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def benefit_selector(p: np.ndarray, b0: float, gamma1: float) -> np.ndarray:
    """I(p) = max{b0, p^(1/(gamma1-1))}.  Optimal benefit given shadow price p."""
    p = np.asarray(p, dtype=float)
    exp = 1.0 / (gamma1 - 1.0)   # negative since gamma1 < 1
    return np.maximum(b0, p ** exp)


def I_of_inv_h(h: np.ndarray, b0: float, gamma1: float) -> np.ndarray:
    """I(1/h) = max{b0, h^(1/(1-gamma1))} analytically."""
    h = np.asarray(h, dtype=float)
    q = 1.0 / (1.0 - gamma1)
    return np.maximum(b0, h ** q)


def U2_prime(x: np.ndarray, params: ModelParams) -> np.ndarray:
    """U2'(x) = kappa1^gamma2 * x^(gamma2-1) + lambda * x^(gamma1-1)."""
    x = np.asarray(x, dtype=float)
    t1 = params.kappa1 ** params.gamma2 * x ** (params.gamma2 - 1.0)
    t2 = params.lambda_ * x ** (params.gamma1 - 1.0)
    return t1 + t2


def U2_value(x: np.ndarray, params: ModelParams) -> np.ndarray:
    """U2(x) = (kappa1*x)^gamma2/gamma2 + lambda*x^gamma1/gamma1."""
    x = np.asarray(x, dtype=float)
    t1 = (params.kappa1 * x) ** params.gamma2 / params.gamma2
    t2 = params.lambda_ * x ** params.gamma1 / params.gamma1
    return t1 + t2


# ---------------------------------------------------------------------------
# BVP solver
# ---------------------------------------------------------------------------

def _make_ode_rhs(params: ModelParams, sc: ScenarioParams):
    """Return a vectorized ODE RHS for use with solve_bvp."""
    b0, gamma1 = sc.b0, params.gamma1

    def ode_rhs(h: np.ndarray, state: np.ndarray) -> np.ndarray:
        H, dH = state[0], state[1]
        H_safe = np.maximum(H, 1e-4)
        I_val = I_of_inv_h(h, b0, gamma1)
        U2p = U2_prime(H_safe, params)
        numer = (
            -(params.r - params.delta) * h * dH
            - U2p * h ** 2 * dH
            + params.r * H_safe
            - I_val
        )
        dH2 = numer / (0.5 * params.theta ** 2 * h ** 2)
        return np.vstack([dH, dH2])

    return ode_rhs


def _make_bcs(h_max: float, q: float):
    """Return boundary conditions for solve_bvp.

    Left (h=y0):  H'(y0) = 0  (smooth pasting)
    Right (h=h_max): H'(h_max)*h_max - q*H(h_max) = 0  (power-law slope)
    """
    def bcs(ya: np.ndarray, yb: np.ndarray) -> np.ndarray:
        return np.array([ya[1], yb[1] * h_max - q * yb[0]])

    return bcs


def solve_bvp_case1(
    h_min: float,
    params: ModelParams,
    sc: ScenarioParams,
    h_max: float = 200.0,
    n_init: int = 200,
    tol: float = 1e-6,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, bool]:
    """Solve the Case I BVP on [h_min, h_max].

    Returns (h_grid, H_grid, dH_grid, success).
    """
    q = 1.0 / (1.0 - params.gamma1)
    # Asymptotic constant from unconstrained benchmark (matches large-h limit).
    # For heterogeneous gamma, temporarily override gamma2=gamma1 to get a usable
    # asymptotic scale for the initial guess.
    from src.benchmark import UnconstainedBenchmark, solve_b_bar
    try:
        bm = UnconstainedBenchmark(params)
        C_un = bm.b_bar ** q
    except AssertionError:
        # Heterogeneous gamma: use gamma1-only approximation for the initial guess
        from src.model_params import ModelParams
        params_eq = ModelParams(
            r1=params.r1, kappa1=params.kappa1, kappa2=params.kappa2,
            mu=params.mu, sigma=params.sigma, lambda_=params.lambda_,
            rho=params.rho, gamma1=params.gamma1, gamma2=params.gamma1,
        )
        try:
            bm_eq = UnconstainedBenchmark(params_eq)
            C_un = bm_eq.b_bar ** q
        except Exception:
            C_un = 15.0 ** q  # robust fallback

    h_mesh = np.logspace(np.log10(h_min), np.log10(h_max), n_init)
    H_guess = C_un * h_mesh ** q
    dH_guess = q * C_un * h_mesh ** (q - 1.0)
    y_init = np.vstack([H_guess, dH_guess])

    ode_rhs = _make_ode_rhs(params, sc)
    bcs_fn = _make_bcs(h_max, q)

    sol = solve_bvp(
        fun=ode_rhs,
        bc=bcs_fn,
        x=h_mesh,
        y=y_init,
        max_nodes=8000,
        tol=tol,
        verbose=0,
    )
    if not sol.success:
        return None, None, None, False

    h_out = sol.x
    H_out = sol.y[0]
    dH_out = sol.y[1]
    return h_out, H_out, dH_out, True


def H_at_left(
    h_min: float,
    params: ModelParams,
    sc: ScenarioParams,
    h_max: float = 200.0,
) -> float:
    """Solve BVP and return H at the left endpoint h_min.

    Used as objective for finding y0: H_at_left(y0) = x_under.
    """
    h_arr, H_arr, dH_arr, ok = solve_bvp_case1(h_min, params, sc, h_max)
    if not ok or H_arr is None:
        return 1e6  # large sentinel when BVP fails
    return float(H_arr[0])


def find_y0_bvp(
    params: ModelParams,
    sc: ScenarioParams,
    h_max: float = 200.0,
    tol: float = 1e-6,
) -> float:
    """Find y0 such that H_at_left(y0) = x_under via brentq.

    Returns y0.
    """
    x_under = sc.x_under

    def objective(h_min: float) -> float:
        return H_at_left(h_min, params, sc, h_max) - x_under

    # Bracket: scan to find sign change
    h_lo = 0.005
    h_hi_max = 0.09   # y_bar is typically around 0.07 for baseline params
    n_scan = 20
    h_cands = np.geomspace(h_lo, h_hi_max, n_scan)

    blo, bhi = None, None
    prev_val = objective(h_cands[0])
    for hc in h_cands[1:]:
        curr_val = objective(hc)
        if prev_val * curr_val <= 0.0:
            blo = h_cands[np.where(h_cands == hc)[0][0] - 1]
            bhi = hc
            break
        prev_val = curr_val

    if blo is None:
        raise RuntimeError(
            f"find_y0_bvp: no sign change found. "
            f"H values at boundary: {[H_at_left(h, params, sc, h_max) for h in h_cands[[0,-1]]]}"
        )

    y0 = brentq(objective, blo, bhi, xtol=tol, rtol=tol)
    return y0


# ---------------------------------------------------------------------------
# Full solver class
# ---------------------------------------------------------------------------

class Case1Solver:
    """Solve the Case I constrained decumulation problem.

    Uses scipy.integrate.solve_bvp for numerical stability (forward ODE shooting
    is numerically unstable due to exponential growth of the spurious solution).
    # Custom -- Context7 confirmed no library solves this transform-based HJB.
    """

    def __init__(
        self,
        params: ModelParams,
        sc: ScenarioParams,
        h_max: float = 200.0,
        n_h: int = 500,
        x_grid_size: int = 500,
        bvp_tol: float = 1e-6,
    ) -> None:
        assert sc.is_case1, "Case1Solver requires Case I (alpha*x0 > b0/r)."
        self.params = params
        self.sc = sc
        self.h_max = h_max
        self.n_h = n_h
        self.x_grid_size = x_grid_size
        self.bvp_tol = bvp_tol
        self._solved = False

        # outputs set by solve()
        self.y0: float
        self.p1: float
        self.p2: float
        self.h_grid: np.ndarray
        self.H_grid: np.ndarray
        self.dH_grid: np.ndarray
        self.x_grid: np.ndarray
        self.b_star: np.ndarray
        self.pi_star: np.ndarray
        self.payout_ratio: np.ndarray
        self.h_of_x: np.ndarray

    def solve(self) -> None:
        """Full Case I solution: find y0 via BVP, build policy maps."""
        self.p1, self.p2 = self.params.compute_roots()

        print(
            f"  [Case1] Finding y0 (x_under={self.sc.x_under:.4f}, "
            f"b0={self.sc.b0:.4f}, alpha={self.sc.alpha:.4f})..."
        )
        self.y0 = find_y0_bvp(self.params, self.sc, self.h_max, self.bvp_tol)
        print(f"  [Case1] y0 = {self.y0:.6f}")

        # Final high-resolution BVP solve from y0 to h_max
        h_grid, H_grid, dH_grid, ok = solve_bvp_case1(
            self.y0, self.params, self.sc, self.h_max, n_init=self.n_h
        )
        if not ok:
            raise RuntimeError("Case1Solver: final BVP solve failed.")

        self.h_grid = h_grid
        self.H_grid = H_grid
        self.dH_grid = dH_grid

        # Build interpolators (monotone PCHIP)
        self.H_interp = PchipInterpolator(h_grid, H_grid)
        self.dH_interp = PchipInterpolator(h_grid, dH_grid)

        # Build x grid
        x_min = self.sc.x_under
        x_max = float(np.clip(np.max(H_grid), x_min + 0.01, 25.0))
        self.x_grid = np.linspace(x_min, x_max, self.x_grid_size)
        self._build_policy_maps()
        self._solved = True

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

        # b*(x) = I(1/h(x)) = max{b0, h^q}
        q = 1.0 / (1.0 - self.params.gamma1)
        self.b_star = np.maximum(
            I_of_inv_h(h_vals, self.sc.b0, self.params.gamma1),
            self.sc.b0,
        )

        # pi*(x) = theta * h(x) * H'(h(x)) / (sigma * x)
        self.pi_star = np.maximum(
            self.params.theta * h_vals * dH_vals / (self.params.sigma * self.x_grid),
            0.0,
        )

        self.payout_ratio = self.b_star / self.x_grid

    def marginal_value(self, x: np.ndarray) -> np.ndarray:
        """v'(x) = 1/h(x) via interpolation."""
        x = np.asarray(x, dtype=float)
        h_at_x = np.interp(x, self.x_grid, self.h_of_x)
        return 1.0 / np.maximum(h_at_x, 1e-12)


# ---------------------------------------------------------------------------
# Analytical G_bar function (used for validation / tests)
# G_bar(y) = r*x_under - p1*y^p1 * ∫_y^∞ I(1/z) * z^{-p1-1} dz
# This ignores the U2' correction term present in the full G(y).
# ---------------------------------------------------------------------------

def compute_G_bar(
    y: float,
    params: ModelParams,
    sc: ScenarioParams,
    p1: float,
) -> float:
    """Analytical G_bar(y) using the purely I(1/z) integrand (no U2' term).

    G_bar(y) = r*x_under - p1*y^p1 * ∫_y^∞ I(1/z) z^{-p1-1} dz

    I(1/z) = max{b0, z^q}, with q = 1/(1-gamma1).
    Split the integral at h_star = b0^{1-gamma1} where the floor becomes inactive.
    """
    q = 1.0 / (1.0 - params.gamma1)
    b0 = sc.b0
    # Critical h: z^q = b0 => z = b0^{1/q} = b0^{1-gamma1}
    h_star = b0 ** (1.0 - params.gamma1)

    def integral_from_y_to_inf(y_val: float) -> float:
        """∫_y^∞ I(1/z) z^{-p1-1} dz analytically."""
        if y_val >= h_star:
            # All in the unconstrained region: I(1/z) = z^q
            # ∫_y^∞ z^{q-p1-1} dz = y^{q-p1}/(p1-q)  (since p1 > q)
            return y_val ** (q - p1) / (p1 - q)
        else:
            # Two pieces:
            # ∫_y^{h_star} b0 z^{-p1-1} dz = b0*(y^{-p1} - h_star^{-p1})/p1
            # ∫_{h_star}^∞ z^{q-p1-1} dz = h_star^{q-p1}/(p1-q)
            piece1 = b0 * (y_val ** (-p1) - h_star ** (-p1)) / p1
            piece2 = h_star ** (q - p1) / (p1 - q)
            return piece1 + piece2

    g_bar = params.r * sc.x_under - p1 * y ** p1 * integral_from_y_to_inf(y)
    return g_bar
