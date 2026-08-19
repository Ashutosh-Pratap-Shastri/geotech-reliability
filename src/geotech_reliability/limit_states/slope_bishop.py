"""
Bishop's Simplified method of slices for circular slip surfaces.

The core calculation (`bishop_simplified_fos`) is a pure function with no
dependency on the reliability engine, so it can be tested and used
standalone. `SlopeBishopLimitState` wraps it to plug into Monte Carlo/FORM.

References
----------
Bishop, A.W. (1955). "The use of the slip circle in the stability
analysis of slopes." Geotechnique, 5(1), 7-17.

Fredlund, D.G. & Krahn, J. (1977). "Comparison of slope stability
methods of analysis." Canadian Geotechnical Journal, 14(3), 429-439.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geotech_reliability.limit_states.base import LimitState
from geotech_reliability.core.distributions import RandomVariable


@dataclass
class CircularSlipSurface:
    """Geometry of a circular slip surface, defined by center and radius."""
    x_center: float
    y_center: float
    radius: float


@dataclass
class SliceGeometry:
    """
    Pre-computed per-slice geometry for a circular slip surface cutting
    through a slope profile. In a full implementation these come from
    intersecting the circle with the ground surface and layer boundaries;
    here they are supplied directly so the FoS routine itself stays a
    pure, testable function independent of surface-generation logic.
    """
    width: np.ndarray       # slice width, b_i (m)
    height: np.ndarray      # average slice height, h_i (m)
    alpha: np.ndarray       # base inclination angle, alpha_i (radians)
    u: np.ndarray           # pore pressure at slice base, u_i (kPa)


def bishop_simplified_fos(
    slices: SliceGeometry,
    gamma: float,
    c: float,
    phi_deg: float,
    kh: float = 0.0,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> float:
    """
    Compute the factor of safety by Bishop's Simplified method, for a
    single homogeneous material (see `bishop_simplified_fos_layered` for
    the two-layer case). Solved iteratively since FoS appears on both
    sides of the governing equation.

    Parameters
    ----------
    slices : SliceGeometry
        Per-slice width, height, base angle, and pore pressure.
    gamma : float
        Unit weight of soil (kN/m^3).
    c : float
        Cohesion (kPa).
    phi_deg : float
        Friction angle (degrees).
    kh : float
        Horizontal pseudo-static seismic coefficient (dimensionless,
        applied as kh * W destabilizing horizontal force per slice).
    max_iter, tol : convergence controls for the fixed-point iteration.

    Returns
    -------
    float
        Factor of safety (dimensionless). Returns np.nan if the
        iteration fails to converge or produces a non-physical result.
    """
    phi = np.radians(phi_deg)
    b = slices.width
    h = slices.height
    alpha = slices.alpha
    u = slices.u

    W = gamma * b * h  # slice weight (kN per m run)

    fos = 1.0  # initial guess
    for _ in range(max_iter):
        m_alpha = np.cos(alpha) + (np.sin(alpha) * np.tan(phi)) / fos

        if np.any(np.abs(m_alpha) < 1e-8):
            return float("nan")

        numerator = c * b + (W - u * b) * np.tan(phi)
        resisting = np.sum(numerator / m_alpha)

        driving = np.sum(W * np.sin(alpha)) + kh * np.sum(W)

        if driving <= 0:
            return float("nan")

        fos_new = resisting / driving

        if not np.isfinite(fos_new) or fos_new <= 0:
            return float("nan")

        if abs(fos_new - fos) < tol:
            return float(fos_new)

        fos = fos_new

    return float("nan")  # did not converge


class SlopeBishopLimitState(LimitState):
    """
    Limit state wrapping Bishop's Simplified FoS for reliability analysis.

    The slip-surface geometry (slice widths/heights/angles) is treated as
    fixed (deterministic) — i.e. the critical surface found under mean
    parameters — while gamma, c, phi, and pore pressure ratio are random.
    This mirrors common practice (fix the geometry, randomize strength/
    pressure inputs) rather than re-searching the critical surface for
    every Monte Carlo realization, which is far more expensive and not
    what most published reliability studies do either.
    """

    def __init__(
        self,
        slices: SliceGeometry,
        gamma_rv: RandomVariable,
        c_rv: RandomVariable,
        phi_rv: RandomVariable,
        kh: float = 0.0,
        ru: float = 0.0,
    ):
        self.slices = slices
        self.kh = kh
        self.ru = ru  # pore pressure ratio, u = ru * gamma * h (kept deterministic here)
        self._rvs = {
            "gamma": gamma_rv,
            "c": c_rv,
            "phi_deg": phi_rv,
        }

    @property
    def random_variables(self) -> dict[str, RandomVariable]:
        return self._rvs

    def evaluate(self, gamma: float, c: float, phi_deg: float) -> float:
        u = self.ru * gamma * self.slices.height
        local_slices = SliceGeometry(
            width=self.slices.width,
            height=self.slices.height,
            alpha=self.slices.alpha,
            u=u,
        )
        fos = bishop_simplified_fos(local_slices, gamma, c, phi_deg, kh=self.kh)
        if np.isnan(fos):
            # Treat non-convergence as failure (conservative) rather than
            # silently dropping the sample, which would bias P(f) low.
            return -1.0
        return fos - 1.0
