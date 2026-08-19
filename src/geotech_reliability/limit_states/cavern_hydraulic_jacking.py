"""
Hydraulic-jacking limit state for a lined rock cavern under internal gas
pressure.

Physics
-------
Classical minimum-principal-stress cover criterion: to avoid hydraulic
jacking (tensile fracturing / gas leakage along the path of least
resistance), the minimum in-situ principal stress must exceed the
internal pressure by a specified safety margin F (typically 1.0-1.5 for
underground gas storage):

    sigma_h,min >= F * Pi

Taking sigma_h = k0 * sigma_v as the (assumed) minimum horizontal
principal stress and sigma_v = gamma * H:

    g(x) = k0 * gamma * H / 1000  -  F * Pi        (all stresses in MPa)

References
----------
Classical minimum-principal-stress / hydraulic-fracturing cover
criterion used widely in underground gas and compressed-air storage
design guidance (e.g. Bergman & Bergman-style cover checks; also see
general underground gas storage design literature).
"""

from __future__ import annotations

from geotech_reliability.limit_states.base import LimitState
from geotech_reliability.core.distributions import RandomVariable


def min_principal_stress_margin(H: float, gamma: float, k0: float, Pi: float, F: float) -> float:
    """
    g(x) = sigma_h,min - F * Pi, in MPa.

    Parameters
    ----------
    H : float
        Depth to crown / representative depth (m).
    gamma : float
        Rock unit weight (kN/m^3).
    k0 : float
        In-situ stress ratio, sigma_h / sigma_v (minimum horizontal
        stress assumed governed by k0 here; use the lesser of k0 and 1/k0
        upstream if the maximum k0 direction is not the relevant one).
    Pi : float
        Internal cavern pressure (MPa).
    F : float
        Required hydraulic-jacking safety factor (deterministic, per
        design guidance; typically 1.0-1.5).

    Returns
    -------
    float
        g(x) in MPa. g < 0 -> hydraulic jacking failure.
    """
    sigma_v = gamma * H / 1000.0  # kPa -> MPa
    sigma_h_min = k0 * sigma_v
    return float(sigma_h_min - F * Pi)


class CavernHydraulicJackingLimitState(LimitState):
    """g(x) = sigma_h,min - F * Pi. g < 0 -> hydraulic jacking failure."""

    def __init__(
        self,
        H_rv: RandomVariable,
        gamma_rv: RandomVariable,
        k0_rv: RandomVariable,
        Pi: float,
        F: float = 1.2,
    ):
        self.Pi = Pi
        self.F = F
        self._rvs = {
            "H": H_rv,
            "gamma": gamma_rv,
            "k0": k0_rv,
        }

    @property
    def random_variables(self) -> dict[str, RandomVariable]:
        return self._rvs

    def evaluate(self, H: float, gamma: float, k0: float) -> float:
        return min_principal_stress_margin(H, gamma, k0, self.Pi, self.F)
