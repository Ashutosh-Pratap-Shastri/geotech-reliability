"""
Wall-crushing limit state for a circular lined rock cavern under internal
gas pressure and anisotropic in-situ stress.

Physics
-------
Hoop stress at the cavern wall from the closed-form Kirsch (1898)
solution for a circular opening in a biaxial far-field stress field with
internal pressure. At the wall (r = a), the tangential (hoop) stress
varies with position theta around the opening; the crown/springline
extremes are:

    sigma_theta(0)   = 3*sigma_h - sigma_v - Pi   (springline, theta=0)
    sigma_theta(90)  = 3*sigma_v - sigma_h - Pi   (crown, theta=90)

where sigma_v = gamma * H (vertical far-field stress), sigma_h = k0 * sigma_v.
The governing (maximum compressive) hoop stress is the larger of the two.

Rock mass compressive strength is estimated from intact UCS reduced by
GSI via the simplified Hoek-Brown relation (Hoek, Carranza-Torres &
Corkum, 2002):

    sigma_cm ~= sigma_ci * exp((GSI - 100) / 24)   [D = 0, mb/s-based
                                                      approximation]

This is a standard rough estimate, not a full generalized Hoek-Brown
solve (which also needs mi and disturbance factor D); it is adequate for
a screening-level reliability check but should be replaced with a full
Hoek-Brown parameter fit for design use.

References
----------
Kirsch, G. (1898). "Die Theorie der Elastizitat und die Bedurfnisse der
Festigkeitslehre." Zeitschrift des Vereines Deutscher Ingenieure, 42.

Hoek, E., Carranza-Torres, C., Corkum, B. (2002). "Hoek-Brown failure
criterion - 2002 edition." Proc. NARMS-TAC Conference, Toronto.
"""

from __future__ import annotations

import numpy as np

from geotech_reliability.limit_states.base import LimitState
from geotech_reliability.core.distributions import RandomVariable


def kirsch_max_hoop_stress(H: float, gamma: float, k0: float, Pi: float) -> float:
    """
    Maximum compressive hoop (tangential) stress at the wall of a
    circular opening, taken as the larger of the crown and springline
    values from the Kirsch solution.

    Parameters
    ----------
    H : float
        Depth to crown / representative depth (m).
    gamma : float
        Rock unit weight (kN/m^3).
    k0 : float
        In-situ stress ratio, sigma_h / sigma_v.
    Pi : float
        Internal cavern pressure (MPa). Converted consistently with
        sigma_v/sigma_h, which are computed in MPa (gamma in kN/m^3,
        H in m -> gamma*H in kPa -> /1000 for MPa).

    Returns
    -------
    float
        Maximum compressive hoop stress (MPa). Sign convention:
        compression positive.
    """
    sigma_v = gamma * H / 1000.0  # kPa -> MPa
    sigma_h = k0 * sigma_v

    springline = 3 * sigma_h - sigma_v - Pi
    crown = 3 * sigma_v - sigma_h - Pi

    return float(max(springline, crown))


def hoek_brown_rock_mass_strength(sigma_ci: float, GSI: float) -> float:
    """
    Simplified Hoek-Brown estimate of rock mass compressive strength,
    D = 0 (undisturbed), following Hoek, Carranza-Torres & Corkum (2002).

    Parameters
    ----------
    sigma_ci : float
        Intact rock UCS (MPa).
    GSI : float
        Geological Strength Index, 0-100.

    Returns
    -------
    float
        Estimated rock mass compressive strength sigma_cm (MPa).
    """
    GSI = np.clip(GSI, 1.0, 100.0)
    return float(sigma_ci * np.exp((GSI - 100.0) / 24.0))


class CavernWallCrushingLimitState(LimitState):
    """
    g(x) = sigma_cm(rock mass strength) - sigma_theta(hoop stress)

    g < 0  ->  hoop stress exceeds rock mass strength -> wall crushing.
    """

    def __init__(
        self,
        H_rv: RandomVariable,
        gamma_rv: RandomVariable,
        k0_rv: RandomVariable,
        sigma_ci_rv: RandomVariable,
        GSI_rv: RandomVariable,
        Pi: float,
    ):
        self.Pi = Pi
        self._rvs = {
            "H": H_rv,
            "gamma": gamma_rv,
            "k0": k0_rv,
            "sigma_ci": sigma_ci_rv,
            "GSI": GSI_rv,
        }

    @property
    def random_variables(self) -> dict[str, RandomVariable]:
        return self._rvs

    def evaluate(self, H: float, gamma: float, k0: float, sigma_ci: float, GSI: float) -> float:
        sigma_theta = kirsch_max_hoop_stress(H, gamma, k0, self.Pi)
        sigma_cm = hoek_brown_rock_mass_strength(sigma_ci, GSI)
        return sigma_cm - sigma_theta
