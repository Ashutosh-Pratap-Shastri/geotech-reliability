"""
Example: reliability of a slope against Bishop's Simplified failure,
with uncertain shear strength parameters.

Run with:  python docs/examples/slope_reliability.py
"""

import numpy as np

from geotech_reliability.core.distributions import RandomVariable
from geotech_reliability.core.monte_carlo import run_monte_carlo
from geotech_reliability.limit_states.slope_bishop import SliceGeometry, SlopeBishopLimitState


def main():
    # A representative slip surface: 40 slices across a moderately steep
    # cut, base angle varying from toe to crest.
    n_slices = 40
    slices = SliceGeometry(
        width=np.full(n_slices, 1.0),
        height=np.linspace(1.0, 8.0, n_slices)[::-1] * np.sin(np.linspace(0.2, 1.0, n_slices)) + 2.0,
        alpha=np.linspace(np.radians(-10), np.radians(35), n_slices),
        u=np.zeros(n_slices),  # overwritten via ru inside the limit state
    )

    gamma = RandomVariable("gamma", mean=19.0, cov=0.05, dist="normal")   # kN/m3
    c = RandomVariable("c", mean=3.0, cov=0.35, dist="lognormal")          # kPa (must stay positive)
    phi = RandomVariable("phi_deg", mean=22.0, cov=0.12, dist="normal")    # degrees

    limit_state = SlopeBishopLimitState(
        slices=slices, gamma_rv=gamma, c_rv=c, phi_rv=phi, kh=0.12, ru=0.30
    )

    result = run_monte_carlo(limit_state, n=200_000, seed=1)

    beta_str = f"{result.beta:.3f}" if result.beta is not None else "undefined (pf=0 or 1)"
    print(f"P(f) = {result.pf:.5f}   beta = {beta_str}")
    lo, hi = result.pf_confidence_interval(0.95)
    print(f"95% CI on P(f): [{lo:.5f}, {hi:.5f}]")
    print(f"Mean FoS (over all samples, from g = FoS - 1): {result.mean_g + 1:.3f}")


if __name__ == "__main__":
    main()
