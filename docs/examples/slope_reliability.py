"""
Example: reliability of a slope against Bishop's Simplified failure,
with uncertain shear strength parameters.

Run with:  python docs/examples/slope_reliability.py
"""

import numpy as np

from geotech_reliability.core.distributions import RandomVariable
from geotech_reliability.core.monte_carlo import run_monte_carlo
from geotech_reliability.core.form import run_form
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
    print("Monte Carlo:")
    print(f"  P(f) = {result.pf:.5f}   beta = {beta_str}")
    lo, hi = result.pf_confidence_interval(0.95)
    print(f"  95% CI on P(f): [{lo:.5f}, {hi:.5f}]")
    print(f"  Mean FoS (over all samples, from g = FoS - 1): {result.mean_g + 1:.3f}")

    form_result = run_form(limit_state)
    print("\nFORM (Hasofer-Lind / HL-RF):")
    print(f"  P(f) = {form_result.pf:.5f}   beta = {form_result.beta:.3f}")
    print(f"  converged = {form_result.converged}   iterations = {form_result.n_iter}")
    print(f"  design point (most likely failure combination): {form_result.design_point_x}")
    print(
        "\nNote: FORM linearizes the (nonlinear) Bishop limit state at the "
        "design point, so it will not match Monte Carlo exactly -- some "
        "difference is expected and normal, not an error."
    )


if __name__ == "__main__":
    main()
