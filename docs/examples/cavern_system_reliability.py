"""
Example: system reliability of a lined rock cavern under two competing
failure modes (wall crushing, hydraulic jacking), sharing common
geometry/stress random variables.

Run with:  python docs/examples/cavern_system_reliability.py
"""

from geotech_reliability.core.distributions import RandomVariable
from geotech_reliability.core.monte_carlo import run_system_monte_carlo
from geotech_reliability.limit_states.cavern_wall_crushing import CavernWallCrushingLimitState
from geotech_reliability.limit_states.cavern_hydraulic_jacking import CavernHydraulicJackingLimitState


def main():
    # Deep hard-rock cavern, moderate cyclic H2 storage pressure.
    H = RandomVariable("H", mean=800, cov=0.05, dist="normal")           # depth to crown, m
    gamma = RandomVariable("gamma", mean=27, cov=0.05, dist="normal")    # rock unit weight, kN/m3
    k0 = RandomVariable("k0", mean=1.1, cov=0.15, dist="normal")         # in-situ stress ratio
    sigma_ci = RandomVariable("sigma_ci", mean=120, cov=0.25, dist="lognormal")  # intact UCS, MPa
    GSI = RandomVariable("GSI", mean=65, cov=0.15, dist="normal")        # geological strength index

    Pi = 15.0  # MPa, max cyclic internal pressure
    F = 1.2    # required hydraulic-jacking safety factor

    wall_crushing = CavernWallCrushingLimitState(H, gamma, k0, sigma_ci, GSI, Pi=Pi)
    hydraulic_jacking = CavernHydraulicJackingLimitState(H, gamma, k0, Pi=Pi, F=F)

    result = run_system_monte_carlo(
        {"wall_crushing": wall_crushing, "hydraulic_jacking": hydraulic_jacking},
        n=500_000,
        seed=42,
        # H, gamma, k0 are the *same* physical quantities in both limit
        # states and must be drawn once, not resampled independently.
        shared_variable_names=["H", "gamma", "k0"],
    )

    print("Limit state reliability:")
    for label, r in result["individual"].items():
        beta_str = f"{r.beta:.3f}" if r.beta is not None else "undefined (pf=0 or 1)"
        print(f"  {label:20s} P(f) = {r.pf:.5f}   beta = {beta_str}")

    beta_sys = result["beta_system"]
    beta_sys_str = f"{beta_sys:.3f}" if beta_sys is not None else "undefined"
    print(f"\nSystem (union): P(f) = {result['pf_system']:.5f}   beta = {beta_sys_str}")


if __name__ == "__main__":
    main()
