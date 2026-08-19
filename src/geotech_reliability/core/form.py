"""
First-Order Reliability Method (FORM / Hasofer-Lind).

NOT YET IMPLEMENTED. `RandomVariable.to_standard_normal` and
`.from_standard_normal` (see core/distributions.py) already provide the
transforms FORM needs; the HL-RF search for the design point in
standard-normal space is not yet written.

Planned interface (subject to change):

    def run_form(limit_state: LimitState, max_iter: int = 50, tol: float = 1e-4) -> FORMResult

mirroring `run_monte_carlo`'s signature so the two methods are easily
compared on the same limit state.
"""

from geotech_reliability.limit_states.base import LimitState


def run_form(limit_state: "LimitState", max_iter: int = 50, tol: float = 1e-4):
    raise NotImplementedError(
        "FORM is not yet implemented. Use core.monte_carlo.run_monte_carlo instead. "
        "See module docstring for the planned interface."
    )
