"""
Sensitivity analysis (parametric sweeps, Sobol indices).

NOT YET IMPLEMENTED. Planned: a `parametric_sweep(limit_state, variable_name,
values)` helper that re-runs Monte Carlo while holding one variable's mean
fixed at each sweep value, and a Sobol-index estimator built on top of the
existing `RandomVariable` sampling.
"""

from geotech_reliability.limit_states.base import LimitState


def parametric_sweep(limit_state: "LimitState", variable_name: str, values: list[float], **kwargs):
    raise NotImplementedError(
        "Parametric sensitivity sweeps are not yet implemented. "
        "See module docstring for the planned interface."
    )
