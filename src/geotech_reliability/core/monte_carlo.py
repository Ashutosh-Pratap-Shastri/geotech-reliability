"""
Monte Carlo reliability analysis.

Works with any object implementing the LimitState interface. The engine
has no knowledge of slopes, caverns, or any specific physics — it just
samples the random variables, evaluates g(x), and estimates P(f) and the
reliability index beta.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from geotech_reliability.limit_states.base import LimitState


@dataclass
class MonteCarloResult:
    pf: float                 # probability of failure, P(g < 0)
    beta: float | None        # reliability index = -Phi^-1(pf)
    n: int                    # number of samples
    g: np.ndarray             # array of g(x) realizations
    samples: dict[str, np.ndarray]  # sampled inputs, name -> array

    @property
    def mean_g(self) -> float:
        return float(np.mean(self.g))

    @property
    def std_g(self) -> float:
        return float(np.std(self.g, ddof=1))

    def pf_confidence_interval(self, confidence: float = 0.95) -> tuple[float, float]:
        """
        Approximate confidence interval on P(f) using the normal
        approximation to the binomial proportion. Valid when n * pf and
        n * (1 - pf) are both reasonably large (rule of thumb >= 5-10).
        """
        z = stats.norm.ppf(0.5 + confidence / 2)
        se = np.sqrt(self.pf * (1 - self.pf) / self.n) if 0 < self.pf < 1 else 0.0
        lo = max(0.0, self.pf - z * se)
        hi = min(1.0, self.pf + z * se)
        return lo, hi


def run_monte_carlo(
    limit_state: LimitState,
    n: int = 100_000,
    seed: int | None = None,
) -> MonteCarloResult:
    """
    Run a crude (direct-sampling) Monte Carlo reliability analysis.

    Parameters
    ----------
    limit_state : LimitState
        Any object implementing evaluate() and random_variables.
    n : int
        Number of samples to draw.
    seed : int, optional
        Seed for reproducibility.

    Returns
    -------
    MonteCarloResult
    """
    if n < 1:
        raise ValueError("n must be >= 1")

    rng = np.random.default_rng(seed)
    rvs = limit_state.random_variables

    samples = {name: rv.sample(n, rng) for name, rv in rvs.items()}

    g = np.empty(n)
    for i in range(n):
        kwargs = {name: samples[name][i] for name in samples}
        g[i] = limit_state.evaluate(**kwargs)

    n_fail = int(np.sum(g < 0))
    pf = n_fail / n

    if pf == 0:
        beta = None  # no failures observed; beta is unbounded (report as None, not inf)
    elif pf == 1:
        beta = None
    else:
        beta = float(-stats.norm.ppf(pf))

    return MonteCarloResult(pf=pf, beta=beta, n=n, g=g, samples=samples)


def run_system_monte_carlo(
    limit_states: dict[str, LimitState],
    n: int = 100_000,
    seed: int | None = None,
    shared_variable_names: list[str] | None = None,
) -> dict:
    """
    Run Monte Carlo for multiple limit states and compute the system
    (union) probability of failure: P(f, system) = P(any limit state
    fails).

    Parameters
    ----------
    limit_states : dict[str, LimitState]
        Name -> limit state, e.g. {"wall_crushing": ..., "hydraulic_jacking": ...}
    n : int
        Number of samples.
    seed : int, optional
        Seed for reproducibility.
    shared_variable_names : list[str], optional
        Variable names that represent the *same physical quantity* across
        multiple limit states (e.g. "H" appearing in both a wall-crushing
        and a hydraulic-jacking limit state should be drawn once and
        reused, not resampled independently). Sharing is opt-in and
        explicit: by default, variables are sampled independently per
        limit state even if their names happen to coincide, to avoid
        silently conflating two different quantities that share a name
        (e.g. "c" meaning cohesion in one limit state and something
        else in another). If two limit states share a name in
        `shared_variable_names` but define it with different
        RandomVariable parameters, a ValueError is raised rather than
        silently picking one.

    Returns
    -------
    dict with keys:
        "individual": dict[name -> MonteCarloResult]
        "pf_system": float
        "beta_system": float | None
    """
    rng = np.random.default_rng(seed)
    shared_names = set(shared_variable_names or [])

    # Sample explicitly shared variables once, validating consistency.
    shared_rv_defs: dict[str, object] = {}
    for ls in limit_states.values():
        for name, rv in ls.random_variables.items():
            if name not in shared_names:
                continue
            if name in shared_rv_defs:
                prev = shared_rv_defs[name]
                if (prev.mean, prev.cov, prev.dist, prev.bounds) != (rv.mean, rv.cov, rv.dist, rv.bounds):
                    raise ValueError(
                        f"Variable '{name}' listed as shared but defined "
                        f"inconsistently across limit states "
                        f"({prev} vs {rv})."
                    )
            shared_rv_defs[name] = rv

    shared_samples = {name: rv.sample(n, rng) for name, rv in shared_rv_defs.items()}

    individual: dict[str, MonteCarloResult] = {}
    any_fail = np.zeros(n, dtype=bool)

    for label, ls in limit_states.items():
        # Non-shared variables get their own independent draw stream,
        # derived deterministically from the master rng so the whole
        # run stays reproducible under a single seed.
        local_samples = {}
        for name, rv in ls.random_variables.items():
            if name in shared_names:
                local_samples[name] = shared_samples[name]
            else:
                local_samples[name] = rv.sample(n, rng)

        g = np.empty(n)
        for i in range(n):
            kwargs = {name: local_samples[name][i] for name in local_samples}
            g[i] = ls.evaluate(**kwargs)

        fail = g < 0
        any_fail |= fail
        pf = float(np.mean(fail))
        beta = None if pf in (0, 1) else float(-stats.norm.ppf(pf))
        individual[label] = MonteCarloResult(pf=pf, beta=beta, n=n, g=g, samples=local_samples)

    pf_system = float(np.mean(any_fail))
    beta_system = None if pf_system in (0, 1) else float(-stats.norm.ppf(pf_system))

    return {
        "individual": individual,
        "pf_system": pf_system,
        "beta_system": beta_system,
    }
