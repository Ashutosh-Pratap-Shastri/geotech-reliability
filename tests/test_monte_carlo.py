"""
Tests for the generic Monte Carlo engine.

Validation strategy
--------------------
Use a trivial linear limit state g = R - S with R, S independent normal
random variables. In that special case P(f) has an exact closed form:

    mu_g = mu_R - mu_S
    sigma_g = sqrt(sigma_R^2 + sigma_S^2)
    P(f) = Phi(-mu_g / sigma_g)
    beta = mu_g / sigma_g

This lets us check the Monte Carlo engine converges to a known-correct
answer without depending on any geotechnical physics.
"""

import numpy as np
import pytest
from scipy import stats

from geotech_reliability.core.distributions import RandomVariable
from geotech_reliability.core.monte_carlo import run_monte_carlo, run_system_monte_carlo
from geotech_reliability.limit_states.base import LimitState


class LinearRminusS(LimitState):
    """g(x) = R - S, both normal. Closed-form P(f) is known exactly."""

    def __init__(self, mean_r, cov_r, mean_s, cov_s):
        self._rvs = {
            "R": RandomVariable(name="R", mean=mean_r, cov=cov_r, dist="normal"),
            "S": RandomVariable(name="S", mean=mean_s, cov=cov_s, dist="normal"),
        }

    @property
    def random_variables(self):
        return self._rvs

    def evaluate(self, R, S):
        return R - S


def closed_form_pf(mean_r, cov_r, mean_s, cov_s):
    sigma_r = mean_r * cov_r
    sigma_s = mean_s * cov_s
    mu_g = mean_r - mean_s
    sigma_g = np.sqrt(sigma_r**2 + sigma_s**2)
    beta = mu_g / sigma_g
    pf = stats.norm.cdf(-beta)
    return pf, beta


def test_monte_carlo_matches_closed_form_linear_case():
    ls = LinearRminusS(mean_r=100, cov_r=0.15, mean_s=70, cov_s=0.20)
    result = run_monte_carlo(ls, n=200_000, seed=123)

    pf_exact, beta_exact = closed_form_pf(100, 0.15, 70, 0.20)

    assert result.pf == pytest.approx(pf_exact, abs=0.01)
    assert result.beta == pytest.approx(beta_exact, abs=0.1)


def test_monte_carlo_reproducible_with_seed():
    ls = LinearRminusS(mean_r=100, cov_r=0.15, mean_s=70, cov_s=0.20)
    r1 = run_monte_carlo(ls, n=10_000, seed=7)
    r2 = run_monte_carlo(ls, n=10_000, seed=7)
    assert r1.pf == r2.pf
    np.testing.assert_array_equal(r1.g, r2.g)


def test_monte_carlo_deterministic_inputs_give_zero_or_one_pf():
    ls = LinearRminusS(mean_r=100, cov_r=0.0, mean_s=70, cov_s=0.0)
    result = run_monte_carlo(ls, n=1000, seed=1)
    assert result.pf == 0.0  # R > S always, deterministically


def test_monte_carlo_rejects_zero_samples():
    ls = LinearRminusS(mean_r=100, cov_r=0.1, mean_s=70, cov_s=0.1)
    with pytest.raises(ValueError):
        run_monte_carlo(ls, n=0)


def test_pf_confidence_interval_contains_point_estimate():
    ls = LinearRminusS(mean_r=100, cov_r=0.15, mean_s=70, cov_s=0.20)
    result = run_monte_carlo(ls, n=50_000, seed=5)
    lo, hi = result.pf_confidence_interval(0.95)
    assert lo <= result.pf <= hi


def test_system_monte_carlo_union_pf_at_least_max_of_individual():
    """P(f, system) for a union of failure modes must be >= max individual P(f)."""
    ls_a = LinearRminusS(mean_r=100, cov_r=0.15, mean_s=70, cov_s=0.20)
    ls_b = LinearRminusS(mean_r=100, cov_r=0.15, mean_s=90, cov_s=0.20)

    result = run_system_monte_carlo({"mode_a": ls_a, "mode_b": ls_b}, n=100_000, seed=9)

    pf_a = result["individual"]["mode_a"].pf
    pf_b = result["individual"]["mode_b"].pf
    pf_system = result["pf_system"]

    assert pf_system >= max(pf_a, pf_b) - 1e-9
    assert pf_system <= pf_a + pf_b + 1e-9  # union bound (Boole's inequality)


def test_system_monte_carlo_independent_by_default_even_with_same_variable_names():
    """
    Two limit states that both happen to use a variable named "S" but
    with DIFFERENT distributions must NOT be silently conflated: without
    declaring "S" as shared, each limit state must get its own draw.
    """
    ls_a = LinearRminusS(mean_r=100, cov_r=0.15, mean_s=70, cov_s=0.20)
    ls_b = LinearRminusS(mean_r=100, cov_r=0.15, mean_s=90, cov_s=0.20)

    result = run_system_monte_carlo({"a": ls_a, "b": ls_b}, n=200_000, seed=9)
    pf_a = result["individual"]["a"].pf
    pf_b = result["individual"]["b"].pf

    # b has higher demand (mean_s=90 vs 70) so must fail strictly more often
    assert pf_b > pf_a


def test_system_monte_carlo_shared_variable_reused_not_resampled():
    ls_a = LinearRminusS(mean_r=100, cov_r=0.15, mean_s=70, cov_s=0.20)
    ls_b = LinearRminusS(mean_r=100, cov_r=0.15, mean_s=90, cov_s=0.20)

    result = run_system_monte_carlo(
        {"a": ls_a, "b": ls_b}, n=1000, seed=9, shared_variable_names=["R"]
    )
    np.testing.assert_array_equal(
        result["individual"]["a"].samples["R"], result["individual"]["b"].samples["R"]
    )


def test_system_monte_carlo_raises_on_inconsistent_shared_definition():
    ls_a = LinearRminusS(mean_r=100, cov_r=0.15, mean_s=70, cov_s=0.20)
    ls_c = LinearRminusS(mean_r=999, cov_r=0.15, mean_s=70, cov_s=0.20)  # different R mean

    with pytest.raises(ValueError):
        run_system_monte_carlo({"a": ls_a, "c": ls_c}, n=1000, seed=1, shared_variable_names=["R"])
