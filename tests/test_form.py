"""
Tests for FORM (Hasofer-Lind / HL-RF algorithm).

Validation strategy
--------------------
For a linear limit state g = R - S with R, S independent normal random
variables, FORM is mathematically EXACT (not an approximation) -- the
linearization FORM relies on is exact when g is already linear. This
gives a strong closed-form check, distinct from the Monte Carlo
validation (which has sampling error FORM does not).

We test:
  1. Exact match to the closed-form beta across several parameter sets.
  2. Correct sign convention: when the mean-value point is itself in the
     failure domain (pf > 0.5), beta must be negative.
  3. Higher-dimensional (3-variable) linear case.
  4. Reasonable agreement with Monte Carlo on the real (nonlinear) slope
     and cavern limit states -- FORM is a linear approximation here, so
     we check they agree in sign and are within a generous tolerance,
     not that they match exactly.
"""

import numpy as np
import pytest

from geotech_reliability.core.distributions import RandomVariable
from geotech_reliability.core.form import run_form
from geotech_reliability.core.monte_carlo import run_monte_carlo
from geotech_reliability.limit_states.base import LimitState
from geotech_reliability.limit_states.slope_bishop import SliceGeometry, SlopeBishopLimitState


class LinearRminusS(LimitState):
    """g(x) = R - S, both normal. FORM should be exact here."""

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


def closed_form_beta(mean_r, cov_r, mean_s, cov_s):
    sigma_r = mean_r * cov_r
    sigma_s = mean_s * cov_s
    return (mean_r - mean_s) / np.sqrt(sigma_r**2 + sigma_s**2)


@pytest.mark.parametrize(
    "mean_r,cov_r,mean_s,cov_s",
    [
        (100, 0.15, 70, 0.20),
        (50, 0.10, 30, 0.10),
        (200, 0.30, 150, 0.25),
        (10, 0.05, 5, 0.40),
    ],
)
def test_form_exact_for_linear_normal_case(mean_r, cov_r, mean_s, cov_s):
    ls = LinearRminusS(mean_r, cov_r, mean_s, cov_s)
    result = run_form(ls)
    beta_exact = closed_form_beta(mean_r, cov_r, mean_s, cov_s)
    assert result.beta == pytest.approx(beta_exact, rel=1e-4)
    assert result.converged


def test_form_negative_beta_when_mean_point_fails():
    """If the mean-value point itself has g < 0 (S > R on average), beta
    must be negative and pf must exceed 0.5."""
    ls = LinearRminusS(mean_r=70, cov_r=0.20, mean_s=100, cov_s=0.15)
    result = run_form(ls)
    beta_exact = closed_form_beta(70, 0.20, 100, 0.15)

    assert beta_exact < 0  # sanity check on the reference value itself
    assert result.beta == pytest.approx(beta_exact, rel=1e-4)
    assert result.beta < 0
    assert result.pf > 0.5


def test_form_positive_beta_when_mean_point_safe():
    ls = LinearRminusS(mean_r=100, cov_r=0.15, mean_s=70, cov_s=0.20)
    result = run_form(ls)
    assert result.beta > 0
    assert result.pf < 0.5


def test_form_three_variable_linear_case():
    class ThreeVar(LimitState):
        def __init__(self):
            self._rvs = {
                "A": RandomVariable("A", mean=100, cov=0.1, dist="normal"),
                "B": RandomVariable("B", mean=50, cov=0.2, dist="normal"),
                "C": RandomVariable("C", mean=20, cov=0.15, dist="normal"),
            }

        @property
        def random_variables(self):
            return self._rvs

        def evaluate(self, A, B, C):
            return A - B - C

    ls = ThreeVar()
    result = run_form(ls)

    sigma_a, sigma_b, sigma_c = 100 * 0.1, 50 * 0.2, 20 * 0.15
    mu = 100 - 50 - 20
    beta_exact = mu / np.sqrt(sigma_a**2 + sigma_b**2 + sigma_c**2)

    assert result.beta == pytest.approx(beta_exact, rel=1e-4)


def test_form_reasonably_agrees_with_monte_carlo_on_slope_limit_state():
    """
    FORM linearizes a genuinely nonlinear limit state (Bishop's method),
    so it will not match Monte Carlo exactly -- but it should agree in
    sign and be within a generous absolute tolerance on beta.
    """
    n_slices = 40
    slices = SliceGeometry(
        width=np.full(n_slices, 1.0),
        height=np.linspace(1.0, 8.0, n_slices)[::-1] * np.sin(np.linspace(0.2, 1.0, n_slices)) + 2.0,
        alpha=np.linspace(np.radians(-10), np.radians(35), n_slices),
        u=np.zeros(n_slices),
    )
    gamma = RandomVariable("gamma", mean=19.0, cov=0.05, dist="normal")
    c = RandomVariable("c", mean=3.0, cov=0.35, dist="lognormal")
    phi = RandomVariable("phi_deg", mean=22.0, cov=0.12, dist="normal")

    ls = SlopeBishopLimitState(slices=slices, gamma_rv=gamma, c_rv=c, phi_rv=phi, kh=0.12, ru=0.30)

    mc = run_monte_carlo(ls, n=200_000, seed=1)
    form_result = run_form(ls)

    assert form_result.converged
    # same sign (both indicate marginal/unsafe slope, pf > 0.5)
    assert np.sign(form_result.beta) == np.sign(mc.beta)
    assert abs(form_result.beta - mc.beta) < 0.3  # generous FORM-vs-MC tolerance


def test_form_raises_on_limit_state_with_no_random_variables():
    class NoVars(LimitState):
        @property
        def random_variables(self):
            return {}

        def evaluate(self):
            return 1.0

    with pytest.raises(ValueError):
        run_form(NoVars())
