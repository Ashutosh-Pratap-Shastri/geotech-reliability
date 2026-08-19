import numpy as np
import pytest

from geotech_reliability.core.distributions import RandomVariable


def test_deterministic_returns_constant():
    rv = RandomVariable(name="x", mean=5.0, dist="deterministic")
    samples = rv.sample(1000, np.random.default_rng(0))
    assert np.all(samples == 5.0)


def test_zero_cov_returns_constant_regardless_of_dist():
    rv = RandomVariable(name="x", mean=5.0, cov=0.0, dist="normal")
    samples = rv.sample(1000, np.random.default_rng(0))
    assert np.all(samples == 5.0)


def test_normal_sample_statistics():
    rv = RandomVariable(name="x", mean=10.0, cov=0.2, dist="normal")
    samples = rv.sample(500_000, np.random.default_rng(42))
    assert samples.mean() == pytest.approx(10.0, rel=0.01)
    assert samples.std() == pytest.approx(2.0, rel=0.02)


def test_lognormal_sample_statistics_match_physical_mean_and_cov():
    rv = RandomVariable(name="x", mean=30.0, cov=0.3, dist="lognormal")
    samples = rv.sample(500_000, np.random.default_rng(1))
    assert samples.mean() == pytest.approx(30.0, rel=0.02)
    assert (samples.std() / samples.mean()) == pytest.approx(0.3, rel=0.05)
    assert np.all(samples > 0)  # lognormal must never go negative


def test_uniform_sample_bounds():
    rv = RandomVariable(name="x", mean=10.0, dist="uniform", bounds=(8.0, 12.0))
    samples = rv.sample(10_000, np.random.default_rng(2))
    assert samples.min() >= 8.0
    assert samples.max() <= 12.0


def test_negative_cov_raises():
    with pytest.raises(ValueError):
        RandomVariable(name="x", mean=5.0, cov=-0.1, dist="normal")


def test_invalid_distribution_raises():
    with pytest.raises(ValueError):
        RandomVariable(name="x", mean=5.0, cov=0.1, dist="triangular")


def test_standard_normal_round_trip_normal():
    rv = RandomVariable(name="x", mean=20.0, cov=0.25, dist="normal")
    x = 23.5
    u = rv.to_standard_normal(x)
    x_back = rv.from_standard_normal(u)
    assert x_back == pytest.approx(x, rel=1e-8)


def test_standard_normal_round_trip_lognormal():
    rv = RandomVariable(name="x", mean=40.0, cov=0.3, dist="lognormal")
    x = 45.0
    u = rv.to_standard_normal(x)
    x_back = rv.from_standard_normal(u)
    assert x_back == pytest.approx(x, rel=1e-6)
