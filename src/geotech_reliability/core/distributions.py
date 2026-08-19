"""
Random variable definitions for reliability analysis.

Wraps scipy.stats so that geotechnical parameters can be specified the
way engineers naturally think about them: mean and coefficient of
variation (COV), plus a distribution family.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy import stats

DistName = Literal["normal", "lognormal", "uniform", "deterministic"]


@dataclass
class RandomVariable:
    """
    A single random input to a limit-state function.

    Parameters
    ----------
    name : str
        Identifier matching the keyword argument expected by the
        limit-state function.
    mean : float
        Mean value, in the variable's natural (physical) units.
    cov : float
        Coefficient of variation (std / mean), as a fraction (0.1 = 10%).
        Ignored if dist == "deterministic".
    dist : {"normal", "lognormal", "uniform", "deterministic"}
        Distribution family.
    bounds : tuple[float, float], optional
        Only used for "uniform"; (low, high). If omitted for uniform,
        derived from mean and cov assuming a symmetric range.
    """

    name: str
    mean: float
    cov: float = 0.0
    dist: DistName = "normal"
    bounds: tuple[float, float] | None = None

    def __post_init__(self):
        if self.dist != "deterministic" and self.cov < 0:
            raise ValueError(f"{self.name}: cov must be >= 0, got {self.cov}")
        if self.dist not in ("normal", "lognormal", "uniform", "deterministic"):
            raise ValueError(f"{self.name}: unsupported distribution '{self.dist}'")

    @property
    def std(self) -> float:
        return self.mean * self.cov

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw n samples from this variable's distribution."""
        if self.dist == "deterministic" or self.cov == 0:
            return np.full(n, self.mean)

        if self.dist == "normal":
            return rng.normal(loc=self.mean, scale=self.std, size=n)

        if self.dist == "lognormal":
            # Match mean and std of the underlying normal (in log-space)
            # to the specified physical mean/std of the lognormal variable.
            sigma2 = np.log(1.0 + self.cov**2)
            sigma = np.sqrt(sigma2)
            mu = np.log(self.mean) - 0.5 * sigma2
            return rng.lognormal(mean=mu, sigma=sigma, size=n)

        if self.dist == "uniform":
            if self.bounds is not None:
                low, high = self.bounds
            else:
                half_width = self.std * np.sqrt(3)  # match variance of uniform
                low, high = self.mean - half_width, self.mean + half_width
            return rng.uniform(low, high, size=n)

        raise AssertionError("unreachable")

    def to_standard_normal(self, x: float) -> float:
        """
        Transform a physical-space value x to standard normal space u,
        used by FORM. Only exact for normal/lognormal; for uniform this
        uses a Nataf-style approximation via the CDF.
        """
        if self.dist == "deterministic" or self.cov == 0:
            return 0.0

        if self.dist == "normal":
            return (x - self.mean) / self.std

        if self.dist == "lognormal":
            sigma2 = np.log(1.0 + self.cov**2)
            sigma = np.sqrt(sigma2)
            mu = np.log(self.mean) - 0.5 * sigma2
            return (np.log(x) - mu) / sigma

        if self.dist == "uniform":
            if self.bounds is not None:
                low, high = self.bounds
            else:
                half_width = self.std * np.sqrt(3)
                low, high = self.mean - half_width, self.mean + half_width
            p = np.clip((x - low) / (high - low), 1e-12, 1 - 1e-12)
            return stats.norm.ppf(p)

        raise AssertionError("unreachable")

    def from_standard_normal(self, u: float) -> float:
        """Inverse of to_standard_normal: map u back to physical space x."""
        if self.dist == "deterministic" or self.cov == 0:
            return self.mean

        if self.dist == "normal":
            return self.mean + u * self.std

        if self.dist == "lognormal":
            sigma2 = np.log(1.0 + self.cov**2)
            sigma = np.sqrt(sigma2)
            mu = np.log(self.mean) - 0.5 * sigma2
            return float(np.exp(mu + u * sigma))

        if self.dist == "uniform":
            if self.bounds is not None:
                low, high = self.bounds
            else:
                half_width = self.std * np.sqrt(3)
                low, high = self.mean - half_width, self.mean + half_width
            p = stats.norm.cdf(u)
            return low + p * (high - low)

        raise AssertionError("unreachable")
