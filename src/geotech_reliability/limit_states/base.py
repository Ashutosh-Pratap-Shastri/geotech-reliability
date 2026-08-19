"""
Abstract interface every limit state must implement.

A limit state exposes:
  1. `random_variables` — the input distributions it depends on
  2. `evaluate(**kwargs)` — g(x) = capacity - demand, with g < 0 = failure

The reliability engine (Monte Carlo, FORM) only ever talks to this
interface, so new physical problems can be added without touching the
engine code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from geotech_reliability.core.distributions import RandomVariable


class LimitState(ABC):
    @property
    @abstractmethod
    def random_variables(self) -> dict[str, RandomVariable]:
        """Mapping of variable name -> RandomVariable definition."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, **kwargs: float) -> float:
        """
        Evaluate g(x) = capacity - demand for one realization of the
        random variables (passed as keyword args matching the names in
        `random_variables`).

        Returns
        -------
        float
            g < 0  -> failure realization
            g >= 0 -> safe realization
        """
        raise NotImplementedError

    def variable_names(self) -> list[str]:
        return list(self.random_variables.keys())
