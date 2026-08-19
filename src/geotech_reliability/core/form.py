"""
First-Order Reliability Method (FORM), solved by the Hasofer-Lind /
Rackwitz-Fiessler (HL-RF) iterative algorithm.

FORM approximates the reliability index beta as the distance, in
standard-normal ("u") space, from the origin to the nearest point on the
limit-state surface g(u) = 0 (the "design point"). It uses a first-order
(linear) approximation of g at that point to estimate P(f) = Phi(-beta).

This is far cheaper than Monte Carlo (tens of iterations of gradient
evaluation vs. hundreds of thousands of samples) and gives an *exact*
answer when g is linear in the physical variables and those variables
are normal -- which is also the case we validate against, since it lets
us check the algorithm against a closed-form result rather than only
against Monte Carlo (which has its own sampling error).

Algorithm (HL-RF)
------------------
Starting from u = 0:
  1. Map u -> physical space x via each variable's from_standard_normal.
  2. Evaluate g(x) and its gradient dg/du (via finite differences in
     u-space, chain-ruled through the x = from_standard_normal(u) map).
  3. Move to a new u via the HL-RF update:
         u_new = (1 / |grad_u g|^2) * (grad_u g . u - g(x)) * grad_u g
  4. Repeat until u converges (design point found).
  5. beta = |u*|, P(f) ~= Phi(-beta).

References
----------
Hasofer, A.M. & Lind, N.C. (1974). "Exact and invariant second-moment
code format." Journal of the Engineering Mechanics Division, ASCE.

Rackwitz, R. & Fiessler, B. (1978). "Structural reliability under
combined random load sequences." Computers & Structures, 9(5), 489-494.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from geotech_reliability.limit_states.base import LimitState


@dataclass
class FORMResult:
    beta: float
    pf: float
    design_point_u: dict[str, float]
    design_point_x: dict[str, float]
    converged: bool
    n_iter: int
    history: list[float] = field(default_factory=list)


def run_form(
    limit_state: LimitState,
    max_iter: int = 50,
    tol: float = 1e-4,
    fd_step: float = 1e-4,
) -> FORMResult:
    """
    Run FORM via the HL-RF algorithm.

    Parameters
    ----------
    limit_state : LimitState
        Any object implementing evaluate() and random_variables.
    max_iter : int
        Maximum HL-RF iterations.
    tol : float
        Convergence tolerance on beta between iterations.
    fd_step : float
        Step size (in standard-normal units) for the central-difference
        gradient estimate.

    Returns
    -------
    FORMResult
    """
    rvs = limit_state.random_variables
    names = list(rvs.keys())
    n = len(names)

    if n == 0:
        raise ValueError("limit_state has no random variables")

    def to_x(u: np.ndarray) -> dict[str, float]:
        return {name: rvs[name].from_standard_normal(u[i]) for i, name in enumerate(names)}

    def g_of_u(u: np.ndarray) -> float:
        return limit_state.evaluate(**to_x(u))

    def grad_g_u(u: np.ndarray) -> np.ndarray:
        grad = np.zeros(n)
        for i in range(n):
            u_plus = u.copy()
            u_minus = u.copy()
            u_plus[i] += fd_step
            u_minus[i] -= fd_step
            grad[i] = (g_of_u(u_plus) - g_of_u(u_minus)) / (2 * fd_step)
        return grad

    u = np.zeros(n)
    beta_history: list[float] = []
    converged = False
    n_iter = 0

    # Sign convention: if the mean-value point (u=0) is itself in the
    # failure domain (g(0) < 0), the design point search still finds the
    # nearest point *on* g=0, but beta must be reported as negative
    # (giving pf = Phi(-beta) > 0.5), since more than half the
    # probability mass is on the failure side. Without this correction,
    # pf = Phi(-beta) can never exceed 0.5 regardless of how unsafe the
    # mean-value case actually is.
    g_at_origin = g_of_u(np.zeros(n))
    sign = 1.0 if g_at_origin >= 0 else -1.0

    for it in range(1, max_iter + 1):
        n_iter = it
        g0 = g_of_u(u)
        grad = grad_g_u(u)
        norm_grad_sq = float(np.dot(grad, grad))

        if norm_grad_sq < 1e-14:
            beta_history.append(sign * float(np.linalg.norm(u)))
            break

        u_new = (np.dot(grad, u) - g0) / norm_grad_sq * grad
        beta_new = sign * float(np.linalg.norm(u_new))
        beta_history.append(beta_new)

        if it > 1 and abs(beta_new - beta_history[-2]) < tol:
            u = u_new
            converged = True
            break

        u = u_new

    beta = sign * float(np.linalg.norm(u))
    pf = float(stats.norm.cdf(-beta))

    return FORMResult(
        beta=beta,
        pf=pf,
        design_point_u={name: float(u[i]) for i, name in enumerate(names)},
        design_point_x=to_x(u),
        converged=converged,
        n_iter=n_iter,
        history=beta_history,
    )
