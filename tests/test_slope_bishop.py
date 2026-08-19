"""
Tests for Bishop's Simplified method.

Validation strategy
--------------------
Bishop's Simplified method should reduce to the classical infinite-slope
factor of safety when the slip surface is shallow, planar, and parallel
to the ground surface (single "slice" spanning a long uniform slope,
alpha constant, no arc curvature effects). That closed-form solution is:

    FoS = c / (gamma * h * cos(beta) * sin(beta)) + tan(phi) / tan(beta)

(dry slope, no pore pressure, no seismic load). This is a standard
textbook cross-check (e.g. Duncan, Wright & Brandon, "Soil Strength and
Slope Stability") and lets us verify the iterative solver converges to
the right answer without requiring a full published slice table.
"""

import numpy as np
import pytest

from geotech_reliability.limit_states.slope_bishop import (
    SliceGeometry,
    bishop_simplified_fos,
)


def infinite_slope_fos(gamma, c, phi_deg, h, beta_deg):
    beta = np.radians(beta_deg)
    phi = np.radians(phi_deg)
    return c / (gamma * h * np.cos(beta) * np.sin(beta)) + np.tan(phi) / np.tan(beta)


@pytest.mark.parametrize(
    "gamma,c,phi_deg,h,beta_deg",
    [
        (18.0, 5.0, 30.0, 5.0, 25.0),
        (19.0, 0.0, 35.0, 3.0, 30.0),   # cohesionless
        (20.0, 15.0, 20.0, 8.0, 20.0),  # cohesive, gentle slope
    ],
)
def test_bishop_matches_infinite_slope_closed_form(gamma, c, phi_deg, h, beta_deg):
    """
    Approximate an infinite slope as many identical narrow slices at a
    constant base angle equal to the slope angle, with no arc curvature
    (all alpha_i equal). Bishop's method should then converge to the
    closed-form infinite-slope FoS.
    """
    n_slices = 200
    slices = SliceGeometry(
        width=np.full(n_slices, 0.5),
        height=np.full(n_slices, h),
        alpha=np.full(n_slices, np.radians(beta_deg)),
        u=np.zeros(n_slices),
    )

    fos_bishop = bishop_simplified_fos(slices, gamma=gamma, c=c, phi_deg=phi_deg)
    fos_closed_form = infinite_slope_fos(gamma, c, phi_deg, h, beta_deg)

    assert fos_bishop == pytest.approx(fos_closed_form, rel=1e-3)


def test_bishop_fos_decreases_with_seismic_coefficient():
    """Adding horizontal pseudo-static load must reduce FoS, all else equal."""
    n_slices = 50
    slices = SliceGeometry(
        width=np.full(n_slices, 1.0),
        height=np.full(n_slices, 4.0),
        alpha=np.linspace(np.radians(5), np.radians(35), n_slices),
        u=np.zeros(n_slices),
    )
    fos_static = bishop_simplified_fos(slices, gamma=18.0, c=5.0, phi_deg=28.0, kh=0.0)
    fos_seismic = bishop_simplified_fos(slices, gamma=18.0, c=5.0, phi_deg=28.0, kh=0.15)

    assert fos_seismic < fos_static


def test_bishop_fos_increases_with_cohesion():
    n_slices = 50
    slices = SliceGeometry(
        width=np.full(n_slices, 1.0),
        height=np.full(n_slices, 4.0),
        alpha=np.linspace(np.radians(5), np.radians(35), n_slices),
        u=np.zeros(n_slices),
    )
    fos_low_c = bishop_simplified_fos(slices, gamma=18.0, c=2.0, phi_deg=28.0)
    fos_high_c = bishop_simplified_fos(slices, gamma=18.0, c=20.0, phi_deg=28.0)

    assert fos_high_c > fos_low_c


def test_bishop_fos_increases_with_friction_angle():
    n_slices = 50
    slices = SliceGeometry(
        width=np.full(n_slices, 1.0),
        height=np.full(n_slices, 4.0),
        alpha=np.linspace(np.radians(5), np.radians(35), n_slices),
        u=np.zeros(n_slices),
    )
    fos_low_phi = bishop_simplified_fos(slices, gamma=18.0, c=5.0, phi_deg=15.0)
    fos_high_phi = bishop_simplified_fos(slices, gamma=18.0, c=5.0, phi_deg=35.0)

    assert fos_high_phi > fos_low_phi


def test_bishop_pore_pressure_reduces_fos():
    n_slices = 50
    height = np.full(n_slices, 4.0)
    alpha = np.linspace(np.radians(5), np.radians(35), n_slices)
    width = np.full(n_slices, 1.0)

    dry = SliceGeometry(width=width, height=height, alpha=alpha, u=np.zeros(n_slices))
    wet = SliceGeometry(width=width, height=height, alpha=alpha, u=np.full(n_slices, 20.0))

    fos_dry = bishop_simplified_fos(dry, gamma=18.0, c=5.0, phi_deg=28.0)
    fos_wet = bishop_simplified_fos(wet, gamma=18.0, c=5.0, phi_deg=28.0)

    assert fos_wet < fos_dry


def test_bishop_returns_nan_on_zero_driving_force():
    """A flat surface (alpha = 0 everywhere) has no driving shear force."""
    n_slices = 10
    slices = SliceGeometry(
        width=np.full(n_slices, 1.0),
        height=np.full(n_slices, 4.0),
        alpha=np.zeros(n_slices),
        u=np.zeros(n_slices),
    )
    fos = bishop_simplified_fos(slices, gamma=18.0, c=5.0, phi_deg=28.0)
    assert np.isnan(fos)
