"""Points to check: behaviours that differ from what we would expect.

Each test describes the behaviour we would expect, what we observe
instead, and a possible explanation. They may reveal bugs, or simply
choices we have misunderstood: they are meant as a starting point for a
discussion, not as a verdict.

The tests are marked ``xfail(strict=True, raises=...)``, so they do not
break the test suite while the point is open. If the code changes so that
a test passes, pytest reports it, and the marker can then be removed so
the test keeps guarding that behaviour. If the current behaviour turns out
to be intended, the test should simply be removed.

To see what each test observes, run:

    pytest tests/test_points_to_check.py --runxfail
"""

import numpy as np
import pytest

from nmrdfrommd import NMRD
from nmrdfrommd.correlation import compute_correlation_time
from nmrdfrommd.fourier import compute_spectral_density
from nmrdfrommd.units import HZ_TO_MHZ, PS_TO_S


def to_check(reason, raises=AssertionError):
    """Mark a test whose expectation is currently not met (by raising `raises`)."""
    return pytest.mark.xfail(strict=True, raises=raises, reason=reason)


# --- Correlation time -----------------------------------------------------


@to_check("possibly reads gij[0][m] (component 0, lag m) instead of gij[m][0]")
def test_correlation_time_of_each_component():
    """We expect tau_m = J_m(0) / (2 G_m(0)), computed from component m.

    Here all three components are built with tau = 1 ps, and we would
    expect 1 ps for each. We observe 1, 4 and 16 ps. A possible
    explanation is that J_m(0) is divided by gij[0][m] (the m-th lag of
    component 0) rather than by gij[m][0].
    """
    gij = np.array([
        [1.0, 0.5, 0.25],
        [2.0, 1.0, 0.50],
        [4.0, 2.0, 1.00],
    ])  # fmt: skip
    # J_m(0) = 2 tau G_m(0) with tau = 1 ps, converted to s.
    J = PS_TO_S * np.array([
        [2.0, 0.0, 0.0],
        [4.0, 0.0, 0.0],
        [8.0, 0.0, 0.0],
    ])  # fmt: skip

    tau_ps = compute_correlation_time(J, gij, dim=3)

    np.testing.assert_allclose(tau_ps, [1.0, 1.0, 1.0])


@to_check("the integral, already in ps, may be converted from seconds again")
def test_correlation_time_from_the_integral():
    """For G(t) = exp(-t / tau), we expect the integral of G(t) / G(0) to be tau.

    The docstring says the result is in ps. With tau = 2 ps, we observe
    2e12 rather than 2. A possible explanation is that the integral over t
    (given in ps) is already in ps, and is then divided by 1e-12 as if it
    were in seconds.
    """
    t_ps = np.linspace(0, 100, 10001)
    gij = np.exp(-t_ps / 2.0)

    tau_ps = compute_correlation_time(
        None, gij, dim=1, integral=True, t=t_ps, oneDarray=True
    )

    np.testing.assert_allclose(tau_ps, [2.0], rtol=1e-3)


# --- Trajectory handling --------------------------------------------------


@to_check("box angles seem to be read even when pbc=False", raises=TypeError)
def test_trajectory_without_box_when_pbc_is_off(make_rigid_pair):
    """We expect a trajectory without box information to work when pbc=False.

    Without periodic boundary conditions the box should not be needed, and
    some trajectories (e.g. of isolated molecules) have no box at all. We
    observe a TypeError: the box angles appear to be checked at every
    frame, whatever the value of pbc.
    """
    u = make_rigid_pair(box=None)

    NMRD(u, u.atoms, pbc=False).run_analysis()


# --- Spectral density and relaxation rates --------------------------------


@to_check("rates are reported up to f_max, where J(2f) has to be extrapolated")
def test_rates_are_only_reported_where_J_2f_is_sampled(rigid_dipole_universe):
    """We expect f, R1 and R2 to stop at half the highest sampled frequency.

    R1(f) needs J(2f), but J is only sampled up to the Nyquist frequency
    f_max = 1 / (2 dt). For f > f_max / 2, J(2f) is therefore not known,
    and it is currently extrapolated linearly. For a curved J this can
    give inaccurate values: with a Lorentzian J, R1 even becomes negative.
    We observe that f, R1 and R2 are reported up to f_max. Not reporting
    the upper half is one option; another could be to flag it, or to
    document the limitation. The values below f_max / 2 are checked in
    test_relaxation.py.
    """
    u, group, _ = rigid_dipole_universe

    results = NMRD(u, group).run_analysis()

    f_max = 1 / (2 * u.trajectory.dt * PS_TO_S) * HZ_TO_MHZ
    assert results["f"].max() <= f_max / 2
    assert len(results["R1"]) == len(results["f"])
    assert len(results["R2"]) == len(results["f"])


@to_check("G(0) may get a full weight instead of half in the discrete transform")
def test_spectral_density_of_an_exponential_at_zero_frequency():
    """For G(t) = exp(-t / tau), we expect J(0) = 2 tau.

    J(f) = 2 * integral from 0 to infinity of G(t) cos(2 pi f t) dt. With
    dt = tau / 10, we observe J(0) = 2.1 tau, i.e. 5 % above the
    expected value. A possible explanation is that the integral is
    approximated by dt * sum(G), which gives G(0) a full weight rather
    than half (trapezoid rule) and adds G(0) dt to J at all frequencies.
    The effect is small when dt is much shorter than the correlation
    time, but the relative difference grows at higher frequencies, where
    J is small.
    """
    tau_ps, dt_ps = 1.0, 0.1
    t = np.arange(0, 50 * tau_ps, dt_ps)  # G(t_max) = exp(-50): no truncation error
    G = np.exp(-t / tau_ps)

    _, J = compute_spectral_density(t, G[np.newaxis, :], dim=1)

    np.testing.assert_allclose(J[0, 0], 2 * tau_ps * PS_TO_S, rtol=1e-3)
