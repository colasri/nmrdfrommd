import numpy as np

from nmrdfrommd.relaxation import compute_relaxation_rates
from nmrdfrommd.units import ANGSTROM_TO_M


def test_isotropic_constant_density_gives_equal_R1_R2():
    """In the extreme-narrowing limit (frequency-independent J), R1 == R2."""
    f = np.linspace(0, 10, 50)
    J = np.full((1, len(f)), 2.0)

    R1, R2 = compute_relaxation_rates(f, J, K=1.0, isotropic=True)

    np.testing.assert_allclose(R1, R2)


def test_isotropic_linear_density_matches_closed_form():
    """For exactly linear J0(f), extrapolation is exact, giving a closed
    form to check the formula against."""
    a, b = 3.0, 0.5
    f = np.linspace(0.0, 10.0, 11)
    J = (a + b * f)[np.newaxis, :]
    K = 2.0

    R1, R2 = compute_relaxation_rates(f, J, K, isotropic=True)

    prefactor = K / (1e-10) ** 6  # cst.angstrom
    R1_expected = prefactor * (5 * a + 9 * b * f) / 6
    R2_expected = prefactor * (5 * a + 4.5 * b * f) / 6

    np.testing.assert_allclose(R1, R1_expected, rtol=1e-10)
    np.testing.assert_allclose(R2, R2_expected, rtol=1e-10)


def test_anisotropic_linear_density_matches_closed_form():
    """Same idea, exercising the dim=3 branch."""
    a0, b0 = 1.0, 0.1
    a1, b1 = 2.0, 0.2
    a2, b2 = 3.0, 0.3
    f = np.linspace(0.0, 10.0, 11)
    J = np.stack([a0 + b0 * f, a1 + b1 * f, a2 + b2 * f])
    K = 1.5

    R1, R2 = compute_relaxation_rates(f, J, K, isotropic=False)

    prefactor = K / (1e-10) ** 6
    R1_expected = prefactor * ((a1 + a2) + (b1 + 2 * b2) * f)
    R2_expected = prefactor * (1 / 4) * ((a0 + 10 * a1 + a2) + (10 * b1 + 2 * b2) * f)

    np.testing.assert_allclose(R1, R1_expected, rtol=1e-10)
    np.testing.assert_allclose(R2, R2_expected, rtol=1e-10)


def test_zero_frequency_term_uses_nearest_value_not_first_index():
    """R2's constant term must anchor on the frequency nearest to zero,
    not silently assume f[0] == 0 (regression test for the find_nearest fix)."""
    a, b = 1.0, 2.0
    f = np.array([-2.0, -0.1, 0.05, 3.0])  # nearest to zero is index 2
    J = (a + b * f)[np.newaxis, :]

    _, R2 = compute_relaxation_rates(f, J, K=1.0, isotropic=True)

    prefactor = 1.0 / (1e-10) ** 6
    J0_at_zero = a + b * f[2]
    J02 = a + 2 * b * f
    R2_expected = prefactor * (3 / 2 * J0_at_zero + 5 / 2 * (a + b * f) + J02) / 6

    np.testing.assert_allclose(R2, R2_expected, rtol=1e-10)


def test_isotropic_R1_matches_closed_form_for_lorentzian_density():
    """For a Lorentzian J(f), R1 = (J(f) + 4 J(2f)) / 6 wherever J(2f) is sampled.

    Unlike the linear densities above, a Lorentzian is curved, so linear
    extrapolation would not be exact. Only frequencies up to f_max / 2 are
    compared: above that, J(2f) is outside the sampled range.
    """

    def lorentzian(f, f_c=2.0):
        return 1 / (1 + (f / f_c) ** 2)

    f = np.linspace(0, 10, 1001)
    K = ANGSTROM_TO_M**6  # makes the prefactor of compute_relaxation_rates equal to 1

    R1, _ = compute_relaxation_rates(f, lorentzian(f)[np.newaxis, :], K, isotropic=True)

    expected = (lorentzian(f) + 4 * lorentzian(2 * f)) / 6
    sampled = 2 * f <= f.max()
    np.testing.assert_allclose(R1[sampled], expected[sampled], rtol=1e-3)
