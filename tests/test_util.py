import numpy as np
import numpy.testing as npt
import pytest

from pygsm.util import planck_law, smooth, tcmb2trj, trj2tcmb


class TestPlanckLaw:
    def test_positive_output(self):
        result = planck_law(19.6, np.array([100.0, 353.0]))
        assert np.all(result > 0)

    def test_higher_temp_higher_output(self):
        """At fixed frequency, higher temperature gives more emission."""
        low = planck_law(10.0, np.array([353.0]))
        high = planck_law(30.0, np.array([353.0]))
        assert high > low

    def test_scalar_frequency(self):
        result = planck_law(19.6, np.array([100.0]))
        assert result.shape == (1,)


class TestTemperatureConversion:
    def test_trj2tcmb_tcmb2trj_inverse(self):
        """trj2tcmb and tcmb2trj should be exact inverses."""
        freqs = np.array([27.0, 39.0, 93.0, 145.0, 225.0, 280.0, 353.0])
        product = trj2tcmb(freqs) * tcmb2trj(freqs)
        npt.assert_allclose(product, 1.0, rtol=1e-12)

    def test_trj2tcmb_low_freq_limit(self):
        """At low frequencies (x << 1), RJ and CMB temperatures converge."""
        low_freq = np.array([0.1])
        npt.assert_allclose(trj2tcmb(low_freq), 1.0, atol=1e-3)

    def test_tcmb2trj_positive(self):
        freqs = np.array([27.0, 93.0, 353.0])
        assert np.all(tcmb2trj(freqs) > 0)


class TestSmooth:
    def test_output_shape(self):
        nside = 8
        npix = 12 * nside**2
        m = np.random.randn(npix)
        result = smooth(m, fwhm=0.1)
        assert result.shape == m.shape
