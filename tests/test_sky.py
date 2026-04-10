import numpy as np
import numpy.testing as npt
import pytest

import pygsm
from pygsm.util import planck_law, tcmb2trj, trj2tcmb

NSIDE = 64
NPIX = 12 * NSIDE**2
LMAX = NSIDE * 3 - 1
NU = np.array([27.0, 93.0, 145.0, 225.0, 353.0])


@pytest.fixture
def sky():
    return pygsm.Sky(nside=NSIDE, nu=NU)


# ------------------------------------------------------------------
# Constructor
# ------------------------------------------------------------------


class TestConstructor:
    def test_basic_attributes(self, sky):
        assert sky.nside == NSIDE
        assert sky.lmax == LMAX
        assert sky.ell.shape == (LMAX + 1,)

    def test_cl2dl_dl2cl_inverse(self, sky):
        """cl2dl and dl2cl should be inverses for ell >= 2."""
        product = sky.cl2dl[2:] * sky.dl2cl[2:]
        npt.assert_allclose(product, 1.0, rtol=1e-12)

    def test_dl2cl_monopole_dipole_zero(self, sky):
        assert sky.dl2cl[0] == 0.0
        assert sky.dl2cl[1] == 0.0


# ------------------------------------------------------------------
# Init guards
# ------------------------------------------------------------------


class TestInitGuards:
    def test_cmb_cls_guard(self, sky):
        with pytest.raises(RuntimeError, match="CMB not initialized"):
            sky.get_cmb_theory_cls()

    def test_cmb_maps_guard(self, sky):
        with pytest.raises(RuntimeError, match="CMB not initialized"):
            sky.get_cmb_maps()

    def test_dust_cls_guard(self, sky):
        with pytest.raises(RuntimeError, match="Dust not initialized"):
            sky.get_dust_theory_cls()

    def test_dust_maps_guard(self, sky):
        with pytest.raises(RuntimeError, match="Dust not initialized"):
            sky.get_dust_maps()

    def test_sync_cls_guard(self, sky):
        with pytest.raises(RuntimeError, match="Synchrotron not initialized"):
            sky.get_sync_theory_cls()

    def test_sync_maps_guard(self, sky):
        with pytest.raises(RuntimeError, match="Synchrotron not initialized"):
            sky.get_sync_maps()

    def test_noise_guard(self, sky):
        with pytest.raises(RuntimeError, match="White noise not initialized"):
            sky.get_white_noise_maps()


# ------------------------------------------------------------------
# Frequency resolution
# ------------------------------------------------------------------


class TestFrequencyResolution:
    def test_no_frequency_raises(self):
        sky = pygsm.Sky(nside=NSIDE)
        sky.init_dust()
        with pytest.raises(ValueError, match="No frequencies provided"):
            sky.get_dust_theory_cls()

    def test_override_frequency(self, sky):
        sky.init_dust()
        nu_override = np.array([100.0, 200.0])
        cls = sky.get_dust_theory_cls(nu=nu_override)
        assert cls.shape[0] == 2

    def test_set_freqs(self):
        sky = pygsm.Sky(nside=NSIDE)
        sky.init_dust()
        sky.set_freqs(np.array([150.0]))
        cls = sky.get_dust_theory_cls()
        assert cls.shape[0] == 1


# ------------------------------------------------------------------
# CMB power spectrum (deterministic)
# ------------------------------------------------------------------


class TestCMBSpectrum:
    def test_cls_shape(self, sky):
        sky.init_cmb()
        cls = sky.get_cmb_theory_cls()
        assert cls.shape == (4, LMAX + 1)

    def test_cls_monopole_dipole_zero(self, sky):
        """Cl at ell=0,1 should be zero due to dl2cl zeroing."""
        sky.init_cmb()
        cls = sky.get_cmb_theory_cls()
        npt.assert_array_equal(cls[:, 0], 0.0)
        npt.assert_array_equal(cls[:, 1], 0.0)

    def test_no_tensor_bb_suppressed(self, sky):
        """With r=0 and A_lens=1, BB should be lensing-only (small)."""
        sky.init_cmb(A_lens=1.0, r_tensor=0.0)
        cls = sky.get_cmb_theory_cls()
        # BB (index 2) should be much smaller than EE (index 1)
        assert np.max(cls[2, 2:]) < np.max(cls[1, 2:])

    def test_no_lensing_bb_zero(self, sky):
        """With A_lens=0 and r=0, BB should be zero (no lensing, no tensor)."""
        sky.init_cmb(A_lens=0.0, r_tensor=0.0)
        cls = sky.get_cmb_theory_cls()
        npt.assert_array_equal(cls[2], 0.0)

    def test_tensor_increases_bb(self, sky):
        """Increasing r_tensor should increase BB power."""
        sky.init_cmb(A_lens=1.0, r_tensor=0.0)
        bb_r0 = sky.get_cmb_theory_cls()[2].copy()

        sky.init_cmb(A_lens=1.0, r_tensor=0.1)
        bb_r01 = sky.get_cmb_theory_cls()[2]

        # BB with r=0.1 should exceed lensing-only BB at low ell
        assert np.sum(bb_r01[2:100]) > np.sum(bb_r0[2:100])

    def test_cls_deterministic(self, sky):
        """Same parameters should give identical Cls."""
        sky.init_cmb(A_lens=0.5, r_tensor=0.05)
        cls1 = sky.get_cmb_theory_cls().copy()

        sky.init_cmb(A_lens=0.5, r_tensor=0.05)
        cls2 = sky.get_cmb_theory_cls()

        npt.assert_array_equal(cls1, cls2)


# ------------------------------------------------------------------
# CMB maps
# ------------------------------------------------------------------


class TestCMBMaps:
    def test_maps_shape(self, sky):
        sky.init_cmb()
        maps = sky.get_cmb_maps()
        assert maps.shape == (3, NPIX)

    def test_lazy_generation(self, sky):
        """Maps should not exist until get_cmb_maps is called."""
        sky.init_cmb()
        assert sky._cmb_maps is None
        sky.get_cmb_maps()
        assert sky._cmb_maps is not None

    def test_cached_maps(self, sky):
        """Repeated get_cmb_maps returns the same object (cached)."""
        sky.init_cmb()
        maps1 = sky.get_cmb_maps()
        maps2 = sky.get_cmb_maps()
        assert maps1 is maps2

    def test_reinit_invalidates_cache(self, sky):
        """Calling init_cmb again should invalidate cached maps."""
        sky.init_cmb()
        sky.get_cmb_maps()
        assert sky._cmb_maps is not None
        sky.init_cmb()
        assert sky._cmb_maps is None


# ------------------------------------------------------------------
# Foreground reference Cls (deterministic)
# ------------------------------------------------------------------


class TestForegroundRefCls:
    def test_ref_cls_shape(self, sky):
        sky.init_dust()
        assert sky._dust_cls_ref.shape == (4, LMAX + 1)

    def test_ref_cls_tt_te_zero(self, sky):
        """TT (index 0) and TE (index 3) should be zero (polarization only)."""
        sky.init_dust()
        npt.assert_array_equal(sky._dust_cls_ref[0], 0.0)
        npt.assert_array_equal(sky._dust_cls_ref[3], 0.0)

    def test_ref_cls_at_pivot(self, sky):
        """At the pivot ell=80, Dl = amplitude, so Cl = amp * dl2cl."""
        amp_ee = 56.0
        amp_bb = 28.0
        sky.init_dust(amp_d_ee=amp_ee, amp_d_bb=amp_bb)
        # At ell=80: (ell/80)^alpha = 1, so Cl = amp * dl2cl[80]
        npt.assert_allclose(sky._dust_cls_ref[1, 80], amp_ee * sky.dl2cl[80])
        npt.assert_allclose(sky._dust_cls_ref[2, 80], amp_bb * sky.dl2cl[80])

    def test_ref_cls_deterministic(self, sky):
        """Same parameters produce identical reference Cls."""
        sky.init_dust(amp_d_ee=10.0, amp_d_bb=5.0)
        cls1 = sky._dust_cls_ref.copy()

        sky.init_dust(amp_d_ee=10.0, amp_d_bb=5.0)
        cls2 = sky._dust_cls_ref

        npt.assert_array_equal(cls1, cls2)

    def test_sync_ref_cls_at_pivot(self, sky):
        """Synchrotron pivot check at ell=80."""
        amp_ee = 9.0
        amp_bb = 1.6
        sky.init_sync(amp_s_ee=amp_ee, amp_s_bb=amp_bb)
        npt.assert_allclose(sky._sync_cls_ref[1, 80], amp_ee * sky.dl2cl[80])
        npt.assert_allclose(sky._sync_cls_ref[2, 80], amp_bb * sky.dl2cl[80])


# ------------------------------------------------------------------
# Dust power spectrum scaling (deterministic)
# ------------------------------------------------------------------


class TestDustTheoryCls:
    def test_shape(self, sky):
        sky.init_dust()
        cls = sky.get_dust_theory_cls()
        assert cls.shape == (NU.shape[0], 4, LMAX + 1)

    def test_tt_te_zero(self, sky):
        """TT and TE channels should remain zero after scaling."""
        sky.init_dust()
        cls = sky.get_dust_theory_cls()
        npt.assert_array_equal(cls[:, 0, :], 0.0)
        npt.assert_array_equal(cls[:, 3, :], 0.0)

    def test_at_reference_frequency(self, sky):
        """At nu0, scaled Cls should equal reference Cls."""
        sky.init_dust(nu0_d=353.0)
        cls_at_nu0 = sky.get_dust_theory_cls(nu=np.array([353.0]))
        npt.assert_allclose(cls_at_nu0[0], sky._dust_cls_ref, rtol=1e-10)

    def test_frequency_scaling_increases(self, sky):
        """Dust power should increase with frequency (in CMB units near 353 GHz)."""
        sky.init_dust()
        cls = sky.get_dust_theory_cls(nu=np.array([150.0, 353.0]))
        # Total EE power at 353 > 150 GHz
        assert np.sum(cls[1, 1, 2:]) > np.sum(cls[0, 1, 2:])

    def test_deterministic(self, sky):
        """Same inputs produce identical output."""
        sky.init_dust()
        cls1 = sky.get_dust_theory_cls().copy()
        cls2 = sky.get_dust_theory_cls()
        npt.assert_array_equal(cls1, cls2)

    def test_scaling_formula(self, sky):
        """Verify the scaling matches the expected modified blackbody formula."""
        nu0 = 353.0
        beta = 1.54
        temp = 19.6
        sky.init_dust(beta_d=beta, temp_d=temp, nu0_d=nu0)

        nu_test = np.array([150.0])
        cls = sky.get_dust_theory_cls(nu=nu_test)

        # Expected scale factor (amplitude, squared for Cl)
        expected_scale = (
            (nu_test[0] / nu0) ** beta
            * planck_law(temp, nu_test[0])
            / planck_law(temp, nu0)
        )
        # Ratio of scaled EE to reference EE, accounting for unit conversion
        ref_rj = sky._dust_cls_ref[1] * tcmb2trj(nu0) ** 2
        expected_rj = ref_rj * expected_scale**2
        expected_cmb = expected_rj * trj2tcmb(nu_test[0]) ** 2

        npt.assert_allclose(cls[0, 1], expected_cmb, rtol=1e-12)


# ------------------------------------------------------------------
# Synchrotron power spectrum scaling (deterministic)
# ------------------------------------------------------------------


class TestSyncTheoryCls:
    def test_shape(self, sky):
        sky.init_sync()
        cls = sky.get_sync_theory_cls()
        assert cls.shape == (NU.shape[0], 4, LMAX + 1)

    def test_at_reference_frequency(self, sky):
        """At nu0, scaled Cls should equal reference Cls."""
        sky.init_sync(nu0_s=23.0)
        cls_at_nu0 = sky.get_sync_theory_cls(nu=np.array([23.0]))
        npt.assert_allclose(cls_at_nu0[0], sky._sync_cls_ref, rtol=1e-10)

    def test_frequency_scaling_decreases(self, sky):
        """Synchrotron power should decrease with frequency (beta_s < 0)."""
        sky.init_sync()
        cls = sky.get_sync_theory_cls(nu=np.array([23.0, 93.0]))
        # Total EE power at 23 GHz > 93 GHz
        assert np.sum(cls[0, 1, 2:]) > np.sum(cls[1, 1, 2:])

    def test_scaling_formula(self, sky):
        """Verify the scaling matches the expected power-law formula."""
        nu0 = 23.0
        beta = -3.0
        sky.init_sync(beta_s=beta, nu0_s=nu0)

        nu_test = np.array([40.0])
        cls = sky.get_sync_theory_cls(nu=nu_test)

        # Expected power-law scale (squared for Cl)
        expected_scale = (nu_test[0] / nu0) ** beta
        ref_rj = sky._sync_cls_ref[1] * tcmb2trj(nu0) ** 2
        expected_rj = ref_rj * expected_scale**2
        expected_cmb = expected_rj * trj2tcmb(nu_test[0]) ** 2

        npt.assert_allclose(cls[0, 1], expected_cmb, rtol=1e-12)


# ------------------------------------------------------------------
# Foreground maps
# ------------------------------------------------------------------


class TestForegroundMaps:
    def test_dust_maps_shape(self, sky):
        sky.init_dust()
        maps = sky.get_dust_maps()
        assert maps.shape == (NU.shape[0], 3, NPIX)

    def test_sync_maps_shape(self, sky):
        sky.init_sync()
        maps = sky.get_sync_maps()
        assert maps.shape == (NU.shape[0], 3, NPIX)

    def test_dust_lazy_generation(self, sky):
        """Maps should not exist until get_dust_maps is called."""
        sky.init_dust()
        assert sky._dust_map_ref is None
        sky.get_dust_maps()
        assert sky._dust_map_ref is not None

    def test_dust_cached_maps(self, sky):
        """Repeated get_dust_maps uses the same reference maps."""
        sky.init_dust()
        sky.get_dust_maps()
        ref1 = sky._dust_map_ref.copy()
        sky.get_dust_maps()
        npt.assert_array_equal(sky._dust_map_ref, ref1)

    def test_dust_reinit_invalidates(self, sky):
        """Calling init_dust again invalidates cached maps."""
        sky.init_dust()
        sky.get_dust_maps()
        assert sky._dust_map_ref is not None
        sky.init_dust()
        assert sky._dust_map_ref is None

    def test_sync_lazy_generation(self, sky):
        sky.init_sync()
        assert sky._sync_map_ref is None
        sky.get_sync_maps()
        assert sky._sync_map_ref is not None


# ------------------------------------------------------------------
# White noise
# ------------------------------------------------------------------


class TestWhiteNoise:
    def test_shape(self, sky):
        dt = np.array([100.0] * NU.shape[0])
        dp = np.array([100.0] * NU.shape[0])
        sky.init_white_noise(sensitivity_t=dt, sensitivity_p=dp)
        maps = sky.get_white_noise_maps()
        assert maps.shape == (NU.shape[0], 3, NPIX)

    def test_mismatched_length_raises(self, sky):
        with pytest.raises(ValueError, match="same length"):
            sky.init_white_noise(
                sensitivity_t=np.array([100.0]),
                sensitivity_p=np.array([100.0, 200.0]),
            )

    def test_independent_realizations(self, sky):
        """Each call to get_white_noise_maps gives a different realization."""
        sky.init_white_noise()
        maps1 = sky.get_white_noise_maps()
        maps2 = sky.get_white_noise_maps()
        assert not np.array_equal(maps1, maps2)

    def test_noise_level_scaling(self, sky):
        """Higher sensitivity value should produce higher noise RMS."""
        sky.init_white_noise(
            sensitivity_t=np.array([10.0]),
            sensitivity_p=np.array([10.0]),
        )
        np.random.seed(42)
        maps_low = sky.get_white_noise_maps()

        sky.init_white_noise(
            sensitivity_t=np.array([1000.0]),
            sensitivity_p=np.array([1000.0]),
        )
        np.random.seed(42)
        maps_high = sky.get_white_noise_maps()

        assert np.std(maps_high) > np.std(maps_low)


# ------------------------------------------------------------------
# Beam and pixel window utilities
# ------------------------------------------------------------------


class TestUtilities:
    def test_pixwin_returns_tuple(self, sky):
        pw = sky.pixwin()
        assert isinstance(pw, tuple) or (isinstance(pw, np.ndarray) and pw.ndim == 2)

    def test_gauss_beam_shape(self, sky):
        bl = sky.gauss_beam(fwhm=30.0)
        assert bl.shape[0] == LMAX + 1


# ------------------------------------------------------------------
# Input coercion (float, int, list -> ndarray)
# ------------------------------------------------------------------


class TestInputCoercion:
    def test_constructor_float_nu(self):
        sky = pygsm.Sky(nside=NSIDE, nu=93.0)
        sky.init_dust()
        cls = sky.get_dust_theory_cls()
        assert cls.shape == (1, 4, LMAX + 1)

    def test_constructor_int_nu(self):
        sky = pygsm.Sky(nside=NSIDE, nu=93)
        sky.init_dust()
        maps = sky.get_dust_maps()
        assert maps.shape == (1, 3, NPIX)

    def test_constructor_list_nu(self):
        sky = pygsm.Sky(nside=NSIDE, nu=[93, 145, 225])
        sky.init_sync()
        cls = sky.get_sync_theory_cls()
        assert cls.shape == (3, 4, LMAX + 1)

    def test_get_methods_float_nu(self, sky):
        sky.init_dust()
        cls = sky.get_dust_theory_cls(nu=93.0)
        assert cls.shape == (1, 4, LMAX + 1)
        maps = sky.get_dust_maps(nu=93.0)
        assert maps.shape == (1, 3, NPIX)

    def test_get_methods_list_nu(self, sky):
        sky.init_sync()
        cls = sky.get_sync_theory_cls(nu=[23.0, 93.0])
        assert cls.shape == (2, 4, LMAX + 1)

    def test_set_freqs_float(self):
        sky = pygsm.Sky(nside=NSIDE)
        sky.set_freqs(93.0)
        sky.init_dust()
        assert sky.get_dust_theory_cls().shape == (1, 4, LMAX + 1)

    def test_set_freqs_list(self):
        sky = pygsm.Sky(nside=NSIDE)
        sky.set_freqs([93, 145])
        sky.init_dust()
        assert sky.get_dust_theory_cls().shape == (2, 4, LMAX + 1)

    def test_white_noise_scalar_sensitivity(self, sky):
        sky.init_white_noise(sensitivity_t=100.0, sensitivity_p=100.0)
        maps = sky.get_white_noise_maps()
        assert maps.shape == (1, 3, NPIX)

    def test_white_noise_list_sensitivity(self, sky):
        sky.init_white_noise(sensitivity_t=[100, 200], sensitivity_p=[100, 200])
        maps = sky.get_white_noise_maps()
        assert maps.shape == (2, 3, NPIX)


# ------------------------------------------------------------------
# nside guard for CAMB data
# ------------------------------------------------------------------


class TestNsideGuard:
    def test_high_nside_raises(self):
        sky = pygsm.Sky(nside=1024)
        with pytest.raises(ValueError, match="exceeds the bundled CAMB spectra"):
            sky.init_cmb()

    def test_max_valid_nside(self):
        """nside=650 gives lmax=1949, which is within CAMB range."""
        sky = pygsm.Sky(nside=650)
        sky.init_cmb()  # should not raise
        assert sky.get_cmb_theory_cls().shape == (4, 650 * 3)


# ------------------------------------------------------------------
# Seed reproducibility
# ------------------------------------------------------------------


class TestSeedReproducibility:
    def test_cmb_maps_same_seed(self):
        sky1 = pygsm.Sky(nside=NSIDE)
        sky1.init_cmb(seed=42)
        maps1 = sky1.get_cmb_maps()

        sky2 = pygsm.Sky(nside=NSIDE)
        sky2.init_cmb(seed=42)
        maps2 = sky2.get_cmb_maps()

        npt.assert_array_equal(maps1, maps2)

    def test_cmb_maps_different_seed(self):
        sky1 = pygsm.Sky(nside=NSIDE)
        sky1.init_cmb(seed=42)
        maps1 = sky1.get_cmb_maps()

        sky2 = pygsm.Sky(nside=NSIDE)
        sky2.init_cmb(seed=99)
        maps2 = sky2.get_cmb_maps()

        assert not np.array_equal(maps1, maps2)

    def test_dust_maps_same_seed(self):
        sky1 = pygsm.Sky(nside=NSIDE, nu=NU)
        sky1.init_dust(seed=42)
        maps1 = sky1.get_dust_maps()

        sky2 = pygsm.Sky(nside=NSIDE, nu=NU)
        sky2.init_dust(seed=42)
        maps2 = sky2.get_dust_maps()

        npt.assert_array_equal(maps1, maps2)

    def test_sync_maps_same_seed(self):
        sky1 = pygsm.Sky(nside=NSIDE, nu=NU)
        sky1.init_sync(seed=42)
        maps1 = sky1.get_sync_maps()

        sky2 = pygsm.Sky(nside=NSIDE, nu=NU)
        sky2.init_sync(seed=42)
        maps2 = sky2.get_sync_maps()

        npt.assert_array_equal(maps1, maps2)

    def test_white_noise_same_seed(self, sky):
        sky.init_white_noise()
        maps1 = sky.get_white_noise_maps(seed=42)
        maps2 = sky.get_white_noise_maps(seed=42)
        npt.assert_array_equal(maps1, maps2)

    def test_white_noise_different_seed(self, sky):
        sky.init_white_noise()
        maps1 = sky.get_white_noise_maps(seed=42)
        maps2 = sky.get_white_noise_maps(seed=99)
        assert not np.array_equal(maps1, maps2)

    def test_seed_does_not_affect_global_state(self):
        """Using a seed should not alter numpy's global random state."""
        np.random.seed(123)
        expected = np.random.randn(5).copy()

        np.random.seed(123)
        sky = pygsm.Sky(nside=NSIDE)
        sky.init_cmb(seed=42)
        sky.get_cmb_maps()
        actual = np.random.randn(5)

        npt.assert_array_equal(actual, expected)
