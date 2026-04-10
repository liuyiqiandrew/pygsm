from __future__ import annotations

import contextlib
import os
from typing import Optional, Union

import healpy as hp
import numpy as np

from .util import planck_law, tcmb2trj, trj2tcmb

# Maximum ell available in bundled CAMB spectra
_CAMB_LMAX = 1950


class Sky:
    """PyGSM Sky object for generating Gaussian CMB and foreground simulations.

    Parameters
    ----------
    nside : int
        HEALPix nside resolution parameter. Default is 256.
    nu : np.ndarray, optional
        Array of observation frequencies in GHz. Can also be set later
        via :meth:`set_freqs` or passed directly to ``get_*`` methods.
    """

    def __init__(
        self,
        nside: int = 256,
        nu: Optional[Union[float, list, np.ndarray]] = None,
    ) -> None:
        self.nside: int = nside
        self.lmax: int = nside * 3 - 1
        self.ell: np.ndarray = np.arange(self.lmax + 1)

        # Cl <-> Dl conversion factors: Dl = ell*(ell+1)/(2*pi) * Cl
        # Both are zero at ell=0,1 (monopole/dipole undefined for this conversion)
        self.cl2dl: np.ndarray = np.zeros_like(self.ell, dtype=float)
        self.cl2dl[2:] = self.ell[2:] * (self.ell[2:] + 1) / (2 * np.pi)
        self.dl2cl: np.ndarray = np.zeros_like(self.ell, dtype=float)
        self.dl2cl[2:] = 1.0 / self.cl2dl[2:]

        self._nu: Optional[np.ndarray] = (
            np.atleast_1d(np.asarray(nu, dtype=float)) if nu is not None else None
        )

        # CMB state
        self._cmb_cls: Optional[np.ndarray] = None
        self._cmb_maps: Optional[np.ndarray] = None
        self._cmb_seed: Optional[int] = None

        # White noise state
        self._noise_sensitivity_t: Optional[np.ndarray] = None
        self._noise_sensitivity_p: Optional[np.ndarray] = None

        # Dust state
        self.temp_d: Optional[float] = None
        self.beta_d: Optional[float] = None
        self.nu0_d: Optional[float] = None
        self._dust_cls_ref: Optional[np.ndarray] = None
        self._dust_map_ref: Optional[np.ndarray] = None
        self._dust_seed: Optional[int] = None

        # Synchrotron state
        self.beta_s: Optional[float] = None
        self.nu0_s: Optional[float] = None
        self._sync_cls_ref: Optional[np.ndarray] = None
        self._sync_map_ref: Optional[np.ndarray] = None
        self._sync_seed: Optional[int] = None

    def set_freqs(self, nu: Union[float, list, np.ndarray]) -> None:
        """Set or update the observation frequency array.

        Parameters
        ----------
        nu : float, list, or np.ndarray
            Frequency or frequencies in GHz. Scalars and lists are
            converted to a 1-D numpy array.
        """
        self._nu = np.atleast_1d(np.asarray(nu, dtype=float))

    # ------------------------------------------------------------------
    # Shared foreground internals
    # ------------------------------------------------------------------

    def _gen_foreground_ref_cls(
        self,
        amp_ee: float,
        amp_bb: float,
        alpha_ee: float,
        alpha_bb: float,
    ) -> np.ndarray:
        """Generate reference power spectra for a foreground component.

        Computes power-law Cls at a pivot multipole of ell=80,
        following the standard foreground parametrization:
            Cl_XX = amp_XX * (ell / 80)^alpha_XX

        Parameters
        ----------
        amp_ee : float
            EE amplitude at ell=80 in CMB uK^2.
        amp_bb : float
            BB amplitude at ell=80 in CMB uK^2.
        alpha_ee : float
            EE angular power spectral index.
        alpha_bb : float
            BB angular power spectral index.

        Returns
        -------
        np.ndarray
            Reference Cls of shape ``(4, lmax+1)`` ordered as
            ``[TT, EE, BB, TE]``. TT and TE are zero (polarization only).
        """
        cl0 = np.zeros(self.lmax + 1)
        tmp_ell = np.copy(self.ell).astype(float)
        tmp_ell[0] = 1.0  # avoid division by zero at ell=0
        cl_ee = amp_ee * (tmp_ell / 80.0) ** alpha_ee * self.dl2cl
        cl_bb = amp_bb * (tmp_ell / 80.0) ** alpha_bb * self.dl2cl
        return np.array([cl0, cl_ee, cl_bb, cl0])

    def _ensure_maps(
        self,
        cls_ref: np.ndarray,
        map_ref: Optional[np.ndarray],
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Generate maps from reference Cls if not already cached.

        Parameters
        ----------
        cls_ref : np.ndarray
            Reference power spectra of shape ``(4, lmax+1)``.
        map_ref : np.ndarray or None
            Cached maps, or None if not yet generated.
        seed : int, optional
            Random seed for reproducible map generation.

        Returns
        -------
        np.ndarray
            HEALPix maps of shape ``(3, npix)``.
        """
        if map_ref is not None:
            return map_ref
        return self._synfast_with_seed(cls_ref, seed)

    def _synfast_with_seed(
        self, cls: np.ndarray, seed: Optional[int] = None
    ) -> np.ndarray:
        """Run healpy.synfast, optionally with a fixed random seed.

        healpy.synfast uses numpy's legacy random state, so we
        save/restore it around the call when a seed is given.

        Parameters
        ----------
        cls : np.ndarray
            Power spectra of shape ``(4, lmax+1)``.
        seed : int, optional
            If provided, sets the numpy random seed before synfast
            and restores the previous state afterward.

        Returns
        -------
        np.ndarray
            HEALPix maps of shape ``(3, npix)``.
        """
        if seed is None:
            return hp.synfast(cls, self.nside, new=True)
        state = np.random.get_state()
        try:
            np.random.seed(seed)
            return hp.synfast(cls, self.nside, new=True)
        finally:
            np.random.set_state(state)

    def _scale_cls(
        self,
        cls_ref: np.ndarray,
        nu0: float,
        nu: np.ndarray,
        scale: np.ndarray,
    ) -> np.ndarray:
        """Scale reference Cls to target frequencies.

        Converts reference Cls from CMB to RJ units, applies the
        frequency-dependent scale factors (squared for power spectra),
        then converts back to CMB units.

        Parameters
        ----------
        cls_ref : np.ndarray
            Reference Cls at nu0, shape ``(4, lmax+1)``, in CMB uK^2.
        nu0 : float
            Reference frequency in GHz.
        nu : np.ndarray
            Target frequencies in GHz, shape ``(nfreq,)``.
        scale : np.ndarray
            Frequency scaling factors (amplitude, not squared),
            shape ``(nfreq,)``.

        Returns
        -------
        np.ndarray
            Scaled Cls of shape ``(nfreq, 4, lmax+1)`` in CMB uK^2.
        """
        # Convert reference Cls to RJ units
        rj_cls0 = cls_ref * tcmb2trj(nu0) ** 2
        # Tile across frequencies and apply scale^2 (power spectrum)
        rj_cls = np.tile(rj_cls0, (nu.shape[0], 1)).reshape(*nu.shape, *rj_cls0.shape)
        rj_cls *= scale[:, None, None] ** 2
        # Convert back to CMB units
        return rj_cls * (trj2tcmb(nu) ** 2)[:, None, None]

    def _scale_maps(
        self,
        map_ref: np.ndarray,
        nu0: float,
        nu: np.ndarray,
        scale: np.ndarray,
    ) -> np.ndarray:
        """Scale reference maps to target frequencies.

        Converts reference maps from CMB to RJ units, applies the
        frequency-dependent scale factors, then converts back to CMB units.

        Parameters
        ----------
        map_ref : np.ndarray
            Reference maps at nu0, shape ``(3, npix)``, in CMB uK.
        nu0 : float
            Reference frequency in GHz.
        nu : np.ndarray
            Target frequencies in GHz, shape ``(nfreq,)``.
        scale : np.ndarray
            Frequency scaling factors, shape ``(nfreq,)``.

        Returns
        -------
        np.ndarray
            Scaled maps of shape ``(nfreq, 3, npix)`` in CMB uK.
        """
        # Convert reference maps to RJ units
        rj_map0 = map_ref * tcmb2trj(nu0)
        # Tile across frequencies and apply scale
        rj_map = np.tile(rj_map0, (nu.shape[0], 1)).reshape(*nu.shape, *rj_map0.shape)
        rj_map *= scale[:, None, None]
        # Convert back to CMB units
        return rj_map * trj2tcmb(nu)[:, None, None]

    def _resolve_nu(self, nu: Optional[Union[float, list, np.ndarray]]) -> np.ndarray:
        """Resolve frequency array from argument or stored default.

        Scalars and lists are converted to a 1-D numpy array.

        Parameters
        ----------
        nu : float, list, np.ndarray, or None
            Frequencies passed by the caller, or None to use stored default.

        Returns
        -------
        np.ndarray
            Resolved 1-D frequency array.

        Raises
        ------
        ValueError
            If no frequencies are available.
        """
        if nu is not None:
            return np.atleast_1d(np.asarray(nu, dtype=float))
        if self._nu is not None:
            return self._nu
        raise ValueError(
            "No frequencies provided. Pass nu as an argument or set "
            "frequencies via the constructor or set_freqs()."
        )

    # ------------------------------------------------------------------
    # Dust
    # ------------------------------------------------------------------

    def init_dust(
        self,
        amp_d_ee: float = 56.0,
        amp_d_bb: float = 28.0,
        alpha_d_ee: float = -0.32,
        alpha_d_bb: float = -0.16,
        temp_d: float = 19.6,
        beta_d: float = 1.54,
        nu0_d: float = 353.0,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize or reinitialize the dust component.

        Computes reference power spectra at the reference frequency.
        Map generation is deferred to the first :meth:`get_dust_maps` call.
        Calling this method again invalidates any cached maps, so the next
        :meth:`get_dust_maps` call will produce a new random realization.

        Parameters
        ----------
        amp_d_ee : float
            Dust EE amplitude at ell=80 in CMB uK^2. Default is 56.
        amp_d_bb : float
            Dust BB amplitude at ell=80 in CMB uK^2. Default is 28.
        alpha_d_ee : float
            Dust EE angular power spectral index. Default is -0.32.
        alpha_d_bb : float
            Dust BB angular power spectral index. Default is -0.16.
        temp_d : float
            Dust temperature in Kelvin. Default is 19.6.
        beta_d : float
            Dust frequency spectral index. Default is 1.54.
        nu0_d : float
            Dust reference frequency in GHz. Default is 353.
        seed : int, optional
            Random seed for reproducible map realizations. If None
            (default), maps are non-reproducible.
        """
        self._dust_cls_ref = self._gen_foreground_ref_cls(
            amp_d_ee, amp_d_bb, alpha_d_ee, alpha_d_bb
        )
        # Invalidate cached maps so next get_dust_maps() generates fresh
        self._dust_map_ref = None
        self._dust_seed = seed
        self.temp_d = temp_d
        self.beta_d = beta_d
        self.nu0_d = nu0_d

    def _dust_scale(self, nu: np.ndarray) -> np.ndarray:
        """Compute dust SED scaling factors.

        Modified blackbody: (nu/nu0)^beta * B(T, nu) / B(T, nu0)

        Parameters
        ----------
        nu : np.ndarray
            Target frequencies in GHz.

        Returns
        -------
        np.ndarray
            Scale factors, shape ``(nfreq,)``.
        """
        return (nu / self.nu0_d) ** self.beta_d * (
            planck_law(self.temp_d, nu) / planck_law(self.temp_d, self.nu0_d)
        )

    def get_dust_theory_cls(
        self, nu: Optional[Union[float, list, np.ndarray]] = None
    ) -> np.ndarray:
        """Generate dust power spectra at the given frequencies.

        Does not require map generation (cheap operation).

        Parameters
        ----------
        nu : float, list, np.ndarray, or None
            Frequencies in GHz. Scalars and lists are accepted.
            If None, uses the frequencies set in the constructor
            or via :meth:`set_freqs`.

        Returns
        -------
        np.ndarray
            Dust Cls of shape ``(nfreq, 4, lmax+1)`` in CMB uK^2.

        Raises
        ------
        RuntimeError
            If :meth:`init_dust` has not been called.
        """
        if self._dust_cls_ref is None:
            raise RuntimeError("Dust not initialized. Call init_dust() first.")
        nu = self._resolve_nu(nu)
        scale = self._dust_scale(nu)
        return self._scale_cls(self._dust_cls_ref, self.nu0_d, nu, scale)

    def get_dust_maps(
        self, nu: Optional[Union[float, list, np.ndarray]] = None
    ) -> np.ndarray:
        """Generate dust maps at the given frequencies.

        On first call after :meth:`init_dust`, generates reference maps
        via ``healpy.synfast`` and caches them. Subsequent calls reuse
        the cached maps. Call :meth:`init_dust` again to get a new
        random realization.

        Parameters
        ----------
        nu : float, list, np.ndarray, or None
            Frequencies in GHz. Scalars and lists are accepted.
            If None, uses the frequencies set in the constructor
            or via :meth:`set_freqs`.

        Returns
        -------
        np.ndarray
            Dust maps of shape ``(nfreq, 3, npix)`` in CMB uK.

        Raises
        ------
        RuntimeError
            If :meth:`init_dust` has not been called.
        """
        if self._dust_cls_ref is None:
            raise RuntimeError("Dust not initialized. Call init_dust() first.")
        nu = self._resolve_nu(nu)
        # Lazy map generation: run synfast only on first call
        self._dust_map_ref = self._ensure_maps(
            self._dust_cls_ref, self._dust_map_ref, self._dust_seed
        )
        scale = self._dust_scale(nu)
        return self._scale_maps(self._dust_map_ref, self.nu0_d, nu, scale)

    # ------------------------------------------------------------------
    # Synchrotron
    # ------------------------------------------------------------------

    def init_sync(
        self,
        amp_s_ee: float = 9.0,
        amp_s_bb: float = 1.6,
        alpha_s_ee: float = -0.7,
        alpha_s_bb: float = -0.93,
        beta_s: float = -3.0,
        nu0_s: float = 23.0,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize or reinitialize the synchrotron component.

        Computes reference power spectra at the reference frequency.
        Map generation is deferred to the first :meth:`get_sync_maps` call.
        Calling this method again invalidates any cached maps, so the next
        :meth:`get_sync_maps` call will produce a new random realization.

        Parameters
        ----------
        amp_s_ee : float
            Synchrotron EE amplitude at ell=80 in CMB uK^2. Default is 9.
        amp_s_bb : float
            Synchrotron BB amplitude at ell=80 in CMB uK^2. Default is 1.6.
        alpha_s_ee : float
            Synchrotron EE angular power spectral index. Default is -0.7.
        alpha_s_bb : float
            Synchrotron BB angular power spectral index. Default is -0.93.
        beta_s : float
            Synchrotron frequency spectral index. Default is -3.
        nu0_s : float
            Synchrotron reference frequency in GHz. Default is 23.
        seed : int, optional
            Random seed for reproducible map realizations. If None
            (default), maps are non-reproducible.
        """
        self._sync_cls_ref = self._gen_foreground_ref_cls(
            amp_s_ee, amp_s_bb, alpha_s_ee, alpha_s_bb
        )
        # Invalidate cached maps so next get_sync_maps() generates fresh
        self._sync_map_ref = None
        self._sync_seed = seed
        self.beta_s = beta_s
        self.nu0_s = nu0_s

    def _sync_scale(self, nu: np.ndarray) -> np.ndarray:
        """Compute synchrotron SED scaling factors.

        Power law: (nu/nu0)^beta_s

        Parameters
        ----------
        nu : np.ndarray
            Target frequencies in GHz.

        Returns
        -------
        np.ndarray
            Scale factors, shape ``(nfreq,)``.
        """
        return (nu / self.nu0_s) ** self.beta_s

    def get_sync_theory_cls(
        self, nu: Optional[Union[float, list, np.ndarray]] = None
    ) -> np.ndarray:
        """Generate synchrotron power spectra at the given frequencies.

        Does not require map generation (cheap operation).

        Parameters
        ----------
        nu : float, list, np.ndarray, or None
            Frequencies in GHz. Scalars and lists are accepted.
            If None, uses the frequencies set in the constructor
            or via :meth:`set_freqs`.

        Returns
        -------
        np.ndarray
            Synchrotron Cls of shape ``(nfreq, 4, lmax+1)`` in CMB uK^2.

        Raises
        ------
        RuntimeError
            If :meth:`init_sync` has not been called.
        """
        if self._sync_cls_ref is None:
            raise RuntimeError("Synchrotron not initialized. Call init_sync() first.")
        nu = self._resolve_nu(nu)
        scale = self._sync_scale(nu)
        return self._scale_cls(self._sync_cls_ref, self.nu0_s, nu, scale)

    def get_sync_maps(
        self, nu: Optional[Union[float, list, np.ndarray]] = None
    ) -> np.ndarray:
        """Generate synchrotron maps at the given frequencies.

        On first call after :meth:`init_sync`, generates reference maps
        via ``healpy.synfast`` and caches them. Subsequent calls reuse
        the cached maps. Call :meth:`init_sync` again to get a new
        random realization.

        Parameters
        ----------
        nu : float, list, np.ndarray, or None
            Frequencies in GHz. Scalars and lists are accepted.
            If None, uses the frequencies set in the constructor
            or via :meth:`set_freqs`.

        Returns
        -------
        np.ndarray
            Synchrotron maps of shape ``(nfreq, 3, npix)`` in CMB uK.

        Raises
        ------
        RuntimeError
            If :meth:`init_sync` has not been called.
        """
        if self._sync_cls_ref is None:
            raise RuntimeError("Synchrotron not initialized. Call init_sync() first.")
        nu = self._resolve_nu(nu)
        # Lazy map generation: run synfast only on first call
        self._sync_map_ref = self._ensure_maps(
            self._sync_cls_ref, self._sync_map_ref, self._sync_seed
        )
        scale = self._sync_scale(nu)
        return self._scale_maps(self._sync_map_ref, self.nu0_s, nu, scale)

    # ------------------------------------------------------------------
    # CMB
    # ------------------------------------------------------------------

    def init_cmb(
        self,
        A_lens: float = 1.0,
        r_tensor: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        """Initialize or reinitialize the CMB component.

        Loads pre-computed CAMB power spectra and combines them according
        to the lensing amplitude and tensor-to-scalar ratio:
            Cl = A_lens * Cl_lens + r_tensor * (Cl_r1 - Cl_lens)

        Map generation is deferred to the first :meth:`get_cmb_maps` call.
        Calling this method again invalidates any cached maps.

        Parameters
        ----------
        A_lens : float
            Lensing amplitude. 1 = full lensing, 0 = no lensing.
            Default is 1.
        r_tensor : float
            Tensor-to-scalar ratio. Default is 0 (no primordial
            gravitational waves).
        seed : int, optional
            Random seed for reproducible map realizations. If None
            (default), maps are non-reproducible.

        Raises
        ------
        ValueError
            If lmax exceeds the bundled CAMB spectra range (ell <= 1950).
        """
        if self.lmax > _CAMB_LMAX:
            raise ValueError(
                f"nside={self.nside} gives lmax={self.lmax}, which exceeds "
                f"the bundled CAMB spectra (lmax={_CAMB_LMAX}). "
                f"Use nside <= {(_CAMB_LMAX + 1) // 3} or provide your own spectra."
            )

        cur_dir = os.path.abspath(os.path.dirname(__file__))

        # Load CAMB spectra: columns are [ell, TT, EE, BB, TE]
        camb_lens_dl = np.loadtxt(cur_dir + "/data/cmb_spec/camb_lens_nobb.dat")
        camb_lens_dl = np.concatenate((np.zeros((1, 5)), camb_lens_dl), axis=0)
        camb_r1_dl = np.loadtxt(cur_dir + "/data/cmb_spec/camb_lens_r1.dat")
        camb_r1_dl = np.concatenate((np.zeros((1, 5)), camb_r1_dl), axis=0)

        # Select ells up to lmax and transpose to (4, lmax+1)
        camb_lens_dl = camb_lens_dl[self.ell, 1:].T
        camb_r1_dl = camb_r1_dl[self.ell, 1:].T

        # Combine lensing and tensor contributions, convert Dl -> Cl
        self._cmb_cls = (
            A_lens * camb_lens_dl + r_tensor * (camb_r1_dl - camb_lens_dl)
        ) * self.dl2cl

        # Invalidate cached maps
        self._cmb_maps = None
        self._cmb_seed = seed

    def get_cmb_theory_cls(self) -> np.ndarray:
        """Get CMB power spectra.

        Returns
        -------
        np.ndarray
            CMB Cls of shape ``(4, lmax+1)`` ordered as ``[TT, EE, BB, TE]``,
            in CMB uK^2. Frequency-independent.

        Raises
        ------
        RuntimeError
            If :meth:`init_cmb` has not been called.
        """
        if self._cmb_cls is None:
            raise RuntimeError("CMB not initialized. Call init_cmb() first.")
        return self._cmb_cls

    def get_cmb_maps(self) -> np.ndarray:
        """Get CMB maps.

        On first call after :meth:`init_cmb`, generates maps via
        ``healpy.synfast`` and caches them. Call :meth:`init_cmb`
        again to get a new random realization.

        Returns
        -------
        np.ndarray
            CMB maps of shape ``(3, npix)`` ordered as ``[T, Q, U]``,
            in CMB uK. Frequency-independent.

        Raises
        ------
        RuntimeError
            If :meth:`init_cmb` has not been called.
        """
        if self._cmb_cls is None:
            raise RuntimeError("CMB not initialized. Call init_cmb() first.")
        # Lazy map generation
        if self._cmb_maps is None:
            self._cmb_maps = self._synfast_with_seed(self._cmb_cls, self._cmb_seed)
        return self._cmb_maps

    # ------------------------------------------------------------------
    # White noise
    # ------------------------------------------------------------------

    def init_white_noise(
        self,
        sensitivity_t: Union[float, list, np.ndarray] = np.array([100.0]),
        sensitivity_p: Union[float, list, np.ndarray] = np.array([100.0]),
    ) -> None:
        """Initialize white noise parameters.

        Stores per-frequency temperature and polarization sensitivities.
        Each call to :meth:`get_white_noise_maps` generates a fresh
        random realization from these sensitivities.

        Parameters
        ----------
        sensitivity_t : float, list, or np.ndarray
            Temperature noise level(s) in uK-arcmin, one per frequency.
            Scalars and lists are converted to a 1-D array.
            Default is ``[100.]``.
        sensitivity_p : float, list, or np.ndarray
            Polarization (Q, U) noise level(s) in uK-arcmin, one per
            frequency. Divide by sqrt(2) if your input is total
            polarization sensitivity. Default is ``[100.]``.

        Raises
        ------
        ValueError
            If sensitivity_t and sensitivity_p have different lengths.
        """
        sensitivity_t = np.atleast_1d(np.asarray(sensitivity_t, dtype=float))
        sensitivity_p = np.atleast_1d(np.asarray(sensitivity_p, dtype=float))
        if len(sensitivity_t) != len(sensitivity_p):
            raise ValueError(
                "sensitivity_t and sensitivity_p must have the same length."
            )
        self._noise_sensitivity_t = sensitivity_t
        self._noise_sensitivity_p = sensitivity_p

    def get_white_noise_maps(self, seed: Optional[int] = None) -> np.ndarray:
        """Generate a white noise realization.

        Each call produces an independent random realization using the
        sensitivities set in :meth:`init_white_noise`.

        Parameters
        ----------
        seed : int, optional
            Random seed for a reproducible realization. Uses
            ``numpy.random.default_rng`` for generation. If None
            (default), each call produces a different realization.

        Returns
        -------
        np.ndarray
            Noise maps of shape ``(nfreq, 3, npix)`` in CMB uK.

        Raises
        ------
        RuntimeError
            If :meth:`init_white_noise` has not been called.
        """
        if self._noise_sensitivity_t is None:
            raise RuntimeError(
                "White noise not initialized. Call init_white_noise() first."
            )
        rng = np.random.default_rng(seed)
        pix_res = hp.nside2resol(self.nside, True)  # pixel size in arcmin
        npix = hp.nside2npix(self.nside)
        nfreq = self._noise_sensitivity_t.shape[0]

        # Temperature noise: sigma = sensitivity / pixel_size
        t_n = (
            self._noise_sensitivity_t[:, None]
            / pix_res
            * rng.standard_normal((nfreq, npix))
        )
        # Polarization noise: independent Q and U realizations
        t_p = (
            self._noise_sensitivity_p[:, None]
            / pix_res
            * rng.standard_normal((2, nfreq, npix))
        )
        # Stack T, Q, U -> (3, nfreq, npix), then swap to (nfreq, 3, npix)
        noise = np.concatenate((t_n[None, :], t_p), axis=0)
        return np.swapaxes(noise, 0, 1)

    # ------------------------------------------------------------------
    # Beam and pixel window utilities
    # ------------------------------------------------------------------

    def pixwin(self) -> tuple:
        """Get the HEALPix pixel window function for the current nside.

        Returns
        -------
        tuple
            ``(pw_T, pw_P)`` pixel window functions from healpy.
        """
        return hp.pixwin(self.nside, pol=True)

    def gauss_beam(self, fwhm: float) -> np.ndarray:
        """Get a Gaussian beam window function.

        Parameters
        ----------
        fwhm : float
            Full width at half maximum of the beam in arcmin.

        Returns
        -------
        np.ndarray
            Gaussian beam window function from healpy, evaluated
            up to ``lmax``.
        """
        fwhm_rad = fwhm / 60 / 180 * np.pi
        return hp.gauss_beam(fwhm=fwhm_rad, lmax=self.lmax, pol=True)
