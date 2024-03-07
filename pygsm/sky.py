import numpy as np
import healpy as hp
import os

from .util import *

class Sky:
    """
        PyGSM Sky object
    """
    def __init__(self, nside=256, nu=None) -> None:
        """
            Constructor for sky
            ==========
            Parameters:
                nside: int
                    nside of map, Default 256
                nu: np.ndarray
                    array of frequencies (in GHz)
        """
        self.nside:int = nside
        self.lmax:int = nside * 3 - 1
        self.ell:int = np.arange(self.lmax + 1)
        self.cl2dl:np.ndarray = self.ell * (self.ell + 1) / 2 / np.pi
        tmp = np.copy(self.cl2dl)
        tmp[0] = 1
        self.dl2cl:np.ndarray = 1 / tmp
        self.dl2cl[[0, 1]] = 0.

        self.__nu:np.ndarray = nu

        self.cmb_cls:np.ndarray = None
        self.cmb_maps:np.ndarray = None
        
        self.noise_dtemp:np.ndarray = None
        self.noise_dpol:np.ndarray = None

        self.temp_d:float = None
        self.beta_d:float = None
        self.nu0_d:float = None
        self.__dust_cls_ref:np.ndarray = None
        self.__dust_map_ref:np.ndarray = None
        
        self.beta_s:float = None
        self.__sync_cls_ref:np.ndarray = None 
        self.__sync_map_ref:np.ndarray = None
        
    def set_freqs(self, nu:np.ndarray):
        """
            reset frequency arrays
            ==========
            Parameters:
                nu: np.ndarray
                    array of frequencies (in GHz)
        """
        self.__nu = nu

    def __gen_dust_ref(self, 
                    amp_d_ee=56., 
                    amp_d_bb=28., 
                    alpha_d_ee=-0.32, 
                    alpha_d_bb=-0.16,
                    ) -> None:
        """ Generate dust reference cls and maps at nu0 """
        cl0 = np.zeros_like(self.ell)
        tmp_ell = self.ell
        tmp_ell[0] = 1.
        cl_ee = amp_d_ee * (tmp_ell / 80.)**alpha_d_ee * self.dl2cl
        cl_bb = amp_d_bb * (tmp_ell / 80.)**alpha_d_bb * self.dl2cl
        self.__dust_cls_ref = np.array([cl0, cl_ee, cl_bb, cl0]) #TODO: add temperature
        self.__dust_map_ref = hp.synfast(self.__dust_cls_ref, self.nside, new=True)

    def init_dust(self,
                amp_d_ee=56., 
                amp_d_bb=28., 
                alpha_d_ee=-0.32, 
                alpha_d_bb=-0.16,
                temp_d=19.6,
                beta_d=1.54,
                nu0_d=353.,
                re_init=False) -> None:
        """
            Initialize / regenerated dust
            ==========
            Parameters:
                amp_d_ee: float 
                    Dust EE amplitude in CMB unit, default 56.
                amp_d_bb: float
                    Dust BB amplitude in CMB unit, default 28.
                alpha_d_ee: float
                    Dust EE Angular power spectral index, default -0.32.
                alpha_d_bb: float
                    Dust BB Angular power spectral index, default -0.16.
                temp_d: float
                    Dust temperature, dsfault 19.6
                beta_d: float
                    Dust frequency spectral index, default 1.54
                nu0_d: float
                    Dust reference frequency, default 353.
                re_init: bool
                    Regenerate dust when function is called, dafault False
        """
        if not re_init and self.__dust_map_ref is not None:
            return
        self.__gen_dust_ref(amp_d_ee=amp_d_ee, amp_d_bb=amp_d_bb, alpha_d_ee=alpha_d_ee,
                        alpha_d_bb=alpha_d_bb)
        self.temp_d = temp_d
        self.beta_d = beta_d
        self.nu0_d = nu0_d

    def get_dust_theory_cls(self, nu:np.ndarray=None):
        """ 
            generate dust cls given frequencies
            ==========
            Inputs:
                nu: np.ndarray
                    Array of frequencies to generate the power spectra cl. If None, use the 
                    frequency passed into the constructor. Default None.
            ==========
            Returns:
                cls: np.ndarray
                    Array of dust cls of shape (Nfreq, 4, Lmax + 1)
        """
        if nu is None:
            nu = self.__nu
        rj_cls0 = self.__dust_cls_ref * tcmb2trj(self.nu0_d)**2
        scale = ((nu / self.nu0_d)**self.beta_d * \
            (planck_law(self.temp_d, nu) / planck_law(self.temp_d, self.nu0_d)))**2
        rj_cls = np.tile(rj_cls0, (nu.shape[0], 1)).reshape(*nu.shape, *rj_cls0.shape)
        rj_cls *= scale[:, None, None]
        return rj_cls * (trj2tcmb(nu)**2)[:, None, None] 
        
    def get_dust_maps(self, nu:np.ndarray=None):
        if nu is None:
            nu = self.__nu
        """ 
            generate dust maps given frequencies 
            ==========
            Inputs:
                nu: np.ndarray
                    Array of frequencies to generate the dust maps. If None, use the 
                    frequency passed into the constructor, default None.
            ==========
            Returns:
                maps: np.ndarray
                    Array of dust maps of shape (Nfreq, 3, Npix)
        """
        rj_map0 = self.__dust_map_ref * tcmb2trj(self.nu0_d)
        scale = (nu / self.nu0_d)**self.beta_d * \
            (planck_law(self.temp_d, nu) / planck_law(self.temp_d, self.nu0_d))
        rj_map = np.tile(rj_map0, (nu.shape[0], 1)).reshape(*nu.shape, *rj_map0.shape)
        rj_map *= scale[:, None, None]
        return rj_map * trj2tcmb(nu)[:, None, None]
        
    def __gen_sync_ref(self, 
                    amp_s_ee=56., 
                    amp_s_bb=28., 
                    alpha_s_ee=-0.32, 
                    alpha_s_bb=-0.16) -> None:
        """ generate sync cls and maps at nu0 """
        cl0 = np.zeros_like(self.ell)
        tmp_ell = self.ell
        tmp_ell[0] = 1.
        cl_ee = amp_s_ee * (tmp_ell / 80.)**alpha_s_ee * self.dl2cl
        cl_bb = amp_s_bb * (tmp_ell / 80.)**alpha_s_bb * self.dl2cl
        self.__sync_cls_ref = np.array([cl0, cl_ee, cl_bb, cl0]) #TODO: add temperature
        self.__sync_map_ref = hp.synfast(self.__sync_cls_ref, self.nside, new=True)
        
    def init_sync(self,
                amp_s_ee=9., 
                amp_s_bb=1.6, 
                alpha_s_ee=-0.7, 
                alpha_s_bb=-0.93,
                beta_s=-3,
                nu0_s=23.,
                re_init=False
                ) -> None:
        """
            Initialize / regenerated synchrotron
            ==========
            Parameters:
                amp_s_ee: float 
                    Synchrotron EE amplitude in CMB unit, default 56.
                amp_s_bb: float
                    Synchrotron BB amplitude in CMB unit, default 28.
                alpha_s_ee: float
                    Synchrotron EE Angular power spectral index, default -0.32.
                alpha_s_bb: float
                    Synchrotron BB Angular power spectral index, default -0.16.
                beta_s: float
                    Synchrotron frequency spectral index, default -3.
                nu0_s: float
                    Synchrotron reference frequency, default 23.
                re_init: bool
                    Regenerate synchrotron when function is called, default False.
        """
        if not re_init and self.__sync_map_ref is not None:
            return
        self.__gen_sync_ref(amp_s_ee=amp_s_ee, amp_s_bb=amp_s_bb, alpha_s_ee=alpha_s_ee,
                        alpha_s_bb=alpha_s_bb)
        self.beta_s = beta_s
        self.nu0_s = nu0_s

    def get_sync_theory_cls(self, nu:np.ndarray=None):
        """ 
            generate dust cls given frequencies
            ==========
            Inputs:
                nu: np.ndarray
                    Array of frequencies to generate the power spectra cl. If None, use the 
                    frequency passed into the constructor, default None.
            ==========
            Returns:
                cls: np.ndarray
                    Array of synchrotron cls of shape (Nfreq, 4, Lmax + 1)
        """
        if nu is None:
            nu = self.__nu
        rj_cls0 = self.__sync_cls_ref * tcmb2trj(self.nu0_s)**2
        scale = (nu / self.nu0_s)**(self.beta_s*2)
        rj_cls = np.tile(rj_cls0, (nu.shape[0], 1)).reshape(*nu.shape, *rj_cls0.shape)
        rj_cls *= scale[:, None, None]
        return rj_cls * (trj2tcmb(nu)**2)[:, None, None] 
        
    def get_sync_maps(self, nu:np.ndarray=None):
        """ 
            Generate synchrotron maps given frequencies 
            ==========
            Inputs:
                nu: np.ndarray
                    Array of frequencies to generate the dust maps. If None, use the 
                    frequency passed into the constructor, default None.
            ==========
            Returns:
                maps: np.ndarray
                    Array of dust maps of shape (Nfreq, 3, Npix)
        """
        if nu is None:
            nu = self.__nu
        rj_map0 = self.__sync_map_ref * tcmb2trj(self.nu0_s)
        scale = (nu / self.nu0_s)**self.beta_s
        rj_map = np.tile(rj_map0, (nu.shape[0], 1)).reshape(*nu.shape, *rj_map0.shape)
        rj_map *= scale[:, None, None]
        return rj_map * trj2tcmb(nu)[:, None, None]
    
    def init_cmb(self, A_lens=1., r_tensor=0., re_init=False):
        """
            Initialize / regenerated CMB
            ==========
            Parameters:
                A_lens: float 
                    Lensing amplitude default 1.
                r_tensor: float
                    Tensor r, default 0.
                re_init: bool
                    Regenerate cmb maps when function is called, default False.
        """
        if not re_init and self.cmb_maps is not None:
            return
        cur_dir = os.path.abspath(os.path.dirname(__file__))
        camb_lens_dl = np.loadtxt(cur_dir + "/data/cmb_spec/camb_lens_nobb.dat")
        camb_lens_dl = np.concatenate((np.zeros((1,5)), camb_lens_dl), axis=0)
        camb_r1_dl = np.loadtxt(cur_dir + "/data/cmb_spec/camb_lens_r1.dat")
        camb_r1_dl = np.concatenate((np.zeros((1,5)), camb_r1_dl), axis=0)
        camb_lens_dl = camb_lens_dl[self.ell, 1:].T
        camb_r1_dl = camb_r1_dl[self.ell, 1:].T
        self.cmb_cls = (A_lens * camb_lens_dl + r_tensor * (camb_r1_dl - camb_lens_dl)) * self.dl2cl
        self.cmb_maps = hp.synfast(self.cmb_cls, self.nside, new=True)

    def get_cmb_theory_cls(self):
        """ Get CMB cls after calling init_cmb """
        return self.cmb_cls
    
    def get_cmb_maps(self):
        """ Get CMB maps after calling init_cmb """
        return self.cmb_maps

    def init_white_noise(self, dt:np.ndarray=np.array([100.]), dp:np.ndarray=np.array([100.])):
        """ 
            Initialize parameters for generating white noise
            ==========
            dt: np.ndarray
                array of T sensitivity, in uK-arcmin, default np.array([100])
            dp: np.ndarray
                array of QU sensitivity, in uK-arcmin, there may be a need to divide by 
                $\sqrt{2}$ if polarization sensitivity is given,
                default np.array([100])
        """
        self.noise_dtemp = dt
        self.noise_dpol = dp

    def get_white_noise_maps(self):
        """ Generate a realization of white noise with initalized sensitivities """
        pix_res = hp.nside2resol(self.nside, True)
        npix = hp.nside2npix(self.nside)
        t_n = self.noise_dtemp[:, None] / pix_res \
            * np.random.randn(self.noise_dtemp.shape[0], npix)
        t_p = self.noise_dpol[:, None] / pix_res \
            * np.random.randn(2, self.noise_dtemp.shape[0], npix)
        noise = np.concatenate((t_n[None, :], t_p), axis=0)
        return np.swapaxes(noise, 0, 1)
    
    def pixwin(self):
        """ 
            Get pixel window function for the given nside
            ==========
            Returns:
                pixwin: tuple
                    Tuple of pixel window function (pw_T, pw_P) from healpy
        """
        return hp.pixwin(self.nside, pol=True)
    
    def gauss_beam(self, fwhm):
        """
            Get the gaussian beam window function given fwhm
            ==========
            Inputs:
                fwhm: float
                    fwhm of beam in arcmin
            ==========
            Returns: np.ndarray
                Gaussian beam window function from healpy.
        """
        fwhm_rad = fwhm / 60 / 180 * np.pi
        return hp.gauss_beam(fwhm=fwhm_rad, lmax=self.lmax, pol=True)
    
    def smooth(self, maps, fwhm):
        fwhm_rad = fwhm / 60 / 180 * np.pi
        return hp.smoothing(maps, fwhm=fwhm_rad)