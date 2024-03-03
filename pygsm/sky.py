import numpy as np
import healpy as hp

from util import *

class Sky:
    def __init__(self, nside=256) -> None:
        self.nside = nside
        self.lmax = nside * 3 - 1
        self.ell = np.arange(self.lmax + 1)
        self.cl2dl = self.ell * (self.ell + 1) / 2 / np.pi
        tmp = np.copy(self.cl2dl)
        tmp[0] = 1
        self.dl2cl = 1 / tmp
        self.dl2cl[0] = 0.

        self.cmb_cls = None
        self.cmb_maps = None
        
        self.temp_d = None
        self.beta_d = None
        self.nu0_d = None
        self.__dust_cls0 = None
        self.__dust_map0 = None
        
        self.__sync_cls0 = None 
        self.__sync_map0 = None

    def __gen_dust0(self, 
                    amp_d_ee=56., 
                    amp_d_bb=28., 
                    alpha_d_ee=-0.32, 
                    alpha_d_bb=-0.16,
                    ):
        """ generate dust cls at nu0 """
        cl0 = np.zeros_like(self.ell)
        dl_ee = amp_d_bb * (self.ell / 80.)**alpha_d_ee * self.dl2cl
        dl_bb = amp_d_ee * (self.ell / 80.)**alpha_d_bb * self.dl2cl
        self.__dust_cls0 = np.array([cl0, dl_ee, dl_bb, cl0]) #TODO: add temperature
        self.__dust_map0 = hp.synfast(self.__dust_cls0, self.nside, new=True)

    def add_dust(self,
                amp_d_ee=56., 
                amp_d_bb=28., 
                alpha_d_ee=-0.32, 
                alpha_d_bb=-0.16,
                temp_d=19.6,
                beta_d=1.54,
                nu0_d=353.
                ):
        self.__gen_dust0(amp_d_ee=amp_d_ee, amp_d_bb=amp_d_bb, alpha_d_ee=alpha_d_ee,
                        alpha_d_bb=alpha_d_bb)
        self.temp_d = temp_d
        self.beta_d = beta_d
        self.nu0_d = nu0_d

    def get_dust_cls(self, nu:np.ndarray):
        """ 
            generate dust cls given frequencies
        """
        rj_cls0 = self.__dust_cls0 * tcmb2trj(self.nu0_d)**2
        scale = ((nu / self.nu0_d)**self.beta_d * \
            (planck_law(self.temp_d, nu) / planck_law(self.temp_d, self.nu0_d)))**2
        rj_cls = np.tile(rj_cls0, (nu.shape[0], 1)).reshape(*nu.shape, *rj_cls0.shape)
        rj_cls *= scale[:, None, None]
        return rj_cls * (trj2tcmb(nu)**2)[:, None, None] 
        
    def get_dust_maps(self, nu:np.ndarray):
        """ generate dust maps given frequencies """
        rj_map0 = self.__dust_map0 * tcmb2trj(self.nu0_d)
        scale = (nu / self.nu0_d)**self.beta_d * \
            (planck_law(self.temp_d, nu) / planck_law(self.temp_d, self.nu0_d))
        rj_map = np.tile(rj_map0, (nu.shape[0], 1)).reshape(*nu.shape, *rj_map0.shape)
        rj_map *= scale[:, None, None]
        return rj_map * trj2tcmb(nu)[:, None, None]
        

        

