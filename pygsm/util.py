import numpy as np
import healpy as hp

def planck_law(T, nu):
    # in RJ unit (not flux so no nu**3)
    # fac = 2 * 6.626e-34 / (3e8)**2
    expo = 6.62607015e-34 * nu * 1e9 / 1.380649e-23 / T
    return nu / (np.exp(expo) - 1)


def trj2tcmb(freq):
    x = 6.62607015e-34 * freq * 1e9 / 2.725 / 1.380649e-23
    return (np.exp(x) - 1)**2 / x**2 / np.exp(x)


def tcmb2trj(freq):
    x = 6.62607015e-34 * freq * 1e9 / 2.725 / 1.380649e-23
    return x**2 * np.exp(x) / (np.exp(x) - 1)**2


def smooth(maps, fwhm):
    smoothed = hp.smoothing(maps, fwhm)
    return smoothed