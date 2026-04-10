import numpy as np
import healpy as hp

# Physical constants
H_PLANCK = 6.62607015e-34  # Planck constant [J·s]
K_BOLTZMANN = 1.380649e-23  # Boltzmann constant [J/K]
T_CMB = 2.725  # CMB monopole temperature [K]
C_LIGHT = 3e8  # Speed of light [m/s]


def planck_law(T: float, nu: np.ndarray) -> np.ndarray:
    """Planck law in Rayleigh-Jeans brightness temperature units.

    Computes the frequency-dependent part of the Planck spectrum
    (without the nu^2 flux prefactor), used for modified blackbody
    SED scaling of dust emission.

    Parameters
    ----------
    T : float
        Temperature in Kelvin.
    nu : np.ndarray
        Frequencies in GHz.

    Returns
    -------
    np.ndarray
        Planck spectrum evaluated at each frequency, in RJ units.
    """
    expo = H_PLANCK * nu * 1e9 / K_BOLTZMANN / T
    return nu / (np.exp(expo) - 1)


def trj2tcmb(freq: np.ndarray) -> np.ndarray:
    """Convert Rayleigh-Jeans temperature to CMB thermodynamic temperature.

    Parameters
    ----------
    freq : np.ndarray
        Frequencies in GHz.

    Returns
    -------
    np.ndarray
        Conversion factor: multiply RJ temperature by this to get CMB temperature.
    """
    x = H_PLANCK * freq * 1e9 / T_CMB / K_BOLTZMANN
    return (np.exp(x) - 1) ** 2 / x**2 / np.exp(x)


def tcmb2trj(freq: np.ndarray) -> np.ndarray:
    """Convert CMB thermodynamic temperature to Rayleigh-Jeans temperature.

    Inverse of :func:`trj2tcmb`.

    Parameters
    ----------
    freq : np.ndarray
        Frequencies in GHz.

    Returns
    -------
    np.ndarray
        Conversion factor: multiply CMB temperature by this to get RJ temperature.
    """
    x = H_PLANCK * freq * 1e9 / T_CMB / K_BOLTZMANN
    return x**2 * np.exp(x) / (np.exp(x) - 1) ** 2


def smooth(maps: np.ndarray, fwhm: float) -> np.ndarray:
    """Smooth HEALPix maps with a Gaussian beam.

    Parameters
    ----------
    maps : np.ndarray
        HEALPix map(s) to smooth.
    fwhm : float
        Full width at half maximum of the Gaussian beam, in radians.

    Returns
    -------
    np.ndarray
        Smoothed map(s), same shape as input.
    """
    return hp.smoothing(maps, fwhm)
