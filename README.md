# PyGSM

A lightweight Gaussian sky simulation package for CMB analysis, built as a supplement to [PySM3](https://github.com/galsci/pysm). PyGSM generates polarized CMB, dust, and synchrotron power spectra and HEALPix maps with frequency-dependent SED scaling, plus white noise realizations.

## Features

- **CMB**: Lensed and tensor B-mode spectra from pre-computed CAMB data, with adjustable lensing amplitude and tensor-to-scalar ratio
- **Dust**: Modified blackbody SED with configurable temperature, spectral index, and angular power spectrum
- **Synchrotron**: Power-law SED with configurable spectral index and angular power spectrum
- **White noise**: Per-frequency T/Q/U noise realizations from sensitivity inputs
- **Lazy map generation**: Power spectra are computed cheaply at init time; expensive `healpy.synfast` map generation is deferred until first use and cached
- **Seed control**: Optional random seeds for reproducible realizations

## Installation

```bash
git clone https://github.com/liuyiqiandrew/pygsm.git
cd pygsm
pip install -e .
```

Requires Python >= 3.8, `numpy`, and `healpy`.

## Quick Start

```python
import numpy as np
import pygsm

# Create a Sky object with HEALPix nside and observation frequencies (GHz)
freqs = np.array([27.0, 39.0, 93.0, 145.0, 225.0, 280.0])
sky = pygsm.Sky(nside=256, nu=freqs)

# Initialize components
sky.init_cmb(A_lens=1.0, r_tensor=0.0)
sky.init_dust()
sky.init_sync()
sky.init_white_noise(sensitivity_t=100.0, sensitivity_p=100.0)

# Get power spectra (cheap, no map generation)
cmb_cls = sky.get_cmb_theory_cls()        # (4, lmax+1)
dust_cls = sky.get_dust_theory_cls()       # (nfreq, 4, lmax+1)
sync_cls = sky.get_sync_theory_cls()       # (nfreq, 4, lmax+1)

# Get maps (generates on first call, cached afterward)
cmb_maps = sky.get_cmb_maps()             # (3, npix)
dust_maps = sky.get_dust_maps()           # (nfreq, 3, npix)
sync_maps = sky.get_sync_maps()           # (nfreq, 3, npix)
noise_maps = sky.get_white_noise_maps()   # (nfreq, 3, npix)

# Combine: CMB broadcasts across frequencies
total = dust_maps + sync_maps + noise_maps + cmb_maps
```

## Components

### CMB

Combines pre-computed CAMB lensed and tensor spectra:

$$C_\ell = A_\text{lens} \cdot C_\ell^{\text{lens}} + r \cdot (C_\ell^{r=1} - C_\ell^{\text{lens}})$$

```python
sky.init_cmb(A_lens=1.0, r_tensor=0.0, seed=42)
cls = sky.get_cmb_theory_cls()   # (4, lmax+1): [TT, EE, BB, TE]
maps = sky.get_cmb_maps()        # (3, npix):   [T, Q, U]
```

CMB is frequency-independent: maps have shape `(3, npix)` and broadcast directly onto `(nfreq, 3, npix)` foreground arrays. The bundled CAMB spectra go up to ell = 1950, so `nside` must be <= 650.

### Dust

Modified blackbody SED at reference frequency `nu0`:

$$D_\ell(\nu) = A \left(\frac{\ell}{80}\right)^\alpha \left(\frac{\nu}{\nu_0}\right)^{2\beta} \left(\frac{B_\nu(T_d)}{B_{\nu_0}(T_d)}\right)^2$$

```python
sky.init_dust(
    amp_d_ee=56.0,       # EE amplitude at ell=80 [CMB uK^2]
    amp_d_bb=28.0,       # BB amplitude at ell=80 [CMB uK^2]
    alpha_d_ee=-0.32,    # EE spectral index
    alpha_d_bb=-0.16,    # BB spectral index
    temp_d=19.6,         # dust temperature [K]
    beta_d=1.54,         # frequency spectral index
    nu0_d=353.0,         # reference frequency [GHz]
    seed=42,
)
cls = sky.get_dust_theory_cls(nu=freqs)   # (nfreq, 4, lmax+1)
maps = sky.get_dust_maps(nu=freqs)        # (nfreq, 3, npix)
```

### Synchrotron

Power-law SED at reference frequency `nu0`:

$$D_\ell(\nu) = A \left(\frac{\ell}{80}\right)^\alpha \left(\frac{\nu}{\nu_0}\right)^{2\beta}$$

```python
sky.init_sync(
    amp_s_ee=9.0,        # EE amplitude at ell=80 [CMB uK^2]
    amp_s_bb=1.6,        # BB amplitude at ell=80 [CMB uK^2]
    alpha_s_ee=-0.7,     # EE spectral index
    alpha_s_bb=-0.93,    # BB spectral index
    beta_s=-3.0,         # frequency spectral index
    nu0_s=23.0,          # reference frequency [GHz]
    seed=42,
)
cls = sky.get_sync_theory_cls(nu=freqs)   # (nfreq, 4, lmax+1)
maps = sky.get_sync_maps(nu=freqs)        # (nfreq, 3, npix)
```

### White Noise

Converts per-frequency sensitivities (uK-arcmin) to white noise map realizations. Each call to `get_white_noise_maps()` produces a fresh realization.

```python
sens_t = np.array([100.0, 100.0, 50.0, 50.0, 100.0, 100.0])
sens_p = sens_t * np.sqrt(2)

sky.init_white_noise(sensitivity_t=sens_t, sensitivity_p=sens_p)
noise = sky.get_white_noise_maps(seed=42)   # (nfreq, 3, npix)
```

## Output Shapes and Conventions

| Component | Power spectra | Maps |
|-----------|--------------|------|
| CMB | `(4, lmax+1)` | `(3, npix)` |
| Dust | `(nfreq, 4, lmax+1)` | `(nfreq, 3, npix)` |
| Synchrotron | `(nfreq, 4, lmax+1)` | `(nfreq, 3, npix)` |
| White noise | — | `(nfreq, 3, npix)` |

- Power spectra are ordered `[TT, EE, BB, TE]`
- Maps are ordered `[T, Q, U]`
- Foregrounds are **polarization-only**: TT and TE spectra are zero, T maps are zero. Stokes I is not implemented
- All values are in **CMB uK** (maps) or **CMB uK^2** (spectra)
- `lmax = 3 * nside - 1`, `npix = 12 * nside^2`

## Frequency Handling

Frequencies can be set in three ways, all accepting floats, lists, or arrays:

```python
# 1. At construction
sky = pygsm.Sky(nside=256, nu=150.0)

# 2. Via set_freqs()
sky.set_freqs([93.0, 145.0, 225.0])

# 3. As an override in get methods (foregrounds only)
cls = sky.get_dust_theory_cls(nu=353.0)
```

CMB and white noise are frequency-independent and do not accept `nu` arguments.

## Lazy Generation and Re-initialization

`init_*()` computes reference power spectra (cheap). Maps are generated on the first `get_*_maps()` call and cached. To get a new random realization, call `init_*()` again — this invalidates the cache and the next `get_*_maps()` produces a fresh realization.

```python
sky.init_dust(seed=1)
maps_a = sky.get_dust_maps()    # generates and caches
maps_b = sky.get_dust_maps()    # returns cached (same as maps_a)

sky.init_dust(seed=2)           # invalidates cache
maps_c = sky.get_dust_maps()    # generates new realization
```

## Utilities

The `Sky` object provides convenience methods tied to its `nside` and `lmax`:

```python
sky.cl2dl          # Cl -> Dl conversion factor array
sky.dl2cl          # Dl -> Cl conversion factor array
sky.gauss_beam(fwhm=10.0)  # Gaussian beam window function (fwhm in arcmin)
sky.pixwin()       # HEALPix pixel window functions (T, P)
```

`pygsm.util` provides physical helper functions:

```python
from pygsm.util import planck_law, trj2tcmb, tcmb2trj, smooth
```

## Example

See [`example/pygsm_example.ipynb`](example/pygsm_example.ipynb) for a walkthrough with plots.

## License

MIT License. See [LICENSE.txt](LICENSE.txt).

## Contact

Yiqi (Andrew) Liu — andrew.liu@princeton.edu
