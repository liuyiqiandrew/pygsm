# API Reference

Full reference for the `pygsm` public API.

## `pygsm.Sky`

```python
class Sky(nside=256, nu=None)
```

Main class for generating Gaussian CMB and foreground simulations.

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `nside` | `int` | `256` | HEALPix resolution parameter |
| `nu` | `float`, `list`, or `ndarray` | `None` | Observation frequencies in GHz |

**Attributes:**

| Name | Type | Description |
|------|------|-------------|
| `nside` | `int` | HEALPix resolution |
| `lmax` | `int` | Maximum multipole (`3 * nside - 1`) |
| `ell` | `ndarray` | Multipole array `[0, 1, ..., lmax]` |
| `cl2dl` | `ndarray` | Conversion factor `ell*(ell+1)/(2*pi)`, zero at ell=0,1 |
| `dl2cl` | `ndarray` | Inverse of `cl2dl`, zero at ell=0,1 |
| `temp_d` | `float` | Dust temperature [K] (set by `init_dust`) |
| `beta_d` | `float` | Dust frequency spectral index (set by `init_dust`) |
| `nu0_d` | `float` | Dust reference frequency [GHz] (set by `init_dust`) |
| `beta_s` | `float` | Synchrotron frequency spectral index (set by `init_sync`) |
| `nu0_s` | `float` | Synchrotron reference frequency [GHz] (set by `init_sync`) |

---

### Frequency Management

#### `set_freqs(nu)`

Set or update the observation frequency array.

| Parameter | Type | Description |
|-----------|------|-------------|
| `nu` | `float`, `list`, or `ndarray` | Frequencies in GHz |

---

### CMB

#### `init_cmb(A_lens=1.0, r_tensor=0.0, seed=None)`

Initialize the CMB component from pre-computed CAMB spectra.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `A_lens` | `float` | `1.0` | Lensing amplitude (0 = no lensing, 1 = full) |
| `r_tensor` | `float` | `0.0` | Tensor-to-scalar ratio |
| `seed` | `int` or `None` | `None` | Random seed for reproducible maps |

**Raises:** `ValueError` if `lmax` exceeds the bundled CAMB range (ell <= 1950, i.e. `nside` > 650).

Calling again invalidates cached maps.

#### `get_cmb_theory_cls() -> ndarray`

Returns CMB power spectra.

- **Shape:** `(4, lmax+1)` — ordered `[TT, EE, BB, TE]`
- **Units:** CMB uK^2
- **Raises:** `RuntimeError` if `init_cmb()` has not been called

#### `get_cmb_maps() -> ndarray`

Returns CMB maps. Generated lazily on first call and cached.

- **Shape:** `(3, npix)` — ordered `[T, Q, U]`
- **Units:** CMB uK
- **Raises:** `RuntimeError` if `init_cmb()` has not been called

---

### Dust

#### `init_dust(amp_d_ee=56.0, amp_d_bb=28.0, alpha_d_ee=-0.32, alpha_d_bb=-0.16, temp_d=19.6, beta_d=1.54, nu0_d=353.0, seed=None)`

Initialize the dust component with a modified blackbody SED.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `amp_d_ee` | `float` | `56.0` | EE amplitude at ell=80 [CMB uK^2] |
| `amp_d_bb` | `float` | `28.0` | BB amplitude at ell=80 [CMB uK^2] |
| `alpha_d_ee` | `float` | `-0.32` | EE angular power spectral index |
| `alpha_d_bb` | `float` | `-0.16` | BB angular power spectral index |
| `temp_d` | `float` | `19.6` | Dust temperature [K] |
| `beta_d` | `float` | `1.54` | Dust frequency spectral index |
| `nu0_d` | `float` | `353.0` | Reference frequency [GHz] |
| `seed` | `int` or `None` | `None` | Random seed for reproducible maps |

Calling again invalidates cached maps.

#### `get_dust_theory_cls(nu=None) -> ndarray`

Returns dust power spectra scaled to the given frequencies.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nu` | `float`, `list`, `ndarray`, or `None` | `None` | Frequencies in GHz (falls back to stored frequencies) |

- **Shape:** `(nfreq, 4, lmax+1)`
- **Units:** CMB uK^2
- **Raises:** `RuntimeError` if `init_dust()` has not been called; `ValueError` if no frequencies available

#### `get_dust_maps(nu=None) -> ndarray`

Returns dust maps scaled to the given frequencies. Generated lazily on first call and cached.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nu` | `float`, `list`, `ndarray`, or `None` | `None` | Frequencies in GHz (falls back to stored frequencies) |

- **Shape:** `(nfreq, 3, npix)`
- **Units:** CMB uK
- **Raises:** `RuntimeError` if `init_dust()` has not been called; `ValueError` if no frequencies available

---

### Synchrotron

#### `init_sync(amp_s_ee=9.0, amp_s_bb=1.6, alpha_s_ee=-0.7, alpha_s_bb=-0.93, beta_s=-3.0, nu0_s=23.0, seed=None)`

Initialize the synchrotron component with a power-law SED.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `amp_s_ee` | `float` | `9.0` | EE amplitude at ell=80 [CMB uK^2] |
| `amp_s_bb` | `float` | `1.6` | BB amplitude at ell=80 [CMB uK^2] |
| `alpha_s_ee` | `float` | `-0.7` | EE angular power spectral index |
| `alpha_s_bb` | `float` | `-0.93` | BB angular power spectral index |
| `beta_s` | `float` | `-3.0` | Synchrotron frequency spectral index |
| `nu0_s` | `float` | `23.0` | Reference frequency [GHz] |
| `seed` | `int` or `None` | `None` | Random seed for reproducible maps |

Calling again invalidates cached maps.

#### `get_sync_theory_cls(nu=None) -> ndarray`

Returns synchrotron power spectra scaled to the given frequencies.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nu` | `float`, `list`, `ndarray`, or `None` | `None` | Frequencies in GHz (falls back to stored frequencies) |

- **Shape:** `(nfreq, 4, lmax+1)`
- **Units:** CMB uK^2
- **Raises:** `RuntimeError` if `init_sync()` has not been called; `ValueError` if no frequencies available

#### `get_sync_maps(nu=None) -> ndarray`

Returns synchrotron maps scaled to the given frequencies. Generated lazily on first call and cached.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `nu` | `float`, `list`, `ndarray`, or `None` | `None` | Frequencies in GHz (falls back to stored frequencies) |

- **Shape:** `(nfreq, 3, npix)`
- **Units:** CMB uK
- **Raises:** `RuntimeError` if `init_sync()` has not been called; `ValueError` if no frequencies available

---

### White Noise

#### `init_white_noise(sensitivity_t=[100.0], sensitivity_p=[100.0])`

Initialize white noise parameters.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sensitivity_t` | `float`, `list`, or `ndarray` | `[100.0]` | Temperature noise level(s) [uK-arcmin], one per frequency |
| `sensitivity_p` | `float`, `list`, or `ndarray` | `[100.0]` | Polarization noise level(s) [uK-arcmin], one per frequency |

**Raises:** `ValueError` if `sensitivity_t` and `sensitivity_p` have different lengths.

#### `get_white_noise_maps(seed=None) -> ndarray`

Generate a white noise realization. Each call produces a fresh realization.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `seed` | `int` or `None` | `None` | Random seed for reproducibility (uses `numpy.random.default_rng`) |

- **Shape:** `(nfreq, 3, npix)`
- **Units:** CMB uK
- **Raises:** `RuntimeError` if `init_white_noise()` has not been called

---

### Beam and Pixel Window Utilities

#### `pixwin() -> tuple`

Returns the HEALPix pixel window functions `(pw_T, pw_P)` for the current `nside`.

#### `gauss_beam(fwhm) -> ndarray`

Returns a Gaussian beam window function.

| Parameter | Type | Description |
|-----------|------|-------------|
| `fwhm` | `float` | Full width at half maximum [arcmin] |

- **Returns:** Beam window function array evaluated up to `lmax`

---

## `pygsm.util`

Physical constants and helper functions.

### Constants

| Name | Value | Description |
|------|-------|-------------|
| `H_PLANCK` | `6.62607015e-34` | Planck constant [J s] |
| `K_BOLTZMANN` | `1.380649e-23` | Boltzmann constant [J/K] |
| `T_CMB` | `2.725` | CMB monopole temperature [K] |
| `C_LIGHT` | `3e8` | Speed of light [m/s] |

### Functions

#### `planck_law(T, nu) -> ndarray`

Planck spectrum in Rayleigh-Jeans brightness temperature units (without the nu^2 flux prefactor). Used for modified blackbody SED scaling.

| Parameter | Type | Description |
|-----------|------|-------------|
| `T` | `float` | Temperature [K] |
| `nu` | `ndarray` | Frequencies [GHz] |

#### `trj2tcmb(freq) -> ndarray`

Conversion factor from Rayleigh-Jeans to CMB thermodynamic temperature.

| Parameter | Type | Description |
|-----------|------|-------------|
| `freq` | `ndarray` | Frequencies [GHz] |

#### `tcmb2trj(freq) -> ndarray`

Conversion factor from CMB thermodynamic to Rayleigh-Jeans temperature. Inverse of `trj2tcmb`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `freq` | `ndarray` | Frequencies [GHz] |

#### `smooth(maps, fwhm) -> ndarray`

Smooth HEALPix maps with a Gaussian beam via `healpy.smoothing`.

| Parameter | Type | Description |
|-----------|------|-------------|
| `maps` | `ndarray` | HEALPix map(s) |
| `fwhm` | `float` | Beam FWHM [radians] |
