# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyGSM is a Gaussian supplement for PySM3 (Python Sky Model) that generates simulated CMB and foreground (dust, synchrotron) sky maps and power spectra for CMB experiment analysis. It produces polarization-only (Q, U Stokes parameters) foreground maps — Stokes I is not implemented. All outputs are in CMB μK units.

## Build, Test & Format

```bash
pip install -e .                              # editable install
conda run -n pygsm pytest tests/ -v           # run tests
conda run -n pygsm black pygsm/ tests/        # format code
```

Dependencies: `numpy`, `healpy`. Dev dependencies: `pytest`, `black`.

## Architecture

The package has a single public class `pygsm.Sky` (defined in `pygsm/sky.py`) and utility functions in `pygsm/util.py`.

### Sky class workflow

1. Construct `Sky(nside, nu)` with HEALPix resolution and frequency array (GHz)
2. Initialize components: `init_cmb()`, `init_dust()`, `init_sync()`, `init_white_noise()`
3. Retrieve spectra (`get_*_theory_cls()`) or maps (`get_*_maps()`)

### Lazy map generation

`init_*()` only computes reference power spectra (cheap). Map generation via `hp.synfast` is deferred to the first `get_*_maps()` call and cached. This separates cheap Cl operations from expensive map realizations. Calling `init_*()` again invalidates cached maps — the next `get_*_maps()` produces a fresh realization.

### Component design

- **CMB**: Loads pre-computed CAMB spectra from `pygsm/data/cmb_spec/`, supports `A_lens` and `r_tensor`. Returns `(4, lmax+1)` cls and `(3, npix)` maps — intentionally no frequency dimension (CMB is frequency-independent, broadcasts onto foreground arrays).
- **Dust**: Modified blackbody SED. Reference Cls at `nu0`, scaled via Planck law.
- **Synchrotron**: Power-law SED. Same reference-then-scale pattern as dust.
- **White noise**: Each `get_white_noise_maps()` call generates a fresh realization.

### Key patterns

- Dust and synchrotron share internal methods (`_gen_foreground_ref_cls`, `_scale_cls`, `_scale_maps`) and differ only in their SED scaling formula.
- `util.py` provides physical constants and RJ/CMB temperature conversion functions.
- Foreground `get_*` methods accept an optional `nu` override; CMB and noise do not (no SED dependence).
