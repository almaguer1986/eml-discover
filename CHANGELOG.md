# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.0] — 2026-04-25 — Registry expansion to 53 formulas + CI

### Added
- **CI workflow** (`.github/workflows/ci.yml`) — pytest + mypy on
  ubuntu / macos / windows × Python 3.10 / 3.11 / 3.12 / 3.13.
- **35 new registry entries** lifting the total from 17 → 53
  formulas across 13 domains:
  - **statistics**: Cauchy PDF, exponential PDF, lognormal PDF,
    chi-square PDF (k=2)
  - **ml**: Mish, ELU (alpha=1), sigmoid derivative
  - **trigonometry**: double-angle sin / cos, tangent half-angle
  - **hyperbolic**: double-angle sinh, tanh half-angle
  - **physics**: Stefan-Boltzmann, kinetic energy, Coulomb's law,
    Newton's gravitation, Lorentz factor, de Broglie wavelength,
    Schwarzschild radius, heat kernel (1D), Planck radiance kernel,
    Maxwell-Boltzmann speed PDF kernel
  - **finance**: continuous compound interest, discrete present
    value, perpetuity, annuity factor
  - **info-theory**: Shannon entropy, KL divergence, cross-entropy
    (single-term forms each)
  - **calculus**: tanh derivative, erf, sinc
  - **biology**: logistic growth
  - **electrical-engineering**: RC voltage decay
  - **physics (more)**: power-law decay
- 5 new test cases covering Lorentz factor, KL divergence, RC
  decay, perpetuity, and a domain-coverage sanity check.

### Fixed
- mypy strict: `_try_match` now declares its `rename` parameter
  type and casts the `sp.simplify == 0` result to `bool` so the
  type checker accepts the return.

## [0.1.0] — 2026-04-25 — Initial release

Two-stage formula identifier (Pfaffian-fingerprint pre-filter +
SymPy variable-rename + simplify equivalence). 17-formula curated
registry across statistics, ML activations, trig/hyperbolic
identities, physics, and finance.
