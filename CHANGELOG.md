# Changelog

Notable changes to NEOlyzer. The version number lives in `src/version.py`
(single source of truth); each bump gets an entry here.

## [3.09] — 2026-07-16

Correctness release: three orbital-mechanics bugs found during a full
project audit (`docs/PROJECT_REVIEW_16Jul26.md`,
`docs/EPOCH_TT_INVESTIGATION_16Jul26.md`).

### Fixed
- **Kepler solver divergence.** The solver never normalized the mean
  anomaly and never checked convergence, so up to ~2 dozen
  high-eccentricity (e ≥ 0.8) NEOs rendered at wildly wrong positions on
  some dates, silently. M is now reduced to [0, 2π) (verified to machine
  precision across 2.5 M solves) and non-convergence is logged.
- **TT/UTC epoch bias.** MPCORB packed epochs (defined in TT) were parsed
  as UTC and converted, adding ΔT (~64–69 s) to every stored `epoch_jd` —
  a sky-position error of ~1″ typically, up to ~155″ for close
  approachers. Epochs are now computed directly from the TT calendar
  date; an automatic idempotent migration corrects existing databases,
  and the position cache should be rebuilt (`scripts/build_cache.py`).
- **Distance-vs-time plot** silently rendered all-NaN (it called a
  method that didn't exist and the error was swallowed). Rebuilt on the
  vectorized calculator; calculation failures are now logged.
- `verify_installation.py` checked a database filename that never
  existed (`neo_orbits.db`), guaranteeing a false failure; it also
  referenced retired launcher names.
- MPC catalog downloads had no timeout and could hang setup/update
  forever.

### Changed
- Version string centralized in `src/version.py` (About dialog, saved
  settings, and recorded scripts previously disagreed: 3.08/3.06/3.06).
- Deleted the broken, never-executed scalar `OrbitCalculator`;
  `FastOrbitCalculator` is the single owner of Kepler propagation.
- Retired `scripts/verify_fixes.py` (hardcoded checks for three
  January 2026 bugs; the pytest suite is the regression net).
- Discovery-tracklets CSV restored to a plain tracked file (was
  briefly a symlink to a dated snapshot); dated snapshots gitignored.
- Test suite grown to 147 tests; Kepler tests now exercise the real
  solver rather than a copy of the algorithm.

## [3.08] — 2026-03-18

- Discovery statistics table
- PHA classification fix
- Catalog update fixes
- Coordinate readback fixes; multi-system display in the info popup
- Milky Way background overlay (Gaia EDR3 color/density, Hipparcos
  density) for public outreach (2026-03-06)
- Pytest test suite (134 tests) and GitHub Actions CI (2026-03-13)
- Observable-shading artifact fixes, opacity control, settings
  save/restore gap fixes

## [3.07] — 2026-02-23

- Release readiness improvements (bare excepts replaced, README/WSL
  clarifications, GitHub issue templates)
- Non-discovery overlay, airmass limits, zenith/nadir markers
- Separate east/west solar elongation limits; observable region shading
- Grid Coords dropdown; toolbar/statusbar reorganization
- Ephemeris caching to prevent file-handle exhaustion

## [3.06 and earlier] — 2026-01 … 2026-02

Initial public development: PyQt6 visualization engine, MPC catalog
pipeline, HDF5 position cache with precision tiers, CLN moon-phase
display, discovery-tracklet integration, MOID filtering via JPL SBDB,
alternate catalogs and blinking, cross-platform install scripts.
