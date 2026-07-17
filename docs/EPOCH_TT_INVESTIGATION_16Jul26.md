# Epoch TT/UTC Investigation — 16 July 2026

Investigation for plan item 5.1 (`docs/IMPROVEMENT_PLAN.md`), following the
16 Jul 2026 project review. **No code changes made** — findings and a
proposed fix, for Rob's approval. Two additional bugs were discovered along
the way (§4, §5).

---

## 1. The bug, confirmed

`src/mpc_loader.py:_unpack_epoch()` decodes the MPCORB packed epoch
(e.g. `K261A`) into a calendar date, then converts it to a Julian Date via:

```python
t = ts.utc(year, month, day)   # interprets the date as UTC
return t.tt                    # returns the TT JD of that UTC instant
```

MPC's MPCORB format documentation states the epoch is **"(in packed form,
.0 TT)"** — the calendar date is already in Terrestrial Time. Interpreting
it as UTC and converting adds ΔT = TT−UTC to every stored `epoch_jd`:
about **64.2 s for 2000-era epochs, 69.2 s for current ones**.

**Database evidence** (definitive): a `.0 TT` epoch is a JD ending in
exactly `.5`. Every one of the 41,441 rows in `asteroids.db` instead ends
in `.50074`–`.50080`; the fractional excess across the catalog is
64.18–69.18 s — exactly ΔT for the corresponding epoch dates. The stored
excess *is* the bug, row by row.

## 2. Quantified effect on sky positions

Method: computed geocentric RA/Dec for the full catalog (41,439 elliptic
orbits) with `FastOrbitCalculator` at JD 2461238.0 (≈ 2026-07-16 TT), using
stored epochs vs corrected epochs (excess removed). Script preserved at the
session scratchpad (`epoch_error_measure.py`); statistics exclude two
objects whose separations were Kepler-solver artifacts (§4):

| statistic | error |
|---|---|
| median | 0.86″ |
| p95 | 3.5″ |
| p99 | 9.1″ |
| worst close approachers (d < 0.08 AU) | 45–125″ |
| 1 display pixel, full-sky map (~2000 px) | ~648″ |

Interpretation: **visually negligible** — even the worst real case
(~125″ for 2022 WG15-class close approachers) is a fifth of a display
pixel. **Scientifically relevant** for any quantitative use: Horizons
comparisons, discovery-circumstance matching, or future features that
report coordinates (the info popup shows RA/Dec to arcsecond precision).
The error scales as (object's apparent motion) × ΔT, so it is largest
exactly where NEO work cares most — fast movers during close approaches.

## 3. Why the existing tests never caught it

`tests/test_orbit_positions.py` fixtures hardcode `epoch_jd` values
directly (e.g. `2460000.5`), so `_unpack_epoch` is never on the tested
path; and `POSITION_TOLERANCE_DEG = 0.5` (1800″) would absorb the error
even if it were. Both facts confirmed by inspection.

## 4. Collateral discovery A — Kepler solver silently diverges (worse bug)

While measuring §2, two objects showed 40–49° "errors" — physically
impossible for a 69 s epoch shift. Root cause, confirmed by direct
experiment: **`_solve_kepler_vectorized` (`orbit_calculator.py:370`) does
not normalize the mean anomaly** before solving. It seeds Newton's method
with E=π for e ≥ 0.8, runs exactly 30 iterations, and **returns the result
with no convergence check**. With M several radians outside [0, 2π) and
high eccentricity, Newton wanders chaotically; e.g. for K11E47L the
returned E had residual **36 rad** — a garbage position, silently plotted.

Prevalence across the live catalog (residual > 10⁻⁶ rad):

| date (JD) | non-converged objects |
|---|---|
| 2460500 | 8 |
| 2461238 | 0 |
| 2462000 | 23 |
| 2465000 | 23 |

All failures have e ≥ 0.81. So **on any given date, up to ~2 dozen
high-eccentricity NEOs may render at wildly wrong positions in today's
app**, date-dependently and with no warning. This is independent of, and
more serious than, the epoch bug.

**Fix, verified:** normalizing M into [0, 2π) before solving
(`M = np.mod(M, 2*np.pi)`) eliminated every failure — 0 non-converged over
60 dates × 41,441 objects (2.5 M solves), worst residual 8.9×10⁻¹⁶ rad.
The scalar `_solve_kepler` (`orbit_calculator.py:146`) has the identical
flaw and should get the identical fix, plus a convergence check/log in both.

## 5. Collateral discovery B — the distance-vs-time plot is dead

The scalar `OrbitCalculator`'s single caller, the distance-time plot
(`neolyzer.py:8267`), calls `calculator.calculate_position(...)` — **a
method that does not exist** (the class defines `elements_to_position`).
Every call raises `AttributeError`, swallowed by the surrounding
`except Exception: distances.append(np.nan)`, so the plot silently renders
all-NaN. The feature has been dead for an unknown period — a textbook case
of the review's `except: pass` finding.

Moreover the scalar path is internally broken anyway: tested against
`FastOrbitCalculator` at the fixture epoch itself, it disagrees by tens of
degrees and returns distance in what appears to be km rather than AU (its
mean motion also omits the Gaussian constant k). Since it never
successfully executes, this strengthens plan item 3.8: **delete
`OrbitCalculator` and rebuild the distance-time plot on
`FastOrbitCalculator`** rather than repairing the scalar code.

## 6. Proposed fix and migration (awaiting approval)

1. **`_unpack_epoch`**: compute the JD directly from the TT calendar date
   with plain arithmetic (standard Gregorian→JD formula; the result ends
   in exactly `.5`). This also removes the current
   `from skyfield.api import load; ts = load.timescale()` executed *per
   parsed line* — a hidden performance cost in catalog imports.
2. **Database migration**: one-time snap of stored epochs to the true TT
   value — `epoch_jd → floor(epoch_jd − 0.5) + 0.5` — for `asteroids` and
   `alternate_asteroids`. Exact because all MPCORB epochs are `.0 TT` and
   the observed excess is < 0.001 d. Avoids a full catalog re-import.
3. **Rebuild the position cache** (`scripts/build_cache.py`): the 18 GB
   `positions.h5` embeds positions computed from the biased epochs.
4. **Kepler solver fix** (can land first, independently): normalize M in
   both solvers, add a convergence check with `logger.warning`.
5. **Tests** (plan 4.3): packed-epoch → exact TT JD (e.g. `K261A` →
   2461041.5); a high-e/large-M regression case for the solver.
6. **Distance-time plot**: separate small fix (plan item added) — port to
   `FastOrbitCalculator`; delete scalar `OrbitCalculator` (plan 3.8).

Suggested order: solver fix (affects today's rendering) → epoch fix +
migration + cache rebuild (one commit + one rebuild session) → plot repair.
