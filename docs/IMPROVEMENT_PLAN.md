# NEOlyzer Improvement Plan

Living document derived from `docs/PROJECT_REVIEW_16Jul26.md` (16 Jul 2026).
Each item is intended to be a small, independently committable change.
Check items off and add notes as work proceeds; reorder freely.

Legend:
- **[DECISION]** — needs a call from Rob before work starts
- **[INVESTIGATE]** — findings first, fix later; the investigation is its own
  deliverable and may change the fix
- Everything else is safe to implement directly

---

## Phase 0 — Repo hygiene (no behavior change)

- [x] 0.1 Delete the empty `{src,data,cache,scripts,docs}` directory at repo
      root (tcsh brace-expansion accident). *(done 2026-07-16)*
- [x] 0.2 Delete 0-byte `data/neo_catalog.db`; remove its `.gitignore` entry.
      *(done 2026-07-16)*
- [x] 0.3 **[DECISION: plain tracked CSV — Rob, 2026-07-16]** Resolved the
      uncommitted tracklets-CSV symlink state: symlink replaced with a plain
      tracked `NEO_discovery_tracklets.csv` (the 10Apr26 data), `_SAVE`
      deleted, dated `NEO_discovery_tracklets_*.csv` pattern gitignored.
- [x] 0.4 Remove the 8 `DEBUG:` prints and stray `traceback.print_exc()`
      from `scripts/setup_database.py`. *(done 2026-07-16)*
- [x] 0.5 **[DECISION: delete — Rob, 2026-07-16]** Deleted stale cache
      backups `positions_jan17_backup.h5`, `positions_mpcorb_sim.h5`,
      `positions_mpcorb_diff.h5` (~4.2 GB reclaimed; `cache/` now holds only
      the active `positions.h5`).
- [x] 0.6 Committed Phase 0 (docs commit + hygiene commit, 2026-07-16).

## Phase 1 — Fix what's broken for users (docs & verify scripts)

- [x] 1.1 Fix `scripts/verify_installation.py:97` — `neo_orbits.db` →
      `asteroids.db`. Also fix the `run_visualizer.sh`/`src/visualizer.py`
      references at `verify_installation.py:247`. *(done 2026-07-16,
      commit a0f7ae9; script now passes all checks end-to-end)*
- [ ] 1.2 **[DECISION]** Retire `scripts/verify_fixes.py` (hardcoded Jan-2026
      bug greps, references retired filenames), or modernize it. Recommended:
      delete; the pytest suite is the regression net now. Update
      CLAUDE.md/README accordingly.
- [ ] 1.3 Add `CHANGELOG.md`; backfill from git tags/commit messages
      (v3.07, v3.08). Going forward, one entry per version bump.
- [ ] 1.4 Single source of truth for the version string: define once (e.g.
      `src/version.py` or a constant in `neolyzer.py`), use it in
      `save_settings` (currently hardcoded `'3.06'` at `neolyzer.py:17383`),
      the About text, and CHANGELOG discipline.
- [ ] 1.5 Document the discovery-tracklets CSV: format (12 columns, dtypes,
      time scale of `avg_mjd_discovery_tracklet`), provenance (psql query
      against mpc_sbn replica on sibyl), and refresh procedure. Check the
      generating `discovery_tracklets.sql` into `scripts/` or `docs/` if
      recoverable.
- [ ] 1.6 Update CLAUDE.md: mention the root-level design docs (ASTRONOMY,
      EPHEMERIS, DATA_FUSION, DELTA_CACHING, etc.), reconcile the Windows
      claim with README's WSL-only stance, drop verify_fixes.py if retired.
- [ ] 1.7 **[DECISION]** Move root-level design `.txt` docs into `docs/`?
      Low cost, breaks any external links/habits. Rob's call.

## Phase 2 — Network robustness

- [x] 2.1 Add timeouts to the two MPC downloads
      (`src/mpc_loader.py:58,100`) — the primary catalog path can currently
      hang forever. *(done 2026-07-16, commit a0f7ae9; (10s connect, 60s
      read) on both)*
- [ ] 2.2 Factor a single shared download helper in `src/` (requests, tqdm
      progress, consistent timeout policy, simple retry-with-backoff).
      Migrate `mpc_loader`, the Gaia download in `setup_database.py:287-302`,
      and `partition_mpcorb.py`'s urllib stack onto it.
- [ ] 2.3 Consolidate the duplicated SSL-fallback ladder
      (`database.py:366-399`, `skyfield_loader.py:145-150`) into the shared
      helper. Make the `verify=False` fallback loud (logger.warning + one-time
      console notice), and add a checksum/size sanity check on downloaded
      `.bsp` files before trusting them.
- [ ] 2.4 **[DECISION]** Whether `verify=False` should require explicit
      opt-in (env var or config flag) vs remain automatic-but-loud. Tension:
      Raspberry Pi users with broken cert stores vs supply-chain integrity.

## Phase 3 — De-duplication (one concern, one owner)

- [ ] 3.1 Remove the naive `unpack_provisional_designation` from
      `src/database.py:223-259`; route callers through
      `designation_utils`. Add a regression test with a high-cycle
      (base-62) designation.
- [ ] 3.2 Single MPCORB fixed-width parser: `partition_mpcorb.py` and
      `diagnose_missing.py` import from `mpc_loader` instead of hardcoding
      offsets (they have already drifted: cols 91–103 vs 92–103 for `a`).
- [ ] 3.3 Add `angular_separation()` / `haversine_separation()` helpers to
      `CoordinateTransformer` (or a new `sky_math` module); replace the ~13
      inline copies in `neolyzer.py` (2166, 3039, 3868, 3911, 4034, 4058,
      4368, 5821, 6098, 6171, 8968, 9177, 9687).
- [ ] 3.4 One named obliquity constant; fix the 23.439 vs 23.43928 mismatch
      (`neolyzer.py:9630` vs `:430,456`) and replace inline
      ecliptic/equatorial math at `neolyzer.py:1721,1729,9720-9728` with
      `CoordinateTransformer` calls.
- [ ] 3.5 Diagnose scripts import shared logic: `diagnose_cln.py` uses
      `database.py`'s CLN constants/functions; `diagnose_sbdb.py` uses the
      shared SBDB fetch. Consider moving all three under
      `scripts/diagnostics/`.
- [ ] 3.6 Deduplicate platform detection (`setup_database.py:43-78` ≈
      `verify_installation.py:31-67`) into one helper.
- [ ] 3.7 Launcher scripts: make `install.sh` the single source (stop
      tracking the generated `run_*.sh`), or stop generating them and track
      only the committed copies. Either way, one source of truth.
- [ ] 3.8 Remove dead code: `CacheBuilder` if truly uncalled (verify
      `scripts/build_cache.py` first); evaluate collapsing scalar
      `OrbitCalculator` into `FastOrbitCalculator` (single caller at
      `neolyzer.py:8247`).

## Phase 4 — Test coverage where bugs actually live

- [ ] 4.1 Parser tests for `mpc_loader.parse_mpc_format` using a few real
      MPCORB lines as fixtures (numbered, provisional, high-cycle, comet-like
      edge cases).
- [ ] 4.2 Headless GUI smoke test in CI: `QT_QPA_PLATFORM=offscreen`, import
      `neolyzer`, instantiate `SkyMapCanvas`, render one frame. Catches
      import-time and first-render regressions cheaply.
- [ ] 4.3 Epoch time-scale regression test (lands with the 5.1 fix): a known
      MPCORB line's epoch must equal the documented TT JD exactly.
- [ ] 4.4 Optional: coverage reporting in CI (informational, no gate yet).

## Phase 5 — Investigations (findings before fixes)

- [x] 5.1 **[INVESTIGATE] TT/UTC epoch bug** (`mpc_loader.py:324-358`).
      *(investigation done 2026-07-16 — see
      `docs/EPOCH_TT_INVESTIGATION_16Jul26.md`)*. Confirmed: MPC docs say
      epoch is ".0 TT"; every DB row carries a 64–69 s excess matching
      ΔT(epoch) exactly. Sky effect: median 0.86″, p99 9″, worst close
      approachers ~125″ — sub-pixel visually, relevant for quantitative
      use. Tests miss it because fixtures hardcode `epoch_jd` and tolerance
      is 0.5°. Proposed fix + migration in the write-up, §6.
      **[DECISION]** approve fix: direct TT→JD in `_unpack_epoch`, one-time
      epoch snap in DB, cache rebuild, regression test (4.3).
- [ ] 5.1a **Kepler solver silently diverges** (found during 5.1; see
      write-up §4). `_solve_kepler_vectorized` and `_solve_kepler` don't
      normalize M and never check convergence; up to ~23 high-e (≥0.81)
      NEOs render at garbage positions on some dates in today's app.
      Fix verified: normalize M into [0, 2π) → 0 failures over 2.5 M
      solves, residual ~1e-16. **[DECISION]** approve fix (normalize M in
      both solvers + convergence warning + regression test).
- [ ] 5.1b **Distance-vs-time plot is dead** (found during 5.1; see
      write-up §5). `neolyzer.py:8267` calls nonexistent
      `calculate_position()`; the AttributeError is swallowed and the plot
      renders all-NaN. Fix by porting to `FastOrbitCalculator`; fold into
      3.8 (delete scalar `OrbitCalculator` — confirmed broken and never
      successfully executed).
- [ ] 5.2 **[INVESTIGATE] Cache invalidation design.** What fingerprint
      belongs in HDF5 metadata (format version, DE kernel name/hash, catalog
      row count + max(updated_at))? How should the app react to a mismatch
      (refuse, warn, rebuild)? Also: real repack for `optimize_cache()`
      (h5repack or copy-rewrite) — currently a no-op flush despite its
      docstring. Design doc first; touches DELTA_CACHING.txt territory.
- [ ] 5.3 **[INVESTIGATE] Settings persistence refactor.** 609 hand-mapped
      lines + four parallel state systems. Evaluate a declarative
      widget-registry (name → getter/setter/default) that drives save,
      restore, factory reset, and script state from one table. Prototype on
      one settings group before committing to the pattern.
- [ ] 5.4 **[INVESTIGATE] QThread workers for network I/O.** Inventory which
      downloads are reachable from the running GUI (SBDB MOID fetch,
      ephemeris download, catalog update), then design one worker pattern
      with progress/finished/error signals. Implement one call site first
      (SBDB fetch is the likeliest freeze).
- [ ] 5.5 **[INVESTIGATE] cython in requirements.txt** — find what needs it;
      remove if nothing does. Also reconcile the PyQt5 fallback with the
      hardcoded `backend_qt5agg` matplotlib import (`neolyzer.py:119-120`)
      and decide whether the PyQt5 path is still supported at all.

## Phase 6 — Larger structural work (each needs its own plan)

- [ ] 6.1 Extract `SkyMapCanvas` (and `CoordinateTransformer`/sky_math) from
      `neolyzer.py` into modules. Precondition for benchmarking and the
      100k-NEO performance push. Do after Phases 3–4 so extraction moves
      tested, deduplicated code.
- [ ] 6.2 Decompose the god-methods: `SettingsDialog.setup_ui` (1,642 lines)
      into per-tab builders; `update_plot` (1,286) and
      `draw_celestial_overlays` (1,080) into per-overlay renderers.
- [ ] 6.3 Config consolidation under `~/.neolyzer/` (move
      `~/.neolyzer_settings.json` in, with backward-compat read of the old
      path).
- [ ] 6.4 Packaging (`pyproject.toml`) — simplifies install and lets scripts
      import `src/` cleanly. **[DECISION]** whether to keep the venv+launcher
      install flow as primary.
- [ ] 6.5 Error-handling sweep: replace bare `except:` in
      `designation_utils.py` and the worst `except: pass` sites in
      `neolyzer.py` with typed excepts + logging. Do incrementally,
      per-subsystem, after the smoke test (4.2) exists.
- [ ] 6.6 100k-NEO performance work (blocked on 6.1 for benchmarkability).

---

## Suggested first session

Phase 0 (with decisions 0.3/0.5 made at session start) + items 1.1, 2.1 —
three small commits that end the dormant/uncommitted state, fix the false
verify failure every new user hits, and remove the hang-forever download
risk. Then 5.1 (the TT/UTC investigation) as the first substantive task.
