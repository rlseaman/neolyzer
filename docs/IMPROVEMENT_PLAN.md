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
- [x] 1.2 **[DECISION: delete — per plan recommendation]** Retired
      `scripts/verify_fixes.py`. *(done 2026-07-16, commit 133a330)*
- [x] 1.3 `CHANGELOG.md` added; 3.07/3.08 backfilled from git history,
      3.09 documents the correctness fixes. *(done 2026-07-16, b1d3717)*
- [x] 1.4 Version single-sourced in `src/version.py` (`__version__ =
      "3.09"`); About dialog, saved settings, and recorded scripts all
      read it (previously 3.08/3.06/3.06). *(done 2026-07-16, b1d3717)*
- [x] 1.5 `docs/DISCOVERY_TRACKLETS.md`: 12-column format, provenance
      (all_neas_from_nea_txt_v4.sql in the sibling NEA_discovery_tracklets
      project, run against mpc_sbn on sibyl), refresh procedure. SQL left
      in its home project (single source of truth) rather than copied.
      *(done 2026-07-16, 5907de9)*
- [x] 1.6 CLAUDE.md updated: three-table DB reality, Windows=WSL-only,
      verify_fixes removed, docs/ contents and root design docs listed,
      test_mpc_loader added. *(done 2026-07-16, 133a330)*
- [ ] 1.7 **[DECISION]** Move root-level design `.txt` docs into `docs/`?
      Low cost, breaks any external links/habits. Rob's call.

## Phase 2 — Network robustness

- [x] 2.1 Add timeouts to the two MPC downloads
      (`src/mpc_loader.py:58,100`) — the primary catalog path can currently
      hang forever. *(done 2026-07-16, commit a0f7ae9; (10s connect, 60s
      read) on both)*
- [x] 2.2 Shared download helper: `src/net_utils.py` (`http_get` +
      `download_file`: one timeout policy, retry with backoff, tqdm,
      atomic .part+rename). All six HTTP call sites migrated (MPC NEA/
      MPCORB, JPL ephemeris, JPL SBDB, Gaia maps, MPCORB.gz partition).
      9 new tests. *(done 2026-07-16, commit 71887ea)*
- [x] 2.3 SSL-fallback ladder consolidated into `net_utils`; unverified
      rung logs a prominent warning; `download_file(min_size=...)` size
      sanity check applied to `.bsp` and all data downloads (true
      checksums unavailable — JPL doesn't publish them for bsp files).
      *(done 2026-07-16, commit 71887ea)*
- [ ] 2.4 **[DECISION]** Whether `verify=False` should require explicit
      opt-in (env var or config flag) vs remain automatic-but-loud. Tension:
      Raspberry Pi users with broken cert stores vs supply-chain integrity.

## Phase 3 — De-duplication (one concern, one owner)

- [x] 3.1 Naive `unpack_provisional_designation` removed from database.py
      (it mis-unpacked every input, even its docstring examples); routes
      through `designation_utils`; regression tests added. *(2026-07-16)*
- [x] 3.2 Single MPCORB parser: `mpc_loader.parse_mpcorb_line` is
      canonical; partition_mpcorb imports it (~90 duplicate lines gone);
      diagnose_missing offsets corrected. *(2026-07-16)*
- [x] 3.3 `src/sky_math.py`: `cos_angular_separation()` +
      `angular_separation_deg()`; all ~15 inline copies in neolyzer.py
      replaced; 10 new tests. *(2026-07-16)*
- [x] 3.4 `OBLIQUITY_J2000_DEG/RAD` in sky_math; 23.439-vs-23.43928
      mismatch fixed; all 7 inline obliquity literals in neolyzer.py and
      orbit_calculator.py now use the shared constant. *(2026-07-16)*
- [x] 3.5 Diagnose scripts moved to `scripts/diagnostics/`; diagnose_cln
      imports CLN constants from database; diagnose_sbdb uses
      net_utils.http_get; .gitignore `diagnostics/` anchored to root.
      *(2026-07-16)*
- [x] 3.6 Platform detection deduplicated into `src/platform_info.py`.
      *(2026-07-16)*
- [x] 3.7 Launchers: committed `run_*.sh` are the single source;
      install.sh no longer regenerates them (chmod + existence check
      only). *(2026-07-16)*
- [x] 3.8 Remove dead code. *(done 2026-07-16, commit 59dcf3d: scalar
      `OrbitCalculator` deleted — it was internally broken and its only
      caller invoked a nonexistent method, see 5.1b. NOTE: `CacheBuilder`
      is NOT dead — `scripts/build_cache.py` uses it; the review's "no
      callers in src/" was literally true but wrong in conclusion.)*

## Phase 4 — Test coverage where bugs actually live

- [x] 4.1 Parser tests with real NEA.txt fixture lines (Eros full-field,
      provisional, header/short-line, a-from-mean-motion fallback), in
      `tests/test_mpc_loader.py`. *(2026-07-16)*
- [x] 4.2 Headless GUI smoke test (`tests/test_gui_smoke.py`): module
      import, offscreen SkyMapCanvas construction + draw,
      CoordinateTransformer round-trip; CI gains offscreen Qt platform
      and Linux Qt system libraries. *(2026-07-16)*
- [x] 4.3 Epoch time-scale regression test. *(done 2026-07-16 with the 5.1
      fix: `tests/test_mpc_loader.py` — known packed epochs → exact TT JD,
      `.0 TT` invariant, malformed-input default)*
- [x] 4.4 Coverage reporting in CI (informational, no gate). *(2026-07-16)*

## Phase 5 — Investigations (findings before fixes)

- [x] 5.1 **[INVESTIGATE] TT/UTC epoch bug** (`mpc_loader.py:324-358`).
      *(investigation done 2026-07-16 — see
      `docs/EPOCH_TT_INVESTIGATION_16Jul26.md`)*. Confirmed: MPC docs say
      epoch is ".0 TT"; every DB row carries a 64–69 s excess matching
      ΔT(epoch) exactly. Sky effect: median 0.86″, p99 9″, worst close
      approachers ~125″ — sub-pixel visually, relevant for quantitative
      use. Tests miss it because fixtures hardcode `epoch_jd` and tolerance
      is 0.5°. Proposed fix + migration in the write-up, §6.
      **[DECISION: approved — Rob, 2026-07-16]** Fix landed (commit
      f795386): direct TT→JD in `_unpack_epoch` (verified identical to
      skyfield `ts.tt()`), idempotent DB migration snapped 41,441 primary
      + 193,685 alternate epochs to `.0 TT`, regression tests in new
      `tests/test_mpc_loader.py` (covers 4.3), cache rebuilt from scratch.
- [x] 5.1a **Kepler solver silently diverges** (found during 5.1; see
      write-up §4). Solvers didn't normalize M and never checked
      convergence; up to ~23 high-e (≥0.81) NEOs rendered at garbage
      positions on some dates. *(fixed 2026-07-16, commit 921cfd3:
      normalize M into [0, 2π) + convergence warning; regression tests;
      tests now import the real solver instead of local copies)*
- [x] 5.1b **Distance-vs-time plot is dead** (found during 5.1; see
      write-up §5). `neolyzer.py` called nonexistent
      `calculate_position()`; the AttributeError was swallowed and the
      plot rendered all-NaN. *(fixed 2026-07-16, commit 59dcf3d: ported
      to `FastOrbitCalculator`, failure now logged; verified 365 finite
      distances for 433 Eros)*
- [x] 5.2 Cache invalidation design written:
      `docs/CACHE_INVALIDATION_DESIGN.md` (fingerprint attrs, mismatch
      behavior, real repack via copy-rewrite, delta-rebuild hook; open
      questions for Rob on size budget and reference-JD snapping).
      **[DECISION]** approve design → implement in the listed order.
      *(investigation done 2026-07-16)*
- [x] 5.3 Settings registry design written:
      `docs/SETTINGS_REGISTRY_DESIGN.md` (one table drives save/restore/
      reset/script-state; prototype scope = Milky Way group; QSettings
      rejected with rationale). **[DECISION]** approve → prototype.
      *(investigation done 2026-07-16)*
- [x] 5.4 Worker-threads design written:
      `docs/GUI_WORKER_THREADS_DESIGN.md`. Inventory corrected the plan's
      assumption: the SBDB fetch is NOT GUI-reachable (scripts only) —
      the real freeze is the ephemeris download (de441 = 3.5 GB on the
      GUI thread). Design: one generic IoWorker + progress callback on
      net_utils.download_file, ephemeris-switch flow first.
      **[DECISION]** approve → implement. *(investigation done 2026-07-16)*
- [x] 5.5 Dependency audit done: cython removed (no .pyx, no build step,
      no import anywhere); matplotlib backend switched to binding-
      agnostic QtAgg (was hardcoded Qt5 shim while running PyQt6); the
      PyQt5 fallback path retained and now actually consistent with the
      backend import. *(2026-07-16)*

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
