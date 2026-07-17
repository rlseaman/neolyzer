# NEOlyzer Project Review — 16 July 2026

Full-project audit: code, dependencies, data standards, docs, infrastructure,
and operating procedures. Conducted by Claude Code at Rob Seaman's request.

- **Repo state at review:** branch `main` @ `73a7849` (committed 2026-03-24),
  in sync with `origin/main`; uncommitted working-tree changes from 2026-04-10.
- **Version:** v3.08
- **Companion document:** `docs/IMPROVEMENT_PLAN.md` (living plan derived from
  this report; the report itself is a snapshot and should not be edited).

---

## Summary

NEOlyzer is a mature, working tool: PyQt6 + matplotlib desktop app displaying
40,000+ NEOs with animation, four coordinate systems, four projections,
discovery-circumstance integration, and Milky Way backgrounds, backed by
SQLite, an HDF5 position cache, and Skyfield/JPL ephemerides. v3.08, 134
passing tests, CI on GitHub Actions (Ubuntu + macOS, Python 3.10/3.12).

**Status: mature but dormant.** Last commit 2026-03-24; the working tree has
sat uncommitted since 2026-04-10. The uncommitted change matters:
`data/NEO_discovery_tracklets.csv` was replaced with a symlink to
`NEO_discovery_tracklets_10Apr26.csv`, which is untracked. Committed as-is, a
fresh clone gets a dangling symlink (and symlinks misbehave on Windows
checkouts). Discovery data is ~3 months stale relative to the review date.

---

## Correctness bugs found

Ranked most significant first:

1. **MPCORB epoch time-scale error (~69 s).** `src/mpc_loader.py:324-358`
   builds the epoch with `ts.utc(...)` then returns `t.tt`. MPCORB epochs are
   defined in TT, so every `epoch_jd` in the database carries a spurious ΔT
   offset (~69 s now, larger historically). The rest of the code consistently
   treats stored JDs as TT, making this conversion the odd one out. Likely
   sub-pixel at display precision for most objects, but systematic, and grows
   for fast movers near perigee. **Needs investigation before fixing** (see
   plan): quantify effect, check whether Horizons-comparison tests absorb it,
   and note that fixing it invalidates the position cache.

2. **`scripts/verify_installation.py:97` checks the wrong database file**
   (`data/neo_orbits.db`; the real file is `data/asteroids.db`). README tells
   users to run this script, so every user following the docs gets a false
   "Database not found" failure. Both verify scripts also reference the
   retired `run_visualizer.sh` / `src/visualizer.py` names
   (`verify_installation.py:247`, `verify_fixes.py:185`).

3. **Duplicate, divergent designation unpackers.** `src/designation_utils.py`
   has the correct base-62 implementation; `src/database.py:223-259` contains
   a second, naive `unpack_provisional_designation` that does not handle
   base-62 cycle counts. Code paths using the naive one will mis-unpack
   high-cycle designations.

4. **Inconsistent obliquity constant.** `CoordinateTransformer` uses
   23.43928° (`neolyzer.py:430,456`); the heliocentric chart uses 23.439°
   (`neolyzer.py:9630`). The sky map and heliocentric chart can subtly
   disagree.

5. **MPC parsing column offsets differ across three independent parsers** —
   `src/mpc_loader.py`, `scripts/partition_mpcorb.py:77-174`, and
   `diagnose_missing.py:20-67` each hardcode MPCORB fixed-width offsets, and
   have already drifted (semi-major axis columns 91–103 vs 92–103 between two
   of them).

---

## Unmitigated risks

- **TLS verification silently disabled as a fallback.** `database.py:397` and
  `skyfield_loader.py:150` fall back to `requests.get(..., verify=False)`
  after SSL errors, with urllib3 warnings suppressed and no checksum on the
  downloaded ephemeris or orbit data. A network adversary or corrupted mirror
  could feed the app bad ephemerides undetected. Minimum fix: log loudly and
  checksum `.bsp` downloads; better: make the insecure fallback opt-in.
- **MPC catalog downloads have no timeout** (`mpc_loader.py:58,100`) and no
  retries anywhere. This is the primary data path for setup and every catalog
  update — a stalled MPC connection hangs forever. Timeout policy is
  inconsistent across endpoints (none / 60 s / 120 s / 300 s).
- **All network and heavy I/O runs on the GUI thread.** Zero uses of
  QThread/QRunnable in `src/`; SBDB fetches (timeout 120 s) and ephemeris
  downloads (300 s) can freeze the window for minutes. Also the single
  biggest UX improvement available.
- **The 18 GB position cache has no invalidation contract.**
  `cache_manager.py` stores a reference JD and creation date but no format
  version, no ephemeris-kernel fingerprint, no catalog hash — if
  `asteroids.db` or the DE kernel changes, nothing detects the mismatch.
  `optimize_cache()` (`cache_manager.py:324-329`) only calls `f.flush()`
  despite its docstring claiming it repacks; deleted HDF5 datasets never
  reclaim space, so `positions.h5` only grows.
- **Ad-hoc schema migrations.** `database.py:833-899` does additive
  `ALTER TABLE` with column-existence checks — no version table, no ordering,
  failures only logged. Fine for a single-user desktop app; brittle the
  moment another user's database diverges.
- **Bus factor and silent failure.** 55 `except: pass` sites in `neolyzer.py`
  and bare `except:` around core designation parsing
  (`designation_utils.py:282,315,322,364,375,384`) swallow data-quality
  problems invisibly. Combined with an 18,000-line monolith and a single
  maintainer, field reports from other users would be hard to debug.

---

## Duplicated / overlapping functionality

The support modules are well-factored (all DB access goes through
`DatabaseManager`; no Kepler math re-rolled in the GUI), but real duplication
exists at the edges:

- Angular-separation formula copy-pasted ~13 times across `neolyzer.py`
  (spherical law of cosines at 2166, 3868, 4368, 5821, 6098, 8968, 9177,
  9687; haversine at 3039, 3911, 4034, 4058, 6171).
- Ecliptic/equatorial rotation math inlined at `neolyzer.py:1721,1729,
  9720-9728` despite `CoordinateTransformer` existing for exactly this.
- Two HTTP download stacks (`requests`+tqdm in `mpc_loader`; hand-rolled
  `urllib` in `partition_mpcorb.py:233-276`).
- SSL-fallback ladder duplicated verbatim (`database.py:366-399` ≈
  `skyfield_loader.py:145-150`).
- Platform detection copy-pasted (`setup_database.py:43-78` ≈
  `verify_installation.py:31-67`).
- The three `diagnose_*.py` scripts reimplement `src/` logic (CLN constants,
  MPC parsing, SBDB fetch) instead of importing it — silent drift hazards.
- `run_neolyzer.sh`/`run_setup.sh` exist both committed and regenerated by
  heredoc in `install.sh:688-723` — two sources of truth.
- Scalar `OrbitCalculator` vs `FastOrbitCalculator` implement Kepler
  propagation twice; the scalar class has one caller (`neolyzer.py:8247`).
  `CacheBuilder` (`cache_manager.py:332`) appears to have no callers.

---

## Gaps

- **Testing:** `mpc_loader.py` — the most bug-prone code (fixed-width
  parsing, all downloads) — has zero tests. The GUI has no test, not even a
  headless import smoke test. Scripts are untested. No coverage measurement
  or lint gate in CI.
- **Docs:** No CHANGELOG despite versioned releases; the version string lives
  in three places with no single source of truth — and `save_settings` still
  writes `'3.06'` (`neolyzer.py:17383`) at v3.08. No CONTRIBUTING. The
  substantive design docs (ASTRONOMY.txt, EPHEMERIS.txt, DATA_FUSION.txt,
  DELTA_CACHING.txt, etc.) sit at repo root while `docs/` holds one file;
  CLAUDE.md's structure section doesn't mention them.
- **Data provenance:** The discovery-tracklets CSV format (12 columns) is
  documented nowhere in the repo, and its origin — a `psql` query against the
  mpc_sbn replica on sibyl — exists only in chat history. The generating SQL
  is not checked in. The April refresh silently dropped 7 objects vs the
  March file (net +272 rows) with no record of why.
- **Settings persistence:** 609 hand-written lines mapping ~100 widgets in
  two must-stay-in-sync methods (`neolyzer.py:17377,17653`), on top of the
  four parallel state systems. Every new control is a five-place edit.
- **Config fragmentation:** `~/.neolyzer/ephemeris.json`,
  `~/.neolyzer_settings.json` (a sibling file, not in the directory), and
  `~/.skyfield`.
- **Dependencies:** minimum-bound pins only (reasonable for diverse user
  environments; CI installing latest is the only canary). `cython` is listed
  but no compiled extension appears to use it. astropy/scipy are commented
  "optional" but installed unconditionally. The PyQt5 fallback path exists in
  code but PyQt5 isn't in requirements, and the matplotlib backend import is
  hardcoded to `backend_qt5agg` (`neolyzer.py:119-120`) regardless of which
  binding loads.
- **No native Windows path** — WSL-only is documented consistently in README,
  but CLAUDE.md lists Windows as a supported platform; those claims disagree.

---

## Housekeeping (quick wins)

- Empty directory literally named `{src,data,cache,scripts,docs}` at repo
  root (tcsh brace-expansion accident, 2026-01-04).
- 0-byte stale `data/neo_catalog.db` (and a `.gitignore` entry for it).
- `NEO_discovery_tracklets.csv_SAVE` and the dated CSV untracked and not
  gitignored.
- ~4.2 GB of stale cache backups (`positions_jan17_backup.h5`,
  `positions_mpcorb_sim.h5`, `positions_mpcorb_diff.h5`) in the 22 GB
  `cache/` directory.
- 8 leftover `DEBUG:` prints in `setup_database.py`
  (:208,:214,:220,:222,:227,:228,:247,:252) plus a stray
  `traceback.print_exc()`.
- `verify_fixes.py` is a hardcoded string-grep for three January bugs and has
  outlived its purpose.

---

## Unexploited opportunities

- **Close the tracklet-provenance loop.** Check `discovery_tracklets.sql`
  into the repo (even though it runs on sibyl), document the CSV format, and
  record generation date + row count in a header or sidecar. Turns an
  out-of-band ritual into a repeatable procedure — precondition for ever
  automating the refresh.
- **Cache fingerprinting + real repack** would let `update_catalog.py` do
  delta rebuilds safely (DELTA_CACHING.txt already points this way).
- **Extract `SkyMapCanvas` and a `sky_math` module from the monolith.**
  Highest-leverage refactor: makes the render path testable, which is the
  precondition for the 100k-NEO performance work — you can't optimize a
  1,286-line `update_plot` you can't benchmark in isolation.
- **A headless smoke test in CI** (import neolyzer.py, instantiate the canvas
  offscreen, render one frame) catches a class of regressions the current
  suite can't see, cheaply.
- **QThread workers for downloads** double as the enabler for in-app catalog
  updates without freezing.
- **Packaging:** no `pyproject.toml`; a proper package would simplify the
  install story and let the diagnose scripts import `src/` cleanly.

---

## Open questions for the project lead

1. **Has the epoch TT/UTC bug ever shown up in validation?** Do the Horizons
   comparison tests pass because tolerance absorbs 69 s, or because fixtures
   bypass `_unpack_epoch`? Fixing it invalidates the 18 GB cache.
2. **What's the intended commit state for the tracklets CSV?** Commit the
   dated-file-plus-symlink scheme, or revert to a plain tracked CSV?
   (Recommendation: plain file — symlinks in git are a portability trap for a
   cross-platform project.)
3. **Who is the user, really?** Public GitHub repo with issue templates, but
   operating procedures (sibyl queries, manual CSV drops, 22 GB local cache)
   assume a single expert operator. If outside adoption is a goal, the
   setup-friction items (false verify failure, no-timeout downloads) are the
   first impressions.
4. **What's the catalog-growth plan?** At current discovery rates the catalog
   crosses 50–60k NEOs soon; the cache is already 18 GB and only grows.
   Smarter tiers, on-demand computation for the tail, or a cache size budget?
5. **Should MPC's newer data services replace the flat-file scrape?** The
   MPCORB fixed-width download is the most fragile external dependency; MPC
   has been pushing toward its API/replica offerings — and mpc_sbn replica
   access on sibyl already exists for sibling projects.
