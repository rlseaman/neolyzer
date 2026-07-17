# Position Cache Invalidation & Repack — Design (plan item 5.2)

Status: **proposal for review** (no code changes). 2026-07-16.

## Problem

`cache/positions.h5` (currently 18 GB) has no invalidation contract:

- Metadata stores only `reference_jd` and a `created` timestamp
  (`cache_manager.py`). No cache format version, no record of which DE
  kernel or which catalog state the positions were computed from.
- If `asteroids.db` changes (catalog update, the July epoch migration)
  or the ephemeris is switched (de421 ↔ de440 ↔ de441), nothing detects
  that the cache is stale — the app serves wrong positions silently.
  The TT/UTC epoch fix required a *manual* full rebuild for exactly
  this reason.
- `optimize_cache()` claims to repack but only calls `f.flush()`;
  deleted/rewritten HDF5 datasets never release space, so the file
  only grows (observed: +660 MB from a single interrupted rebuild).
- Two smaller design smells found during the July rebuild:
  - The date grid is anchored at `ts.now().tt` — a *fractional* JD
    (…238.5574…), so cached instants never land on clean 0h/12h
    boundaries and `JD{int(jd)}` keys truncate unevenly. Snapping the
    reference to the nearest 0h TT would make grid dates predictable.
  - Exact-match tolerance is 0.01 d; a request 0.057 d off the grid
    silently interpolates (or takes nearest), which for a close
    approacher can be arcminutes of motion.

## Proposal

### 1. Fingerprint metadata (new attrs on `/metadata`)

| attr | value | invalidates when |
|---|---|---|
| `cache_format_version` | int, start at 1 | layout changes |
| `ephemeris_file` | e.g. `de440.bsp` | ephemeris switched |
| `catalog_count` | row count of `asteroids` | catalog grew/shrank |
| `catalog_fingerprint` | `max(updated_at)` + count, hashed | any orbit updated |
| `epoch_policy` | `"TT"` | epoch semantics change |

On startup (and after `update_catalog.py` runs), the app compares
fingerprints. Mismatch → status-bar warning + dialog offering
(a) rebuild now, (b) continue with stale cache (labeled), (c) run
with `--no-cache` semantics for this session. Scripts get a
`--check-cache` / non-interactive equivalent.

### 2. Real repack

Replace `optimize_cache()`'s no-op with copy-rewrite: write all live
datasets to `positions.h5.new`, fsync, atomic rename over the old file
(same pattern as `net_utils.download_file`). Run automatically when
`file_size > k × live_data_size` (k ≈ 1.3), or on demand from
`build_cache.py --repack`.

### 3. Delta rebuild hook (enables DELTA_CACHING.txt)

With per-catalog fingerprints, `update_catalog.py` can rebuild only
the date keys affected by changed objects (new discoveries: append
columns; updated orbits: rewrite rows) instead of the full 25-minute
rebuild. This is the payoff that makes the fingerprint work worth it —
catalog updates become minutes, not a full rebuild.

### 4. Open questions for Rob

- Cache size budget as the catalog grows toward 100k? (Tier spacing
  and the ±27 yr low-precision span are the levers; 18 GB today
  scales roughly linearly with object count.)
- Snap `reference_jd` to 0h TT on next rebuild? (Changes all date
  keys once; harmless but the rebuild is the natural moment.)
- Should a stale cache ever be *silently* used? Proposal says no —
  always at least a status-bar label.

## Implementation order (each independently committable)

1. Write fingerprint attrs at build time (backward compatible — old
   caches simply lack them and read as "unknown provenance").
2. Startup check + warning (no dialog yet; log + status bar).
3. Repack.
4. Rebuild-offer dialog.
5. Delta rebuild (own design pass against DELTA_CACHING.txt).
