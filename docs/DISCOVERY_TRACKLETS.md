# Discovery Tracklets CSV — Format and Provenance

`data/NEO_discovery_tracklets.csv` carries discovery-circumstance data for
the NEO catalog: where, when, and how bright each object was at discovery.
NEOlyzer uses it to hide objects before their discovery date, render the
non-discovery overlay, and populate the discovery statistics table.

## Provenance

The file is generated **outside this repository** by
`all_neas_from_nea_txt_v4.sql` in the sibling `NEA_discovery_tracklets`
project (Rob Seaman / CSS). That query runs against `mpc_sbn`, the CSS
PostgreSQL replica of the MPC/SBN database, hosted on the Linux server
`sibyl` (not reachable from every development machine). It joins the
current MPC `NEA.txt` catalog against `obs_sbn` discovery observations
(`disc = '*'`), aggregating each object's discovery tracklet.

Completeness at last generation: 99.995% of NEA.txt (objects with no
discovery observation in the database are simply absent from the CSV and
are handled gracefully by the loader).

## Refresh procedure

1. On a host with `sibyl` access, follow the usage notes in the SQL header
   (download a fresh `NEA.txt`, run the query via `psql`, export CSV).
2. Copy the output into this repo as `data/NEO_discovery_tracklets.csv`
   (plain file — dated snapshots like `NEO_discovery_tracklets_10Apr26.csv`
   are gitignored; don't commit symlinks).
3. Re-ingest: tracklet columns are loaded into the database by
   `load_discovery_tracklets()` (`src/database.py`), which is invoked by
   `scripts/setup_database.py`, `scripts/update_catalog.py`, and
   `scripts/load_alt_catalog.py`.
4. Commit with a message noting the generation date and row count.

## Format

CSV with a header row; 12 columns:

| column | meaning |
|---|---|
| `primary_designation` | MPC packed primary designation (e.g. `00433`, `K24A01A`) — join key to the catalog |
| `packed_primary_provisional_designation` | packed provisional designation |
| `avg_mjd_discovery_tracklet` | mean MJD of the discovery tracklet's observations (UTC, as in obs80/ADES `obstime`) |
| `avg_ra_deg` | mean RA of tracklet observations, degrees (J2000) |
| `avg_dec_deg` | mean Dec, degrees (J2000) |
| `median_v_magnitude` | median V magnitude across the tracklet |
| `nobs` | number of observations in the discovery tracklet |
| `span_hours` | time span of the tracklet, hours |
| `rate_deg_per_day` | apparent sky motion rate, degrees/day |
| `position_angle_deg` | direction of motion, degrees E of N |
| `discovery_site_code` | MPC observatory code (e.g. `G96`) |
| `discovery_site_name` | observatory name |

The loader (`load_discovery_tracklets`) sniffs the header — it detects
this format via the `packed_primary_provisional_designation` column and
also accepts the legacy `NEA_discovery_tracklets.csv` layout.

## History notes

- 2026-04-10 generation: 44,490 rows (+279 / −7 designations vs the
  2026-03-18 generation). Drops occur when MPC designation or discovery
  attribution changes upstream.
- 2026-07-16: file restored from a symlink-to-dated-snapshot arrangement
  to a plain tracked CSV (see `docs/PROJECT_REVIEW_16Jul26.md`).
