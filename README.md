# NEOlyzer

**Near-Earth Object Visualization and Analysis** | Version 3.06

Interactive visualization tool for the NEO catalog supporting Planetary Defense research and operations. Developed at Catalina Sky Survey, University of Arizona.

- **Date Range:** 1550-2650 (DE440 ephemeris, configurable)
- **Objects:** 40,000+ Near-Earth Asteroids (catalog grows daily)

---

## Quick Start

### With Git

```bash
git clone https://github.com/rlseaman/neolyzer.git
cd neolyzer
./install.sh
./run_neolyzer.sh
```

### Without Git

Download and extract: [neolyzer-v3.06.zip](https://github.com/rlseaman/neolyzer/archive/refs/tags/v3.06.zip)

```bash
unzip neolyzer-v3.06.zip
cd neolyzer-3.06
./install.sh
./run_neolyzer.sh
```

Or download from the [Releases](https://github.com/rlseaman/neolyzer/releases) page.

---

## Platform Support

| Platform | Status |
|----------|--------|
| macOS (Intel) | Supported |
| macOS (Apple Silicon) | Supported |
| Linux (RHEL, Debian/Ubuntu) | Supported |
| Linux (Raspberry Pi) | Supported |
| Windows (via WSL) | Supported |

**Requirements:**
- Python 3.10+ (3.12+ recommended)
- Unix-like shell (bash/zsh) for install scripts
- Windows users: use WSL, Git Bash, or run Python directly

Platform-specific notes in [PLATFORM_NOTES.txt](PLATFORM_NOTES.txt).

---

## Features

- **Multiple map projections:** Rectangular, Hammer, Aitoff, Mollweide
- **Coordinate systems:** Equatorial, Ecliptic, Galactic, Opposition
- **Animation:** Variable playback rate and direction
- **Moon phase display:** Catalina Lunation Number (CLN)
- **Discovery tracking:** Hide objects before discovery date, with tracklet details (rate, PA, nobs, span, site name) from bundled `NEO_discovery_tracklets.csv`
- **Earth MOID filtering:** Via JPL SBDB
- **Horizon/twilight overlays:** For observer location
- **Scripted playback:** Full state save/restore
- **Data tables:** Selection and CSV export
- **Constellation boundaries:** IAU boundary overlay
- **Background stars:** Bright star display with magnitude filtering
- **Alternate catalogs:** Load and compare multiple catalog versions
- **Catalog blinking:** Rapidly toggle between catalogs for comparison

---

## Launching

```bash
./run_neolyzer.sh                              # Recommended launcher
./venv/bin/python src/neolyzer.py              # Direct invocation
./venv/bin/python src/neolyzer.py --quiet      # Suppress console output
./venv/bin/python src/neolyzer.py --debug      # Enable debug logging
./venv/bin/python src/neolyzer.py --no-cache   # Skip cache, compute live
```

Exit: Click Exit button, close window, or press Ctrl+C in terminal.

---

## JPL Ephemeris Options

During setup, select which JPL planetary ephemeris to use:

| Ephemeris | Years | Size | Download | Notes |
|-----------|-------|------|----------|-------|
| DE421 | 1900-2050 | 17 MB | ~1 min | Compact, legacy |
| **DE440** | 1550-2650 | 115 MB | ~5 min | **Recommended** |
| DE441 | -13200 to +17191 | 3.5 GB | ~60 min | Extended range |

**DE440** is the default and recommended choice:
- Modern accuracy incorporating spacecraft tracking data through 2020
- Extended range covers historical observations and future predictions
- Improved lunar positions from Lunar Reconnaissance Orbiter data

**Storage requirements (approximate):**
- Ephemeris file: 17-3500 MB (stored in `~/.skyfield/`)
- Position cache: 200-400 MB (stored in `cache/`)
- Database: 50-100 MB (stored in `data/`)
- Discovery tracklets: ~4.5 MB (bundled `data/NEO_discovery_tracklets.csv`)
- Total: 300 MB - 4 GB depending on ephemeris choice

---

## Maintenance

### Basic Commands

```bash
./venv/bin/python scripts/update_catalog.py   # Update catalog from MPC
./venv/bin/python scripts/build_cache.py      # Rebuild position cache
./venv/bin/python scripts/verify_installation.py  # Verify installation
./run_setup.sh                                 # Re-run setup wizard
```

### Updating the Primary Catalog

The primary catalog can be updated from the Minor Planet Center with various options:

```bash
# Daily update (minimal, ~5 min)
./venv/bin/python scripts/update_catalog.py --fetch-moid --mark-stale --clear-cache

# Daily update with quick cache rebuild (~10 min)
./venv/bin/python scripts/update_catalog.py --fetch-moid --mark-stale --quick-cache

# Weekly update with full cache rebuild (~35 min)
./venv/bin/python scripts/update_catalog.py --fetch-moid --mark-stale --rebuild-cache

# Clean sync - delete missing objects (destructive)
./venv/bin/python scripts/update_catalog.py --fetch-moid --sync
```

**Options:**
| Option | Description |
|--------|-------------|
| `--fetch-moid` | Fetch Earth MOID values from JPL SBDB (~2-3 min) |
| `--clear-cache` | Clear position cache (app computes on-the-fly) |
| `--quick-cache` | Rebuild ±1 year high-precision cache only (~5 min) |
| `--rebuild-cache` | Rebuild entire cache - all precision tiers (~30 min) |
| `--sync` | Delete objects not in current MPC catalog (destructive) |
| `--mark-stale` | Mark missing objects as stale with timestamp |
| `--quiet` | Suppress progress output (for cron jobs) |

### Loading Alternate Catalogs

Load historical or comparison catalogs for side-by-side analysis:

```bash
# Basic load (name derived from filename)
./venv/bin/python scripts/load_alt_catalog.py alt_data/NEA_backup.txt

# Load with custom name and MOID data
./venv/bin/python scripts/load_alt_catalog.py alt_data/NEA_jan17.txt --name jan17_backup --fetch-moid

# Full load with MOID, discovery data, and position cache
./venv/bin/python scripts/load_alt_catalog.py alt_data/NEA.txt --name backup_jan \
    --fetch-moid --load-discovery --quick-cache

# Replace existing catalog
./venv/bin/python scripts/load_alt_catalog.py alt_data/NEA.txt --name backup --replace

# List loaded catalogs
./venv/bin/python scripts/load_alt_catalog.py --list

# Show catalog info
./venv/bin/python scripts/load_alt_catalog.py --info jan17_backup

# Delete a catalog
./venv/bin/python scripts/load_alt_catalog.py --delete old_catalog
```

**Options:**
| Option | Description |
|--------|-------------|
| `--name NAME` | Catalog name (default: derived from filename) |
| `--description TEXT` | Optional description for the catalog |
| `--fetch-moid` | Fetch Earth MOID values from JPL SBDB |
| `--load-discovery` | Load discovery circumstances if CSV available |
| `--build-cache` | Build full position cache for this catalog |
| `--quick-cache` | Build ±1 year high-precision cache only |
| `--replace` | Replace existing catalog with same name |
| `--list` | List all loaded alternate catalogs |
| `--info NAME` | Show detailed info about a catalog |
| `--delete NAME` | Delete an alternate catalog |
| `--force` | Force delete without confirmation |

**Using Alternate Catalogs in NEOlyzer:**
1. Select catalog from dropdown (left of Search box)
2. Click **Blink** button to toggle between primary and alternate
3. Adjust blink rate with spinner (0.25s - 5.0s)
4. Yellow background indicates alternate catalog is active
5. Some filters (MOID, discovery) disabled for alternates without that data

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Python version | `brew install python@3.12` (macOS) or `apt install python3.12` |
| HDF5 errors | `brew install hdf5 && export HDF5_DIR=/opt/homebrew/opt/hdf5` |
| PyQt6 issues | See [PLATFORM_NOTES.txt](PLATFORM_NOTES.txt) |
| Permission denied | `chmod +x install.sh run_neolyzer.sh run_setup.sh` |
| Ephemeris errors | Check `~/.skyfield/` for corrupted .bsp files |

---

## Underlying Packages

- **[Skyfield](https://rhodesmill.org/skyfield/)** — High-precision astronomical computations
- **[JPL DE Ephemerides](https://ssd.jpl.nasa.gov/planets/eph_export.html)** — Planetary position data
- **[Minor Planet Center](https://www.minorplanetcenter.net/)** — NEO orbital elements
- **[JPL SBDB](https://ssd.jpl.nasa.gov/tools/sbdb_lookup.html)** — Physical and orbital parameters, MOID data
- **[PyQt6](https://www.riverbankcomputing.com/software/pyqt/)** — Cross-platform GUI framework
- **[Matplotlib](https://matplotlib.org/)** — Scientific visualization
- **[h5py](https://docs.h5py.org/)** — HDF5 position cache storage

---

## License

[MIT License](LICENSE) — Copyright (c) 2026 University of Arizona, Catalina Sky Survey

---

## Contact

**Rob Seaman**
Catalina Sky Survey
Lunar and Planetary Laboratory, University of Arizona
[rseaman@arizona.edu](mailto:rseaman@arizona.edu)
