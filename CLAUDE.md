# NEOlyzer

Interactive visualization tool for the Near-Earth Object catalog, supporting Planetary Defense research and operations.

**Lead:** Rob Seaman, Catalina Sky Survey, Lunar and Planetary Laboratory, University of Arizona (rseaman@arizona.edu)

## What This Project Does

NEOlyzer visualizes NEOs — asteroids and comets approaching within 1.3 AU of the Sun. The tool helps researchers and observers understand orbital dynamics, plan observations, and track catalog growth. NEO impacts are inevitable across geological timescales; this tool supports the human effort to find them first.

## Tech Stack

- **Language:** Python 3.10+ (3.12+ recommended)
- **Database:** SQLite via SQLAlchemy (single table currently; multi-table normalization planned)
- **GUI:** PyQt6 (with PyQt5 fallback) + matplotlib (Qt5Agg backend) for visualization
- **Caching:** HDF5 via h5py for pre-computed positions
- **Astronomy:** Skyfield + configurable JPL ephemeris (DE440 default: 1550–2650)
- **Platforms:** macOS (Intel/Apple Silicon), Linux (RHEL, Debian/Ubuntu, Raspberry Pi), Windows

## Project Structure

```
neolyzer/
├── src/                        # Main source code
│   ├── neolyzer.py             # Main application (entry point, ~450KB)
│   ├── database.py             # SQLite database management via SQLAlchemy
│   ├── orbit_calculator.py     # Orbital mechanics calculations
│   ├── cache_manager.py        # HDF5 position caching
│   ├── mpc_loader.py           # MPC catalog data loading
│   ├── designation_utils.py    # Asteroid designation parsing
│   ├── ephemeris_config.py     # JPL ephemeris selection and metadata
│   └── skyfield_loader.py      # Skyfield ephemeris management
├── scripts/                    # Setup and maintenance scripts
│   ├── setup_database.py       # Initial setup (download data, build cache)
│   ├── update_catalog.py       # Update catalog from MPC
│   ├── build_cache.py          # Rebuild position cache
│   ├── verify_installation.py  # Verify installation
│   └── verify_fixes.py         # Verification tests
├── data/                       # Data files
│   └── NEA_discovery_tracklets.csv  # Discovery circumstances data
├── assets/                     # UI assets
│   └── CSS_logo_transparent.png     # Catalina Sky Survey logo
├── diagnose_*.py               # Diagnostic scripts (CLN, missing NEOs, SBDB)
├── install.sh                  # Cross-platform installation script
├── requirements.txt            # Python dependencies
├── CLAUDE.md                   # This file
├── README.txt                  # User documentation
└── PLATFORM_NOTES.txt          # Platform-specific notes
```

Key components:
- **src/neolyzer.py** — Main visualization engine with PyQt6 GUI
- **src/database.py** — Database operations, MOID fetching, discovery tracklet loading
- **src/orbit_calculator.py** — FastOrbitCalculator for position computation
- **src/cache_manager.py** — HDF5-based position caching with multiple precision tiers
- **scripts/setup_database.py** — Interactive setup wizard

## Commands

```bash
# Full installation (creates venv, installs deps, runs setup)
./install.sh

# Run the application (after install.sh creates launchers)
./run_neolyzer.sh                              # Via launcher script
./venv/bin/python src/neolyzer.py              # Direct invocation
./venv/bin/python src/neolyzer.py --quiet      # Suppress console output
./venv/bin/python src/neolyzer.py --debug      # Debug logging
./venv/bin/python src/neolyzer.py --no-cache   # Skip cache, compute on-the-fly

# Setup: download catalogs and build database (interactive)
./run_setup.sh                                 # Via launcher script
./venv/bin/python scripts/setup_database.py   # Direct invocation

# Maintenance
./venv/bin/python scripts/update_catalog.py   # Update from MPC
./venv/bin/python scripts/build_cache.py      # Rebuild position cache

# Verification
./venv/bin/python scripts/verify_installation.py
./venv/bin/python scripts/verify_fixes.py

# Diagnostics (run from project root)
python diagnose_cln.py      # CLN calculation methods comparison
python diagnose_missing.py  # Check for missing NEOs
python diagnose_sbdb.py     # SBDB/JPL data diagnostics
```

## Current State

**Working well:**
- Displays 40,000+ NEOs with smooth animation
- Multiple map projections (Rectangular, Hammer, Aitoff, Mollweide)
- Multiple coordinate systems (Equatorial, Ecliptic, Galactic, Opposition)
- Time range: 1550–2650 with DE440 (configurable ephemeris selection)
- HDF5 position cache with multiple precision tiers (daily/weekly/monthly)
- Dual magnitude filtering (V and H magnitudes, min and max)
- Animation controls with variable rate (hours/days/months per second)
- Negative rate support for backwards playback
- Moon phase display (CLN — Catalina Lunation Number)
- Discovery tracklet integration (hide objects before discovery date)
- Earth MOID filtering via JPL SBDB API
- Cross-platform support (macOS, Linux, Raspberry Pi)

**Needs work:**
- Performance optimization for 100,000+ NEOs (catalog growth is rapid)
- Additional controls and modes (specific features TBD per session)
- Database schema may need normalization as features expand

## Design Principles

1. **Simplicity for users** — Target audience includes non-expert users; minimize setup friction
2. **Performance matters** — Frame rate and responsiveness are primary concerns at scale
3. **Cross-platform parity** — Features must work on macOS, Linux variants, and Windows
4. **Incremental complexity** — Single-table DB works now; normalize only when features require it
5. **Claude codes, Rob designs** — Architectural and scientific decisions come from Rob; implementation details are Claude's domain

## Working With This Project

- **Before modifying visualization code:** understand the current frame rate constraints
- **Before modifying database schema:** discuss implications with Rob first
- **New features:** implement behind flags or modes so existing functionality isn't disrupted
- **Testing:** use diagnostic scripts; also note any user feedback Rob provides
- **Dependencies:** minimize additions; users have diverse environments

## Domain Context

Orbital mechanics terminology that may appear:
- **AU** — Astronomical Unit (Earth-Sun distance)
- **Ephemeris** — Predicted positions of an object over time
- **NEO** — Near-Earth Object (perihelion < 1.3 AU)
- **PHA** — Potentially Hazardous Asteroid (subset of NEOs with closer approaches)
- **MOID** — Minimum Orbit Intersection Distance (closest approach between two orbits)
- **MPC** — Minor Planet Center (primary catalog source)
- **JPL SBDB** — JPL Small-Body Database (source for MOID data)
- **JPL Horizons** — NASA ephemeris service
- **CLN** — Catalina Lunation Number (lunation count from epoch 1980-01-02)
- **JD** — Julian Date (continuous day count from 4713 BC)

Claude should not oversimplify orbital mechanics in code comments or UI text — the audience understands the domain.

## Session Workflow

Typical interaction pattern:
1. Rob describes a feature need or problem
2. Claude reads relevant files, asks clarifying questions if needed
3. Claude proposes approach (use "think" for complex changes)
4. Rob approves or redirects
5. Claude implements and tests
6. Rob evaluates result

If uncertain about scientific or design intent, **ask** rather than assume.
