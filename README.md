# NEOlyzer

**Near-Earth Object Visualization and Analysis** | Version 3.05

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

Download and extract: [neolyzer-v3.05.zip](https://github.com/rlseaman/neolyzer/archive/refs/tags/v3.05.zip)

```bash
unzip neolyzer-v3.05.zip
cd neolyzer-3.05
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
- **Discovery tracking:** Hide objects before discovery date
- **Earth MOID filtering:** Via JPL SBDB
- **Horizon/twilight overlays:** For observer location
- **Scripted playback:** Full state save/restore
- **Data tables:** Selection and CSV export
- **Constellation boundaries:** IAU boundary overlay
- **Background stars:** Bright star display with magnitude filtering

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
- Total: 300 MB - 4 GB depending on ephemeris choice

---

## Maintenance

```bash
./venv/bin/python scripts/update_catalog.py   # Update catalog from MPC
./venv/bin/python scripts/build_cache.py      # Rebuild position cache
./venv/bin/python scripts/verify_installation.py  # Verify installation
./run_setup.sh                                 # Re-run setup wizard
```

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
