NEOlyzer v3.04 - Near-Earth Object Visualization and Analysis
=============================================================

Interactive visualization tool for the NEO catalog supporting Planetary Defense
research and operations. Developed at Catalina Sky Survey, University of Arizona.

Date Range: 1550-2650 (DE440, configurable)
Objects: 40,000+ Near-Earth Asteroids (catalog grows daily)


QUICK START
===========

Fetch from GitHub:
    git clone https://github.com/rlseaman/neolyzer.git
    cd neolyzer

Or download and extract release:
    tar -xzf neolyzer-v3.04.tar.gz
    cd neolyzer-v3.04

Install and run:
    ./install.sh              # Creates venv, installs deps, runs setup
    ./run_neolyzer.sh         # Launch the application


LAUNCHING
=========

    ./run_neolyzer.sh                              # Recommended launcher
    ./venv/bin/python src/neolyzer.py              # Direct invocation
    ./venv/bin/python src/neolyzer.py --quiet      # Suppress console output
    ./venv/bin/python src/neolyzer.py --debug      # Enable debug logging
    ./venv/bin/python src/neolyzer.py --no-cache   # Skip cache, compute live

Exit: Click Exit button, close window, or press Ctrl+C in terminal.


PLATFORM SUPPORT
================

Supported platforms:
    - macOS (Intel and Apple Silicon)
    - Linux (RHEL, Debian/Ubuntu, Raspberry Pi)
    - Windows (via WSL)

Requirements:
    - Python 3.10+ (3.12+ recommended)
    - Unix-like shell (bash/zsh) for install.sh and run_neolyzer.sh
    - Windows users: use WSL, Git Bash, or run Python directly

Platform-specific notes in PLATFORM_NOTES.txt.


PREFERENCES & CONSTRAINTS
=========================

Dark mode: The application uses a light theme optimized for data visualization.
    System dark mode may cause display issues on some platforms. If controls
    appear illegible, try disabling system dark mode or check PLATFORM_NOTES.txt.

Shell: Installation scripts require bash or compatible shell. On Windows,
    use WSL, Git Bash, or manually create venv and run Python directly.

Display: Designed for 1280x800 minimum resolution. Larger displays recommended
    for full feature visibility.


FEATURES
========

- Multiple map projections (Rectangular, Hammer, Aitoff, Mollweide)
- Multiple coordinate systems (Equatorial, Ecliptic, Galactic, Opposition)
- Animation with variable playback rate and direction
- Moon phase display with Catalina Lunation Number (CLN)
- Discovery tracking (hide objects before discovery date)
- Earth MOID filtering via JPL SBDB
- Horizon/twilight overlays for observer location
- Scripted playback with full state save/restore
- Data tables with selection and export


JPL EPHEMERIS OPTIONS
=====================

During setup, you can select which JPL planetary ephemeris to use. The ephemeris
provides precise positions for the Sun, Moon, and planets.

    Ephemeris   Years              Size      Download   Cache Build   Notes
    ---------   -----              ----      --------   -----------   -----
    DE421       1900-2050          17 MB     ~1 min     ~15 min       Compact, legacy
    DE440       1550-2650          115 MB    ~5 min     ~25 min       Recommended
    DE441       -13200 to +17191   3.5 GB    ~60 min    ~25 min       Extended range

DE440 is the default and recommended choice:
- Modern accuracy incorporating spacecraft tracking data through 2020
- Extended range covers historical observations and future predictions
- Improved lunar positions from Lunar Reconnaissance Orbiter data
- Reasonable file size (115 MB)

Storage requirements (approximate):
- Ephemeris file: 17-3500 MB (stored in ~/.skyfield/)
- Position cache: 200-400 MB (stored in cache/)
- Database: 50-100 MB (stored in data/)
- Total: 300 MB - 4 GB depending on ephemeris choice

Runtime efficiency: All ephemerides provide identical runtime performance.
The position cache pre-computes asteroid positions, so ephemeris choice
only affects setup time, not visualization speed.


UNDERLYING PACKAGES
===================

NEOlyzer builds on several astronomy and visualization packages:

Skyfield - High-precision astronomical computations
    Repository: https://github.com/skyfielders/python-skyfield
    Documentation: https://rhodesmill.org/skyfield/
    Purpose: Ephemeris loading, coordinate transformations, time handling
    API: Python library using JPL SPICE kernels

JPL Development Ephemerides (DE) - Planetary position data
    Source: https://ssd.jpl.nasa.gov/planets/eph_export.html
    Download: https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/
    Format: SPICE Binary SPK (.bsp files)
    Coverage: DE440 spans 1550-2650 with sub-arcsecond accuracy

Minor Planet Center (MPC) - NEO orbital elements
    Website: https://www.minorplanetcenter.net/
    Data: https://www.minorplanetcenter.net/iau/MPCORB.html
    API: HTTP download of orbital element files (MPCORB.DAT, NEA.txt)
    Updates: Daily catalog updates available

JPL Small-Body Database (SBDB) - Physical and orbital parameters
    Website: https://ssd.jpl.nasa.gov/tools/sbdb_lookup.html
    API: https://ssd-api.jpl.nasa.gov/doc/sbdb.html
    Purpose: Earth MOID data, physical parameters, close approach data
    Access: RESTful JSON API with batch query support

PyQt6 - Cross-platform GUI framework
    Repository: https://www.riverbankcomputing.com/software/pyqt/
    Documentation: https://doc.qt.io/qtforpython-6/
    Purpose: Window management, controls, dialogs

Matplotlib - Scientific visualization
    Repository: https://github.com/matplotlib/matplotlib
    Documentation: https://matplotlib.org/
    Purpose: Sky map rendering, charts, map projections
    Backend: Qt5Agg for PyQt integration

HDF5/h5py - Hierarchical data storage
    Repository: https://github.com/h5py/h5py
    Documentation: https://docs.h5py.org/
    Purpose: Position cache storage with compression
    Format: Binary HDF5 with gzip compression


MAINTENANCE
===========

Update catalog:     ./venv/bin/python scripts/update_catalog.py
Rebuild cache:      ./venv/bin/python scripts/build_cache.py
Verify install:     ./venv/bin/python scripts/verify_installation.py
Run setup again:    ./run_setup.sh
Change ephemeris:   Re-run ./run_setup.sh and select new ephemeris


TROUBLESHOOTING
===============

Python version:     brew install python@3.12 (macOS) or apt install python3.12
HDF5 errors:        brew install hdf5 && export HDF5_DIR=/opt/homebrew/opt/hdf5
PyQt6 issues:       See PLATFORM_NOTES.txt for platform-specific Qt guidance
Permission denied:  chmod +x install.sh run_neolyzer.sh run_setup.sh
Ephemeris errors:   Check ~/.skyfield/ for corrupted .bsp files


CONTACT
=======

Rob Seaman, Catalina Sky Survey
Lunar and Planetary Laboratory, University of Arizona
rseaman@arizona.edu
