NEO SKY VISUALIZER v1.0 - Complete Installation Guide
======================================================

Professional Near-Earth Object visualization tool
Date Range: 1899-2053 (154 years from JPL DE421 ephemeris)
Status: Production ready - All bugs fixed


QUICK START
===========

Extract and run the installer:

    tar -xzf neo_visualizer_v1.0_final.tar.gz
    cd neo_visualizer
    ./install.sh

That's it! The installer handles everything automatically (~35 minutes).


LAUNCHING
=========

    ./venv/bin/python src/visualizer.py                # Normal mode
    ./venv/bin/python src/visualizer.py --quiet        # Quiet (no console spam)
    ./venv/bin/python src/visualizer.py --debug        # Debug info
    ./venv/bin/python src/visualizer.py --no-cache     # Smooth animation

Exit: Press Ctrl+C or click Exit button (clean shutdown, no errors)


WHAT'S FIXED
============

✓ Semi-major axis parsing (Eros a = 1.458 AU correct)
✓ Coordinate system mixing (proper ecliptic distribution)
✓ Mean motion calculation (realistic velocities)
✓ Ctrl+C handling (clean shutdown)
✓ Date range (full 1899-2053 coverage)


DATA COVERAGE
=============

Time: 1899-2053 (154 years, full JPL DE421 range)
Objects: 40,509 Near-Earth Asteroids
Cache: 1899-2053 at daily/weekly/monthly resolution


For complete documentation, see sections below.

COMPLETE INSTALLATION
=====================

Automated (Recommended):
    ./install.sh

Manual:
    1. Create venv: python3.14 -m venv venv
    2. Install deps: venv/bin/pip install -r requirements.txt
    3. Setup: venv/bin/python scripts/setup_database.py


FEATURES
========

- 40,509 NEO orbits (1899-2053)
- Multiple projections & coordinates
- Interactive animation
- Zoom, pan, navigate
- Export plots


BUGS FIXED
==========

1. Semi-major axis parsing
2. Coordinate system mixing  
3. Mean motion calculation
4. Ctrl+C clean shutdown


EXPECTED BEHAVIOR
=================

Motion: Median ~0.5-1°/day, 70-80% < 1°/day
Distribution: Mean ecliptic lat ~0°, 75% within ±10°


VERIFICATION
============

    ./venv/bin/python scripts/verify_installation.py
    ./venv/bin/python scripts/verify_fixes.py


TROUBLESHOOTING
===============

Python < 3.12: brew install python@3.14 (macOS)
HDF5 errors: brew install hdf5, export HDF5_DIR=/opt/homebrew/opt/hdf5
Motion too fast: Run verify_fixes.py - should show "ALL BUGS FIXED"


QUICK REFERENCE
===============

Install:      ./install.sh
Verify:       venv/bin/python scripts/verify_fixes.py
Launch:       venv/bin/python src/visualizer.py
Launch quiet: venv/bin/python src/visualizer.py --quiet
Rebuild cache: venv/bin/python scripts/build_cache.py

Happy asteroid hunting! 🌌🔭
