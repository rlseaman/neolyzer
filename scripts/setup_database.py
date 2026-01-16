#!/usr/bin/env python3
"""
Initial Setup Script - Download MPC data and build cache
Run this once to initialize the NEO Visualizer
"""

import sys
import os
import logging
import time
import platform
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from mpc_loader import load_asteroids_from_mpc
from database import DatabaseManager, fetch_moid_batch, load_discovery_tracklets
from orbit_calculator import FastOrbitCalculator
from cache_manager import PositionCache, CacheBuilder
from skyfield.api import load

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def format_duration(seconds):
    """Format duration in human-readable form"""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"


def get_platform_info():
    """Get platform information string"""
    system = platform.system()
    machine = platform.machine()
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    if system == "Darwin":
        os_name = "macOS"
        if machine == "arm64":
            os_name += " (Apple Silicon)"
        else:
            os_name += " (Intel)"
    elif system == "Linux":
        try:
            with open('/etc/os-release') as f:
                for line in f:
                    if line.startswith('PRETTY_NAME='):
                        os_name = line.split('=')[1].strip().strip('"')
                        break
                else:
                    os_name = "Linux"
        except:
            os_name = "Linux"
        
        # Check for Raspberry Pi
        try:
            with open('/proc/device-tree/model') as f:
                model = f.read()
                if 'raspberry' in model.lower():
                    os_name += " (Raspberry Pi)"
        except:
            pass
    else:
        os_name = system
    
    return os_name, machine, python_version


def main():
    """Run initial setup"""
    start_time = time.time()
    
    # Get platform info
    os_name, arch, py_version = get_platform_info()
    
    # Acquire lock to prevent concurrent setup runs
    lock_file = Path(__file__).parent.parent / '.setup.lock'
    lock_fd = None
    
    try:
        import fcntl
        lock_fd = open(lock_file, 'w')
        try:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            lock_fd.write(f"PID: {os.getpid()}\nStarted: {datetime.now().isoformat()}\n")
            lock_fd.flush()
        except (IOError, OSError):
            print("=" * 70)
            print("ERROR: Another setup process is already running!")
            print("=" * 70)
            print()
            print("If you're sure no other setup is running, remove the lock file:")
            print(f"  rm {lock_file}")
            print()
            return 1
    except ImportError:
        # fcntl not available on Windows - skip locking
        pass
    
    print("=" * 70)
    print("NEO Visualizer - Initial Setup")
    print("=" * 70)
    print(f"  Platform:  {os_name}")
    print(f"  Arch:      {arch}")
    print(f"  Python:    {py_version} ({sys.executable})")
    print(f"  Started:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()
    
    # Step 1: Download MPC data
    print("Step 1: Downloading NEO orbital elements...")
    print("-" * 70)
    
    try:
        asteroids = load_asteroids_from_mpc(
            neo_only=True,
            force_download=False  # Use cached if available
        )
        print(f"✓ Downloaded {len(asteroids)} NEO orbits")
        print()
    except Exception as e:
        print(f"✗ Error downloading data: {e}")
        return 1
    
    # Step 2: Fetch Earth MOID from JPL
    print("Step 2: Fetching Earth MOID from JPL API...")
    print("-" * 70)
    print("This retrieves the Minimum Orbit Intersection Distance with Earth.")
    print("Uses JPL's batch query API (single request for all NEOs).")
    print()
    print("Diagnostic files will be saved to help with designation matching:")
    print("  - mpc_designations.txt: List from NEA.txt")
    print("  - jpl_designations.txt: List from JPL SBDB")
    print("  - matching_log.txt: Matching attempts and results")
    print("  - sbdb_moid_cache.json: Raw API response (JSON)")
    print()
    
    response = input("Fetch MOID data? This takes ~30 seconds. (y/n): ").lower().strip()
    print(f"DEBUG: Response received: '{response}'")
    
    if response == 'y':
        try:
            # Create diagnostics directory
            diag_dir = os.path.join(os.path.dirname(__file__), '..', 'diagnostics')
            print(f"DEBUG: Creating diagnostics directory: {diag_dir}")
            os.makedirs(diag_dir, exist_ok=True)
            
            # Also save to data directory for persistence
            data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
            
            print(f"DEBUG: Calling fetch_moid_batch with output_dir={diag_dir}")
            fetch_moid_batch(asteroids, show_progress=True, output_dir=diag_dir)
            print("DEBUG: fetch_moid_batch completed")
            
            # Copy SBDB cache to data directory for persistence
            diag_cache = os.path.join(diag_dir, 'sbdb_moid_cache.json')
            data_cache = os.path.join(data_dir, 'sbdb_moid_cache.json')
            print(f"DEBUG: Checking for cache at {diag_cache}")
            print(f"DEBUG: File exists: {os.path.exists(diag_cache)}")
            if os.path.exists(diag_cache):
                import shutil
                shutil.copy2(diag_cache, data_cache)
                print(f"✓ SBDB cache saved to {data_cache}")
            else:
                print(f"WARNING: Expected cache file not found at {diag_cache}")
            
            moid_count = sum(1 for ast in asteroids if ast.get('earth_moid') is not None)
            print(f"✓ Fetched MOID for {moid_count}/{len(asteroids)} asteroids")
            print(f"✓ Diagnostic files saved to {diag_dir}")
            print()
        except Exception as e:
            import traceback
            print(f"✗ Error fetching MOID data: {e}")
            print("DEBUG: Full traceback:")
            traceback.print_exc()
            print("⚠ Continuing without MOID data...")
            print()
    else:
        print(f"DEBUG: Skipping because response was '{response}', not 'y'")
        print("⚠ Skipping MOID fetch. MOID filtering will not be available.")
        print()
    
    # Step 2b: Load discovery tracklet data
    print("Step 2b: Loading discovery tracklet data...")
    print("-" * 70)
    print("This loads discovery circumstances (date, position, magnitude, site)")
    print("for each NEO from the bundled NEA_discovery_tracklets.csv file.")
    print()
    
    try:
        tracklet_csv = os.path.join(os.path.dirname(__file__), '..', 'data', 'NEA_discovery_tracklets.csv')
        if os.path.exists(tracklet_csv):
            matched = load_discovery_tracklets(asteroids, tracklet_csv, show_progress=True)
            print(f"✓ Loaded discovery data for {matched}/{len(asteroids)} asteroids")
        else:
            print(f"⚠ Discovery tracklet file not found: {tracklet_csv}")
            print("  'Hide before discovery' feature will not be available.")
        print()
    except Exception as e:
        print(f"✗ Error loading discovery tracklets: {e}")
        print("⚠ Continuing without discovery data...")
        print()
    
    # Step 3: Initialize database
    print("Step 3: Setting up database...")
    print("-" * 70)
    
    try:
        # Use SQLite by default (change to PostgreSQL URL if desired)
        db = DatabaseManager(use_sqlite=True)
        print("✓ Database initialized (SQLite)")
        
        # Clear existing data
        response = input("Clear existing data? (y/n): ").lower()
        if response == 'y':
            db.clear_all()
            print("✓ Database cleared")
        
        # Insert asteroids
        print(f"Inserting {len(asteroids)} asteroids...")
        db.insert_asteroids(asteroids)
        print("✓ Data loaded into database")
        
        # Show statistics
        stats = db.get_statistics()
        print()
        print("Database Statistics:")
        print(f"  Total asteroids: {stats['total']}")
        print(f"  NEOs: {stats['neos']}")
        print(f"  PHAs: {stats['phas']}")
        print(f"  By class: {stats.get('by_class', {})}")
        print()
        
    except Exception as e:
        print(f"✗ Error setting up database: {e}")
        return 1
    
    # Step 4: Build position cache
    print("Step 4: Building position cache...")
    print("-" * 70)
    print("This will pre-compute asteroid positions for efficient visualization.")
    print()
    print("Cache options:")
    print("  [f] Full cache - all precision tiers (±6mo daily, ±5yr weekly, full monthly)")
    print("      Best for production use. Takes 10-30 minutes.")
    print("  [q] Quick cache - high precision only (±6 months, daily)")
    print("      Good for testing. Takes 2-5 minutes.")
    print("  [n] No cache - compute positions on-the-fly")
    print("      Slowest but no setup time.")
    print()
    
    response = input("Build cache? [f]ull / [q]uick / [n]o: ").lower().strip()
    
    if response in ['f', 'q']:
        high_precision_only = (response == 'q')
        try:
            # Initialize calculator and cache
            calculator = FastOrbitCalculator()
            cache = PositionCache()
            builder = CacheBuilder(cache, calculator)
            
            # Get current Julian Date
            ts = load.timescale()
            now = ts.now()
            reference_jd = now.tt
            
            print(f"Reference date: {now.utc_datetime()}")
            print(f"Reference JD: {reference_jd:.2f}")
            if high_precision_only:
                print("Mode: Quick (±6 months, daily positions only)")
            else:
                print("Mode: Full (all precision tiers)")
            print()
            
            # Build cache
            print("Building cache (this will take a while)...")
            builder.build_cache(asteroids, reference_jd, show_progress=True, 
                              high_precision_only=high_precision_only)
            
            # Show cache stats
            stats = cache.get_cache_statistics()
            print()
            print("Cache Statistics:")
            print(f"  File size: {stats['file_size_mb']:.1f} MB")
            print(f"  High precision dates: {stats['high_precision_dates']}")
            print(f"  Medium precision dates: {stats['medium_precision_dates']}")
            print(f"  Low precision dates: {stats['low_precision_dates']}")
            print(f"  Total objects: {stats['n_objects']}")
            print()
            print("✓ Cache built successfully")
            
        except Exception as e:
            print(f"✗ Error building cache: {e}")
            logger.error("Cache build error", exc_info=True)
            return 1
    else:
        print("⚠ Skipping cache build. Positions will be computed on-the-fly.")
        print("  (This is slower but works fine for initial testing)")
        print()
    
    # Done!
    elapsed = time.time() - start_time
    
    print("=" * 70)
    print("Setup Complete!")
    print("=" * 70)
    print()
    print(f"Total elapsed time: {format_duration(elapsed)}")
    print()
    print("You can now run the visualizer:")
    print("  python src/visualizer.py")
    print()
    print("To update the catalog daily, run:")
    print("  python scripts/update_catalog.py")
    print()
    
    # Release lock
    if lock_fd:
        lock_fd.close()
        try:
            lock_file.unlink()
        except:
            pass
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
