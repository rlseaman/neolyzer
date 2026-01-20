#!/usr/bin/env python3
"""
Daily Catalog Update Script
Downloads latest NEO data and updates the database
Can be run as a cron job for automated daily updates

Usage:
    ./venv/bin/python scripts/update_catalog.py [options]

Options:
    --fetch-moid      Fetch Earth MOID values from JPL SBDB
    --clear-cache     Clear the position cache (will rebuild on next app launch)
    --rebuild-cache   Rebuild the entire position cache (slow, ~30 min)
    --quick-cache     Rebuild only ±1 year high-precision cache (~5 min)
    --sync            Delete objects not in current catalog (destructive)
    --mark-stale      Mark missing objects as stale instead of deleting
    --quiet           Suppress progress output (for cron)
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timezone
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from mpc_loader import load_asteroids_from_mpc
from database import DatabaseManager, Asteroid, fetch_moid_batch
from cache_manager import PositionCache, CacheBuilder
from orbit_calculator import FastOrbitCalculator
from skyfield.api import load
from sqlalchemy import func

# Create logs directory
Path('logs').mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/catalog_update.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def prompt_collision_resolution(designation, old_data, new_data):
    """
    Prompt user to resolve a collision when a previously stale object reappears.

    Returns: 'update', 'keep_old', or 'skip'
    """
    print()
    print("=" * 70)
    print(f"COLLISION DETECTED: {designation}")
    print("=" * 70)
    print()
    print("This object was previously marked stale but has reappeared in the catalog.")
    print()
    print("Previous data (marked stale):")
    print(f"  H magnitude: {old_data.get('H', 'N/A')}")
    print(f"  Semi-major axis: {old_data.get('a', 'N/A'):.4f} AU" if old_data.get('a') else "  Semi-major axis: N/A")
    print(f"  Stale since: {old_data.get('stale_detected_at', 'N/A')}")
    print()
    print("New data from MPC:")
    print(f"  H magnitude: {new_data.get('H', 'N/A')}")
    print(f"  Semi-major axis: {new_data.get('a', 'N/A'):.4f} AU" if new_data.get('a') else "  Semi-major axis: N/A")
    print()
    print("Options:")
    print("  [1] Update with new data and clear stale flag (recommended)")
    print("  [2] Keep old data, clear stale flag only")
    print("  [3] Skip this object (leave as stale)")
    print()

    while True:
        response = input("Select option [1/2/3] (default: 1): ").strip()
        if response in ('', '1'):
            return 'update'
        elif response == '2':
            return 'keep_old'
        elif response == '3':
            return 'skip'
        else:
            print("Invalid option. Please enter 1, 2, or 3.")


def update_catalog(
    fetch_moid: bool = False,
    clear_cache: bool = False,
    rebuild_cache: bool = False,
    quick_cache: bool = False,
    sync_mode: bool = False,
    mark_stale: bool = False,
    quiet: bool = False
):
    """
    Update the asteroid catalog

    Parameters:
    -----------
    fetch_moid : bool
        Fetch Earth MOID values from JPL SBDB
    clear_cache : bool
        Clear the position cache without rebuilding
    rebuild_cache : bool
        Rebuild the entire position cache (slow)
    quick_cache : bool
        Rebuild only high-precision cache (±1 year, faster)
    sync_mode : bool
        Delete objects not in current catalog
    mark_stale : bool
        Mark missing objects as stale instead of deleting
    quiet : bool
        Suppress progress output
    """
    if sync_mode and mark_stale:
        logger.error("Cannot use both --sync and --mark-stale. Choose one.")
        return False

    logger.info("=" * 70)
    logger.info("NEO Catalog Update - " + datetime.now().isoformat())
    logger.info("=" * 70)

    # Download latest data
    logger.info("Downloading latest NEO data from MPC...")
    try:
        asteroids = load_asteroids_from_mpc(
            neo_only=True,
            force_download=True  # Always download fresh data
        )
        logger.info(f"Downloaded {len(asteroids)} NEO orbits")
    except Exception as e:
        logger.error(f"Failed to download data: {e}")
        return False

    # Create set of designations in new catalog for quick lookup
    new_designations = {ast['designation'] for ast in asteroids}

    # Update database
    logger.info("Updating database...")
    try:
        db = DatabaseManager(use_sqlite=True)
        session = db.get_session()

        # Get old statistics
        old_stats = db.get_statistics()
        old_count = old_stats['total']

        # Get current designations in database
        db_designations = set(
            row[0] for row in session.query(Asteroid.designation).all()
        )

        # Find objects that have disappeared from catalog
        missing_designations = db_designations - new_designations

        # Find previously stale objects that have reappeared
        reappeared = []
        stale_objects = session.query(Asteroid).filter(
            Asteroid.is_stale == True
        ).all()
        stale_designations = {obj.designation for obj in stale_objects}
        reappeared_designations = stale_designations & new_designations

        # Handle collisions (stale objects reappearing)
        collision_actions = {}
        if reappeared_designations:
            logger.info(f"Found {len(reappeared_designations)} previously stale objects reappearing")

            for desig in reappeared_designations:
                stale_obj = session.query(Asteroid).filter_by(designation=desig).first()
                new_data = next((a for a in asteroids if a['designation'] == desig), None)

                if stale_obj and new_data:
                    old_data = {
                        'H': stale_obj.H,
                        'a': stale_obj.a,
                        'stale_detected_at': stale_obj.stale_detected_at
                    }
                    action = prompt_collision_resolution(desig, old_data, new_data)
                    collision_actions[desig] = action

        # Process updates
        now = datetime.now(timezone.utc)
        new_objects = 0
        updated_objects = 0

        for ast_data in asteroids:
            desig = ast_data['designation']
            existing = session.query(Asteroid).filter_by(designation=desig).first()

            if existing:
                # Check if this is a collision case
                if desig in collision_actions:
                    action = collision_actions[desig]
                    if action == 'skip':
                        continue
                    elif action == 'keep_old':
                        # Just clear stale flag
                        existing.is_stale = False
                        existing.last_seen_in_catalog = now
                        continue
                    # 'update' falls through to normal update

                # Update existing object
                for key, value in ast_data.items():
                    setattr(existing, key, value)
                existing.is_stale = False
                existing.stale_detected_at = None
                existing.last_seen_in_catalog = now
                updated_objects += 1
            else:
                # Insert new object
                ast_data['last_seen_in_catalog'] = now
                ast_data['is_stale'] = False
                session.add(Asteroid(**ast_data))
                new_objects += 1

        session.commit()
        logger.info(f"Processed: {new_objects} new, {updated_objects} updated")

        # Handle missing objects
        if missing_designations:
            # Exclude already-stale objects from the count
            newly_missing = missing_designations - stale_designations

            if newly_missing:
                logger.info(f"Found {len(newly_missing)} objects missing from new catalog")

                if sync_mode:
                    # Delete missing objects
                    deleted = session.query(Asteroid).filter(
                        Asteroid.designation.in_(newly_missing)
                    ).delete(synchronize_session='fetch')
                    session.commit()
                    logger.info(f"Deleted {deleted} objects (--sync mode)")

                elif mark_stale:
                    # Mark as stale
                    for desig in newly_missing:
                        obj = session.query(Asteroid).filter_by(designation=desig).first()
                        if obj and not obj.is_stale:
                            obj.is_stale = True
                            obj.stale_detected_at = now
                    session.commit()
                    logger.info(f"Marked {len(newly_missing)} objects as stale (--mark-stale mode)")

                else:
                    logger.info("Missing objects retained (use --sync to delete or --mark-stale to flag)")

        session.close()

        # Get new statistics
        new_stats = db.get_statistics()
        new_count = new_stats['total']

        logger.info(f"Database: {old_count} → {new_count} asteroids")
        logger.info(f"NEOs: {new_stats['neos']}, PHAs: {new_stats['phas']}")

        if new_objects > 0:
            logger.info(f"🌟 {new_objects} new objects discovered!")

    except Exception as e:
        logger.error(f"Database update failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Fetch MOID data if requested
    if fetch_moid:
        logger.info("Fetching Earth MOID values from JPL SBDB...")
        try:
            # Re-query asteroids from database for MOID fetch
            db_asteroids = db.get_asteroids(neo_only=True)
            fetch_moid_batch(db_asteroids, show_progress=not quiet)
            logger.info("MOID data updated")
        except Exception as e:
            logger.error(f"MOID fetch failed: {e}")
            # Continue - this is not fatal

    # Handle cache
    if clear_cache or rebuild_cache or quick_cache:
        try:
            cache = PositionCache()

            if clear_cache and not rebuild_cache and not quick_cache:
                # Just clear, no rebuild - app will compute on-the-fly
                logger.info("Clearing position cache...")
                cache.clear_cache()
                logger.info("Cache cleared (app will compute positions on-the-fly until rebuilt)")

            elif rebuild_cache:
                # Full rebuild - clear all, rebuild all three precision tiers
                logger.info("Rebuilding position cache (full - all precision tiers)...")
                calculator = FastOrbitCalculator()
                builder = CacheBuilder(cache, calculator)

                ts = load.timescale()
                reference_jd = ts.now().tt
                logger.info(f"Reference JD: {reference_jd:.2f}")

                # Clear all precision groups
                cache.clear_cache()

                # Rebuild all tiers
                db_asteroids = db.get_asteroids(neo_only=True)
                builder.build_cache(
                    db_asteroids,
                    reference_jd,
                    show_progress=not quiet,
                    high_precision_only=False
                )

                stats = cache.get_cache_statistics()
                logger.info(f"Cache rebuilt: {stats['file_size_mb']:.1f} MB")

            elif quick_cache:
                # Quick rebuild - clear only high_precision, preserve medium/low
                logger.info("Rebuilding position cache (quick - ±1 year only, preserving long-range)...")
                calculator = FastOrbitCalculator()
                builder = CacheBuilder(cache, calculator)

                ts = load.timescale()
                reference_jd = ts.now().tt
                logger.info(f"Reference JD: {reference_jd:.2f}")

                # Clear only high precision group, preserve medium and low
                cache.clear_cache(groups=['high_precision'])

                # Rebuild only high precision tier
                db_asteroids = db.get_asteroids(neo_only=True)
                builder.build_cache(
                    db_asteroids,
                    reference_jd,
                    show_progress=not quiet,
                    high_precision_only=True
                )

                stats = cache.get_cache_statistics()
                logger.info(f"Cache updated: {stats['file_size_mb']:.1f} MB")

        except Exception as e:
            logger.error(f"Cache operation failed: {e}")
            return False
    else:
        logger.info("Cache unchanged (use --clear-cache, --rebuild-cache, or --quick-cache)")

    logger.info("=" * 70)
    logger.info("Update complete!")
    logger.info("=" * 70)

    return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Update NEO catalog from MPC',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Daily update (no cache rebuild, ~5 min):
    %(prog)s --fetch-moid --mark-stale --clear-cache

  Daily update with quick cache (~10 min):
    %(prog)s --fetch-moid --mark-stale --quick-cache

  Weekly update with full cache rebuild (~35 min):
    %(prog)s --fetch-moid --mark-stale --rebuild-cache

  Clean sync (delete missing objects):
    %(prog)s --fetch-moid --sync
        """
    )
    parser.add_argument(
        '--fetch-moid',
        action='store_true',
        help='Fetch Earth MOID values from JPL SBDB (~2-3 min)'
    )
    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help='Clear position cache (app computes on-the-fly until rebuilt)'
    )
    parser.add_argument(
        '--rebuild-cache',
        action='store_true',
        help='Rebuild entire cache - all precision tiers (~30 min)'
    )
    parser.add_argument(
        '--quick-cache',
        action='store_true',
        help='Rebuild ±1 year only, preserve long-range cache (~5 min)'
    )
    parser.add_argument(
        '--sync',
        action='store_true',
        help='Delete objects not in current catalog (destructive)'
    )
    parser.add_argument(
        '--mark-stale',
        action='store_true',
        help='Mark missing objects as stale with timestamp'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output (for cron jobs)'
    )

    args = parser.parse_args()

    # Run update
    success = update_catalog(
        fetch_moid=args.fetch_moid,
        clear_cache=args.clear_cache,
        rebuild_cache=args.rebuild_cache,
        quick_cache=args.quick_cache,
        sync_mode=args.sync,
        mark_stale=args.mark_stale,
        quiet=args.quiet
    )

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
