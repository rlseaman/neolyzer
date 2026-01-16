#!/usr/bin/env python3
"""
Daily Catalog Update Script
Downloads latest NEO data and updates the database
Can be run as a cron job for automated daily updates
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from mpc_loader import load_asteroids_from_mpc
from database import DatabaseManager
from cache_manager import PositionCache, CacheBuilder
from orbit_calculator import FastOrbitCalculator
from skyfield.api import load

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/catalog_update.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def update_catalog(rebuild_cache: bool = False, use_postgres: bool = False):
    """
    Update the asteroid catalog
    
    Parameters:
    -----------
    rebuild_cache : bool
        Rebuild the entire position cache (slow)
    use_postgres : bool
        Use PostgreSQL instead of SQLite
    """
    logger.info("=" * 70)
    logger.info("NEO Catalog Update - " + datetime.now().isoformat())
    logger.info("=" * 70)
    
    # Download latest data
    logger.info("Downloading latest NEO data...")
    try:
        asteroids = load_asteroids_from_mpc(
            neo_only=True,
            force_download=True  # Always download fresh data
        )
        logger.info(f"Downloaded {len(asteroids)} NEO orbits")
    except Exception as e:
        logger.error(f"Failed to download data: {e}")
        return False
    
    # Update database
    logger.info("Updating database...")
    try:
        db = DatabaseManager(use_sqlite=not use_postgres)
        
        # Get old count
        old_stats = db.get_statistics()
        old_count = old_stats['total']
        
        # Insert/update asteroids
        db.insert_asteroids(asteroids)
        
        # Get new count
        new_stats = db.get_statistics()
        new_count = new_stats['total']
        
        logger.info(f"Database updated: {old_count} → {new_count} asteroids")
        logger.info(f"NEOs: {new_stats['neos']}, PHAs: {new_stats['phas']}")
        
        # Detect new objects
        if new_count > old_count:
            logger.info(f"🌟 {new_count - old_count} new objects discovered!")
        
    except Exception as e:
        logger.error(f"Database update failed: {e}")
        return False
    
    # Update cache if requested
    if rebuild_cache:
        logger.info("Rebuilding position cache...")
        try:
            calculator = FastOrbitCalculator()
            cache = PositionCache()
            builder = CacheBuilder(cache, calculator)
            
            # Use current time as reference
            ts = load.timescale()
            now = ts.now()
            reference_jd = now.tt
            
            logger.info(f"Reference JD: {reference_jd:.2f}")
            
            # Clear old cache
            cache.clear_cache()
            
            # Rebuild
            builder.build_cache(asteroids, reference_jd, show_progress=True)
            
            stats = cache.get_cache_statistics()
            logger.info(f"Cache rebuilt: {stats['file_size_mb']:.1f} MB")
            
        except Exception as e:
            logger.error(f"Cache rebuild failed: {e}")
            return False
    else:
        logger.info("Skipping cache rebuild (use --rebuild-cache to rebuild)")
    
    logger.info("=" * 70)
    logger.info("Update complete!")
    logger.info("=" * 70)
    
    return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Update NEO catalog from MPC'
    )
    parser.add_argument(
        '--rebuild-cache',
        action='store_true',
        help='Rebuild the entire position cache (slow, ~30 min)'
    )
    parser.add_argument(
        '--postgres',
        action='store_true',
        help='Use PostgreSQL instead of SQLite'
    )
    
    args = parser.parse_args()
    
    # Create logs directory
    Path('logs').mkdir(exist_ok=True)
    
    # Run update
    success = update_catalog(
        rebuild_cache=args.rebuild_cache,
        use_postgres=args.postgres
    )
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
