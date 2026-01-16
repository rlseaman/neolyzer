#!/usr/bin/env python3
"""
Build position cache for NEO visualizer
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from cache_manager import PositionCache, CacheBuilder
from database import DatabaseManager
from orbit_calculator import FastOrbitCalculator
from skyfield.api import load
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def main():
    """Build the position cache"""
    
    logger.info("=" * 70)
    logger.info("NEO Position Cache Builder")
    logger.info("=" * 70)
    logger.info("")
    
    # Check database
    db = DatabaseManager(use_sqlite=True)
    stats = db.get_statistics()
    
    if stats['total'] == 0:
        logger.error("Database is empty! Run setup_database.py first.")
        return 1
    
    logger.info(f"Found {stats['total']} asteroids in database")
    logger.info("")
    
    # Get all asteroids
    logger.info("Loading asteroids from database...")
    asteroids = db.get_asteroids()
    logger.info(f"Loaded {len(asteroids)} asteroids")
    logger.info("")
    
    # Initialize
    calculator = FastOrbitCalculator()
    cache = PositionCache()
    builder = CacheBuilder(cache, calculator)
    
    # Get current time as reference
    ts = load.timescale()
    now = ts.now()
    reference_jd = now.tt
    
    logger.info(f"Reference date: {now.utc_datetime()}")
    logger.info(f"Reference JD: {reference_jd:.2f}")
    logger.info("")
    
    # Build cache
    logger.info("Building position cache...")
    logger.info("This will take approximately 25-30 minutes")
    logger.info("")
    logger.info("Cache strategy:")
    logger.info("  • High precision (±6 months): daily positions")
    logger.info("  • Medium precision (±5 years): weekly positions")
    logger.info("  • Low precision (±27 years): monthly positions")
    logger.info("")
    
    try:
        builder.build_cache(asteroids, reference_jd, show_progress=True)
        
        # Show stats
        cache_stats = cache.get_cache_statistics()
        logger.info("")
        logger.info("=" * 70)
        logger.info("Cache Statistics:")
        logger.info(f"  File size: {cache_stats['file_size_mb']:.1f} MB")
        logger.info(f"  High precision dates: {cache_stats['high_precision_dates']}")
        logger.info(f"  Medium precision dates: {cache_stats['medium_precision_dates']}")
        logger.info(f"  Low precision dates: {cache_stats['low_precision_dates']}")
        logger.info(f"  Total objects: {cache_stats['n_objects']}")
        logger.info("=" * 70)
        logger.info("Cache build complete!")
        logger.info("=" * 70)
        
        return 0
        
    except Exception as e:
        logger.error(f"Cache build failed: {e}", exc_info=True)
        return 1

if __name__ == '__main__':
    sys.exit(main())
