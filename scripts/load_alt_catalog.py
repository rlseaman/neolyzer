#!/usr/bin/env python3
"""
Alternate Catalog Loader
Load an NEA.txt-format file as an alternate catalog for comparison/blinking

Usage:
    ./venv/bin/python scripts/load_alt_catalog.py <nea_file> [options]

Options:
    --name NAME         Catalog name (default: derived from filename)
    --description DESC  Optional description
    --fetch-moid        Fetch Earth MOID from JPL SBDB
    --load-discovery    Load discovery circumstances if available
    --build-cache       Build position cache for this catalog
    --quick-cache       Build only ±1 year high-precision cache
    --replace           Replace existing catalog with same name
    --list              List existing alternate catalogs
    --delete NAME       Delete an alternate catalog
    --info NAME         Show info about a catalog
"""

import sys
import logging
import argparse
import os
from pathlib import Path
from datetime import datetime, timezone

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from mpc_loader import MPCLoader
from database import DatabaseManager, load_discovery_tracklets, fetch_moid_batch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def sanitize_name(name: str) -> str:
    """Sanitize a catalog name for use as identifier"""
    # Remove/replace invalid characters
    import re
    name = re.sub(r'[^\w\-]', '_', name)
    name = re.sub(r'_+', '_', name)  # Collapse multiple underscores
    name = name.strip('_')
    return name[:50]  # Max 50 chars


def derive_name_from_file(filepath: str) -> str:
    """Derive a catalog name from the filename"""
    basename = os.path.basename(filepath)
    name, _ = os.path.splitext(basename)
    return sanitize_name(name)


def list_catalogs(db: DatabaseManager):
    """List all alternate catalogs"""
    catalogs = db.list_catalogs()

    if not catalogs:
        print("\nNo alternate catalogs found.")
        print("Use 'load_alt_catalog.py <nea_file>' to load one.")
        return

    print("\n" + "=" * 70)
    print("ALTERNATE CATALOGS")
    print("=" * 70)
    print(f"{'Name':<20} {'Objects':>10} {'MOID':>6} {'Disc':>6} {'Cache':>8} {'Created':<20}")
    print("-" * 70)

    for c in catalogs:
        moid_str = "Yes" if c['has_moid'] else "No"
        disc_str = "Yes" if c['has_discovery'] else "No"
        cache_str = "Yes" if c['cache_file'] else "No"
        created = c['created_at'].strftime('%Y-%m-%d %H:%M') if c['created_at'] else 'Unknown'
        print(f"{c['name']:<20} {c['object_count']:>10} {moid_str:>6} {disc_str:>6} {cache_str:>8} {created:<20}")

    print("-" * 70)
    print(f"Total: {len(catalogs)} catalog(s)")
    print()


def show_catalog_info(db: DatabaseManager, name: str):
    """Show detailed info about a catalog"""
    catalog = db.get_catalog(name)

    if not catalog:
        print(f"\nCatalog '{name}' not found.")
        return

    print("\n" + "=" * 70)
    print(f"CATALOG: {catalog['name']}")
    print("=" * 70)
    print(f"  ID:            {catalog['id']}")
    print(f"  Source file:   {catalog['source_file'] or 'Unknown'}")
    print(f"  Description:   {catalog['description'] or 'None'}")
    print(f"  Object count:  {catalog['object_count']}")
    print(f"  Has MOID:      {'Yes' if catalog['has_moid'] else 'No'}")
    print(f"  Has discovery: {'Yes' if catalog['has_discovery'] else 'No'}")
    print(f"  Cache file:    {catalog['cache_file'] or 'None'}")
    print(f"  Created:       {catalog['created_at']}")
    print(f"  Updated:       {catalog['updated_at']}")
    print()


def delete_catalog(db: DatabaseManager, name: str, force: bool = False):
    """Delete an alternate catalog"""
    catalog = db.get_catalog(name)

    if not catalog:
        print(f"\nCatalog '{name}' not found.")
        return False

    if not force:
        print(f"\nThis will delete catalog '{name}' with {catalog['object_count']} objects.")
        response = input("Are you sure? [y/N]: ").strip().lower()
        if response != 'y':
            print("Cancelled.")
            return False

    # Delete cache file if exists
    if catalog['cache_file'] and os.path.exists(catalog['cache_file']):
        try:
            os.remove(catalog['cache_file'])
            logger.info(f"Deleted cache file: {catalog['cache_file']}")
        except Exception as e:
            logger.warning(f"Could not delete cache file: {e}")

    # Delete from database
    if db.delete_catalog(name):
        print(f"Deleted catalog '{name}'")
        return True
    else:
        print(f"Failed to delete catalog '{name}'")
        return False


def load_alternate_catalog(
    filepath: str,
    name: str = None,
    description: str = None,
    fetch_moid: bool = False,
    load_discovery: bool = False,
    build_cache: bool = False,
    quick_cache: bool = False,
    replace: bool = False
):
    """
    Load an NEA.txt-format file as an alternate catalog

    Parameters:
    -----------
    filepath : str
        Path to NEA.txt-format file
    name : str
        Catalog name (default: derived from filename)
    description : str
        Optional description
    fetch_moid : bool
        Fetch Earth MOID from JPL SBDB
    load_discovery : bool
        Load discovery circumstances if CSV exists
    build_cache : bool
        Build full position cache
    quick_cache : bool
        Build quick (±1 year) position cache
    replace : bool
        Replace existing catalog with same name
    """
    # Validate file exists
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        return False

    # Derive name if not provided
    if not name:
        name = derive_name_from_file(filepath)

    logger.info("=" * 70)
    logger.info(f"Loading Alternate Catalog: {name}")
    logger.info("=" * 70)
    logger.info(f"Source file: {filepath}")

    # Initialize database
    db = DatabaseManager(use_sqlite=True)

    # Check if catalog already exists
    existing = db.get_catalog(name)
    if existing:
        if not replace:
            logger.error(f"Catalog '{name}' already exists. Use --replace to overwrite.")
            return False
        else:
            logger.info(f"Replacing existing catalog '{name}'")
            db.delete_catalog(name)

    # Parse the NEA.txt file
    logger.info("Parsing NEA.txt format file...")
    loader = MPCLoader()
    try:
        asteroids = loader.parse_mpc_format(
            Path(filepath),
            neo_only=True,  # Filter to NEOs
            max_records=None
        )
        logger.info(f"Parsed {len(asteroids)} NEOs")
    except Exception as e:
        logger.error(f"Failed to parse file: {e}")
        return False

    if not asteroids:
        logger.error("No asteroids found in file")
        return False

    # Fetch MOID data if requested
    has_moid = False
    if fetch_moid:
        logger.info("Fetching Earth MOID values from JPL SBDB...")
        try:
            fetch_moid_batch(asteroids, show_progress=True)
            # Check if any have MOID
            moid_count = sum(1 for a in asteroids if a.get('earth_moid') is not None)
            has_moid = moid_count > 0
            logger.info(f"MOID data: {moid_count}/{len(asteroids)} objects")
        except Exception as e:
            logger.error(f"MOID fetch failed: {e}")

    # Load discovery circumstances if requested
    has_discovery = False
    if load_discovery:
        discovery_csv = Path(__file__).parent.parent / 'data' / 'NEA_discovery_tracklets.csv'
        if discovery_csv.exists():
            logger.info("Loading discovery circumstances...")
            try:
                matched = load_discovery_tracklets(asteroids, str(discovery_csv), show_progress=True)
                has_discovery = matched > 0
            except Exception as e:
                logger.error(f"Discovery tracklet loading failed: {e}")
        else:
            logger.warning(f"Discovery CSV not found: {discovery_csv}")

    # Create catalog entry
    logger.info("Creating catalog entry...")
    try:
        catalog_id = db.create_catalog(
            name=name,
            source_file=os.path.abspath(filepath),
            description=description
        )
    except Exception as e:
        logger.error(f"Failed to create catalog: {e}")
        return False

    # Insert asteroids
    logger.info("Inserting asteroids into database...")
    try:
        db.insert_alternate_asteroids(catalog_id, asteroids)
    except Exception as e:
        logger.error(f"Failed to insert asteroids: {e}")
        db.delete_catalog(name)
        return False

    # Update catalog metadata
    db.update_catalog(
        catalog_id,
        has_moid=has_moid,
        has_discovery=has_discovery
    )

    # Build cache if requested
    cache_file = None
    if build_cache or quick_cache:
        logger.info("Building position cache...")
        try:
            from cache_manager import PositionCache, CacheBuilder
            from orbit_calculator import FastOrbitCalculator
            from skyfield.api import load

            # Create cache file path
            cache_file = f"cache/positions_{name}.h5"
            cache = PositionCache(cache_file=cache_file)
            calculator = FastOrbitCalculator()
            builder = CacheBuilder(cache, calculator)

            ts = load.timescale()
            reference_jd = ts.now().tt

            # Get asteroids for cache building
            alt_asteroids = db.get_alternate_asteroids(catalog_id)

            builder.build_cache(
                alt_asteroids,
                reference_jd,
                show_progress=True,
                high_precision_only=quick_cache
            )

            # Update catalog with cache file path
            db.update_catalog(catalog_id, cache_file=cache_file)

            logger.info(f"Cache built: {cache_file}")
        except Exception as e:
            logger.error(f"Cache building failed: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    logger.info("=" * 70)
    logger.info("LOAD COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Catalog name:  {name}")
    logger.info(f"  Objects:       {len(asteroids)}")
    logger.info(f"  Has MOID:      {'Yes' if has_moid else 'No'}")
    logger.info(f"  Has discovery: {'Yes' if has_discovery else 'No'}")
    logger.info(f"  Cache:         {cache_file or 'None (on-the-fly computation)'}")
    logger.info("=" * 70)

    return True


def main():
    parser = argparse.ArgumentParser(
        description='Load alternate NEA catalog for comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Load a catalog from file:
    %(prog)s alt_data/NEA.txt

  Load with custom name and MOID:
    %(prog)s alt_data/sim_2026.txt --name sim_jan2026 --fetch-moid

  Load with full cache:
    %(prog)s alt_data/NEA.txt --name backup_jan --build-cache

  List existing catalogs:
    %(prog)s --list

  Delete a catalog:
    %(prog)s --delete sim_jan2026
        """
    )

    parser.add_argument(
        'nea_file',
        nargs='?',
        help='Path to NEA.txt-format file to load'
    )
    parser.add_argument(
        '--name',
        help='Catalog name (default: derived from filename)'
    )
    parser.add_argument(
        '--description',
        help='Optional description for the catalog'
    )
    parser.add_argument(
        '--fetch-moid',
        action='store_true',
        help='Fetch Earth MOID values from JPL SBDB'
    )
    parser.add_argument(
        '--load-discovery',
        action='store_true',
        help='Load discovery circumstances if CSV available'
    )
    parser.add_argument(
        '--build-cache',
        action='store_true',
        help='Build full position cache for this catalog'
    )
    parser.add_argument(
        '--quick-cache',
        action='store_true',
        help='Build quick (±1 year) position cache'
    )
    parser.add_argument(
        '--replace',
        action='store_true',
        help='Replace existing catalog with same name'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List existing alternate catalogs'
    )
    parser.add_argument(
        '--delete',
        metavar='NAME',
        help='Delete an alternate catalog'
    )
    parser.add_argument(
        '--info',
        metavar='NAME',
        help='Show info about a catalog'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force delete without confirmation'
    )

    args = parser.parse_args()

    # Initialize database for list/delete/info operations
    db = DatabaseManager(use_sqlite=True)

    # Handle list
    if args.list:
        list_catalogs(db)
        return 0

    # Handle info
    if args.info:
        show_catalog_info(db, args.info)
        return 0

    # Handle delete
    if args.delete:
        success = delete_catalog(db, args.delete, force=args.force)
        return 0 if success else 1

    # Load catalog requires a file
    if not args.nea_file:
        parser.print_help()
        print("\nError: NEA file required for loading")
        return 1

    success = load_alternate_catalog(
        filepath=args.nea_file,
        name=args.name,
        description=args.description,
        fetch_moid=args.fetch_moid,
        load_discovery=args.load_discovery,
        build_cache=args.build_cache,
        quick_cache=args.quick_cache,
        replace=args.replace
    )

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
