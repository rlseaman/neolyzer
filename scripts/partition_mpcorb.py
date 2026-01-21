#!/usr/bin/env python3
"""
MPCORB.DAT Partition Script
Downloads and partitions MPCORB.DAT into separate files by dynamical class.

See MINOR_PLANET_CLASSES.txt for classification scheme documentation.

Usage:
    ./venv/bin/python scripts/partition_mpcorb.py [--clobber] [--local FILE]

Options:
    --clobber       Overwrite existing output files
    --local FILE    Use local MPCORB.DAT file instead of downloading
    --no-download   Skip download, use existing data/MPC/MPCORB.DAT
"""

import sys
import os
import gzip
import argparse
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import urllib.request
import shutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MPCORB_URL = "https://minorplanetcenter.net/iau/MPCORB/MPCORB.DAT.gz"
OUTPUT_DIR = Path("data/MPC")

# Classification codes and output filenames
# Note: MPCORB.DAT contains only minor planets (asteroids), not comets.
# Comets are in separate MPC files (COMET.DAT, not processed by this script).
# These 10 classes cover all objects in MPCORB.DAT by orbital elements.
CLASSES = [
    'NEO',  # Near-Earth Objects (q < 1.3 AU)
    'HUN',  # Hungaria family
    'PHO',  # Phocaea family
    'MCA',  # Mars Crossers
    'MB1',  # Inner Main Belt (a < 2.5 AU, catch-all)
    'MB2',  # Middle Main Belt (2.5 <= a < 2.8 AU)
    'MB3',  # Outer Main Belt (2.8 <= a < 3.25 AU)
    'HIL',  # Hilda group (3:2 resonance)
    'JTR',  # Jupiter Trojans (1:1 resonance)
    'OUT',  # Outer Solar System (a >= 3.25 AU, catch-all)
]


def is_numbered_asteroid(packed):
    """
    Check if a packed designation represents a numbered asteroid.

    MPC packed designation format:
    - Numbered asteroids: 5 characters or fewer (after stripping spaces)
      - 00001-99999: 5 digits with leading zeros
      - 100000+: Letter prefix + 4 chars (A0001, B0001, ..., ~xxxx)
    - Provisional designations: 7 characters
      - Century code (I=1800s, J=1900s, K=2000s) + year + half-month + serial

    The key distinction is length: numbered <= 5 chars, provisional = 7 chars.
    """
    if not packed:
        return False

    # After strip(), numbered asteroids are <= 5 chars, provisional are 7
    return len(packed) <= 5


def parse_mpc_line(line):
    """
    Parse a line from MPCORB.DAT in MPC orbit format.

    Returns dict with orbital elements or None if line is a header/invalid.
    """
    # Skip empty lines and header lines
    if len(line) < 160:
        return None

    # Header lines typically start with specific patterns
    if line.startswith('-----') or line.startswith('Des\'t'):
        return None

    try:
        # MPC orbit format (fixed-width columns)
        # See: https://minorplanetcenter.net/iau/info/MPOrbitFormat.html

        packed_desig = line[0:7].strip()

        # Parse H magnitude (can be blank)
        h_str = line[8:13].strip()
        H = float(h_str) if h_str else None

        # Parse G slope (can be blank)
        g_str = line[14:19].strip()
        G = float(g_str) if g_str else 0.15

        # Epoch (packed format)
        epoch = line[20:25].strip()

        # Mean anomaly (degrees)
        M_str = line[26:35].strip()
        M = float(M_str) if M_str else None

        # Argument of perihelion (degrees)
        arg_peri_str = line[37:46].strip()
        arg_peri = float(arg_peri_str) if arg_peri_str else None

        # Longitude of ascending node (degrees)
        node_str = line[48:57].strip()
        node = float(node_str) if node_str else None

        # Inclination (degrees)
        inc_str = line[59:68].strip()
        i = float(inc_str) if inc_str else None

        # Eccentricity
        e_str = line[70:79].strip()
        e = float(e_str) if e_str else None

        # Mean daily motion (degrees/day) - can be used to get semi-major axis
        n_str = line[80:91].strip()
        n = float(n_str) if n_str else None

        # Semi-major axis (AU)
        a_str = line[92:103].strip()
        a = float(a_str) if a_str else None

        # If a is missing but n is available, calculate a
        if a is None and n is not None and n > 0:
            # n (deg/day) -> period (years) -> a (AU)
            # n = 0.9856076686 / a^1.5 (Gauss's constant)
            # a = (0.9856076686 / n)^(2/3)
            k = 0.9856076686  # degrees per day for a=1 AU
            a = (k / n) ** (2/3)

        if a is None or e is None or i is None:
            return None

        # Calculate derived quantities
        q = a * (1 - e)  # Perihelion
        Q = a * (1 + e)  # Aphelion
        P = a ** 1.5     # Orbital period in years

        # Check if numbered or provisional
        numbered = is_numbered_asteroid(packed_desig)

        return {
            'designation': packed_desig,
            'H': H,
            'G': G,
            'a': a,
            'e': e,
            'i': i,
            'q': q,
            'Q': Q,
            'P': P,
            'M': M,
            'arg_peri': arg_peri,
            'node': node,
            'numbered': numbered,
            'line': line,
        }

    except (ValueError, IndexError) as ex:
        logger.debug(f"Failed to parse line: {line[:50]}... ({ex})")
        return None


def classify_object(obj):
    """
    Classify an object into one of the 10 dynamical classes.

    Classification is applied in order; first match wins.
    Returns class code (e.g., 'NEO', 'MB1', etc.)

    Note: These 10 classes are exhaustive for MPCORB.DAT objects.
    """
    a = obj['a']
    e = obj['e']
    i = obj['i']
    q = obj['q']
    Q = obj['Q']

    # NEO: q < 1.3 AU
    if q < 1.3:
        return 'NEO'

    # Hungaria: 1.78 < a < 2.0, e <= 0.18, 16 <= i <= 34
    if 1.78 < a < 2.0 and e <= 0.18 and 16 <= i <= 34:
        return 'HUN'

    # Phocaea: q >= 1.5, 2.2 < a < 2.45, 20 <= i <= 27
    if q >= 1.5 and 2.2 < a < 2.45 and 20 <= i <= 27:
        return 'PHO'

    # Mars Crosser: 1.3 <= q < 1.67, Q > 1.58
    if 1.3 <= q < 1.67 and Q > 1.58:
        return 'MCA'

    # Inner Main Belt: a < 2.5 (catch-all for inner solar system)
    if a < 2.5:
        return 'MB1'

    # Middle Main Belt: 2.5 <= a < 2.8
    if 2.5 <= a < 2.8:
        return 'MB2'

    # Outer Main Belt: 2.8 <= a < 3.25
    if 2.8 <= a < 3.25:
        return 'MB3'

    # Hilda: 3.9 < a < 4.02, e <= 0.4, i <= 18
    if 3.9 < a < 4.02 and e <= 0.4 and i <= 18:
        return 'HIL'

    # Jupiter Trojan: 5.05 < a < 5.35, e <= 0.22, i <= 38
    if 5.05 < a < 5.35 and e <= 0.22 and i <= 38:
        return 'JTR'

    # Outer Solar System: a >= 3.25 (catch-all for everything else)
    # This includes Cybeles, Thule, Centaurs, TNOs, SDOs, etc.
    return 'OUT'


def download_mpcorb(output_path):
    """Download MPCORB.DAT.gz and decompress it."""
    gz_path = output_path.with_suffix('.DAT.gz')

    logger.info(f"Downloading MPCORB.DAT.gz from MPC...")
    logger.info(f"URL: {MPCORB_URL}")

    try:
        # Download with progress indication
        with urllib.request.urlopen(MPCORB_URL, timeout=300) as response:
            total_size = response.getheader('Content-Length')
            if total_size:
                total_size = int(total_size)
                logger.info(f"File size: {total_size / 1024 / 1024:.1f} MB")

            with open(gz_path, 'wb') as f:
                downloaded = 0
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size and downloaded % (10 * 1024 * 1024) < 8192:
                        pct = 100 * downloaded / total_size
                        logger.info(f"  {pct:.0f}% ({downloaded / 1024 / 1024:.1f} MB)")

        logger.info(f"Download complete: {gz_path}")

        # Decompress
        logger.info("Decompressing...")
        with gzip.open(gz_path, 'rb') as f_in:
            with open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Remove compressed file
        gz_path.unlink()

        logger.info(f"Decompressed to: {output_path}")
        return True

    except Exception as ex:
        logger.error(f"Download failed: {ex}")
        return False


def partition_mpcorb(input_path, output_dir, clobber=False):
    """
    Partition MPCORB.DAT into separate files by class.

    Returns dict of statistics.
    """
    # Check for existing output files
    output_files = {cls: output_dir / f"{cls}_MPC.txt" for cls in CLASSES}

    if not clobber:
        existing = [f for f in output_files.values() if f.exists()]
        if existing:
            logger.error("Output files already exist (use --clobber to overwrite):")
            for f in existing[:5]:
                logger.error(f"  {f}")
            if len(existing) > 5:
                logger.error(f"  ... and {len(existing) - 5} more")
            return None

    # Initialize counters and file handles
    counts = defaultdict(int)
    file_handles = {}

    # Statistics tracking
    stats = {
        'total_lines': 0,
        'header_lines': 0,
        'parsed_objects': 0,
        'parse_errors': 0,
        'by_class': defaultdict(int),
        'numbered_asteroids': 0,
        'provisional_asteroids': 0,
        'a_min': float('inf'),
        'a_max': float('-inf'),
        'e_max': 0,
        'i_max': 0,
        'q_min': float('inf'),
        'Q_max': float('-inf'),
        'interesting_objects': [],
    }

    try:
        # Open all output files
        output_dir.mkdir(parents=True, exist_ok=True)
        for cls in CLASSES:
            file_handles[cls] = open(output_files[cls], 'w')

        # Process input file
        logger.info(f"Processing {input_path}...")

        with open(input_path, 'r', encoding='latin-1') as f:
            for line in f:
                stats['total_lines'] += 1

                # Parse the line
                obj = parse_mpc_line(line)

                if obj is None:
                    stats['header_lines'] += 1
                    continue

                stats['parsed_objects'] += 1
                desig = obj['designation']

                # Track numbered vs provisional
                if obj['numbered']:
                    stats['numbered_asteroids'] += 1
                else:
                    stats['provisional_asteroids'] += 1

                # Track extremes
                if obj['a'] < stats['a_min']:
                    stats['a_min'] = obj['a']
                if obj['a'] > stats['a_max']:
                    stats['a_max'] = obj['a']
                    if obj['a'] > 100:
                        stats['interesting_objects'].append(
                            f"Extreme a={obj['a']:.1f} AU: {desig}"
                        )
                if obj['e'] > stats['e_max']:
                    stats['e_max'] = obj['e']
                if obj['i'] > stats['i_max']:
                    stats['i_max'] = obj['i']
                if obj['q'] < stats['q_min']:
                    stats['q_min'] = obj['q']
                if obj['Q'] > stats['Q_max']:
                    stats['Q_max'] = obj['Q']

                # Track interesting objects
                if obj['q'] < 0.1:
                    stats['interesting_objects'].append(
                        f"Very small q={obj['q']:.4f} AU: {desig}"
                    )
                if obj['e'] > 0.99:
                    stats['interesting_objects'].append(
                        f"Near-parabolic e={obj['e']:.6f}: {desig}"
                    )

                # Classify and write
                cls = classify_object(obj)
                stats['by_class'][cls] += 1
                file_handles[cls].write(line)

                # Progress
                if stats['parsed_objects'] % 100000 == 0:
                    logger.info(f"  Processed {stats['parsed_objects']:,} objects...")

        logger.info(f"Processed {stats['parsed_objects']:,} objects")

    finally:
        # Close all file handles
        for fh in file_handles.values():
            fh.close()

    return stats


def write_statistics(stats, output_dir):
    """Write statistics to a summary file."""
    stats_file = output_dir / "MPC_partition_stats.txt"

    with open(stats_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("MPCORB.DAT PARTITION STATISTICS\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("=" * 70 + "\n\n")

        f.write("INPUT SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total lines:        {stats['total_lines']:>12,}\n")
        f.write(f"Header lines:       {stats['header_lines']:>12,}\n")
        f.write(f"Parsed objects:     {stats['parsed_objects']:>12,}\n")
        f.write(f"Parse errors:       {stats['parse_errors']:>12,}\n")
        f.write("\n")

        f.write("DESIGNATION TYPES\n")
        f.write("-" * 40 + "\n")
        f.write(f"Numbered asteroids: {stats['numbered_asteroids']:>12,}\n")
        f.write(f"Provisional:        {stats['provisional_asteroids']:>12,}\n")
        f.write("\n")

        f.write("BY DYNAMICAL CLASS\n")
        f.write("-" * 40 + "\n")
        total = 0
        for cls in CLASSES:
            count = stats['by_class'].get(cls, 0)
            total += count
            if count > 0:
                pct = 100 * count / stats['parsed_objects']
                f.write(f"  {cls:<6} {count:>12,}  ({pct:>5.2f}%)\n")
        f.write("-" * 40 + "\n")
        f.write(f"  TOTAL  {total:>12,}\n")
        f.write("\n")

        f.write("ORBITAL ELEMENT EXTREMES\n")
        f.write("-" * 40 + "\n")
        f.write(f"Semi-major axis (a): {stats['a_min']:.4f} - {stats['a_max']:.1f} AU\n")
        f.write(f"Perihelion (q):      {stats['q_min']:.4f} AU (minimum)\n")
        f.write(f"Aphelion (Q):        {stats['Q_max']:.1f} AU (maximum)\n")
        f.write(f"Eccentricity (e):    {stats['e_max']:.6f} (maximum)\n")
        f.write(f"Inclination (i):     {stats['i_max']:.2f}° (maximum)\n")
        f.write("\n")

        if stats['interesting_objects']:
            f.write("INTERESTING OBJECTS\n")
            f.write("-" * 40 + "\n")
            for item in stats['interesting_objects'][:50]:  # Limit to 50
                f.write(f"  {item}\n")
            if len(stats['interesting_objects']) > 50:
                f.write(f"  ... and {len(stats['interesting_objects']) - 50} more\n")
            f.write("\n")

        f.write("OUTPUT FILES\n")
        f.write("-" * 40 + "\n")
        for cls in CLASSES:
            count = stats['by_class'].get(cls, 0)
            if count > 0:
                f.write(f"  {cls}_MPC.txt: {count:,} objects\n")
        f.write("\n")

        f.write("=" * 70 + "\n")

    logger.info(f"Statistics written to: {stats_file}")
    return stats_file


def main():
    parser = argparse.ArgumentParser(
        description='Partition MPCORB.DAT into dynamical class files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                      # Download and partition
  %(prog)s --clobber            # Overwrite existing files
  %(prog)s --local MPCORB.DAT   # Use local file
  %(prog)s --no-download        # Use existing data/MPC/MPCORB.DAT
        """
    )
    parser.add_argument(
        '--clobber',
        action='store_true',
        help='Overwrite existing output files'
    )
    parser.add_argument(
        '--local',
        metavar='FILE',
        help='Use local MPCORB.DAT file instead of downloading'
    )
    parser.add_argument(
        '--no-download',
        action='store_true',
        help='Skip download, use existing data/MPC/MPCORB.DAT'
    )

    args = parser.parse_args()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Determine input file
    if args.local:
        input_path = Path(args.local)
        if not input_path.exists():
            logger.error(f"Local file not found: {input_path}")
            return 1
    elif args.no_download:
        input_path = OUTPUT_DIR / "MPCORB.DAT"
        if not input_path.exists():
            logger.error(f"MPCORB.DAT not found: {input_path}")
            logger.error("Run without --no-download to download it first")
            return 1
    else:
        input_path = OUTPUT_DIR / "MPCORB.DAT"
        if not download_mpcorb(input_path):
            return 1

    # Partition the file
    logger.info("=" * 60)
    logger.info("PARTITIONING MPCORB.DAT")
    logger.info("=" * 60)

    stats = partition_mpcorb(input_path, OUTPUT_DIR, clobber=args.clobber)

    if stats is None:
        return 1

    # Write statistics
    write_statistics(stats, OUTPUT_DIR)

    # Summary
    logger.info("=" * 60)
    logger.info("PARTITION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total objects: {stats['parsed_objects']:,}")
    logger.info(f"Output directory: {OUTPUT_DIR}")

    # Print class counts
    for cls in CLASSES:
        count = stats['by_class'].get(cls, 0)
        if count > 0:
            logger.info(f"  {cls}: {count:,}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
