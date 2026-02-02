"""
Skyfield Loader - Centralized loader with SSL fallback

This module provides a Skyfield Loader that:
1. Uses a consistent cache directory (~/.skyfield)
2. Downloads ephemeris files with SSL certificate fallback
3. Works on systems with incomplete CA certificate stores (Raspberry Pi, etc.)
4. Supports configurable ephemeris selection (DE421, DE440, DE441)

Usage:
    from skyfield_loader import load_ephemeris, get_current_ephemeris

    # Load the configured ephemeris (downloads with SSL fallback if needed)
    eph = load_ephemeris()

    # Or load a specific ephemeris
    eph = load_ephemeris('de440.bsp')

    # Get timescale
    from skyfield.api import load
    ts = load.timescale()
"""

import logging
from pathlib import Path

from skyfield.api import Loader

logger = logging.getLogger(__name__)

# Create a global Skyfield loader with a fixed directory
# This ensures we always look for/save ephemeris files in the same place
SKYFIELD_DATA_DIR = Path.home() / '.skyfield'
SKYFIELD_DATA_DIR.mkdir(parents=True, exist_ok=True)

# The global loader instance (for timescale, etc.)
skyfield_load = Loader(str(SKYFIELD_DATA_DIR))

# Cached timescale to avoid repeated file opens
# skyfield_load.timescale() opens data files each time; cache to prevent file handle exhaustion
_cached_timescale = None

# Cached ephemeris to avoid repeated file opens
# load_ephemeris() opens the .bsp file each time; cache to prevent file handle exhaustion
_cached_ephemeris = None
_cached_ephemeris_filename = None

def get_timescale():
    """
    Get a cached Skyfield timescale object.

    This avoids repeated calls to skyfield_load.timescale() which can open
    file handles that accumulate over time, especially during rapid updates
    like blink mode.

    Returns:
    --------
    Timescale : Skyfield timescale object
    """
    global _cached_timescale
    if _cached_timescale is None:
        _cached_timescale = skyfield_load.timescale()
    return _cached_timescale


def get_current_ephemeris() -> str:
    """
    Get the currently configured ephemeris filename.

    Returns:
    --------
    str : Ephemeris filename (e.g., 'de440.bsp')
    """
    try:
        from ephemeris_config import get_configured_ephemeris
        return get_configured_ephemeris()
    except ImportError:
        # Fallback if ephemeris_config not available
        return 'de440.bsp'


def ensure_ephemeris(filename=None):
    """
    Ensure the ephemeris file exists, downloading with SSL fallback if needed.

    Skyfield's internal downloader uses urllib which may fail on systems
    with incomplete CA certificate stores (like Raspberry Pi).
    This function downloads the file ourselves with fallback options.

    Parameters:
    -----------
    filename : str, optional
        Name of the ephemeris file. If None, uses configured default.

    Returns:
    --------
    Path : Path to the ephemeris file
    """
    if filename is None:
        filename = get_current_ephemeris()

    cache_path = SKYFIELD_DATA_DIR / filename

    if cache_path.exists():
        logger.debug(f"Ephemeris file already exists: {cache_path}")
        return cache_path

    # Need to download
    url = f"https://ssd.jpl.nasa.gov/ftp/eph/planets/bsp/{filename}"
    logger.info(f"Downloading ephemeris file: {filename}")

    import requests

    # Try to download with SSL fallback
    response = None

    # First try: normal SSL verification
    try:
        response = requests.get(url, stream=True, timeout=300)
        response.raise_for_status()
        logger.info("Downloaded with normal SSL verification")
    except requests.exceptions.SSLError as e:
        logger.warning(f"SSL verification failed: {e}")

        # Second try: use certifi if available
        try:
            import certifi
            logger.info("Retrying with certifi certificate bundle...")
            response = requests.get(url, stream=True, timeout=300,
                                   verify=certifi.where())
            response.raise_for_status()
            logger.info("Downloaded with certifi certificates")
        except ImportError:
            logger.info("certifi not installed")
            response = None
        except requests.exceptions.SSLError:
            logger.warning("Still failing with certifi")
            response = None
        except Exception:
            response = None

        # Third try: disable SSL verification
        if response is None:
            logger.warning("Downloading without SSL verification (less secure)...")
            try:
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            except:
                pass
            response = requests.get(url, stream=True, timeout=300, verify=False)
            response.raise_for_status()
            logger.info("Downloaded without SSL verification")

    # Save the file
    total_size = int(response.headers.get('content-length', 0))
    logger.info(f"Downloading {total_size / 1024 / 1024:.1f} MB...")

    with open(cache_path, 'wb') as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = 100 * downloaded / total_size
                if downloaded % (1024 * 1024) < 8192:  # Log every ~1MB
                    logger.info(f"  {pct:.0f}% ({downloaded / 1024 / 1024:.1f} MB)")

    logger.info(f"Ephemeris downloaded to {cache_path}")
    return cache_path


def load_ephemeris(filename=None):
    """
    Load an ephemeris file, downloading with SSL fallback if needed.

    This function ensures the file is downloaded first (handling SSL issues),
    then loads it using a Loader pointed at the directory where the file exists.
    The loaded ephemeris is cached to prevent file handle exhaustion from
    repeated opens during animation/updates.

    Parameters:
    -----------
    filename : str, optional
        Name of the ephemeris file. If None, uses configured default.

    Returns:
    --------
    SpiceKernel : The loaded ephemeris
    """
    global _cached_ephemeris, _cached_ephemeris_filename

    if filename is None:
        filename = get_current_ephemeris()

    # Return cached ephemeris if same file
    if _cached_ephemeris is not None and _cached_ephemeris_filename == filename:
        return _cached_ephemeris

    # Make sure the file exists (downloads if needed with SSL fallback)
    eph_path = ensure_ephemeris(filename)

    # Create a Loader pointed at the directory containing the file
    # Since the file already exists, the Loader won't try to download
    from skyfield.api import Loader
    loader = Loader(str(eph_path.parent))

    # Load the file - since it exists, no download attempt will be made
    _cached_ephemeris = loader(filename)
    _cached_ephemeris_filename = filename

    return _cached_ephemeris


def clear_ephemeris_cache():
    """Clear the cached ephemeris (call when switching ephemeris files)."""
    global _cached_ephemeris, _cached_ephemeris_filename
    _cached_ephemeris = None
    _cached_ephemeris_filename = None
