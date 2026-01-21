"""
Ephemeris Configuration - Centralized JPL ephemeris selection and metadata

This module provides:
1. Metadata for supported JPL Development Ephemerides (DE)
2. Configuration storage for user's ephemeris selection
3. Helper functions to get current ephemeris settings

Supported ephemerides:
- DE421: 1900-2050, 17MB - Original default, compact
- DE440: 1550-2650, 115MB - Latest planetary data, recommended upgrade
- DE441: -13200 to +17191, 3.5GB - Long-span version (optional)
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Ephemeris metadata: file sizes, date ranges, JD bounds
# JD values are approximate start/end of valid range
EPHEMERIS_DATA = {
    'de421.bsp': {
        'name': 'DE421',
        'description': 'Standard ephemeris (2008)',
        'year_range': (1900, 2050),
        'jd_min': 2414864,   # 1899-07-09
        'jd_max': 2471184,   # 2053-10-08
        'file_size_mb': 17,
        'download_time_min': 1,
        'notes': 'Compact, widely used, sufficient for most applications'
    },
    'de440.bsp': {
        'name': 'DE440',
        'description': 'Latest planetary ephemeris (2020)',
        'year_range': (1550, 2650),
        'jd_min': 2287184,   # 1549-12-21
        'jd_max': 2688976,   # 2650-01-25
        'file_size_mb': 115,
        'download_time_min': 5,
        'notes': 'Improved accuracy, extended range, includes LRO lunar data'
    },
    'de441.bsp': {
        'name': 'DE441',
        'description': 'Long-span ephemeris (2020)',
        'year_range': (-13200, 17191),
        'jd_min': -3027215,  # -13200
        'jd_max': 8000000,   # ~17191
        'file_size_mb': 3500,
        'download_time_min': 60,
        'notes': 'Extended time span for historical/far-future work (very large)'
    }
}

# Default ephemeris (used if no config exists)
DEFAULT_EPHEMERIS = 'de440.bsp'

# Config file location
CONFIG_DIR = Path.home() / '.neolyzer'
CONFIG_FILE = CONFIG_DIR / 'ephemeris.json'


def get_ephemeris_info(filename: str) -> Dict:
    """Get metadata for a specific ephemeris file."""
    return EPHEMERIS_DATA.get(filename, EPHEMERIS_DATA[DEFAULT_EPHEMERIS])


def get_available_ephemerides() -> Dict[str, Dict]:
    """Get all available ephemeris options."""
    return EPHEMERIS_DATA


# Cache for configured ephemeris to avoid repeated file reads
_cached_ephemeris = None

def get_configured_ephemeris() -> str:
    """
    Get the currently configured ephemeris filename.
    Returns default if no config exists.
    Result is cached after first read to prevent file handle exhaustion.
    """
    global _cached_ephemeris
    if _cached_ephemeris is not None:
        return _cached_ephemeris

    try:
        if CONFIG_FILE.exists():
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                filename = config.get('ephemeris', DEFAULT_EPHEMERIS)
                if filename in EPHEMERIS_DATA:
                    _cached_ephemeris = filename
                    return filename
                else:
                    logger.warning(f"Unknown ephemeris in config: {filename}, using default")
    except Exception as e:
        logger.warning(f"Error reading ephemeris config: {e}")

    _cached_ephemeris = DEFAULT_EPHEMERIS
    return DEFAULT_EPHEMERIS


def clear_ephemeris_cache():
    """Clear the cached ephemeris setting (call after changing config)."""
    global _cached_ephemeris
    _cached_ephemeris = None


def set_configured_ephemeris(filename: str) -> bool:
    """
    Set the configured ephemeris.

    Parameters:
    -----------
    filename : str
        Ephemeris filename (e.g., 'de440.bsp')

    Returns:
    --------
    bool : True if successful
    """
    global _cached_ephemeris

    if filename not in EPHEMERIS_DATA:
        logger.error(f"Unknown ephemeris: {filename}")
        return False

    try:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        config = {'ephemeris': filename}

        # Preserve other config settings if they exist
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, 'r') as f:
                    existing = json.load(f)
                    existing['ephemeris'] = filename
                    config = existing
            except:
                pass

        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)

        # Clear cache so next read gets the new value
        _cached_ephemeris = filename

        logger.info(f"Ephemeris set to: {filename}")
        return True
    except Exception as e:
        logger.error(f"Error saving ephemeris config: {e}")
        return False


# Cache for ephemeris bounds
_cached_bounds = None

def get_ephemeris_bounds() -> Tuple[float, float]:
    """
    Get the JD bounds for the currently configured ephemeris.
    Includes a 10-day safety buffer on each end.
    Result is cached after first calculation.

    Returns:
    --------
    Tuple[float, float] : (jd_min, jd_max) with safety buffers
    """
    global _cached_bounds
    if _cached_bounds is not None:
        return _cached_bounds

    filename = get_configured_ephemeris()
    info = get_ephemeris_info(filename)

    # Add 10-day safety buffer
    jd_min = info['jd_min'] + 10
    jd_max = info['jd_max'] - 10

    _cached_bounds = (jd_min, jd_max)
    return _cached_bounds


def get_ephemeris_year_range() -> Tuple[int, int]:
    """
    Get the year range for the currently configured ephemeris.

    Returns:
    --------
    Tuple[int, int] : (start_year, end_year)
    """
    filename = get_configured_ephemeris()
    info = get_ephemeris_info(filename)
    return info['year_range']


def format_ephemeris_table() -> str:
    """
    Format a text table of available ephemerides for display.

    Returns:
    --------
    str : Formatted table
    """
    lines = []
    lines.append("Available JPL Ephemerides:")
    lines.append("-" * 75)
    lines.append(f"{'Name':<10} {'Years':<20} {'Size':<10} {'Download':<12} {'Notes'}")
    lines.append("-" * 75)

    for filename, info in EPHEMERIS_DATA.items():
        year_range = f"{info['year_range'][0]} to {info['year_range'][1]}"
        size = f"{info['file_size_mb']} MB"
        time = f"~{info['download_time_min']} min"
        lines.append(f"{info['name']:<10} {year_range:<20} {size:<10} {time:<12} {info['notes'][:30]}")

    lines.append("-" * 75)
    return "\n".join(lines)
