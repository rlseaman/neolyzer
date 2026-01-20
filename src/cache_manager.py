"""
Cache Manager - Efficient storage and retrieval of pre-computed positions
Uses HDF5 for fast access and compression
Implements variable precision: daily within ±1 year, weekly outside
"""

import h5py
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple, List, Dict
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PositionCache:
    """
    Manage cached asteroid positions with variable precision
    
    Storage strategy:
    - High precision window (±1 year from reference): daily positions
    - Medium precision (±5 years): weekly positions  
    - Low precision (±50 years): monthly positions
    - On-demand computation for anything else
    
    HDF5 structure:
    /high_precision/YYYYMMDD/positions -> (N, 5) [id, ra, dec, dist, mag]
    /medium_precision/YYYYMMDD/positions -> (N, 5)
    /low_precision/YYYYMMDD/positions -> (N, 5)
    /metadata/asteroids -> (N,) array of designations
    /metadata/reference_jd -> scalar, reference date
    """
    
    def __init__(self, cache_file: str = "cache/positions.h5"):
        """
        Initialize position cache
        
        Parameters:
        -----------
        cache_file : str
            Path to HDF5 cache file
        """
        self.cache_file = Path(cache_file)
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Precision windows (days from reference)
        # Will be set dynamically based on ephemeris coverage
        self.HIGH_PRECISION_DAYS = 365  # ±1 year (daily positions)
        self.MEDIUM_PRECISION_DAYS = 1826  # ±5 years (weekly positions)
        
        # Low precision range will be set dynamically in set_reference_date()
        # to stay within ephemeris bounds
        self.LOW_PRECISION_DAYS_FORWARD = None
        self.LOW_PRECISION_DAYS_BACKWARD = None
        
        # Ephemeris safe bounds - loaded from configuration
        # Supports DE421, DE440, DE441 with different date ranges
        try:
            from ephemeris_config import get_ephemeris_bounds
            self.EPHEMERIS_JD_MIN, self.EPHEMERIS_JD_MAX = get_ephemeris_bounds()
        except ImportError:
            # Fallback to DE440 defaults if config not available
            self.EPHEMERIS_JD_MIN = 2287184 + 10  # 1550 + buffer
            self.EPHEMERIS_JD_MAX = 2688976 - 10  # 2650 - buffer
        
        # Sampling intervals
        self.HIGH_INTERVAL = 1  # daily
        self.MEDIUM_INTERVAL = 7  # weekly
        self.LOW_INTERVAL = 30  # monthly
        
        self._initialize_cache()
    
    def _initialize_cache(self):
        """Create cache file structure if it doesn't exist"""
        if not self.cache_file.exists():
            logger.info(f"Creating new cache file: {self.cache_file}")
            with h5py.File(self.cache_file, 'w') as f:
                # Create groups
                f.create_group('high_precision')
                f.create_group('medium_precision')
                f.create_group('low_precision')
                f.create_group('metadata')
                
                # Set reference date to J2000.0
                f['metadata'].attrs['reference_jd'] = 2451545.0
                f['metadata'].attrs['created'] = datetime.now(timezone.utc).isoformat()
    
    def set_reference_date(self, jd: float):
        """Set the reference date for cache (typically current date)
        
        Also calculates safe low precision ranges based on ephemeris coverage.
        """
        # Calculate safe ranges within ephemeris bounds
        max_backward = jd - self.EPHEMERIS_JD_MIN
        max_forward = self.EPHEMERIS_JD_MAX - jd
        
        # Use the maximum safe ranges (with buffers already applied)
        self.LOW_PRECISION_DAYS_BACKWARD = int(max_backward)
        self.LOW_PRECISION_DAYS_FORWARD = int(max_forward)
        
        logger.info(f"Set reference JD to {jd}")
        logger.info(f"Safe cache range: -{self.LOW_PRECISION_DAYS_BACKWARD} to +{self.LOW_PRECISION_DAYS_FORWARD} days")
        logger.info(f"Date range: {self._jd_to_date(jd - self.LOW_PRECISION_DAYS_BACKWARD)} to {self._jd_to_date(jd + self.LOW_PRECISION_DAYS_FORWARD)}")
        
        with h5py.File(self.cache_file, 'a') as f:
            f['metadata'].attrs['reference_jd'] = jd
            f['metadata'].attrs['range_backward_days'] = self.LOW_PRECISION_DAYS_BACKWARD
            f['metadata'].attrs['range_forward_days'] = self.LOW_PRECISION_DAYS_FORWARD
    
    def _jd_to_date(self, jd: float) -> str:
        """Convert JD to readable date string"""
        from skyfield_loader import skyfield_load
        ts = skyfield_load.timescale()
        t = ts.tt_jd(jd)
        return t.utc_datetime().strftime('%Y-%m-%d')
    
    def get_reference_date(self) -> float:
        """Get the reference Julian Date"""
        with h5py.File(self.cache_file, 'r') as f:
            return f['metadata'].attrs.get('reference_jd', 2451545.0)
    
    def store_positions(self, jd: float, positions: np.ndarray, 
                       designations: Optional[List[str]] = None):
        """
        Store positions for a given Julian Date
        
        Parameters:
        -----------
        jd : float
            Julian Date
        positions : ndarray
            Array of shape (N, 5) with [id, ra, dec, distance, magnitude]
        designations : list of str, optional
            Asteroid designations (stored in metadata on first call)
        """
        ref_jd = self.get_reference_date()
        days_from_ref = jd - ref_jd
        
        # Determine which precision tier
        group, interval = self._get_precision_group(days_from_ref)
        
        # Create date key
        date_key = f"JD{int(jd)}"
        
        with h5py.File(self.cache_file, 'a') as f:
            # Store positions with compression
            if date_key in f[group]:
                del f[group][date_key]
            
            dataset = f[group].create_dataset(
                date_key,
                data=positions,
                compression='gzip',
                compression_opts=4,
                chunks=True
            )
            dataset.attrs['jd'] = jd
            dataset.attrs['n_objects'] = len(positions)
            
            # Store designations in metadata (once)
            if designations and 'designations' not in f['metadata']:
                dt = h5py.string_dtype(encoding='utf-8')
                f['metadata'].create_dataset(
                    'designations',
                    data=np.array(designations, dtype=dt),
                    compression='gzip'
                )
    
    def get_positions(self, jd: float, 
                     interpolate: bool = True) -> Optional[np.ndarray]:
        """
        Retrieve positions for a given Julian Date
        
        Parameters:
        -----------
        jd : float
            Julian Date
        interpolate : bool
            If True, interpolate between cached dates
            
        Returns:
        --------
        ndarray : (N, 5) array of [id, ra, dec, distance, magnitude]
                  or None if not cached
        """
        ref_jd = self.get_reference_date()
        days_from_ref = jd - ref_jd
        
        # Check if out of cached range (asymmetric: more backward than forward)
        if days_from_ref < -self.LOW_PRECISION_DAYS_BACKWARD or days_from_ref > self.LOW_PRECISION_DAYS_FORWARD:
            return None
        
        group, interval = self._get_precision_group(days_from_ref)
        
        with h5py.File(self.cache_file, 'r') as f:
            if group not in f:
                return None
            
            # Find nearest cached date(s)
            cached_jds = []
            for key in f[group].keys():
                if key.startswith('JD'):
                    cached_jds.append(f[group][key].attrs['jd'])
            
            if not cached_jds:
                return None
            
            cached_jds = np.array(cached_jds)
            
            # Find exact match
            exact_match = np.where(np.abs(cached_jds - jd) < 0.01)[0]
            if len(exact_match) > 0:
                date_key = f"JD{int(cached_jds[exact_match[0]])}"
                return f[group][date_key][:]
            
            # Interpolate if requested
            if interpolate and len(cached_jds) >= 2:
                # Find bracketing dates
                before = cached_jds[cached_jds <= jd]
                after = cached_jds[cached_jds > jd]
                
                if len(before) > 0 and len(after) > 0:
                    jd1 = before.max()
                    jd2 = after.min()
                    
                    key1 = f"JD{int(jd1)}"
                    key2 = f"JD{int(jd2)}"
                    
                    pos1 = f[group][key1][:]
                    pos2 = f[group][key2][:]
                    
                    # Linear interpolation
                    t = (jd - jd1) / (jd2 - jd1)
                    positions = pos1 * (1 - t) + pos2 * t
                    
                    return positions
            
            # Return nearest if no interpolation
            nearest_idx = np.argmin(np.abs(cached_jds - jd))
            date_key = f"JD{int(cached_jds[nearest_idx])}"
            return f[group][date_key][:]
    
    def _get_precision_group(self, days_from_ref: float) -> Tuple[str, int]:
        """
        Determine which precision group and interval to use
        
        Returns:
        --------
        tuple : (group_name, interval_days)
        """
        abs_days = abs(days_from_ref)
        
        if abs_days <= self.HIGH_PRECISION_DAYS:
            return 'high_precision', self.HIGH_INTERVAL
        elif abs_days <= self.MEDIUM_PRECISION_DAYS:
            return 'medium_precision', self.MEDIUM_INTERVAL
        else:
            return 'low_precision', self.LOW_INTERVAL
    
    def get_cache_dates(self) -> Dict[str, List[float]]:
        """Get all cached Julian Dates by precision group"""
        result = {
            'high_precision': [],
            'medium_precision': [],
            'low_precision': []
        }
        
        with h5py.File(self.cache_file, 'r') as f:
            for group in result.keys():
                if group in f:
                    for key in f[group].keys():
                        if key.startswith('JD'):
                            result[group].append(f[group][key].attrs['jd'])
        
        return {k: sorted(v) for k, v in result.items()}
    
    def get_cache_statistics(self) -> Dict:
        """Get statistics about cached data"""
        stats = {
            'file_size_mb': self.cache_file.stat().st_size / (1024**2) if self.cache_file.exists() else 0,
            'reference_jd': self.get_reference_date(),
        }
        
        with h5py.File(self.cache_file, 'r') as f:
            # Count dates in each group
            for group in ['high_precision', 'medium_precision', 'low_precision']:
                if group in f:
                    stats[f'{group}_dates'] = len([k for k in f[group].keys() if k.startswith('JD')])
                else:
                    stats[f'{group}_dates'] = 0
            
            # Get number of objects
            if 'designations' in f['metadata']:
                stats['n_objects'] = len(f['metadata']['designations'])
            else:
                stats['n_objects'] = 0
        
        return stats
    
    def clear_cache(self, groups: list = None):
        """Clear cached positions (keeps structure)

        Parameters:
        -----------
        groups : list, optional
            List of precision groups to clear: ['high_precision', 'medium_precision', 'low_precision']
            If None, clears all groups.
        """
        if groups is None:
            groups = ['high_precision', 'medium_precision', 'low_precision']

        with h5py.File(self.cache_file, 'a') as f:
            for group in groups:
                if group in f:
                    del f[group]
                    f.create_group(group)

        if len(groups) == 3:
            logger.info("Cache cleared (all precision tiers)")
        else:
            logger.info(f"Cache cleared: {', '.join(groups)}")
    
    def optimize_cache(self):
        """Repack HDF5 file to reclaim space"""
        logger.info("Optimizing cache file...")
        with h5py.File(self.cache_file, 'a') as f:
            f.flush()
        logger.info("Cache optimized")


class CacheBuilder:
    """Build position cache for specified time ranges"""
    
    def __init__(self, cache: PositionCache, orbit_calculator):
        """
        Initialize cache builder
        
        Parameters:
        -----------
        cache : PositionCache
            Position cache instance
        orbit_calculator : OrbitCalculator
            Orbit calculator instance
        """
        self.cache = cache
        self.calc = orbit_calculator
    
    def build_cache(self, 
                   asteroids: List[Dict],
                   reference_jd: float,
                   show_progress: bool = True,
                   high_precision_only: bool = False):
        """
        Build complete cache for all asteroids
        
        Parameters:
        -----------
        asteroids : list of dict
            Asteroid orbital elements
        reference_jd : float
            Reference Julian Date (typically today)
        show_progress : bool
            Show progress bar
        high_precision_only : bool
            If True, only build ±1 year high-precision cache (faster for testing)
        """
        from tqdm import tqdm
        
        self.cache.set_reference_date(reference_jd)
        
        # Extract designations
        designations = [ast['designation'] for ast in asteroids]
        
        # Build date ranges
        date_ranges = self._generate_date_ranges(reference_jd, high_precision_only)
        
        total_dates = sum(len(dates) for dates in date_ranges.values())
        
        logger.info(f"Building cache for {len(asteroids)} asteroids")
        logger.info(f"Total dates to compute: {total_dates}")
        if high_precision_only:
            logger.info("High-precision only mode (±1 year, daily)")
        
        # Progress bar
        pbar = tqdm(total=total_dates, disable=not show_progress, 
                   desc="Building cache")
        
        # Compute and store positions
        for precision, jd_list in date_ranges.items():
            for jd in jd_list:
                positions = self.calc.calculate_batch(asteroids, jd)
                self.cache.store_positions(jd, positions, designations)
                pbar.update(1)
        
        pbar.close()
        logger.info("Cache build complete")
    
    def _generate_date_ranges(self, reference_jd: float, high_precision_only: bool = False) -> Dict[str, List[float]]:
        """Generate Julian Dates for each precision tier"""
        ranges = {}
        
        # High precision: ±1 year, daily
        high_start = reference_jd - self.cache.HIGH_PRECISION_DAYS
        high_end = reference_jd + self.cache.HIGH_PRECISION_DAYS
        ranges['high'] = np.arange(high_start, high_end + 1, 
                                   self.cache.HIGH_INTERVAL).tolist()
        
        # If high_precision_only, return just the high precision range
        if high_precision_only:
            ranges['medium'] = []
            ranges['low'] = []
            return ranges
        
        # Ensure LOW_PRECISION ranges are set (should be set by set_reference_date)
        if self.cache.LOW_PRECISION_DAYS_BACKWARD is None or self.cache.LOW_PRECISION_DAYS_FORWARD is None:
            raise RuntimeError("Cache date ranges not initialized. Call set_reference_date() first.")
        
        # Medium precision: ±5 years (excluding high precision), weekly
        med_start = reference_jd - self.cache.MEDIUM_PRECISION_DAYS
        med_end = reference_jd + self.cache.MEDIUM_PRECISION_DAYS
        med_dates = np.arange(med_start, med_end + 1, 
                             self.cache.MEDIUM_INTERVAL)
        # Exclude high precision range
        med_dates = med_dates[
            (med_dates < high_start) | (med_dates > high_end)
        ]
        ranges['medium'] = med_dates.tolist()
        
        # Low precision: Full safe ephemeris range (excluding medium), monthly
        # Range is calculated dynamically based on ephemeris coverage
        low_start = reference_jd - self.cache.LOW_PRECISION_DAYS_BACKWARD
        low_end = reference_jd + self.cache.LOW_PRECISION_DAYS_FORWARD
        low_dates = np.arange(low_start, low_end + 1, 
                             self.cache.LOW_INTERVAL)
        # Exclude medium precision range
        low_dates = low_dates[
            (low_dates < med_start) | (low_dates > med_end)
        ]
        ranges['low'] = low_dates.tolist()
        
        return ranges
