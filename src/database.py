"""
Database Manager - Handle asteroid orbital elements storage and retrieval
Supports both PostgreSQL (full installation) and SQLite (lightweight/RPi)
"""

import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone
import numpy as np
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, Index, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
import pandas as pd
import requests

logger = logging.getLogger(__name__)

Base = declarative_base()

# CLN constants for discovery CLN calculation
# CLN 0 = Full Moon of 1980-01-02 (JD 2444240.0076)
CLN_EPOCH_JD = 2444240.0076
SYNODIC_MONTH = 29.530588853


def mjd_to_cln(mjd):
    """Convert MJD to Catalina Lunation Number (integer).
    
    Uses average synodic month - fast and consistent for filtering.
    """
    if mjd is None:
        return None
    jd = mjd + 2400000.5
    days_from_epoch = jd - CLN_EPOCH_JD
    lunations = days_from_epoch / SYNODIC_MONTH
    return int(np.floor(lunations))


class Asteroid(Base):
    """Asteroid orbital elements table"""
    __tablename__ = 'asteroids'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    designation = Column(String(20), unique=True, index=True, nullable=False)
    
    # Orbital elements (Keplerian)
    a = Column(Float, nullable=False)  # Semi-major axis (AU)
    e = Column(Float, nullable=False)  # Eccentricity
    i = Column(Float, nullable=False)  # Inclination (degrees)
    node = Column(Float, nullable=False)  # Longitude of ascending node (degrees)
    arg_peri = Column(Float, nullable=False)  # Argument of perihelion (degrees)
    M = Column(Float, nullable=False)  # Mean anomaly at epoch (degrees)
    epoch_jd = Column(Float, nullable=False)  # Epoch (Julian Date)
    mean_motion = Column(Float, nullable=True)  # Mean daily motion (degrees/day)
    
    # Physical parameters
    H = Column(Float, nullable=True)  # Absolute magnitude (can be NULL for new discoveries)
    G = Column(Float, default=0.15)  # Slope parameter
    earth_moid = Column(Float, nullable=True)  # Earth MOID (AU) - Minimum Orbit Intersection Distance
    
    # Classification
    orbit_class = Column(String(20), index=True)  # NEO, MBA, etc.
    neo_flag = Column(Boolean, default=False, index=True)  # Is NEO?
    pha_flag = Column(Boolean, default=False, index=True)  # Potentially Hazardous?
    
    # Orbit quality parameters
    uncertainty = Column(String(2), nullable=True)  # Uncertainty parameter U
    rms_residual = Column(Float, nullable=True)  # RMS residual (arcsec)
    
    # Observation metadata
    num_obs = Column(Integer)  # Number of observations
    num_oppositions = Column(Integer, nullable=True)  # Number of oppositions
    arc_span = Column(String(12), nullable=True)  # Arc span (original format, e.g., "15 days" or "12.5")
    arc_years = Column(Float)  # Observation arc in years (numeric)
    last_obs = Column(String(10), nullable=True)  # Last observation date (YYYYMMDD string)
    
    # Computation metadata
    reference = Column(String(12), nullable=True)  # MPC reference
    computer = Column(String(12), nullable=True)  # Computer/source name
    perturbers_coarse = Column(String(4), nullable=True)  # Coarse perturber indicator
    perturbers_precise = Column(String(4), nullable=True)  # Precise perturber indicator
    hex_flags = Column(String(6), nullable=True)  # 4-digit hex flags
    
    # Human-readable designation
    readable_designation = Column(String(30), nullable=True)  # e.g., "2024 AA" or "Eros"
    
    # Discovery tracklet data (from CNEOS)
    discovery_mjd = Column(Float, nullable=True)  # Average MJD of discovery tracklet
    discovery_cln = Column(Integer, nullable=True)  # Catalina Lunation Number at discovery
    discovery_ra = Column(Float, nullable=True)  # RA at discovery (degrees)
    discovery_dec = Column(Float, nullable=True)  # Dec at discovery (degrees)
    discovery_vmag = Column(Float, nullable=True)  # Median V magnitude at discovery
    discovery_site = Column(String(10), nullable=True)  # MPC observatory code
    
    # Update tracking
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Catalog staleness tracking (for objects that disappear from MPC catalog)
    is_stale = Column(Boolean, default=False, index=True)  # Object missing from current catalog
    stale_detected_at = Column(DateTime, nullable=True)  # When staleness was first detected
    last_seen_in_catalog = Column(DateTime, nullable=True)  # Last time object was in MPC download

    # Indexes for common queries
    __table_args__ = (
        Index('idx_magnitude', 'H'),
        Index('idx_semimajor', 'a'),
        Index('idx_eccentricity', 'e'),
        Index('idx_class_mag', 'orbit_class', 'H'),
        Index('idx_earth_moid', 'earth_moid'),
        Index('idx_discovery_mjd', 'discovery_mjd'),
        Index('idx_discovery_cln', 'discovery_cln'),
    )


class Catalog(Base):
    """Registry of alternate catalogs for comparison"""
    __tablename__ = 'catalogs'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50), unique=True, nullable=False, index=True)  # User-friendly name
    source_file = Column(String(255), nullable=True)  # Original filename
    description = Column(Text, nullable=True)  # Optional description

    # Statistics
    object_count = Column(Integer, default=0)

    # Data availability flags
    has_moid = Column(Boolean, default=False)  # Has Earth MOID data
    has_discovery = Column(Boolean, default=False)  # Has discovery circumstance data

    # Cache information
    cache_file = Column(String(255), nullable=True)  # Path to cache file if built

    # Timestamps
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class AlternateAsteroid(Base):
    """Asteroid data from alternate catalogs (for comparison/blinking)"""
    __tablename__ = 'alternate_asteroids'

    id = Column(Integer, primary_key=True, autoincrement=True)
    catalog_id = Column(Integer, ForeignKey('catalogs.id', ondelete='CASCADE'), nullable=False, index=True)
    designation = Column(String(20), index=True, nullable=False)

    # Orbital elements (Keplerian) - same as primary Asteroid table
    a = Column(Float, nullable=False)  # Semi-major axis (AU)
    e = Column(Float, nullable=False)  # Eccentricity
    i = Column(Float, nullable=False)  # Inclination (degrees)
    node = Column(Float, nullable=False)  # Longitude of ascending node (degrees)
    arg_peri = Column(Float, nullable=False)  # Argument of perihelion (degrees)
    M = Column(Float, nullable=False)  # Mean anomaly at epoch (degrees)
    epoch_jd = Column(Float, nullable=False)  # Epoch (Julian Date)
    mean_motion = Column(Float, nullable=True)  # Mean daily motion (degrees/day)

    # Physical parameters
    H = Column(Float, nullable=True)  # Absolute magnitude
    G = Column(Float, default=0.15)  # Slope parameter
    earth_moid = Column(Float, nullable=True)  # Earth MOID (AU) - optional

    # Classification
    orbit_class = Column(String(20), index=True)
    neo_flag = Column(Boolean, default=False, index=True)
    pha_flag = Column(Boolean, default=False, index=True)

    # Orbit quality parameters
    uncertainty = Column(String(2), nullable=True)
    rms_residual = Column(Float, nullable=True)

    # Observation metadata
    num_obs = Column(Integer)
    num_oppositions = Column(Integer, nullable=True)
    arc_span = Column(String(12), nullable=True)
    arc_years = Column(Float)
    last_obs = Column(String(10), nullable=True)

    # Computation metadata
    reference = Column(String(12), nullable=True)
    computer = Column(String(12), nullable=True)
    perturbers_coarse = Column(String(4), nullable=True)
    perturbers_precise = Column(String(4), nullable=True)
    hex_flags = Column(String(6), nullable=True)

    # Human-readable designation
    readable_designation = Column(String(30), nullable=True)

    # Discovery tracklet data (optional - may not exist for alternate catalogs)
    discovery_mjd = Column(Float, nullable=True)
    discovery_cln = Column(Integer, nullable=True)
    discovery_ra = Column(Float, nullable=True)
    discovery_dec = Column(Float, nullable=True)
    discovery_vmag = Column(Float, nullable=True)
    discovery_site = Column(String(10), nullable=True)

    # Indexes
    __table_args__ = (
        Index('idx_alt_catalog_designation', 'catalog_id', 'designation'),
        Index('idx_alt_magnitude', 'H'),
        Index('idx_alt_neo', 'neo_flag'),
    )


def unpack_provisional_designation(packed: str) -> str:
    """
    Unpack a provisional designation from MPC packed format
    
    Examples:
    - 'K24A00A' -> '2024 AA'
    - 'J99X12B' -> '1999 XB12'
    """
    if not packed or len(packed) < 7:
        return packed
    
    try:
        # First character encodes century
        century_map = {'I': '18', 'J': '19', 'K': '20'}
        century = century_map.get(packed[0], '20')
        
        # Next two characters are year
        year = century + packed[1:3]
        
        # Next character is half-month
        half_month = packed[3]
        
        # Next two characters are cycle and cycle_number
        cycle = packed[4:6]
        
        # Last character is order in half-month
        order = packed[6] if len(packed) > 6 else ''
        
        # Convert cycle number (A=1, B=2, etc., but some have numbers)
        if cycle[1].isdigit():
            cycle_str = cycle
        else:
            cycle_num = ord(cycle[1]) - ord('A') + 1 if cycle[1].isalpha() else 0
            cycle_str = cycle[0] + str(cycle_num) if cycle_num > 0 else cycle[0]
        
        return f"{year} {half_month}{cycle_str}{order}".strip()
    except:
        return packed


def normalize_designation(des: str) -> set:
    """
    Generate all possible variations of a designation for matching
    
    Returns a set of possible designation strings to try
    """
    if not des:
        return set()
    
    variations = set()
    des_clean = des.strip()
    
    # Add original
    variations.add(des_clean)
    
    # Remove parentheses
    no_parens = des_clean.replace('(', '').replace(')', '').strip()
    if no_parens != des_clean:
        variations.add(no_parens)
    
    # Add with parentheses
    if not des_clean.startswith('('):
        variations.add(f"({no_parens})")
    
    # Remove leading zeros from numbered asteroids
    if no_parens.isdigit():
        variations.add(str(int(no_parens)))
    
    # Try unpacking if it looks packed
    if len(des_clean) == 7 and des_clean[0] in 'IJK':
        unpacked = unpack_provisional_designation(des_clean)
        if unpacked != des_clean:
            variations.add(unpacked)
    
    return variations


def fetch_moid_batch(asteroids: List[Dict], show_progress: bool = True, output_dir: str = None) -> None:
    """
    Fetch Earth MOID for a batch of asteroids using JPL SBDB Query API
    
    Parameters:
    -----------
    asteroids : list of dict
        List of asteroid dictionaries to update with MOID
    show_progress : bool
        Show progress messages
    output_dir : str, optional
        Directory to save diagnostic files (e.g., designation lists)
    """
    # Import the designation utils
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from designation_utils import unpack_designation, normalize_designation
    
    if show_progress:
        logger.info(f"Fetching Earth MOID for {len(asteroids)} asteroids from JPL Query API...")
    
    # Save MPC designation list for comparison
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        mpc_list_path = os.path.join(output_dir, 'mpc_designations.txt')
        with open(mpc_list_path, 'w') as f:
            f.write("MPC NEA.txt Designations (Packed Format)\n")
            f.write("=" * 70 + "\n")
            f.write(f"{'Packed':<15} {'Unpacked':<20} {'H Magnitude':<12}\n")
            f.write("-" * 70 + "\n")
            for ast in sorted(asteroids, key=lambda x: x.get('designation', '')):
                des_packed = ast.get('designation', 'N/A')
                try:
                    des_unpacked = unpack_designation(des_packed)
                except:
                    des_unpacked = des_packed
                h = ast.get('H')
                h_str = f"{h:.2f}" if h is not None else "N/A"
                f.write(f"{des_packed:<15} {des_unpacked:<20} {h_str:<12}\n")
        logger.info(f"Saved MPC designation list to {mpc_list_path}")
    
    try:
        # Use JPL SBDB Query API to get all NEOs with MOID in one query
        base_url = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"
        
        # Query parameters for all NEOs with MOID
        # Request pdes (primary designation) - JPL returns unpacked format
        # Also request 'name' which includes alternate designations for numbered asteroids
        # Include MCA (Mars-crossing asteroids) to catch objects with different
        # orbital classifications between MPC and JPL
        params = {
            'fields': 'spkid,pdes,name,moid',
            'sb-class': 'IEO,ATE,APO,AMO,MCA'  # All NEO classes + Mars-crossers
        }
        
        if show_progress:
            logger.info("Querying JPL SBDB for all NEO MOID values...")
        
        # Try to make the request with SSL verification
        # Some systems (like Raspberry Pi) may have incomplete CA certificates
        response = None
        ssl_error = False
        
        # First try: normal SSL verification
        try:
            response = requests.get(base_url, params=params, timeout=120)
            response.raise_for_status()
        except requests.exceptions.SSLError as e:
            ssl_error = True
            logger.warning(f"SSL certificate verification failed: {e}")
            
            # Second try: use certifi if available
            try:
                import certifi
                logger.info("Retrying with certifi certificate bundle...")
                response = requests.get(base_url, params=params, timeout=120, 
                                       verify=certifi.where())
                response.raise_for_status()
                ssl_error = False
            except ImportError:
                logger.info("certifi not installed, cannot use alternative certificates")
            except requests.exceptions.SSLError:
                logger.warning("Still failing with certifi certificates")
            except Exception:
                pass
            
            # Third try: disable SSL verification (with warning)
            if ssl_error:
                logger.warning("Attempting request without SSL verification...")
                logger.warning("*** This is less secure but may be necessary on some systems ***")
                try:
                    import urllib3
                    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                except:
                    pass
                response = requests.get(base_url, params=params, timeout=120, verify=False)
                response.raise_for_status()
                logger.info("Request succeeded without SSL verification")
        
        data = response.json()
        
        # Save raw SBDB response to data directory for future use
        if output_dir:
            sbdb_cache_path = os.path.join(output_dir, 'sbdb_moid_cache.json')
            print(f"  Attempting to save SBDB cache to: {sbdb_cache_path}")
            try:
                import json
                # Ensure directory exists
                os.makedirs(output_dir, exist_ok=True)
                with open(sbdb_cache_path, 'w') as f:
                    json.dump(data, f)
                print(f"  ✓ Saved SBDB MOID data ({len(data.get('data', []))} records)")
                logger.info(f"Saved SBDB MOID data to {sbdb_cache_path}")
            except Exception as e:
                print(f"  ✗ Could not save SBDB cache: {e}")
                logger.warning(f"Could not save SBDB cache: {e}")
        
        if show_progress:
            logger.info(f"Received data from JPL API")
        
        # Parse response
        if 'data' not in data or 'fields' not in data:
            logger.error("Unexpected response format from JPL API")
            return
        
        fields = data['fields']
        rows = data['data']
        
        # Find column indices
        try:
            spkid_idx = fields.index('spkid')
            pdes_idx = fields.index('pdes')
            moid_idx = fields.index('moid')
        except ValueError as e:
            logger.error(f"Required field not found in response: {e}")
            return
        
        if show_progress:
            logger.info(f"Received {len(rows)} objects from JPL")
        
        # Save JPL designation list for comparison
        if output_dir:
            jpl_list_path = os.path.join(output_dir, 'jpl_designations.txt')
            with open(jpl_list_path, 'w') as f:
                f.write("JPL SBDB Designations (Unpacked Format)\n")
                f.write("=" * 90 + "\n")
                f.write(f"{'SPK-ID':<12} {'PDES (Unpacked)':<20} {'Packed':<15} {'Earth MOID (AU)':<15}\n")
                f.write("-" * 90 + "\n")
                for row in sorted(rows, key=lambda x: str(x[pdes_idx]) if x[pdes_idx] else ''):
                    spkid = row[spkid_idx]
                    pdes = row[pdes_idx]
                    moid = row[moid_idx]
                    # Try to pack the designation
                    try:
                        from designation_utils import pack_designation
                        packed = pack_designation(str(pdes)) if pdes else 'N/A'
                    except:
                        packed = 'N/A'
                    f.write(f"{str(spkid):<12} {str(pdes):<20} {packed:<15} {str(moid):<15}\n")
            logger.info(f"Saved JPL designation list to {jpl_list_path}")
        
        # Build MOID lookup dictionary
        # Key: unpacked designation (what JPL uses)
        # Value: MOID
        moid_lookup = {}
        
        # Try to find name field index for alternate designations
        try:
            name_idx = fields.index('name')
        except ValueError:
            name_idx = None
        
        for row in rows:
            pdes = row[pdes_idx]
            moid_value = row[moid_idx]
            
            if moid_value is not None:
                try:
                    moid_float = float(moid_value)
                except (ValueError, TypeError):
                    continue
                    
                # Store with original JPL designation (unpacked)
                if pdes:
                    moid_lookup[str(pdes).strip()] = moid_float
                
                # Also extract alternate designations from name field
                # Name format: "433 Eros (1898 DQ)" or "2024 AA" 
                if name_idx is not None and row[name_idx]:
                    name = str(row[name_idx])
                    # Extract provisional designation from parentheses if present
                    import re
                    prov_match = re.search(r'\((\d{4}\s+[A-Z]{2}\d*)\)', name)
                    if prov_match:
                        prov_des = prov_match.group(1)
                        moid_lookup[prov_des] = moid_float
                    # Also try the name itself if it looks like a designation
                    name_parts = name.split('(')[0].strip()
                    if re.match(r'\d{4}\s+[A-Z]{2}', name_parts):
                        moid_lookup[name_parts] = moid_float
        
        if show_progress:
            logger.info(f"Built MOID lookup with {len(moid_lookup)} entries (including alternate designations)")
        
        # Match asteroids with MOID data
        # Strategy: Unpack MPC designation and match against JPL unpacked
        fetched = 0
        not_found = 0
        matches = []
        mismatches = []
        
        for ast in asteroids:
            designation_packed = ast.get('designation', '')
            if not designation_packed:
                ast['earth_moid'] = None
                not_found += 1
                mismatches.append((designation_packed, 'N/A', 'NO', 'Empty designation'))
                continue
            
            # Unpack the MPC designation
            try:
                designation_unpacked = unpack_designation(designation_packed)
            except Exception as e:
                # If unpacking fails, try as-is
                designation_unpacked = designation_packed
                logger.debug(f"Failed to unpack '{designation_packed}': {e}")
            
            # Try to match against JPL
            moid = moid_lookup.get(designation_unpacked)
            
            if moid is not None:
                ast['earth_moid'] = moid
                fetched += 1
                matches.append((designation_packed, designation_unpacked, 'YES', f'{moid:.6f}'))
            else:
                ast['earth_moid'] = None
                not_found += 1
                mismatches.append((designation_packed, designation_unpacked, 'NO', 'Not in JPL'))
        
        # Save matching diagnostic
        if output_dir:
            match_log_path = os.path.join(output_dir, 'matching_log.txt')
            with open(match_log_path, 'w') as f:
                f.write("Designation Matching Log\n")
                f.write("=" * 80 + "\n\n")
                
                # Write summary
                f.write(f"Total asteroids: {len(asteroids)}\n")
                f.write(f"Matched: {fetched} ({100.0*fetched/len(asteroids):.1f}%)\n")
                f.write(f"Not matched: {not_found} ({100.0*not_found/len(asteroids):.1f}%)\n\n")
                
                # Write sample of successful matches
                f.write("SAMPLE SUCCESSFUL MATCHES (first 50):\n")
                f.write("-" * 80 + "\n")
                f.write(f"{'MPC (Packed)':<15} {'Unpacked':<20} {'MOID':<12} {'Match'}\n")
                f.write("-" * 80 + "\n")
                for item in matches[:50]:
                    f.write(f"{item[0]:<15} {item[1]:<20} {item[3]:<12} {item[2]}\n")
                f.write("\n")
                
                # Write sample of failures
                f.write("SAMPLE FAILED MATCHES (first 50):\n")
                f.write("-" * 80 + "\n")
                f.write(f"{'MPC (Packed)':<15} {'Unpacked':<20} {'Reason'}\n")
                f.write("-" * 80 + "\n")
                for item in mismatches[:50]:
                    f.write(f"{item[0]:<15} {item[1]:<20} {item[3]}\n")
                f.write("\n")
            
            logger.info(f"Saved matching log to {match_log_path}")
        
        if show_progress:
            logger.info(f"MOID match complete: {fetched} matched, {not_found} not found")
            logger.info(f"Coverage: {100.0 * fetched / len(asteroids):.1f}%")
            if output_dir:
                logger.info(f"Diagnostic files saved to {output_dir}")
    
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching MOID data from JPL: {e}")
        logger.error("MOID data will not be available")
        # Set all to None
        for ast in asteroids:
            ast['earth_moid'] = None
    except Exception as e:
        logger.error(f"Unexpected error processing MOID data: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        for ast in asteroids:
            ast['earth_moid'] = None


def load_discovery_tracklets(asteroids: List[Dict], csv_path: str, show_progress: bool = True) -> int:
    """
    Load discovery tracklet data from CSV and merge into asteroid list.
    
    Parameters:
    -----------
    asteroids : List[Dict]
        List of asteroid dictionaries to update
    csv_path : str
        Path to NEA_discovery_tracklets.csv
    show_progress : bool
        Whether to show progress messages
        
    Returns:
    --------
    int : Number of asteroids matched
    """
    import os
    
    if not os.path.exists(csv_path):
        logger.warning(f"Discovery tracklet file not found: {csv_path}")
        return 0
    
    if show_progress:
        logger.info(f"Loading discovery tracklet data from {csv_path}...")
    
    try:
        # Load CSV
        df = pd.read_csv(csv_path)
        
        if show_progress:
            logger.info(f"Loaded {len(df)} discovery tracklet records")
        
        # Build lookup dictionaries by packed designation
        tracklet_by_packed = {}
        for _, row in df.iterrows():
            packed = str(row['packed_designation']).strip()
            disc_mjd = row['avg_mjd_discovery_tracklet'] if pd.notna(row['avg_mjd_discovery_tracklet']) else None
            # Precompute CLN from MJD
            disc_cln = mjd_to_cln(disc_mjd)
            tracklet_by_packed[packed] = {
                'discovery_mjd': disc_mjd,
                'discovery_cln': disc_cln,
                'discovery_ra': row['avg_ra_deg'] if pd.notna(row['avg_ra_deg']) else None,
                'discovery_dec': row['avg_dec_deg'] if pd.notna(row['avg_dec_deg']) else None,
                'discovery_vmag': row['median_v_magnitude'] if pd.notna(row['median_v_magnitude']) else None,
                'discovery_site': str(row['discovery_site_code']).strip() if pd.notna(row['discovery_site_code']) else None,
            }
        
        # Match to asteroids
        matched = 0
        for ast in asteroids:
            des = ast['designation']
            
            if des in tracklet_by_packed:
                tracklet = tracklet_by_packed[des]
                ast.update(tracklet)
                matched += 1
            else:
                # No match - set fields to None
                ast['discovery_mjd'] = None
                ast['discovery_cln'] = None
                ast['discovery_ra'] = None
                ast['discovery_dec'] = None
                ast['discovery_vmag'] = None
                ast['discovery_site'] = None
        
        if show_progress:
            logger.info(f"Matched discovery data for {matched} of {len(asteroids)} asteroids")
            logger.info(f"Coverage: {100.0 * matched / len(asteroids):.1f}%")
        
        return matched
        
    except Exception as e:
        logger.error(f"Error loading discovery tracklets: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        
        # Set all to None on error
        for ast in asteroids:
            ast['discovery_mjd'] = None
            ast['discovery_cln'] = None
            ast['discovery_ra'] = None
            ast['discovery_dec'] = None
            ast['discovery_vmag'] = None
            ast['discovery_site'] = None
        
        return 0


class DatabaseManager:
    """Manage asteroid database operations"""
    
    def __init__(self, db_url: str = None, use_sqlite: bool = False):
        """
        Initialize database connection
        
        Parameters:
        -----------
        db_url : str
            Database connection URL (e.g., 'postgresql://user:pass@localhost/asteroids')
            If None, uses SQLite by default
        use_sqlite : bool
            Force use of SQLite (for RPi or lightweight deployment)
        """
        if use_sqlite or db_url is None:
            # SQLite fallback
            db_url = 'sqlite:///data/asteroids.db'
            logger.info("Using SQLite database")
        else:
            logger.info(f"Using PostgreSQL database: {db_url}")
        
        # Create engine with connection pooling
        self.engine = create_engine(
            db_url,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            echo=False  # Set to True for SQL debugging
        )
        
        # Create tables
        Base.metadata.create_all(self.engine)
        
        # Session factory
        self.Session = sessionmaker(bind=self.engine)
        
        # Check for and perform any needed migrations
        self._check_migrations()
    
    def _check_migrations(self):
        """Check for and perform database migrations"""
        try:
            from sqlalchemy import inspect, text
            inspector = inspect(self.engine)
            columns = [c['name'] for c in inspector.get_columns('asteroids')]

            # Migration: Add discovery_cln column if missing
            if 'discovery_cln' not in columns:
                logger.info("Migrating database: adding discovery_cln column...")
                with self.engine.connect() as conn:
                    conn.execute(text("ALTER TABLE asteroids ADD COLUMN discovery_cln INTEGER"))
                    conn.commit()

                # Populate discovery_cln from discovery_mjd
                self._populate_discovery_cln()
                logger.info("Migration complete: discovery_cln column added")

            # Migration: Add stale tracking columns if missing
            if 'is_stale' not in columns:
                logger.info("Migrating database: adding stale tracking columns...")
                with self.engine.connect() as conn:
                    conn.execute(text("ALTER TABLE asteroids ADD COLUMN is_stale BOOLEAN DEFAULT 0"))
                    conn.execute(text("ALTER TABLE asteroids ADD COLUMN stale_detected_at DATETIME"))
                    conn.execute(text("ALTER TABLE asteroids ADD COLUMN last_seen_in_catalog DATETIME"))
                    conn.commit()
                logger.info("Migration complete: stale tracking columns added")

        except Exception as e:
            logger.warning(f"Migration check failed: {e}")
    
    def _populate_discovery_cln(self):
        """Populate discovery_cln for all records with discovery_mjd"""
        session = self.get_session()
        try:
            # Get all asteroids with discovery_mjd but no discovery_cln
            asteroids = session.query(Asteroid).filter(
                Asteroid.discovery_mjd.isnot(None),
                Asteroid.discovery_cln.is_(None)
            ).all()
            
            if asteroids:
                logger.info(f"Computing discovery_cln for {len(asteroids)} asteroids...")
                for ast in asteroids:
                    ast.discovery_cln = mjd_to_cln(ast.discovery_mjd)
                session.commit()
                logger.info(f"Updated {len(asteroids)} records with discovery_cln")
        except Exception as e:
            session.rollback()
            logger.error(f"Error populating discovery_cln: {e}")
        finally:
            session.close()
    
    def get_session(self) -> Session:
        """Get a new database session"""
        return self.Session()
    
    def insert_asteroids(self, asteroids: List[Dict], batch_size: int = 1000):
        """
        Insert or update asteroids in bulk
        
        Parameters:
        -----------
        asteroids : list of dict
            List of asteroid data dictionaries
        batch_size : int
            Number of records to insert per transaction
        """
        session = self.get_session()
        
        try:
            for i in range(0, len(asteroids), batch_size):
                batch = asteroids[i:i + batch_size]
                
                for ast_data in batch:
                    # Check if exists
                    existing = session.query(Asteroid).filter_by(
                        designation=ast_data['designation']
                    ).first()
                    
                    if existing:
                        # Update
                        for key, value in ast_data.items():
                            setattr(existing, key, value)
                    else:
                        # Insert
                        session.add(Asteroid(**ast_data))
                
                session.commit()
                logger.info(f"Inserted/updated batch {i//batch_size + 1} ({len(batch)} objects)")
            
            logger.info(f"Successfully loaded {len(asteroids)} asteroids")
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error inserting asteroids: {e}")
            raise
        finally:
            session.close()
    
    def get_asteroids(self,
                     orbit_class: Optional[str] = None,
                     neo_only: bool = False,
                     pha_only: bool = False,
                     mag_limit: Optional[float] = None,
                     h_min: Optional[float] = None,
                     h_max: Optional[float] = None,
                     a_min: Optional[float] = None,
                     a_max: Optional[float] = None,
                     moid_min: Optional[float] = None,
                     moid_max: Optional[float] = None,
                     limit: Optional[int] = None) -> List[Dict]:
        """
        Query asteroids with various filters
        
        Parameters:
        -----------
        orbit_class : str, optional
            Filter by orbit class (e.g., 'Apollo', 'Aten', 'Amor', 'MBA')
        neo_only : bool
            Return only NEOs
        pha_only : bool
            Return only PHAs
        mag_limit : float, optional
            Maximum absolute magnitude (brighter than) - DEPRECATED, use h_max
        h_min : float, optional
            Minimum absolute magnitude (faintest to include)
        h_max : float, optional
            Maximum absolute magnitude (brightest to include)
        a_min, a_max : float, optional
            Semi-major axis range (AU)
        moid_min : float, optional
            Minimum Earth MOID (AU) - only include asteroids with MOID >= this value
        moid_max : float, optional
            Maximum Earth MOID (AU) - only include asteroids with MOID <= this value
        limit : int, optional
            Maximum number of results
            
        Returns:
        --------
        list of dict : Asteroid data as dictionaries
        """
        session = self.get_session()
        
        try:
            query = session.query(Asteroid)
            
            # Apply filters
            if orbit_class:
                query = query.filter(Asteroid.orbit_class == orbit_class)
            if neo_only:
                query = query.filter(Asteroid.neo_flag == True)
            if pha_only:
                query = query.filter(Asteroid.pha_flag == True)
            if mag_limit is not None:
                query = query.filter(Asteroid.H <= mag_limit)
            if h_min is not None:
                query = query.filter(Asteroid.H >= h_min)
            if h_max is not None:
                query = query.filter(Asteroid.H <= h_max)
            if a_min is not None:
                query = query.filter(Asteroid.a >= a_min)
            if a_max is not None:
                query = query.filter(Asteroid.a <= a_max)
            if moid_min is not None or moid_max is not None:
                query = query.filter(Asteroid.earth_moid != None)  # Exclude NULL MOIDs
                if moid_min is not None:
                    query = query.filter(Asteroid.earth_moid >= moid_min)
                if moid_max is not None:
                    query = query.filter(Asteroid.earth_moid <= moid_max)
            
            # Apply limit
            if limit:
                query = query.limit(limit)
            
            # Execute and convert to dicts
            results = query.all()
            
            return [self._asteroid_to_dict(ast) for ast in results]
            
        finally:
            session.close()
    
    def get_asteroid_by_designation(self, designation: str) -> Optional[Dict]:
        """Get a single asteroid by designation"""
        session = self.get_session()
        
        try:
            ast = session.query(Asteroid).filter_by(designation=designation).first()
            return self._asteroid_to_dict(ast) if ast else None
        finally:
            session.close()
    
    def get_all_orbital_elements(self, neo_only: bool = False) -> pd.DataFrame:
        """
        Get all orbital elements as a pandas DataFrame for efficient computation
        
        Parameters:
        -----------
        neo_only : bool
            Return only NEOs
            
        Returns:
        --------
        DataFrame with columns: id, designation, a, e, i, node, arg_peri, M, epoch_jd, H, G
        """
        session = self.get_session()
        
        try:
            query = session.query(Asteroid)
            
            if neo_only:
                query = query.filter(Asteroid.neo_flag == True)
            
            # Use pandas to read directly
            df = pd.read_sql(query.statement, session.bind)
            
            return df
            
        finally:
            session.close()
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        session = self.get_session()
        
        try:
            total = session.query(Asteroid).count()
            neos = session.query(Asteroid).filter(Asteroid.neo_flag == True).count()
            phas = session.query(Asteroid).filter(Asteroid.pha_flag == True).count()
            
            # Count by class
            classes = session.query(
                Asteroid.orbit_class, 
                func.count(Asteroid.id)
            ).group_by(Asteroid.orbit_class).all()
            
            # Magnitude distribution (exclude NULL H values)
            mag_bins = session.query(
                func.floor(Asteroid.H),
                func.count(Asteroid.id)
            ).filter(Asteroid.H.isnot(None)).group_by(func.floor(Asteroid.H)).order_by(func.floor(Asteroid.H)).all()
            
            return {
                'total': total,
                'neos': neos,
                'phas': phas,
                'by_class': dict(classes),
                'mag_distribution': dict(mag_bins)
            }
            
        finally:
            session.close()
    
    def _asteroid_to_dict(self, ast: Asteroid) -> Dict:
        """Convert Asteroid ORM object to dictionary"""
        if ast is None:
            return None
        
        return {
            'id': ast.id,
            'designation': ast.designation,
            # Orbital elements
            'a': ast.a,
            'e': ast.e,
            'i': ast.i,
            'node': ast.node,
            'arg_peri': ast.arg_peri,
            'M': ast.M,
            'epoch_jd': ast.epoch_jd,
            'mean_motion': ast.mean_motion,
            # Physical parameters
            'H': ast.H,
            'G': ast.G,
            'earth_moid': ast.earth_moid,
            # Classification
            'orbit_class': ast.orbit_class,
            'neo_flag': ast.neo_flag,
            'pha_flag': ast.pha_flag,
            # Orbit quality
            'uncertainty': ast.uncertainty,
            'rms_residual': ast.rms_residual,
            # Observation metadata
            'num_obs': ast.num_obs,
            'num_oppositions': ast.num_oppositions,
            'arc_span': ast.arc_span,
            'arc_years': ast.arc_years,
            'last_obs': ast.last_obs,
            # Computation metadata
            'reference': ast.reference,
            'computer': ast.computer,
            'perturbers_coarse': ast.perturbers_coarse,
            'perturbers_precise': ast.perturbers_precise,
            'hex_flags': ast.hex_flags,
            # Human-readable designation
            'readable_designation': ast.readable_designation,
            # Discovery tracklet data
            'discovery_mjd': ast.discovery_mjd,
            'discovery_cln': ast.discovery_cln,
            'discovery_ra': ast.discovery_ra,
            'discovery_dec': ast.discovery_dec,
            'discovery_vmag': ast.discovery_vmag,
            'discovery_site': ast.discovery_site,
        }
    
    def clear_all(self):
        """Clear all asteroids (use with caution!)"""
        session = self.get_session()
        try:
            session.query(Asteroid).delete()
            session.commit()
            logger.info("Database cleared")
        except Exception as e:
            session.rollback()
            logger.error(f"Error clearing database: {e}")
            raise
        finally:
            session.close()

    # ==================== Alternate Catalog Methods ====================

    def list_catalogs(self) -> List[Dict]:
        """List all alternate catalogs"""
        session = self.get_session()
        try:
            catalogs = session.query(Catalog).order_by(Catalog.name).all()
            return [{
                'id': c.id,
                'name': c.name,
                'source_file': c.source_file,
                'description': c.description,
                'object_count': c.object_count,
                'has_moid': c.has_moid,
                'has_discovery': c.has_discovery,
                'cache_file': c.cache_file,
                'created_at': c.created_at,
                'updated_at': c.updated_at
            } for c in catalogs]
        finally:
            session.close()

    def get_catalog(self, name: str) -> Optional[Dict]:
        """Get a catalog by name"""
        session = self.get_session()
        try:
            c = session.query(Catalog).filter_by(name=name).first()
            if c is None:
                return None
            return {
                'id': c.id,
                'name': c.name,
                'source_file': c.source_file,
                'description': c.description,
                'object_count': c.object_count,
                'has_moid': c.has_moid,
                'has_discovery': c.has_discovery,
                'cache_file': c.cache_file,
                'created_at': c.created_at,
                'updated_at': c.updated_at
            }
        finally:
            session.close()

    def get_catalog_by_id(self, catalog_id: int) -> Optional[Dict]:
        """Get a catalog by ID"""
        session = self.get_session()
        try:
            c = session.query(Catalog).filter_by(id=catalog_id).first()
            if c is None:
                return None
            return {
                'id': c.id,
                'name': c.name,
                'source_file': c.source_file,
                'description': c.description,
                'object_count': c.object_count,
                'has_moid': c.has_moid,
                'has_discovery': c.has_discovery,
                'cache_file': c.cache_file,
                'created_at': c.created_at,
                'updated_at': c.updated_at
            }
        finally:
            session.close()

    def create_catalog(self, name: str, source_file: str = None,
                      description: str = None) -> int:
        """Create a new alternate catalog entry

        Returns:
        --------
        int : The catalog ID
        """
        session = self.get_session()
        try:
            catalog = Catalog(
                name=name,
                source_file=source_file,
                description=description
            )
            session.add(catalog)
            session.commit()
            catalog_id = catalog.id
            logger.info(f"Created alternate catalog '{name}' with ID {catalog_id}")
            return catalog_id
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating catalog: {e}")
            raise
        finally:
            session.close()

    def update_catalog(self, catalog_id: int, **kwargs):
        """Update catalog metadata"""
        session = self.get_session()
        try:
            catalog = session.query(Catalog).filter_by(id=catalog_id).first()
            if catalog is None:
                raise ValueError(f"Catalog ID {catalog_id} not found")

            for key, value in kwargs.items():
                if hasattr(catalog, key):
                    setattr(catalog, key, value)

            session.commit()
            logger.info(f"Updated catalog ID {catalog_id}")
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating catalog: {e}")
            raise
        finally:
            session.close()

    def delete_catalog(self, name: str) -> bool:
        """Delete an alternate catalog and all its asteroids

        Returns:
        --------
        bool : True if deleted, False if not found
        """
        session = self.get_session()
        try:
            catalog = session.query(Catalog).filter_by(name=name).first()
            if catalog is None:
                return False

            # Delete all asteroids in this catalog (cascade should handle this,
            # but be explicit for clarity)
            deleted_count = session.query(AlternateAsteroid).filter_by(
                catalog_id=catalog.id
            ).delete()

            # Delete the catalog entry
            session.delete(catalog)
            session.commit()

            logger.info(f"Deleted catalog '{name}' with {deleted_count} asteroids")
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting catalog: {e}")
            raise
        finally:
            session.close()

    def insert_alternate_asteroids(self, catalog_id: int, asteroids: List[Dict],
                                   batch_size: int = 1000):
        """Insert asteroids into an alternate catalog

        Parameters:
        -----------
        catalog_id : int
            The catalog ID to insert into
        asteroids : list of dict
            List of asteroid data dictionaries
        batch_size : int
            Number of records to insert per transaction
        """
        session = self.get_session()

        try:
            for i in range(0, len(asteroids), batch_size):
                batch = asteroids[i:i + batch_size]

                for ast_data in batch:
                    # Add catalog_id to the data
                    ast_data['catalog_id'] = catalog_id
                    session.add(AlternateAsteroid(**ast_data))

                session.commit()
                logger.info(f"Inserted batch {i//batch_size + 1} ({len(batch)} objects)")

            # Update object count in catalog
            catalog = session.query(Catalog).filter_by(id=catalog_id).first()
            if catalog:
                catalog.object_count = session.query(AlternateAsteroid).filter_by(
                    catalog_id=catalog_id
                ).count()
                session.commit()

            logger.info(f"Successfully loaded {len(asteroids)} asteroids into catalog ID {catalog_id}")

        except Exception as e:
            session.rollback()
            logger.error(f"Error inserting alternate asteroids: {e}")
            raise
        finally:
            session.close()

    def get_alternate_asteroids(self, catalog_id: int,
                                neo_only: bool = False,
                                limit: Optional[int] = None) -> List[Dict]:
        """Get asteroids from an alternate catalog

        Parameters:
        -----------
        catalog_id : int
            The catalog ID to query
        neo_only : bool
            Return only NEOs
        limit : int, optional
            Maximum number of results

        Returns:
        --------
        list of dict : Asteroid data as dictionaries
        """
        session = self.get_session()

        try:
            query = session.query(AlternateAsteroid).filter_by(catalog_id=catalog_id)

            if neo_only:
                query = query.filter(AlternateAsteroid.neo_flag == True)

            if limit:
                query = query.limit(limit)

            results = query.all()
            return [self._alternate_asteroid_to_dict(ast) for ast in results]

        finally:
            session.close()

    def _alternate_asteroid_to_dict(self, ast: AlternateAsteroid) -> Dict:
        """Convert AlternateAsteroid ORM object to dictionary"""
        if ast is None:
            return None

        return {
            'id': ast.id,
            'catalog_id': ast.catalog_id,
            'designation': ast.designation,
            # Orbital elements
            'a': ast.a,
            'e': ast.e,
            'i': ast.i,
            'node': ast.node,
            'arg_peri': ast.arg_peri,
            'M': ast.M,
            'epoch_jd': ast.epoch_jd,
            'mean_motion': ast.mean_motion,
            # Physical parameters
            'H': ast.H,
            'G': ast.G,
            'earth_moid': ast.earth_moid,
            # Classification
            'orbit_class': ast.orbit_class,
            'neo_flag': ast.neo_flag,
            'pha_flag': ast.pha_flag,
            # Orbit quality
            'uncertainty': ast.uncertainty,
            'rms_residual': ast.rms_residual,
            # Observation metadata
            'num_obs': ast.num_obs,
            'num_oppositions': ast.num_oppositions,
            'arc_span': ast.arc_span,
            'arc_years': ast.arc_years,
            'last_obs': ast.last_obs,
            # Computation metadata
            'reference': ast.reference,
            'computer': ast.computer,
            'perturbers_coarse': ast.perturbers_coarse,
            'perturbers_precise': ast.perturbers_precise,
            'hex_flags': ast.hex_flags,
            # Human-readable designation
            'readable_designation': ast.readable_designation,
            # Discovery tracklet data
            'discovery_mjd': ast.discovery_mjd,
            'discovery_cln': ast.discovery_cln,
            'discovery_ra': ast.discovery_ra,
            'discovery_dec': ast.discovery_dec,
            'discovery_vmag': ast.discovery_vmag,
            'discovery_site': ast.discovery_site,
        }


# Import after Asteroid is defined
from sqlalchemy import func
