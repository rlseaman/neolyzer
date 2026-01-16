"""
MPC Data Loader - Download and parse Minor Planet Center orbital elements
Supports NEA.txt and MPCORB.DAT formats
"""

import requests
import logging
from typing import List, Dict, Optional
from datetime import datetime
import re
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)


class MPCLoader:
    """Download and parse MPC orbital element files"""
    
    # MPC data sources
    NEA_URL = "https://www.minorplanetcenter.net/iau/MPCORB/NEA.txt"
    MPCORB_URL = "https://minorplanetcenter.net/iau/MPCORB/MPCORB.DAT"
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize MPC loader
        
        Parameters:
        -----------
        data_dir : str
            Directory to store downloaded files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def download_nea(self, force: bool = False) -> Path:
        """
        Download NEA.txt file
        
        Parameters:
        -----------
        force : bool
            Force re-download even if file exists
            
        Returns:
        --------
        Path : Path to downloaded file
        """
        filepath = self.data_dir / "NEA.txt"
        
        if filepath.exists() and not force:
            logger.info(f"NEA.txt already exists at {filepath}")
            return filepath
        
        logger.info(f"Downloading NEA.txt from {self.NEA_URL}")
        
        try:
            response = requests.get(self.NEA_URL, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, 
                         desc="Downloading NEA.txt") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            logger.info(f"Downloaded to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error downloading NEA.txt: {e}")
            raise
    
    def download_mpcorb(self, force: bool = False) -> Path:
        """
        Download MPCORB.DAT file (large!)
        
        Parameters:
        -----------
        force : bool
            Force re-download even if file exists
            
        Returns:
        --------
        Path : Path to downloaded file
        """
        filepath = self.data_dir / "MPCORB.DAT"
        
        if filepath.exists() and not force:
            logger.info(f"MPCORB.DAT already exists at {filepath}")
            return filepath
        
        logger.info(f"Downloading MPCORB.DAT from {self.MPCORB_URL}")
        logger.warning("This is a large file (~200 MB), download may take several minutes")
        
        try:
            response = requests.get(self.MPCORB_URL, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filepath, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True,
                         desc="Downloading MPCORB.DAT") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            logger.info(f"Downloaded to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error downloading MPCORB.DAT: {e}")
            raise
    
    def parse_mpc_format(self, filepath: Path, 
                        neo_only: bool = True,
                        max_records: Optional[int] = None) -> List[Dict]:
        """
        Parse MPC orbital elements file
        
        MPC format specification:
        https://www.minorplanetcenter.net/iau/info/MPOrbitFormat.html
        
        Parameters:
        -----------
        filepath : Path
            Path to MPC data file
        neo_only : bool
            Only parse NEOs (a < 4.2 AU)
        max_records : int, optional
            Limit number of records parsed (for testing)
            
        Returns:
        --------
        list of dict : Parsed asteroid data
        """
        asteroids = []
        
        logger.info(f"Parsing {filepath.name}")
        
        with open(filepath, 'r', encoding='latin-1') as f:
            # Skip header lines (varies by file)
            # NEA.txt has no header, MPCORB.DAT does
            header_found = False
            for i, line in enumerate(f):
                if line.startswith('------') or line.startswith('Des'):
                    header_found = True
                    break
                # If we don't find header in first 10 lines, assume no header
                if i >= 10:
                    break
            
            # If no header found, rewind to start
            if not header_found:
                f.seek(0)
                logger.debug("No header found, parsing from beginning")
            
            # Parse data lines
            line_count = 0
            for line in tqdm(f, desc="Parsing records"):
                if max_records and line_count >= max_records:
                    break
                
                if len(line.strip()) < 160:  # Minimum line length
                    continue
                
                try:
                    asteroid = self._parse_mpc_line(line)
                    
                    # Filter NEOs if requested
                    # NEO definition: perihelion q < 1.3 AU
                    # q = a * (1 - e)
                    if neo_only:
                        q = asteroid['a'] * (1 - asteroid['e'])
                        if q < 1.3:  # Perihelion inside 1.3 AU = NEO
                            asteroids.append(asteroid)
                    else:
                        asteroids.append(asteroid)
                    
                    line_count += 1
                    
                except Exception as e:
                    logger.debug(f"Error parsing line: {e}")
                    continue
        
        logger.info(f"Parsed {len(asteroids)} asteroids")
        
        # Classify orbits
        self._classify_orbits(asteroids)
        
        return asteroids
    
    def _parse_mpc_line(self, line: str) -> Dict:
        """
        Parse a single line from MPC format
        
        Format positions (1-indexed, converted to 0-indexed for Python):
        1-7:     Packed designation
        9-13:    Absolute magnitude H
        15-19:   Slope parameter G
        21-25:   Epoch (packed format)
        27-35:   Mean anomaly M (degrees)
        38-46:   Argument of perihelion ω (degrees)
        49-57:   Longitude of ascending node Ω (degrees)
        60-68:   Inclination i (degrees)
        71-79:   Eccentricity e
        81-91:   Mean daily motion n (degrees/day)
        93-103:  Semi-major axis a (AU)
        106:     Uncertainty parameter U
        108-116: Reference
        118-122: Number of observations
        124-126: Number of oppositions
        128-136: Arc span (days/years)
        138-141: r.m.s. residual (arcsec)
        143-145: Coarse indicator of perturbers
        147-149: Precise indicator of perturbers
        151-160: Computer name
        162-165: 4-digit hex flags
        167-194: Readable designation
        195-202: Date of last observation (YYYYMMDD)
        """
        designation = line[0:7].strip()
        
        # Magnitude and slope - H can be missing for newly discovered objects
        h_str = line[8:13].strip()
        H = float(h_str) if h_str else None
        G = float(line[14:19].strip()) if line[14:19].strip() else 0.15
        
        # Epoch (packed format - convert to JD)
        epoch_str = line[20:25].strip()
        epoch_jd = self._unpack_epoch(epoch_str)
        
        # Orbital elements
        M = float(line[26:35].strip())  # Mean anomaly
        arg_peri = float(line[37:46].strip())  # Argument of perihelion
        node = float(line[48:57].strip())  # Ascending node
        i = float(line[59:68].strip())  # Inclination
        e = float(line[70:79].strip())  # Eccentricity
        
        # Mean motion and semi-major axis
        n_str = line[80:91].strip()
        mean_motion = float(n_str) if n_str else None
        a = float(line[92:103].strip())  # Semi-major axis
        
        # Uncertainty parameter (single character)
        uncertainty = line[105:106].strip() if len(line) > 105 else ''
        
        # Reference
        reference = line[107:116].strip() if len(line) > 107 else ''
        
        # Observation info
        num_obs_str = line[117:122].strip() if len(line) > 117 else ''
        num_obs = int(num_obs_str) if num_obs_str else 0
        
        num_opp_str = line[123:126].strip() if len(line) > 123 else ''
        num_oppositions = int(num_opp_str) if num_opp_str else 0
        
        # Arc span (can be "days" or years format)
        arc_span = line[127:136].strip() if len(line) > 127 else ''
        # Also extract numeric arc_years for backwards compatibility
        arc_years_str = line[127:131].strip() if len(line) > 127 else ''
        try:
            arc_years = float(arc_years_str) if arc_years_str and not arc_years_str.endswith('days') else 0.0
        except ValueError:
            arc_years = 0.0
        
        # RMS residual
        rms_str = line[137:141].strip() if len(line) > 137 else ''
        try:
            rms_residual = float(rms_str) if rms_str else None
        except ValueError:
            rms_residual = None
        
        # Perturbers
        perturbers_coarse = line[142:145].strip() if len(line) > 142 else ''
        perturbers_precise = line[146:149].strip() if len(line) > 146 else ''
        
        # Computer name
        computer = line[150:160].strip() if len(line) > 150 else ''
        
        # Hex flags
        hex_flags = line[161:165].strip() if len(line) > 161 else ''
        
        # Readable designation (human-readable form)
        readable_designation = line[166:194].strip() if len(line) > 166 else ''
        
        # Last observation date (YYYYMMDD)
        last_obs = line[194:202].strip() if len(line) > 194 else ''
        
        return {
            'designation': designation,
            'H': H,
            'G': G,
            'epoch_jd': epoch_jd,
            'M': M,
            'arg_peri': arg_peri,
            'node': node,
            'i': i,
            'e': e,
            'a': a,
            'mean_motion': mean_motion,
            'uncertainty': uncertainty,
            'reference': reference,
            'num_obs': num_obs,
            'num_oppositions': num_oppositions,
            'arc_span': arc_span,
            'arc_years': arc_years,
            'rms_residual': rms_residual,
            'perturbers_coarse': perturbers_coarse,
            'perturbers_precise': perturbers_precise,
            'computer': computer,
            'hex_flags': hex_flags,
            'readable_designation': readable_designation,
            'last_obs': last_obs,
            'orbit_class': None,  # Will be determined
            'neo_flag': False,
            'pha_flag': False,
        }
    
    def _unpack_epoch(self, packed: str) -> float:
        """
        Convert MPC packed epoch format to Julian Date
        
        Format: KYYMD where:
        K = century code (J=2000, K=2010, etc.)
        YY = year within century
        M = month code (1-9=Jan-Sep, A=Oct, B=Nov, C=Dec)
        D = day code (01-31)
        """
        if not packed or len(packed) < 5:
            return 2451545.0  # Default to J2000.0
        
        # Century
        century_codes = {
            'I': 1800, 'J': 1900, 'K': 2000, 'L': 2100
        }
        century = century_codes.get(packed[0], 2000)
        
        # Year
        year = century + int(packed[1:3])
        
        # Month
        month_codes = '123456789ABC'
        month = month_codes.index(packed[3]) + 1 if packed[3] in month_codes else 1
        
        # Day
        day_codes = '0123456789ABCDEFGHIJKLMNOPQRSTUV'
        day = day_codes.index(packed[4]) if packed[4] in day_codes else 1
        
        # Convert to JD
        from skyfield.api import load
        ts = load.timescale()
        t = ts.utc(year, month, day)
        return t.tt
    
    def _classify_orbits(self, asteroids: List[Dict]):
        """
        Classify orbits into NEO categories
        
        Categories:
        - Atiras: a < 1.0 AU, Q < 0.983 AU
        - Atens: a < 1.0 AU, Q > 0.983 AU
        - Apollos: a > 1.0 AU, q < 1.017 AU
        - Amors: 1.017 AU < q < 1.3 AU
        - MBA: Main Belt (a > 2.0 AU)
        
        PHA: H < 22.0 and MOID < 0.05 AU (simplified)
        """
        for ast in asteroids:
            a = ast['a']
            e = ast['e']
            
            # Calculate perihelion (q) and aphelion (Q)
            q = a * (1 - e)
            Q = a * (1 + e)
            
            # Classify
            if a < 1.0 and Q < 0.983:
                ast['orbit_class'] = 'Atira'
                ast['neo_flag'] = True
            elif a < 1.0 and Q > 0.983:
                ast['orbit_class'] = 'Aten'
                ast['neo_flag'] = True
            elif a > 1.0 and q < 1.017:
                ast['orbit_class'] = 'Apollo'
                ast['neo_flag'] = True
            elif 1.017 < q < 1.3:
                ast['orbit_class'] = 'Amor'
                ast['neo_flag'] = True
            elif a > 2.0:
                ast['orbit_class'] = 'MBA'
                ast['neo_flag'] = False
            else:
                ast['orbit_class'] = 'Other'
                ast['neo_flag'] = False
            
            # PHA determination (simplified - needs MOID calculation for accuracy)
            # H must be known and < 22.0, and close approach possible
            if ast['neo_flag'] and ast['H'] is not None and ast['H'] < 22.0 and q < 0.05:
                ast['pha_flag'] = True


def load_asteroids_from_mpc(neo_only: bool = True, 
                            force_download: bool = False,
                            max_records: Optional[int] = None) -> List[Dict]:
    """
    Convenience function to download and parse MPC data
    
    Parameters:
    -----------
    neo_only : bool
        Load only NEOs
    force_download : bool
        Force re-download of data files
    max_records : int, optional
        Limit number of records (for testing)
        
    Returns:
    --------
    list of dict : Parsed asteroid data
    """
    loader = MPCLoader()
    
    # Download appropriate file
    if neo_only:
        filepath = loader.download_nea(force=force_download)
    else:
        filepath = loader.download_mpcorb(force=force_download)
    
    # Parse
    asteroids = loader.parse_mpc_format(
        filepath, 
        neo_only=neo_only,
        max_records=max_records
    )
    
    return asteroids
