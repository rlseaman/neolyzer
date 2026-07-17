"""
Orbit Calculator - Efficient Keplerian orbit propagation for asteroids
Uses Skyfield for high-accuracy orbital mechanics
"""

import numpy as np
from typing import List, Dict, Optional
import logging

# Use our centralized loader with SSL fallback
from skyfield_loader import skyfield_load, load_ephemeris

logger = logging.getLogger(__name__)


# Vectorized version for extreme performance
class FastOrbitCalculator:
    """
    Vectorized orbit calculator for computing thousands of positions simultaneously
    Uses NumPy broadcasting for 10-100x speedup
    """
    
    def __init__(self):
        # Load ephemeris using our SSL-fallback loader (uses configured ephemeris)
        self.eph = load_ephemeris()
        
        # Use our loader for timescale
        self.ts = skyfield_load.timescale()
        self.sun = self.eph['sun']
        self.earth = self.eph['earth']
    
    def calculate_positions_vectorized(self,
                                      a_arr: np.ndarray,
                                      e_arr: np.ndarray,
                                      i_arr: np.ndarray,
                                      omega_arr: np.ndarray,
                                      Omega_arr: np.ndarray,
                                      M0_arr: np.ndarray,
                                      H_arr: np.ndarray,
                                      epoch_jd_arr: np.ndarray,
                                      time_jd: float,
                                      mag_limit: Optional[float] = None) -> np.ndarray:
        """
        Vectorized calculation for maximum performance
        
        All arrays should have the same length (number of asteroids)
        Returns: (N, 4) array of [ra, dec, distance, magnitude]
        """
        n_objects = len(a_arr)
        
        # Convert to radians
        i_rad = np.deg2rad(i_arr)
        omega_rad = np.deg2rad(omega_arr)
        Omega_rad = np.deg2rad(Omega_arr)
        M0_rad = np.deg2rad(M0_arr)
        
        # Mean motion (rad/day) - using Gaussian gravitational constant
        # k = 0.01720209895 (AU^3/2 per day for Sun's gravitational parameter)
        k = 0.01720209895
        n = k / np.sqrt(a_arr**3)  # Mean motion in rad/day
        
        # Mean anomaly at time_jd
        dt = time_jd - epoch_jd_arr
        M = M0_rad + n * dt
        
        # Solve Kepler's equation (vectorized)
        E = self._solve_kepler_vectorized(M, e_arr)
        
        # True anomaly
        nu = 2.0 * np.arctan2(
            np.sqrt(1 + e_arr) * np.sin(E / 2),
            np.sqrt(1 - e_arr) * np.cos(E / 2)
        )
        
        # Distance and orbital plane coordinates
        r = a_arr * (1 - e_arr * np.cos(E))
        x_orb = r * np.cos(nu)
        y_orb = r * np.sin(nu)
        
        # Pre-compute trig functions
        cos_omega, sin_omega = np.cos(omega_rad), np.sin(omega_rad)
        cos_Omega, sin_Omega = np.cos(Omega_rad), np.sin(Omega_rad)
        cos_i, sin_i = np.cos(i_rad), np.sin(i_rad)
        
        # Transform to ecliptic (vectorized)
        x = (cos_Omega * cos_omega - sin_Omega * sin_omega * cos_i) * x_orb + \
            (-cos_Omega * sin_omega - sin_Omega * cos_omega * cos_i) * y_orb
        y = (sin_Omega * cos_omega + cos_Omega * sin_omega * cos_i) * x_orb + \
            (-sin_Omega * sin_omega + cos_Omega * cos_omega * cos_i) * y_orb
        z = (sin_i * sin_omega) * x_orb + (sin_i * cos_omega) * y_orb
        
        # Heliocentric positions (AU) - in ecliptic coordinates
        positions_helio = np.column_stack([x, y, z])
        
        # Get Earth position (Skyfield returns EQUATORIAL coordinates)
        t = self.ts.tt_jd(time_jd)
        earth_pos_eq = self.earth.at(t).position.au
        
        # CRITICAL FIX: Convert Earth position from equatorial to ecliptic
        # Obliquity of ecliptic (J2000.0) = 23.43928 degrees
        epsilon = np.radians(23.43928)
        cos_eps = np.cos(epsilon)
        sin_eps = np.sin(epsilon)
        
        # Rotation from equatorial to ecliptic (inverse of the usual rotation)
        x_eq, y_eq, z_eq = earth_pos_eq
        x_earth_ecl = x_eq
        y_earth_ecl = cos_eps * y_eq + sin_eps * z_eq
        z_earth_ecl = -sin_eps * y_eq + cos_eps * z_eq
        earth_pos_ecl = np.array([x_earth_ecl, y_earth_ecl, z_earth_ecl])
        
        # Geocentric positions (now both in ecliptic coordinates)
        positions_geo_ecliptic = positions_helio - earth_pos_ecl
        
        # Transform from ecliptic to equatorial coordinates
        x_ecl, y_ecl, z_ecl = positions_geo_ecliptic.T
        
        # Rotation matrix from ecliptic to equatorial
        x_eq = x_ecl
        y_eq = cos_eps * y_ecl - sin_eps * z_ecl
        z_eq = sin_eps * y_ecl + cos_eps * z_ecl
        
        # Convert to RA/Dec (now in equatorial coordinates)
        distance = np.sqrt(x_eq**2 + y_eq**2 + z_eq**2)
        ra = np.rad2deg(np.arctan2(y_eq, x_eq))
        ra = np.where(ra < 0, ra + 360, ra)
        dec = np.rad2deg(np.arcsin(z_eq / distance))
        
        # Calculate magnitudes
        r_sun = np.sqrt(np.sum(positions_helio**2, axis=1))
        mag = H_arr + 5 * np.log10(r_sun * distance)
        
        # Filter by magnitude if requested
        if mag_limit is not None:
            mask = mag < mag_limit
            ra = ra[mask]
            dec = dec[mask]
            distance = distance[mask]
            mag = mag[mask]
        
        return np.column_stack([ra, dec, distance, mag])
    
    @staticmethod
    def _solve_kepler_vectorized(M: np.ndarray, e: np.ndarray,
                                 tol: float = 1e-10) -> np.ndarray:
        """Vectorized Kepler equation solver

        M is reduced to [0, 2*pi) before iterating: the E=pi seed used
        for high-e orbits diverges chaotically when M lies far outside
        that range (M0 + n*dt grows without bound), silently producing
        garbage positions for e >= ~0.8 objects at some dates.
        """
        M = np.mod(M, 2.0 * np.pi)
        E = np.where(e < 0.8, M, np.full_like(M, np.pi))

        for _ in range(30):
            f = E - e * np.sin(E) - M
            fp = 1 - e * np.cos(E)
            delta = f / fp
            E -= delta

            if np.all(np.abs(delta) < tol):
                break

        residual = np.abs(E - e * np.sin(E) - M)
        n_bad = int(np.count_nonzero(residual > 1e-6))
        if n_bad:
            logger.warning(f"Kepler solver did not converge for {n_bad} "
                           f"objects (max residual {residual.max():.2e} rad)")
        return E
    
    def calculate_batch(self, 
                       elements_list: List[Dict],
                       time_jd: float,
                       mag_limit: Optional[float] = None) -> np.ndarray:
        """
        Calculate positions for multiple asteroids (compatibility wrapper)
        Converts list of dicts to arrays and calls vectorized method
        
        Parameters:
        -----------
        elements_list : list of dict
            List of orbital element dictionaries
        time_jd : float
            Julian date
        mag_limit : float, optional
            Only return objects brighter than this magnitude
            
        Returns:
        --------
        ndarray : Array of (id, ra, dec, dist, mag) for each object
        """
        if not elements_list:
            return np.array([]).reshape(0, 5)
        
        # Convert list of dicts to arrays
        n = len(elements_list)
        a_arr = np.array([e['a'] for e in elements_list])
        e_arr = np.array([e['e'] for e in elements_list])
        i_arr = np.array([e['i'] for e in elements_list])
        omega_arr = np.array([e['arg_peri'] for e in elements_list])
        Omega_arr = np.array([e['node'] for e in elements_list])
        M0_arr = np.array([e['M'] for e in elements_list])
        # Handle None H values (new discoveries without measured magnitude)
        H_arr = np.array([e['H'] if e['H'] is not None else 99.0 for e in elements_list])
        epoch_jd_arr = np.array([e['epoch_jd'] for e in elements_list])
        
        # Call vectorized method
        positions = self.calculate_positions_vectorized(
            a_arr, e_arr, i_arr, omega_arr, Omega_arr, M0_arr, H_arr, epoch_jd_arr,
            time_jd, mag_limit
        )
        
        # Add IDs (first column)
        if len(positions) > 0:
            ids = np.arange(len(positions)).reshape(-1, 1)
            return np.hstack([ids, positions])
        else:
            return np.array([]).reshape(0, 5)
