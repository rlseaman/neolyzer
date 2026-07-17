"""
Shared spherical-astronomy math for NEOlyzer.

Single home for the J2000 obliquity constant and great-circle
separation formulas that were previously copy-pasted (with drifting
precision) across neolyzer.py and orbit_calculator.py.

All angles are radians unless a function name says otherwise.
Functions accept scalars or NumPy arrays (broadcasting applies).
"""

import numpy as np

# Mean obliquity of the ecliptic at J2000.0, degrees (IAU 2006 value to
# the precision used throughout NEOlyzer). One truncation, one place —
# the heliocentric chart previously used 23.439 while the sky map used
# 23.43928, making the two subtly disagree.
OBLIQUITY_J2000_DEG = 23.43928
OBLIQUITY_J2000_RAD = np.radians(OBLIQUITY_J2000_DEG)


def cos_angular_separation(ra1, dec1, ra2, dec2):
    """
    Cosine of the great-circle separation (spherical law of cosines),
    clipped into [-1, 1]. Use when the cosine itself feeds a further
    computation (e.g. the law-of-cosines heliocentric distance); use
    angular_separation_deg() when you want the angle.
    """
    c = (np.sin(dec1) * np.sin(dec2)
         + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2))
    return np.clip(c, -1.0, 1.0)


def angular_separation_deg(ra1, dec1, ra2, dec2):
    """
    Great-circle separation in degrees via the haversine formula,
    which stays accurate at small separations where the law of
    cosines loses precision.
    """
    a = (np.sin((dec2 - dec1) / 2.0) ** 2
         + np.cos(dec1) * np.cos(dec2) * np.sin((ra2 - ra1) / 2.0) ** 2)
    return 2.0 * np.degrees(np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))
