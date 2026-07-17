"""Tests for src/sky_math.py — the shared spherical-astronomy helpers
that replaced ~15 copy-pasted formula blocks in neolyzer.py."""

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sky_math import (OBLIQUITY_J2000_DEG, OBLIQUITY_J2000_RAD,
                      cos_angular_separation, angular_separation_deg)

r = np.radians


class TestObliquity:
    def test_value(self):
        assert OBLIQUITY_J2000_DEG == 23.43928
        assert OBLIQUITY_J2000_RAD == np.radians(23.43928)


class TestCosAngularSeparation:
    def test_coincident(self):
        assert cos_angular_separation(r(10), r(20), r(10), r(20)) == 1.0

    def test_orthogonal(self):
        assert abs(cos_angular_separation(0, 0, r(90), 0)) < 1e-15

    def test_antipodal(self):
        assert cos_angular_separation(0, 0, r(180), 0) == -1.0

    def test_clipping(self):
        # identical points must never exceed 1.0 through rounding
        c = cos_angular_separation(r(123.456), r(-45.678),
                                   r(123.456), r(-45.678))
        assert c <= 1.0

    def test_vectorized(self):
        ra = r(np.array([0.0, 90.0, 180.0]))
        dec = np.zeros(3)
        c = cos_angular_separation(ra, dec, 0.0, 0.0)
        np.testing.assert_allclose(c, [1.0, 0.0, -1.0], atol=1e-15)


class TestAngularSeparationDeg:
    def test_matches_arccos_at_moderate_angles(self):
        rng = [(10, 20, 50, -30), (0, 0, 120, 45), (300, 80, 100, -80)]
        for ra1, dec1, ra2, dec2 in rng:
            hav = angular_separation_deg(r(ra1), r(dec1), r(ra2), r(dec2))
            loc = np.degrees(np.arccos(
                cos_angular_separation(r(ra1), r(dec1), r(ra2), r(dec2))))
            assert abs(hav - loc) < 1e-9

    def test_small_angle_accuracy(self):
        # 1 arcsecond apart in dec — haversine keeps full precision here
        sep = angular_separation_deg(0.0, 0.0, 0.0, r(1 / 3600))
        assert abs(sep - 1 / 3600) < 1e-12

    def test_symmetry(self):
        a = angular_separation_deg(r(10), r(5), r(200), r(-40))
        b = angular_separation_deg(r(200), r(-40), r(10), r(5))
        assert a == b

    def test_ra_wraparound(self):
        # 359° and 1° RA on the equator are 2° apart
        sep = angular_separation_deg(r(359), 0.0, r(1), 0.0)
        assert abs(sep - 2.0) < 1e-9
