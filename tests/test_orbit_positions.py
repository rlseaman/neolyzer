"""
Integration tests: FastOrbitCalculator positions vs JPL Horizons reference data.

These tests use osculating orbital elements at a specific epoch and verify
that the computed RA/Dec matches JPL Horizons geocentric astrometric positions.

Reference data source: JPL Horizons API (https://ssd.jpl.nasa.gov/horizons/)
Queried 2026-03-13 with EPHEM_TYPE=OBSERVER, CENTER=500@399 (geocentric).

Using elements AT the observation epoch minimizes propagation error,
isolating the coordinate-transform and geometry code.
"""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from orbit_calculator import FastOrbitCalculator


# ── Reference data from JPL Horizons ─────────────────────────────

# Osculating elements at the observation epoch (heliocentric ecliptic J2000)
# All angles in degrees, a in AU

EROS_ELEMENTS_JD2460000 = {
    'a': 1.4581,
    'e': 0.22278,
    'i': 10.828,
    'node': 304.287,
    'arg_peri': 178.927,
    'M': 110.778,
    'epoch_jd': 2460000.5,
    'H': 10.39,
    'G': 0.15,
}

GANYMED_ELEMENTS_JD2460000 = {
    'a': 2.6669,
    'e': 0.53302,
    'i': 26.686,
    'node': 215.499,
    'arg_peri': 132.468,
    'M': 231.158,
    'epoch_jd': 2460000.5,
    'H': 9.17,
    'G': 0.15,
}

EROS_ELEMENTS_JD2460400 = {
    'a': 1.4582,
    'e': 0.22271,
    'i': 10.828,
    'node': 304.277,
    'arg_peri': 178.895,
    'M': 334.727,
    'epoch_jd': 2460400.5,
    'H': 10.39,
    'G': 0.15,
}

# JPL Horizons geocentric astrometric RA/Dec (degrees)
HORIZONS_POSITIONS = {
    'eros_jd2460000': {'ra': 286.425, 'dec': -29.111},
    'ganymed_jd2460000': {'ra': 192.703, 'dec': -22.134},
    'eros_jd2460400': {'ra': 46.650, 'dec': 22.507},
}

# Tolerance: Keplerian propagation vs full N-body differs, plus
# osculating elements are rounded. 0.5° is generous but catches
# coordinate system bugs (which typically produce errors of many degrees).
POSITION_TOLERANCE_DEG = 0.5


# ── Fixtures ─────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def calculator():
    """Shared FastOrbitCalculator instance (loads ephemeris once)."""
    return FastOrbitCalculator()


# ── Position tests ───────────────────────────────────────────────

class TestHorizonsPositions:
    """Verify computed positions against JPL Horizons reference data."""

    def _compute_position(self, calc, elements, jd):
        """Compute RA/Dec for a single object at a given JD."""
        result = calc.calculate_batch([elements], jd)
        assert result.shape[0] == 1, "Expected exactly one result"
        # result columns: [id, ra, dec, distance, magnitude]
        return result[0, 1], result[0, 2]  # ra, dec

    def test_eros_jd2460000(self, calculator):
        """433 Eros on 2023-Feb-25 (elements at epoch)."""
        ra, dec = self._compute_position(
            calculator, EROS_ELEMENTS_JD2460000, 2460000.5
        )
        ref = HORIZONS_POSITIONS['eros_jd2460000']
        assert abs(ra - ref['ra']) < POSITION_TOLERANCE_DEG, \
            f"RA: got {ra:.3f}°, expected {ref['ra']:.3f}°"
        assert abs(dec - ref['dec']) < POSITION_TOLERANCE_DEG, \
            f"Dec: got {dec:.3f}°, expected {ref['dec']:.3f}°"

    def test_ganymed_jd2460000(self, calculator):
        """1036 Ganymed on 2023-Feb-25 (elements at epoch)."""
        ra, dec = self._compute_position(
            calculator, GANYMED_ELEMENTS_JD2460000, 2460000.5
        )
        ref = HORIZONS_POSITIONS['ganymed_jd2460000']
        assert abs(ra - ref['ra']) < POSITION_TOLERANCE_DEG, \
            f"RA: got {ra:.3f}°, expected {ref['ra']:.3f}°"
        assert abs(dec - ref['dec']) < POSITION_TOLERANCE_DEG, \
            f"Dec: got {dec:.3f}°, expected {ref['dec']:.3f}°"

    def test_eros_jd2460400(self, calculator):
        """433 Eros on 2024-Mar-31 (elements at epoch)."""
        ra, dec = self._compute_position(
            calculator, EROS_ELEMENTS_JD2460400, 2460400.5
        )
        ref = HORIZONS_POSITIONS['eros_jd2460400']
        # RA near 0°/360° wrap — handle wrap-around
        ra_diff = abs(ra - ref['ra'])
        if ra_diff > 180:
            ra_diff = 360 - ra_diff
        assert ra_diff < POSITION_TOLERANCE_DEG, \
            f"RA: got {ra:.3f}°, expected {ref['ra']:.3f}°"
        assert abs(dec - ref['dec']) < POSITION_TOLERANCE_DEG, \
            f"Dec: got {dec:.3f}°, expected {ref['dec']:.3f}°"


class TestBatchCalculation:
    """Verify batch mode produces same results as individual calculations."""

    def test_batch_matches_individual(self, calculator):
        """Multiple objects in one batch should match individual results."""
        elements_list = [
            EROS_ELEMENTS_JD2460000,
            GANYMED_ELEMENTS_JD2460000,
        ]
        jd = 2460000.5

        # Batch
        batch_result = calculator.calculate_batch(elements_list, jd)
        assert batch_result.shape[0] == 2

        # Individual
        for idx, elements in enumerate(elements_list):
            individual = calculator.calculate_batch([elements], jd)
            np.testing.assert_allclose(
                batch_result[idx, 1:],  # skip id column
                individual[0, 1:],
                atol=1e-10,
                err_msg=f"Batch/individual mismatch for object {idx}",
            )

    def test_empty_batch(self, calculator):
        """Empty input should return empty array with correct shape."""
        result = calculator.calculate_batch([], 2460000.5)
        assert result.shape == (0, 5)


class TestMagnitudes:
    """Verify magnitude calculations are physically reasonable."""

    def test_magnitude_finite(self, calculator):
        """Computed magnitudes should be finite numbers."""
        result = calculator.calculate_batch(
            [EROS_ELEMENTS_JD2460000], 2460000.5
        )
        mag = result[0, 4]
        assert np.isfinite(mag)

    def test_magnitude_reasonable_range(self, calculator):
        """NEO apparent magnitudes should be in a plausible range."""
        result = calculator.calculate_batch(
            [EROS_ELEMENTS_JD2460000], 2460000.5
        )
        mag = result[0, 4]
        # Eros H=10.39; apparent mag depends on geometry but should be
        # somewhere between ~8 (close approach) and ~25 (far/faint)
        assert 5 < mag < 30, f"Magnitude {mag} outside plausible range"

    def test_null_H_returns_sentinel(self, calculator):
        """Objects with H=None should get magnitude 99.0 sentinel."""
        elements = dict(EROS_ELEMENTS_JD2460000)
        elements['H'] = None
        result = calculator.calculate_batch([elements], 2460000.5)
        mag = result[0, 4]
        # H=None is replaced with 99.0 internally, so mag should be very large
        assert mag > 90


class TestPropagation:
    """Verify that propagation over moderate intervals stays reasonable."""

    def test_propagation_30_days(self, calculator):
        """Position 30 days from epoch should still be on the sky."""
        ra, dec = calculator.calculate_batch(
            [EROS_ELEMENTS_JD2460000], 2460000.5 + 30
        )[0, 1:3]
        assert 0 <= ra < 360
        assert -90 <= dec <= 90

    def test_propagation_one_year(self, calculator):
        """Position 1 year from epoch — larger error expected but still valid."""
        ra, dec = calculator.calculate_batch(
            [EROS_ELEMENTS_JD2460000], 2460000.5 + 365.25
        )[0, 1:3]
        assert 0 <= ra < 360
        assert -90 <= dec <= 90

    def test_position_changes_over_time(self, calculator):
        """Object should move between two dates."""
        pos1 = calculator.calculate_batch(
            [EROS_ELEMENTS_JD2460000], 2460000.5
        )[0, 1:3]
        pos2 = calculator.calculate_batch(
            [EROS_ELEMENTS_JD2460000], 2460000.5 + 30
        )[0, 1:3]
        # Should have moved at least a little
        assert not np.allclose(pos1, pos2, atol=0.01)
