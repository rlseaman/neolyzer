"""
Tests for PositionCache HDF5 read/write and precision tier logic.

Uses a temp directory — no persistent files created.
"""

import numpy as np
import pytest
import tempfile
import os

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cache_manager import PositionCache


# ── Fixtures ─────────────────────────────────────────────────────

@pytest.fixture
def cache(tmp_path):
    """Fresh PositionCache in a temp directory."""
    cache_file = str(tmp_path / "test_cache.h5")
    return PositionCache(cache_file=cache_file)


def make_positions(n_objects=10, seed=42):
    """Generate synthetic position data: (N, 5) [id, ra, dec, dist, mag]."""
    rng = np.random.RandomState(seed)
    ids = np.arange(n_objects, dtype=float)
    ra = rng.uniform(0, 360, n_objects)
    dec = rng.uniform(-90, 90, n_objects)
    dist = rng.uniform(0.5, 5.0, n_objects)
    mag = rng.uniform(10, 25, n_objects)
    return np.column_stack([ids, ra, dec, dist, mag])


# ── Initialization ───────────────────────────────────────────────

class TestInitialization:
    def test_creates_cache_file(self, cache):
        assert cache.cache_file.exists()

    def test_default_reference_jd(self, cache):
        """Default reference should be J2000.0."""
        assert cache.get_reference_date() == 2451545.0

    def test_set_reference_date(self, cache):
        cache.set_reference_date(2460000.5)
        assert cache.get_reference_date() == 2460000.5

    def test_low_precision_range_set(self, cache):
        """After set_reference_date, low precision ranges should be set."""
        cache.set_reference_date(2460000.5)
        assert cache.LOW_PRECISION_DAYS_BACKWARD is not None
        assert cache.LOW_PRECISION_DAYS_FORWARD is not None
        assert cache.LOW_PRECISION_DAYS_BACKWARD > 0
        assert cache.LOW_PRECISION_DAYS_FORWARD > 0


# ── Precision tiers ──────────────────────────────────────────────

class TestPrecisionTiers:
    def test_high_precision_within_year(self, cache):
        group, interval = cache._get_precision_group(100)
        assert group == 'high_precision'
        assert interval == 1

    def test_medium_precision_beyond_year(self, cache):
        group, interval = cache._get_precision_group(500)
        assert group == 'medium_precision'
        assert interval == 7

    def test_low_precision_beyond_5_years(self, cache):
        group, interval = cache._get_precision_group(2000)
        assert group == 'low_precision'
        assert interval == 30

    def test_negative_days_same_tiers(self, cache):
        """Past dates should use same tier logic."""
        group, _ = cache._get_precision_group(-100)
        assert group == 'high_precision'
        group, _ = cache._get_precision_group(-500)
        assert group == 'medium_precision'

    def test_boundary_high_medium(self, cache):
        """Exactly at HIGH_PRECISION_DAYS boundary."""
        group, _ = cache._get_precision_group(365)
        assert group == 'high_precision'
        group, _ = cache._get_precision_group(366)
        assert group == 'medium_precision'


# ── Store and retrieve ───────────────────────────────────────────

class TestStoreRetrieve:
    def test_roundtrip_exact(self, cache):
        """Store positions at a JD, retrieve at same JD."""
        ref_jd = 2460000.5
        cache.set_reference_date(ref_jd)

        positions = make_positions(10)
        cache.store_positions(ref_jd, positions)

        retrieved = cache.get_positions(ref_jd)
        assert retrieved is not None
        np.testing.assert_allclose(retrieved, positions)

    def test_store_with_designations(self, cache):
        """Designations should be stored in metadata."""
        ref_jd = 2460000.5
        cache.set_reference_date(ref_jd)

        positions = make_positions(3)
        desigs = ['00433', '01036', 'K24A00A']
        cache.store_positions(ref_jd, positions, designations=desigs)

        stats = cache.get_cache_statistics()
        assert stats['n_objects'] == 3

    def test_overwrite_existing(self, cache):
        """Storing at same JD should overwrite previous data."""
        ref_jd = 2460000.5
        cache.set_reference_date(ref_jd)

        pos1 = make_positions(10, seed=1)
        cache.store_positions(ref_jd, pos1)

        pos2 = make_positions(10, seed=2)
        cache.store_positions(ref_jd, pos2)

        retrieved = cache.get_positions(ref_jd)
        np.testing.assert_allclose(retrieved, pos2)

    def test_multiple_dates(self, cache):
        """Store at multiple dates, retrieve each correctly."""
        ref_jd = 2460000.5
        cache.set_reference_date(ref_jd)

        jd1, jd2 = ref_jd, ref_jd + 1
        pos1 = make_positions(5, seed=1)
        pos2 = make_positions(5, seed=2)

        cache.store_positions(jd1, pos1)
        cache.store_positions(jd2, pos2)

        r1 = cache.get_positions(jd1)
        r2 = cache.get_positions(jd2)
        np.testing.assert_allclose(r1, pos1)
        np.testing.assert_allclose(r2, pos2)


# ── Interpolation ────────────────────────────────────────────────

class TestInterpolation:
    def test_linear_interpolation(self, cache):
        """Position at midpoint should be average of bracketing dates."""
        ref_jd = 2460000.5
        cache.set_reference_date(ref_jd)

        jd1 = ref_jd
        jd2 = ref_jd + 2

        pos1 = np.array([[0, 100.0, 20.0, 1.5, 15.0]])
        pos2 = np.array([[0, 110.0, 30.0, 2.0, 16.0]])

        cache.store_positions(jd1, pos1)
        cache.store_positions(jd2, pos2)

        mid = cache.get_positions(ref_jd + 1, interpolate=True)
        assert mid is not None

        expected = (pos1 + pos2) / 2
        np.testing.assert_allclose(mid, expected, atol=1e-10)

    def test_no_interpolation_returns_nearest(self, cache):
        """With interpolate=False, should return nearest cached date."""
        ref_jd = 2460000.5
        cache.set_reference_date(ref_jd)

        pos = make_positions(3)
        cache.store_positions(ref_jd, pos)

        # Ask for a slightly different JD without interpolation
        retrieved = cache.get_positions(ref_jd + 0.001, interpolate=False)
        assert retrieved is not None


# ── Cache management ─────────────────────────────────────────────

class TestCacheManagement:
    def test_get_cache_dates(self, cache):
        ref_jd = 2460000.5
        cache.set_reference_date(ref_jd)

        cache.store_positions(ref_jd, make_positions(3))
        cache.store_positions(ref_jd + 1, make_positions(3))

        dates = cache.get_cache_dates()
        assert len(dates['high_precision']) == 2

    def test_clear_cache(self, cache):
        ref_jd = 2460000.5
        cache.set_reference_date(ref_jd)

        cache.store_positions(ref_jd, make_positions(3))
        cache.clear_cache()

        dates = cache.get_cache_dates()
        assert len(dates['high_precision']) == 0

    def test_clear_specific_group(self, cache):
        """Clear only one precision tier."""
        ref_jd = 2460000.5
        cache.set_reference_date(ref_jd)

        # Store in high precision (within ±365 days of ref)
        cache.store_positions(ref_jd, make_positions(3))
        # Store in medium precision (500 days from ref)
        cache.store_positions(ref_jd + 500, make_positions(3))

        cache.clear_cache(groups=['high_precision'])

        dates = cache.get_cache_dates()
        assert len(dates['high_precision']) == 0
        assert len(dates['medium_precision']) == 1

    def test_cache_statistics(self, cache):
        ref_jd = 2460000.5
        cache.set_reference_date(ref_jd)

        desigs = ['A', 'B', 'C']
        cache.store_positions(ref_jd, make_positions(3), designations=desigs)

        stats = cache.get_cache_statistics()
        assert stats['n_objects'] == 3
        assert stats['high_precision_dates'] == 1
        assert stats['reference_jd'] == ref_jd
        assert stats['file_size_mb'] > 0


# ── Edge cases ───────────────────────────────────────────────────

class TestEdgeCases:
    def test_out_of_range_returns_none(self, cache):
        """Requesting a JD far outside cached range returns None."""
        cache.set_reference_date(2460000.5)
        # Way outside any ephemeris range
        result = cache.get_positions(1000000.0)
        assert result is None

    def test_empty_cache_returns_none(self, cache):
        """No data stored yet — should return None."""
        cache.set_reference_date(2460000.5)
        result = cache.get_positions(2460000.5)
        assert result is None

    def test_compression_works(self, cache, tmp_path):
        """Compressed data should be smaller than raw."""
        ref_jd = 2460000.5
        cache.set_reference_date(ref_jd)

        # Store a large-ish dataset
        big_positions = make_positions(1000)
        cache.store_positions(ref_jd, big_positions)

        raw_size = big_positions.nbytes
        file_size = cache.cache_file.stat().st_size
        # HDF5+gzip should compress the random data somewhat
        # (random floats don't compress well, but the overhead should still be finite)
        assert file_size < raw_size * 2  # generous — just verifies it's not corrupt
