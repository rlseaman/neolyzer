"""
Tests for DatabaseManager using in-memory SQLite.

No external data files needed — all tests use synthetic fixture data.
"""

import pytest
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from database import DatabaseManager, mjd_to_cln, CLN_EPOCH_JD, SYNODIC_MONTH


# ── Fixtures ─────────────────────────────────────────────────────

def make_asteroid(designation, **overrides):
    """Build a minimal asteroid dict with sensible defaults."""
    base = {
        'designation': designation,
        'a': 1.458,
        'e': 0.223,
        'i': 10.83,
        'node': 304.29,
        'arg_peri': 178.93,
        'M': 110.78,
        'epoch_jd': 2460000.5,
        'H': 15.0,
        'G': 0.15,
        'neo_flag': True,
        'pha_flag': False,
        'orbit_class': 'Amor',
        'num_obs': 100,
        'arc_years': 5.0,
    }
    base.update(overrides)
    return base


@pytest.fixture
def db():
    """In-memory SQLite database, fresh for each test."""
    return DatabaseManager(db_url='sqlite://')


@pytest.fixture
def populated_db(db):
    """Database with a handful of test asteroids."""
    asteroids = [
        make_asteroid('00433', H=10.39, neo_flag=True, pha_flag=False,
                       orbit_class='Amor', earth_moid=0.15, readable_designation='Eros'),
        make_asteroid('K24A00A', H=22.5, neo_flag=True, pha_flag=True,
                       orbit_class='Apollo', earth_moid=0.02),
        make_asteroid('01036', H=9.17, neo_flag=True, pha_flag=False,
                       orbit_class='Amor', earth_moid=0.34, readable_designation='Ganymed',
                       a=2.667, e=0.533, i=26.69),
        make_asteroid('A0004', H=18.0, neo_flag=True, pha_flag=False,
                       orbit_class='Aten', earth_moid=0.08),
        make_asteroid('J79X00B', H=None, neo_flag=True, pha_flag=False,
                       orbit_class='Apollo'),  # No H magnitude
    ]
    db.insert_asteroids(asteroids)
    return db


# ── CLN conversion ──────────────────────────────────────────────

class TestMjdToCln:
    def test_epoch_is_cln_zero(self):
        """The CLN epoch (JD 2444240.0076) should give CLN 0."""
        mjd = CLN_EPOCH_JD - 2400000.5
        assert mjd_to_cln(mjd) == 0

    def test_one_lunation_later(self):
        """One synodic month after epoch should be CLN 1."""
        mjd = (CLN_EPOCH_JD + SYNODIC_MONTH) - 2400000.5
        assert mjd_to_cln(mjd) == 1

    def test_none_input(self):
        assert mjd_to_cln(None) is None

    def test_negative_cln(self):
        """Before the epoch should give negative CLN."""
        mjd = (CLN_EPOCH_JD - 60) - 2400000.5  # ~2 lunations before
        assert mjd_to_cln(mjd) < 0

    def test_known_date(self):
        """2024-01-01 (MJD 60310) should give a reasonable CLN (~550)."""
        cln = mjd_to_cln(60310)
        assert 540 < cln < 560


# ── Insert and retrieve ─────────────────────────────────────────

class TestInsertAndRetrieve:
    def test_insert_and_count(self, populated_db):
        """Should have 5 asteroids after fixture insertion."""
        results = populated_db.get_asteroids()
        assert len(results) == 5

    def test_retrieve_by_designation(self, populated_db):
        """Look up Eros by packed designation."""
        ast = populated_db.get_asteroid_by_designation('00433')
        assert ast is not None
        assert ast['H'] == 10.39
        assert ast['readable_designation'] == 'Eros'

    def test_missing_designation_returns_none(self, populated_db):
        result = populated_db.get_asteroid_by_designation('ZZZZZ')
        assert result is None

    def test_dict_has_orbital_elements(self, populated_db):
        """Returned dict should have all orbital element keys."""
        ast = populated_db.get_asteroid_by_designation('00433')
        for key in ['a', 'e', 'i', 'node', 'arg_peri', 'M', 'epoch_jd', 'H', 'G']:
            assert key in ast

    def test_null_H_preserved(self, populated_db):
        """Objects with H=None should have None in the dict."""
        ast = populated_db.get_asteroid_by_designation('J79X00B')
        assert ast['H'] is None

    def test_upsert_updates_existing(self, populated_db):
        """Re-inserting with same designation should update, not duplicate."""
        updated = [make_asteroid('00433', H=11.0)]
        populated_db.insert_asteroids(updated)
        ast = populated_db.get_asteroid_by_designation('00433')
        assert ast['H'] == 11.0
        # Total count unchanged
        assert len(populated_db.get_asteroids()) == 5


# ── Filters ──────────────────────────────────────────────────────

class TestFilters:
    def test_neo_only(self, populated_db):
        """All test data is NEO, so neo_only should return all."""
        results = populated_db.get_asteroids(neo_only=True)
        assert len(results) == 5

    def test_pha_only(self, populated_db):
        """Only one PHA in fixture data."""
        results = populated_db.get_asteroids(pha_only=True)
        assert len(results) == 1
        assert results[0]['designation'] == 'K24A00A'

    def test_orbit_class_filter(self, populated_db):
        results = populated_db.get_asteroids(orbit_class='Amor')
        assert len(results) == 2  # Eros and Ganymed

    def test_h_max_filter(self, populated_db):
        """H <= 15 should exclude the H=22.5 and H=18 objects (and None)."""
        results = populated_db.get_asteroids(h_max=15.0)
        # H=10.39, H=9.17, H=15.0 (default for J79X00B is None, excluded by filter)
        designations = {r['designation'] for r in results}
        assert '00433' in designations  # H=10.39
        assert '01036' in designations  # H=9.17
        assert 'K24A00A' not in designations  # H=22.5

    def test_h_min_filter(self, populated_db):
        """H >= 15 should include only fainter objects."""
        results = populated_db.get_asteroids(h_min=15.0)
        for r in results:
            assert r['H'] is not None and r['H'] >= 15.0

    def test_moid_max_filter(self, populated_db):
        """MOID <= 0.05 should return only the closest approacher."""
        results = populated_db.get_asteroids(moid_max=0.05)
        assert len(results) == 1
        assert results[0]['designation'] == 'K24A00A'

    def test_moid_filter_excludes_null(self, populated_db):
        """MOID filters should exclude objects with no MOID data."""
        results = populated_db.get_asteroids(moid_max=1.0)
        for r in results:
            assert r['earth_moid'] is not None

    def test_limit(self, populated_db):
        results = populated_db.get_asteroids(limit=2)
        assert len(results) == 2

    def test_a_range_filter(self, populated_db):
        """Semi-major axis filter."""
        results = populated_db.get_asteroids(a_min=2.0, a_max=3.0)
        assert len(results) == 1
        assert results[0]['designation'] == '01036'  # Ganymed, a=2.667


# ── Statistics ───────────────────────────────────────────────────

class TestStatistics:
    def test_statistics_counts(self, populated_db):
        stats = populated_db.get_statistics()
        assert stats['total'] == 5
        assert stats['neos'] == 5
        assert stats['phas'] == 1

    def test_statistics_by_class(self, populated_db):
        stats = populated_db.get_statistics()
        assert stats['by_class']['Amor'] == 2
        assert stats['by_class']['Apollo'] == 2
        assert stats['by_class']['Aten'] == 1


# ── Clear ────────────────────────────────────────────────────────

class TestClear:
    def test_clear_all(self, populated_db):
        populated_db.clear_all()
        assert len(populated_db.get_asteroids()) == 0


# ── DataFrame export ─────────────────────────────────────────────

class TestDataFrameExport:
    def test_get_all_orbital_elements(self, populated_db):
        df = populated_db.get_all_orbital_elements()
        assert len(df) == 5
        assert 'a' in df.columns
        assert 'e' in df.columns

    def test_neo_only_dataframe(self, populated_db):
        df = populated_db.get_all_orbital_elements(neo_only=True)
        assert len(df) == 5  # all are NEOs in fixture
