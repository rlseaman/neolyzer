"""Tests for MPC catalog parsing in mpc_loader.py — currently the packed
epoch decoder (regression tests for the TT/UTC epoch bug, plan item 4.3;
see docs/EPOCH_TT_INVESTIGATION_16Jul26.md)."""

from datetime import date

import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mpc_loader import MPCLoader

# JD of 2000-01-01.0 TT — anchor for computing expected values
# independently of the implementation's calendar arithmetic
JD_2000_JAN_1 = 2451544.5


def expected_jd(year, month, day):
    """JD at 0h TT of a Gregorian calendar date, via datetime.date."""
    return JD_2000_JAN_1 + (date(year, month, day) - date(2000, 1, 1)).days


class TestUnpackEpoch:
    """MPC packed epoch (KYYMD) → Julian Date, .0 TT."""

    @pytest.mark.parametrize("packed,ymd", [
        ("K0011", (2000, 1, 1)),    # J2000 era
        ("K261A", (2026, 1, 10)),   # day code A = 10
        ("K25AV", (2025, 10, 31)),  # month code A = Oct, day V = 31
        ("K25C1", (2025, 12, 1)),   # month code C = Dec
        ("J9611", (1996, 1, 1)),    # 1900s century code
        ("J699N", (1969, 9, 23)),   # older epoch
        ("K2029", (2020, 2, 9)),    # leap year, February
    ])
    def test_known_dates(self, packed, ymd):
        assert MPCLoader._unpack_epoch(packed) == expected_jd(*ymd)

    def test_epoch_is_0h_tt(self):
        """A .0 TT epoch is a JD ending in exactly .5 — the TT/UTC bug
        added ΔT (~64-69 s, fraction .50074-.50080) to every epoch."""
        for packed in ["K0011", "K261A", "K25AV", "J9611"]:
            jd = MPCLoader._unpack_epoch(packed)
            assert jd % 1.0 == 0.5, f"{packed}: epoch {jd} not at 0h TT"

    def test_malformed_defaults_to_j2000(self):
        for packed in ["", None, "K26", "X"]:
            assert MPCLoader._unpack_epoch(packed) == 2451545.0
