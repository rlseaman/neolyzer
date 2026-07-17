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


# Real lines from MPC NEA.txt (2026-04-10 generation) — fixed-width
# format fixtures for the canonical parser (plan item 4.1)
EROS_LINE = ("00433   10.40  0.15 K25BL 310.55432  178.92976  304.27010  "
             " 10.82847  0.2228360  0.55977529   1.4581210  0 E2026-G89 "
             "17462  59 1893-2026 0.62 M-v 3Ek MPCORBFIT  1804    (433) "
             "Eros               20260408")
XB79_LINE = ("J79X00B 18.6   0.15 K25BL 310.15706   77.08472   84.35362  "
             " 24.55290  0.7097545  0.29882912   2.2157501  9 MPO 70208  "
             "  16   1    4 days 1.39 M-v 3Eh MPCW       A803          "
             "1979 XB            19791215")


class TestParseMpcorbLine:
    """The canonical MPCORB fixed-width parser (parse_mpcorb_line) —
    partition_mpcorb.py and the diagnostics must consume this, not
    their own column offsets."""

    def test_eros_line(self):
        from mpc_loader import parse_mpcorb_line
        ast = parse_mpcorb_line(EROS_LINE)
        assert ast['designation'] == '00433'
        assert ast['H'] == 10.40
        assert ast['G'] == 0.15
        assert ast['epoch_jd'] == expected_jd(2025, 11, 21)  # K25BL
        assert ast['M'] == 310.55432
        assert ast['arg_peri'] == 178.92976
        assert ast['node'] == 304.27010
        assert ast['i'] == 10.82847
        assert ast['e'] == 0.2228360
        assert ast['mean_motion'] == 0.55977529
        assert ast['a'] == 1.4581210
        assert ast['num_obs'] == 17462
        assert ast['num_oppositions'] == 59
        assert '(433) Eros' in ast['readable_designation']

    def test_provisional_line(self):
        from mpc_loader import parse_mpcorb_line
        ast = parse_mpcorb_line(XB79_LINE)
        assert ast['designation'] == 'J79X00B'
        assert ast['e'] == 0.7097545
        assert ast['a'] == 2.2157501

    def test_header_and_short_lines_return_none(self):
        from mpc_loader import parse_mpcorb_line
        assert parse_mpcorb_line('') is None
        assert parse_mpcorb_line('Des\'n     H     G   Epoch') is None
        assert parse_mpcorb_line('-' * 200) is None

    def test_semi_major_axis_from_mean_motion(self):
        """When a is blank, derive it from mean motion n."""
        from mpc_loader import parse_mpcorb_line
        blanked = EROS_LINE[:92] + ' ' * 11 + EROS_LINE[103:]
        ast = parse_mpcorb_line(blanked)
        assert abs(ast['a'] - 1.4581210) < 1e-4
