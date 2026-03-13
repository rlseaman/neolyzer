"""Tests for MPC packed designation format handling."""

import pytest
from designation_utils import (
    encode_base62,
    decode_base62,
    pack_numbered_designation,
    unpack_numbered_designation,
    pack_provisional_designation,
    unpack_provisional_designation,
    pack_designation,
    unpack_designation,
    normalize_designation,
)


# ── Base-62 encoding ──────────────────────────────────────────────

class TestBase62:
    def test_digits(self):
        for i in range(10):
            assert encode_base62(i) == str(i)
            assert decode_base62(str(i)) == i

    def test_uppercase(self):
        assert encode_base62(10) == 'A'
        assert encode_base62(35) == 'Z'
        assert decode_base62('A') == 10
        assert decode_base62('Z') == 35

    def test_lowercase(self):
        assert encode_base62(36) == 'a'
        assert encode_base62(61) == 'z'
        assert decode_base62('a') == 36
        assert decode_base62('z') == 61

    def test_roundtrip(self):
        for i in range(62):
            assert decode_base62(encode_base62(i)) == i

    def test_out_of_range(self):
        with pytest.raises(ValueError):
            encode_base62(-1)
        with pytest.raises(ValueError):
            encode_base62(62)

    def test_invalid_char(self):
        with pytest.raises(ValueError):
            decode_base62('!')


# ── Numbered designations ────────────────────────────────────────

class TestNumberedDesignations:
    @pytest.mark.parametrize("number, expected", [
        (1, "00001"),
        (433, "00433"),
        (1036, "01036"),
        (99999, "99999"),
    ])
    def test_pack_simple(self, number, expected):
        assert pack_numbered_designation(number) == expected

    @pytest.mark.parametrize("number, expected", [
        (100004, "A0004"),
        (200000, "K0000"),
        (360000, "a0000"),
        (619999, "z9999"),
    ])
    def test_pack_base62_prefix(self, number, expected):
        assert pack_numbered_designation(number) == expected

    @pytest.mark.parametrize("number, expected", [
        (620000, "~0000"),
        (620061, "~000z"),
        (3140113, "~AZaz"),
    ])
    def test_pack_tilde_range(self, number, expected):
        assert pack_numbered_designation(number) == expected

    def test_pack_negative_raises(self):
        with pytest.raises(ValueError):
            pack_numbered_designation(-1)

    @pytest.mark.parametrize("packed, expected", [
        ("00433", "433"),
        ("01036", "1036"),
        ("A0004", "100004"),
        ("a0000", "360000"),
        ("~0000", "620000"),
        ("~000z", "620061"),
        ("~AZaz", "3140113"),
    ])
    def test_unpack(self, packed, expected):
        assert unpack_numbered_designation(packed) == expected

    def test_unpack_invalid_length(self):
        with pytest.raises(ValueError):
            unpack_numbered_designation("1234")
        with pytest.raises(ValueError):
            unpack_numbered_designation("123456")

    @pytest.mark.parametrize("number", [0, 1, 433, 99999, 100000, 100004,
                                         360000, 619999, 620000, 620061, 3140113])
    def test_roundtrip(self, number):
        packed = pack_numbered_designation(number)
        assert int(unpack_numbered_designation(packed)) == number


# ── Provisional designations ─────────────────────────────────────

class TestProvisionalDesignations:
    @pytest.mark.parametrize("desig, expected", [
        ("1979 XB", "J79X00B"),
        ("2024 AA", "K24A00A"),
        ("1801 AA", "I01A00A"),
    ])
    def test_pack_basic(self, desig, expected):
        assert pack_provisional_designation(desig) == expected

    @pytest.mark.parametrize("desig, expected", [
        ("1998 SQ108", "J98SA8Q"),   # cycle 108: base62(10)='A', ones=8
        ("2007 TA418", "K07Tf8A"),   # cycle 418: base62(41)='f', ones=8
    ])
    def test_pack_high_cycle(self, desig, expected):
        assert pack_provisional_designation(desig) == expected

    def test_pack_invalid_format(self):
        with pytest.raises(ValueError):
            pack_provisional_designation("not a designation")

    def test_pack_year_out_of_range(self):
        with pytest.raises(ValueError):
            pack_provisional_designation("1799 AA")
        with pytest.raises(ValueError):
            pack_provisional_designation("2100 AA")

    @pytest.mark.parametrize("packed, expected", [
        ("J79X00B", "1979 XB"),
        ("K24A00A", "2024 AA"),
    ])
    def test_unpack_basic(self, packed, expected):
        assert unpack_provisional_designation(packed) == expected

    @pytest.mark.parametrize("packed, expected", [
        ("J98SA8Q", "1998 SQ108"),
        ("K07Tf8A", "2007 TA418"),
    ])
    def test_unpack_high_cycle(self, packed, expected):
        assert unpack_provisional_designation(packed) == expected

    def test_unpack_invalid(self):
        with pytest.raises(ValueError):
            unpack_provisional_designation("")
        with pytest.raises(ValueError):
            unpack_provisional_designation("short")

    @pytest.mark.parametrize("desig", [
        "1979 XB", "2024 AA", "1998 SQ108",
    ])
    def test_roundtrip(self, desig):
        packed = pack_provisional_designation(desig)
        assert unpack_provisional_designation(packed) == desig


# ── Generic pack/unpack (auto-detect format) ─────────────────────

class TestGenericDesignation:
    def test_pack_numbered(self):
        assert pack_designation("433") == "00433"

    def test_pack_provisional(self):
        assert pack_designation("2024 AA") == "K24A00A"

    def test_pack_palomar_leiden(self):
        assert pack_designation("6344 P-L") == "PLS6344"

    def test_unpack_palomar_leiden(self):
        assert unpack_designation("PLS6344") == "6344 P-L"

    def test_pack_strips_parens(self):
        assert pack_designation("(433)") == "00433"

    def test_pack_already_packed(self):
        # Unknown format returned as-is
        assert pack_designation("J79X00B") == "J79X00B"

    def test_unpack_plain_number(self):
        assert unpack_designation("433") == "433"

    def test_unpack_unknown_format(self):
        assert unpack_designation("???") == "???"


# ── Normalization ────────────────────────────────────────────────

class TestNormalization:
    def test_empty(self):
        assert normalize_designation("") == set()
        assert normalize_designation(None) == set()

    def test_numbered_produces_both_forms(self):
        variations = normalize_designation("433")
        assert "433" in variations
        assert "00433" in variations

    def test_packed_and_unpacked_share_common(self):
        from_unpacked = normalize_designation("433")
        from_packed = normalize_designation("00433")
        assert from_unpacked & from_packed  # non-empty intersection

    def test_provisional(self):
        variations = normalize_designation("2024 AA")
        assert "2024 AA" in variations
        assert "K24A00A" in variations
