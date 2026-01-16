"""
Designation Packing/Unpacking Utilities
Implements MPC packed designation format as per:
https://www.minorplanetcenter.net/iau/info/PackedDes.html
"""

import re
import logging

logger = logging.getLogger(__name__)


# Character encoding for packed format
# Used for numbers > 9 in various fields
PACK_CHARS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

def encode_base62(num):
    """Convert number to base-62 character (0-9, A-Z, a-z)"""
    if num < 0 or num >= 62:
        raise ValueError(f"Number {num} out of range for base-62 encoding")
    return PACK_CHARS[num]

def decode_base62(char):
    """Convert base-62 character to number"""
    if char in PACK_CHARS:
        return PACK_CHARS.index(char)
    raise ValueError(f"Invalid character '{char}' for base-62 decoding")


def pack_numbered_designation(number):
    """
    Pack a numbered asteroid designation
    
    Examples:
    - 433 → 00433
    - 100004 → A0004
    - 360000 → a0000
    - 620000 → ~0000
    - 620061 → ~000z
    - 3140113 → ~AZaz
    
    Format: ~NNNNN where ~ can be 0-9 for numbers < 100000
    or A-Z, a-z for 100000-619999, or ~ + base-62 for >= 620000
    """
    number = int(number)
    
    if number < 0:
        raise ValueError("Negative asteroid numbers not allowed")
    
    if number < 100000:
        # Simple case: pad with leading zeros
        return f"{number:05d}"
    elif number < 620000:
        # Use first character as base-62 for 100000s place
        hundreds_thousands = number // 10000
        remainder = number % 10000
        return encode_base62(hundreds_thousands) + f"{remainder:04d}"
    else:
        # High numbers: ~XXXX where XXXX is base-62 encoded (number - 620000)
        value = number - 620000
        # Encode as 4-digit base-62
        encoded = ''
        for _ in range(4):
            encoded = encode_base62(value % 62) + encoded
            value //= 62
        return '~' + encoded


def unpack_numbered_designation(packed):
    """
    Unpack a numbered asteroid designation
    
    Examples:
    - 00433 → 433
    - A0004 → 100004
    - a0000 → 360000
    - ~0000 → 620000
    - ~000z → 620061
    - ~AZaz → 3140113
    """
    if not packed or len(packed) != 5:
        raise ValueError(f"Invalid packed numbered designation: {packed}")
    
    first_char = packed[0]
    
    if first_char == '~':
        # High numbered asteroids (>= 620000)
        # Format: ~XXXX where XXXX is base-62 encoded (number - 620000)
        encoded = packed[1:5]
        # Decode as base-62 number
        value = 0
        for char in encoded:
            value = value * 62 + decode_base62(char)
        number = value + 620000
        return str(number)
    elif first_char.isdigit():
        # Simple case: all digits
        return str(int(packed))
    else:
        # First character is base-62 encoded (for 100000-619999)
        hundreds_thousands = decode_base62(first_char)
        remainder = int(packed[1:5])
        number = hundreds_thousands * 10000 + remainder
        return str(number)


def pack_provisional_designation(desig):
    """
    Pack a provisional designation
    
    Examples:
    - 1979 XB → J79X00B
    - 1998 SQ108 → J98SQ8H
    - 2024 AA → K24A00A
    - 2007 TA418 → K07Ts8I
    
    Format: CYYHC##L where:
    - C: century (I=18, J=19, K=20)
    - YY: year within century
    - H: half-month letter (A-Y, skipping I)
    - C: cycle letter (A-Z for cycles 1-26, or base-62 for higher)
    - ##: cycle number in base-62
    - L: second letter if designation has two letters (e.g., SQ)
    
    The format is actually CYYHMLLN where:
    - H is half-month
    - M is first letter of cycle indicator
    - LL is cycle number (can be encoded)
    - N is second half-month letter if present
    """
    # Parse provisional designation: YYYY XX### or YYYY X# or YYYY XX
    match = re.match(r'^(\d{4})\s+([A-Z])([A-Z]?)(\d*)$', desig.strip())
    if not match:
        raise ValueError(f"Invalid provisional designation format: {desig}")
    
    year_str, first_letter, second_letter, number_str = match.groups()
    year = int(year_str)
    
    # Century encoding
    if 1800 <= year < 1900:
        century = 'I'
    elif 1900 <= year < 2000:
        century = 'J'
    elif 2000 <= year < 2100:
        century = 'K'
    else:
        raise ValueError(f"Year {year} outside supported range (1800-2099)")
    
    # Year within century
    year_in_century = year % 100
    
    # Half-month letter
    half_month = first_letter
    
    # Cycle number
    cycle_num = int(number_str) if number_str else 0
    
    # Encode cycle number
    if cycle_num == 0:
        # No cycle
        cycle_enc = '00'
    elif cycle_num < 10:
        # Single digit - encode as 0# where # is the digit
        cycle_enc = f"0{cycle_num}"
    elif cycle_num < 100:
        # Two digits - encode directly
        cycle_enc = f"{cycle_num:02d}"
    else:
        # Larger number - use base-62 encoding
        # Cycle 100 = 10*10, so we encode tens and ones separately
        tens_digit = cycle_num // 10
        ones_digit = cycle_num % 10
        cycle_enc = encode_base62(tens_digit) + str(ones_digit)
    
    # Second letter (if present)
    second_char = second_letter if second_letter else ''
    
    # Build packed designation
    packed = f"{century}{year_in_century:02d}{half_month}{cycle_enc}{second_char}"
    
    return packed


def unpack_provisional_designation(packed):
    """
    Unpack a provisional designation
    
    Examples:
    - J79X00B → 1979 XB
    - J98SQ8H → 1998 SQ108
    - K24A00A → 2024 AA
    - K07Ts8I → 2007 TA418
    
    Format: CYYHMLLN where:
    - C: century (I=18, J=19, K=20)
    - YY: year within century
    - H: half-month letter
    - MLL: cycle encoded in base-62
    - N: second letter (if present)
    """
    if not packed or len(packed) < 6:
        raise ValueError(f"Invalid packed provisional designation: {packed}")
    
    # Century
    century_char = packed[0]
    if century_char == 'I':
        century = 1800
    elif century_char == 'J':
        century = 1900
    elif century_char == 'K':
        century = 2000
    else:
        raise ValueError(f"Invalid century character: {century_char}")
    
    # Year within century
    year_in_century = int(packed[1:3])
    year = century + year_in_century
    
    # Half-month letter
    half_month = packed[3]
    
    # Cycle encoding (positions 4-5)
    cycle_enc = packed[4:6]
    
    # Decode cycle
    try:
        if cycle_enc == '00':
            cycle_num = 0
        elif cycle_enc[0] == '0':
            # Single digit cycle
            cycle_num = int(cycle_enc[1])
        elif cycle_enc[0].isdigit() and cycle_enc[1].isdigit():
            # Two digit cycle
            cycle_num = int(cycle_enc)
        else:
            # Base-62 encoded
            tens = decode_base62(cycle_enc[0])
            ones = int(cycle_enc[1])
            cycle_num = tens * 10 + ones
    except Exception as e:
        logger.debug(f"Error decoding cycle from '{cycle_enc}': {e}")
        cycle_num = 0
    
    # Second letter (position 6 if present)
    second_letter = packed[6] if len(packed) > 6 and packed[6].isalpha() else ''
    
    # Build unpacked designation
    if cycle_num == 0:
        # No cycle number
        if second_letter:
            unpacked = f"{year} {half_month}{second_letter}"
        else:
            unpacked = f"{year} {half_month}"
    else:
        # Has cycle number
        if second_letter:
            unpacked = f"{year} {half_month}{second_letter}{cycle_num}"
        else:
            unpacked = f"{year} {half_month}{cycle_num}"
    
    return unpacked


def pack_designation(desig):
    """
    Pack any designation (numbered or provisional)
    
    Examples:
    - '433' → '00433'
    - '100004' → 'A0004'
    - '1979 XB' → 'J79X00B'
    - '2024 AA' → 'K24A00A'
    - '6344 P-L' → 'PLS6344' (Palomar-Leiden Survey special case)
    """
    desig = desig.strip().replace('(', '').replace(')', '')
    
    # Special case: Palomar-Leiden Survey format (e.g., "6344 P-L")
    if ' P-L' in desig:
        try:
            number = desig.split()[0]  # Extract number before space
            return f"PLS{number}"
        except:
            pass
    
    # Check if it's a numbered asteroid (all digits)
    if desig.isdigit():
        return pack_numbered_designation(int(desig))
    
    # Check if it's a provisional designation (YYYY XX...)
    if re.match(r'^\d{4}\s+[A-Z]', desig):
        return pack_provisional_designation(desig)
    
    # Already packed or unknown format
    return desig


def unpack_designation(packed):
    """
    Unpack any designation (numbered or provisional)
    
    Examples:
    - '00433' → '433'
    - 'A0004' → '100004'
    - 'J79X00B' → '1979 XB'
    - 'K24A00A' → '2024 AA'
    - 'PLS6344' → '6344 P-L' (Palomar-Leiden Survey special case)
    """
    packed = packed.strip()
    
    # Special case: Palomar-Leiden Survey format (e.g., PLS6344)
    if packed.startswith('PLS') and len(packed) == 7:
        try:
            number = packed[3:]  # Extract the number part
            return f"{number} P-L"
        except:
            pass
    
    # Check if it's 5 characters - could be numbered
    if len(packed) == 5:
        try:
            return unpack_numbered_designation(packed)
        except:
            pass
    
    # Check if it starts with I/J/K - likely provisional
    if len(packed) >= 6 and packed[0] in 'IJK':
        try:
            return unpack_provisional_designation(packed)
        except Exception as e:
            logger.debug(f"Failed to unpack provisional '{packed}': {e}")
            pass
    
    # Check if it's all digits (unpacked numbered)
    if packed.isdigit():
        return packed
    
    # Unknown format - return as-is
    return packed


def normalize_designation(desig):
    """
    Generate all reasonable variations of a designation for matching
    
    Returns a set of possible forms (both packed and unpacked)
    """
    if not desig:
        return set()
    
    variations = set()
    desig_clean = desig.strip().replace('(', '').replace(')', '')
    
    # Add original
    variations.add(desig_clean)
    
    try:
        # Try unpacking if it looks packed
        unpacked = unpack_designation(desig_clean)
        variations.add(unpacked)
        
        # Also add packed version of unpacked
        repacked = pack_designation(unpacked)
        variations.add(repacked)
    except:
        pass
    
    try:
        # Try packing if it looks unpacked
        packed = pack_designation(desig_clean)
        variations.add(packed)
        
        # Also add unpacked version of packed
        reunpacked = unpack_designation(packed)
        variations.add(reunpacked)
    except:
        pass
    
    # For numbered asteroids, also try without leading zeros
    if desig_clean.isdigit() or (len(desig_clean) == 5 and desig_clean[0].isdigit()):
        try:
            num = int(desig_clean)
            variations.add(str(num))  # Without leading zeros
            variations.add(f"{num:05d}")  # With leading zeros
        except:
            pass
    
    return variations


if __name__ == '__main__':
    # Test cases
    test_cases = [
        ('433', '00433'),
        ('1036', '01036'),
        ('100004', 'A0004'),
        ('620000', '~0000'),
        ('620061', '~000z'),
        ('3140113', '~AZaz'),
        ('1979 XB', 'J79X00B'),
        ('2024 AA', 'K24A00A'),
        ('6344 P-L', 'PLS6344'),  # Palomar-Leiden Survey
    ]
    
    print("Testing pack/unpack functions:")
    print("=" * 60)
    
    for unpacked, expected_packed in test_cases:
        print(f"\nTest: {unpacked} ↔ {expected_packed}")
        
        # Test packing
        try:
            packed = pack_designation(unpacked)
            print(f"  Pack: {unpacked} → {packed} {'✓' if packed == expected_packed else '✗ Expected: ' + expected_packed}")
        except Exception as e:
            print(f"  Pack error: {e}")
        
        # Test unpacking
        try:
            result = unpack_designation(expected_packed)
            print(f"  Unpack: {expected_packed} → {result} {'✓' if result == unpacked else '✗ Expected: ' + unpacked}")
        except Exception as e:
            print(f"  Unpack error: {e}")
        
        # Test normalization
        try:
            vars_from_unpacked = normalize_designation(unpacked)
            vars_from_packed = normalize_designation(expected_packed)
            common = vars_from_unpacked & vars_from_packed
            print(f"  Common variations: {common}")
        except Exception as e:
            print(f"  Normalization error: {e}")
