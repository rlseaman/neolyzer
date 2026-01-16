#!/usr/bin/env python3
"""
CLN Diagnostic Script - Compare different CLN calculation methods

WHY TWO METHODS?
================

The display and filter use the "precise" method (jd_to_cln_precise) which:
- Finds the ACTUAL last Full Moon using Skyfield ephemeris
- CLN changes exactly at each Full Moon moment
- Requires loading de421.bsp ephemeris file
- Takes ~10-50ms per calculation

The database stores discovery_cln using the "average" method which:
- Uses simple arithmetic: floor((JD - epoch) / 29.530588853)
- Is ~1000x faster (no ephemeris lookup)
- Can be computed for 40,000+ asteroids in <100ms
- Has ~1 day discrepancy near Full Moon boundaries

This means:
- The CLN shown in the UI (and used for filtering) is precise
- The discovery_cln stored in database uses average method
- Near Full Moon, an asteroid discovered on "CLN 569" (average method)
  might display under "CLN 568" (precise method) for about 1 day

This discrepancy is acceptable because:
1. It only affects ~1 day out of 29.5 days per lunation
2. It only matters for discoveries made within ~1 day of Full Moon
3. Computing precise CLN for all 40,000+ asteroids would be too slow
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from datetime import datetime, timedelta
from skyfield.api import load as skyfield_load
from skyfield.api import utc

# CLN constants
CLN_EPOCH_JD = 2444240.0076
SYNODIC_MONTH = 29.530588853


def jd_to_cln_average(jd):
    """Average-based CLN (floor) - USED BY DATABASE for discovery_cln"""
    days_from_epoch = jd - CLN_EPOCH_JD
    lunations = days_from_epoch / SYNODIC_MONTH
    cln = int(np.floor(lunations))
    days_offset = (lunations - cln) * SYNODIC_MONTH
    return cln, days_offset


def compute_moon_phase(jd):
    """Compute lunar phase angle (0=New, 180=Full)"""
    try:
        ts = skyfield_load.timescale()
        t = ts.tt_jd(jd)
        eph = skyfield_load('de421.bsp')
        
        sun = eph['sun']
        moon = eph['moon']
        earth = eph['earth']
        
        e = earth.at(t)
        sun_pos = e.observe(sun).apparent()
        moon_pos = e.observe(moon).apparent()
        
        _, sun_lon, _ = sun_pos.ecliptic_latlon()
        _, moon_lon, _ = moon_pos.ecliptic_latlon()
        
        phase = (moon_lon.degrees - sun_lon.degrees) % 360
        return phase
    except Exception as e:
        print(f"Error computing phase: {e}")
        return None


def find_last_full_moon(jd):
    """Find precise JD of last Full Moon before jd"""
    phase = compute_moon_phase(jd)
    if phase is None:
        return None
    
    # Estimate how many days since Full Moon (phase 180)
    if phase > 180:
        days_since_full = (phase - 180) / (360 / SYNODIC_MONTH)
    else:
        days_since_full = (phase + 180) / (360 / SYNODIC_MONTH)
    
    estimated_full_jd = jd - days_since_full
    
    # Refine
    for _ in range(5):
        est_phase = compute_moon_phase(estimated_full_jd)
        if est_phase is None:
            break
        phase_error = est_phase - 180
        if abs(phase_error) < 0.01:
            break
        day_correction = (phase_error / 360) * SYNODIC_MONTH
        estimated_full_jd -= day_correction
    
    return estimated_full_jd


def jd_to_cln_precise(jd):
    """Precise CLN using actual Full Moon detection - USED BY DISPLAY/FILTER"""
    last_full_jd = find_last_full_moon(jd)
    if last_full_jd is None:
        return None
    
    days_offset = jd - last_full_jd
    lunations_from_epoch = (last_full_jd - CLN_EPOCH_JD) / SYNODIC_MONTH
    cln = int(round(lunations_from_epoch))
    
    return cln, days_offset


def jd_to_datetime(jd):
    """Convert JD to datetime string"""
    ts = skyfield_load.timescale()
    t = ts.tt_jd(jd)
    dt = t.utc_datetime()
    return dt.strftime('%Y-%m-%d %H:%M UTC')


def main():
    print("=" * 80)
    print("CLN DIAGNOSTIC - Comparing calculation methods")
    print("=" * 80)
    print()
    
    # Get current time
    ts = skyfield_load.timescale()
    t = ts.now()
    current_jd = t.tt
    
    print(f"Current time: {jd_to_datetime(current_jd)}")
    print(f"Current JD: {current_jd:.4f}")
    print()
    print("Methods:")
    print("  DB CLN: floor((JD - epoch) / synodic) - Used for discovery_cln in database")
    print("  Precise: Uses actual Full Moon detection - Used for display/filter")
    print()
    
    # Test around current time
    print("Testing +/- 20 days from now:")
    print("-" * 80)
    print(f"{'Date':<22} {'Phase':>7} {'DB CLN':>8} {'Precise':>8} {'Days':>6} {'Note':<15}")
    print("-" * 80)
    
    for delta in range(-20, 21):
        test_jd = current_jd + delta
        
        phase = compute_moon_phase(test_jd)
        avg_cln, avg_days = jd_to_cln_average(test_jd)
        
        precise_result = jd_to_cln_precise(test_jd)
        if precise_result:
            prec_cln, prec_days = precise_result
        else:
            prec_cln = "N/A"
            prec_days = 0
        
        # Determine note
        note = ""
        if phase and (phase > 175 and phase < 185):
            note = "FULL MOON"
        elif phase and (phase < 5 or phase > 355):
            note = "NEW MOON"
        
        # Check for discrepancy
        if isinstance(prec_cln, int) and prec_cln != avg_cln:
            note = f"*DIFF* ({avg_cln - prec_cln:+d})"
        
        date_str = jd_to_datetime(test_jd)[:16]
        phase_str = f"{phase:.1f}°" if phase else "N/A"
        days_str = f"{prec_days:.1f}" if isinstance(prec_cln, int) else ""
        
        print(f"{date_str:<22} {phase_str:>7} {avg_cln:>8} {prec_cln:>8} {days_str:>6} {note:<15}")
    
    print()
    print("=" * 80)
    print("Key:")
    print("  Phase: Moon-Sun ecliptic longitude (180° = Full Moon)")
    print("  DB CLN: Database method (may be ~1 day off near Full Moon)")
    print("  Precise: Display method (changes exactly at Full Moon)")
    print("  Days: Days since last Full Moon (from Precise method)")
    print("  *DIFF*: Discrepancy between DB and Precise (expected near Full Moon)")
    print("=" * 80)


if __name__ == '__main__':
    main()
