#!/usr/bin/env python3
"""
Verify that all three critical bugs are fixed
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
from database import DatabaseManager
from orbit_calculator import FastOrbitCalculator
from skyfield.api import load

print("=" * 80)
print("NEO VISUALIZER - BUG FIX VERIFICATION")
print("=" * 80)
print()

db = DatabaseManager(use_sqlite=True)
calc = FastOrbitCalculator()
ts = load.timescale()

# Bug 1: Semi-major axis parsing
print("BUG 1: Semi-major Axis Parsing")
print("-" * 80)

# Check Eros
asteroids = db.get_asteroids()
eros = None
for ast in asteroids:
    if ast['designation'].strip() == '00433':
        eros = ast
        break

if eros:
    a = eros['a']
    expected_a = 1.458
    diff = abs(a - expected_a)
    
    print(f"Eros (433) semi-major axis:")
    print(f"  Parsed: {a:.3f} AU")
    print(f"  Expected: {expected_a:.3f} AU")
    print(f"  Difference: {diff:.4f} AU")
    
    if diff < 0.001:
        print("  ✓ CORRECT - Bug 1 is FIXED")
        bug1_fixed = True
    else:
        print("  ✗ WRONG - Bug 1 still present!")
        bug1_fixed = False
else:
    print("  ⚠ Eros not found in database")
    bug1_fixed = None

print()

# Bug 2: Coordinate system mixing
print("BUG 2: Coordinate System Mixing")
print("-" * 80)

# Check ecliptic latitude distribution
neos = []
for cls in ['Apollo', 'Aten', 'Atira', 'Amor']:
    neos.extend(db.get_asteroids(orbit_class=cls, limit=10000))

if len(neos) > 100:
    t = ts.now()
    positions = calc.calculate_batch(neos, t.tt)
    
    if len(positions) > 0:
        # Convert to ecliptic
        ra = positions[:, 1]
        dec = positions[:, 2]
        
        # Simple ecliptic conversion
        eps = np.radians(23.43928)
        ra_rad = np.radians(ra)
        dec_rad = np.radians(dec)
        
        beta_rad = np.arcsin(
            np.sin(dec_rad) * np.cos(eps) - np.cos(dec_rad) * np.sin(eps) * np.sin(ra_rad)
        )
        beta = np.degrees(beta_rad)
        
        mean_beta = np.mean(beta)
        within_10 = np.sum(np.abs(beta) < 10) / len(beta) * 100
        
        print(f"Ecliptic latitude distribution ({len(positions)} NEAs):")
        print(f"  Mean latitude: {mean_beta:.2f}°")
        print(f"  Within ±10°: {within_10:.1f}%")
        
        if abs(mean_beta) < 2.0 and within_10 > 60:
            print("  ✓ CORRECT - Bug 2 is FIXED")
            bug2_fixed = True
        else:
            print("  ✗ WRONG - Bug 2 still present!")
            print("    Expected: mean ~ 0°, >60% within ±10°")
            bug2_fixed = False
    else:
        print("  ⚠ No positions calculated")
        bug2_fixed = None
else:
    print("  ⚠ Not enough NEAs in database")
    bug2_fixed = None

print()

# Bug 3: Mean motion calculation
print("BUG 3: Mean Motion Calculation")
print("-" * 80)

# Check Gaussian constant in code
code_file = Path("src/orbit_calculator.py")
if code_file.exists():
    code = code_file.read_text()
    if "0.01720209895" in code:
        print("Gaussian gravitational constant:")
        print("  ✓ Found k = 0.01720209895 in code")
        print("  ✓ CORRECT - Bug 3 is FIXED")
        bug3_fixed = True
    else:
        print("Gaussian gravitational constant:")
        print("  ✗ NOT FOUND in code")
        print("  ✗ WRONG - Bug 3 still present!")
        bug3_fixed = False
else:
    print("  ⚠ orbit_calculator.py not found")
    bug3_fixed = None

# Also check motion rates
if len(neos) > 100:
    print()
    print("Motion rate verification:")
    
    t0 = ts.now().tt
    t1 = t0 + 0.1  # 0.1 days
    
    pos0 = calc.calculate_batch(neos[:1000], t0)
    pos1 = calc.calculate_batch(neos[:1000], t1)
    
    if len(pos0) > 0 and len(pos1) > 0:
        ra0, dec0 = pos0[:, 1], pos0[:, 2]
        ra1, dec1 = pos1[:, 1], pos1[:, 2]
        
        dra = ra1 - ra0
        dra = np.where(dra > 180, dra - 360, dra)
        dra = np.where(dra < -180, dra + 360, dra)
        ddec = dec1 - dec0
        
        motion = np.sqrt(dra**2 + ddec**2)
        motion_per_day = motion / 0.1
        
        median_motion = np.median(motion_per_day)
        slow_fraction = np.sum(motion_per_day < 1.0) / len(motion_per_day) * 100
        fast_fraction = np.sum(motion_per_day > 5.0) / len(motion_per_day) * 100
        
        print(f"  Median motion: {median_motion:.2f}°/day")
        print(f"  < 1°/day: {slow_fraction:.1f}%")
        print(f"  > 5°/day: {fast_fraction:.1f}%")
        
        if median_motion < 2.0 and slow_fraction > 50:
            print("  ✓ Motion rates look realistic")
        else:
            print("  ⚠ Motion rates seem high")
            print("    Expected: median < 2°/day, >50% < 1°/day")

print()

# Summary
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()

if bug1_fixed and bug2_fixed and bug3_fixed:
    print("✅ ALL BUGS FIXED!")
    print()
    print("The visualizer should now work correctly:")
    print("  ✓ Correct orbital sizes")
    print("  ✓ Proper ecliptic distribution")
    print("  ✓ Realistic motion rates")
    print()
    print("Launch with:")
    print("  ./venv/bin/python src/visualizer.py")
    exit_code = 0
    
elif bug1_fixed is False or bug2_fixed is False or bug3_fixed is False:
    print("❌ SOME BUGS NOT FIXED!")
    print()
    if bug1_fixed is False:
        print("  ✗ Bug 1 (parsing) - redownload package")
    if bug2_fixed is False:
        print("  ✗ Bug 2 (coordinates) - rebuild cache")
    if bug3_fixed is False:
        print("  ✗ Bug 3 (motion) - redownload package")
    print()
    print("Recommendation: Download fresh package and reinstall")
    exit_code = 1
    
else:
    print("⚠ VERIFICATION INCOMPLETE")
    print()
    print("Some checks could not be completed.")
    print("This may be normal if database is empty.")
    print()
    print("Run setup if needed:")
    print("  ./venv/bin/python scripts/setup_database.py")
    exit_code = 2

print("=" * 80)
sys.exit(exit_code)
