#!/usr/bin/env python3
"""
Diagnostic script to identify NEA.txt entries missing from the database.

Usage:
    cd neolyzer
    source venv/bin/activate
    python3 diagnose_missing.py
"""

import sqlite3
from pathlib import Path

def parse_designation(line):
    """Extract designation from MPC format line"""
    if len(line) >= 7:
        return line[0:7].strip()
    return None

def analyze_line(line):
    """Analyze why a line might fail to parse"""
    issues = []
    
    stripped = line.rstrip('\n\r')
    if len(stripped) < 160:
        issues.append(f"Line too short: {len(stripped)} chars (need >= 160)")
        return issues
    
    try:
        H = line[8:13].strip()
        if not H:
            issues.append("Missing H magnitude (blank)")
        elif H:
            float(H)
    except ValueError:
        issues.append(f"Invalid H magnitude: '{line[8:13]}'")
    
    try:
        G = line[14:19].strip()
        if G:
            float(G)
    except ValueError:
        issues.append(f"Invalid G slope: '{line[14:19]}'")
    
    try:
        epoch = line[20:25].strip()
        if not epoch or len(epoch) < 5:
            issues.append(f"Invalid epoch: '{epoch}'")
    except:
        issues.append("Can't read epoch")
    
    for name, start, end in [
        ('mean anomaly', 26, 35),
        ('arg_peri', 37, 46),
        ('node', 48, 57),
        ('inclination', 59, 68),
        ('eccentricity', 70, 79),
        ('semi-major axis', 91, 103),
    ]:
        try:
            val = line[start:end].strip()
            if val:
                float(val)
        except ValueError:
            issues.append(f"Invalid {name}: '{line[start:end]}'")
    
    return issues

def main():
    script_dir = Path(__file__).parent
    data_dir = script_dir / 'data'
    nea_file = data_dir / 'NEA.txt'
    db_file = data_dir / 'asteroids.db'
    
    if not nea_file.exists():
        print(f"ERROR: {nea_file} not found")
        return 1
    
    if not db_file.exists():
        print(f"ERROR: {db_file} not found")
        return 1
    
    # Read all designations from NEA.txt
    print(f"Reading {nea_file}...", flush=True)
    nea_designations = {}
    line_data = {}
    
    with open(nea_file, 'r', encoding='latin-1') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 10000 == 0:
                print(f"  ...line {line_num}", flush=True)
            des = parse_designation(line)
            if des:
                nea_designations[des] = line_num
                line_data[des] = line
    
    print(f"Found {len(nea_designations)} designations in NEA.txt", flush=True)
    
    # Query database directly with sqlite3 (faster, no ORM overhead)
    print(f"\nQuerying database {db_file}...", flush=True)
    
    try:
        conn = sqlite3.connect(str(db_file))
        cursor = conn.cursor()
        
        # Get count first
        cursor.execute("SELECT COUNT(*) FROM asteroids")
        count = cursor.fetchone()[0]
        print(f"Database has {count} records", flush=True)
        
        # Get all designations
        print("Fetching designations...", flush=True)
        cursor.execute("SELECT designation FROM asteroids")
        db_designations = {row[0] for row in cursor.fetchall()}
        
        conn.close()
        print(f"Found {len(db_designations)} designations in database", flush=True)
        
    except Exception as e:
        print(f"Database error: {e}")
        return 1
    
    # Find missing
    missing = set(nea_designations.keys()) - db_designations
    extra = db_designations - set(nea_designations.keys())
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"NEA.txt entries:     {len(nea_designations)}")
    print(f"Database entries:    {len(db_designations)}")
    print(f"Missing from DB:     {len(missing)}")
    print(f"Extra in DB:         {len(extra)}")
    
    if missing:
        print(f"\n{'='*60}")
        print(f"MISSING ENTRIES ({len(missing)})")
        print(f"{'='*60}")
        
        for des in sorted(missing):
            line_num = nea_designations[des]
            line = line_data[des]
            issues = analyze_line(line)
            
            print(f"\n{des} (line {line_num}):")
            if issues:
                for issue in issues:
                    print(f"  - {issue}")
            else:
                print(f"  - No obvious parse issues detected")
                print(f"  - Line length: {len(line.rstrip())}")
                print(f"  - Raw: {repr(line[:100])}...")
    
    if extra:
        print(f"\n{'='*60}")
        print(f"EXTRA IN DATABASE ({len(extra)})")
        print(f"{'='*60}")
        for des in sorted(extra):
            print(f"  {des}")
    
    print("\nDone.")
    return 0

if __name__ == '__main__':
    exit(main())
