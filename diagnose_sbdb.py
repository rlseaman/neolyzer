#!/usr/bin/env python3
"""
SBDB Diagnostic Tool - Analyze NEA.txt vs JPL SBDB matching

Usage:
    cd neolyzer
    ./venv/bin/python3 diagnose_sbdb.py [--fetch] [--analyze]

Options:
    --fetch    Fetch fresh data from JPL SBDB (takes ~30 seconds)
    --analyze  Analyze existing data (default if cache exists)
    
If no options given and cache exists, will analyze. Otherwise will fetch.
"""

import sys
import os
import json
import sqlite3
import argparse
from pathlib import Path
from collections import defaultdict

# Add src to path for designation_utils
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def load_nea_designations(nea_file):
    """Load all designations from NEA.txt"""
    designations = {}
    with open(nea_file, 'r', encoding='latin-1') as f:
        for line_num, line in enumerate(f, 1):
            if len(line) >= 7:
                packed = line[0:7].strip()
                if packed:
                    designations[packed] = line_num
    return designations

def fetch_sbdb_data(output_file):
    """Fetch fresh SBDB data from JPL"""
    import requests
    
    print("Fetching SBDB data from JPL...", flush=True)
    
    base_url = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"
    params = {
        'fields': 'spkid,pdes,name,full_name,class,moid,H',
        'sb-class': 'IEO,ATE,APO,AMO,MCA'
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=120)
        response.raise_for_status()
        data = response.json()
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved to {output_file}", flush=True)
        return data
        
    except Exception as e:
        print(f"Error fetching SBDB data: {e}")
        return None

def load_sbdb_cache(cache_file):
    """Load SBDB data from cache file"""
    with open(cache_file, 'r') as f:
        return json.load(f)

def unpack_designation(packed):
    """Convert packed MPC designation to unpacked format"""
    if not packed:
        return packed
    
    # Numbered asteroids (all digits or leading spaces + digits)
    packed = packed.strip()
    if packed.isdigit():
        return packed.lstrip('0') or '0'
    
    # Check for packed number format (A0000, B0000, etc.)
    if len(packed) >= 5 and packed[0].isalpha() and packed[1:].isdigit():
        first_char = packed[0]
        number_part = int(packed[1:])
        
        char_values = {
            'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15,
            'G': 16, 'H': 17, 'I': 18, 'J': 19, 'K': 20, 'L': 21,
            'M': 22, 'N': 23, 'O': 24, 'P': 25, 'Q': 26, 'R': 27,
            'S': 28, 'T': 29, 'U': 30, 'V': 31, 'W': 32, 'X': 33,
            'Y': 34, 'Z': 35, 'a': 36, 'b': 37, 'c': 38, 'd': 39,
            'e': 40, 'f': 41, 'g': 42, 'h': 43, 'i': 44, 'j': 45,
            'k': 46, 'l': 47, 'm': 48, 'n': 49, 'o': 50, 'p': 51,
            'q': 52, 'r': 53, 's': 54, 't': 55, 'u': 56, 'v': 57,
            'w': 58, 'x': 59, 'y': 60, 'z': 61
        }
        
        if first_char in char_values:
            full_number = char_values[first_char] * 10000 + number_part
            return str(full_number)
    
    # Provisional designations (e.g., K24A00A -> 2024 AA)
    if len(packed) == 7:
        century_codes = {'I': '18', 'J': '19', 'K': '20', 'L': '21'}
        half_month_codes = 'ABCDEFGHJKLMNOPQRSTUVWXY'
        second_letter_codes = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        
        century_char = packed[0]
        year_digits = packed[1:3]
        half_month = packed[3]
        cycle_code = packed[4:6]
        second_letter = packed[6]
        
        if century_char in century_codes and half_month in half_month_codes:
            year = century_codes[century_char] + year_digits
            
            # Decode cycle number
            if cycle_code == '00':
                cycle_num = 0
            else:
                cycle_num = 0
                if cycle_code[0] in second_letter_codes:
                    cycle_num += second_letter_codes.index(cycle_code[0]) * 62
                if cycle_code[1] in second_letter_codes:
                    cycle_num += second_letter_codes.index(cycle_code[1])
            
            if cycle_num == 0:
                return f"{year} {half_month}{second_letter}"
            else:
                return f"{year} {half_month}{second_letter}{cycle_num}"
    
    # If we can't unpack, return as-is
    return packed

def analyze_matching(nea_designations, sbdb_data):
    """Analyze matching between NEA.txt and SBDB"""
    import re
    
    fields = sbdb_data['fields']
    rows = sbdb_data['data']
    
    # Find column indices
    try:
        pdes_idx = fields.index('pdes')
        spkid_idx = fields.index('spkid')
        class_idx = fields.index('class') if 'class' in fields else None
        moid_idx = fields.index('moid') if 'moid' in fields else None
        h_idx = fields.index('H') if 'H' in fields else None
        fullname_idx = fields.index('full_name') if 'full_name' in fields else None
        name_idx = fields.index('name') if 'name' in fields else None
    except ValueError as e:
        print(f"Missing required field: {e}")
        return
    
    print(f"\nSBDB Query Results:")
    print(f"  Fields: {fields}")
    print(f"  Total rows: {len(rows)}")
    
    # Build SBDB lookup by unpacked designation AND alternate designations
    sbdb_lookup = {}
    sbdb_by_class = defaultdict(list)
    
    for row in rows:
        pdes = str(row[pdes_idx]).strip() if row[pdes_idx] else None
        if pdes:
            sbdb_lookup[pdes] = row
            if class_idx is not None:
                sbdb_by_class[row[class_idx]].append(pdes)
        
        # Extract alternate designations from name field
        if name_idx is not None and row[name_idx]:
            name = str(row[name_idx])
            # Extract provisional designation from parentheses: "433 Eros (1898 DQ)"
            prov_match = re.search(r'\((\d{4}\s+[A-Z]{2}\d*)\)', name)
            if prov_match:
                prov_des = prov_match.group(1)
                if prov_des not in sbdb_lookup:
                    sbdb_lookup[prov_des] = row
            # Also check if name itself looks like a designation
            name_parts = name.split('(')[0].strip()
            if re.match(r'\d{4}\s+[A-Z]{2}', name_parts) and name_parts not in sbdb_lookup:
                sbdb_lookup[name_parts] = row
    
    print(f"\n  SBDB objects by class:")
    for cls, objs in sorted(sbdb_by_class.items()):
        print(f"    {cls}: {len(objs)}")
    
    # Show sample SBDB designations for comparison
    print(f"\n  Sample SBDB designations (first 20):")
    sample_pdes = sorted(sbdb_lookup.keys())[:20]
    for pdes in sample_pdes:
        print(f"    '{pdes}'")
    
    # Try to match NEA designations
    matched = []
    unmatched_nea = []
    
    # Debug: collect some examples for analysis
    debug_examples = []
    
    for packed, line_num in nea_designations.items():
        unpacked = unpack_designation(packed)
        
        if unpacked in sbdb_lookup:
            matched.append((packed, unpacked, sbdb_lookup[unpacked]))
        else:
            # Try alternate forms
            found = False
            
            # Try without spaces
            unpacked_nospace = unpacked.replace(' ', '')
            for sbdb_pdes in sbdb_lookup:
                if sbdb_pdes.replace(' ', '') == unpacked_nospace:
                    matched.append((packed, unpacked, sbdb_lookup[sbdb_pdes]))
                    found = True
                    break
            
            if not found:
                unmatched_nea.append((packed, unpacked, line_num))
                # Collect debug examples
                if len(debug_examples) < 20:
                    debug_examples.append((packed, unpacked))
    
    # Find SBDB objects not in NEA.txt
    nea_unpacked = {unpack_designation(p) for p in nea_designations}
    unmatched_sbdb = []
    for pdes, row in sbdb_lookup.items():
        if pdes not in nea_unpacked:
            # Check without spaces too
            found = False
            for nea_u in nea_unpacked:
                if nea_u.replace(' ', '') == pdes.replace(' ', ''):
                    found = True
                    break
            if not found:
                unmatched_sbdb.append((pdes, row))
    
    print(f"\n" + "="*70)
    print("MATCHING SUMMARY")
    print("="*70)
    print(f"NEA.txt entries:           {len(nea_designations)}")
    print(f"SBDB entries:              {len(sbdb_lookup)}")
    print(f"Matched:                   {len(matched)}")
    print(f"In NEA.txt but not SBDB:   {len(unmatched_nea)}")
    print(f"In SBDB but not NEA.txt:   {len(unmatched_sbdb)}")
    
    # Show unmatched NEA entries
    if unmatched_nea:
        print(f"\n" + "="*70)
        print(f"NEA.txt ENTRIES NOT IN SBDB ({len(unmatched_nea)})")
        print("="*70)
        print(f"{'Packed':<10} {'Unpacked':<20} {'Line#':<8}")
        print("-"*70)
        
        # Show first 50
        for packed, unpacked, line_num in sorted(unmatched_nea)[:50]:
            print(f"{packed:<10} {unpacked:<20} {line_num:<8}")
        
        if len(unmatched_nea) > 50:
            print(f"... and {len(unmatched_nea) - 50} more")
    
    # Show unmatched SBDB entries
    if unmatched_sbdb:
        print(f"\n" + "="*70)
        print(f"SBDB ENTRIES NOT IN NEA.txt ({len(unmatched_sbdb)})")
        print("="*70)
        print(f"{'PDES':<20} {'Class':<6} {'MOID':<10} {'H':<8}")
        print("-"*70)
        
        # Show first 50
        for pdes, row in sorted(unmatched_sbdb)[:50]:
            cls = row[class_idx] if class_idx is not None else 'N/A'
            if moid_idx is not None and row[moid_idx] is not None:
                try:
                    moid = f"{float(row[moid_idx]):.4f}"
                except (ValueError, TypeError):
                    moid = str(row[moid_idx])
            else:
                moid = 'N/A'
            if h_idx is not None and row[h_idx] is not None:
                try:
                    h = f"{float(row[h_idx]):.1f}"
                except (ValueError, TypeError):
                    h = str(row[h_idx])
            else:
                h = 'N/A'
            print(f"{pdes:<20} {cls:<6} {moid:<10} {h:<8}")
        
        if len(unmatched_sbdb) > 50:
            print(f"... and {len(unmatched_sbdb) - 50} more")
    
    # Analyze patterns in unmatched NEA entries
    if unmatched_nea:
        print(f"\n" + "="*70)
        print("ANALYSIS OF UNMATCHED NEA.txt ENTRIES")
        print("="*70)
        
        # Show what we're searching for vs what exists
        print("\nSample unmatched - showing our unpacked vs SBDB samples:")
        for packed, unpacked in debug_examples[:10]:
            # Find similar SBDB entries
            similar = []
            unpacked_base = unpacked.split()[0] if ' ' in unpacked else unpacked[:4]
            for sbdb_pdes in sbdb_lookup:
                if sbdb_pdes.startswith(unpacked_base):
                    similar.append(sbdb_pdes)
                    if len(similar) >= 3:
                        break
            print(f"  {packed} -> '{unpacked}'")
            if similar:
                print(f"    Similar in SBDB: {similar}")
            else:
                print(f"    No similar entries found starting with '{unpacked_base}'")
        
        # Group by year
        by_year = defaultdict(list)
        for packed, unpacked, line_num in unmatched_nea:
            if len(unpacked) >= 4 and unpacked[:4].isdigit():
                year = unpacked[:4]
            elif packed.startswith('K'):
                year = '20' + packed[1:3]
            elif packed.startswith('J'):
                year = '19' + packed[1:3]
            else:
                year = 'Unknown'
            by_year[year].append(packed)
        
        print("\nUnmatched by discovery year:")
        for year in sorted(by_year.keys()):
            count = len(by_year[year])
            if count > 0:
                print(f"  {year}: {count}")
    
    return {
        'matched': matched,
        'unmatched_nea': unmatched_nea,
        'unmatched_sbdb': unmatched_sbdb
    }

def main():
    parser = argparse.ArgumentParser(description='SBDB Diagnostic Tool')
    parser.add_argument('--fetch', action='store_true', help='Fetch fresh data from JPL SBDB')
    parser.add_argument('--analyze', action='store_true', help='Analyze existing data')
    args = parser.parse_args()
    
    script_dir = Path(__file__).parent
    data_dir = script_dir / 'data'
    diag_dir = script_dir / 'diagnostics'
    
    nea_file = data_dir / 'NEA.txt'
    cache_file = data_dir / 'sbdb_moid_cache.json'
    diag_cache = diag_dir / 'sbdb_moid_cache.json'
    
    # Check for NEA.txt
    if not nea_file.exists():
        print(f"ERROR: {nea_file} not found")
        print("Run ./install.sh first to download NEA.txt")
        return 1
    
    # Load NEA designations
    print(f"Loading {nea_file}...", flush=True)
    nea_designations = load_nea_designations(nea_file)
    print(f"Found {len(nea_designations)} designations in NEA.txt")
    
    # Determine whether to fetch or use cache
    sbdb_data = None
    
    if args.fetch:
        diag_dir.mkdir(exist_ok=True)
        sbdb_data = fetch_sbdb_data(str(diag_cache))
        if sbdb_data:
            # Copy to data dir
            import shutil
            shutil.copy2(str(diag_cache), str(cache_file))
            print(f"Copied to {cache_file}")
    
    if sbdb_data is None:
        # Try to load from cache
        if cache_file.exists():
            print(f"\nLoading SBDB cache from {cache_file}...", flush=True)
            sbdb_data = load_sbdb_cache(cache_file)
        elif diag_cache.exists():
            print(f"\nLoading SBDB cache from {diag_cache}...", flush=True)
            sbdb_data = load_sbdb_cache(diag_cache)
        else:
            print(f"\nNo SBDB cache found. Use --fetch to download.")
            return 1
    
    # Analyze
    results = analyze_matching(nea_designations, sbdb_data)
    
    # Save detailed results
    output_file = diag_dir / 'sbdb_analysis.txt'
    diag_dir.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write("SBDB vs NEA.txt Analysis\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"NEA.txt entries: {len(nea_designations)}\n")
        f.write(f"Matched: {len(results['matched'])}\n")
        f.write(f"Unmatched NEA: {len(results['unmatched_nea'])}\n")
        f.write(f"Unmatched SBDB: {len(results['unmatched_sbdb'])}\n\n")
        
        f.write("UNMATCHED NEA.txt ENTRIES (full list)\n")
        f.write("-"*70 + "\n")
        for packed, unpacked, line_num in sorted(results['unmatched_nea']):
            f.write(f"{packed}\t{unpacked}\t{line_num}\n")
        
        f.write("\n\nUNMATCHED SBDB ENTRIES (full list)\n")
        f.write("-"*70 + "\n")
        for pdes, row in sorted(results['unmatched_sbdb']):
            f.write(f"{pdes}\t{row}\n")
    
    print(f"\nDetailed results saved to {output_file}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
