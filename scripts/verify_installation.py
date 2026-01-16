#!/usr/bin/env python3
"""
Verify NEO Visualizer Installation
Checks all components are properly installed and configured
"""

import sys
import os
import platform

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def check_mark(success):
    return "✓" if success else "✗"


def check_python_version():
    """Check Python version"""
    version = sys.version_info
    ok = version.major >= 3 and version.minor >= 10
    print(f"  {check_mark(ok)} Python {version.major}.{version.minor}.{version.micro}", end="")
    if not ok:
        print(" (need 3.10+)")
    else:
        print()
    return ok


def check_platform():
    """Show platform info"""
    system = platform.system()
    machine = platform.machine()
    
    if system == "Darwin":
        os_name = "macOS"
        if machine == "arm64":
            os_name += " (Apple Silicon)"
        else:
            os_name += " (Intel)"
    elif system == "Linux":
        # Try to get distribution
        try:
            with open('/etc/os-release') as f:
                for line in f:
                    if line.startswith('PRETTY_NAME='):
                        os_name = line.split('=')[1].strip().strip('"')
                        break
                else:
                    os_name = "Linux"
        except:
            os_name = "Linux"
        
        # Check for Raspberry Pi
        try:
            with open('/proc/device-tree/model') as f:
                model = f.read()
                if 'raspberry' in model.lower():
                    os_name += " (Raspberry Pi)"
        except:
            pass
    else:
        os_name = f"{system} {machine}"
    
    print(f"  ℹ Platform: {os_name} ({machine})")
    return True


def check_import(module_name, display_name=None):
    """Check if a module can be imported"""
    if display_name is None:
        display_name = module_name
    try:
        __import__(module_name)
        print(f"  {check_mark(True)} {display_name}")
        return True
    except ImportError as e:
        print(f"  {check_mark(False)} {display_name}: {e}")
        return False


def check_pyqt6():
    """Check PyQt6 with more detail"""
    try:
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import QT_VERSION_STR
        print(f"  {check_mark(True)} PyQt6 (Qt {QT_VERSION_STR})")
        return True
    except ImportError as e:
        print(f"  {check_mark(False)} PyQt6: {e}")
        return False


def check_database():
    """Check database exists and has data"""
    db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'neo_orbits.db')
    if os.path.exists(db_path):
        size_mb = os.path.getsize(db_path) / (1024 * 1024)
        print(f"  {check_mark(True)} Database: {size_mb:.1f} MB")
        
        # Check record count
        try:
            import sqlite3
            conn = sqlite3.connect(db_path)
            cursor = conn.execute("SELECT COUNT(*) FROM asteroids")
            count = cursor.fetchone()[0]
            conn.close()
            print(f"  {check_mark(True)} NEO records: {count:,}")
            return count > 0
        except Exception as e:
            print(f"  {check_mark(False)} Database query failed: {e}")
            return False
    else:
        print(f"  {check_mark(False)} Database not found")
        print(f"      Run: python scripts/setup_database.py")
        return False


def check_cache():
    """Check position cache exists"""
    cache_path = os.path.join(os.path.dirname(__file__), '..', 'cache', 'positions.h5')
    if os.path.exists(cache_path):
        size_mb = os.path.getsize(cache_path) / (1024 * 1024)
        print(f"  {check_mark(True)} Position cache: {size_mb:.1f} MB")
        return True
    else:
        print(f"  {check_mark(False)} Position cache not found (optional)")
        print(f"      Positions will be computed on-the-fly")
        return True  # Cache is optional


def check_ephemeris():
    """Check JPL ephemeris file"""
    try:
        from skyfield.api import load
        eph = load('de421.bsp')
        print(f"  {check_mark(True)} JPL ephemeris (de421.bsp)")
        return True
    except Exception as e:
        print(f"  {check_mark(False)} JPL ephemeris: {e}")
        return False


def check_display():
    """Check if display is available (for GUI)"""
    if platform.system() == "Darwin":
        # macOS always has display
        print(f"  {check_mark(True)} Display available (macOS)")
        return True
    
    display = os.environ.get('DISPLAY')
    if display:
        print(f"  {check_mark(True)} Display available ({display})")
        return True
    
    # Check for Wayland
    wayland = os.environ.get('WAYLAND_DISPLAY')
    if wayland:
        print(f"  {check_mark(True)} Display available (Wayland: {wayland})")
        return True
    
    print(f"  {check_mark(False)} No display found (set DISPLAY or use X forwarding)")
    return False


def check_config():
    """Check installation config file"""
    config_path = os.path.join(os.path.dirname(__file__), '..', '.neo_config')
    if os.path.exists(config_path):
        print(f"  {check_mark(True)} Installation config found")
        with open(config_path) as f:
            for line in f:
                if line.startswith('INSTALL_DATE='):
                    date = line.split('=')[1].strip().strip('"')
                    print(f"      Installed: {date}")
                elif line.startswith('NEO_VERSION='):
                    version = line.split('=')[1].strip().strip('"')
                    print(f"      Version: {version}")
        return True
    else:
        print(f"  {check_mark(False)} No installation config (.neo_config)")
        return False


def main():
    print("=" * 60)
    print("NEO Visualizer - Installation Verification")
    print("=" * 60)
    print()
    
    all_ok = True
    
    # Platform info
    print("Platform:")
    check_platform()
    all_ok &= check_python_version()
    print()
    
    # Core packages
    print("Core Packages:")
    all_ok &= check_import('numpy')
    all_ok &= check_import('pandas')
    all_ok &= check_import('sqlalchemy', 'SQLAlchemy')
    print()
    
    # Astronomy packages
    print("Astronomy Packages:")
    all_ok &= check_import('skyfield')
    all_ok &= check_import('jplephem')
    check_import('astropy', 'astropy (optional)')  # Optional
    print()
    
    # Visualization
    print("Visualization Packages:")
    all_ok &= check_import('matplotlib')
    all_ok &= check_pyqt6()
    print()
    
    # HDF5 (optional for cache)
    print("Cache Support:")
    check_import('h5py', 'h5py (for position cache)')
    print()
    
    # Data files
    print("Data Files:")
    check_config()
    all_ok &= check_database()
    check_cache()
    check_ephemeris()
    print()
    
    # Display
    print("Display:")
    display_ok = check_display()
    print()
    
    # Summary
    print("=" * 60)
    if all_ok:
        print("✓ All checks passed!")
        if display_ok:
            print()
            print("Ready to run:")
            print("  ./run_visualizer.sh")
            print("  # or")
            print("  ./venv/bin/python src/visualizer.py")
        else:
            print()
            print("⚠ No display available. For remote Linux systems:")
            print("  - Use X11 forwarding: ssh -X user@host")
            print("  - Or set DISPLAY: export DISPLAY=:0")
    else:
        print("✗ Some checks failed. Please review errors above.")
        print()
        print("To reinstall:")
        print("  ./install.sh")
    print("=" * 60)
    
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
