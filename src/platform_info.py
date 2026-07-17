"""
Platform description helper — single home for the OS/architecture
detection previously copy-pasted between setup_database.py and
verify_installation.py.
"""

import platform
import sys


def describe_platform():
    """
    Return (os_name, machine, python_version) with a human-readable
    OS name: macOS Intel/Apple Silicon, Linux distro from os-release
    (with Raspberry Pi detection), or the raw platform name.
    """
    system = platform.system()
    machine = platform.machine()
    python_version = (f"{sys.version_info.major}.{sys.version_info.minor}"
                      f".{sys.version_info.micro}")

    if system == "Darwin":
        os_name = "macOS"
        os_name += " (Apple Silicon)" if machine == "arm64" else " (Intel)"
    elif system == "Linux":
        try:
            with open('/etc/os-release') as f:
                for line in f:
                    if line.startswith('PRETTY_NAME='):
                        os_name = line.split('=')[1].strip().strip('"')
                        break
                else:
                    os_name = "Linux"
        except OSError:
            os_name = "Linux"

        # Raspberry Pi detection
        try:
            with open('/proc/device-tree/model') as f:
                if 'raspberry' in f.read().lower():
                    os_name += " (Raspberry Pi)"
        except OSError:
            pass
    else:
        os_name = system

    return os_name, machine, python_version
