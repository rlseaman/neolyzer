"""Shared test configuration and fixtures."""

import sys
from pathlib import Path

# Add src/ to path so tests can import project modules directly
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
