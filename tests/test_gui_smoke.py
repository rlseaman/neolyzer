"""Headless GUI smoke test (plan item 4.2).

Imports the 18k-line neolyzer module and instantiates the matplotlib
canvas offscreen. This catches import-time regressions (bad imports,
syntax/indentation damage, module-level API misuse) and first-render
crashes that the pure-math test suite can't see. No database, cache,
or ephemeris download is required.
"""

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

pytest.importorskip("PyQt6.QtWidgets")


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def test_neolyzer_imports():
    import neolyzer
    assert hasattr(neolyzer, "__version__")
    assert hasattr(neolyzer, "SkyMapCanvas")
    assert hasattr(neolyzer, "NEOVisualizer")


def test_skymap_canvas_constructs_and_draws(qapp):
    import neolyzer
    canvas = neolyzer.SkyMapCanvas()
    # Render whatever the empty canvas draws — exercises the figure,
    # projection setup, and matplotlib backend end to end
    canvas.fig.canvas.draw()
    assert canvas.fig is not None


def test_coordinate_transformer_roundtrip():
    """CoordinateTransformer lives inside neolyzer.py — exercise it via
    the real import (not a copy) now that the module imports headlessly."""
    import numpy as np
    import neolyzer
    ra, dec = 123.456, -21.5
    lon, lat = neolyzer.CoordinateTransformer.equatorial_to_ecliptic(ra, dec)
    ra2, dec2 = neolyzer.CoordinateTransformer.ecliptic_to_equatorial(lon, lat)
    assert abs((ra2 - ra + 180) % 360 - 180) < 1e-9
    assert abs(dec2 - dec) < 1e-9
