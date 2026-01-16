"""
NEOlyzer v3.02 - Near-Earth Object Visualization and Analysis

FEATURES:
- Horizontal control layout (compact, filters on top, time/animation below)
- Smooth animation (on-the-fly calculation, no interpolation artifacts)
- Negative rate support for backwards playback
- Logging via command-line (--quiet, --verbose, --debug)
- Dual magnitude limits (min and max for both V and H)
- Better rate controls (hours/days/months per second)
- Compressed UI (controls side-by-side where possible)
- Map projections (Rectangular, Hammer, Aitoff, Mollweide)
- Coordinate systems (Equatorial, Ecliptic, Galactic, Opposition)
- Performance optimizations (caching, smart overlays)
- Cross-platform (macOS, Linux, Raspberry Pi)
"""

import sys
import os
import numpy as np
import warnings
from datetime import datetime, timedelta
from typing import Optional, List
import logging
import argparse
import time

# Suppress matplotlib projection warnings (sqrt of negative in geo projections)
warnings.filterwarnings('ignore', message='invalid value encountered in sqrt')
warnings.filterwarnings('ignore', category=RuntimeWarning, module='matplotlib.projections.geo')

# PyQt5/PyQt6 Compatibility Layer
# Try PyQt6 first, fall back to PyQt5
PYQT_VERSION = None
try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QSlider, QSpinBox, QDoubleSpinBox, QComboBox,
        QCheckBox, QGroupBox, QDateTimeEdit, QSplitter, QStatusBar, QGridLayout,
        QScrollArea, QSizePolicy, QDialog, QTextBrowser, QLineEdit,
        QTableWidget, QTableWidgetItem, QMenu, QFrame
    )
    from PyQt6.QtCore import Qt, QDateTime, QTimer, pyqtSignal, QDate, QTime
    from PyQt6.QtGui import QFont, QPixmap, QCursor, QShortcut, QKeySequence
    PYQT_VERSION = 6
    
    # PyQt6 enums are nested (Qt.AlignmentFlag.AlignLeft)
    # Create compatibility aliases
    class QtCompat:
        AlignLeft = Qt.AlignmentFlag.AlignLeft
        AlignRight = Qt.AlignmentFlag.AlignRight
        AlignCenter = Qt.AlignmentFlag.AlignCenter
        AlignTop = Qt.AlignmentFlag.AlignTop
        AlignBottom = Qt.AlignmentFlag.AlignBottom
        AlignVCenter = Qt.AlignmentFlag.AlignVCenter
        AlignHCenter = Qt.AlignmentFlag.AlignHCenter
        Horizontal = Qt.Orientation.Horizontal
        Vertical = Qt.Orientation.Vertical
        PointingHandCursor = Qt.CursorShape.PointingHandCursor
        KeepAspectRatio = Qt.AspectRatioMode.KeepAspectRatio
        SmoothTransformation = Qt.TransformationMode.SmoothTransformation
        TextSelectableByMouse = Qt.TextInteractionFlag.TextSelectableByMouse
        TextSelectableByKeyboard = Qt.TextInteractionFlag.TextSelectableByKeyboard
        LinksAccessibleByMouse = Qt.TextInteractionFlag.LinksAccessibleByMouse
        # Arrow keys
        Key_Left = Qt.Key.Key_Left
        Key_Right = Qt.Key.Key_Right
        Key_Up = Qt.Key.Key_Up
        Key_Down = Qt.Key.Key_Down
        # Time spec for UTC handling
        UTC = Qt.TimeSpec.UTC
        
except ImportError:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QSlider, QSpinBox, QDoubleSpinBox, QComboBox,
        QCheckBox, QGroupBox, QDateTimeEdit, QSplitter, QStatusBar, QGridLayout,
        QScrollArea, QSizePolicy, QDialog, QTextBrowser, QLineEdit,
        QTableWidget, QTableWidgetItem, QMenu, QFrame, QShortcut
    )
    from PyQt5.QtCore import Qt, QDateTime, QTimer, pyqtSignal, QDate, QTime
    from PyQt5.QtGui import QFont, QPixmap, QCursor, QKeySequence
    PYQT_VERSION = 5
    
    # PyQt5 enums are flat (Qt.AlignLeft)
    class QtCompat:
        AlignLeft = Qt.AlignLeft
        AlignRight = Qt.AlignRight
        AlignCenter = Qt.AlignCenter
        AlignTop = Qt.AlignTop
        AlignBottom = Qt.AlignBottom
        AlignVCenter = Qt.AlignVCenter
        AlignHCenter = Qt.AlignHCenter
        Horizontal = Qt.Horizontal
        Vertical = Qt.Vertical
        PointingHandCursor = Qt.PointingHandCursor
        KeepAspectRatio = Qt.KeepAspectRatio
        SmoothTransformation = Qt.SmoothTransformation
        TextSelectableByMouse = Qt.TextSelectableByMouse
        TextSelectableByKeyboard = Qt.TextSelectableByKeyboard
        LinksAccessibleByMouse = Qt.LinksAccessibleByMouse
        # Arrow keys
        Key_Left = Qt.Key_Left
        Key_Right = Qt.Key_Right
        Key_Up = Qt.Key_Up
        Key_Down = Qt.Key_Down
        # Time spec for UTC handling
        UTC = Qt.UTC

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.patches import Circle
import matplotlib.pyplot as plt

from orbit_calculator import FastOrbitCalculator
from database import DatabaseManager
from cache_manager import PositionCache
from skyfield_loader import skyfield_load, ensure_ephemeris
from skyfield.api import utc

logger = logging.getLogger(__name__)


def setup_logging(level):
    """Setup logging with specified level"""
    logging.basicConfig(
        level=level,
        format='%(levelname)s: %(message)s',
        force=True,  # Force reconfiguration for PyQt
        stream=sys.stdout
    )


# Catalina Lunation Number (CLN) constants
# CLN 0 = Full Moon of 1980-01-02 (precise time determined by lunar phase)
# Reference: JD 2444240.0076 is approximately the Full Moon of 1980-01-02 06:11 UTC
CLN_EPOCH_JD = 2444240.0076  # More precise Full Moon time
SYNODIC_MONTH = 29.530588853  # Average lunation period in days

# DE421 ephemeris range (approximate JD values)
# 1899-07-29 = JD 2414986
# 2053-10-09 = JD 2471184
EPHEMERIS_MIN_JD = 2414986
EPHEMERIS_MAX_JD = 2471184


def compute_moon_phase(jd):
    """Compute the lunar phase angle at a given Julian Date.
    
    Returns phase angle in degrees where:
    - 0° = New Moon
    - 90° = First Quarter
    - 180° = Full Moon
    - 270° = Last Quarter
    
    Returns None if date is outside ephemeris range.
    """
    # Check if within ephemeris range
    if jd < EPHEMERIS_MIN_JD or jd > EPHEMERIS_MAX_JD:
        return None
    
    try:
        ts = skyfield_load.timescale()
        t = ts.tt_jd(jd)
        eph = skyfield_load('de421.bsp')
        
        sun = eph['sun']
        moon = eph['moon']
        earth = eph['earth']
        
        # Get positions from Earth
        e = earth.at(t)
        sun_pos = e.observe(sun).apparent()
        moon_pos = e.observe(moon).apparent()
        
        # Get ecliptic longitudes
        _, sun_lon, _ = sun_pos.ecliptic_latlon()
        _, moon_lon, _ = moon_pos.ecliptic_latlon()
        
        # Phase angle is difference in ecliptic longitude
        phase = (moon_lon.degrees - sun_lon.degrees) % 360
        return phase
    except Exception:
        return None


def get_moon_phase_name(jd):
    """Get the moon phase name and color for display.
    
    Returns (phase_name, color) where color is:
    - 'red' for Full Moon
    - 'green' for New Moon  
    - 'black' for other phases (including quarters)
    
    Full Moon, New Moon, 1st Quarter, and 3rd Quarter are shown ±1 day from their precise times.
    Other phases transition at boundaries between these windows.
    """
    # 1 day in phase angle
    ONE_DAY_DEGREES = 360.0 / SYNODIC_MONTH  # ~12.19°
    
    # Try to get precise phase angle
    phase = compute_moon_phase(jd)
    
    if phase is not None:
        # Precise calculation using phase angle
        # New Moon: 0° ± 1 day (wraps around 360°)
        if phase >= 360 - ONE_DAY_DEGREES or phase <= ONE_DAY_DEGREES:
            return 'New Moon', 'green'
        
        # 1st Quarter: 90° ± 1 day
        if 90 - ONE_DAY_DEGREES <= phase <= 90 + ONE_DAY_DEGREES:
            return '1st Quarter Moon', 'black'
        
        # Full Moon: 180° ± 1 day
        if 180 - ONE_DAY_DEGREES <= phase <= 180 + ONE_DAY_DEGREES:
            return 'Full Moon', 'red'
        
        # 3rd Quarter: 270° ± 1 day
        if 270 - ONE_DAY_DEGREES <= phase <= 270 + ONE_DAY_DEGREES:
            return '3rd Quarter Moon', 'black'
        
        # Phases between the major points
        # Waxing Crescent: after New Moon window until 1st Quarter window
        if ONE_DAY_DEGREES < phase < 90 - ONE_DAY_DEGREES:
            return 'Waxing Crescent', 'black'
        
        # Waxing Gibbous: after 1st Quarter window until Full Moon window
        if 90 + ONE_DAY_DEGREES < phase < 180 - ONE_DAY_DEGREES:
            return 'Waxing Gibbous', 'black'
        
        # Waning Gibbous: after Full Moon window until 3rd Quarter window
        if 180 + ONE_DAY_DEGREES < phase < 270 - ONE_DAY_DEGREES:
            return 'Waning Gibbous', 'black'
        
        # Waning Crescent: after 3rd Quarter window until New Moon window
        if 270 + ONE_DAY_DEGREES < phase < 360 - ONE_DAY_DEGREES:
            return 'Waning Crescent', 'black'
        
        # Fallback (shouldn't happen)
        return 'Moon', 'black'
    
    else:
        # Fallback: use CLN days offset for approximate phase
        try:
            cln, days_offset = jd_to_cln_average(jd)
            
            # Full Moon: around day 0 (±1 day)
            if days_offset <= 1.0 or days_offset >= SYNODIC_MONTH - 1.0:
                return 'Full Moon', 'red'
            
            # Waning Gibbous: days ~1-6.4
            if 1.0 < days_offset < 6.38:
                return 'Waning Gibbous', 'black'
            
            # 3rd Quarter: around day 7.38 (±1 day)
            if 6.38 <= days_offset <= 8.38:
                return '3rd Quarter Moon', 'black'
            
            # Waning Crescent: days ~8.4-13.8
            if 8.38 < days_offset < 13.77:
                return 'Waning Crescent', 'black'
            
            # New Moon: around day 14.77 (±1 day)
            if 13.77 <= days_offset <= 15.77:
                return 'New Moon', 'green'
            
            # Waxing Crescent: days ~15.8-21.1
            if 15.77 < days_offset < 21.15:
                return 'Waxing Crescent', 'black'
            
            # 1st Quarter: around day 22.15 (±1 day)
            if 21.15 <= days_offset <= 23.15:
                return '1st Quarter Moon', 'black'
            
            # Waxing Gibbous: days ~23.2-28.5
            if 23.15 < days_offset < SYNODIC_MONTH - 1.0:
                return 'Waxing Gibbous', 'black'
            
            return 'Moon', 'black'
        except Exception:
            return 'Moon', 'black'


def find_last_full_moon(jd):
    """Find the Julian Date of the most recent Full Moon before or at the given JD.
    
    Uses iterative refinement to find the precise Full Moon time.
    Returns None if date is outside ephemeris range.
    """
    phase = compute_moon_phase(jd)
    if phase is None:
        return None
    
    # Phase 180 = Full Moon. If phase < 180, we're before the halfway point
    # If phase > 180, we're past it and heading toward new moon
    if phase < 180:
        # Days since last Full Moon ≈ (phase / 360) * synodic_month + half_month
        days_since_full = (phase / 360) * SYNODIC_MONTH + SYNODIC_MONTH / 2
    else:
        # Days since last Full Moon ≈ ((phase - 180) / 360) * synodic_month
        days_since_full = ((phase - 180) / 360) * SYNODIC_MONTH
    
    # Estimate of last Full Moon
    estimated_full_jd = jd - days_since_full
    
    # Refine by checking the phase at the estimate
    # Full Moon is when phase ≈ 180
    for _ in range(5):  # A few iterations should suffice
        est_phase = compute_moon_phase(estimated_full_jd)
        if est_phase is None:
            break
        # Adjust: if phase < 180, we're too early; if > 180, we're too late
        phase_error = est_phase - 180
        if abs(phase_error) < 0.01:  # Close enough
            break
        # Adjust by the phase error converted to days
        day_correction = (phase_error / 360) * SYNODIC_MONTH
        estimated_full_jd -= day_correction
    
    return estimated_full_jd


def jd_to_cln_precise(jd):
    """Convert Julian Date to CLN using precise lunar phase calculations.
    
    Returns (cln, days_offset) or None if outside ephemeris range.
    
    CLN is determined by finding the actual last Full Moon and counting
    lunations from the epoch Full Moon. This ensures CLN changes exactly
    at each Full Moon.
    """
    last_full_jd = find_last_full_moon(jd)
    if last_full_jd is None:
        return None
    
    # Days since that Full Moon (precise)
    days_offset = jd - last_full_jd
    
    # Count lunations from epoch to that Full Moon
    # Use round() because we're counting Full Moons, and any drift
    # between average and actual should round to the correct integer
    lunations_from_epoch = (last_full_jd - CLN_EPOCH_JD) / SYNODIC_MONTH
    cln = int(round(lunations_from_epoch))
    
    return cln, days_offset


def jd_to_cln_average(jd):
    """Convert Julian Date to CLN using average synodic month (fallback).
    
    Less precise but works for any date.
    """
    days_from_epoch = jd - CLN_EPOCH_JD
    lunations = days_from_epoch / SYNODIC_MONTH
    cln = int(np.floor(lunations))
    days_offset = (lunations - cln) * SYNODIC_MONTH
    return cln, days_offset


def jd_to_cln(jd):
    """Convert Julian Date to Catalina Lunation Number and days since Full Moon.
    
    Returns (cln, days_offset) where:
    - cln is the integer lunation number (changes exactly at each Full Moon)
    - days_offset is the decimal days since that Full Moon (0.0 to ~29.53)
    
    Uses precise lunar phase calculations when possible (1899-2053),
    falls back to average-based calculation for dates outside ephemeris range.
    """
    # Try precise calculation first
    result = jd_to_cln_precise(jd)
    if result is not None:
        return result
    
    # Fall back to average-based calculation
    return jd_to_cln_average(jd)


def cln_to_jd(cln, days_offset=0.0):
    """Convert Catalina Lunation Number and days offset to Julian Date.
    
    Args:
        cln: Integer lunation number
        days_offset: Decimal days after that Full Moon
    
    Returns Julian Date (approximate - uses average synodic month)
    """
    # Estimate the Full Moon JD for this CLN
    full_moon_jd = CLN_EPOCH_JD + (cln * SYNODIC_MONTH)
    return full_moon_jd + days_offset


def make_utc_qdatetime(year, month, day, hour=0, minute=0, second=0):
    """Create a QDateTime with UTC timeSpec.
    
    This ensures the datetime is treated as UTC, not local time.
    Uses QDate and QTime objects for PyQt5/6 compatibility.
    """
    date = QDate(year, month, day)
    time = QTime(hour, minute, second)
    qdt = QDateTime(date, time, QtCompat.UTC)
    return qdt


def python_datetime_to_utc_qdatetime(dt):
    """Convert a Python datetime (assumed UTC) to a QDateTime with UTC timeSpec."""
    return make_utc_qdatetime(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)


class CoordinateTransformer:
    """Transform between coordinate systems"""
    
    @staticmethod
    def equatorial_to_ecliptic(ra, dec):
        """Convert equatorial (RA/Dec) to ecliptic (lambda/beta)"""
        # Obliquity of ecliptic
        eps = np.radians(23.43928)
        
        ra_rad = np.radians(ra)
        dec_rad = np.radians(dec)
        
        # Convert to ecliptic
        lambda_rad = np.arctan2(
            np.sin(ra_rad) * np.cos(eps) + np.tan(dec_rad) * np.sin(eps),
            np.cos(ra_rad)
        )
        beta_rad = np.arcsin(
            np.sin(dec_rad) * np.cos(eps) - np.cos(dec_rad) * np.sin(eps) * np.sin(ra_rad)
        )
        
        lambda_deg = np.degrees(lambda_rad)
        beta_deg = np.degrees(beta_rad)
        
        # Wrap to 0-360
        lambda_deg = np.where(lambda_deg < 0, lambda_deg + 360, lambda_deg)
        
        return lambda_deg, beta_deg
    
    @staticmethod
    def equatorial_to_galactic(ra, dec):
        """Convert equatorial (RA/Dec) to galactic (l/b)"""
        try:
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            
            coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
            galactic = coord.galactic
            
            return galactic.l.deg, galactic.b.deg
        except ImportError:
            logger.warning("astropy not available, galactic coordinates disabled")
            return ra, dec
    
    @staticmethod
    def galactic_to_equatorial(l, b):
        """Convert galactic (l/b) to equatorial (RA/Dec)"""
        try:
            from astropy.coordinates import SkyCoord
            import astropy.units as u
            
            coord = SkyCoord(l=l*u.deg, b=b*u.deg, frame='galactic')
            icrs = coord.icrs
            
            return icrs.ra.deg, icrs.dec.deg
        except ImportError:
            logger.warning("astropy not available, galactic coordinates disabled")
            return l, b


class SkyMapCanvas(FigureCanvas):
    """Professional sky map with projections and coordinate systems"""
    
    # CNEOS Discovery Site color mapping
    CNEOS_SITE_COLORS = {
        # LINEAR - blue
        '704': '#4263D8', 'G45': '#4263D8', 'P07': '#4263D8',
        # NEAT - orange
        '566': '#F58231', '608': '#F58231', '644': '#F58231',
        # Spacewatch - red
        '691': '#E6184B', '291': '#E6184B',
        # LONEOS - yellow
        '699': '#FFE117',
        # Catalina - green
        '703': '#3DB44B', 'G96': '#3DB44B', 'E12': '#3DB44B', 'I52': '#3DB44B', 'V06': '#3DB44B',
        # Pan-STARRS - magenta
        'F51': '#EF33E6', 'F52': '#EF33E6',
        # NEOWISE - teal
        'C51': '#479990',
        # ATLAS - cyan
        'T05': '#43D4F4', 'T08': '#43D4F4', 'W68': '#43D4F4', 'M22': '#43D4F4', 'T07': '#43D4F4', 'R17': '#43D4F4',
        # Other-US - brown
        'V00': '#9A6324', 'W84': '#9A6324', 'I41': '#9A6324', 'U68': '#9A6324', 'U74': '#9A6324',
    }
    CNEOS_DEFAULT_COLOR = '#A9A9A9'  # Gray for others
    
    CNEOS_LEGEND_ENTRIES = [
        ('LINEAR', '#4263D8'),
        ('NEAT', '#F58231'),
        ('Spacewatch', '#E6184B'),
        ('LONEOS', '#FFE117'),
        ('Catalina', '#3DB44B'),
        ('Pan-STARRS', '#EF33E6'),
        ('NEOWISE', '#479990'),
        ('ATLAS', '#43D4F4'),
        ('Other-US', '#9A6324'),
        ('Others', '#A9A9A9'),
    ]
    
    def __init__(self, parent=None, show_fps=False):
        # Dynamic figure size - will resize with window
        self.fig = Figure(dpi=100)
        super().__init__(self.fig)
        self.setParent(parent)

        # Make canvas expand to fill available space
        from PyQt6.QtWidgets import QSizePolicy
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # FPS display flag
        self._show_fps = show_fps

        # Settings
        self.projection = 'hammer'  # Default to Hammer
        self.coord_system = 'equatorial'
        self.cbar_min = 19.0  # Default minimum magnitude
        self.cbar_max = 23.0
        self.cmap = 'viridis_r'  # Default colormap
        self.h_resolution = 30  # Horizontal grid spacing in degrees
        self.v_resolution = 15  # Vertical grid spacing in degrees
        
        # Plot elements
        self.ax = None
        self.scatter = None
        self.cbar = None
        self.stats_text = None
        self.calendar_text = None
        self.phase_text = None
        self.stats_visible = True  # Track stats box visibility
        
        # Click-to-identify data
        self.current_asteroids = None  # List of asteroid dicts
        self.current_positions = None  # Position array (id, ra, dec, dist, mag)
        self.visible_indices = None    # Indices into current_asteroids for visible points
        self.plot_offsets = None       # Screen coordinates of plotted points
        self.current_jd = None         # Current Julian date
        
        # Selection mode
        self.selection_mode = None  # None, 'rectangle', or 'circle'
        self.selector = None
        self.selected_indices = None  # Indices of selected objects
        
        # Trailing
        self.trailing_settings = {
            'enabled': False,
            'length': 50,
            'weight': 1.0,
            'color': '#00AA00'
        }
        self.trail_history = {}  # Dict: asteroid_id -> list of (jd, x, y) or None for breaks
        self.trail_lines = []    # List of matplotlib line objects
        self.animation_playing = False  # Track animation state
        self.animation_paused = False   # Track if paused (vs stopped)

        # Blitting support for animation performance
        self._background = None
        self._use_blitting = False
        self._animated_artists = []

        # FPS tracking
        self._frame_times = []
        self._last_fps_print = 0

        # Connect click event
        self.mpl_connect('button_press_event', self.on_click)
        
        self.setup_plot()
    
    def setup_plot(self):
        """Setup plot with current projection"""
        self.fig.clear()
        self.trail_lines = []  # Clear trail line references since fig.clear() removes them
        
        # Use tight margins for all projections
        # Give more space on the right for colorbar labels
        self.fig.subplots_adjust(left=0.02, right=0.95, top=0.97, bottom=0.04)
        
        # Create subplot with projection
        if self.projection == 'rectangular':
            self.ax = self.fig.add_subplot(111)
            if self.coord_system == 'opposition':
                # Opposition coords are -180 to +180, centered at 0
                self.ax.set_xlim(180, -180)  # Inverted: +180 on left, -180 on right
            else:
                self.ax.set_xlim(360, 0)  # Inverted: 360 on left, 0 on right
            self.ax.set_ylim(-90, 90)
            # Use auto aspect to fill available space better
            self.ax.set_aspect('auto')
        elif self.projection in ['hammer', 'aitoff', 'mollweide']:
            self.ax = self.fig.add_subplot(111, projection=self.projection)
            # Geographic projections maintain their own aspect ratio automatically
            # They will use maximum available space with minimal margins
        else:
            self.ax = self.fig.add_subplot(111)
            self.ax.invert_xaxis()  # Left-handed coordinates
        
        # Labels based on coordinate system
        if self.coord_system == 'equatorial':
            xlabel, ylabel = 'Right Ascension (°)', 'Declination (°)'
            # Add N/S/E/W labels for equatorial in rectangular
            if self.projection == 'rectangular':
                self.ax.text(0.5, 1.02, 'N', ha='center', va='bottom', transform=self.ax.transAxes, fontweight='bold')
                self.ax.text(0.5, -0.02, 'S', ha='center', va='top', transform=self.ax.transAxes, fontweight='bold')
                self.ax.text(-0.02, 0.5, 'E', ha='right', va='center', transform=self.ax.transAxes, fontweight='bold')
                self.ax.text(1.02, 0.5, 'W', ha='left', va='center', transform=self.ax.transAxes, fontweight='bold')
        elif self.coord_system == 'ecliptic':
            xlabel, ylabel = 'Ecliptic Longitude (°)', 'Ecliptic Latitude (°)'
        elif self.coord_system == 'galactic':
            xlabel, ylabel = 'Galactic Longitude (°)', 'Galactic Latitude (°)'
        elif self.coord_system == 'opposition':
            xlabel, ylabel = 'Opposition-Centered Longitude (°)', 'Ecliptic Latitude (°)'
        else:
            xlabel, ylabel = 'Longitude (°)', 'Latitude (°)'
        
        # For geographic projections, simplify labels
        if self.projection in ['hammer', 'aitoff', 'mollweide']:
            # Keep axis labels but hide tick labels to avoid confusion with flipped axis
            self.ax.set_xlabel(xlabel, fontsize=9, fontweight='bold')
            self.ax.set_ylabel(ylabel, fontsize=9, fontweight='bold')
            # Hide tick labels (they would show wrong values after flip)
            self.ax.set_xticklabels([])
            self.ax.set_yticklabels([])
        else:
            self.ax.set_xlabel(xlabel, fontsize=9, fontweight='bold')
            self.ax.set_ylabel(ylabel, fontsize=9, fontweight='bold')
            self.ax.tick_params(labelsize=8)  # Smaller tick labels
        
        self.ax.grid(True, alpha=0.3, linestyle='--')
        
        # Set grid spacing based on resolution settings
        if self.projection == 'rectangular':
            # For rectangular, use MultipleLocator for custom grid spacing
            from matplotlib.ticker import MultipleLocator
            self.ax.xaxis.set_major_locator(MultipleLocator(self.h_resolution))
            self.ax.yaxis.set_major_locator(MultipleLocator(self.v_resolution))
        elif self.projection in ['hammer', 'aitoff', 'mollweide']:
            # For geographic projections, set the grid lines directly
            # These projections use radians internally (-π to π for lon, -π/2 to π/2 for lat)
            import numpy as np
            # Generate grid line positions in degrees, then convert to radians
            lon_ticks = np.arange(-180, 181, self.h_resolution) * np.pi / 180
            lat_ticks = np.arange(-90, 91, self.v_resolution) * np.pi / 180
            # Filter to valid range
            lon_ticks = lon_ticks[(lon_ticks >= -np.pi) & (lon_ticks <= np.pi)]
            lat_ticks = lat_ticks[(lat_ticks >= -np.pi/2) & (lat_ticks <= np.pi/2)]
            self.ax.set_xticks(lon_ticks)
            self.ax.set_yticks(lat_ticks)
        
        title = f'NEO Sky Map ({self.coord_system.title()})'
        self.ax.set_title(title, fontsize=11, fontweight='bold', pad=4)
        
        # Create empty scatter for near-side NEOs (filled circles)
        self.scatter = self.ax.scatter(
            [], [], s=10, c=[], 
            cmap=self.cmap, alpha=0.7,
            vmin=self.cbar_min, vmax=self.cbar_max,
            zorder=10
        )
        
        # Create empty scatter for far-side NEOs (hollow circles)
        self.scatter_far = self.ax.scatter(
            [], [], s=10,
            facecolors='none', edgecolors='gray', linewidths=1.5,
            alpha=0.7, zorder=10
        )
        
        # Create highlight scatter for selected object (red ring)
        self.scatter_highlight = self.ax.scatter(
            [], [], s=300,
            facecolors='none', edgecolors='red', linewidths=2.5,
            alpha=1.0, zorder=100
        )

        # Density map elements (created on demand)
        self.density_hexbin = None
        self.contour_set = None
        self.display_mode = getattr(self, 'display_mode', 'points')
        
        # Colorbar - MINIMAL size
        self.cbar = self.fig.colorbar(
            self.scatter, ax=self.ax,
            fraction=0.02, pad=0.02,  # Minimal size
            label='Visual Magnitude'
        )
        self.cbar.ax.tick_params(labelsize=8)  # Smaller tick labels
        
        self.draw()  # Fixed: self.draw() not self.canvas.draw()
        
        # Calendar date text (top left, bold and larger) - in a container box
        # Click on this to restore stats if hidden
        self.calendar_text = self.ax.text(
            0.02, 0.98, '',
            transform=self.ax.transAxes,
            verticalalignment='top',
            fontsize=11, fontweight='bold', family='sans-serif',
            color='#1a1a2e',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f4f8', edgecolor='#c0c8d0', alpha=0.9),
            picker=True  # Enable click detection to restore stats
        )
        
        # Phase text (below calendar date, same styling, colored for Full/New Moon)
        # Will be positioned dynamically below calendar text
        self.phase_text = self.ax.text(
            0.02, 0.94, '',
            transform=self.ax.transAxes,
            verticalalignment='top',
            fontsize=11, fontweight='bold', family='sans-serif',
            color='black'
        )
        
        # Stats text (below phase) - contains UTC, CLN, counts, filters
        # Click to dismiss; click calendar to restore
        self.stats_text = self.ax.text(
            0.02, 0.98, '',
            transform=self.ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='#e8eef5', alpha=0.85),  # Light gray-blue
            fontsize=8, family='monospace',
            picker=True  # Enable click detection
        )

        # List of artists to animate during blitting (marked animated only when blitting active)
        self._animated_artists = [
            self.scatter, self.scatter_far, self.scatter_highlight,
            self.stats_text, self.calendar_text, self.phase_text
        ]

        # Connect click handler for stats dismissal/restoration
        self.mpl_connect('pick_event', self.on_stats_pick)
        
        # Set figure background to slight neutral tint (plot remains white)
        self.fig.patch.set_facecolor('#f5f5f5')  # Very light gray
        self.ax.set_facecolor('white')  # Keep plot white
        
        # Don't use tight_layout - manual margins for maximum plot area
        self.draw()

    def capture_background(self):
        """Capture static background for blitting."""
        # Mark all animated artists so they're excluded from background capture
        for artist in self._animated_artists:
            artist.set_animated(True)
        self.fig.canvas.draw()
        self._background = self.fig.canvas.copy_from_bbox(self.ax.bbox)

    def disable_blitting(self):
        """Disable blitting and restore normal drawing."""
        self._use_blitting = False
        self._background = None
        # Unmark artists so they render normally
        for artist in self._animated_artists:
            artist.set_animated(False)
        # Also unmark any trail lines that were added
        for line in self.trail_lines:
            line.set_animated(False)

    def blit_update(self):
        """Fast blitted update - only redraws animated artists."""
        if self._background is None:
            self.draw()
            return
        self.fig.canvas.restore_region(self._background)
        for artist in self._animated_artists:
            if artist.get_visible():
                self.ax.draw_artist(artist)
        self.fig.canvas.blit(self.ax.bbox)

    def set_projection(self, projection):
        """Change projection"""
        self.projection = projection
        self.trail_history.clear()  # Clear trails - old coordinates invalid
        self._clear_trails()
        self._background = None  # Invalidate blitting background
        logger.debug(f"TRAIL: Cleared due to projection change to {projection}")
        self.setup_plot()
    
    def set_coordinate_system(self, coord_system):
        """Change coordinate system"""
        self.coord_system = coord_system
        self.trail_history.clear()  # Clear trails - old coordinates invalid
        self._clear_trails()
        self._background = None  # Invalidate blitting background
        logger.debug(f"TRAIL: Cleared due to coordinate system change to {coord_system}")
        self.setup_plot()
    
    def set_colormap(self, cmap):
        """Change colormap"""
        self.cmap = cmap
        if self.scatter:
            self.scatter.set_cmap(cmap)
            self.draw()
        else:
            self.setup_plot()
    
    def set_colorbar_range(self, vmin, vmax):
        """Update colorbar range"""
        self.cbar_min = vmin
        self.cbar_max = vmax
        if self.scatter:
            self.scatter.set_clim(vmin, vmax)
            self.draw()
    
    def set_resolution(self, h_res, v_res):
        """Set grid resolution in degrees"""
        self.h_resolution = h_res
        self.v_resolution = v_res
        self.setup_plot()
    
    def set_display_mode(self, mode):
        """Set display mode: 'points', 'density', 'contours', or 'points+contours'"""
        self.display_mode = mode
        # Clear density/contour elements when switching modes
        self._clear_density_elements()
    
    def set_trailing_settings(self, settings):
        """Update trailing settings"""
        self.trailing_settings = settings
        # Clear trail history if trailing is disabled
        if not settings.get('enabled', False):
            self.trail_history.clear()
            self._clear_trails()
    
    def _clear_trails(self):
        """Remove all trail lines from the plot"""
        for line in self.trail_lines:
            try:
                line.remove()
            except:
                pass
        self.trail_lines.clear()
    
    def _update_trails(self, asteroids, positions, offsets, jd):
        """Update trail history and draw trails.
        
        Args:
            asteroids: List of asteroid dicts
            positions: Array with [id, ra, dec, dist, mag] for visible asteroids
            offsets: Array with [x, y] plot coordinates matching positions
            jd: Current Julian date
        """
        if not self.trailing_settings.get('enabled', False):
            return
        
        if positions is None or len(positions) == 0 or offsets is None or len(offsets) == 0:
            logger.debug(f"TRAIL: No positions or offsets - clearing all trails")
            # Clear BOTH visual lines AND history when no objects visible
            self._clear_trails()
            self.trail_history.clear()
            return
        
        # Ensure positions and offsets match
        if len(positions) != len(offsets):
            logger.warning(f"TRAIL: positions ({len(positions)}) != offsets ({len(offsets)})")
            return
        
        trail_length = self.trailing_settings.get('length', 50)
        trail_color = self.trailing_settings.get('color', '#00AA00')
        trail_weight = self.trailing_settings.get('weight', 1.0)
        
        # Build lookup for asteroid data
        ast_lookup = {int(ast['id']): ast for ast in asteroids}
        
        # Get set of currently visible asteroid IDs
        visible_ids = set(int(positions[i, 0]) for i in range(len(positions)))
        
        # IMPORTANT: Prune trail_history to only keep entries for currently visible objects
        # This prevents orphan trails from accumulating
        orphan_ids = set(self.trail_history.keys()) - visible_ids
        if orphan_ids:
            logger.debug(f"TRAIL: Removing {len(orphan_ids)} orphan trail histories")
            for orphan_id in orphan_ids:
                del self.trail_history[orphan_id]
        
        # Log first few objects for debugging
        if len(positions) > 0:
            logger.debug(f"TRAIL: JD={jd:.4f}, {len(positions)} visible, {len(self.trail_history)} in history")
            for i in range(min(3, len(positions))):
                ast_id = int(positions[i, 0])
                ra, dec = positions[i, 1], positions[i, 2]
                x, y = offsets[i, 0], offsets[i, 1]
                logger.debug(f"  Object {ast_id}: RA={ra:.2f}, Dec={dec:.2f} -> offset=({x:.4f}, {y:.4f})")
        
        # Maximum time gap (in days) before we consider it a discontinuity
        # If animation advances more than this, we start a new trail segment
        max_time_gap = 1.0  # days - matches clearing threshold for manual navigation
        
        # Update trail history for each visible asteroid
        for i in range(len(positions)):
            ast_id = int(positions[i, 0])
            x, y = offsets[i, 0], offsets[i, 1]
            
            # Get or create history for this asteroid
            if ast_id not in self.trail_history:
                self.trail_history[ast_id] = []
            
            history = self.trail_history[ast_id]
            
            # Check for time discontinuity - if object was not visible for a while,
            # insert a None marker to indicate trail break
            if len(history) > 0:
                last_jd = history[-1][0] if history[-1] is not None else None
                if last_jd is not None and abs(jd - last_jd) > max_time_gap:
                    # Large time gap - insert break marker
                    history.append(None)
                    logger.debug(f"  Object {ast_id}: Time gap {abs(jd - last_jd):.1f} days, inserting break")
            
            # Store plot coordinates directly
            history.append((jd, x, y))
            
            # Trim to max length (count non-None entries)
            while len([h for h in history if h is not None]) > trail_length:
                history.pop(0)
                # Also remove break markers at the start
                while history and history[0] is None:
                    history.pop(0)
        
        # Log trail history stats
        total_points = sum(len([h for h in hist if h is not None]) for hist in self.trail_history.values())
        logger.debug(f"TRAIL: {len(self.trail_history)} objects in history, {total_points} total points, {len(visible_ids)} currently visible")
        
        # Clear existing trail lines
        self._clear_trails()
        
        # Draw trails ONLY for currently visible objects
        lines_drawn = 0
        for ast_id in visible_ids:
            if ast_id not in self.trail_history:
                continue
                
            history = self.trail_history[ast_id]
            if len(history) < 2:
                continue
            
            # Build segments, breaking at None markers and wraparound
            segments = []
            current_segment = []
            last_x = None
            
            for entry in history:
                if entry is None:
                    # Break marker - end current segment
                    if len(current_segment) >= 2:
                        segments.append(current_segment)
                    current_segment = []
                    last_x = None
                    continue
                
                pt_jd, x, y = entry
                
                # Check for coordinate wraparound
                if last_x is not None:
                    x_diff = abs(x - last_x)
                    # For rectangular: jump > 180 degrees
                    # For Hammer etc: jump > pi radians
                    threshold = 180 if self.projection == 'rectangular' else np.pi
                    if x_diff > threshold:
                        # Wraparound - end current segment
                        if len(current_segment) >= 2:
                            segments.append(current_segment)
                        current_segment = []
                
                current_segment.append((x, y))
                last_x = x
            
            # Add final segment
            if len(current_segment) >= 2:
                segments.append(current_segment)
            
            # Draw segments
            for seg in segments:
                xs = [p[0] for p in seg]
                ys = [p[1] for p in seg]
                line, = self.ax.plot(xs, ys, color=trail_color, linewidth=trail_weight,
                                     alpha=0.7, zorder=1)
                self.trail_lines.append(line)
                lines_drawn += 1
        
        logger.debug(f"TRAIL: Drew {lines_drawn} trail lines for {len(visible_ids)} visible objects")
    
    def _clear_density_elements(self):
        """Clear density map and contour elements"""
        if self.density_hexbin is not None:
            try:
                self.density_hexbin.remove()
            except:
                pass
            self.density_hexbin = None
        
        if self.contour_set is not None:
            try:
                # Try direct removal first (matplotlib 3.8+)
                self.contour_set.remove()
            except AttributeError:
                # Fallback for older matplotlib
                try:
                    if hasattr(self.contour_set, 'collections'):
                        for coll in list(self.contour_set.collections):
                            try:
                                coll.remove()
                            except:
                                pass
                except:
                    pass
            except Exception as e:
                logger.debug(f"Error clearing contours: {e}")
            
            # Clear labels separately (clabel creates Text objects)
            try:
                if hasattr(self.contour_set, 'labelTexts'):
                    for txt in list(self.contour_set.labelTexts):
                        try:
                            txt.remove()
                        except:
                            pass
            except:
                pass
            
            self.contour_set = None
    
    def _draw_density_map(self, offsets):
        """Draw density map - hexbin for rectangular, pcolormesh for projections"""
        self._clear_density_elements()
        
        if len(offsets) < 3:
            return
        
        x, y = offsets[:, 0], offsets[:, 1]
        
        # Get settings
        settings = getattr(self, 'display_settings', {})
        gridsize = settings.get('density_gridsize', 35)
        colormap = settings.get('density_colormap', 'YlOrRd')
        scale = settings.get('density_scale', 'auto')
        
        # Determine vmin/vmax
        if scale == 'auto':
            vmin, vmax = None, None
        else:
            vmin, vmax = scale
        
        if self.projection == 'rectangular':
            # Hexbin works well for rectangular projection
            self.density_hexbin = self.ax.hexbin(
                x, y, gridsize=gridsize, cmap=colormap, 
                mincnt=1, alpha=0.8, zorder=5,
                vmin=vmin, vmax=vmax
            )
            mappable = self.density_hexbin
        else:
            # For projections, use pcolormesh with computed density grid
            # This avoids gaps that hexbin creates on curved projections
            
            # Create grid in radians for projection
            n_lon = gridsize * 2
            n_lat = gridsize
            lon_edges = np.linspace(-np.pi, np.pi, n_lon + 1)
            lat_edges = np.linspace(-np.pi/2, np.pi/2, n_lat + 1)
            
            # Compute 2D histogram
            H, _, _ = np.histogram2d(x, y, bins=[lon_edges, lat_edges])
            H = H.T  # Transpose to match meshgrid orientation
            
            # Create meshgrid for pcolormesh
            Lon, Lat = np.meshgrid(lon_edges, lat_edges)
            
            # Mask zero values for transparency
            H_masked = np.ma.masked_where(H == 0, H)
            
            self.density_hexbin = self.ax.pcolormesh(
                Lon, Lat, H_masked, cmap=colormap,
                alpha=0.8, zorder=5, shading='flat',
                vmin=vmin, vmax=vmax
            )
            mappable = self.density_hexbin
        
        # Update existing colorbar instead of creating new one (preserves layout)
        if hasattr(self, 'cbar') and self.cbar is not None:
            self.cbar.update_normal(mappable)
            self.cbar.set_label('NEO Count')
            self.cbar.ax.set_visible(True)
        
        # Re-apply fixed layout to prevent any drift
        self.fig.subplots_adjust(left=0.02, right=0.95, top=0.97, bottom=0.04)
    
    def _draw_contours(self, offsets):
        """Draw density contours"""
        if self.contour_set is not None:
            try:
                for coll in self.contour_set.collections:
                    coll.remove()
            except:
                pass
            self.contour_set = None
        
        if len(offsets) < 10:
            return
        
        # Get settings
        settings = getattr(self, 'display_settings', {})
        levels = settings.get('contour_levels', 8)
        smoothing = settings.get('contour_smoothing', 0.15)
        
        x, y = offsets[:, 0], offsets[:, 1]
        
        # Create grid for density estimation
        if self.projection in ['hammer', 'aitoff', 'mollweide']:
            # For projections, use radian-based grid
            xi = np.linspace(-np.pi, np.pi, 80)
            yi = np.linspace(-np.pi/2, np.pi/2, 40)
        else:
            # For rectangular, use degree-based grid
            xi = np.linspace(0, 360, 90)
            yi = np.linspace(-90, 90, 45)
        
        try:
            # Try scipy KDE first
            from scipy import ndimage
            from scipy.stats import gaussian_kde
            
            Xi, Yi = np.meshgrid(xi, yi)
            positions = np.vstack([Xi.ravel(), Yi.ravel()])
            values = np.vstack([x, y])
            kernel = gaussian_kde(values, bw_method=smoothing)
            Z = np.reshape(kernel(positions).T, Xi.shape)
            
            # Apply Gaussian smoothing
            Z = ndimage.gaussian_filter(Z, sigma=1.0)
            
        except ImportError:
            # Fallback: use numpy histogram2d
            Xi, Yi = np.meshgrid(xi, yi)
            Z, _, _ = np.histogram2d(x, y, bins=[len(xi)-1, len(yi)-1], 
                                     range=[[xi.min(), xi.max()], [yi.min(), yi.max()]])
            Z = Z.T  # Transpose to match meshgrid orientation
            # Simple smoothing using convolution
            from numpy.lib.stride_tricks import sliding_window_view
            # Pad and smooth
            Z = np.pad(Z, 1, mode='edge')
            kern = np.ones((3, 3)) / 9
            Z_smooth = np.zeros_like(Z[1:-1, 1:-1])
            for i in range(Z_smooth.shape[0]):
                for j in range(Z_smooth.shape[1]):
                    Z_smooth[i, j] = np.sum(Z[i:i+3, j:j+3] * kern)
            Z = Z_smooth
            # Resize Xi, Yi to match
            Xi = Xi[:-1, :-1]
            Yi = Yi[:-1, :-1]
        
        try:
            self.contour_set = self.ax.contour(
                Xi, Yi, Z, levels=levels,
                colors='darkblue', linewidths=0.8, alpha=0.7, zorder=8
            )
            # Only label if we have valid contours
            if len(self.contour_set.levels) > 0:
                self.ax.clabel(self.contour_set, inline=True, fontsize=7, fmt='%.1f')
        except Exception as e:
            logger.debug(f"Contour drawing failed: {e}")
    
    def draw_celestial_overlays(self, jd):
        """Draw ecliptic, equator, galactic plane, poles, sun, and solar opposition"""
        # Remove old overlay artists - clear all lines from previous overlays
        if hasattr(self, 'overlay_artists'):
            for artist in self.overlay_artists:
                try:
                    artist.remove()
                except Exception as e:
                    logger.debug(f"Error removing artist: {e}")
        self.overlay_artists = []
        
        # Pre-calculate sun's ecliptic longitude (needed for opposition coordinates)
        opp_offset = 0
        if self.coord_system == 'opposition':
            try:
                ts = skyfield_load.timescale()
                t = ts.tt_jd(jd)
                ensure_ephemeris('de421.bsp')
                eph = skyfield_load('de421.bsp')
                earth = eph['earth']
                sun = eph['sun']
                astrometric = earth.at(t).observe(sun)
                ra_sun, dec_sun, _ = astrometric.radec()
                sun_ecl_lon, _ = CoordinateTransformer.equatorial_to_ecliptic(ra_sun.degrees, dec_sun.degrees)
                self.sun_ecl_lon = sun_ecl_lon
                opp_offset = (sun_ecl_lon + 180) % 360  # Opposition point in ecliptic coords
            except Exception as e:
                logger.error(f"Error calculating sun position for opposition coords: {e}")
                opp_offset = 0
        
        # Get plane settings (use defaults if not set)
        plane_settings = getattr(self, 'plane_settings', None)
        if plane_settings is None:
            # Default settings based on coordinate system
            plane_settings = {
                'equator': {'enabled': self.coord_system not in ['equatorial'], 'color': '#00FFFF', 'pole': self.coord_system not in ['equatorial']},
                'ecliptic': {'enabled': self.coord_system not in ['ecliptic', 'opposition'], 'color': '#4169E1', 'pole': self.coord_system not in ['ecliptic', 'opposition']},
                'galaxy': {'enabled': self.coord_system not in ['galactic'], 'color': '#FF00FF', 'pole': self.coord_system not in ['galactic']}
            }
        
        # zorder: grid=0, planes=5, NEOs=10, poles=15, sun/opposition=20
        PLANE_ZORDER = 5
        POLE_ZORDER = 15
        
        def draw_plane_line(lon_array, lat_array, color, alpha=0.8):
            """Helper to draw a plane line handling wrap-around"""
            if self.projection in ['hammer', 'aitoff', 'mollweide']:
                lon_array = np.where(lon_array > 180, lon_array - 360, lon_array)
                lon_diff = np.abs(np.diff(lon_array))
                breaks = np.where(lon_diff > 180)[0]
                
                if len(breaks) > 0:
                    start = 0
                    for break_idx in breaks:
                        if break_idx > start:
                            seg_lon = lon_array[start:break_idx+1]
                            seg_lat = lat_array[start:break_idx+1]
                            lon_rad = np.radians(-seg_lon)
                            lat_rad = np.radians(seg_lat)
                            line = self.ax.plot(lon_rad, lat_rad, color=color, alpha=alpha, linewidth=1.5, zorder=PLANE_ZORDER)[0]
                            self.overlay_artists.append(line)
                        start = break_idx + 1
                    if start < len(lon_array):
                        seg_lon = lon_array[start:]
                        seg_lat = lat_array[start:]
                        lon_rad = np.radians(-seg_lon)
                        lat_rad = np.radians(seg_lat)
                        line = self.ax.plot(lon_rad, lat_rad, color=color, alpha=alpha, linewidth=1.5, zorder=PLANE_ZORDER)[0]
                        self.overlay_artists.append(line)
                else:
                    lon_rad = np.radians(-lon_array)
                    lat_rad = np.radians(lat_array)
                    line = self.ax.plot(lon_rad, lat_rad, color=color, alpha=alpha, linewidth=1.5, zorder=PLANE_ZORDER)[0]
                    self.overlay_artists.append(line)
            else:
                lon_diff = np.abs(np.diff(lon_array))
                breaks = np.where(lon_diff > 180)[0]
                if len(breaks) > 0:
                    start = 0
                    for break_idx in breaks:
                        if break_idx > start:
                            line = self.ax.plot(lon_array[start:break_idx+1], lat_array[start:break_idx+1], 
                                              color=color, alpha=alpha, linewidth=1.5, zorder=PLANE_ZORDER)[0]
                            self.overlay_artists.append(line)
                        start = break_idx + 1
                    if start < len(lon_array):
                        line = self.ax.plot(lon_array[start:], lat_array[start:], 
                                          color=color, alpha=alpha, linewidth=1.5, zorder=PLANE_ZORDER)[0]
                        self.overlay_artists.append(line)
                else:
                    line = self.ax.plot(lon_array, lat_array, color=color, alpha=alpha, linewidth=1.5, zorder=PLANE_ZORDER)[0]
                    self.overlay_artists.append(line)
        
        def draw_pole_marker(lon, lat, color, marker_type):
            """Draw a pole marker at given coordinates using scatter markers (no distortion)"""
            if self.projection in ['hammer', 'aitoff', 'mollweide']:
                lon_plot = lon if lon <= 180 else lon - 360
                lon_rad = np.radians(-lon_plot)
                lat_rad = np.radians(lat)
            else:
                lon_rad = lon
                lat_rad = lat
            
            # Use matplotlib markers which don't distort with projection
            if marker_type == 'plus':  # Ecliptic poles - use '+' marker
                m = self.ax.scatter([lon_rad], [lat_rad], marker='+', s=120, c=color,
                                   linewidths=2, zorder=POLE_ZORDER)
                self.overlay_artists.append(m)
            elif marker_type == 'x':  # Galactic poles - use 'x' marker
                m = self.ax.scatter([lon_rad], [lat_rad], marker='x', s=100, c=color,
                                   linewidths=2, zorder=POLE_ZORDER)
                self.overlay_artists.append(m)
            elif marker_type == 'triangle':  # Celestial poles - use '^' marker
                m = self.ax.scatter([lon_rad], [lat_rad], marker='^', s=80, c=color,
                                   edgecolors='black', linewidths=0.5, zorder=POLE_ZORDER)
                self.overlay_artists.append(m)
        
        # Draw ecliptic plane and poles
        if plane_settings['ecliptic']['enabled']:
            try:
                color = plane_settings['ecliptic']['color']
                if self.coord_system == 'equatorial':
                    ecl_lon = np.linspace(0, 360, 360)
                    eps = np.radians(23.43928)
                    ecl_lon_rad = np.radians(ecl_lon)
                    ra = np.degrees(ecl_lon_rad)
                    dec = np.degrees(np.arcsin(np.sin(eps) * np.sin(ecl_lon_rad)))
                    draw_plane_line(ra, dec, color)
                elif self.coord_system == 'galactic':
                    # Convert ecliptic to galactic
                    ecl_lon = np.linspace(0, 360, 360)
                    eps = np.radians(23.43928)
                    ra = ecl_lon
                    dec = np.degrees(np.arcsin(np.sin(eps) * np.sin(np.radians(ecl_lon))))
                    try:
                        gal_l, gal_b = CoordinateTransformer.equatorial_to_galactic(ra, dec)
                        draw_plane_line(gal_l, gal_b, color)
                    except:
                        pass
                elif self.coord_system == 'opposition':
                    # Ecliptic plane in opposition coords - horizontal line at lat=0
                    # but longitude is shifted by opposition point
                    # Don't draw - it's the coordinate system baseline
                    pass
                # Ecliptic poles in equatorial: RA=270°, Dec=+66.56° (NEP) and RA=90°, Dec=-66.56° (SEP)
                if plane_settings['ecliptic']['pole']:
                    nep_ra, nep_dec = 270.0, 66.56
                    sep_ra, sep_dec = 90.0, -66.56
                    if self.coord_system == 'galactic':
                        try:
                            nep_ra, nep_dec = CoordinateTransformer.equatorial_to_galactic(nep_ra, nep_dec)
                            sep_ra, sep_dec = CoordinateTransformer.equatorial_to_galactic(sep_ra, sep_dec)
                        except:
                            pass
                    elif self.coord_system == 'ecliptic':
                        nep_ra, nep_dec = 0, 90  # North ecliptic pole in ecliptic coords
                        sep_ra, sep_dec = 0, -90
                    elif self.coord_system == 'opposition':
                        nep_ra, nep_dec = 0, 90  # North ecliptic pole (lat=90)
                        sep_ra, sep_dec = 0, -90  # South ecliptic pole (lat=-90)
                    draw_pole_marker(nep_ra, nep_dec, color, 'plus')
                    draw_pole_marker(sep_ra, sep_dec, color, 'plus')
            except Exception as e:
                logger.error(f"Error drawing ecliptic: {e}")
        
        # Draw equator plane and poles
        if plane_settings['equator']['enabled']:
            try:
                color = plane_settings['equator']['color']
                if self.coord_system == 'ecliptic':
                    eq_ra = np.linspace(0, 360, 360)
                    eq_dec = np.zeros(360)
                    ecl_lon, ecl_lat = CoordinateTransformer.equatorial_to_ecliptic(eq_ra, eq_dec)
                    draw_plane_line(ecl_lon, ecl_lat, color)
                elif self.coord_system == 'galactic':
                    eq_ra = np.linspace(0, 360, 360)
                    eq_dec = np.zeros(360)
                    try:
                        gal_l, gal_b = CoordinateTransformer.equatorial_to_galactic(eq_ra, eq_dec)
                        draw_plane_line(gal_l, gal_b, color)
                    except:
                        pass
                elif self.coord_system == 'opposition':
                    # Convert equator to ecliptic, then shift by opposition
                    eq_ra = np.linspace(0, 360, 360)
                    eq_dec = np.zeros(360)
                    ecl_lon, ecl_lat = CoordinateTransformer.equatorial_to_ecliptic(eq_ra, eq_dec)
                    # Shift longitude so opposition is at center
                    ecl_lon = ecl_lon - opp_offset
                    ecl_lon = np.where(ecl_lon > 180, ecl_lon - 360, ecl_lon)
                    ecl_lon = np.where(ecl_lon < -180, ecl_lon + 360, ecl_lon)
                    draw_plane_line(ecl_lon, ecl_lat, color)
                # Celestial poles
                if plane_settings['equator']['pole']:
                    ncp_ra, ncp_dec = 0, 90  # North celestial pole
                    scp_ra, scp_dec = 0, -90  # South celestial pole
                    if self.coord_system == 'ecliptic':
                        ncp_ra, ncp_dec = CoordinateTransformer.equatorial_to_ecliptic(0, 90)
                        scp_ra, scp_dec = CoordinateTransformer.equatorial_to_ecliptic(0, -90)
                    elif self.coord_system == 'galactic':
                        try:
                            ncp_ra, ncp_dec = CoordinateTransformer.equatorial_to_galactic(0, 90)
                            scp_ra, scp_dec = CoordinateTransformer.equatorial_to_galactic(0, -90)
                        except:
                            pass
                    elif self.coord_system == 'opposition':
                        # Convert to ecliptic then shift
                        ncp_ecl_lon, ncp_ecl_lat = CoordinateTransformer.equatorial_to_ecliptic(0, 90)
                        scp_ecl_lon, scp_ecl_lat = CoordinateTransformer.equatorial_to_ecliptic(0, -90)
                        ncp_ra = (ncp_ecl_lon - opp_offset + 180) % 360 - 180
                        ncp_dec = ncp_ecl_lat
                        scp_ra = (scp_ecl_lon - opp_offset + 180) % 360 - 180
                        scp_dec = scp_ecl_lat
                    draw_pole_marker(ncp_ra, ncp_dec, color, 'triangle')
                    draw_pole_marker(scp_ra, scp_dec, color, 'triangle')
            except Exception as e:
                logger.error(f"Error drawing equator: {e}")
        
        # Draw galactic plane and poles
        if plane_settings['galaxy']['enabled']:
            try:
                color = plane_settings['galaxy']['color']
                gal_l = np.linspace(0, 360, 360)
                gal_b = np.zeros(360)
                
                if self.coord_system == 'equatorial':
                    try:
                        gal_lon, gal_lat = CoordinateTransformer.galactic_to_equatorial(gal_l, gal_b)
                        draw_plane_line(gal_lon, gal_lat, color)
                    except:
                        pass
                elif self.coord_system == 'ecliptic':
                    try:
                        eq_ra, eq_dec = CoordinateTransformer.galactic_to_equatorial(gal_l, gal_b)
                        gal_lon, gal_lat = CoordinateTransformer.equatorial_to_ecliptic(eq_ra, eq_dec)
                        draw_plane_line(gal_lon, gal_lat, color)
                    except:
                        pass
                elif self.coord_system == 'opposition':
                    try:
                        # Convert galactic to equatorial, then to ecliptic, then shift
                        eq_ra, eq_dec = CoordinateTransformer.galactic_to_equatorial(gal_l, gal_b)
                        gal_lon, gal_lat = CoordinateTransformer.equatorial_to_ecliptic(eq_ra, eq_dec)
                        # Shift longitude so opposition is at center
                        gal_lon = gal_lon - opp_offset
                        gal_lon = np.where(gal_lon > 180, gal_lon - 360, gal_lon)
                        gal_lon = np.where(gal_lon < -180, gal_lon + 360, gal_lon)
                        draw_plane_line(gal_lon, gal_lat, color)
                    except:
                        pass
                else:  # galactic coordinates - don't draw plane on itself
                    pass
                
                # Galactic poles: NGP at RA=192.85°, Dec=+27.13° and SGP at RA=12.85°, Dec=-27.13°
                if plane_settings['galaxy']['pole']:
                    ngp_ra, ngp_dec = 192.85, 27.13
                    sgp_ra, sgp_dec = 12.85, -27.13
                    if self.coord_system == 'ecliptic':
                        ngp_ra, ngp_dec = CoordinateTransformer.equatorial_to_ecliptic(ngp_ra, ngp_dec)
                        sgp_ra, sgp_dec = CoordinateTransformer.equatorial_to_ecliptic(sgp_ra, sgp_dec)
                    elif self.coord_system == 'galactic':
                        ngp_ra, ngp_dec = 0, 90  # North galactic pole in galactic coords
                        sgp_ra, sgp_dec = 0, -90
                    elif self.coord_system == 'opposition':
                        # Convert to ecliptic then shift
                        ngp_ecl_lon, ngp_ecl_lat = CoordinateTransformer.equatorial_to_ecliptic(192.85, 27.13)
                        sgp_ecl_lon, sgp_ecl_lat = CoordinateTransformer.equatorial_to_ecliptic(12.85, -27.13)
                        ngp_ra = (ngp_ecl_lon - opp_offset + 180) % 360 - 180
                        ngp_dec = ngp_ecl_lat
                        sgp_ra = (sgp_ecl_lon - opp_offset + 180) % 360 - 180
                        sgp_dec = sgp_ecl_lat
                    draw_pole_marker(ngp_ra, ngp_dec, color, 'x')
                    draw_pole_marker(sgp_ra, sgp_dec, color, 'x')
            except Exception as e:
                logger.error(f"Error drawing galactic plane: {e}")
        
        # Draw galactic exclusion band edge lines
        galactic_settings = getattr(self, 'galactic_settings', None)
        if galactic_settings and galactic_settings.get('enabled', False) and galactic_settings.get('show_bounds', True):
            try:
                offset = galactic_settings.get('offset', 15.0)
                color = galactic_settings.get('color', '#FF99FF')
                
                # Generate band edges (galactic b = +offset and -offset)
                gal_l = np.linspace(0, 360, 360)
                
                def draw_band_edge(gal_b_val):
                    """Draw a single band edge line at galactic latitude gal_b_val"""
                    gal_b = np.full(360, gal_b_val)
                    
                    # Transform to current coordinate system
                    if self.coord_system == 'galactic':
                        edge_lon, edge_lat = gal_l.copy(), gal_b.copy()
                    else:
                        eq_ra, eq_dec = CoordinateTransformer.galactic_to_equatorial(gal_l, gal_b)
                        
                        if self.coord_system == 'ecliptic':
                            edge_lon, edge_lat = CoordinateTransformer.equatorial_to_ecliptic(eq_ra, eq_dec)
                        elif self.coord_system == 'opposition':
                            edge_lon, edge_lat = CoordinateTransformer.equatorial_to_ecliptic(eq_ra, eq_dec)
                            edge_lon = edge_lon - opp_offset
                            edge_lon = np.where(edge_lon > 180, edge_lon - 360, edge_lon)
                            edge_lon = np.where(edge_lon < -180, edge_lon + 360, edge_lon)
                        else:  # equatorial
                            edge_lon, edge_lat = eq_ra, eq_dec
                    
                    # Sort by longitude for proper line drawing
                    sort_idx = np.argsort(edge_lon)
                    edge_lon = edge_lon[sort_idx]
                    edge_lat = edge_lat[sort_idx]
                    
                    # Use draw_plane_line style for consistent rendering
                    if self.projection in ['hammer', 'aitoff', 'mollweide']:
                        edge_lon = np.where(edge_lon > 180, edge_lon - 360, edge_lon)
                        lon_diff = np.abs(np.diff(edge_lon))
                        breaks = np.where(lon_diff > 180)[0]
                        
                        if len(breaks) > 0:
                            start = 0
                            for break_idx in breaks:
                                if break_idx > start:
                                    seg_lon = edge_lon[start:break_idx+1]
                                    seg_lat = edge_lat[start:break_idx+1]
                                    lon_rad = np.radians(-seg_lon)
                                    lat_rad = np.radians(seg_lat)
                                    line = self.ax.plot(lon_rad, lat_rad, color=color, alpha=0.6, linewidth=1.5, linestyle='--', zorder=4)[0]
                                    self.overlay_artists.append(line)
                                start = break_idx + 1
                            if start < len(edge_lon):
                                seg_lon = edge_lon[start:]
                                seg_lat = edge_lat[start:]
                                lon_rad = np.radians(-seg_lon)
                                lat_rad = np.radians(seg_lat)
                                line = self.ax.plot(lon_rad, lat_rad, color=color, alpha=0.6, linewidth=1.5, linestyle='--', zorder=4)[0]
                                self.overlay_artists.append(line)
                        else:
                            lon_rad = np.radians(-edge_lon)
                            lat_rad = np.radians(edge_lat)
                            line = self.ax.plot(lon_rad, lat_rad, color=color, alpha=0.6, linewidth=1.5, linestyle='--', zorder=4)[0]
                            self.overlay_artists.append(line)
                    else:
                        # Rectangular projection
                        lon_diff = np.abs(np.diff(edge_lon))
                        breaks = np.where(lon_diff > 180)[0]
                        if len(breaks) > 0:
                            start = 0
                            for break_idx in breaks:
                                if break_idx > start:
                                    line = self.ax.plot(edge_lon[start:break_idx+1], edge_lat[start:break_idx+1], 
                                                       color=color, alpha=0.6, linewidth=1.5, linestyle='--', zorder=4)[0]
                                    self.overlay_artists.append(line)
                                start = break_idx + 1
                            if start < len(edge_lon):
                                line = self.ax.plot(edge_lon[start:], edge_lat[start:], 
                                                   color=color, alpha=0.6, linewidth=1.5, linestyle='--', zorder=4)[0]
                                self.overlay_artists.append(line)
                        else:
                            line = self.ax.plot(edge_lon, edge_lat, color=color, alpha=0.6, linewidth=1.5, linestyle='--', zorder=4)[0]
                            self.overlay_artists.append(line)
                
                # Draw both edges
                draw_band_edge(offset)   # Upper edge (b = +offset)
                draw_band_edge(-offset)  # Lower edge (b = -offset)
                
            except Exception as e:
                logger.error(f"Error drawing galactic band: {e}")
        
        # Calculate and draw Sun position (save for distance calculations)
        self.sun_position = None  # Store for NEO marker calculations
        try:
            ts = skyfield_load.timescale()
            t = ts.tt_jd(jd)
            ensure_ephemeris('de421.bsp')
            eph = skyfield_load('de421.bsp')
            earth = eph['earth']
            sun = eph['sun']
            
            # Sun's apparent position from Earth
            astrometric = earth.at(t).observe(sun)
            ra_sun, dec_sun, dist_sun = astrometric.radec()
            ra_sun_deg = ra_sun.degrees
            dec_sun_deg = dec_sun.degrees
            
            # Store for elongation calculations
            self.sun_ra = ra_sun_deg
            self.sun_dec = dec_sun_deg
            self.sun_dist = dist_sun.au
            
            # Always calculate and store sun's ecliptic longitude (needed for opposition coords)
            sun_ecl_lon, sun_ecl_lat = CoordinateTransformer.equatorial_to_ecliptic(ra_sun_deg, dec_sun_deg)
            self.sun_ecl_lon = sun_ecl_lon
            
            # Transform to current coordinate system
            if self.coord_system == 'ecliptic':
                lon_sun, lat_sun = sun_ecl_lon, sun_ecl_lat
            elif self.coord_system == 'galactic':
                lon_sun, lat_sun = CoordinateTransformer.equatorial_to_galactic(ra_sun_deg, dec_sun_deg)
            elif self.coord_system == 'opposition':
                # In opposition coords, sun is always at -180° (opposite side from center)
                lon_sun, lat_sun = -180, sun_ecl_lat
            else:
                lon_sun, lat_sun = ra_sun_deg, dec_sun_deg
            
            # Solar opposition (180° away)
            if self.coord_system == 'opposition':
                # Opposition is at center (0°) in opposition coords
                lon_opp, lat_opp = 0, -sun_ecl_lat
            else:
                lon_opp = (lon_sun + 180) % 360
                lat_opp = -lat_sun
            
            # Plot markers
            if self.projection in ['hammer', 'aitoff', 'mollweide']:
                lon_sun_plot = lon_sun if lon_sun <= 180 else lon_sun - 360
                lon_opp_plot = lon_opp if lon_opp <= 180 else lon_opp - 360
                # FLIP: Negate longitude to put East on left
                lon_sun_rad = np.radians(-lon_sun_plot)
                lat_sun_rad = np.radians(lat_sun)
                lon_opp_rad = np.radians(-lon_opp_plot)
                lat_opp_rad = np.radians(lat_opp)
            else:
                lon_sun_rad = lon_sun
                lat_sun_rad = lat_sun
                lon_opp_rad = lon_opp
                lat_opp_rad = lat_opp
            
            # Draw opposition benefit circle shading
            opposition_settings = getattr(self, 'opposition_settings', None)
            if opposition_settings and opposition_settings.get('enabled', False) and opposition_settings.get('show_bounds', True):
                try:
                    radius = opposition_settings.get('radius', 5.0)
                    opp_color = opposition_settings.get('color', '#90EE90')
                    
                    if self.projection in ['hammer', 'aitoff', 'mollweide']:
                        # Draw a circle using scatter with large size
                        # Size in points^2, need to convert radius degrees to plot units
                        # For these projections, use a Circle patch approach
                        from matplotlib.patches import Circle
                        radius_rad = np.radians(radius)
                        circle_patch = Circle((lon_opp_rad, lat_opp_rad), radius_rad,
                                            color=opp_color, alpha=0.2, zorder=3,
                                            transform=self.ax.transData)
                        self.ax.add_patch(circle_patch)
                        self.overlay_artists.append(circle_patch)
                    else:
                        # For rectangular projection
                        from matplotlib.patches import Circle
                        circle_patch = Circle((lon_opp_rad, lat_opp_rad), radius,
                                            color=opp_color, alpha=0.2, zorder=3)
                        self.ax.add_patch(circle_patch)
                        self.overlay_artists.append(circle_patch)
                except Exception as e:
                    logger.error(f"Error drawing opposition circle: {e}")
            
            # Get sun/moon display settings
            sunmoon_settings = getattr(self, 'sunmoon_settings', {'show_sun': True, 'show_moon': True, 'show_phases': False,
                                    'lunar_exclusion_enabled': False, 'lunar_radius': 30.0, 
                                    'lunar_penalty': 3.0, 'lunar_color': '#228B22', 'lunar_show_bounds': True})
            
            # Sun marker: yellow circle with red border (1.5 weight)
            # zorder=17 so Moon (zorder=19) can appear in front during eclipses
            if sunmoon_settings.get('show_sun', True):
                sun_marker = self.ax.plot(lon_sun_rad, lat_sun_rad, 'o', 
                                         color='yellow', markersize=12,
                                         markeredgecolor='red', markeredgewidth=1.5, zorder=17)[0]
                self.overlay_artists.append(sun_marker)
            
            # === MOON POSITION AND PHASE ===
            if sunmoon_settings.get('show_moon', True):
                try:
                    moon = eph['moon']
                    astrometric_moon = earth.at(t).observe(moon)
                    ra_moon, dec_moon, dist_moon = astrometric_moon.radec()
                    ra_moon_deg = ra_moon.degrees
                    dec_moon_deg = dec_moon.degrees
                    
                    # Store for reference
                    self.moon_ra = ra_moon_deg
                    self.moon_dec = dec_moon_deg
                    self.moon_dist = dist_moon.au
                    
                    # Calculate lunar phase (illumination fraction)
                    ra1, dec1 = np.radians(ra_moon_deg), np.radians(dec_moon_deg)
                    ra2, dec2 = np.radians(ra_sun_deg), np.radians(dec_sun_deg)
                    cos_elong = (np.sin(dec1) * np.sin(dec2) + 
                                np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2))
                    elongation = np.degrees(np.arccos(np.clip(cos_elong, -1, 1)))
                    
                    illumination = (1 - np.cos(np.radians(elongation))) / 2
                    ra_diff = (ra_moon_deg - ra_sun_deg + 360) % 360
                    waxing = ra_diff < 180
                    
                    # Determine phase name
                    if illumination < 0.03:
                        phase_name = 'New'
                    elif illumination > 0.97:
                        phase_name = 'Full'
                    elif 0.45 < illumination < 0.55:
                        phase_name = 'First Quarter' if waxing else 'Last Quarter'
                    elif illumination < 0.5:
                        phase_name = 'Waxing Crescent' if waxing else 'Waning Crescent'
                    else:
                        phase_name = 'Waxing Gibbous' if waxing else 'Waning Gibbous'
                    
                    # Transform Moon position to current coordinate system
                    if self.coord_system == 'ecliptic':
                        lon_moon, lat_moon = CoordinateTransformer.equatorial_to_ecliptic(ra_moon_deg, dec_moon_deg)
                    elif self.coord_system == 'galactic':
                        lon_moon, lat_moon = CoordinateTransformer.equatorial_to_galactic(ra_moon_deg, dec_moon_deg)
                    elif self.coord_system == 'opposition':
                        moon_ecl_lon, moon_ecl_lat = CoordinateTransformer.equatorial_to_ecliptic(ra_moon_deg, dec_moon_deg)
                        lon_moon = moon_ecl_lon - self.sun_ecl_lon - 180
                        if lon_moon < -180:
                            lon_moon += 360
                        if lon_moon > 180:
                            lon_moon -= 360
                        lat_moon = moon_ecl_lat
                    else:
                        lon_moon, lat_moon = ra_moon_deg, dec_moon_deg
                    
                    # Convert for plotting
                    if self.projection in ['hammer', 'aitoff', 'mollweide']:
                        lon_moon_plot = lon_moon if lon_moon <= 180 else lon_moon - 360
                        lon_moon_rad = np.radians(-lon_moon_plot)
                        lat_moon_rad = np.radians(lat_moon)
                        moon_radius = 0.035  # Same size as Sun
                    else:
                        lon_moon_rad = lon_moon
                        lat_moon_rad = lat_moon
                        moon_radius = 1.5
                    
                    show_phases = sunmoon_settings.get('show_phases', True)
                    
                    if show_phases:
                        # Draw realistic moon phase using patches
                        from matplotlib.patches import Circle, Path as MplPath, PathPatch
                        transform = self.ax.transData
                        
                        # Base dark circle with forest green border
                        dark_circle = Circle((lon_moon_rad, lat_moon_rad), moon_radius,
                                            facecolor='#1a3a1a', edgecolor='#228B22',  # Dark green base, forest green border
                                            linewidth=1.5, zorder=19, transform=transform)
                        self.ax.add_patch(dark_circle)
                        self.overlay_artists.append(dark_circle)
                        
                        # Draw illuminated portion (pale green)
                        if illumination > 0.03:
                            n_points = 30
                            theta = np.linspace(-np.pi/2, np.pi/2, n_points)
                            
                            if illumination >= 0.97:
                                # Full moon - pale green
                                bright_circle = Circle((lon_moon_rad, lat_moon_rad), moon_radius * 0.95,
                                                      facecolor='#98FB98', edgecolor='none',
                                                      zorder=19.5, transform=transform)
                                self.ax.add_patch(bright_circle)
                                self.overlay_artists.append(bright_circle)
                            else:
                                # Crescent/Quarter/Gibbous
                                if illumination < 0.5:
                                    curve = -(1 - 2*illumination)
                                else:
                                    curve = 2*illumination - 1
                                
                                lit_side = 1 if waxing else -1
                                
                                outer_x = lit_side * moon_radius * np.cos(theta)
                                outer_y = moon_radius * np.sin(theta)
                                inner_x = lit_side * moon_radius * curve * np.cos(theta)
                                inner_y = moon_radius * np.sin(theta)
                                
                                verts = []
                                codes = []
                                
                                for i, (ox, oy) in enumerate(zip(outer_x, outer_y)):
                                    verts.append((lon_moon_rad + ox, lat_moon_rad + oy))
                                    codes.append(MplPath.MOVETO if i == 0 else MplPath.LINETO)
                                
                                for ix, iy in zip(inner_x[::-1], inner_y[::-1]):
                                    verts.append((lon_moon_rad + ix, lat_moon_rad + iy))
                                    codes.append(MplPath.LINETO)
                                
                                codes.append(MplPath.CLOSEPOLY)
                                verts.append(verts[0])
                                
                                path = MplPath(verts, codes)
                                patch = PathPatch(path, facecolor='#98FB98', edgecolor='none',
                                                 zorder=19.5, transform=transform)
                                self.ax.add_patch(patch)
                                self.overlay_artists.append(patch)
                    else:
                        # Simple circle (no phase detail) - pale green
                        moon_marker = self.ax.plot(lon_moon_rad, lat_moon_rad, 'o',
                                                  color='#98FB98', markersize=12,
                                                  markeredgecolor='#228B22', markeredgewidth=1.5, zorder=19)[0]
                        self.overlay_artists.append(moon_marker)
                    
                    # Draw lunar exclusion circle if enabled
                    if sunmoon_settings.get('lunar_exclusion_enabled', False) and sunmoon_settings.get('lunar_show_bounds', True):
                        try:
                            # Radius scales with illumination (0 at new moon, max at full moon)
                            max_radius = sunmoon_settings.get('lunar_radius', 30.0)
                            current_radius = max_radius * illumination
                            
                            if current_radius > 0.5:  # Only draw if radius is meaningful
                                lunar_color = sunmoon_settings.get('lunar_color', '#32CD32')
                                
                                if self.projection in ['hammer', 'aitoff', 'mollweide']:
                                    # Draw dashed circle in radians
                                    circle_theta = np.linspace(0, 2*np.pi, 72)
                                    radius_rad = np.radians(current_radius)
                                    cx = lon_moon_rad + radius_rad * np.cos(circle_theta)
                                    cy = lat_moon_rad + radius_rad * np.sin(circle_theta)
                                    lunar_circle = self.ax.plot(cx, cy, '--', color=lunar_color, 
                                                               linewidth=1.5, alpha=0.7, zorder=18)[0]
                                else:
                                    # Rectangular - use degrees
                                    circle_theta = np.linspace(0, 2*np.pi, 72)
                                    cx = lon_moon_rad + current_radius * np.cos(circle_theta)
                                    cy = lat_moon_rad + current_radius * np.sin(circle_theta)
                                    lunar_circle = self.ax.plot(cx, cy, '--', color=lunar_color,
                                                               linewidth=1.5, alpha=0.7, zorder=18)[0]
                                self.overlay_artists.append(lunar_circle)
                        except Exception as e:
                            logger.debug(f"Could not draw lunar exclusion circle: {e}")
                    
                    # Store phase info
                    self.moon_phase = illumination
                    self.moon_phase_name = phase_name
                    self.moon_waxing = waxing
                    
                    logger.debug(f"Drew Moon at ({lon_moon:.1f}, {lat_moon:.1f}), phase: {phase_name} ({illumination*100:.0f}%)")
                    
                except Exception as e:
                    logger.debug(f"Could not draw Moon: {e}")
            
            # === HORIZON AND TWILIGHT BOUNDARIES ===
            horizon_settings = getattr(self, 'horizon_settings', None)
            if horizon_settings and horizon_settings.get('enabled', False):
                try:
                    self._draw_horizon_boundaries(jd, horizon_settings)
                except Exception as e:
                    logger.debug(f"Could not draw horizon boundaries: {e}")
            
            # Opposition marker: circular reticle with crossed lines
            # Use scatter marker for circle to avoid distortion in Mollweide
            opp_circle = self.ax.scatter([lon_opp_rad], [lat_opp_rad], marker='o', s=250,
                                        facecolors='white', edgecolors='red', linewidths=1.5,
                                        alpha=0.7, zorder=20)
            self.overlay_artists.append(opp_circle)
            
            # Draw crossed red lines (1.5 weight)
            if self.projection in ['hammer', 'aitoff', 'mollweide']:
                line_len = 0.05  # Reduced from 0.06
            else:
                line_len = 2.0  # Reduced from 2.5
            
            # Horizontal line (1.5 weight)
            h_line = self.ax.plot([lon_opp_rad - line_len, lon_opp_rad + line_len],
                                 [lat_opp_rad, lat_opp_rad], 
                                 'r-', linewidth=1.5, zorder=21)[0]
            # Vertical line (1.5 weight)
            v_line = self.ax.plot([lon_opp_rad, lon_opp_rad],
                                 [lat_opp_rad - line_len, lat_opp_rad + line_len],
                                 'r-', linewidth=1.5, zorder=21)[0]
            
            self.overlay_artists.extend([h_line, v_line])
            logger.debug(f"Drew Sun at ({lon_sun:.1f}, {lat_sun:.1f}) and opposition")
            
        except Exception as e:
            logger.error(f"Could not draw Sun/opposition: {e}")
            import traceback
            logger.debug(traceback.format_exc())
    
    def _draw_horizon_boundaries(self, jd, horizon_settings):
        """Draw horizon and twilight boundaries for the observer location.
        
        These lines show where the horizon and twilight boundaries fall on the
        celestial sphere for a given observer location and time.
        
        Args:
            jd: Julian Date
            horizon_settings: dict with observer location and display options
        """
        HORIZON_ZORDER = 4  # Below planes (5) but above grid
        
        observer_lat = horizon_settings.get('observer_lat', 32.2226)  # Tucson default
        observer_lon = horizon_settings.get('observer_lon', -110.9747)
        line_style = horizon_settings.get('line_style', 'solid')
        line_weight = horizon_settings.get('line_weight', 1.5)
        
        # Convert line style to matplotlib format
        linestyle_map = {'solid': '-', 'dashed': '--', 'dotted': ':'}
        ls = linestyle_map.get(line_style, '-')
        
        # Calculate Greenwich Mean Sidereal Time from JD
        # Using simplified formula: GMST = 18.697374558 + 24.06570982441908 * D
        # where D is days since J2000.0 (JD 2451545.0)
        D = jd - 2451545.0
        GMST_hours = (18.697374558 + 24.06570982441908 * D) % 24
        
        # Local Sidereal Time = GMST + longitude (in hours)
        LST_hours = (GMST_hours + observer_lon / 15.0) % 24
        LST_deg = LST_hours * 15.0  # Convert to degrees
        
        # Observer latitude in radians
        lat_obs_rad = np.radians(observer_lat)
        
        def altitude_circle_to_radec(altitude_deg):
            """Compute RA/Dec points for a constant altitude circle.
            
            For each azimuth, compute the RA/Dec of the point at that azimuth
            and the given altitude.
            
            Returns arrays of (ra, dec) in degrees.
            """
            alt_rad = np.radians(altitude_deg)
            
            # Sample azimuths (0 = North, 90 = East, etc.)
            azimuths = np.linspace(0, 360, 361)  # Extra point for closure
            az_rad = np.radians(azimuths)
            
            # Compute declination for each azimuth
            # sin(dec) = sin(alt)*sin(lat) + cos(alt)*cos(lat)*cos(az)
            sin_dec = np.sin(alt_rad) * np.sin(lat_obs_rad) + \
                      np.cos(alt_rad) * np.cos(lat_obs_rad) * np.cos(az_rad)
            sin_dec = np.clip(sin_dec, -1, 1)
            dec_rad = np.arcsin(sin_dec)
            
            # Compute hour angle for each azimuth
            # cos(H) = (sin(alt) - sin(dec)*sin(lat)) / (cos(dec)*cos(lat))
            # sin(H) = -sin(az)*cos(alt) / cos(dec)
            cos_dec = np.cos(dec_rad)
            # Avoid division by zero at poles
            cos_dec = np.where(np.abs(cos_dec) < 1e-10, 1e-10, cos_dec)
            
            sin_H = -np.sin(az_rad) * np.cos(alt_rad) / cos_dec
            cos_H = (np.sin(alt_rad) - np.sin(dec_rad) * np.sin(lat_obs_rad)) / \
                    (cos_dec * np.cos(lat_obs_rad + 1e-10))  # Small offset to avoid /0 at poles
            
            # Clamp to valid range
            sin_H = np.clip(sin_H, -1, 1)
            cos_H = np.clip(cos_H, -1, 1)
            
            H_rad = np.arctan2(sin_H, cos_H)
            
            # RA = LST - H
            ra_deg = (LST_deg - np.degrees(H_rad)) % 360
            dec_deg = np.degrees(dec_rad)
            
            return ra_deg, dec_deg
        
        def draw_altitude_boundary(altitude_deg, color, label=None):
            """Draw a boundary line at the given altitude."""
            try:
                ra, dec = altitude_circle_to_radec(altitude_deg)
                
                # Transform to current coordinate system
                if self.coord_system == 'ecliptic':
                    lon, lat = CoordinateTransformer.equatorial_to_ecliptic(ra, dec)
                elif self.coord_system == 'galactic':
                    lon, lat = CoordinateTransformer.equatorial_to_galactic(ra, dec)
                elif self.coord_system == 'opposition':
                    ecl_lon, ecl_lat = CoordinateTransformer.equatorial_to_ecliptic(ra, dec)
                    lon = ecl_lon - self.sun_ecl_lon - 180
                    lon = np.where(lon < -180, lon + 360, lon)
                    lon = np.where(lon > 180, lon - 360, lon)
                    lat = ecl_lat
                else:
                    lon, lat = ra, dec
                
                # Handle projection-specific plotting
                if self.projection in ['hammer', 'aitoff', 'mollweide']:
                    lon_plot = np.where(lon > 180, lon - 360, lon)
                    lon_rad = np.radians(-lon_plot)  # Flip for East-left convention
                    lat_rad = np.radians(lat)
                    
                    # Find discontinuities (large jumps in longitude)
                    lon_diff = np.abs(np.diff(lon_rad))
                    breaks = np.where(lon_diff > np.pi)[0]
                    
                    # Draw segments between breaks
                    if len(breaks) > 0:
                        start = 0
                        for break_idx in breaks:
                            if break_idx > start:
                                seg_lon = lon_rad[start:break_idx+1]
                                seg_lat = lat_rad[start:break_idx+1]
                                line = self.ax.plot(seg_lon, seg_lat, ls, color=color,
                                                   linewidth=line_weight, alpha=0.8,
                                                   zorder=HORIZON_ZORDER)[0]
                                self.overlay_artists.append(line)
                            start = break_idx + 1
                        # Final segment
                        if start < len(lon_rad):
                            seg_lon = lon_rad[start:]
                            seg_lat = lat_rad[start:]
                            line = self.ax.plot(seg_lon, seg_lat, ls, color=color,
                                               linewidth=line_weight, alpha=0.8,
                                               zorder=HORIZON_ZORDER)[0]
                            self.overlay_artists.append(line)
                    else:
                        line = self.ax.plot(lon_rad, lat_rad, ls, color=color,
                                           linewidth=line_weight, alpha=0.8,
                                           zorder=HORIZON_ZORDER)[0]
                        self.overlay_artists.append(line)
                else:
                    # Rectangular projection
                    lon_diff = np.abs(np.diff(lon))
                    breaks = np.where(lon_diff > 180)[0]
                    
                    if len(breaks) > 0:
                        start = 0
                        for break_idx in breaks:
                            if break_idx > start:
                                line = self.ax.plot(lon[start:break_idx+1], lat[start:break_idx+1],
                                                   ls, color=color, linewidth=line_weight,
                                                   alpha=0.8, zorder=HORIZON_ZORDER)[0]
                                self.overlay_artists.append(line)
                            start = break_idx + 1
                        if start < len(lon):
                            line = self.ax.plot(lon[start:], lat[start:], ls, color=color,
                                               linewidth=line_weight, alpha=0.8,
                                               zorder=HORIZON_ZORDER)[0]
                            self.overlay_artists.append(line)
                    else:
                        line = self.ax.plot(lon, lat, ls, color=color,
                                           linewidth=line_weight, alpha=0.8,
                                           zorder=HORIZON_ZORDER)[0]
                        self.overlay_artists.append(line)
                        
            except Exception as e:
                logger.debug(f"Error drawing altitude boundary at {altitude_deg}°: {e}")
        
        # Draw enabled boundaries
        if horizon_settings.get('show_horizon', True):
            draw_altitude_boundary(0, horizon_settings.get('horizon_color', '#FF6600'))
        
        if horizon_settings.get('show_civil', False):
            draw_altitude_boundary(-6, horizon_settings.get('civil_color', '#FF9933'))
        
        if horizon_settings.get('show_nautical', False):
            draw_altitude_boundary(-12, horizon_settings.get('nautical_color', '#CC66FF'))
        
        if horizon_settings.get('show_astro', True):
            draw_altitude_boundary(-18, horizon_settings.get('astro_color', '#6666FF'))
        
        logger.debug(f"Drew horizon boundaries for observer at ({observer_lat:.2f}, {observer_lon:.2f})")
    
    def update_plot(self, positions, mag_min, mag_max, jd=None, show_hollow=True,
                    h_min=None, h_max=None, selected_classes=None,
                    moid_enabled=False, moid_min=None, moid_max=None,
                    orb_filters=None, asteroids=None, size_settings=None,
                    galactic_settings=None, opposition_settings=None, hide_before_discovery=False,
                    hide_missing_discovery=False, color_by='V magnitude', show_legend=False,
                    site_filter=None):
        """Update plot with positions"""
        _profile = self._show_fps
        if _profile:
            _t0 = time.time()
            _times = {}

        # Store data for click-to-identify
        self.current_positions = positions
        self.current_asteroids = asteroids
        self.current_jd = jd
        
        # Default size settings if not provided
        if size_settings is None:
            size_settings = {
                'size_by': 'V magnitude',
                'size_min': 10,
                'size_max': 150,
                'data_min': 19.0,
                'data_max': 23.0,
                'invert': True
            }
        self.size_settings = size_settings
        self.hide_before_discovery = hide_before_discovery
        self.hide_missing_discovery = hide_missing_discovery
        self.color_by = color_by
        self.show_legend = show_legend
        self.site_filter = site_filter
        
        # Store galactic and opposition settings for overlay drawing
        self.galactic_settings = galactic_settings
        self.opposition_settings = opposition_settings
        
        # Draw celestial overlays FIRST to ensure Sun position is available
        # for far-side calculations
        if jd is not None:
            if not hasattr(self, 'last_jd') or self.last_jd is None or abs(jd - self.last_jd) > 0.01:
                self.draw_celestial_overlays(jd)
                self.last_jd = jd

        if _profile:
            _times['overlays'] = time.time() - _t0

        if positions is None or len(positions) == 0:
            self.scatter.set_offsets(np.empty((0, 2)))
            self.scatter.set_array(np.array([]))
            self.scatter_far.set_offsets(np.empty((0, 2)))  # Clear hollow too
            self.stats_text.set_text('No objects')
            # Clear trails when no objects visible
            if self.trailing_settings.get('enabled', False):
                self._clear_trails()
                self.trail_history.clear()
                logger.debug("TRAIL: No positions - cleared trails")
            self.draw()  # Force immediate redraw
            return
        
        # Filter by magnitude (both min and max)
        # Note: We expand mag_max by opposition benefit to allow objects that could become
        # visible due to the benefit to pass initial filtering
        mag = positions[:, 4]
        opposition_max_benefit = 0.0
        if opposition_settings and opposition_settings.get('enabled', False):
            opposition_max_benefit = opposition_settings.get('benefit', 2.0)
        expanded_mag_max = mag_max + opposition_max_benefit
        mask = (mag >= mag_min) & (mag < expanded_mag_max)
        visible = positions[mask]
        
        if len(visible) == 0:
            self.scatter.set_offsets(np.empty((0, 2)))
            self.scatter.set_array(np.array([]))
            self.stats_text.set_text(f'No objects {mag_min:.1f} < mag < {mag_max:.1f}')
            # Clear trails when no objects visible
            if self.trailing_settings.get('enabled', False):
                self._clear_trails()
                self.trail_history.clear()
                logger.debug("TRAIL: No objects in initial mag range - cleared trails")
            self.draw()  # Force immediate redraw
            return
        
        # Extract RA/Dec
        ra = visible[:, 1]
        dec = visible[:, 2]
        mag = visible[:, 4]
        display_mag = mag.copy()  # Keep original magnitudes for color display
        
        # Calculate effective magnitude combining galactic penalty and opposition benefit
        effective_mag = display_mag.copy()
        
        # Track which effects apply to each object (for status display)
        in_galactic_band = np.zeros(len(ra), dtype=bool)
        near_opposition = np.zeros(len(ra), dtype=bool)
        
        # Apply galactic band penalty
        galactic_enabled = galactic_settings and galactic_settings.get('enabled', False)
        if galactic_enabled:
            offset = galactic_settings.get('offset', 15.0)
            penalty = galactic_settings.get('penalty', 2.0)
            
            # Calculate galactic latitude for each object
            _, gal_b = CoordinateTransformer.equatorial_to_galactic(ra, dec)
            in_galactic_band = np.abs(gal_b) < offset
            
            # Add penalty to effective magnitude
            effective_mag[in_galactic_band] += penalty
        
        # Apply lunar exclusion penalty (scales with phase)
        sunmoon_settings = getattr(self, 'sunmoon_settings', None)
        near_moon = np.zeros(len(ra), dtype=bool)
        lunar_penalty_applied = 0.0
        if sunmoon_settings and sunmoon_settings.get('lunar_exclusion_enabled', False):
            if hasattr(self, 'moon_ra') and hasattr(self, 'moon_dec') and hasattr(self, 'moon_phase'):
                max_radius = sunmoon_settings.get('lunar_radius', 30.0)
                max_penalty = sunmoon_settings.get('lunar_penalty', 3.0)
                
                # Current radius and penalty scale with illumination
                current_radius = max_radius * self.moon_phase
                current_penalty = max_penalty * self.moon_phase
                lunar_penalty_applied = current_penalty
                
                if current_radius > 0.5 and current_penalty > 0.01:
                    # Angular separation from moon
                    d_ra = np.radians(ra - self.moon_ra)
                    d_dec = np.radians(dec - self.moon_dec)
                    dec_rad = np.radians(dec)
                    moon_dec_rad = np.radians(self.moon_dec)
                    
                    a = np.sin(d_dec/2)**2 + np.cos(dec_rad) * np.cos(moon_dec_rad) * np.sin(d_ra/2)**2
                    angular_sep = 2 * np.degrees(np.arcsin(np.sqrt(np.clip(a, 0, 1))))
                    
                    near_moon = angular_sep < current_radius
                    
                    # Add penalty to effective magnitude
                    effective_mag[near_moon] += current_penalty
        
        # Apply opposition benefit (subtracts from effective magnitude)
        opposition_enabled = opposition_settings and opposition_settings.get('enabled', False)
        if opposition_enabled and hasattr(self, 'sun_ra') and hasattr(self, 'sun_dec'):
            radius = opposition_settings.get('radius', 5.0)
            benefit = opposition_settings.get('benefit', 2.0)
            
            # Opposition RA/Dec (180° from sun)
            opp_ra = (self.sun_ra + 180) % 360
            opp_dec = -self.sun_dec
            
            # Angular separation using haversine-like formula
            d_ra = np.radians(ra - opp_ra)
            d_dec = np.radians(dec - opp_dec)
            dec_rad = np.radians(dec)
            opp_dec_rad = np.radians(opp_dec)
            
            a = np.sin(d_dec/2)**2 + np.cos(dec_rad) * np.cos(opp_dec_rad) * np.sin(d_ra/2)**2
            angular_sep = 2 * np.degrees(np.arcsin(np.sqrt(np.clip(a, 0, 1))))
            
            near_opposition = angular_sep < radius
            
            # Subtract benefit from effective magnitude (makes objects appear brighter)
            effective_mag[near_opposition] -= benefit
        
        # Filter by effective magnitude
        keep_mask = (effective_mag >= mag_min) & (effective_mag < mag_max)
        visible = visible[keep_mask]
        ra = ra[keep_mask]
        dec = dec[keep_mask]
        display_mag = display_mag[keep_mask]  # Original mags for display
        in_galactic_band = in_galactic_band[keep_mask]
        near_opposition = near_opposition[keep_mask]
        
        if len(visible) == 0:
            self.scatter.set_offsets(np.empty((0, 2)))
            self.scatter.set_array(np.array([]))
            self.stats_text.set_text('No objects in effective magnitude range')
            # Clear trails when no objects visible
            if self.trailing_settings.get('enabled', False):
                self._clear_trails()
                self.trail_history.clear()
                logger.debug("TRAIL: No objects in mag range - cleared trails")
            self.draw()
            return
        
        if _profile:
            _times['magfilter'] = time.time() - _t0

        # Filter by discovery date if enabled
        if hide_before_discovery and asteroids is not None and jd is not None:
            current_mjd = jd - 2400000.5  # Convert JD to MJD
            discovery_mask = np.ones(len(visible), dtype=bool)
            for i in range(len(visible)):
                ast_idx = int(visible[i, 0])
                if ast_idx < len(asteroids):
                    discovery_mjd = asteroids[ast_idx].get('discovery_mjd')
                    if discovery_mjd is not None and current_mjd < discovery_mjd:
                        discovery_mask[i] = False
            
            visible = visible[discovery_mask]
            ra = ra[discovery_mask]
            dec = dec[discovery_mask]
            display_mag = display_mag[discovery_mask]
            in_galactic_band = in_galactic_band[discovery_mask]
            near_opposition = near_opposition[discovery_mask]
            
            if len(visible) == 0:
                self.scatter.set_offsets(np.empty((0, 2)))
                self.scatter.set_array(np.array([]))
                self.stats_text.set_text('No objects discovered yet at this date')
                # Clear trails when no objects visible
                if self.trailing_settings.get('enabled', False):
                    self._clear_trails()
                    self.trail_history.clear()
                    logger.debug("TRAIL: No objects discovered yet - cleared trails")
                self.draw()
                return
        
        # Filter out objects with missing discovery tracklets if enabled
        if hide_missing_discovery and asteroids is not None:
            missing_mask = np.ones(len(visible), dtype=bool)
            for i in range(len(visible)):
                ast_idx = int(visible[i, 0])
                if ast_idx < len(asteroids):
                    discovery_mjd = asteroids[ast_idx].get('discovery_mjd')
                    if discovery_mjd is None:
                        missing_mask[i] = False
            
            visible = visible[missing_mask]
            ra = ra[missing_mask]
            dec = dec[missing_mask]
            display_mag = display_mag[missing_mask]
            in_galactic_band = in_galactic_band[missing_mask]
            near_opposition = near_opposition[missing_mask]
            
            if len(visible) == 0:
                self.scatter.set_offsets(np.empty((0, 2)))
                self.scatter.set_array(np.array([]))
                self.stats_text.set_text('No objects (all have missing discovery data)')
                # Clear trails when no objects visible
                if self.trailing_settings.get('enabled', False):
                    self._clear_trails()
                    self.trail_history.clear()
                    logger.debug("TRAIL: No objects (missing discovery) - cleared trails")
                self.draw()
                return
        
        # Filter by discovery site (whitelist/blacklist)
        if site_filter and asteroids is not None:
            whitelist_enabled = site_filter.get('whitelist_enabled', False)
            whitelist = site_filter.get('whitelist', [])
            blacklist_enabled = site_filter.get('blacklist_enabled', False)
            blacklist = site_filter.get('blacklist', [])
            
            if whitelist_enabled and whitelist or blacklist_enabled and blacklist:
                site_mask = np.ones(len(visible), dtype=bool)
                for i in range(len(visible)):
                    ast_idx = int(visible[i, 0])
                    if ast_idx < len(asteroids):
                        site = asteroids[ast_idx].get('discovery_site', '')
                        site = str(site).strip() if site else ''
                        
                        # Apply whitelist (must be in list)
                        if whitelist_enabled and whitelist:
                            if site not in whitelist:
                                site_mask[i] = False
                                continue
                        
                        # Apply blacklist (must not be in list)
                        if blacklist_enabled and blacklist:
                            if site in blacklist:
                                site_mask[i] = False
                
                visible = visible[site_mask]
                ra = ra[site_mask]
                dec = dec[site_mask]
                display_mag = display_mag[site_mask]
                in_galactic_band = in_galactic_band[site_mask]
                near_opposition = near_opposition[site_mask]
                
                if len(visible) == 0:
                    self.scatter.set_offsets(np.empty((0, 2)))
                    self.scatter.set_array(np.array([]))
                    self.stats_text.set_text('No objects match site filter')
                    # Clear trails when no objects visible
                    if self.trailing_settings.get('enabled', False):
                        self._clear_trails()
                        self.trail_history.clear()
                        logger.debug("TRAIL: No objects match site filter - cleared trails")
                    self.draw()
                    return
        
        if _profile:
            _times['sitefilter'] = time.time() - _t0

        # Use display_mag for color mapping
        mag = display_mag

        # Transform coordinates
        if self.coord_system == 'ecliptic':
            lon, lat = CoordinateTransformer.equatorial_to_ecliptic(ra, dec)
        elif self.coord_system == 'galactic':
            lon, lat = CoordinateTransformer.equatorial_to_galactic(ra, dec)
        elif self.coord_system == 'opposition':
            # Convert to ecliptic first
            lon, lat = CoordinateTransformer.equatorial_to_ecliptic(ra, dec)
            # Adjust longitude so opposition is at center (0°)
            # Opposition is 180° from sun, so we subtract (sun_ecl_lon + 180)
            if hasattr(self, 'sun_ecl_lon'):
                opp_lon = (self.sun_ecl_lon + 180) % 360
                lon = lon - opp_lon
                # Normalize to -180 to +180
                lon = np.where(lon > 180, lon - 360, lon)
                lon = np.where(lon < -180, lon + 360, lon)
        else:
            lon, lat = ra, dec
        
        # Convert for projection
        if self.projection in ['hammer', 'aitoff', 'mollweide']:
            # These need -180 to +180
            lon = np.where(lon > 180, lon - 360, lon)
            # FLIP: Negate longitude to put East on left (matching rectangular projection)
            lon_rad = np.radians(-lon)
            lat_rad = np.radians(lat)
            
            # Filter out invalid coordinates (outside projection bounds)
            # For Hammer: valid range is approximately -180 to +180 lon, -90 to +90 lat
            valid_mask = (np.abs(lon_rad) <= np.pi) & (np.abs(lat_rad) <= np.pi/2)
            
            lon_rad = lon_rad[valid_mask]
            lat_rad = lat_rad[valid_mask]
            mag = mag[valid_mask]
            dist = visible[:, 3][valid_mask]
            visible = visible[valid_mask]
            
            offsets = np.column_stack([lon_rad, lat_rad])
        else:
            offsets = np.column_stack([lon, lat])
            dist = visible[:, 3]
        
        # Sizes based on selected property with user-configurable mapping
        if len(mag) > 0:
            size_by = size_settings.get('size_by', 'V magnitude')
            size_min = size_settings.get('size_min', 10)
            size_max = size_settings.get('size_max', 150)
            data_min = size_settings.get('data_min', 19.0)
            data_max = size_settings.get('data_max', 23.0)
            invert = size_settings.get('invert', True)
            
            # Get data values based on size_by
            if size_by == 'V magnitude':
                data_vals = mag
            elif size_by == 'H magnitude' and asteroids is not None:
                h_vals = np.array([asteroids[int(visible[i, 0])].get('H') for i in range(len(visible))])
                data_vals = np.array([h if h is not None else 22.0 for h in h_vals], dtype=float)
            elif size_by == 'Distance':
                data_vals = visible[:, 3]  # geocentric distance
            elif size_by == 'Earth MOID' and asteroids is not None:
                data_vals = np.array([asteroids[int(visible[i, 0])].get('earth_moid', 0.05) or 0.05 for i in range(len(visible))])
            elif size_by == 'Period' and asteroids is not None:
                # Period from semi-major axis: P = a^1.5
                data_vals = np.array([asteroids[int(visible[i, 0])].get('a', 1.0) ** 1.5 for i in range(len(visible))])
            elif size_by == 'Eccentricity' and asteroids is not None:
                data_vals = np.array([asteroids[int(visible[i, 0])].get('e', 0.5) for i in range(len(visible))])
            else:
                # Default to V magnitude
                data_vals = mag
            
            # Linear mapping from data range to size range
            # Normalize data values to 0-1 range
            data_range = data_max - data_min
            if abs(data_range) > 0.0001:
                normalized = (data_vals - data_min) / data_range
            else:
                normalized = np.full(len(data_vals), 0.5)
            
            # Clip to 0-1 range
            normalized = np.clip(normalized, 0, 1)
            
            # Apply invert if requested
            if invert:
                normalized = 1 - normalized
            
            # Map to size range
            size_range = size_max - size_min
            sizes = size_min + normalized * size_range
            
            # Final clipping
            sizes = np.clip(sizes, 1, 500)
        else:
            sizes = np.array([])
        
        # Determine which NEOs are on far side of Sun
        # Filled circles: NEO closer to Earth than to Sun (Earth-NEO < Sun-NEO)
        # Hollow circles: NEO closer to Sun than to Earth (Sun-NEO < Earth-NEO)
        far_side_mask = np.zeros(len(mag), dtype=bool)
        
        if hasattr(self, 'sun_ra') and hasattr(self, 'sun_dec') and hasattr(self, 'sun_dist'):
            # Calculate angular separation between Sun and each NEO
            # Use original RA/Dec (before coordinate filtering)
            # Need to track which rows survived filtering
            ra_original = visible[:, 1]
            dec_original = visible[:, 2]
            
            # Angular separation formula (great circle distance)
            ra1 = np.radians(ra_original)
            dec1 = np.radians(dec_original)
            ra2 = np.radians(self.sun_ra)
            dec2 = np.radians(self.sun_dec)
            
            cos_sep = (np.sin(dec1) * np.sin(dec2) + 
                      np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2))
            # Clamp to avoid numerical errors
            cos_sep = np.clip(cos_sep, -1, 1)
            
            # Law of cosines: c² = a² + b² - 2ab*cos(C)
            # where a = Earth-NEO distance, b = Earth-Sun distance, C = angular separation
            # and c = Sun-NEO distance
            dist_earth_neo = dist
            dist_earth_sun = self.sun_dist
            
            dist_sun_neo_sq = (dist_earth_neo**2 + dist_earth_sun**2 - 
                              2 * dist_earth_neo * dist_earth_sun * cos_sep)
            dist_sun_neo = np.sqrt(np.abs(dist_sun_neo_sq))
            
            # Hollow if Sun-NEO < Earth-NEO (NEO closer to Sun)
            # Filled if Earth-NEO < Sun-NEO (NEO closer to Earth)
            far_side_mask = dist_sun_neo < dist_earth_neo
        
        # Split into near and far side for different markers
        near_offsets = offsets[~far_side_mask]
        near_sizes = sizes[~far_side_mask]
        near_mag = mag[~far_side_mask]
        near_visible = visible[~far_side_mask]  # Keep full data for click lookup
        
        far_offsets = offsets[far_side_mask]
        far_sizes = sizes[far_side_mask]
        far_mag = mag[far_side_mask]
        far_visible = visible[far_side_mask]  # Keep full data for click lookup
        
        # Store for click-to-identify (combine near and far)
        if show_hollow:
            self.plot_offsets = np.vstack([near_offsets, far_offsets]) if len(far_offsets) > 0 else near_offsets
            self.visible_data = np.vstack([near_visible, far_visible]) if len(far_visible) > 0 else near_visible
        else:
            self.plot_offsets = near_offsets
            self.visible_data = near_visible
        
        # Store near-side data separately for trails (don't trail objects behind sun)
        self.near_offsets = near_offsets
        self.near_visible = near_visible
        
        # Handle display mode
        display_mode = getattr(self, 'display_mode', 'points')
        
        # Force points mode during animation (density/contours too slow)
        if getattr(self, 'animation_playing', False) and display_mode != 'points':
            display_mode = 'points'
            # Clear any density elements from previous frame
            self._clear_density_elements()
        
        show_points = display_mode in ['points', 'points+contours']
        show_density = display_mode == 'density'
        show_contours = display_mode in ['contours', 'points+contours']
        
        # Clear density elements when not in density mode
        if not show_density:
            self._clear_density_elements()
            # Restore colorbar to magnitude mode when not showing density
            if hasattr(self, 'cbar') and self.cbar is not None:
                self.cbar.update_normal(self.scatter)
                self.cbar.set_label('Visual Magnitude')
                # Ensure consistent layout
                self.fig.subplots_adjust(left=0.02, right=0.95, top=0.97, bottom=0.04)
        
        # Draw density map or contours
        if show_density and len(self.plot_offsets) > 2:
            self._draw_density_map(self.plot_offsets)
        
        if show_contours and len(self.plot_offsets) > 9:
            self._draw_contours(self.plot_offsets)
        
        # If only showing density/contours (not points), hide scatter plots
        if not show_points:
            self.scatter.set_offsets(np.empty((0, 2)))
            self.scatter.set_sizes([])
            self.scatter_far.set_offsets(np.empty((0, 2)))
            self.scatter_far.set_sizes([])
            self._clear_cneos_legend()
        else:
            # Determine coloring mode
            use_cneos_colors = (color_by == 'CNEOS Discovery Site') and asteroids is not None
            
            if use_cneos_colors:
                # Get colors based on discovery site for near-side objects
                near_colors = []
                for i in range(len(near_visible)):
                    ast_idx = int(near_visible[i, 0])
                    if ast_idx < len(asteroids):
                        site = asteroids[ast_idx].get('discovery_site', '')
                        site = str(site).strip() if site else ''
                        color = self.CNEOS_SITE_COLORS.get(site, self.CNEOS_DEFAULT_COLOR)
                    else:
                        color = self.CNEOS_DEFAULT_COLOR
                    near_colors.append(color)
                
                # Update main scatter with discrete colors
                self.scatter.set_offsets(near_offsets)
                self.scatter.set_sizes(near_sizes)
                self.scatter.set_array(None)  # Clear colormap array
                self.scatter.set_facecolor(near_colors)
                
                # Hide colorbar in CNEOS mode
                if hasattr(self, 'cbar') and self.cbar is not None:
                    self.cbar.ax.set_visible(False)
                
                # Draw legend
                if show_legend:
                    self._draw_cneos_legend()
                else:
                    self._clear_cneos_legend()
            else:
                # Standard V magnitude coloring
                self.scatter.set_offsets(near_offsets)
                self.scatter.set_sizes(near_sizes)
                self.scatter.set_array(near_mag)
                
                # Show colorbar in magnitude mode (but not if showing density)
                if hasattr(self, 'cbar') and self.cbar is not None and not show_density:
                    self.cbar.ax.set_visible(True)
                
                # Clear legend
                self._clear_cneos_legend()
            
            # Update far side scatter (hollow circles) - only if show_hollow is True
            if show_hollow:
                # For far side, we need to color the edges
                if len(far_mag) > 0:
                    if use_cneos_colors:
                        # Get colors based on discovery site for far-side objects
                        far_colors = []
                        for i in range(len(far_visible)):
                            ast_idx = int(far_visible[i, 0])
                            if ast_idx < len(asteroids):
                                site = asteroids[ast_idx].get('discovery_site', '')
                                site = str(site).strip() if site else ''
                                color = self.CNEOS_SITE_COLORS.get(site, self.CNEOS_DEFAULT_COLOR)
                            else:
                                color = self.CNEOS_DEFAULT_COLOR
                            far_colors.append(color)
                    else:
                        # Magnitude-based colors
                        import matplotlib
                        norm = plt.Normalize(vmin=self.cbar_min, vmax=self.cbar_max)
                        cmap_obj = matplotlib.colormaps.get_cmap(self.cmap)
                        far_colors = cmap_obj(norm(far_mag))
                else:
                    far_colors = []
                
                self.scatter_far.set_offsets(far_offsets)
                self.scatter_far.set_sizes(far_sizes)
                self.scatter_far.set_edgecolors(far_colors)
            else:
                # Hide hollow symbols by clearing them
                self.scatter_far.set_offsets(np.empty((0, 2)))
                self.scatter_far.set_sizes([])
                self.scatter_far.set_edgecolors([])
        
        # Enhanced Stats
        if len(mag) > 0:
            # Date and time
            ts = skyfield_load.timescale()
            t = ts.tt_jd(jd) if jd is not None else ts.now()
            dt = t.utc_datetime()
            
            # Calendar date (top line, bold) - format: "13 Jan 2026, Tuesday"
            calendar_str = dt.strftime('%d %b %Y, %A')
            if self.calendar_text:
                self.calendar_text.set_text(calendar_str)
                self.calendar_text.set_position((0.02, 0.98))
            
            # UTC date/time string
            date_str = dt.strftime('%Y-%m-%d %H:%M UTC')
            
            # Catalina Lunation Number
            current_jd = jd if jd is not None else t.tt
            try:
                cln, cln_days = jd_to_cln(current_jd)
                # Format: "CLN X, Y days past Full" with integer days (rounded down)
                days_past_full = int(cln_days)  # Round down
                # Add "~" if outside precise ephemeris range (approximate calculation)
                if current_jd < EPHEMERIS_MIN_JD or current_jd > EPHEMERIS_MAX_JD:
                    cln_str = f'CLN ~{cln}, ~{days_past_full} days past Full'
                else:
                    cln_str = f'CLN {cln}, {days_past_full} days past Full'
            except Exception:
                cln_str = 'CLN unavailable'
            
            # Moon phase (displayed below calendar date in same container style)
            phase_name, phase_color = get_moon_phase_name(current_jd)
            
            # Update phase text element below calendar date with appropriate color
            if self.phase_text:
                self.phase_text.set_text(phase_name)
                self.phase_text.set_color(phase_color)
                self.phase_text.set_position((0.02, 0.945))
            
            # NEO counts (filled + hollow if showing hollow)
            n_near = len(near_offsets)
            n_far = len(far_offsets)
            if show_hollow:
                neos_str = f'NEOs: {n_near} + {n_far} behind sun'
            else:
                neos_str = f'NEOs: {n_near}'
            
            # V magnitude range (aligned with fixed width)
            v_str = f' {mag_min:4.1f} ≤ V ≤ {mag_max:4.1f}'
            
            # H magnitude range (aligned with fixed width)
            h_str = f' {h_min:4.1f} ≤ H ≤ {h_max:4.1f}' if h_min is not None and h_max is not None else ''
            
            # NEO classes (singular forms)
            if selected_classes:
                class_names = []
                has_amor_near = 'Amor, q≤1.15' in selected_classes
                has_amor_far = 'Amor, q>1.15' in selected_classes
                
                for cls in selected_classes:
                    if cls == 'Atira':
                        class_names.append('Atira')
                    elif cls == 'Aten':
                        class_names.append('Aten')
                    elif cls == 'Apollo':
                        class_names.append('Apollo')
                
                # Handle Amors specially (singular)
                if has_amor_near and has_amor_far:
                    class_names.append('Amor')
                elif has_amor_near:
                    class_names.append('Amor (q ≤ 1.15)')
                elif has_amor_far:
                    class_names.append('Amor (q > 1.15)')
                
                classes_str = ' '.join(class_names)
            else:
                classes_str = 'All classes'
            
            # MOID filter (if enabled)
            moid_str = ''
            if moid_enabled and moid_min is not None and moid_max is not None:
                moid_str = f'{moid_min:.3f} ≤ MOID ≤ {moid_max:.3f} AU'
            
            # Orbital element filters (if non-default)
            per_str = ''
            ecc_str = ''
            inc_str = ''
            if orb_filters:
                if orb_filters.get('period_enabled'):
                    per_str = f"Per: {orb_filters['period_min']:.2f}-{orb_filters['period_max']:.2f} yr"
                if orb_filters.get('ecc_enabled'):
                    ecc_str = f"Ecc: {orb_filters['ecc_min']:.3f}-{orb_filters['ecc_max']:.3f}"
                if orb_filters.get('inc_enabled'):
                    inc_str = f"Inc: {orb_filters['inc_min']:.1f}°-{orb_filters['inc_max']:.1f}°"
            
            # Build stats text (calendar date and phase are separate, above)
            stats_lines = [date_str, cln_str, neos_str, v_str]
            if h_str:
                stats_lines.append(h_str)
            stats_lines.append(classes_str)
            if moid_str:
                stats_lines.append(moid_str)
            if per_str:
                stats_lines.append(per_str)
            if ecc_str:
                stats_lines.append(ecc_str)
            if inc_str:
                stats_lines.append(inc_str)
            
            stats = '\n'.join(stats_lines)
            # Position stats text below the calendar date and phase
            self.stats_text.set_position((0.02, 0.91))
        else:
            if self.calendar_text:
                self.calendar_text.set_text('')
            if self.phase_text:
                self.phase_text.set_text('')
            stats = 'No objects (filtered)'
            self.stats_text.set_position((0.02, 0.98))
        
        self.stats_text.set_text(stats)
        self.stats_text.set_fontsize(9)  # Slightly larger for readability
        self.stats_text.set_family('monospace')  # Use monospace for alignment
        
        # Update trails if enabled (use near-side objects only)
        if self.trailing_settings.get('enabled', False) and asteroids is not None:
            # Use near-side data only (don't trail objects behind sun)
            trail_positions = getattr(self, 'near_visible', None)
            trail_offsets = getattr(self, 'near_offsets', None)
            if trail_positions is not None and trail_offsets is not None and len(trail_positions) > 0:
                self._update_trails(asteroids, trail_positions, trail_offsets, jd)
            else:
                # No visible objects - clear BOTH trail lines AND history
                self._clear_trails()
                self.trail_history.clear()
                logger.debug("TRAIL: No visible objects - cleared all trails")
        elif self.trailing_settings.get('enabled', False):
            # Trailing enabled but no asteroids data - clear trails
            self._clear_trails()
            self.trail_history.clear()
            logger.debug("TRAIL: No asteroids data - cleared all trails")
        
        # Ensure consistent layout before drawing
        self.fig.subplots_adjust(left=0.02, right=0.95, top=0.97, bottom=0.04)

        if _profile:
            _times['bindraw'] = time.time() - _t0

        # FPS tracking during animation (enabled with --fps flag)
        if self._show_fps and getattr(self, 'animation_playing', False):
            now = time.time()
            self._frame_times.append(now)
            self._frame_times = self._frame_times[-30:]  # Keep last 30 frames
            if len(self._frame_times) > 1 and now - self._last_fps_print > 1.0:
                fps = (len(self._frame_times) - 1) / (self._frame_times[-1] - self._frame_times[0])
                print(f"FPS: {fps:.1f}")
                self._last_fps_print = now

        # Use blitting for faster animation updates
        if self._use_blitting and getattr(self, 'animation_playing', False):
            # Ensure trail lines are marked as animated
            for line in self.trail_lines:
                if line not in self._animated_artists:
                    line.set_animated(True)
                    self._animated_artists.append(line)
            self.blit_update()
        else:
            self.draw()  # Force immediate redraw

        if _profile:
            _times['draw'] = time.time() - _t0
            # Print breakdown
            overlays = _times.get('overlays', 0) * 1000
            magf = (_times.get('magfilter', 0) - _times.get('overlays', 0)) * 1000
            sitef = (_times.get('sitefilter', 0) - _times.get('magfilter', 0)) * 1000
            bindraw = (_times.get('bindraw', 0) - _times.get('sitefilter', 0)) * 1000
            draw = (_times.get('draw', 0) - _times.get('bindraw', 0)) * 1000
            print(f"    overlays:{overlays:.0f} mag:{magf:.0f} site:{sitef:.0f} bindraw:{bindraw:.0f} draw:{draw:.0f}ms")

    def on_stats_pick(self, event):
        """Handle pick event on stats/calendar text - toggle stats visibility"""
        if event.artist == self.stats_text and self.stats_visible:
            # Click on stats - dismiss
            self.stats_visible = False
            self.stats_text.set_visible(False)
            self.draw()
        elif event.artist == self.calendar_text and not self.stats_visible:
            # Click on calendar when stats hidden - restore
            self.stats_visible = True
            self.stats_text.set_visible(True)
            self.draw()
    
    def toggle_stats_visibility(self):
        """Toggle stats box visibility"""
        self.stats_visible = not self.stats_visible
        self.stats_text.set_visible(self.stats_visible)
        self.draw()
    
    def on_click(self, event):
        """Handle mouse click to identify NEO under cursor"""
        # Don't process clicks while animation is playing
        if getattr(self, 'animation_playing', False):
            return
        
        # If in selection mode, let the selector handle it
        if self.selection_mode is not None:
            return
        
        # Only process clicks inside the plot area
        if event.inaxes != self.ax:
            return
        
        # Check if we have data to search
        if self.plot_offsets is None or len(self.plot_offsets) == 0:
            return
        if self.current_asteroids is None:
            return
        
        # Get click coordinates
        click_x, click_y = event.xdata, event.ydata
        if click_x is None or click_y is None:
            return
        
        # Find nearest point
        # Calculate distances in data coordinates
        offsets = self.plot_offsets
        distances = np.sqrt((offsets[:, 0] - click_x)**2 + (offsets[:, 1] - click_y)**2)
        nearest_idx = np.argmin(distances)
        min_distance = distances[nearest_idx]
        
        # Check if click is close enough (threshold depends on projection)
        if self.projection in ['hammer', 'aitoff', 'mollweide']:
            threshold = 0.1  # radians - about 5.7 degrees
        else:
            threshold = 10  # degrees for rectangular
        
        if min_distance > threshold:
            return  # Click too far from any point
        
        # Get the asteroid data
        visible_row = self.visible_data[nearest_idx]
        ast_idx = int(visible_row[0])  # Column 0 is the index
        
        if ast_idx < 0 or ast_idx >= len(self.current_asteroids):
            return
        
        asteroid = self.current_asteroids[ast_idx]
        
        # Current position data from visible_row
        current_ra = visible_row[1]
        current_dec = visible_row[2]
        current_dist = visible_row[3]
        current_mag = visible_row[4]
        
        # Close existing dialog first (without triggering highlight clear)
        if hasattr(self, 'current_info_dialog') and self.current_info_dialog is not None:
            try:
                # Temporarily disconnect canvas to prevent highlight clear
                self.current_info_dialog.canvas = None
                self.current_info_dialog.close()
            except:
                pass
        
        # Highlight the selected object
        plot_pos = offsets[nearest_idx]
        self.highlight_object(plot_pos[0], plot_pos[1])
        
        # Force immediate GUI update to show highlight
        QApplication.processEvents()
        
        # Get mag_max and calculator from parent window
        parent_window = self.parent()
        mag_max = None
        calculator = None
        if parent_window and hasattr(parent_window, 'magnitude_panel'):
            _, mag_max = parent_window.magnitude_panel.get_magnitude_limits()
        if parent_window and hasattr(parent_window, 'calculator'):
            calculator = parent_window.calculator
        
        # Calculate screen position from canvas event coordinates
        # Convert matplotlib canvas coords to global screen coords
        click_screen_pos = None
        try:
            canvas_widget = self.fig.canvas
            # Get global position of canvas
            global_pos = canvas_widget.mapToGlobal(canvas_widget.rect().topLeft())
            # event.x and event.y are in canvas pixel coordinates (origin at bottom-left)
            # Convert to screen coordinates
            click_screen_x = global_pos.x() + event.x
            click_screen_y = global_pos.y() + (canvas_widget.height() - event.y)  # Flip Y
            click_screen_pos = (click_screen_x, click_screen_y)
        except:
            pass
        
        # Show info dialog (non-blocking, passing canvas reference)
        self.current_info_dialog = NEOInfoDialog(asteroid, current_ra, current_dec, current_dist, 
                              current_mag, self.current_jd, self.parent(), canvas=self,
                              mag_max=mag_max, calculator=calculator, click_screen_pos=click_screen_pos)
        self.current_info_dialog.show()
    
    def highlight_object(self, x, y):
        """Highlight an object at the given plot coordinates"""
        if hasattr(self, 'scatter_highlight'):
            self.scatter_highlight.set_offsets([[x, y]])
            # Use multiple draw methods for cross-platform reliability
            self.draw()
            try:
                self.fig.canvas.flush_events()
            except:
                pass
            self.update()  # Qt widget update
    
    def clear_highlight(self):
        """Clear the highlight marker"""
        if hasattr(self, 'scatter_highlight'):
            self.scatter_highlight.set_offsets(np.empty((0, 2)))
            self.draw()
            self.update()
    
    def _draw_cneos_legend(self):
        """Draw CNEOS discovery site legend"""
        # Remove existing legend if any
        self._clear_cneos_legend()
        
        # Create legend handles
        from matplotlib.lines import Line2D
        handles = []
        labels = []
        for name, color in self.CNEOS_LEGEND_ENTRIES:
            handle = Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                          markersize=8, linestyle='None')
            handles.append(handle)
            labels.append(name)
        
        # Add legend to plot - lower right since colorbar is hidden in CNEOS mode
        self.cneos_legend = self.ax.legend(handles, labels, loc='lower right',
                                           fontsize=7, framealpha=0.8,
                                           title='Discovery Site', title_fontsize=8,
                                           borderpad=0.3, labelspacing=0.2,
                                           handletextpad=0.3)
    
    def _clear_cneos_legend(self):
        """Remove CNEOS legend from plot"""
        if hasattr(self, 'cneos_legend') and self.cneos_legend is not None:
            try:
                self.cneos_legend.remove()
            except:
                pass
            self.cneos_legend = None
        # Also clear any other legends
        legend = self.ax.get_legend()
        if legend is not None:
            legend.remove()
    
    def show_neo_popup(self, asteroid, ra, dec, dist, mag, jd):
        """Show NEO info popup for a given asteroid"""
        # Close any existing dialog
        if hasattr(self, 'current_info_dialog') and self.current_info_dialog is not None:
            try:
                self.current_info_dialog.canvas = None
                self.current_info_dialog.close()
            except:
                pass
        
        # Get mag_max and calculator from parent
        mag_max = None
        calculator = None
        if self.parent() and hasattr(self.parent(), 'magnitude_panel'):
            mag_max = self.parent().magnitude_panel.mag_max_spin.value()
        if self.parent() and hasattr(self.parent(), 'calculator'):
            calculator = self.parent().calculator
        
        # Create and show dialog
        self.current_info_dialog = NEOInfoDialog(asteroid, ra, dec, dist, 
                              mag, jd, self.parent(), canvas=self,
                              mag_max=mag_max, calculator=calculator, click_screen_pos=None)
        self.current_info_dialog.show()
    
    def find_and_highlight_designation(self, designation):
        """
        Search for an object by designation and highlight it if visible.
        Searches by: packed designation, unpacked designation, readable_designation (name),
        and asteroid number.
        Returns (asteroid_dict, ra, dec, dist, mag, visible_on_plot) or (None, ...) if not found.
        """
        from designation_utils import pack_designation, unpack_designation
        
        if self.current_asteroids is None:
            return None, None, None, None, None, False
        
        # Normalize the input designation
        search_desig = designation.strip()
        search_lower = search_desig.lower()
        search_nospace = search_desig.replace(' ', '')
        
        # Try multiple matching strategies
        found_ast = None
        found_idx = None
        
        for idx, ast in enumerate(self.current_asteroids):
            packed = ast['designation']
            
            # 1. Direct match with packed designation
            if packed == search_desig:
                found_ast = ast
                found_idx = idx
                break
            
            # 2. Match readable_designation (name like "Apophis", "Eros", or "2024 AA")
            readable = ast.get('readable_designation', '').strip()
            if readable:
                # Case-insensitive match
                if readable.lower() == search_lower:
                    found_ast = ast
                    found_idx = idx
                    break
                # Match without spaces
                if readable.replace(' ', '').lower() == search_lower.replace(' ', ''):
                    found_ast = ast
                    found_idx = idx
                    break
                # Match if search is contained in readable (e.g., "Eros" in "433 Eros")
                if search_lower in readable.lower():
                    found_ast = ast
                    found_idx = idx
                    break
            
            # 3. Try unpacking the database designation and comparing
            try:
                unpacked = unpack_designation(packed)
                if unpacked == search_desig or unpacked.replace(' ', '') == search_nospace:
                    found_ast = ast
                    found_idx = idx
                    break
            except:
                pass
            
            # 4. Try packing the search string and comparing
            try:
                packed_search = pack_designation(search_desig)
                if packed == packed_search:
                    found_ast = ast
                    found_idx = idx
                    break
            except:
                pass
            
            # 5. For numbered asteroids, try matching number in various forms
            if search_desig.isdigit():
                search_num = search_desig.lstrip('0') or '0'
                # Check packed - numbered asteroids have right-justified numbers
                packed_num = packed.lstrip('0').strip()
                if packed_num == search_num:
                    found_ast = ast
                    found_idx = idx
                    break
                # Check readable designation for the number
                if readable:
                    # Match "99942 Apophis" style
                    if readable.split()[0] == search_num:
                        found_ast = ast
                        found_idx = idx
                        break
        
        if found_ast is None:
            return None, None, None, None, None, False
        
        # Found the asteroid - now check if it's currently visible on the plot
        visible_on_plot = False
        plot_x, plot_y = None, None
        current_ra, current_dec, current_dist, current_mag = None, None, None, None
        
        if self.visible_data is not None and len(self.visible_data) > 0:
            # Search for this asteroid in visible_data
            for i, row in enumerate(self.visible_data):
                if int(row[0]) == found_idx:
                    visible_on_plot = True
                    current_ra = row[1]
                    current_dec = row[2]
                    current_dist = row[3]
                    current_mag = row[4]
                    
                    # Get plot position
                    if self.plot_offsets is not None and i < len(self.plot_offsets):
                        plot_x, plot_y = self.plot_offsets[i]
                        self.highlight_object(plot_x, plot_y)
                    break
        
        # If not visible, try to calculate current position from all positions
        if not visible_on_plot and self.current_positions is not None:
            for row in self.current_positions:
                if int(row[0]) == found_idx:
                    current_ra = row[1]
                    current_dec = row[2]
                    current_dist = row[3]
                    current_mag = row[4]
                    break
        
        return found_ast, current_ra, current_dec, current_dist, current_mag, visible_on_plot
    
    def start_selection(self, mode='rectangle'):
        """Start selection mode for lasso selection of objects"""
        from matplotlib.widgets import RectangleSelector, EllipseSelector
        
        # Clear any existing selector first
        self.clear_selection()
        
        self.selection_mode = mode
        
        if mode == 'rectangle':
            self.selector = RectangleSelector(
                self.ax, self._on_select,
                useblit=True,
                button=[1],  # Left button only
                minspanx=5, minspany=5,
                spancoords='pixels',
                interactive=True,
                props=dict(facecolor='cyan', edgecolor='blue', alpha=0.3, linewidth=2)
            )
        elif mode == 'ellipse':
            self.selector = EllipseSelector(
                self.ax, self._on_select,
                useblit=True,
                button=[1],
                minspanx=5, minspany=5,
                spancoords='pixels',
                interactive=True,
                props=dict(facecolor='cyan', edgecolor='blue', alpha=0.3, linewidth=2)
            )
        
        self.draw_idle()
        return True
    
    def _on_select(self, eclick, erelease):
        """Callback when selection is made"""
        if self.visible_data is None or len(self.visible_data) == 0:
            return
        
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        
        if x1 is None or x2 is None:
            return
        
        # Get selection bounds
        xmin, xmax = min(x1, x2), max(x1, x2)
        ymin, ymax = min(y1, y2), max(y1, y2)
        
        # Find objects within selection
        selected = []
        
        if self.selection_mode == 'rectangle':
            for i, offset in enumerate(self.plot_offsets):
                px, py = offset
                if xmin <= px <= xmax and ymin <= py <= ymax:
                    selected.append(i)
        elif self.selection_mode == 'ellipse':
            # Ellipse selection - check if point is within ellipse
            cx = (xmin + xmax) / 2
            cy = (ymin + ymax) / 2
            rx = (xmax - xmin) / 2
            ry = (ymax - ymin) / 2
            
            for i, offset in enumerate(self.plot_offsets):
                px, py = offset
                # Check if point is inside ellipse: ((x-cx)/rx)^2 + ((y-cy)/ry)^2 <= 1
                if rx > 0 and ry > 0:
                    if ((px - cx) / rx) ** 2 + ((py - cy) / ry) ** 2 <= 1:
                        selected.append(i)
        
        self.selected_indices = selected
        
        # Emit signal for table update
        if hasattr(self, 'selection_complete') and self.selection_complete:
            self.selection_complete(selected)
    
    def get_selected_data(self):
        """Get visible_data rows for selected objects"""
        if self.selected_indices is None or self.visible_data is None:
            return None
        
        return self.visible_data[self.selected_indices]
    
    def clear_selection(self):
        """Clear selection mode and selector"""
        if self.selector is not None:
            self.selector.set_active(False)
            # Remove the selector's artists from the axes
            try:
                for artist in self.selector.artists:
                    artist.remove()
            except:
                pass
            self.selector = None
        self.selection_mode = None
        self.selected_indices = None
        self.draw_idle()


class NEOInfoDialog(QDialog):
    """Dialog showing detailed information about a clicked NEO"""
    
    # Class variable to remember last position
    last_position = None
    
    def __init__(self, asteroid, ra, dec, dist, mag, jd, parent=None, canvas=None, 
                 mag_max=None, calculator=None, click_screen_pos=None):
        super().__init__(parent)
        self.canvas = canvas  # Reference to canvas for clearing highlight
        self.asteroid = asteroid
        self.mag_max = mag_max
        self.calculator = calculator
        self.current_jd = jd
        self.click_screen_pos = click_screen_pos  # Store click position
        
        # Use readable_designation if available for window title
        readable = asteroid.get('readable_designation', '').strip()
        title_name = readable if readable else asteroid['designation']
        self.setWindowTitle(f"NEO: {title_name}")
        
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(5)
        
        # Create text display with copy support
        self.text = QTextBrowser()
        self.text.setOpenExternalLinks(True)
        # Enable text selection for copy-paste
        self.text.setTextInteractionFlags(
            QtCompat.TextSelectableByMouse | 
            QtCompat.TextSelectableByKeyboard |
            QtCompat.LinksAccessibleByMouse
        )
        
        # Format the information
        info = self._format_info(asteroid, ra, dec, dist, mag, jd)
        self.text.setHtml(info)
        
        # Size text browser to content - calculate ideal size
        self.text.document().setTextWidth(340)
        doc_height = int(self.text.document().size().height())
        doc_width = int(self.text.document().idealWidth())
        
        # Set fixed height based on content (no extra space)
        self.text.setFixedHeight(min(doc_height + 10, 600))
        self.text.setMinimumWidth(max(doc_width + 20, 340))
        
        layout.addWidget(self.text)
        
        # Close button (set as default for blue tint on macOS)
        close_btn = QPushButton("Close")
        close_btn.setDefault(True)
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)
        
        self.setLayout(layout)
        self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        
        # Adjust size to fit content tightly
        self.adjustSize()
        
        # Store position to apply in showEvent (for macOS reliability)
        # Priority: 1) User's last moved position, 2) Calculate from click to avoid occlusion
        self._pending_position = None
        if NEOInfoDialog.last_position is not None:
            # User has moved a previous dialog - use that position
            self._pending_position = NEOInfoDialog.last_position
        elif click_screen_pos is not None:
            # First time or no user movement - position away from click
            self._pending_position = self._calc_position_from_click(click_screen_pos)
        else:
            # Fallback - position at top-left with offset
            self._pending_position = self._get_fallback_position()
    
    def _get_fallback_position(self):
        """Get a fallback position when click position is unknown"""
        try:
            from PyQt6.QtCore import QPoint
        except ImportError:
            from PyQt5.QtCore import QPoint
        
        screen = QApplication.primaryScreen()
        if screen:
            screen_rect = screen.availableGeometry()
            # Position at top-right by default
            dialog_width = self.sizeHint().width() or 360
            return QPoint(screen_rect.right() - dialog_width - 50, screen_rect.top() + 50)
        return QPoint(100, 100)
    
    def _calc_position_from_click(self, click_pos):
        """Calculate dialog position to avoid occluding clicked NEO"""
        try:
            from PyQt6.QtCore import QPoint
        except ImportError:
            from PyQt5.QtCore import QPoint
        
        click_x, click_y = click_pos
        dialog_width = self.sizeHint().width() or 360
        dialog_height = self.sizeHint().height() or 600
        
        # Get screen geometry
        screen = QApplication.primaryScreen()
        if screen:
            screen_rect = screen.availableGeometry()
            
            # Always position dialog on opposite side from click
            # Use screen center as dividing line
            center_x = screen_rect.center().x()
            
            if click_x > center_x:
                # Click on right half - put dialog on left side with margin from edge
                new_x = screen_rect.left() + 30
            else:
                # Click on left half - put dialog on right side
                new_x = screen_rect.right() - dialog_width - 30
            
            # Vertical: position near top of screen but not at edge
            new_y = screen_rect.top() + 50
            
            return QPoint(int(new_x), int(new_y))
        return None
    
    def showEvent(self, event):
        """Apply position after dialog is shown (needed for macOS)"""
        super().showEvent(event)
        if self._pending_position is not None:
            self.move(self._pending_position)
    
    def moveEvent(self, event):
        """Track when user moves the dialog"""
        super().moveEvent(event)
        # Update last_position when dialog is moved (after initial positioning)
        if self.isVisible():
            NEOInfoDialog.last_position = self.pos()
    
    def closeEvent(self, event):
        """Remember position and clear highlight when closing"""
        NEOInfoDialog.last_position = self.pos()
        if self.canvas:
            self.canvas.clear_highlight()
        super().closeEvent(event)
    
    def _is_behind_sun(self, ra, dec, dist, jd):
        """Check if object is on far side of Sun (heliocentric distance > geocentric)"""
        try:
            ts = skyfield_load.timescale()
            t = ts.tt_jd(jd)
            
            eph = skyfield_load('de421.bsp')
            earth = eph['earth']
            sun = eph['sun']
            
            sun_astrometric = earth.at(t).observe(sun)
            sun_ra_deg, sun_dec_deg, sun_dist_au = sun_astrometric.radec()
            sun_ra = sun_ra_deg._degrees
            sun_dec = sun_dec_deg.degrees
            sun_dist = sun_dist_au.au
            
            # Angular separation
            ra1 = np.radians(ra)
            dec1 = np.radians(dec)
            ra2 = np.radians(sun_ra)
            dec2 = np.radians(sun_dec)
            
            cos_sep = (np.sin(dec1) * np.sin(dec2) + 
                      np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2))
            cos_sep = np.clip(cos_sep, -1, 1)
            
            # Heliocentric distance via law of cosines
            helio_dist_sq = dist**2 + sun_dist**2 - 2 * dist * sun_dist * cos_sep
            helio_dist = np.sqrt(abs(helio_dist_sq))
            
            return helio_dist < dist  # Behind sun if heliocentric < geocentric (closer to Sun than Earth)
        except:
            return False
    
    def _check_visibility_at_jd(self, jd):
        """
        Check if asteroid is visible at given JD.
        Returns (is_visible, mag) or (None, None) if calculation fails.
        """
        if self.calculator is None or self.mag_max is None:
            return None, None
        
        try:
            # Calculate position at this JD
            positions = self.calculator.calculate_batch([self.asteroid], jd)
            if positions is None or len(positions) == 0:
                return None, None
            
            ra, dec, dist, mag_at_jd = positions[0, 1], positions[0, 2], positions[0, 3], positions[0, 4]
            
            # Check magnitude
            if mag_at_jd >= self.mag_max:
                return False, mag_at_jd
            
            # Check if behind sun
            if self._is_behind_sun(ra, dec, dist, jd):
                return False, mag_at_jd
            
            return True, mag_at_jd
        except:
            return None, None
    
    def _find_visibility_transitions(self, current_jd, is_currently_visible):
        """
        Find last visible and next visible dates.
        - Last visible: End of the PREVIOUS visibility period (when it faded)
        - Next visible: Start of the NEXT visibility period (when it brightens)
        Excludes any current visibility period.
        Returns (last_visible_str, next_visible_str)
        """
        if self.calculator is None or self.mag_max is None:
            return "Unknown", "Unknown"
        
        # Ephemeris range limits (de421.bsp: 1900-2050)
        MIN_JD = 2415020.5
        MAX_JD = 2469807.5
        COARSE_STEP = 30  # days
        
        ts = skyfield_load.timescale()
        
        def binary_search_fade(jd_visible, jd_invisible):
            """Find last day visible before fading. jd_visible < jd_invisible."""
            while jd_invisible - jd_visible > 1.0:
                mid_jd = (jd_visible + jd_invisible) / 2
                vis, _ = self._check_visibility_at_jd(mid_jd)
                if vis is True:
                    jd_visible = mid_jd
                else:
                    jd_invisible = mid_jd
            return jd_visible  # Last visible day
        
        def binary_search_brighten(jd_invisible, jd_visible):
            """Find first day visible after brightening. jd_invisible < jd_visible."""
            while jd_visible - jd_invisible > 1.0:
                mid_jd = (jd_invisible + jd_visible) / 2
                vis, _ = self._check_visibility_at_jd(mid_jd)
                if vis is True:
                    jd_visible = mid_jd
                else:
                    jd_invisible = mid_jd
            return jd_visible  # First visible day
        
        def coarse_search(start_jd, direction, find_state):
            """
            Search in direction (+1 forward, -1 backward) for find_state (True=visible, False=invisible).
            Returns JD or None.
            """
            test_jd = start_jd + direction * COARSE_STEP
            limit = MAX_JD if direction > 0 else MIN_JD
            while (direction > 0 and test_jd <= limit) or (direction < 0 and test_jd >= limit):
                vis, _ = self._check_visibility_at_jd(test_jd)
                if vis is find_state:
                    return test_jd
                test_jd += direction * COARSE_STEP
            return None
        
        last_visible_str = "Unknown"
        next_visible_str = "Unknown"
        
        # =====================
        # LAST VISIBLE
        # =====================
        # Goal: Find end of PREVIOUS visibility period (excluding current)
        
        # Step 1: Get to an invisible point (skip current visibility if needed)
        if is_currently_visible:
            gap_jd = coarse_search(current_jd, -1, False)  # Search backward for invisible
        else:
            gap_jd = current_jd  # Already invisible
        
        if gap_jd is not None:
            # Step 2: From gap, find a visible point in previous period
            prev_vis_jd = coarse_search(gap_jd, -1, True)  # Search backward for visible
            
            if prev_vis_jd is not None:
                # Step 3: From that visible point, find END of that visible period
                # Search forward until we hit invisible
                end_invis_jd = coarse_search(prev_vis_jd, +1, False)
                
                if end_invis_jd is not None:
                    # Step 4: Binary search to find exact fade date
                    fade_jd = binary_search_fade(prev_vis_jd, end_invis_jd)
                    t = ts.tt_jd(fade_jd)
                    last_visible_str = t.utc_datetime().strftime('%Y-%m-%d')
                else:
                    # Visible forward to limit? Shouldn't happen if gap_jd exists
                    last_visible_str = "Unknown"
            else:
                last_visible_str = "Before 1900"
        else:
            # No invisible gap found backward - visible since 1900?
            last_visible_str = "Before 1900"
        
        # =====================
        # NEXT VISIBLE  
        # =====================
        # Goal: Find start of NEXT visibility period (excluding current)
        
        # Step 1: Get to an invisible point (skip current visibility if needed)
        if is_currently_visible:
            gap_jd = coarse_search(current_jd, +1, False)  # Search forward for invisible
            if gap_jd is None:
                next_visible_str = "Continuously visible"
                return last_visible_str, next_visible_str
        else:
            gap_jd = current_jd  # Already invisible
        
        # Step 2: From gap, find next visible point
        next_vis_jd = coarse_search(gap_jd, +1, True)  # Search forward for visible
        
        if next_vis_jd is not None:
            # Step 3: Binary search to find exact brighten date
            brighten_jd = binary_search_brighten(gap_jd, next_vis_jd)
            t = ts.tt_jd(brighten_jd)
            next_visible_str = t.utc_datetime().strftime('%Y-%m-%d')
        else:
            next_visible_str = "After 2050"
        
        return last_visible_str, next_visible_str
    
    def _format_discovery_date(self, discovery_mjd):
        """Format discovery date for display, including CLN"""
        if discovery_mjd is None:
            return '<span style="color: gray; font-style: italic;">Unknown</span>'
        
        ts = skyfield_load.timescale()
        discovery_jd = discovery_mjd + 2400000.5
        date = ts.tt_jd(discovery_jd).utc_datetime().strftime('%Y-%m-%d')
        
        # Get CLN for discovery date
        try:
            disc_cln, _ = jd_to_cln(discovery_jd)
            return f'{date} (CLN {disc_cln})'
        except:
            return date
    
    def _format_discovery_site(self, site):
        """Format discovery site with survey name"""
        if not site:
            return '<span style="color: gray; font-style: italic;">Unknown</span>'
        
        site = str(site).strip()
        
        # Site code to name mapping (based on CNEOS categories)
        site_names = {
            # LINEAR
            '704': 'LINEAR', 'G45': 'LINEAR', 'P07': 'LINEAR',
            # NEAT
            '566': 'NEAT', '608': 'NEAT', '644': 'NEAT',
            # Spacewatch
            '691': 'Spacewatch', '291': 'Spacewatch',
            # LONEOS
            '699': 'LONEOS',
            # Catalina
            '703': 'Mt. Lemmon (CSS)', 'G96': 'Mt. Lemmon (CSS)', 
            'E12': 'Siding Spring (CSS)', 'I52': 'Catalina', 'V06': 'Catalina',
            # Pan-STARRS
            'F51': 'Pan-STARRS 1', 'F52': 'Pan-STARRS 2',
            # NEOWISE
            'C51': 'NEOWISE',
            # ATLAS
            'T05': 'ATLAS-HKO', 'T08': 'ATLAS-MLO', 'W68': 'ATLAS',
            'M22': 'ATLAS-STH', 'T07': 'ATLAS', 'R17': 'ATLAS',
            # Other US
            'V00': 'Lunar & Planetary Lab', 'W84': 'Cerro Tololo',
            'I41': 'Palomar', 'U68': 'Steward', 'U74': 'Steward',
        }
        
        name = site_names.get(site, '')
        if name:
            return f'{site} ({name})'
        else:
            return site
    
    def _format_info(self, ast, ra, dec, dist, mag, jd):
        """Format asteroid info as HTML"""
        from designation_utils import unpack_designation
        
        # Calculate derived values
        a = ast['a']
        e = ast['e']
        i = ast['i']
        
        # Orbital period (years) from semi-major axis
        period = a ** 1.5
        
        # Perihelion and aphelion
        q = a * (1 - e)  # Perihelion distance
        Q_aph = a * (1 + e)  # Aphelion distance
        
        # Format RA as hours:minutes:seconds
        ra_hours = ra / 15.0
        ra_h = int(ra_hours)
        ra_m = int((ra_hours - ra_h) * 60)
        ra_s = ((ra_hours - ra_h) * 60 - ra_m) * 60
        ra_sexagesimal = f"{ra_h:02d}h {ra_m:02d}m {ra_s:05.2f}s"
        
        # Format Dec as degrees:arcmin:arcsec
        dec_sign = '+' if dec >= 0 else '-'
        dec_abs = abs(dec)
        dec_d = int(dec_abs)
        dec_m = int((dec_abs - dec_d) * 60)
        dec_s = ((dec_abs - dec_d) * 60 - dec_m) * 60
        dec_sexagesimal = f"{dec_sign}{dec_d:02d}° {dec_m:02d}' {dec_s:05.2f}\""
        
        # Calculate heliocentric distance
        helio_dist = None
        if jd:
            try:
                ts = skyfield_load.timescale()
                t = ts.tt_jd(jd)
                date_str = t.utc_datetime().strftime('%Y-%m-%d %H:%M')
                
                # Get Sun position from Earth
                eph = skyfield_load('de421.bsp')
                earth = eph['earth']
                sun = eph['sun']
                
                # Sun's geocentric position
                sun_astrometric = earth.at(t).observe(sun)
                sun_ra_deg, sun_dec_deg, sun_dist_au = sun_astrometric.radec()
                sun_ra = sun_ra_deg._degrees
                sun_dec = sun_dec_deg.degrees
                sun_dist = sun_dist_au.au
                
                # Calculate heliocentric distance using law of cosines
                # Angular separation between Sun and NEO
                ra1 = np.radians(ra)
                dec1 = np.radians(dec)
                ra2 = np.radians(sun_ra)
                dec2 = np.radians(sun_dec)
                
                cos_sep = (np.sin(dec1) * np.sin(dec2) + 
                          np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2))
                cos_sep = np.clip(cos_sep, -1, 1)
                
                # Law of cosines: helio² = geo² + sun² - 2*geo*sun*cos(sep)
                helio_dist_sq = (dist**2 + sun_dist**2 - 2 * dist * sun_dist * cos_sep)
                helio_dist = np.sqrt(abs(helio_dist_sq))
            except Exception as exc:
                import logging
                logging.debug(f"Error calculating heliocentric distance: {exc}")
                date_str = "Unknown"
                helio_dist = None
        else:
            date_str = "Unknown"
        
        # Unpack designation to full format
        packed_desig = ast['designation']
        try:
            full_desig = unpack_designation(packed_desig)
        except:
            full_desig = packed_desig
        
        # Use readable_designation from MPC if available (e.g., "Eros" or "2024 AA")
        readable_desig = ast.get('readable_designation', '').strip()
        if readable_desig:
            display_name = readable_desig
        else:
            display_name = full_desig
        
        # Orbit class
        orbit_class = ast.get('orbit_class', 'Unknown')
        
        # Earth MOID from database
        moid = ast.get('earth_moid')
        moid_str = f"{moid:.4f} AU" if moid else "Unknown"
        
        # PHA status - calculated from H ≤ 22 and MOID ≤ 0.05 AU
        h_mag = ast.get('H', 99)
        is_pha = (h_mag <= 22.0) and (moid is not None and moid <= 0.05)
        pha_str = '<span style="color: red; font-weight: bold;">Yes</span>' if is_pha else 'No'
        
        # Format heliocentric distance
        helio_str = f"{helio_dist:.4f} AU" if helio_dist else "Unknown"
        
        # Check galactic band and opposition circle status
        in_galactic_band = False
        near_opposition = False
        galactic_penalty = 0.0
        opposition_benefit = 0.0
        
        if self.canvas:
            galactic_settings = getattr(self.canvas, 'galactic_settings', None)
            opposition_settings = getattr(self.canvas, 'opposition_settings', None)
            
            # Check galactic band
            if galactic_settings and galactic_settings.get('enabled', False):
                offset = galactic_settings.get('offset', 15.0)
                galactic_penalty = galactic_settings.get('penalty', 2.0)
                _, gal_b = CoordinateTransformer.equatorial_to_galactic(ra, dec)
                in_galactic_band = abs(gal_b) < offset
            
            # Check opposition circle
            if opposition_settings and opposition_settings.get('enabled', False):
                radius = opposition_settings.get('radius', 5.0)
                opposition_benefit = opposition_settings.get('benefit', 2.0)
                if hasattr(self.canvas, 'sun_ra') and hasattr(self.canvas, 'sun_dec'):
                    opp_ra = (self.canvas.sun_ra + 180) % 360
                    opp_dec = -self.canvas.sun_dec
                    d_ra = np.radians(ra - opp_ra)
                    d_dec = np.radians(dec - opp_dec)
                    dec_rad = np.radians(dec)
                    opp_dec_rad = np.radians(opp_dec)
                    a = np.sin(d_dec/2)**2 + np.cos(dec_rad) * np.cos(opp_dec_rad) * np.sin(d_ra/2)**2
                    angular_sep = 2 * np.degrees(np.arcsin(np.sqrt(np.clip(a, 0, 1))))
                    near_opposition = angular_sep < radius
        
        # Calculate effective magnitude
        effective_mag = mag
        if in_galactic_band:
            effective_mag += galactic_penalty
        if near_opposition:
            effective_mag -= opposition_benefit
        
        # Check discovery date status
        discovery_mjd = ast.get('discovery_mjd')
        discovery_site = ast.get('discovery_site')
        is_before_discovery = False
        discovery_status_line = None
        
        if discovery_mjd is not None and jd is not None:
            current_mjd = jd - 2400000.5
            if current_mjd < discovery_mjd:
                is_before_discovery = True
                # Convert discovery MJD to date string
                ts = skyfield_load.timescale()
                discovery_jd = discovery_mjd + 2400000.5
                discovery_date = ts.tt_jd(discovery_jd).utc_datetime().strftime('%Y-%m-%d')
                discovery_status_line = f'<br><span style="color: #6666CC; font-size: 9pt;">Not yet discovered (disc. {discovery_date})</span>'
        elif discovery_mjd is None:
            discovery_status_line = '<br><span style="color: #999999; font-size: 9pt; font-style: italic;">Missing discovery tracklet data</span>'
        
        # Determine current visibility status
        is_behind_sun = self._is_behind_sun(ra, dec, dist, jd) if jd else False
        is_too_faint = (self.mag_max is not None and effective_mag >= self.mag_max)
        
        # Build status string with details
        if is_behind_sun:
            visibility_status = '<span style="color: orange;">Invisible (behind Sun)</span>'
            if discovery_status_line:
                visibility_status += discovery_status_line
        elif is_too_faint:
            status_parts = [f'<span style="color: gray;">Invisible (V<sub>eff</sub> ≥ {self.mag_max:.1f})</span>']
            if in_galactic_band:
                status_parts.append(f'<br><span style="color: #CC6666; font-size: 9pt;">MW penalty: +{galactic_penalty:.1f} mag</span>')
            if near_opposition:
                status_parts.append(f'<br><span style="color: #66AA66; font-size: 9pt;">Opp benefit: -{opposition_benefit:.1f} mag</span>')
            if discovery_status_line:
                status_parts.append(discovery_status_line)
            visibility_status = ''.join(status_parts)
        else:
            status_parts = ['<span style="color: green; font-weight: bold;">Visible</span>']
            if in_galactic_band:
                status_parts.append(f'<br><span style="color: #CC6666; font-size: 9pt;">MW penalty: +{galactic_penalty:.1f} mag</span>')
            if near_opposition:
                status_parts.append(f'<br><span style="color: #66AA66; font-size: 9pt;">Opp benefit: -{opposition_benefit:.1f} mag</span>')
            if discovery_status_line:
                status_parts.append(discovery_status_line)
            visibility_status = ''.join(status_parts)
        
        # Build HTML - use cellpadding=0 cellspacing=0 style to minimize spacing
        html = f"""
        <h2 style="margin-top: 0; margin-bottom: 5px;">{display_name}</h2>
        <p style="color: gray; margin-top: 0; margin-bottom: 4px;">Packed: {packed_desig} | Unpacked: {full_desig}</p>
        <p style="font-size: 10pt; margin-top: 0; margin-bottom: 8px;">
        <a href="https://neofixer.arizona.edu/site/500/{packed_desig}">CSS NEOfixer</a> |
        <a href="https://ssd.jpl.nasa.gov/tools/sbdb_lookup.html#/?sstr={packed_desig}">JPL SBDB</a>
        </p>
        
        <h3 style="margin-bottom: 3px;">Classification</h3>
        <table cellpadding="0" cellspacing="0" style="border-collapse: collapse;">
        <tr><td style="padding-right: 8px;"><b>Orbit Class:</b></td><td>{orbit_class}</td></tr>
        <tr><td style="padding-right: 8px;"><b>PHA:</b></td><td>{pha_str}</td></tr>
        <tr><td style="padding-right: 8px;"><b>Absolute mag (H):</b></td><td>{f"{ast['H']:.2f}" if ast['H'] is not None else 'Unknown'}</td></tr>
        <tr><td style="padding-right: 8px;"><b>Earth MOID:</b></td><td>{moid_str}</td></tr>
        <tr><td style="padding-right: 8px; vertical-align: top;"><b>Current status:</b></td><td>{visibility_status}</td></tr>
        </table>
        
        <h3 style="margin-bottom: 3px;">Current Position</h3>
        <table cellpadding="0" cellspacing="0" style="border-collapse: collapse;">
        <tr><td style="padding-right: 4px;"><b>UTC:</b></td><td>{date_str}</td></tr>
        <tr><td style="padding-right: 4px;"><b>RA:</b></td><td>{ra:.4f}° ({ra_sexagesimal})</td></tr>
        <tr><td style="padding-right: 4px;"><b>Dec:</b></td><td>{dec:+.4f}° ({dec_sexagesimal})</td></tr>
        </table>
        <table cellpadding="0" cellspacing="0" style="border-collapse: collapse; margin-top: 8px;">
        <tr><td style="padding-right: 8px;"><b>Geocentric dist (r):</b></td><td>{dist:.4f} AU</td></tr>
        <tr><td style="padding-right: 8px;"><b>Heliocentric dist (Δ):</b></td><td>{helio_str}</td></tr>
        <tr><td style="padding-right: 8px;"><b>Visual mag:</b></td><td>{mag:.2f}</td></tr>
        </table>
        
        <h3 style="margin-bottom: 3px;">Keplerian Elements</h3>
        <table cellpadding="0" cellspacing="0" style="border-collapse: collapse;">
        <tr><td style="padding-right: 8px;"><b>Semi-major axis (a):</b></td><td>{a:.6f} AU</td></tr>
        <tr><td style="padding-right: 8px;"><b>Eccentricity (e):</b></td><td>{e:.6f}</td></tr>
        <tr><td style="padding-right: 8px;"><b>Inclination (i):</b></td><td>{i:.4f}°</td></tr>
        </table>
        <table cellpadding="0" cellspacing="0" style="border-collapse: collapse; margin-top: 8px;">
        <tr><td style="padding-right: 8px;"><b>Long. of Asc. Node (Ω):</b></td><td>{ast['node']:.4f}°</td></tr>
        <tr><td style="padding-right: 8px;"><b>Arg. of Perihelion (ω):</b></td><td>{ast['arg_peri']:.4f}°</td></tr>
        <tr><td style="padding-right: 8px;"><b>Mean Anomaly (M):</b></td><td>{ast['M']:.4f}°</td></tr>
        </table>
        
        <h3 style="margin-bottom: 3px;">Other Orbital Elements</h3>
        <table cellpadding="0" cellspacing="0" style="border-collapse: collapse;">
        <tr><td style="padding-right: 8px;"><b>Period (P):</b></td><td>{period:.4f} years ({period * 365.25:.1f} days)</td></tr>
        <tr><td style="padding-right: 8px;"><b>Perihelion (q):</b></td><td>{q:.6f} AU</td></tr>
        <tr><td style="padding-right: 8px;"><b>Aphelion (Q):</b></td><td>{Q_aph:.6f} AU</td></tr>
        </table>
        
        <h3 style="margin-bottom: 3px;">Discovery</h3>
        <table cellpadding="0" cellspacing="0" style="border-collapse: collapse;">
        <tr><td style="padding-right: 8px;"><b>Date:</b></td><td>{self._format_discovery_date(discovery_mjd)}</td></tr>
        <tr><td style="padding-right: 8px;"><b>Site:</b></td><td>{self._format_discovery_site(discovery_site)}</td></tr>
        </table>
        """
        
        return html


class NEOTableDialog(QDialog):
    """Non-modal dialog displaying a table of currently visible NEOs"""
    
    # Class variable to remember last position/size
    last_geometry = None
    MAX_TABLE_ROWS = 5000  # Limit to prevent memory issues
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("NEO Table")
        self.setMinimumSize(900, 500)
        
        # Restore last geometry if available
        if NEOTableDialog.last_geometry:
            self.setGeometry(NEOTableDialog.last_geometry)
        
        # Make non-modal so user can interact with main window
        self.setModal(False)
        
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        
        # Filter row
        filter_layout = QHBoxLayout()
        filter_label = QLabel("Filter:")
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Type to filter by designation, name, class... (press Enter)")
        self.filter_input.returnPressed.connect(self.apply_filter)
        self.filter_input.setMaximumWidth(300)
        filter_layout.addWidget(filter_label)
        filter_layout.addWidget(self.filter_input)
        
        self.filter_btn = QPushButton("Apply")
        self.filter_btn.clicked.connect(self.apply_filter)
        filter_layout.addWidget(self.filter_btn)
        
        self.clear_filter_btn = QPushButton("Clear")
        self.clear_filter_btn.clicked.connect(self.clear_filter)
        filter_layout.addWidget(self.clear_filter_btn)
        
        filter_layout.addStretch()
        layout.addLayout(filter_layout)
        
        # Info/warning label
        self.info_label = QLabel("Showing 0 NEOs")
        self.info_label.setStyleSheet("font-weight: bold; margin-bottom: 5px;")
        layout.addWidget(self.info_label)
        
        # Warning label (hidden by default)
        self.warning_label = QLabel("")
        self.warning_label.setStyleSheet("color: #CC6600; font-style: italic;")
        self.warning_label.hide()
        layout.addWidget(self.warning_label)
        
        # Table widget
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.horizontalHeader().setStretchLastSection(True)
        
        # Connect row click
        self.table.cellClicked.connect(self.on_row_clicked)
        
        # Store mapping from row to asteroid index
        self.row_to_ast_idx = {}
        self.current_asteroids = None
        self.current_visible_data = None
        self.current_jd = None
        self.current_filter = ""
        
        # Define columns
        self.columns = [
            ('Designation', 120),
            ('Class', 70),
            ('H', 50),
            ('V mag', 60),
            ('Distance (AU)', 90),
            ('RA (°)', 70),
            ('Dec (°)', 70),
            ('Earth MOID', 80),
            ('Period (yr)', 75),
            ('e', 55),
            ('i (°)', 55),
            ('Discovered', 90),
            ('Site', 45)
        ]
        
        self.table.setColumnCount(len(self.columns))
        headers = [col[0] for col in self.columns]
        self.table.setHorizontalHeaderLabels(headers)
        
        # Set column widths
        for i, (name, width) in enumerate(self.columns):
            self.table.setColumnWidth(i, width)
        
        layout.addWidget(self.table)
        
        # Button row
        btn_layout = QHBoxLayout()
        
        # Export button
        export_btn = QPushButton("📥 Export CSV")
        export_btn.clicked.connect(self.export_csv)
        btn_layout.addWidget(export_btn)
        
        # Analysis button with dropdown
        self.analysis_btn = QPushButton("📊 Analysis")
        analysis_menu = QMenu(self)
        analysis_menu.addAction("Histogram: H magnitude", lambda: self.show_histogram('H'))
        analysis_menu.addAction("Histogram: Eccentricity", lambda: self.show_histogram('e'))
        analysis_menu.addAction("Histogram: Inclination", lambda: self.show_histogram('i'))
        analysis_menu.addAction("Histogram: Semi-major axis", lambda: self.show_histogram('a'))
        analysis_menu.addAction("Histogram: V magnitude", lambda: self.show_histogram('V'))
        analysis_menu.addSeparator()
        analysis_menu.addAction("Scatter: a vs e", lambda: self.show_scatter('a', 'e'))
        analysis_menu.addAction("Scatter: a vs i", lambda: self.show_scatter('a', 'i'))
        analysis_menu.addAction("Scatter: H vs distance", lambda: self.show_scatter('H', 'dist'))
        analysis_menu.addSeparator()
        analysis_menu.addAction("Summary Statistics", self.show_statistics)
        self.analysis_btn.setMenu(analysis_menu)
        btn_layout.addWidget(self.analysis_btn)
        
        btn_layout.addStretch()
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        btn_layout.addWidget(close_btn)
        
        layout.addLayout(btn_layout)
        self.setLayout(layout)
    
    def update_table(self, visible_data, asteroids, jd):
        """
        Update table with current visible NEOs
        
        Parameters:
        -----------
        visible_data : np.ndarray
            Array of [idx, ra, dec, dist, mag] for visible NEOs
        asteroids : list
            List of asteroid dictionaries
        jd : float
            Current Julian Date
        """
        # Store references for click handling and filtering
        self.current_visible_data = visible_data
        self.current_asteroids = asteroids
        self.current_jd = jd
        self.row_to_ast_idx = {}
        
        if visible_data is None or asteroids is None or len(visible_data) == 0:
            self.table.setRowCount(0)
            self.info_label.setText("No NEOs currently visible")
            self.warning_label.hide()
            return
        
        from designation_utils import unpack_designation
        from skyfield.api import load as skyfield_load
        
        # Get current date for display
        ts = skyfield_load.timescale()
        current_date = ts.tt_jd(jd).utc_datetime().strftime('%Y-%m-%d') if jd else ""
        
        # Apply filter if set
        filter_text = self.current_filter.lower().strip()
        
        # Build list of rows to display
        display_rows = []
        for pos_row in visible_data:
            ast_idx = int(pos_row[0])
            if ast_idx >= len(asteroids):
                continue
            
            ast = asteroids[ast_idx]
            
            # Check filter
            if filter_text:
                # Search in multiple fields (handle None values)
                packed = (ast.get('designation') or '').lower()
                readable = (ast.get('readable_designation') or '').lower()
                orbit_class = (ast.get('orbit_class') or '').lower()
                site = (ast.get('discovery_site') or '').lower()
                
                try:
                    unpacked = unpack_designation(ast.get('designation', '')).lower()
                except:
                    unpacked = ''
                
                # Match if filter is found in any field
                if not (filter_text in packed or 
                        filter_text in readable or 
                        filter_text in unpacked or
                        filter_text in orbit_class or
                        filter_text in site):
                    continue
            
            display_rows.append((pos_row, ast_idx, ast))
        
        total_filtered = len(display_rows)
        
        # Check row limit
        if total_filtered > self.MAX_TABLE_ROWS:
            self.warning_label.setText(
                f"⚠ Limiting display to {self.MAX_TABLE_ROWS} rows (of {total_filtered}) to prevent memory issues. "
                "Use the filter to narrow results, or export to CSV."
            )
            self.warning_label.show()
            display_rows = display_rows[:self.MAX_TABLE_ROWS]
        else:
            self.warning_label.hide()
        
        self.table.setSortingEnabled(False)  # Disable during update
        self.table.setRowCount(len(display_rows))
        
        for row_idx, (pos_row, ast_idx, ast) in enumerate(display_rows):
            ra = pos_row[1]
            dec = pos_row[2]
            dist = pos_row[3]
            vmag = pos_row[4]
            
            # Store mapping (will be updated after sorting)
            self.row_to_ast_idx[row_idx] = (ast_idx, row_idx)
            
            # Designation - store ast_idx in UserRole for retrieval after sorting
            packed = ast.get('designation', '')
            readable = ast.get('readable_designation', '').strip()
            if readable:
                display_des = readable
            else:
                try:
                    display_des = unpack_designation(packed)
                except:
                    display_des = packed
            item = QTableWidgetItem(display_des)
            item.setData(Qt.ItemDataRole.UserRole + 1, ast_idx)  # Store asteroid index
            item.setData(Qt.ItemDataRole.UserRole + 2, row_idx)  # Store visible_data index
            self.table.setItem(row_idx, 0, item)
            
            # Orbit class
            orbit_class = ast.get('orbit_class', '')
            self._set_cell(row_idx, 1, orbit_class)
            
            # H magnitude
            h_mag = ast.get('H', None)
            self._set_cell(row_idx, 2, f"{h_mag:.2f}" if h_mag else "")
            
            # V magnitude
            self._set_cell(row_idx, 3, f"{vmag:.2f}")
            
            # Distance
            self._set_cell(row_idx, 4, f"{dist:.4f}")
            
            # RA
            self._set_cell(row_idx, 5, f"{ra:.3f}")
            
            # Dec
            self._set_cell(row_idx, 6, f"{dec:+.3f}")
            
            # Earth MOID
            moid = ast.get('earth_moid', None)
            self._set_cell(row_idx, 7, f"{moid:.4f}" if moid else "")
            
            # Period
            a = ast.get('a', None)
            if a:
                period = a ** 1.5
                self._set_cell(row_idx, 8, f"{period:.3f}")
            else:
                self._set_cell(row_idx, 8, "")
            
            # Eccentricity
            e = ast.get('e', None)
            self._set_cell(row_idx, 9, f"{e:.4f}" if e else "")
            
            # Inclination
            i = ast.get('i', None)
            self._set_cell(row_idx, 10, f"{i:.2f}" if i else "")
            
            # Discovery date
            disc_mjd = ast.get('discovery_mjd', None)
            if disc_mjd:
                disc_jd = disc_mjd + 2400000.5
                disc_date = ts.tt_jd(disc_jd).utc_datetime().strftime('%Y-%m-%d')
                self._set_cell(row_idx, 11, disc_date)
            else:
                self._set_cell(row_idx, 11, "")
            
            # Discovery site
            site = ast.get('discovery_site', '')
            self._set_cell(row_idx, 12, site if site else "")
        
        self.table.setSortingEnabled(True)
        
        # Update info label
        filter_info = f" (filtered)" if filter_text else ""
        total_info = f" of {len(visible_data)}" if len(display_rows) != len(visible_data) else ""
        self.info_label.setText(f"Showing {len(display_rows)}{total_info} NEOs{filter_info} as of {current_date}")
    
    def apply_filter(self):
        """Apply the current filter text"""
        self.current_filter = self.filter_input.text()
        if self.current_visible_data is not None:
            self.update_table(self.current_visible_data, self.current_asteroids, self.current_jd)
    
    def clear_filter(self):
        """Clear the filter"""
        self.filter_input.clear()
        self.current_filter = ""
        if self.current_visible_data is not None:
            self.update_table(self.current_visible_data, self.current_asteroids, self.current_jd)
    
    def _set_cell(self, row, col, text):
        """Helper to set cell value with proper sorting"""
        item = QTableWidgetItem(str(text))
        # For numeric columns, set data for proper sorting
        if col in [2, 3, 4, 5, 6, 7, 8, 9, 10]:  # Numeric columns
            try:
                item.setData(Qt.ItemDataRole.UserRole, float(text) if text else float('inf'))
            except:
                pass
        self.table.setItem(row, col, item)
    
    def export_csv(self):
        """Export table to CSV file"""
        from PyQt6.QtWidgets import QFileDialog
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", "neo_table.csv", "CSV files (*.csv)"
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    # Header
                    headers = [col[0] for col in self.columns]
                    f.write(','.join(headers) + '\n')
                    
                    # Data rows
                    for row in range(self.table.rowCount()):
                        row_data = []
                        for col in range(self.table.columnCount()):
                            item = self.table.item(row, col)
                            text = item.text() if item else ''
                            # Quote if contains comma
                            if ',' in text:
                                text = f'"{text}"'
                            row_data.append(text)
                        f.write(','.join(row_data) + '\n')
                
                self.info_label.setText(f"Exported {self.table.rowCount()} rows to {filename}")
            except Exception as e:
                self.info_label.setText(f"Export error: {e}")
    
    def on_row_clicked(self, row, col):
        """Handle click on table row - highlight on plot and show popup"""
        # Get the designation item (column 0) which stores the indices
        des_item = self.table.item(row, 0)
        if des_item is None:
            return
        
        # Retrieve stored indices
        ast_idx = des_item.data(Qt.ItemDataRole.UserRole + 1)
        vis_idx = des_item.data(Qt.ItemDataRole.UserRole + 2)
        
        if ast_idx is None or self.current_asteroids is None:
            return
        
        if ast_idx >= len(self.current_asteroids):
            return
        
        ast = self.current_asteroids[ast_idx]
        
        # Get position data
        if vis_idx is not None and self.current_visible_data is not None and vis_idx < len(self.current_visible_data):
            pos_row = self.current_visible_data[vis_idx]
            ra = pos_row[1]
            dec = pos_row[2]
            dist = pos_row[3]
            mag = pos_row[4]
        else:
            return
        
        # Get parent window (NEOVisualizer)
        parent = self.parent()
        if parent is None:
            return
        
        # Find plot position and highlight
        if hasattr(parent, 'canvas') and parent.canvas is not None:
            canvas = parent.canvas
            
            # Find this object in the plot offsets
            if hasattr(canvas, 'visible_data') and canvas.visible_data is not None:
                for i, vd in enumerate(canvas.visible_data):
                    if int(vd[0]) == ast_idx:
                        if hasattr(canvas, 'plot_offsets') and i < len(canvas.plot_offsets):
                            plot_x, plot_y = canvas.plot_offsets[i]
                            canvas.highlight_object(plot_x, plot_y)
                        break
            
            # Show popup
            canvas.show_neo_popup(ast, ra, dec, dist, mag, self.current_jd)
    
    def _get_table_data(self):
        """Extract data from currently displayed table rows"""
        if self.current_asteroids is None or self.current_visible_data is None:
            return None
        
        data = []
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)  # Designation column
            if item is None:
                continue
            ast_idx = item.data(Qt.ItemDataRole.UserRole + 1)
            if ast_idx is not None and ast_idx < len(self.current_asteroids):
                ast = self.current_asteroids[ast_idx]
                # Find corresponding visible_data entry
                for vd in self.current_visible_data:
                    if int(vd[0]) == ast_idx:
                        data.append({
                            'ast': ast,
                            'ra': vd[1],
                            'dec': vd[2],
                            'dist': vd[3],
                            'V': vd[4]
                        })
                        break
        return data
    
    def show_histogram(self, field):
        """Show histogram of a field for current table data"""
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        
        data = self._get_table_data()
        if not data:
            return
        
        # Extract values
        values = []
        labels = {
            'H': ('Absolute Magnitude (H)', 'H magnitude'),
            'e': ('Eccentricity', 'e'),
            'i': ('Inclination', 'Inclination (°)'),
            'a': ('Semi-major Axis', 'a (AU)'),
            'V': ('Visual Magnitude', 'V magnitude')
        }
        
        for d in data:
            if field == 'V':
                val = d['V']
            else:
                val = d['ast'].get(field)
            if val is not None:
                values.append(val)
        
        if not values:
            return
        
        # Create dialog with histogram
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Histogram: {labels[field][0]}")
        dialog.setMinimumSize(500, 400)
        
        layout = QVBoxLayout()
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(values, bins=30, edgecolor='black', alpha=0.7)
        ax.set_xlabel(labels[field][1])
        ax.set_ylabel('Count')
        ax.set_title(f'{labels[field][0]} Distribution (n={len(values)})')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        
        canvas = FigureCanvasQTAgg(fig)
        layout.addWidget(canvas)
        
        # Stats label
        import numpy as np
        arr = np.array(values)
        stats_text = f"Mean: {np.mean(arr):.3f}  Median: {np.median(arr):.3f}  Std: {np.std(arr):.3f}  Min: {np.min(arr):.3f}  Max: {np.max(arr):.3f}"
        stats_label = QLabel(stats_text)
        layout.addWidget(stats_label)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)
        
        dialog.setLayout(layout)
        dialog.show()
    
    def show_scatter(self, x_field, y_field):
        """Show scatter plot of two fields"""
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        
        data = self._get_table_data()
        if not data:
            return
        
        labels = {
            'H': 'H magnitude',
            'e': 'Eccentricity',
            'i': 'Inclination (°)',
            'a': 'Semi-major axis (AU)',
            'V': 'V magnitude',
            'dist': 'Distance (AU)'
        }
        
        x_vals, y_vals = [], []
        for d in data:
            if x_field in ['V', 'dist']:
                x = d['V'] if x_field == 'V' else d['dist']
            else:
                x = d['ast'].get(x_field)
            
            if y_field in ['V', 'dist']:
                y = d['V'] if y_field == 'V' else d['dist']
            else:
                y = d['ast'].get(y_field)
            
            if x is not None and y is not None:
                x_vals.append(x)
                y_vals.append(y)
        
        if not x_vals:
            return
        
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Scatter: {labels[x_field]} vs {labels[y_field]}")
        dialog.setMinimumSize(500, 400)
        
        layout = QVBoxLayout()
        
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(x_vals, y_vals, alpha=0.5, s=10)
        ax.set_xlabel(labels[x_field])
        ax.set_ylabel(labels[y_field])
        ax.set_title(f'{labels[x_field]} vs {labels[y_field]} (n={len(x_vals)})')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        
        canvas = FigureCanvasQTAgg(fig)
        layout.addWidget(canvas)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)
        
        dialog.setLayout(layout)
        dialog.show()
    
    def show_statistics(self):
        """Show summary statistics for current table data"""
        import numpy as np
        
        data = self._get_table_data()
        if not data:
            return
        
        # Collect values
        H_vals = [d['ast'].get('H') for d in data if d['ast'].get('H') is not None]
        e_vals = [d['ast'].get('e') for d in data if d['ast'].get('e') is not None]
        i_vals = [d['ast'].get('i') for d in data if d['ast'].get('i') is not None]
        a_vals = [d['ast'].get('a') for d in data if d['ast'].get('a') is not None]
        V_vals = [d['V'] for d in data]
        dist_vals = [d['dist'] for d in data]
        
        # Count by class
        class_counts = {}
        for d in data:
            cls = d['ast'].get('orbit_class', 'Unknown')
            class_counts[cls] = class_counts.get(cls, 0) + 1
        
        # Count PHAs - calculate from H and MOID (same as popup)
        pha_count = 0
        for d in data:
            h_mag = d['ast'].get('H')
            moid = d['ast'].get('earth_moid')
            if h_mag is not None and h_mag <= 22.0 and moid is not None and moid <= 0.05:
                pha_count += 1
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Summary Statistics")
        dialog.setMinimumSize(400, 500)
        
        layout = QVBoxLayout()
        
        text = QTextBrowser()
        text.setOpenExternalLinks(False)
        
        def stat_line(name, vals):
            if not vals:
                return f"<tr><td>{name}</td><td colspan='5'>No data</td></tr>"
            arr = np.array(vals)
            return f"<tr><td>{name}</td><td>{np.mean(arr):.3f}</td><td>{np.median(arr):.3f}</td><td>{np.std(arr):.3f}</td><td>{np.min(arr):.3f}</td><td>{np.max(arr):.3f}</td></tr>"
        
        html = f"""
        <h3>Summary Statistics (n={len(data)} objects)</h3>
        
        <h4>Numerical Fields</h4>
        <table border='1' cellpadding='4'>
        <tr><th>Field</th><th>Mean</th><th>Median</th><th>Std</th><th>Min</th><th>Max</th></tr>
        {stat_line('H magnitude', H_vals)}
        {stat_line('V magnitude', V_vals)}
        {stat_line('Eccentricity', e_vals)}
        {stat_line('Inclination', i_vals)}
        {stat_line('Semi-major axis', a_vals)}
        {stat_line('Distance (AU)', dist_vals)}
        </table>
        
        <h4>Classification</h4>
        <table border='1' cellpadding='4'>
        <tr><th>Class</th><th>Count</th><th>%</th></tr>
        """
        
        for cls in sorted(class_counts.keys()):
            pct = 100 * class_counts[cls] / len(data)
            html += f"<tr><td>{cls}</td><td>{class_counts[cls]}</td><td>{pct:.1f}%</td></tr>"
        
        html += f"""
        </table>
        
        <p><b>PHAs:</b> {pha_count} ({100*pha_count/len(data):.1f}%)</p>
        """
        
        text.setHtml(html)
        layout.addWidget(text)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)
        
        dialog.setLayout(layout)
        dialog.show()
    
    def closeEvent(self, event):
        """Save geometry before closing"""
        NEOTableDialog.last_geometry = self.geometry()
        event.accept()


class CollapsiblePanel(QWidget):
    """A panel with a clickable header that can collapse/expand its content"""
    
    collapsed_changed = pyqtSignal(bool)  # Emits True when collapsed
    
    def __init__(self, title, content_widget, parent=None):
        super().__init__(parent)
        self.content_widget = content_widget
        self.collapsed = False
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header button that shows title and collapse indicator
        self.header = QPushButton(f"▼ {title}")
        self.header.setStyleSheet("""
            QPushButton {
                text-align: left;
                padding: 4px 8px;
                background: #e0e0e0;
                border: 1px solid #c0c0c0;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #d0d0d0;
            }
        """)
        self.header.clicked.connect(self.toggle)
        self.title = title
        layout.addWidget(self.header)
        
        # Content container
        self.content_container = QWidget()
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addWidget(content_widget)
        self.content_container.setLayout(content_layout)
        layout.addWidget(self.content_container)
        
        self.setLayout(layout)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
    
    def toggle(self):
        """Toggle collapsed state"""
        self.collapsed = not self.collapsed
        self.content_container.setVisible(not self.collapsed)
        self.header.setText(f"{'▶' if self.collapsed else '▼'} {self.title}")
        self.collapsed_changed.emit(self.collapsed)
    
    def set_collapsed(self, collapsed):
        """Set collapsed state programmatically"""
        if self.collapsed != collapsed:
            self.toggle()
    
    def is_collapsed(self):
        """Return current collapsed state"""
        return self.collapsed


class CollapsibleGroupBox(QWidget):
    """A collapsible section styled like QGroupBox with a clickable title bar"""
    
    def __init__(self, title, parent=None, collapsed=True):
        super().__init__(parent)
        self.full_title = title
        self.collapsed = collapsed
        
        self._main_layout = QVBoxLayout()
        self._main_layout.setContentsMargins(0, 0, 0, 8)
        self._main_layout.setSpacing(0)
        
        # Use a QGroupBox for native styling
        self.group_box = QGroupBox()
        self.group_box.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #c0c0c0;
                border-radius: 5px;
                margin-top: 0px;
                padding-top: 5px;
                background-color: #f8f8f8;
            }
        """)
        
        group_layout = QVBoxLayout()
        group_layout.setContentsMargins(0, 0, 0, 0)
        group_layout.setSpacing(0)
        
        # Title bar (clickable button inside the group box)
        self.title_btn = QPushButton()
        self.title_btn.setStyleSheet("""
            QPushButton {
                text-align: left;
                padding: 6px 10px;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #f5f5f5, stop:1 #e8e8e8);
                border: none;
                border-bottom: 1px solid #d0d0d0;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                font-weight: bold;
                color: #333;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #e8e8e8, stop:1 #dcdcdc);
            }
        """)
        self.title_btn.clicked.connect(self.toggle)
        group_layout.addWidget(self.title_btn)
        
        # Content container inside the group box - NO stylesheet to preserve native widgets
        self.content_frame = QWidget()
        self.content_layout = QVBoxLayout()
        self.content_layout.setContentsMargins(10, 8, 10, 10)
        self.content_layout.setSpacing(4)
        self.content_frame.setLayout(self.content_layout)
        group_layout.addWidget(self.content_frame)
        
        self.group_box.setLayout(group_layout)
        self._main_layout.addWidget(self.group_box)
        
        super().setLayout(self._main_layout)
        
        # Apply initial state
        self._update_display()
    
    def setLayout(self, layout):
        """Add all widgets from layout to content_layout"""
        # Transfer widgets from the provided layout to our content_layout
        while layout.count():
            item = layout.takeAt(0)
            if item.widget():
                self.content_layout.addWidget(item.widget())
            elif item.layout():
                self.content_layout.addLayout(item.layout())
    
    def layout(self):
        """Return content layout for adding widgets"""
        return self.content_layout
    
    def toggle(self):
        """Toggle collapsed state"""
        self.collapsed = not self.collapsed
        self._update_display()
    
    def _update_display(self):
        """Update display based on collapsed state"""
        indicator = "▶" if self.collapsed else "▼"
        self.title_btn.setText(f"{indicator} {self.full_title}")
        self.content_frame.setVisible(not self.collapsed)


class TimeControlPanel(QWidget):
    """Enhanced time controls with better rate options"""
    
    time_changed = pyqtSignal(float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.animation_timer = QTimer()
        # Connection to advance_time made in ControlsPanel
        
        ts = skyfield_load.timescale()
        self.current_jd = ts.now().tt
        
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)  # No margins for alignment
        layout.setSpacing(0)
        
        # Date and Time - COMPACT (datetime and quick jumps in one group)
        dt_group = QGroupBox("Date and Time")
        dt_layout = QVBoxLayout()
        
        # DateTime and Now button on same line
        dt_row = QHBoxLayout()
        self.datetime_edit = QDateTimeEdit()
        self.datetime_edit.setCalendarPopup(True)
        # CRITICAL: Set the widget itself to use UTC, not just the QDateTime values
        self.datetime_edit.setTimeSpec(QtCompat.UTC)
        self.datetime_edit.setDateTime(QDateTime.currentDateTimeUtc())
        self.datetime_edit.setDisplayFormat("yyyy-MM-dd HH:mm 'UTC'")
        self.datetime_edit.dateTimeChanged.connect(self.on_datetime_changed)
        dt_row.addWidget(self.datetime_edit)
        
        now_btn = QPushButton("Now")
        now_btn.setMaximumWidth(50)
        now_btn.clicked.connect(self.set_to_now)
        dt_row.addWidget(now_btn)
        dt_layout.addLayout(dt_row)
        
        # Quick jumps below
        jump_layout = QGridLayout()
        
        jumps = [
            ("◀ Yr", -365, 0, 0),
            ("◀ Mo", -30, 0, 1),
            ("◀ Day", -1, 0, 2),
            ("Day ▶", 1, 1, 0),
            ("Mo ▶", 30, 1, 1),
            ("Yr ▶", 365, 1, 2)
        ]
        
        for label, days, row, col in jumps:
            btn = QPushButton(label)
            btn.clicked.connect(lambda _, d=days: self.jump_days(d))
            jump_layout.addWidget(btn, row, col)
        
        dt_layout.addLayout(jump_layout)
        
        # CLN (Catalina Lunation Number) input row
        cln_row = QHBoxLayout()
        cln_row.addWidget(QLabel("CLN:"))
        
        self.cln_spin = QSpinBox()
        self.cln_spin.setRange(-10000, 10000)  # Allow historical and future dates
        self.cln_spin.setValue(0)
        self.cln_spin.setToolTip("Catalina Lunation Number (0 = Full Moon of 1980-01-02)")
        self.cln_spin.valueChanged.connect(self.on_cln_changed)
        cln_row.addWidget(self.cln_spin)
        
        cln_row.addWidget(QLabel("+"))
        
        self.cln_days_spin = QDoubleSpinBox()
        self.cln_days_spin.setRange(0.0, 29.53)
        self.cln_days_spin.setValue(0.0)
        self.cln_days_spin.setDecimals(2)
        self.cln_days_spin.setSingleStep(1.0)
        self.cln_days_spin.setSuffix(" days")
        self.cln_days_spin.setToolTip("Days after the specified Full Moon")
        self.cln_days_spin.valueChanged.connect(self.on_cln_changed)
        cln_row.addWidget(self.cln_days_spin)
        
        dt_layout.addLayout(cln_row)
        
        dt_group.setLayout(dt_layout)
        layout.addWidget(dt_group)
        
        self.setLayout(layout)
        
        # Animation timer (controlled externally by ControlsPanel)
        self.animation_timer = QTimer()
        
        # Flag to prevent recursive updates between datetime and CLN
        self._updating_from_cln = False
        self._updating_from_datetime = False
        
        # Initialize CLN display from current datetime
        self._update_cln_from_jd(self.current_jd)
    
    def on_datetime_changed(self, qdatetime):
        if self._updating_from_cln:
            return
        
        dt = qdatetime.toPyDateTime()
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=utc)
        
        ts = skyfield_load.timescale()
        t = ts.from_datetime(dt)
        self.current_jd = t.tt
        
        # Update CLN spinboxes
        self._updating_from_datetime = True
        self._update_cln_from_jd(self.current_jd)
        self._updating_from_datetime = False
        
        self.time_changed.emit(self.current_jd)
    
    def on_cln_changed(self):
        """Handle CLN or days spinbox change"""
        if self._updating_from_datetime:
            return
        
        cln = self.cln_spin.value()
        days_offset = self.cln_days_spin.value()
        
        # Convert CLN to JD
        new_jd = cln_to_jd(cln, days_offset)
        self.current_jd = new_jd
        
        # Update datetime edit without triggering on_datetime_changed
        self._updating_from_cln = True
        ts = skyfield_load.timescale()
        t = ts.tt_jd(new_jd)
        dt = t.utc_datetime()
        self.datetime_edit.setDateTime(python_datetime_to_utc_qdatetime(dt))
        self._updating_from_cln = False
        
        self.time_changed.emit(self.current_jd)
    
    def _update_cln_from_jd(self, jd):
        """Update CLN spinboxes from Julian Date"""
        try:
            cln, days_offset = jd_to_cln(jd)
            self.cln_spin.blockSignals(True)
            self.cln_days_spin.blockSignals(True)
            self.cln_spin.setValue(cln)
            self.cln_days_spin.setValue(days_offset)
            self.cln_spin.blockSignals(False)
            self.cln_days_spin.blockSignals(False)
        except Exception as e:
            # If CLN calculation fails, just log and continue
            logger.debug(f"CLN calculation error: {e}")
    
    def set_to_now(self):
        self.datetime_edit.setDateTime(QDateTime.currentDateTimeUtc())
    
    def set_jd(self, jd):
        """Set the datetime from a Julian Date"""
        ts = skyfield_load.timescale()
        t = ts.tt_jd(jd)
        dt = t.utc_datetime()
        self.datetime_edit.setDateTime(python_datetime_to_utc_qdatetime(dt))
    
    def jump_days(self, days):
        current = self.datetime_edit.dateTime().toPyDateTime()
        new_dt = current + timedelta(days=days)
        self.datetime_edit.setDateTime(python_datetime_to_utc_qdatetime(new_dt))
    

class ControlsPanel(QWidget):
    """Controls panel with animation and action buttons"""
    
    def __init__(self, time_panel, parent=None):
        super().__init__(parent)
        self.time_panel = time_panel  # Reference to TimeControlPanel
        self.parent_window = parent
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)  # No margins for alignment
        layout.setSpacing(0)
        
        # Controls group - contains animation and buttons
        controls_group = QGroupBox("Controls")
        controls_layout = QVBoxLayout()
        controls_layout.setSpacing(2)
        
        # Play button and FPS on first line
        play_layout = QHBoxLayout()
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.setMaximumWidth(70)
        self.play_btn.clicked.connect(self.toggle_play)
        play_layout.addWidget(self.play_btn)
        
        play_layout.addWidget(QLabel("FPS:"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(10)
        self.fps_spin.setMaximumWidth(50)
        play_layout.addWidget(self.fps_spin)
        play_layout.addStretch()
        
        controls_layout.addLayout(play_layout)
        
        # Rate controls on second line (left-aligned)
        rate_layout = QHBoxLayout()
        rate_layout.addWidget(QLabel("Rate:"))
        
        self.rate_spin = QDoubleSpinBox()
        self.rate_spin.setRange(-1000, 1000)  # Allow negative for backwards playback
        self.rate_spin.setValue(1.0)
        self.rate_spin.setDecimals(3)
        self.rate_spin.setMaximumWidth(80)  # Wider to fit minus sign
        rate_layout.addWidget(self.rate_spin)
        
        self.rate_unit = QComboBox()
        self.rate_unit.addItems(['hours/sec', 'days/sec', 'days/min', 'months/min'])
        self.rate_unit.setCurrentText('days/sec')
        self.rate_unit.setMaximumWidth(90)
        rate_layout.addWidget(self.rate_unit)
        rate_layout.addStretch()  # Push everything to the left
        
        controls_layout.addLayout(rate_layout)
        
        # Annual step mode (for catalog growth visualization)
        annual_layout = QHBoxLayout()
        self.annual_step_check = QCheckBox("Annual step")
        self.annual_step_check.setChecked(False)
        self.annual_step_check.setToolTip("Step one sidereal year at a time (for catalog growth)\nUse negative values to go backwards")
        self.annual_step_check.stateChanged.connect(self.on_annual_mode_changed)
        annual_layout.addWidget(self.annual_step_check)
        
        annual_layout.addWidget(QLabel("sec/yr:"))
        self.annual_pace_spin = QDoubleSpinBox()
        self.annual_pace_spin.setRange(-10.0, 10.0)  # Allow negative for backwards
        self.annual_pace_spin.setValue(1.0)
        self.annual_pace_spin.setSingleStep(0.5)
        self.annual_pace_spin.setDecimals(1)
        self.annual_pace_spin.setMaximumWidth(60)
        self.annual_pace_spin.setToolTip("Seconds between annual steps (negative = backwards)")
        self.annual_pace_spin.valueChanged.connect(self.on_annual_pace_changed)
        annual_layout.addWidget(self.annual_pace_spin)
        annual_layout.addStretch()
        
        controls_layout.addLayout(annual_layout)
        
        # Action buttons on one line - Table and More only (Help/Reset/Exit moved to status bar)
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(5)
        
        # Table button with dropdown menu
        self.table_btn = QPushButton("📋 Table")
        self.table_btn.setMaximumWidth(80)
        
        table_menu = QMenu(self)
        table_menu.addAction("📋 Show All Visible", self.show_table_all_visible)
        table_menu.addSeparator()
        table_menu.addAction("🔲 Select Rectangle...", self.start_rectangle_select)
        table_menu.addAction("⬭ Select Ellipse...", self.start_ellipse_select)
        table_menu.addSeparator()
        table_menu.addAction("🔍 Search Catalog...", self.search_catalog_for_table)
        table_menu.addSeparator()
        table_menu.addAction("❌ Clear Selection", self.clear_selection)
        
        self.table_btn.setMenu(table_menu)
        self.table_btn.setToolTip("Show table of NEOs (all visible or select region)")
        buttons_layout.addWidget(self.table_btn)
        
        # Charts button with dropdown
        charts_btn = QPushButton("📈 Charts")
        charts_btn.setMaximumWidth(80)
        
        charts_menu = QMenu(self)
        charts_menu.addAction("📏 Distance vs Time (selected object)...", self.show_distance_time_chart)
        charts_menu.addAction("⚠️ MOID vs H (hazard space)", self.show_moid_h_chart)
        charts_menu.addAction("📅 Discovery Timeline", self.show_discovery_timeline)
        charts_menu.addSeparator()
        charts_menu.addAction("🌍 Solar Elongation vs Distance", self.show_elongation_distance_chart)
        charts_menu.addAction("🔄 a vs e (orbital element space)", self.show_a_e_chart)
        charts_menu.addSeparator()
        charts_menu.addAction("🌙 Lunar Phases (year view)", self.show_lunar_phases_chart)
        charts_menu.addSeparator()
        charts_menu.addAction("☀️ Heliocentric View (polar)", self.show_heliocentric_chart)
        
        charts_btn.setMenu(charts_menu)
        charts_btn.setToolTip("Open analysis charts")
        buttons_layout.addWidget(charts_btn)
        
        buttons_layout.addStretch()
        controls_layout.addLayout(buttons_layout)
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        self.setLayout(layout)
    
    def toggle_play(self):
        if self.time_panel.animation_timer.isActive():
            # PAUSING - stop animation but preserve trails
            self.time_panel.animation_timer.stop()
            self.play_btn.setText("▶ Play")
            # Notify canvas that animation stopped (paused)
            if self.parent_window and hasattr(self.parent_window, 'canvas'):
                self.parent_window.canvas.animation_playing = False
                self.parent_window.canvas.animation_paused = True  # Mark as paused
                # Disable blitting since animation stopped
                self.parent_window.canvas.disable_blitting()
                # Trigger redraw to restore display mode (density/contours)
                self.parent_window.update_display()
            # Sync statusbar button if it exists
            if self.parent_window and hasattr(self.parent_window, 'statusbar_play_btn'):
                self.parent_window.statusbar_play_btn.setText("▶ Play")
        else:
            # STARTING or RESUMING
            # Check if we're resuming from pause
            is_resuming = False
            if self.parent_window and hasattr(self.parent_window, 'canvas'):
                is_resuming = getattr(self.parent_window.canvas, 'animation_paused', False)
            
            if self.annual_step_check.isChecked():
                # Annual step mode: timer interval = |seconds per year| * 1000
                # Sign determines direction in advance_time, abs value is the delay
                pace = abs(self.annual_pace_spin.value())
                if pace < 0.1:
                    pace = 0.1  # Minimum pace
                interval_ms = int(pace * 1000)
                # Clear trails for annual step mode (large jumps) - always, even on resume
                if self.parent_window and hasattr(self.parent_window, 'canvas'):
                    self.parent_window.canvas.trail_history.clear()
                    self.parent_window.canvas._clear_trails()
                    logger.debug("TRAIL: Cleared for annual step mode")
            else:
                # Normal mode: use FPS
                fps = self.fps_spin.value()
                interval_ms = 1000 // fps
            
            self.time_panel.animation_timer.start(interval_ms)
            self.play_btn.setText("⏸ Pause")
            
            # Notify canvas that animation started
            if self.parent_window and hasattr(self.parent_window, 'canvas'):
                self.parent_window.canvas.animation_playing = True
                self.parent_window.canvas.animation_paused = False

                # Enable blitting for faster animation updates
                self.parent_window.canvas._use_blitting = True
                self.parent_window.canvas.capture_background()

                # Only clear trails on FRESH start, not on resume from pause
                if not is_resuming:
                    self.parent_window.canvas.trail_history.clear()
                    self.parent_window.canvas._clear_trails()
                    logger.debug("TRAIL: Cleared for fresh animation start")
                else:
                    logger.debug("TRAIL: Resuming from pause - preserving trails")
            # Sync statusbar button if it exists
            if self.parent_window and hasattr(self.parent_window, 'statusbar_play_btn'):
                self.parent_window.statusbar_play_btn.setText("⏸ Pause")
    
    def on_annual_mode_changed(self, state):
        """Handle annual step mode toggle"""
        is_annual = (state == 2)  # Qt.Checked = 2
        # Disable/enable normal rate controls
        self.rate_spin.setEnabled(not is_annual)
        self.rate_unit.setEnabled(not is_annual)
        self.fps_spin.setEnabled(not is_annual)
        self.annual_pace_spin.setEnabled(is_annual)
        
        # If currently playing, restart with new interval
        if self.time_panel.animation_timer.isActive():
            self.toggle_play()  # Stop
            self.toggle_play()  # Restart with new settings
    
    def on_annual_pace_changed(self, value):
        """Handle annual pace change"""
        # If playing in annual mode, restart with new interval
        if self.annual_step_check.isChecked() and self.time_panel.animation_timer.isActive():
            self.toggle_play()  # Stop
            self.toggle_play()  # Restart with new settings
    
    def stop_animation(self):
        self.time_panel.animation_timer.stop()
        self.play_btn.setText("▶ Play")
        # Notify canvas that animation stopped (hard stop, not pause)
        if self.parent_window and hasattr(self.parent_window, 'canvas'):
            self.parent_window.canvas.animation_playing = False
            self.parent_window.canvas.animation_paused = False  # Not paused - fully stopped
            # Disable blitting since animation stopped
            self.parent_window.canvas.disable_blitting()
        # Sync statusbar button if it exists
        if self.parent_window and hasattr(self.parent_window, 'statusbar_play_btn'):
            self.parent_window.statusbar_play_btn.setText("▶ Play")
    
    def advance_time(self):
        """Advance time based on rate and units or annual step mode"""
        # Check for annual step mode
        if self.annual_step_check.isChecked():
            # Sidereal year = 365.25636 days (Earth returns to same position relative to stars)
            # This keeps the Sun's ecliptic longitude approximately constant
            SIDEREAL_YEAR = 365.25636
            pace = self.annual_pace_spin.value()
            # Direction is determined by sign of pace
            if pace < 0:
                self.time_panel.jump_days(-SIDEREAL_YEAR)
            else:
                self.time_panel.jump_days(SIDEREAL_YEAR)
            return
        
        # Normal mode
        rate = self.rate_spin.value()
        unit = self.rate_unit.currentText()
        fps = self.fps_spin.value()
        
        # Convert to days per frame
        if unit == 'hours/sec':
            days_per_frame = (rate / 24.0) / fps
        elif unit == 'days/sec':
            days_per_frame = rate / fps
        elif unit == 'days/min':
            days_per_frame = (rate / 60.0) / fps
        elif unit == 'months/min':
            days_per_frame = (rate * 30.0 / 60.0) / fps
        else:
            days_per_frame = rate / fps
        
        self.time_panel.jump_days(days_per_frame)
    
    def show_settings(self):
        """Show Settings dialog"""
        if self.parent_window:
            self.parent_window.show_settings()
    
    def show_help(self):
        """Show help dialog"""
        if self.parent_window:
            self.parent_window.show_help()
    
    def reset_all(self):
        """Reset all controls"""
        if self.parent_window:
            self.parent_window.reset_all()
    
    def show_table_all_visible(self):
        """Show table with all currently visible NEOs"""
        if self.parent_window:
            # Clear any selection first
            if hasattr(self.parent_window, 'canvas') and self.parent_window.canvas:
                self.parent_window.canvas.clear_selection()
            self.parent_window.toggle_table()
    
    def start_rectangle_select(self):
        """Start rectangle selection mode"""
        if self.parent_window and hasattr(self.parent_window, 'canvas'):
            canvas = self.parent_window.canvas
            canvas.start_selection('rectangle')
            canvas.selection_complete = self._on_selection_complete
            self.parent_window.status_label.setText("Drag to select region (rectangle). Press Escape to cancel.")
    
    def start_ellipse_select(self):
        """Start ellipse selection mode"""
        if self.parent_window and hasattr(self.parent_window, 'canvas'):
            canvas = self.parent_window.canvas
            canvas.start_selection('ellipse')
            canvas.selection_complete = self._on_selection_complete
            self.parent_window.status_label.setText("Drag to select region (ellipse). Press Escape to cancel.")
    
    def search_catalog_for_table(self):
        """Open dialog to search catalog and add objects to table"""
        if not self.parent_window:
            return
        
        from PyQt6.QtWidgets import QInputDialog
        
        text, ok = QInputDialog.getText(
            self.parent_window, 
            "Search Catalog",
            "Enter designations to find (comma-separated):\n\n"
            "Examples: Apophis, Eros, 2024 AA, 99942\n",
            QLineEdit.EchoMode.Normal,
            ""
        )
        
        if ok and text.strip():
            self._search_and_show_objects(text)
    
    def _search_and_show_objects(self, search_text):
        """Search for objects and show them in table"""
        if not self.parent_window or not hasattr(self.parent_window, 'canvas'):
            return
        
        canvas = self.parent_window.canvas
        if canvas.current_asteroids is None:
            self.parent_window.status_label.setText("No asteroid data loaded.")
            return
        
        from designation_utils import pack_designation, unpack_designation
        
        # Parse search terms
        search_terms = [t.strip() for t in search_text.split(',') if t.strip()]
        
        found_indices = []
        not_found = []
        
        for term in search_terms:
            term_lower = term.lower()
            term_nospace = term.replace(' ', '')
            found = False
            
            for idx, ast in enumerate(canvas.current_asteroids):
                packed = ast['designation']
                readable = (ast.get('readable_designation') or '').strip()
                
                # Check various match types
                if packed == term or packed.strip() == term:
                    found_indices.append(idx)
                    found = True
                    break
                
                if readable and (readable.lower() == term_lower or 
                                term_lower in readable.lower()):
                    found_indices.append(idx)
                    found = True
                    break
                
                try:
                    unpacked = unpack_designation(packed)
                    if unpacked == term or unpacked.replace(' ', '') == term_nospace:
                        found_indices.append(idx)
                        found = True
                        break
                except:
                    pass
                
                # Number match
                if term.isdigit():
                    packed_num = packed.lstrip('0').strip()
                    if packed_num == term.lstrip('0'):
                        found_indices.append(idx)
                        found = True
                        break
                    if readable and readable.split()[0] == term.lstrip('0'):
                        found_indices.append(idx)
                        found = True
                        break
            
            if not found:
                not_found.append(term)
        
        if not found_indices:
            self.parent_window.status_label.setText(f"No objects found for: {', '.join(not_found)}")
            return
        
        # Build visible_data for found objects
        # We need to calculate positions for these objects
        if canvas.current_positions is None:
            self.parent_window.status_label.setText("Position data not available. Try after plot updates.")
            return
        
        # Find matching position data
        selected_data = []
        for idx in found_indices:
            for pos in canvas.current_positions:
                if int(pos[0]) == idx:
                    selected_data.append(pos)
                    break
        
        if selected_data:
            import numpy as np
            selected_data = np.array(selected_data)
            self.parent_window.show_table_with_selection(selected_data)
            
            msg = f"Found {len(selected_data)} objects."
            if not_found:
                msg += f" Not found: {', '.join(not_found[:3])}"
                if len(not_found) > 3:
                    msg += f" (+{len(not_found)-3} more)"
            self.parent_window.status_label.setText(msg)
        else:
            self.parent_window.status_label.setText("Objects found but no position data available.")
    
    def _on_selection_complete(self, selected_indices):
        """Callback when selection is complete"""
        if not selected_indices:
            self.parent_window.status_label.setText("No objects selected in region.")
            return
        
        # Get the selected data
        if self.parent_window and hasattr(self.parent_window, 'canvas'):
            canvas = self.parent_window.canvas
            
            # Build filtered visible_data for just selected objects
            if canvas.visible_data is not None and len(selected_indices) > 0:
                selected_data = canvas.visible_data[selected_indices]
                
                # Show table with selected data
                self.parent_window.show_table_with_selection(selected_data)
                self.parent_window.status_label.setText(f"Selected {len(selected_indices)} objects. Table showing selection.")
    
    def clear_selection(self):
        """Clear selection mode"""
        if self.parent_window and hasattr(self.parent_window, 'canvas'):
            self.parent_window.canvas.clear_selection()
            self.parent_window.status_label.setText("Selection cleared.")
    
    def toggle_table(self):
        """Toggle table display (for backwards compatibility)"""
        self.show_table_all_visible()
    
    def exit_app(self):
        """Exit application"""
        if self.parent_window:
            self.parent_window.close()
    
    # =========== CHART METHODS ===========
    
    def show_distance_time_chart(self):
        """Show distance vs time chart for a selected object"""
        if not self.parent_window:
            return
        
        from PyQt6.QtWidgets import QInputDialog
        
        # Ask user for object designation
        text, ok = QInputDialog.getText(
            self.parent_window,
            "Distance vs Time",
            "Enter object designation:\n(e.g., Apophis, 99942, 2024 AA)",
            QLineEdit.EchoMode.Normal,
            ""
        )
        
        if not ok or not text.strip():
            return
        
        # Find the object
        canvas = self.parent_window.canvas
        if canvas.current_asteroids is None:
            self.parent_window.status_label.setText("No asteroid data loaded.")
            return
        
        # Search for the object
        result = canvas.find_and_highlight_designation(text.strip())
        ast, ra, dec, dist, mag, visible = result
        
        if ast is None:
            self.parent_window.status_label.setText(f"Object '{text}' not found.")
            return
        
        # Calculate distances over time range
        self._show_distance_time_plot(ast)
    
    def _show_distance_time_plot(self, ast):
        """Generate and show distance vs time plot for an asteroid"""
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        import numpy as np
        
        # Get orbital elements
        from orbit_calculator import OrbitCalculator
        calculator = OrbitCalculator()
        
        # Time range: 1 year centered on current time
        current_jd = self.time_panel.current_jd
        jd_start = current_jd - 182.5  # 6 months ago
        jd_end = current_jd + 182.5    # 6 months ahead
        
        # Calculate positions at 1-day intervals
        jd_range = np.arange(jd_start, jd_end, 1.0)
        distances = []
        
        elements = {
            'a': ast['a'], 'e': ast['e'], 'i': ast['i'],
            'node': ast['node'], 'arg_peri': ast['arg_peri'],
            'M': ast['M'], 'epoch_jd': ast['epoch_jd']
        }
        
        for jd in jd_range:
            try:
                ra, dec, dist, mag = calculator.calculate_position(elements, jd, ast.get('H', 20), ast.get('G', 0.15))
                distances.append(dist)
            except:
                distances.append(np.nan)
        
        # Convert JD to dates for x-axis
        from datetime import datetime, timedelta
        base_date = datetime(2000, 1, 1, 12, 0, 0)  # J2000
        dates = [base_date + timedelta(days=jd - 2451545.0) for jd in jd_range]
        
        # Get readable name
        readable = ast.get('readable_designation', '').strip()
        name = readable if readable else ast['designation']
        
        # Create dialog with plot
        dialog = QDialog(self.parent_window)
        dialog.setWindowTitle(f"Distance vs Time: {name}")
        dialog.setMinimumSize(700, 500)
        
        layout = QVBoxLayout()
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(dates, distances, 'b-', linewidth=1.5)
        
        # Mark current time
        current_idx = len(jd_range) // 2
        ax.axvline(dates[current_idx], color='red', linestyle='--', alpha=0.7, label='Now')
        ax.plot(dates[current_idx], distances[current_idx], 'ro', markersize=8)
        
        # Mark perihelion distance line
        q = ast['a'] * (1 - ast['e'])
        ax.axhline(q, color='orange', linestyle=':', alpha=0.5, label=f'Perihelion (q={q:.3f} AU)')
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Geocentric Distance (AU)')
        ax.set_title(f'{name} - Geocentric Distance Over Time')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Format x-axis dates
        fig.autofmt_xdate()
        fig.tight_layout()
        
        canvas_widget = FigureCanvasQTAgg(fig)
        layout.addWidget(canvas_widget)
        
        # Stats
        distances_arr = np.array(distances)
        valid = ~np.isnan(distances_arr)
        if np.any(valid):
            stats_text = f"Current: {distances_arr[current_idx]:.4f} AU | Min: {np.nanmin(distances_arr):.4f} AU | Max: {np.nanmax(distances_arr):.4f} AU"
        else:
            stats_text = "Unable to calculate distances"
        stats_label = QLabel(stats_text)
        layout.addWidget(stats_label)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)
        
        dialog.setLayout(layout)
        dialog.show()
        
        self.parent_window.status_label.setText(f"Showing distance chart for {name}")
    
    def show_moid_h_chart(self):
        """Show MOID vs H magnitude scatter plot (hazard space)"""
        if not self.parent_window or not hasattr(self.parent_window, 'canvas'):
            return
        
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        import numpy as np
        
        canvas = self.parent_window.canvas
        if canvas.current_asteroids is None:
            self.parent_window.status_label.setText("No asteroid data loaded.")
            return
        
        # Collect MOID and H values
        moid_vals = []
        h_vals = []
        is_pha = []
        
        for ast in canvas.current_asteroids:
            moid = ast.get('earth_moid')
            h = ast.get('H')
            if moid is not None and h is not None:
                moid_vals.append(moid)
                h_vals.append(h)
                # PHA: H <= 22 and MOID <= 0.05
                is_pha.append(h <= 22.0 and moid <= 0.05)
        
        if not moid_vals:
            self.parent_window.status_label.setText("No MOID/H data available.")
            return
        
        moid_arr = np.array(moid_vals)
        h_arr = np.array(h_vals)
        pha_arr = np.array(is_pha)
        
        # Create dialog
        dialog = QDialog(self.parent_window)
        dialog.setWindowTitle("MOID vs H Magnitude (Hazard Space)")
        dialog.setMinimumSize(700, 550)
        
        layout = QVBoxLayout()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot non-PHAs
        non_pha = ~pha_arr
        ax.scatter(moid_arr[non_pha], h_arr[non_pha], s=5, alpha=0.3, c='blue', label='Non-PHA')
        
        # Plot PHAs
        ax.scatter(moid_arr[pha_arr], h_arr[pha_arr], s=15, alpha=0.7, c='red', label='PHA')
        
        # Draw PHA boundary box
        ax.axvline(0.05, color='red', linestyle='--', alpha=0.5)
        ax.axhline(22.0, color='red', linestyle='--', alpha=0.5)
        
        # Shade PHA region
        ax.fill_between([0, 0.05], [0, 0], [22, 22], alpha=0.1, color='red', label='PHA region')
        
        ax.set_xlabel('Earth MOID (AU)')
        ax.set_ylabel('Absolute Magnitude (H)')
        ax.set_title('NEO Hazard Space: MOID vs H')
        ax.set_xlim(0, max(0.5, np.percentile(moid_arr, 99)))
        ax.set_ylim(min(h_arr) - 1, max(h_arr) + 1)
        ax.invert_yaxis()  # Brighter (smaller H) at top
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        fig.tight_layout()
        
        canvas_widget = FigureCanvasQTAgg(fig)
        layout.addWidget(canvas_widget)
        
        # Stats
        pha_count = np.sum(pha_arr)
        stats_text = f"Total: {len(moid_arr)} | PHAs: {pha_count} ({100*pha_count/len(moid_arr):.1f}%) | MOID range: {np.min(moid_arr):.4f} - {np.max(moid_arr):.4f} AU"
        stats_label = QLabel(stats_text)
        layout.addWidget(stats_label)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)
        
        dialog.setLayout(layout)
        dialog.show()
        
        self.parent_window.status_label.setText(f"Showing MOID vs H chart ({pha_count} PHAs)")
    
    def show_discovery_timeline(self):
        """Show discovery timeline histogram"""
        if not self.parent_window or not hasattr(self.parent_window, 'canvas'):
            return
        
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        import numpy as np
        from datetime import datetime
        
        canvas = self.parent_window.canvas
        if canvas.current_asteroids is None:
            self.parent_window.status_label.setText("No asteroid data loaded.")
            return
        
        # Collect discovery years
        discovery_years = []
        
        for ast in canvas.current_asteroids:
            disc_mjd = ast.get('discovery_mjd')
            if disc_mjd is not None:
                # Convert MJD to year
                jd = disc_mjd + 2400000.5
                # Approximate year from JD
                year = 2000 + (jd - 2451545.0) / 365.25
                discovery_years.append(int(year))
        
        if not discovery_years:
            self.parent_window.status_label.setText("No discovery date data available.")
            return
        
        years_arr = np.array(discovery_years)
        min_year = max(1990, np.min(years_arr))
        max_year = np.max(years_arr)
        
        # Count by year
        year_range = range(min_year, max_year + 1)
        counts = [np.sum(years_arr == y) for y in year_range]
        cumulative = np.cumsum(counts)
        
        # Create dialog
        dialog = QDialog(self.parent_window)
        dialog.setWindowTitle("NEO Discovery Timeline")
        dialog.setMinimumSize(800, 550)
        
        layout = QVBoxLayout()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        
        # Bar chart of discoveries per year
        bars = ax1.bar(list(year_range), counts, color='steelblue', edgecolor='navy', alpha=0.7)
        ax1.set_ylabel('Discoveries per Year')
        ax1.set_title('NEO Discovery Rate')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Highlight major surveys with annotations
        # CSS started ~2004, Pan-STARRS ~2010, ATLAS ~2015
        ax1.axvline(2004, color='green', linestyle='--', alpha=0.5)
        ax1.axvline(2010, color='orange', linestyle='--', alpha=0.5)
        ax1.axvline(2015, color='purple', linestyle='--', alpha=0.5)
        
        # Cumulative plot
        ax2.plot(list(year_range), cumulative, 'b-', linewidth=2)
        ax2.fill_between(list(year_range), cumulative, alpha=0.3)
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Cumulative Total')
        ax2.set_title('Cumulative NEO Catalog Size')
        ax2.grid(True, alpha=0.3)
        
        # Mark current catalog size
        ax2.axhline(cumulative[-1], color='red', linestyle=':', alpha=0.5)
        ax2.text(min_year + 1, cumulative[-1] * 0.95, f'{cumulative[-1]:,} total', fontsize=10, color='red')
        
        fig.tight_layout()
        
        canvas_widget = FigureCanvasQTAgg(fig)
        layout.addWidget(canvas_widget)
        
        # Stats
        recent_5yr = np.sum(years_arr >= max_year - 4)
        stats_text = f"Total with dates: {len(years_arr):,} | Last 5 years: {recent_5yr:,} ({100*recent_5yr/len(years_arr):.1f}%) | Peak year: {list(year_range)[np.argmax(counts)]} ({max(counts):,})"
        stats_label = QLabel(stats_text)
        layout.addWidget(stats_label)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)
        
        dialog.setLayout(layout)
        dialog.show()
        
        self.parent_window.status_label.setText(f"Showing discovery timeline ({len(years_arr):,} objects with dates)")
    
    def show_elongation_distance_chart(self):
        """Show solar elongation vs geocentric distance for visible NEOs"""
        if not self.parent_window or not hasattr(self.parent_window, 'canvas'):
            return
        
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        import numpy as np
        
        canvas = self.parent_window.canvas
        if canvas.visible_data is None or len(canvas.visible_data) == 0:
            self.parent_window.status_label.setText("No visible NEOs to plot.")
            return
        
        # Calculate solar elongation for each visible object
        sun_ra = getattr(canvas, 'sun_ra', None)
        sun_dec = getattr(canvas, 'sun_dec', None)
        
        if sun_ra is None or sun_dec is None:
            self.parent_window.status_label.setText("Sun position not available.")
            return
        
        elongations = []
        distances = []
        magnitudes = []
        
        for row in canvas.visible_data:
            ra, dec, dist, mag = row[1], row[2], row[3], row[4]
            
            # Calculate angular separation from Sun
            ra1, dec1 = np.radians(ra), np.radians(dec)
            ra2, dec2 = np.radians(sun_ra), np.radians(sun_dec)
            
            cos_sep = np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2)
            cos_sep = np.clip(cos_sep, -1, 1)
            elongation = np.degrees(np.arccos(cos_sep))
            
            elongations.append(elongation)
            distances.append(dist)
            magnitudes.append(mag)
        
        elong_arr = np.array(elongations)
        dist_arr = np.array(distances)
        mag_arr = np.array(magnitudes)
        
        # Create dialog
        dialog = QDialog(self.parent_window)
        dialog.setWindowTitle("Solar Elongation vs Distance")
        dialog.setMinimumSize(700, 550)
        
        layout = QVBoxLayout()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        scatter = ax.scatter(elong_arr, dist_arr, c=mag_arr, cmap='viridis_r', s=10, alpha=0.6)
        cbar = fig.colorbar(scatter, ax=ax, label='V magnitude')
        
        # Mark solar exclusion zone
        ax.axvline(30, color='orange', linestyle='--', alpha=0.7, label='30° elongation')
        ax.axvspan(0, 30, alpha=0.1, color='yellow', label='Solar exclusion')
        
        ax.set_xlabel('Solar Elongation (degrees)')
        ax.set_ylabel('Geocentric Distance (AU)')
        ax.set_title('NEO Observability: Elongation vs Distance')
        ax.set_xlim(0, 180)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        fig.tight_layout()
        
        canvas_widget = FigureCanvasQTAgg(fig)
        layout.addWidget(canvas_widget)
        
        # Stats
        near_sun = np.sum(elong_arr < 30)
        at_opposition = np.sum(elong_arr > 150)
        stats_text = f"Total: {len(elong_arr)} | Near Sun (<30°): {near_sun} | Near Opposition (>150°): {at_opposition}"
        stats_label = QLabel(stats_text)
        layout.addWidget(stats_label)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)
        
        dialog.setLayout(layout)
        dialog.show()
        
        self.parent_window.status_label.setText(f"Showing elongation vs distance ({len(elong_arr)} visible NEOs)")
    
    def show_a_e_chart(self):
        """Show semi-major axis vs eccentricity plot with orbit class regions"""
        if not self.parent_window or not hasattr(self.parent_window, 'canvas'):
            return
        
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        import numpy as np
        
        canvas = self.parent_window.canvas
        if canvas.current_asteroids is None:
            self.parent_window.status_label.setText("No asteroid data loaded.")
            return
        
        # Collect a, e values by orbit class
        data_by_class = {}
        
        for ast in canvas.current_asteroids:
            a = ast.get('a')
            e = ast.get('e')
            cls = ast.get('orbit_class', 'Unknown')
            
            if a is not None and e is not None:
                if cls not in data_by_class:
                    data_by_class[cls] = {'a': [], 'e': []}
                data_by_class[cls]['a'].append(a)
                data_by_class[cls]['e'].append(e)
        
        if not data_by_class:
            self.parent_window.status_label.setText("No orbital element data available.")
            return
        
        # Create dialog
        dialog = QDialog(self.parent_window)
        dialog.setWindowTitle("Orbital Element Space: a vs e")
        dialog.setMinimumSize(750, 600)
        
        layout = QVBoxLayout()
        
        fig, ax = plt.subplots(figsize=(9, 7))
        
        # Color map for classes
        class_colors = {
            'Atira': 'purple',
            'Aten': 'red',
            'Apollo': 'orange',
            'Amor': 'green',
            'Unknown': 'gray'
        }
        
        # Plot each class
        for cls, vals in data_by_class.items():
            color = class_colors.get(cls, 'blue')
            ax.scatter(vals['a'], vals['e'], s=5, alpha=0.4, c=color, label=f'{cls} ({len(vals["a"])})')
        
        # Draw key boundaries
        # Earth's orbit: a=1, e variable
        a_line = np.linspace(0.5, 4, 100)
        
        # q = 1.0 line (perihelion at Earth): a(1-e) = 1 -> e = 1 - 1/a
        e_q1 = 1 - 1/a_line
        ax.plot(a_line[a_line > 1], e_q1[a_line > 1], 'k--', alpha=0.5, label='q = 1.0 AU')
        
        # Q = 1.0 line (aphelion at Earth): a(1+e) = 1 -> e = 1/a - 1
        e_Q1 = 1/a_line - 1
        valid = (e_Q1 >= 0) & (e_Q1 <= 1)
        ax.plot(a_line[valid], e_Q1[valid], 'k:', alpha=0.5, label='Q = 1.0 AU')
        
        # q = 1.3 line (Amor boundary)
        e_q13 = 1 - 1.3/a_line
        valid = (e_q13 >= 0) & (e_q13 <= 1)
        ax.plot(a_line[valid], e_q13[valid], 'g--', alpha=0.3, label='q = 1.3 AU')
        
        # Earth line
        ax.axvline(1.0, color='blue', linestyle='-', alpha=0.3, linewidth=2)
        ax.text(1.02, 0.95, 'Earth', fontsize=9, color='blue')
        
        ax.set_xlabel('Semi-major Axis (AU)')
        ax.set_ylabel('Eccentricity')
        ax.set_title('NEO Orbital Element Space')
        ax.set_xlim(0.3, 4.5)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
        
        fig.tight_layout()
        
        canvas_widget = FigureCanvasQTAgg(fig)
        layout.addWidget(canvas_widget)
        
        # Stats
        total = sum(len(v['a']) for v in data_by_class.values())
        class_summary = ' | '.join([f'{cls}: {len(v["a"])}' for cls, v in sorted(data_by_class.items())])
        stats_label = QLabel(f"Total: {total} | {class_summary}")
        layout.addWidget(stats_label)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)
        
        dialog.setLayout(layout)
        dialog.show()
        
        self.parent_window.status_label.setText(f"Showing a vs e chart ({total} objects)")
    
    def show_lunar_phases_chart(self):
        """Show lunar phases through the year centered on current date"""
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        from matplotlib.patches import Circle, Wedge, Path as MplPath, PathPatch
        import matplotlib.dates as mdates
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Lunar Phases - Year View")
        dialog.resize(900, 500)
        layout = QVBoxLayout()
        
        # Get current JD from time panel
        current_jd = self.time_panel.current_jd
        
        # Load ephemeris
        try:
            ensure_ephemeris('de421.bsp')
            eph = skyfield_load('de421.bsp')
            ts = skyfield_load.timescale()
            earth = eph['earth']
            moon = eph['moon']
            sun = eph['sun']
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not load ephemeris: {e}")
            return
        
        # Calculate phases for 6 months before and after current date
        days_range = 183  # ~6 months each direction
        jd_start = current_jd - days_range
        jd_end = current_jd + days_range
        
        # Sample every day
        jds = np.arange(jd_start, jd_end, 1.0)
        dates = []
        illuminations = []
        waxings = []
        
        for jd in jds:
            t = ts.tt_jd(jd)
            
            # Get positions
            earth_pos = earth.at(t)
            moon_astrometric = earth_pos.observe(moon)
            sun_astrometric = earth_pos.observe(sun)
            
            ra_moon, dec_moon, _ = moon_astrometric.radec()
            ra_sun, dec_sun, _ = sun_astrometric.radec()
            
            # Calculate elongation
            ra1, dec1 = np.radians(ra_moon.degrees), np.radians(dec_moon.degrees)
            ra2, dec2 = np.radians(ra_sun.degrees), np.radians(dec_sun.degrees)
            cos_elong = (np.sin(dec1) * np.sin(dec2) + 
                        np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2))
            elongation = np.degrees(np.arccos(np.clip(cos_elong, -1, 1)))
            
            # Illumination
            illum = (1 - np.cos(np.radians(elongation))) / 2
            illuminations.append(illum)
            
            # Waxing/waning
            ra_diff = (ra_moon.degrees - ra_sun.degrees + 360) % 360
            waxings.append(ra_diff < 180)
            
            # Convert JD to datetime
            dt = t.utc_datetime()
            dates.append(dt)
        
        illuminations = np.array(illuminations)
        
        # Find new moons and full moons (local minima and maxima)
        new_moons = []
        full_moons = []
        for i in range(1, len(illuminations) - 1):
            if illuminations[i] < illuminations[i-1] and illuminations[i] < illuminations[i+1]:
                if illuminations[i] < 0.1:
                    new_moons.append(i)
            if illuminations[i] > illuminations[i-1] and illuminations[i] > illuminations[i+1]:
                if illuminations[i] > 0.9:
                    full_moons.append(i)
        
        # Create figure
        fig = Figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        
        # Plot illumination curve
        ax.fill_between(dates, illuminations, alpha=0.3, color='silver', label='Illumination')
        ax.plot(dates, illuminations, 'k-', linewidth=0.5, alpha=0.5)
        
        # Draw moon phase symbols at regular intervals (every ~7 days)
        symbol_interval = 7
        for i in range(0, len(dates), symbol_interval):
            dt = dates[i]
            illum = illuminations[i]
            waxing = waxings[i]
            
            # Draw small moon phase icon
            self._draw_moon_phase_icon(ax, dt, 0.5, illum, waxing, size=0.03)
        
        # Mark new moons
        for idx in new_moons:
            ax.axvline(dates[idx], color='#303030', linestyle=':', alpha=0.5, linewidth=1)
            ax.plot(dates[idx], illuminations[idx], 'o', color='#303030', markersize=8)
        
        # Mark full moons
        for idx in full_moons:
            ax.axvline(dates[idx], color='#FFD700', linestyle=':', alpha=0.5, linewidth=1)
            ax.plot(dates[idx], illuminations[idx], 'o', color='#FFFACD', 
                   markeredgecolor='#FFD700', markersize=10, markeredgewidth=1.5)
        
        # Mark current date
        current_t = ts.tt_jd(current_jd)
        current_dt = current_t.utc_datetime()
        ax.axvline(current_dt, color='red', linestyle='-', linewidth=2, alpha=0.7)
        ax.text(current_dt, 1.05, 'Now', ha='center', va='bottom', color='red', fontsize=10, fontweight='bold')
        
        # Find current illumination
        current_idx = np.argmin(np.abs(jds - current_jd))
        current_illum = illuminations[current_idx]
        current_waxing = waxings[current_idx]
        
        # Determine current phase name
        if current_illum < 0.03:
            phase_name = 'New Moon'
        elif current_illum > 0.97:
            phase_name = 'Full Moon'
        elif 0.45 < current_illum < 0.55:
            phase_name = 'First Quarter' if current_waxing else 'Last Quarter'
        elif current_illum < 0.5:
            phase_name = 'Waxing Crescent' if current_waxing else 'Waning Crescent'
        else:
            phase_name = 'Waxing Gibbous' if current_waxing else 'Waning Gibbous'
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Illumination Fraction')
        ax.set_title(f'Lunar Phases — Current: {phase_name} ({current_illum*100:.0f}% illuminated)')
        ax.set_ylim(-0.05, 1.15)
        ax.set_xlim(dates[0], dates[-1])
        
        # Format x-axis
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_minor_locator(mdates.WeekdayLocator())
        fig.autofmt_xdate()
        
        ax.grid(True, alpha=0.3)
        
        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#303030', markersize=8, label='New Moon'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#FFFACD', markeredgecolor='#FFD700', markersize=10, label='Full Moon'),
            Line2D([0], [0], color='red', linewidth=2, label='Current Date'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        fig.tight_layout()
        
        canvas_widget = FigureCanvasQTAgg(fig)
        layout.addWidget(canvas_widget)
        
        # Info label
        info_text = f"Showing {days_range*2} days ({days_range//30} months before and after current date)"
        info_label = QLabel(info_text)
        layout.addWidget(info_label)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)
        
        dialog.setLayout(layout)
        dialog.show()
        
        self.parent_window.status_label.setText(f"Showing lunar phases chart - {phase_name}")
    
    def _draw_moon_phase_icon(self, ax, x_date, y, illumination, waxing, size=0.03):
        """Draw a small moon phase icon at the given position"""
        from matplotlib.patches import Circle, Path as MplPath, PathPatch
        import matplotlib.dates as mdates
        
        # Convert date to numeric for positioning
        x = mdates.date2num(x_date)
        
        # Use axis transform to get consistent size
        # Size in data coordinates
        radius = size
        
        # Dark background
        dark = Circle((x, y), radius, facecolor='#404040', edgecolor='#228B22', 
                      linewidth=0.5, transform=ax.transData, zorder=10)
        ax.add_patch(dark)
        
        if illumination > 0.03 and illumination < 0.97:
            # Draw illuminated portion
            n_points = 20
            theta = np.linspace(-np.pi/2, np.pi/2, n_points)
            
            # Terminator curve
            if illumination < 0.5:
                curve = -(1 - 2*illumination)
            else:
                curve = 2*illumination - 1
            
            lit_side = 1 if waxing else -1
            
            outer_x = lit_side * radius * np.cos(theta)
            outer_y = radius * np.sin(theta)
            inner_x = lit_side * radius * curve * np.cos(theta)
            inner_y = radius * np.sin(theta)
            
            verts = []
            codes = []
            
            for i, (ox, oy) in enumerate(zip(outer_x, outer_y)):
                verts.append((x + ox, y + oy))
                codes.append(MplPath.MOVETO if i == 0 else MplPath.LINETO)
            
            for ix, iy in zip(inner_x[::-1], inner_y[::-1]):
                verts.append((x + ix, y + iy))
                codes.append(MplPath.LINETO)
            
            codes.append(MplPath.CLOSEPOLY)
            verts.append(verts[0])
            
            path = MplPath(verts, codes)
            patch = PathPatch(path, facecolor='#FFFACD', edgecolor='none',
                             transform=ax.transData, zorder=11)
            ax.add_patch(patch)
        elif illumination >= 0.97:
            # Full moon
            bright = Circle((x, y), radius * 0.9, facecolor='#FFFACD', edgecolor='none',
                           transform=ax.transData, zorder=11)
            ax.add_patch(bright)
    
    def show_heliocentric_chart(self):
        """Show heliocentric polar chart - Sun-centered view of NEO positions"""
        if not self.parent_window or not hasattr(self.parent_window, 'canvas'):
            return
        
        canvas = self.parent_window.canvas
        if canvas.current_asteroids is None:
            self.parent_window.status_label.setText("No asteroid data loaded.")
            return
        
        # Create the chart dialog
        dialog = HelicentricChartDialog(self.parent_window, self.time_panel)
        dialog.show()
        
        self.parent_window.status_label.setText("Heliocentric chart opened")


class HelicentricChartDialog(QDialog):
    """Heliocentric polar chart showing NEOs in Sun-centered ecliptic coordinates"""
    
    def __init__(self, parent_window, time_panel, parent=None):
        super().__init__(parent or parent_window)
        self.parent_window = parent_window
        self.time_panel = time_panel
        self.setWindowTitle("☀️ Heliocentric View")
        self.resize(800, 800)
        
        # Animation state
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.advance_time)
        self.animation_playing = False
        self.current_jd = time_panel.current_jd
        self.animation_rate = 1.0  # days per frame
        
        # Display mode
        self.density_mode = False
        self.color_by = 'V magnitude'
        
        # Selected object for click-to-identify
        self.selected_neo = None
        
        # Store computed positions for click detection
        self.neo_positions = []  # List of (theta, r, asteroid_dict)
        
        self.setup_ui()
        self.update_plot()
    
    def setup_ui(self):
        """Set up the dialog UI"""
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
        from matplotlib.figure import Figure
        
        layout = QVBoxLayout()
        layout.setSpacing(5)
        
        # Control bar - row 1
        control_layout = QHBoxLayout()
        
        # Play/Pause button
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.setMaximumWidth(70)
        self.play_btn.clicked.connect(self.toggle_play)
        control_layout.addWidget(self.play_btn)
        
        # Animation speed
        control_layout.addWidget(QLabel("Rate:"))
        self.rate_spin = QDoubleSpinBox()
        self.rate_spin.setRange(0.1, 30.0)
        self.rate_spin.setValue(1.0)
        self.rate_spin.setSingleStep(0.5)
        self.rate_spin.setSuffix(" d/frame")
        self.rate_spin.setMaximumWidth(100)
        self.rate_spin.valueChanged.connect(self.on_rate_changed)
        control_layout.addWidget(self.rate_spin)
        
        control_layout.addWidget(QLabel(" "))
        
        # Display mode
        self.density_check = QCheckBox("Density map")
        self.density_check.setChecked(False)
        self.density_check.stateChanged.connect(self.on_mode_changed)
        control_layout.addWidget(self.density_check)
        
        control_layout.addWidget(QLabel(" "))
        
        # Max radius control
        control_layout.addWidget(QLabel("Max AU:"))
        self.max_r_spin = QDoubleSpinBox()
        self.max_r_spin.setRange(1.0, 10.0)
        self.max_r_spin.setValue(4.0)
        self.max_r_spin.setSingleStep(0.5)
        self.max_r_spin.setMaximumWidth(70)
        self.max_r_spin.valueChanged.connect(self.update_plot)
        control_layout.addWidget(self.max_r_spin)
        
        control_layout.addStretch()
        
        # Sync with main button
        sync_btn = QPushButton("📡 Sync")
        sync_btn.setToolTip("Sync time with main display")
        sync_btn.clicked.connect(self.sync_with_main)
        control_layout.addWidget(sync_btn)
        
        layout.addLayout(control_layout)
        
        # Control bar - row 2 (NEO and planet options)
        options_layout = QHBoxLayout()
        
        # Show all NEOs checkbox
        self.show_all_check = QCheckBox("Show all NEOs")
        self.show_all_check.setChecked(False)
        self.show_all_check.setToolTip("Show all NEOs regardless of V magnitude filter")
        self.show_all_check.stateChanged.connect(self.update_plot)
        options_layout.addWidget(self.show_all_check)
        
        # Sync show_all with main panel
        if hasattr(self.parent_window, 'magnitude_panel'):
            self.show_all_check.setChecked(self.parent_window.magnitude_panel.show_all_neos_check.isChecked())
        
        options_layout.addWidget(QLabel(" │ "))
        
        # Show planets checkbox (not Earth - Earth always shown)
        self.show_planets_check = QCheckBox("Show planets")
        self.show_planets_check.setChecked(True)
        self.show_planets_check.setToolTip("Show Mercury, Venus, Mars, Jupiter positions (Earth always shown)")
        self.show_planets_check.stateChanged.connect(self.update_plot)
        options_layout.addWidget(self.show_planets_check)
        
        # Show orbits checkbox
        self.show_orbits_check = QCheckBox("Orbits")
        self.show_orbits_check.setChecked(True)
        self.show_orbits_check.setToolTip("Show planetary orbit circles")
        self.show_orbits_check.stateChanged.connect(self.update_plot)
        options_layout.addWidget(self.show_orbits_check)
        
        options_layout.addStretch()
        layout.addLayout(options_layout)
        
        # Row 3: Coordinate system and filtering options
        coord_layout = QHBoxLayout()
        
        # Coordinate system dropdown
        coord_layout.addWidget(QLabel("Longitude:"))
        self.coord_combo = QComboBox()
        self.coord_combo.addItems(["Geocentric ecliptic", "True heliocentric"])
        self.coord_combo.setToolTip("Geocentric: angle as seen from Earth\nTrue heliocentric: angle from Sun's perspective")
        self.coord_combo.setMaximumWidth(150)
        self.coord_combo.currentIndexChanged.connect(self.update_plot)
        coord_layout.addWidget(self.coord_combo)
        
        coord_layout.addWidget(QLabel(" │ "))
        
        # Behind sun filter (helio < geo distance)
        self.hide_behind_sun_check = QCheckBox("Hide behind sun")
        self.hide_behind_sun_check.setChecked(False)
        self.hide_behind_sun_check.setToolTip("Hide NEOs where heliocentric distance < geocentric distance (syncs with main display)")
        self.hide_behind_sun_check.stateChanged.connect(self.update_plot)
        coord_layout.addWidget(self.hide_behind_sun_check)
        
        coord_layout.addWidget(QLabel(" │ "))
        
        # Mark behind-sun NEOs (visual indicator)
        self.mark_behind_sun_check = QCheckBox("Mark behind sun")
        self.mark_behind_sun_check.setChecked(False)
        self.mark_behind_sun_check.setToolTip("Show NEOs with heliocentric < geocentric distance as hollow red circles")
        self.mark_behind_sun_check.stateChanged.connect(self.update_plot)
        coord_layout.addWidget(self.mark_behind_sun_check)
        
        coord_layout.addStretch()
        layout.addLayout(coord_layout)
        
        # Create matplotlib figure with polar projection
        self.fig = Figure(figsize=(8, 8))
        self.ax = self.fig.add_subplot(111, projection='polar')
        
        self.canvas_widget = FigureCanvasQTAgg(self.fig)
        self.canvas_widget.mpl_connect('button_press_event', self.on_click)
        layout.addWidget(self.canvas_widget)
        
        # Toolbar
        toolbar = NavigationToolbar2QT(self.canvas_widget, self)
        layout.addWidget(toolbar)
        
        # Status bar
        status_layout = QHBoxLayout()
        self.status_label = QLabel("")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()
        
        self.date_label = QLabel("")
        status_layout.addWidget(self.date_label)
        layout.addLayout(status_layout)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)
        
        self.setLayout(layout)
    
    def toggle_play(self):
        """Toggle animation play/pause"""
        if self.animation_playing:
            self.animation_timer.stop()
            self.animation_playing = False
            self.play_btn.setText("▶ Play")
        else:
            self.animation_timer.start(100)  # 10 fps
            self.animation_playing = True
            self.play_btn.setText("⏸ Pause")
    
    def advance_time(self):
        """Advance time for animation"""
        self.current_jd += self.animation_rate
        self.update_plot()
    
    def on_rate_changed(self):
        """Handle animation rate change"""
        self.animation_rate = self.rate_spin.value()
    
    def on_mode_changed(self):
        """Handle display mode change"""
        self.density_mode = self.density_check.isChecked()
        self.update_plot()
    
    def sync_with_main(self):
        """Sync time with main display"""
        self.current_jd = self.time_panel.current_jd
        self.update_plot()
    
    def compute_heliocentric_positions(self, jd):
        """Compute heliocentric positions for all visible NEOs"""
        canvas = self.parent_window.canvas
        if canvas.current_asteroids is None:
            return [], None
        
        # Get coordinate system choice
        use_true_helio = self.coord_combo.currentIndex() == 1  # "True heliocentric"
        hide_behind_sun = self.hide_behind_sun_check.isChecked()
        
        # Get Sun position and store for calculations
        try:
            ts = skyfield_load.timescale()
            t = ts.tt_jd(jd)
            eph = skyfield_load('de421.bsp')
            earth = eph['earth']
            sun = eph['sun']
            
            sun_astrometric = earth.at(t).observe(sun)
            sun_ra_deg, sun_dec_deg, sun_dist_au = sun_astrometric.radec()
            sun_ra = np.radians(sun_ra_deg._degrees)
            sun_dec = np.radians(sun_dec_deg.degrees)
            sun_dist = sun_dist_au.au
            
            # Earth's heliocentric ecliptic longitude (opposite of Sun's geocentric RA)
            earth_ecl_lon = (sun_ra_deg._degrees + 180) % 360
            
            # For true heliocentric: get Sun and Earth positions in ICRS
            sun_pos_icrs = sun.at(t).position.au  # Sun in ICRS (barycentric)
            earth_pos_icrs = earth.at(t).position.au  # Earth in ICRS
            
            # Store sun info for planet calculations
            sun_info = {
                'ra': sun_ra_deg._degrees,
                'dec': sun_dec_deg.degrees,
                'dist': sun_dist,
                'earth_lon': earth_ecl_lon,
                't': t,
                'eph': eph,
                'earth': earth,
                'sun_pos_icrs': sun_pos_icrs,
                'earth_pos_icrs': earth_pos_icrs
            }
        except Exception as e:
            logger.error(f"Ephemeris error: {e}")
            return [], None
        
        # Obliquity for ecliptic conversion
        obliquity = np.radians(23.439)
        
        # Get filters from main window
        mag_min, mag_max = self.parent_window.magnitude_panel.get_magnitude_limits()
        selected_classes = self.parent_window.neo_classes_panel.get_selected_classes()
        show_all = self.show_all_check.isChecked()
        
        # Filter asteroids by class first
        filtered_asteroids = []
        for ast in canvas.current_asteroids:
            cls = ast.get('orbit_class', 'Unknown')
            if cls in selected_classes:
                filtered_asteroids.append(ast)
        
        if not filtered_asteroids:
            return [], sun_info
        
        # Compute all positions at once using calculate_batch
        calculator = self.parent_window.calculator
        if calculator is None:
            return [], sun_info
        
        try:
            all_positions = calculator.calculate_batch(filtered_asteroids, jd)
        except Exception as e:
            logger.error(f"Position calculation error: {e}")
            return [], sun_info
        
        if all_positions is None or len(all_positions) == 0:
            return [], sun_info
        
        positions = []
        
        for i, ast in enumerate(filtered_asteroids):
            if i >= len(all_positions):
                break
            
            # Extract position data: [idx, ra, dec, dist, mag]
            ra = all_positions[i, 1]
            dec = all_positions[i, 2]
            geo_dist = all_positions[i, 3]  # geocentric distance
            mag = all_positions[i, 4]
            
            # Skip invalid data
            if np.isnan(ra) or np.isnan(dec) or np.isnan(geo_dist):
                continue
            
            # Skip invalid magnitudes (unless show_all)
            if not show_all:
                if mag is None or np.isnan(mag) or not (mag_min <= mag <= mag_max):
                    continue
            
            # Convert geocentric RA/Dec to radians
            ra_rad = np.radians(ra)
            dec_rad = np.radians(dec)
            
            # Calculate heliocentric distance using law of cosines
            cos_sep = (np.sin(dec_rad) * np.sin(sun_dec) + 
                      np.cos(dec_rad) * np.cos(sun_dec) * np.cos(ra_rad - sun_ra))
            cos_sep = np.clip(cos_sep, -1, 1)
            
            helio_dist_sq = geo_dist**2 + sun_dist**2 - 2 * geo_dist * sun_dist * cos_sep
            helio_dist = np.sqrt(abs(helio_dist_sq))
            
            # "Behind sun" = heliocentric distance < geocentric distance
            is_behind_sun = helio_dist < geo_dist
            
            # Apply hide_behind_sun filter
            if hide_behind_sun and is_behind_sun:
                continue
            
            # Calculate longitude based on coordinate system
            if use_true_helio:
                # True heliocentric: compute Sun→NEO vector and find its ecliptic longitude
                # NEO position in ICRS (geocentric)
                neo_x_geo = geo_dist * np.cos(dec_rad) * np.cos(ra_rad)
                neo_y_geo = geo_dist * np.cos(dec_rad) * np.sin(ra_rad)
                neo_z_geo = geo_dist * np.sin(dec_rad)
                
                # Convert geocentric to heliocentric (Sun-centered)
                # NEO_helio = Earth_pos + NEO_geo - Sun_pos
                # But Earth and Sun are in barycentric, NEO is geocentric
                # So: NEO_helio = NEO_geo + (Earth - Sun)_bary
                earth_sun_vec = sun_info['earth_pos_icrs'] - sun_info['sun_pos_icrs']
                
                neo_x_helio = neo_x_geo + earth_sun_vec[0]
                neo_y_helio = neo_y_geo + earth_sun_vec[1]
                neo_z_helio = neo_z_geo + earth_sun_vec[2]
                
                # Convert ICRS (equatorial) to ecliptic coordinates
                # Rotation around X axis by obliquity
                neo_y_ecl = neo_y_helio * np.cos(obliquity) + neo_z_helio * np.sin(obliquity)
                neo_x_ecl = neo_x_helio
                
                # Ecliptic longitude
                ecl_lon = np.degrees(np.arctan2(neo_y_ecl, neo_x_ecl)) % 360
            else:
                # Geocentric ecliptic longitude (original method)
                sin_ecl_lon = np.sin(ra_rad) * np.cos(obliquity) + np.tan(dec_rad) * np.sin(obliquity)
                cos_ecl_lon = np.cos(ra_rad)
                ecl_lon = np.degrees(np.arctan2(sin_ecl_lon, cos_ecl_lon)) % 360
            
            # Convert to radians for polar plot (angle from top, clockwise)
            theta = np.radians(90 - ecl_lon)
            
            positions.append({
                'theta': theta,
                'r': helio_dist,
                'mag': mag if mag and not np.isnan(mag) else 25.0,
                'ast': ast,
                'ra': ra,
                'dec': dec,
                'geo_dist': geo_dist,
                'ecl_lon': ecl_lon,
                'is_behind_sun': is_behind_sun
            })
        
        return positions, sun_info
    
    def update_plot(self):
        """Update the polar plot"""
        self.ax.clear()
        
        # Compute positions
        result = self.compute_heliocentric_positions(self.current_jd)
        positions, sun_info = result
        self.neo_positions = positions
        
        # Get date string
        try:
            ts = skyfield_load.timescale()
            t = ts.tt_jd(self.current_jd)
            date_str = t.utc_datetime().strftime('%Y-%m-%d %H:%M UTC')
        except:
            date_str = f"JD {self.current_jd:.2f}"
        
        self.date_label.setText(date_str)
        
        max_r = self.max_r_spin.value()
        show_orbits = self.show_orbits_check.isChecked()
        show_planets = self.show_planets_check.isChecked()
        mark_behind_sun = self.mark_behind_sun_check.isChecked()
        use_true_helio = self.coord_combo.currentIndex() == 1
        
        # Planet orbital radii (semi-major axes in AU)
        planet_orbits = {
            'Mercury': (0.387, '#A0522D'),  # sienna
            'Venus': (0.723, '#DEB887'),    # burlywood  
            'Earth': (1.000, '#4169E1'),    # royal blue
            'Mars': (1.524, '#CD5C5C'),     # indian red
            'Jupiter': (5.203, '#DAA520')   # goldenrod
        }
        
        # Draw orbital circles
        theta_circle = np.linspace(0, 2*np.pi, 100)
        if show_orbits:
            for name, (radius, color) in planet_orbits.items():
                if radius <= max_r:
                    self.ax.plot(theta_circle, np.ones_like(theta_circle) * radius, 
                                color=color, linestyle=':', linewidth=1, alpha=0.4)
        
        # Draw NEOs
        if self.density_mode and len(positions) > 10:
            # Density map mode - use full 0 to 2π range
            thetas = [p['theta'] for p in positions if p['r'] <= max_r]
            rs = [p['r'] for p in positions if p['r'] <= max_r]
            
            if len(thetas) > 0:
                # Normalize theta to 0 to 2π range for histogram
                thetas_norm = [(t + 2*np.pi) % (2*np.pi) for t in thetas]
                
                theta_bins = np.linspace(0, 2*np.pi, 37)  # 10° bins, full circle
                r_bins = np.linspace(0, max_r, 21)
                
                H, theta_edges, r_edges = np.histogram2d(thetas_norm, rs, bins=[theta_bins, r_bins])
                
                # Create mesh for pcolormesh
                theta_centers = (theta_edges[:-1] + theta_edges[1:]) / 2
                r_centers = (r_edges[:-1] + r_edges[1:]) / 2
                
                Theta, R = np.meshgrid(theta_centers, r_centers)
                self.ax.pcolormesh(Theta, R, H.T, cmap='hot', shading='auto', zorder=1)
        else:
            # Point mode
            if positions:
                # Separate normal and behind-sun NEOs
                normal = [p for p in positions if p['r'] <= max_r and not p.get('is_behind_sun', False)]
                behind_sun = [p for p in positions if p['r'] <= max_r and p.get('is_behind_sun', False)]
                
                # Plot normal NEOs
                if normal:
                    thetas = [p['theta'] for p in normal]
                    rs = [p['r'] for p in normal]
                    mags = [p['mag'] for p in normal]
                    
                    self.ax.scatter(thetas, rs, c=mags, cmap='viridis_r', 
                                   s=10, alpha=0.6, vmin=19, vmax=24, zorder=10)
                
                # Plot behind-sun NEOs (with different style if marking)
                if behind_sun:
                    thetas = [p['theta'] for p in behind_sun]
                    rs = [p['r'] for p in behind_sun]
                    mags = [p['mag'] for p in behind_sun]
                    
                    if mark_behind_sun:
                        # Hollow red circles for behind-sun
                        self.ax.scatter(thetas, rs, s=15, alpha=0.6,
                                       facecolors='none', edgecolors='red', linewidths=0.5, zorder=10)
                    else:
                        self.ax.scatter(thetas, rs, c=mags, cmap='viridis_r',
                                       s=10, alpha=0.6, vmin=19, vmax=24, zorder=10)
        
        # Draw Sun at center
        self.ax.plot(0, 0, 'o', color='#FFD700', markersize=15, 
                    markeredgecolor='#FFA500', markeredgewidth=2, zorder=100)
        
        # Draw planets (excluding Earth - Earth drawn separately)
        if show_planets and sun_info:
            self._draw_other_planets(sun_info, max_r)
        
        # ALWAYS draw Earth and opposition arrow
        if sun_info:
            self._draw_earth(sun_info, max_r)
        
        # Configure polar plot
        self.ax.set_theta_zero_location('N')  # 0° at top
        self.ax.set_theta_direction(-1)  # Clockwise
        self.ax.set_rlim(0, max_r)
        
        # Count behind-sun
        n_behind = len([p for p in positions if p.get('is_behind_sun', False)])
        coord_label = "helio" if use_true_helio else "geo"
        title = f"Heliocentric View ({len(positions)} NEOs, {coord_label} lon)"
        if mark_behind_sun and n_behind > 0:
            title = f"Heliocentric View ({len(positions)} NEOs, {n_behind} behind sun)"
        self.ax.set_title(title, pad=20)
        
        self.fig.tight_layout()
        self.canvas_widget.draw()
        
        self.status_label.setText(f"{len(positions)} NEOs displayed")
    
    def _draw_earth(self, sun_info, max_r):
        """Always draw Earth position and opposition arrow"""
        try:
            sun_ra = sun_info['ra']
            earth_theta = np.radians(90 - (sun_ra + 180) % 360)
            self.ax.plot(earth_theta, 1.0, 'o', color='#4169E1', markersize=10,
                        markeredgecolor='white', markeredgewidth=1.5, zorder=99, label='Earth')
            
            # Mark opposition direction (green arrow pointing outward from Earth)
            self.ax.annotate('', xy=(earth_theta, max_r * 0.95), xytext=(earth_theta, 1.2),
                           arrowprops=dict(arrowstyle='->', color='green', alpha=0.7, lw=2))
        except Exception as e:
            logger.debug(f"Error drawing Earth: {e}")
    
    def _draw_other_planets(self, sun_info, max_r):
        """Draw other planet positions (Mercury, Venus, Mars, Jupiter)"""
        try:
            t = sun_info['t']
            eph = sun_info['eph']
            
            # Planet data: name, ephemeris key, color, size
            planets = [
                ('Mercury', 'mercury', '#A0522D', 6),
                ('Venus', 'venus', '#DEB887', 8),
                ('Mars', 'mars', '#CD5C5C', 7),
                ('Jupiter', 'jupiter barycenter', '#DAA520', 10)
            ]
            
            sun = eph['sun']
            
            for name, eph_key, color, size in planets:
                try:
                    planet = eph[eph_key]
                    
                    # Planet's position from Sun
                    sun_pos = sun.at(t)
                    planet_pos = planet.at(t)
                    
                    # Vector from Sun to planet
                    sun_to_planet = planet_pos.position.au - sun_pos.position.au
                    
                    # Heliocentric distance
                    helio_dist = np.sqrt(np.sum(sun_to_planet**2))
                    
                    if helio_dist > max_r:
                        continue
                    
                    # Ecliptic longitude (x-y plane in ICRS is equatorial, need rotation)
                    # For now use simplified version
                    ecl_lon = np.degrees(np.arctan2(sun_to_planet[1], sun_to_planet[0])) % 360
                    
                    theta = np.radians(90 - ecl_lon)
                    
                    self.ax.plot(theta, helio_dist, 'o', color=color, markersize=size,
                                markeredgecolor='white', markeredgewidth=0.5, zorder=99,
                                label=name)
                except Exception as e:
                    logger.debug(f"Could not plot {name}: {e}")
            
        except Exception as e:
            logger.error(f"Planet drawing error: {e}")
    
    def on_click(self, event):
        """Handle click on plot for object identification"""
        if event.inaxes != self.ax or self.density_mode:
            return
        
        # Get click position in polar coordinates
        click_theta = event.xdata  # radians
        click_r = event.ydata  # AU
        
        if click_theta is None or click_r is None:
            return
        
        # Find nearest NEO
        min_dist = float('inf')
        nearest = None
        
        for pos in self.neo_positions:
            # Angular distance (accounting for wrap-around)
            d_theta = abs(pos['theta'] - click_theta)
            if d_theta > np.pi:
                d_theta = 2 * np.pi - d_theta
            
            # Scale theta difference by radius for distance calculation
            dist = np.sqrt((d_theta * pos['r'])**2 + (pos['r'] - click_r)**2)
            
            if dist < min_dist:
                min_dist = dist
                nearest = pos
        
        # Check if close enough (within 0.3 AU equivalent)
        if nearest and min_dist < 0.3:
            ast = nearest['ast']
            mag = nearest['mag']
            helio_r = nearest['r']
            geo_dist = nearest['geo_dist']
            is_behind_sun = nearest.get('is_behind_sun', False)
            
            designation = ast.get('designation', 'Unknown')
            cls = ast.get('orbit_class', 'Unknown')
            
            status = f"{designation} ({cls}) | V={mag:.1f} | Helio: {helio_r:.3f} AU | Geo: {geo_dist:.3f} AU"
            if is_behind_sun:
                status += " | BEHIND SUN"
            
            self.status_label.setText(status)
    
    def closeEvent(self, event):
        """Clean up on close"""
        self.animation_timer.stop()
        super().closeEvent(event)




class MagnitudeRangesPanel(QWidget):
    """Compact magnitude ranges panel for main window"""
    
    filters_changed = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)  # No margins for alignment
        layout.setSpacing(0)
        
        # Magnitude Ranges group (compact)
        mag_group = QGroupBox("Magnitude Ranges")
        mag_layout = QVBoxLayout()
        mag_layout.setSpacing(2)
        
        # V magnitude on one line
        v_row = QHBoxLayout()
        v_row.addWidget(QLabel("V:"))
        self.mag_min_spin = QDoubleSpinBox()
        self.mag_min_spin.setRange(5.0, 30.0)
        self.mag_min_spin.setValue(19.0)
        self.mag_min_spin.setDecimals(2)
        self.mag_min_spin.setSingleStep(0.25)
        self.mag_min_spin.setMaximumWidth(65)
        self.mag_min_spin.valueChanged.connect(self.on_filters_changed)
        v_row.addWidget(self.mag_min_spin)
        
        v_row.addWidget(QLabel("-"))
        self.mag_max_spin = QDoubleSpinBox()
        self.mag_max_spin.setRange(5.0, 30.0)
        self.mag_max_spin.setValue(22.0)
        self.mag_max_spin.setDecimals(2)
        self.mag_max_spin.setSingleStep(0.25)
        self.mag_max_spin.setMaximumWidth(65)
        self.mag_max_spin.valueChanged.connect(self.on_filters_changed)
        v_row.addWidget(self.mag_max_spin)
        
        v_reset = QPushButton("reset")
        v_reset.setMaximumWidth(45)
        v_reset.clicked.connect(self.reset_v_mag)
        v_row.addWidget(v_reset)
        v_row.addStretch()
        mag_layout.addLayout(v_row)
        
        # H magnitude on one line
        h_row = QHBoxLayout()
        h_row.addWidget(QLabel("H:"))
        self.h_min_spin = QDoubleSpinBox()
        self.h_min_spin.setRange(5.0, 35.0)
        self.h_min_spin.setValue(9.0)
        self.h_min_spin.setDecimals(2)
        self.h_min_spin.setSingleStep(0.25)
        self.h_min_spin.setMaximumWidth(65)
        self.h_min_spin.valueChanged.connect(self.on_filters_changed)
        h_row.addWidget(self.h_min_spin)
        
        h_row.addWidget(QLabel("-"))
        self.h_max_spin = QDoubleSpinBox()
        self.h_max_spin.setRange(5.0, 35.0)
        self.h_max_spin.setValue(22.0)
        self.h_max_spin.setDecimals(2)
        self.h_max_spin.setSingleStep(0.25)
        self.h_max_spin.setMaximumWidth(65)
        self.h_max_spin.valueChanged.connect(self.on_filters_changed)
        h_row.addWidget(self.h_max_spin)
        
        h_reset = QPushButton("reset")
        h_reset.setMaximumWidth(45)
        h_reset.clicked.connect(self.reset_h_mag)
        h_row.addWidget(h_reset)
        h_row.addStretch()
        mag_layout.addLayout(h_row)
        
        # Show all NEOs checkbox (bypasses V/H filtering)
        self.show_all_neos_check = QCheckBox("Show all NEOs (override V/H)")
        self.show_all_neos_check.setChecked(False)
        self.show_all_neos_check.setToolTip("Display all NEOs regardless of V and H magnitude limits")
        self.show_all_neos_check.stateChanged.connect(self.on_show_all_changed)
        mag_layout.addWidget(self.show_all_neos_check)
        
        # Show only discoveries per lunation
        self.lunation_discoveries_check = QCheckBox("Show discoveries per lunation")
        self.lunation_discoveries_check.setChecked(False)
        self.lunation_discoveries_check.setToolTip("Show only NEOs discovered during the current lunation (CLN)")
        self.lunation_discoveries_check.stateChanged.connect(self.on_filters_changed)
        mag_layout.addWidget(self.lunation_discoveries_check)
        
        mag_group.setLayout(mag_layout)
        layout.addWidget(mag_group)
        
        self.setLayout(layout)
    
    def reset_v_mag(self):
        """Reset V magnitude to defaults (19.0 - 22.0)"""
        self.mag_min_spin.setValue(19.0)
        self.mag_max_spin.setValue(22.0)
    
    def reset_h_mag(self):
        """Reset H magnitude to defaults (9.0 - 22.0)"""
        self.h_min_spin.setValue(9.0)
        self.h_max_spin.setValue(22.0)
    
    def get_magnitude_limits(self):
        return self.mag_min_spin.value(), self.mag_max_spin.value()
    
    def get_h_limits(self):
        return self.h_min_spin.value(), self.h_max_spin.value()
    
    def get_show_all_neos(self):
        """Return whether to show all NEOs regardless of V/H limits"""
        return self.show_all_neos_check.isChecked()
    
    def get_lunation_discoveries_only(self):
        """Return whether to show only NEOs discovered during current lunation"""
        return self.lunation_discoveries_check.isChecked()
    
    def on_show_all_changed(self, state):
        """Handle show all NEOs checkbox change"""
        show_all = (state == 2)  # Qt.Checked = 2
        # Disable/enable magnitude controls when showing all
        self.mag_min_spin.setEnabled(not show_all)
        self.mag_max_spin.setEnabled(not show_all)
        self.h_min_spin.setEnabled(not show_all)
        self.h_max_spin.setEnabled(not show_all)
        self.filters_changed.emit()
    
    def on_filters_changed(self):
        self.filters_changed.emit()


class NEOClassesPanel(QWidget):
    """Standalone NEO/PHA Classes panel for main window"""
    
    filters_changed = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)  # No margins for alignment
        layout.setSpacing(0)
        
        # NEO/PHA Classes group
        class_group = QGroupBox("NEO/PHA Classes")
        class_layout = QVBoxLayout()
        
        # Main NEO classes in grid (3 columns)
        self.class_checks = {}
        neo_grid = QGridLayout()
        neo_classes = [
            ('Atira', 0, 0),
            ('Aten', 0, 1),
            ('Apollo', 0, 2),
            ('Amor, q≤1.15', 1, 0),
            ('Amor, q>1.15', 1, 1),
        ]
        
        for cls, row, col in neo_classes:
            cb = QCheckBox(cls)
            cb.setChecked(True)
            cb.stateChanged.connect(self.on_filters_changed)
            self.class_checks[cls] = cb
            neo_grid.addWidget(cb, row, col)
        
        # All button in grid
        all_btn = QPushButton("All Classes")
        all_btn.setMaximumWidth(80)
        all_btn.clicked.connect(self.select_all)
        neo_grid.addWidget(all_btn, 1, 2)
        
        class_layout.addLayout(neo_grid)
        
        # MOID filter controls below NEO classes
        moid_row = QHBoxLayout()
        
        self.moid_enabled_check = QCheckBox("MOID filter")
        self.moid_enabled_check.setChecked(False)
        self.moid_enabled_check.setToolTip("Filter by Earth MOID (Minimum Orbit Intersection Distance)")
        self.moid_enabled_check.stateChanged.connect(self.on_filters_changed)
        moid_row.addWidget(self.moid_enabled_check)
        
        moid_row.addWidget(QLabel("Min:"))
        self.moid_min_spin = QDoubleSpinBox()
        self.moid_min_spin.setRange(0.0, 1.0)
        self.moid_min_spin.setValue(0.0)
        self.moid_min_spin.setDecimals(3)
        self.moid_min_spin.setSingleStep(0.01)
        self.moid_min_spin.setSuffix(" AU")
        self.moid_min_spin.setToolTip("Minimum Earth MOID in AU")
        self.moid_min_spin.valueChanged.connect(self.on_filters_changed)
        moid_row.addWidget(self.moid_min_spin)
        
        moid_row.addWidget(QLabel("Max:"))
        self.moid_max_spin = QDoubleSpinBox()
        self.moid_max_spin.setRange(0.0, 1.0)
        self.moid_max_spin.setValue(0.05)
        self.moid_max_spin.setDecimals(3)
        self.moid_max_spin.setSingleStep(0.01)
        self.moid_max_spin.setSuffix(" AU")
        self.moid_max_spin.setToolTip("Maximum Earth MOID in AU")
        self.moid_max_spin.valueChanged.connect(self.on_filters_changed)
        moid_row.addWidget(self.moid_max_spin)
        
        class_layout.addLayout(moid_row)
        
        # Display options row (moved from Magnitude Ranges)
        display_row = QHBoxLayout()
        
        self.show_hollow_check = QCheckBox("Show behind sun")
        self.show_hollow_check.setChecked(False)
        self.show_hollow_check.setToolTip("Show hollow circles for NEOs on far side of Sun")
        display_row.addWidget(self.show_hollow_check)
        
        self.hide_before_discovery_check = QCheckBox("Hide before discovery")
        self.hide_before_discovery_check.setChecked(False)
        self.hide_before_discovery_check.setToolTip("Hide NEOs at dates before their discovery")
        display_row.addWidget(self.hide_before_discovery_check)
        
        display_row.addStretch()
        class_layout.addLayout(display_row)
        
        class_group.setLayout(class_layout)
        layout.addWidget(class_group)
        
        self.setLayout(layout)
    
    def get_selected_classes(self):
        """Returns list of selected classes"""
        selected = [c for c, cb in self.class_checks.items() if cb.isChecked()]
        if not selected:
            return None
        return selected
    
    def get_moid_filter(self):
        """Get Earth MOID filter settings"""
        enabled = self.moid_enabled_check.isChecked()
        return enabled, self.moid_min_spin.value(), self.moid_max_spin.value()
    
    def select_all(self):
        for cb in self.class_checks.values():
            cb.setChecked(True)
    
    def on_filters_changed(self):
        self.filters_changed.emit()
    
    def mousePressEvent(self, event):
        """Handle mouse clicks - cycle classes if click is in empty area"""
        # Get the clicked widget
        child = self.childAt(event.pos())
        
        # If click is on empty space (not on checkboxes, spinboxes, buttons, labels)
        if child is None or isinstance(child, QGroupBox):
            self.cycle_classes()
        else:
            widget_type = type(child).__name__
            if widget_type not in ['QCheckBox', 'QSpinBox', 'QDoubleSpinBox', 'QLabel', 
                                   'QPushButton', 'QLineEdit']:
                self.cycle_classes()
        
        super().mousePressEvent(event)
    
    def cycle_classes(self):
        """Cycle through NEO classes one at a time: Atira -> Aten -> Apollo -> Amor q≤1.15 -> Amor q>1.15 -> Atira"""
        # Define the cycle order
        cycle_order = ['Atira', 'Aten', 'Apollo', 'Amor, q≤1.15', 'Amor, q>1.15']
        
        # Get currently selected classes
        selected = [c for c, cb in self.class_checks.items() if cb.isChecked()]
        
        # Determine starting point
        if len(selected) == 1 and selected[0] in cycle_order:
            # Start from the single selected class
            current_idx = cycle_order.index(selected[0])
        else:
            # Start with Atira (index -1 so next will be 0)
            current_idx = -1
        
        # Move to next class in cycle
        next_idx = (current_idx + 1) % len(cycle_order)
        next_class = cycle_order[next_idx]
        
        # Uncheck all, then check only the next class
        for cls, cb in self.class_checks.items():
            cb.setChecked(cls == next_class)



class OrbitalElementsPanel(QWidget):
    """Orbital elements filters (Period, Eccentricity, Inclination) with min/max spinboxes"""
    
    filters_changed = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)
        
        # Period filter (in years) - min/max spinboxes
        period_row = QHBoxLayout()
        period_row.addWidget(QLabel("Period:"))
        
        self.period_min_spin = QDoubleSpinBox()
        self.period_min_spin.setRange(0.0, 100.0)
        self.period_min_spin.setValue(0.0)
        self.period_min_spin.setDecimals(3)
        self.period_min_spin.setSingleStep(0.1)
        self.period_min_spin.setSuffix(" yr")
        self.period_min_spin.setMaximumWidth(85)
        self.period_min_spin.valueChanged.connect(self.on_filters_changed)
        period_row.addWidget(self.period_min_spin)
        
        period_row.addWidget(QLabel("-"))
        
        self.period_max_spin = QDoubleSpinBox()
        self.period_max_spin.setRange(0.0, 100.0)
        self.period_max_spin.setValue(10.0)
        self.period_max_spin.setDecimals(3)
        self.period_max_spin.setSingleStep(0.1)
        self.period_max_spin.setSuffix(" yr")
        self.period_max_spin.setMaximumWidth(85)
        self.period_max_spin.valueChanged.connect(self.on_filters_changed)
        period_row.addWidget(self.period_max_spin)
        
        period_default = QPushButton("reset")
        period_default.setMaximumWidth(60)
        period_default.clicked.connect(self.clear_period)
        period_row.addWidget(period_default)
        period_row.addStretch()
        
        layout.addLayout(period_row)
        
        # Eccentricity filter - min/max spinboxes
        ecc_row = QHBoxLayout()
        ecc_row.addWidget(QLabel("Eccentricity:"))
        
        self.ecc_min_spin = QDoubleSpinBox()
        self.ecc_min_spin.setRange(0.0, 1.0)
        self.ecc_min_spin.setValue(0.0)
        self.ecc_min_spin.setDecimals(3)
        self.ecc_min_spin.setSingleStep(0.01)
        self.ecc_min_spin.setMaximumWidth(70)
        self.ecc_min_spin.valueChanged.connect(self.on_filters_changed)
        ecc_row.addWidget(self.ecc_min_spin)
        
        ecc_row.addWidget(QLabel("-"))
        
        self.ecc_max_spin = QDoubleSpinBox()
        self.ecc_max_spin.setRange(0.0, 1.0)
        self.ecc_max_spin.setValue(1.0)
        self.ecc_max_spin.setDecimals(3)
        self.ecc_max_spin.setSingleStep(0.01)
        self.ecc_max_spin.setMaximumWidth(70)
        self.ecc_max_spin.valueChanged.connect(self.on_filters_changed)
        ecc_row.addWidget(self.ecc_max_spin)
        
        ecc_default = QPushButton("reset")
        ecc_default.setMaximumWidth(60)
        ecc_default.clicked.connect(self.clear_ecc)
        ecc_row.addWidget(ecc_default)
        ecc_row.addStretch()
        
        layout.addLayout(ecc_row)
        
        # Inclination filter - min/max spinboxes
        inc_row = QHBoxLayout()
        inc_row.addWidget(QLabel("Inclination:"))
        
        self.inc_min_spin = QDoubleSpinBox()
        self.inc_min_spin.setRange(0.0, 180.0)
        self.inc_min_spin.setValue(0.0)
        self.inc_min_spin.setDecimals(3)
        self.inc_min_spin.setSingleStep(1.0)
        self.inc_min_spin.setSuffix("°")
        self.inc_min_spin.setMaximumWidth(75)
        self.inc_min_spin.valueChanged.connect(self.on_filters_changed)
        inc_row.addWidget(self.inc_min_spin)
        
        inc_row.addWidget(QLabel("-"))
        
        self.inc_max_spin = QDoubleSpinBox()
        self.inc_max_spin.setRange(0.0, 180.0)
        self.inc_max_spin.setValue(180.0)
        self.inc_max_spin.setDecimals(3)
        self.inc_max_spin.setSingleStep(1.0)
        self.inc_max_spin.setSuffix("°")
        self.inc_max_spin.setMaximumWidth(75)
        self.inc_max_spin.valueChanged.connect(self.on_filters_changed)
        inc_row.addWidget(self.inc_max_spin)
        
        inc_default = QPushButton("reset")
        inc_default.setMaximumWidth(60)
        inc_default.clicked.connect(self.clear_inc)
        inc_row.addWidget(inc_default)
        inc_row.addStretch()
        
        layout.addLayout(inc_row)
        
        self.setLayout(layout)
    
    def clear_period(self):
        self.period_min_spin.setValue(0.0)
        self.period_max_spin.setValue(10.0)
    
    def clear_ecc(self):
        self.ecc_min_spin.setValue(0.0)
        self.ecc_max_spin.setValue(1.0)
    
    def clear_inc(self):
        self.inc_min_spin.setValue(0.0)
        self.inc_max_spin.setValue(180.0)
    
    def get_orbital_filters(self):
        """Get orbital element filter settings"""
        period_min = self.period_min_spin.value()
        period_max = self.period_max_spin.value()
        ecc_min = self.ecc_min_spin.value()
        ecc_max = self.ecc_max_spin.value()
        inc_min = self.inc_min_spin.value()
        inc_max = self.inc_max_spin.value()
        
        # Filter is active if range is not at full extent
        return {
            'period_enabled': period_min > 0.0 or period_max < 10.0,
            'period_min': period_min,
            'period_max': period_max,
            'ecc_enabled': ecc_min > 0.0 or ecc_max < 1.0,
            'ecc_min': ecc_min,
            'ecc_max': ecc_max,
            'inc_enabled': inc_min > 0.0 or inc_max < 180.0,
            'inc_min': inc_min,
            'inc_max': inc_max
        }
    
    def set_ranges_from_data(self, period_range, ecc_range, inc_range):
        """Update ranges based on actual data"""
        # Not needed anymore - using fixed defaults
        pass
    
    def on_filters_changed(self):
        self.filters_changed.emit()


class ProjectionPanel(QWidget):
    """Map projection and coordinate system controls (minimal for main window)"""
    
    settings_changed = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Map Projection and Coordinates - COMPACT (both in one group)
        proj_group = QGroupBox("Map Projection")
        proj_layout = QVBoxLayout()
        
        # Projection selector (left-justified)
        proj_row = QHBoxLayout()
        proj_row.addWidget(QLabel("Type:"))
        self.proj_combo = QComboBox()
        self.proj_combo.addItems([
            'Rectangular',
            'Hammer',
            'Aitoff',
            'Mollweide'
        ])
        self.proj_combo.setCurrentText('Hammer')  # Default to Hammer
        self.proj_combo.currentTextChanged.connect(self.on_settings_changed)
        proj_row.addWidget(self.proj_combo)
        proj_row.addStretch()  # Left-justify
        proj_layout.addLayout(proj_row)
        
        # Coordinate system selector (left-justified)
        coord_row = QHBoxLayout()
        coord_row.addWidget(QLabel("Coords:"))
        self.coord_combo = QComboBox()
        self.coord_combo.addItems(['Equatorial', 'Ecliptic', 'Galactic', 'Opposition'])
        self.coord_combo.currentTextChanged.connect(self.on_settings_changed)
        coord_row.addWidget(self.coord_combo)
        coord_row.addStretch()  # Left-justify
        proj_layout.addLayout(coord_row)
        
        # Grid spacing options (horizontal and vertical in degrees)
        res_row = QHBoxLayout()
        res_row.addWidget(QLabel("Grid:"))
        res_row.addWidget(QLabel("H:"))
        self.h_resolution = QDoubleSpinBox()
        self.h_resolution.setRange(1, 90)
        self.h_resolution.setValue(30)
        self.h_resolution.setDecimals(0)
        self.h_resolution.setSingleStep(15)
        self.h_resolution.setSuffix("°")
        self.h_resolution.setMaximumWidth(55)
        self.h_resolution.setToolTip("Horizontal grid spacing in degrees")
        self.h_resolution.valueChanged.connect(self.on_settings_changed)
        res_row.addWidget(self.h_resolution)
        
        res_row.addWidget(QLabel("V:"))
        self.v_resolution = QDoubleSpinBox()
        self.v_resolution.setRange(1, 90)
        self.v_resolution.setValue(15)  # Default: 15 degrees vertical
        self.v_resolution.setDecimals(0)
        self.v_resolution.setSingleStep(15)
        self.v_resolution.setSuffix("°")
        self.v_resolution.setMaximumWidth(55)
        self.v_resolution.setToolTip("Vertical grid spacing in degrees")
        self.v_resolution.valueChanged.connect(self.on_settings_changed)
        res_row.addWidget(self.v_resolution)
        res_row.addStretch()  # Left-justify
        proj_layout.addLayout(res_row)
        
        # Display mode selector with settings
        display_row = QHBoxLayout()
        display_row.addWidget(QLabel("Display:"))
        self.display_mode = QComboBox()
        self.display_mode.addItems(['Points', 'Density Map', 'Contours', 'Points + Contours'])
        self.display_mode.setCurrentText('Points')
        self.display_mode.setToolTip("How to visualize NEO positions:\n"
                                     "Points: Individual symbols\n"
                                     "Density Map: Heat map of clustering\n"
                                     "Contours: Density contour lines\n"
                                     "Points + Contours: Both overlaid")
        self.display_mode.currentTextChanged.connect(self.on_settings_changed)
        display_row.addWidget(self.display_mode)
        
        # Settings button for density/contour options
        self.display_settings_btn = QPushButton("⚙")
        self.display_settings_btn.setMaximumWidth(25)
        self.display_settings_btn.setToolTip("Display mode settings")
        display_settings_menu = QMenu(self)
        
        # Density map settings - flat menu with Grid Size submenu
        density_menu = display_settings_menu.addMenu("Density Map")
        
        # Grid size (higher = finer detail)
        self.density_gridsize_group = density_menu.addMenu("Resolution")
        for size, label in [(50, "Fine (50)"), (35, "Medium (35)"), (25, "Coarse (25)"), (15, "Very Coarse (15)")]:
            action = self.density_gridsize_group.addAction(label)
            action.setCheckable(True)
            action.setChecked(size == 35)  # Default medium
            action.triggered.connect(lambda checked, s=size: self.set_density_gridsize(s))
        
        # Colormap
        self.density_cmap_group = density_menu.addMenu("Colormap")
        for cmap, label in [('YlOrRd', 'Yellow-Orange-Red'), ('hot', 'Hot'), ('plasma', 'Plasma'), 
                            ('viridis', 'Viridis'), ('Blues', 'Blues'), ('Greys', 'Greys')]:
            action = self.density_cmap_group.addAction(label)
            action.setCheckable(True)
            action.setChecked(cmap == 'YlOrRd')  # Default
            action.triggered.connect(lambda checked, c=cmap: self.set_density_colormap(c))
        
        # Scale range - just Auto and Manual
        self.density_scale_group = density_menu.addMenu("Scale Range")
        self.density_scale_auto = self.density_scale_group.addAction("Auto")
        self.density_scale_auto.setCheckable(True)
        self.density_scale_auto.setChecked(True)
        self.density_scale_auto.triggered.connect(lambda: self.set_density_scale('auto'))
        
        self.density_scale_manual = self.density_scale_group.addAction("Manual...")
        self.density_scale_manual.setCheckable(True)
        self.density_scale_manual.triggered.connect(self.show_density_scale_dialog)
        
        # Contour settings submenu
        contour_menu = display_settings_menu.addMenu("Contours")
        self.contour_levels_group = contour_menu.addMenu("Levels")
        for levels, label in [(5, "Few (5)"), (8, "Medium (8)"), (12, "Many (12)"), (20, "Dense (20)")]:
            action = self.contour_levels_group.addAction(label)
            action.setCheckable(True)
            action.setChecked(levels == 8)  # Default
            action.triggered.connect(lambda checked, l=levels: self.set_contour_levels(l))
        
        self.contour_smooth_group = contour_menu.addMenu("Smoothing")
        for smooth, label in [(0.08, "Sharp"), (0.15, "Medium"), (0.25, "Smooth"), (0.4, "Very Smooth")]:
            action = self.contour_smooth_group.addAction(label)
            action.setCheckable(True)
            action.setChecked(smooth == 0.15)  # Default
            action.triggered.connect(lambda checked, s=smooth: self.set_contour_smoothing(s))
        
        self.display_settings_btn.setMenu(display_settings_menu)
        display_row.addWidget(self.display_settings_btn)
        
        display_row.addStretch()
        proj_layout.addLayout(display_row)
        
        # Initialize display settings
        self.density_gridsize = 35
        self.density_colormap = 'YlOrRd'
        self.density_scale = 'auto'  # 'auto' or (min, max) tuple
        self.contour_levels = 8
        self.contour_smoothing = 0.15
        
        proj_group.setLayout(proj_layout)
        layout.addWidget(proj_group)
        
        # Store reference for click handling
        self.proj_group = proj_group
        
        self.setLayout(layout)
    
    def get_projection(self):
        return self.proj_combo.currentText().lower()
    
    def get_coordinate_system(self):
        return self.coord_combo.currentText().lower()
    
    def get_resolution(self):
        """Return horizontal and vertical resolution in degrees"""
        return int(self.h_resolution.value()), int(self.v_resolution.value())
    
    def get_display_mode(self):
        """Return display mode: 'points', 'density', 'contours', or 'points+contours'"""
        mode = self.display_mode.currentText().lower()
        if 'density' in mode:
            return 'density'
        elif '+' in mode:
            return 'points+contours'
        elif 'contour' in mode:
            return 'contours'
        return 'points'
    
    def get_display_settings(self):
        """Return display settings dict"""
        return {
            'density_gridsize': getattr(self, 'density_gridsize', 35),
            'density_colormap': getattr(self, 'density_colormap', 'YlOrRd'),
            'density_scale': getattr(self, 'density_scale', 'auto'),
            'contour_levels': getattr(self, 'contour_levels', 8),
            'contour_smoothing': getattr(self, 'contour_smoothing', 0.15)
        }
    
    def set_density_gridsize(self, size):
        """Set density map grid size (higher = finer)"""
        self.density_gridsize = size
        # Update checkmarks
        for action in self.density_gridsize_group.actions():
            action.setChecked(str(size) in action.text())
        self.on_settings_changed()
    
    def set_density_colormap(self, cmap):
        """Set density map colormap"""
        self.density_colormap = cmap
        # Update checkmarks
        for action in self.density_cmap_group.actions():
            action.setChecked(cmap in action.text() or 
                            (cmap == 'YlOrRd' and 'Yellow' in action.text()))
        self.on_settings_changed()
    
    def set_density_scale(self, scale):
        """Set density map scale range - 'auto' or (min, max) tuple"""
        self.density_scale = scale
        # Update checkmarks for Auto vs Manual
        self.density_scale_auto.setChecked(scale == 'auto')
        self.density_scale_manual.setChecked(scale != 'auto')
        self.on_settings_changed()
    
    def show_density_scale_dialog(self):
        """Show dialog for manual density scale range"""
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QPushButton
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Manual Scale Range")
        dialog.setModal(True)
        
        layout = QVBoxLayout()
        
        # Get current values - try to get from canvas data range if available
        current = getattr(self, 'density_scale', 'auto')
        if isinstance(current, tuple):
            current_min, current_max = current
        else:
            # Try to get the auto range from the canvas
            current_min, current_max = 1, 50  # Default fallback
            try:
                parent = self.parent()
                while parent and not hasattr(parent, 'canvas'):
                    parent = parent.parent()
                if parent and hasattr(parent, 'canvas'):
                    canvas = parent.canvas
                    if hasattr(canvas, 'density_hexbin') and canvas.density_hexbin is not None:
                        # Get the actual data range from the density plot
                        array = canvas.density_hexbin.get_array()
                        if array is not None and len(array) > 0:
                            valid = array[array > 0]  # Exclude zeros
                            if len(valid) > 0:
                                current_min = int(valid.min())
                                current_max = int(valid.max())
            except:
                pass
        
        # Info label
        info_label = QLabel(f"Current data range: {current_min} - {current_max}")
        info_label.setStyleSheet("color: gray; font-style: italic;")
        layout.addWidget(info_label)
        
        # Min value
        min_row = QHBoxLayout()
        min_row.addWidget(QLabel("Minimum:"))
        min_spin = QSpinBox()
        min_spin.setRange(0, 10000)
        min_spin.setValue(current_min)
        min_row.addWidget(min_spin)
        layout.addLayout(min_row)
        
        # Max value
        max_row = QHBoxLayout()
        max_row.addWidget(QLabel("Maximum:"))
        max_spin = QSpinBox()
        max_spin.setRange(1, 10000)
        max_spin.setValue(current_max)
        max_row.addWidget(max_spin)
        layout.addLayout(max_row)
        
        # Buttons
        btn_row = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(dialog.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)
        
        dialog.setLayout(layout)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.set_density_scale((min_spin.value(), max_spin.value()))
    
    def set_contour_levels(self, levels):
        """Set number of contour levels"""
        self.contour_levels = levels
        # Update checkmarks
        for action in self.contour_levels_group.actions():
            action.setChecked(str(levels) in action.text())
        self.on_settings_changed()
    
    def set_contour_smoothing(self, smoothing):
        """Set contour smoothing (KDE bandwidth)"""
        self.contour_smoothing = smoothing
        # Update checkmarks
        labels = {0.08: "Sharp", 0.15: "Medium", 0.25: "Smooth", 0.4: "Very Smooth"}
        for action in self.contour_smooth_group.actions():
            action.setChecked(labels.get(smoothing, "") in action.text())
        self.on_settings_changed()
    
    def reset_defaults(self):
        """Reset to default values"""
        self.proj_combo.setCurrentText('Hammer')
        self.coord_combo.setCurrentText('Equatorial')
        self.h_resolution.setValue(30)
        self.v_resolution.setValue(15)
        self.display_mode.setCurrentText('Points')
        self.density_gridsize = 35
        self.density_colormap = 'YlOrRd'
        self.density_scale = 'auto'
        self.contour_levels = 8
        self.contour_smoothing = 0.15
    
    def mousePressEvent(self, event):
        """Handle mouse clicks - cycle projections if click is in empty area"""
        # Get the clicked widget
        child = self.childAt(event.pos())
        
        # If click is on the panel but not on a specific control widget, cycle projection
        # Check if click is within the group box area
        if child is None or isinstance(child, QGroupBox):
            self.cycle_projection()
        else:
            # Check if click is on empty space within group box
            # (not on combo boxes, spinboxes, labels, or buttons)
            widget_type = type(child).__name__
            if widget_type not in ['QComboBox', 'QSpinBox', 'QDoubleSpinBox', 'QLabel', 
                                   'QPushButton', 'QLineEdit', 'QCheckBox']:
                self.cycle_projection()
        
        super().mousePressEvent(event)
    
    def cycle_projection(self):
        """Cycle through projection types: Rectangular -> Hammer -> Aitoff -> Mollweide -> Rectangular"""
        projections = ['Rectangular', 'Hammer', 'Aitoff', 'Mollweide']
        current = self.proj_combo.currentText()
        try:
            current_idx = projections.index(current)
            next_idx = (current_idx + 1) % len(projections)
            self.proj_combo.setCurrentText(projections[next_idx])
        except ValueError:
            self.proj_combo.setCurrentText('Hammer')
    
    def on_settings_changed(self):
        self.settings_changed.emit()


class ColorbarPanel(QWidget):
    """Colorbar and symbol size controls (for Settings dialog)"""
    
    settings_changed = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)
        
        # Color by selection
        colorby_row = QHBoxLayout()
        colorby_row.addWidget(QLabel("Color by:"))
        self.color_by_combo = QComboBox()
        self.color_by_combo.addItems([
            'V magnitude',
            'CNEOS Discovery Site'
        ])
        self.color_by_combo.setToolTip("Select what property controls symbol color")
        self.color_by_combo.currentTextChanged.connect(self.on_color_by_changed)
        colorby_row.addWidget(self.color_by_combo)
        colorby_row.addStretch()
        layout.addLayout(colorby_row)
        
        # Colormap selection (left-justified) - only for V magnitude mode
        cmap_row = QHBoxLayout()
        self.cmap_label = QLabel("Colormap:")
        cmap_row.addWidget(self.cmap_label)
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems([
            'viridis_r',
            'plasma_r',
            'inferno_r',
            'magma_r',
            'cividis_r',
            'twilight',
            'coolwarm',
            'RdYlBu_r'
        ])
        self.cmap_combo.currentTextChanged.connect(self.on_settings_changed)
        cmap_row.addWidget(self.cmap_combo)
        cmap_row.addStretch()
        layout.addLayout(cmap_row)
        
        # Min and Max on same line (left-justified) - only for V magnitude mode
        range_row = QHBoxLayout()
        self.range_min_label = QLabel("Min:")
        range_row.addWidget(self.range_min_label)
        self.cbar_min = QDoubleSpinBox()
        self.cbar_min.setRange(5, 30)
        self.cbar_min.setValue(19.0)
        self.cbar_min.setDecimals(1)
        self.cbar_min.setSingleStep(0.5)
        self.cbar_min.setMaximumWidth(60)
        self.cbar_min.valueChanged.connect(self.on_settings_changed)
        range_row.addWidget(self.cbar_min)
        
        self.range_max_label = QLabel("Max:")
        range_row.addWidget(self.range_max_label)
        self.cbar_max = QDoubleSpinBox()
        self.cbar_max.setRange(5, 30)
        self.cbar_max.setValue(23)
        self.cbar_max.setDecimals(1)
        self.cbar_max.setSingleStep(0.5)
        self.cbar_max.setMaximumWidth(60)
        self.cbar_max.valueChanged.connect(self.on_settings_changed)
        range_row.addWidget(self.cbar_max)
        
        self.cbar_reset = QPushButton("reset")
        self.cbar_reset.setMaximumWidth(45)
        self.cbar_reset.clicked.connect(self.reset_colorbar)
        range_row.addWidget(self.cbar_reset)
        range_row.addStretch()
        layout.addLayout(range_row)
        
        # Legend checkbox (for CNEOS mode)
        legend_row = QHBoxLayout()
        self.show_legend_check = QCheckBox("Show legend")
        self.show_legend_check.setChecked(True)
        self.show_legend_check.setToolTip("Show/hide discovery site legend on plot")
        self.show_legend_check.stateChanged.connect(self.on_settings_changed)
        self.show_legend_check.setVisible(False)  # Hidden until CNEOS mode selected
        legend_row.addWidget(self.show_legend_check)
        
        # CNEOS link
        self.cneos_link = QLabel('<a href="https://cneos.jpl.nasa.gov/stats/site_all.html">CNEOS Stats</a>')
        self.cneos_link.setOpenExternalLinks(True)
        self.cneos_link.setVisible(False)  # Hidden until CNEOS mode selected
        legend_row.addWidget(self.cneos_link)
        legend_row.addStretch()
        layout.addLayout(legend_row)
        
        # Symbol Size controls
        layout.addWidget(QLabel(""))  # Spacer
        size_label = QLabel("<b>Symbol Size</b>")
        layout.addWidget(size_label)
        
        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Based on:"))
        self.size_combo = QComboBox()
        self.size_combo.addItems([
            'V magnitude',
            'H magnitude',
            'Distance',
            'Earth MOID',
            'Period',
            'Eccentricity'
        ])
        self.size_combo.setToolTip("Select which property controls symbol size")
        self.size_combo.currentTextChanged.connect(self.on_size_by_changed)
        size_row.addWidget(self.size_combo)
        size_row.addStretch()
        layout.addLayout(size_row)
        
        # Symbol size range (min/max pixels²)
        symbol_row = QHBoxLayout()
        symbol_row.addWidget(QLabel("Symbol:"))
        symbol_row.addWidget(QLabel("Min"))
        self.size_min_spin = QSpinBox()
        self.size_min_spin.setRange(1, 500)
        self.size_min_spin.setValue(10)
        self.size_min_spin.setMaximumWidth(55)
        self.size_min_spin.setToolTip("Minimum symbol size (pixels²)")
        self.size_min_spin.valueChanged.connect(self.on_settings_changed)
        symbol_row.addWidget(self.size_min_spin)
        symbol_row.addWidget(QLabel("Max"))
        self.size_max_spin = QSpinBox()
        self.size_max_spin.setRange(1, 500)
        self.size_max_spin.setValue(150)
        self.size_max_spin.setMaximumWidth(55)
        self.size_max_spin.setToolTip("Maximum symbol size (pixels²)")
        self.size_max_spin.valueChanged.connect(self.on_settings_changed)
        symbol_row.addWidget(self.size_max_spin)
        symbol_row.addWidget(QLabel("px²"))
        symbol_row.addStretch()
        layout.addLayout(symbol_row)
        
        # Data range (min/max values)
        data_row = QHBoxLayout()
        data_row.addWidget(QLabel("Data:"))
        data_row.addWidget(QLabel("Min"))
        self.data_min_spin = QDoubleSpinBox()
        self.data_min_spin.setRange(-1000, 1000)
        self.data_min_spin.setValue(19.0)
        self.data_min_spin.setDecimals(2)
        self.data_min_spin.setSingleStep(0.5)
        self.data_min_spin.setMaximumWidth(65)
        self.data_min_spin.setToolTip("Data value that maps to minimum symbol size")
        self.data_min_spin.valueChanged.connect(self.on_settings_changed)
        data_row.addWidget(self.data_min_spin)
        data_row.addWidget(QLabel("Max"))
        self.data_max_spin = QDoubleSpinBox()
        self.data_max_spin.setRange(-1000, 1000)
        self.data_max_spin.setValue(23.0)
        self.data_max_spin.setDecimals(2)
        self.data_max_spin.setSingleStep(0.5)
        self.data_max_spin.setMaximumWidth(65)
        self.data_max_spin.setToolTip("Data value that maps to maximum symbol size")
        self.data_max_spin.valueChanged.connect(self.on_settings_changed)
        data_row.addWidget(self.data_max_spin)
        self.data_unit_label = QLabel("mag")
        self.data_unit_label.setMinimumWidth(30)
        data_row.addWidget(self.data_unit_label)
        data_row.addStretch()
        layout.addLayout(data_row)
        
        # Invert checkbox and reset button
        options_row = QHBoxLayout()
        self.size_invert_check = QCheckBox("Invert")
        self.size_invert_check.setChecked(True)  # V mag: brighter = bigger
        self.size_invert_check.setToolTip("Invert mapping (e.g., brighter magnitudes = bigger symbols)")
        self.size_invert_check.stateChanged.connect(self.on_settings_changed)
        options_row.addWidget(self.size_invert_check)
        self.size_reset_btn = QPushButton("reset")
        self.size_reset_btn.setMaximumWidth(45)
        self.size_reset_btn.setToolTip("Reset size controls to defaults for current property")
        self.size_reset_btn.clicked.connect(self.reset_size_defaults)
        options_row.addWidget(self.size_reset_btn)
        options_row.addStretch()
        layout.addLayout(options_row)
        
        self.setLayout(layout)
        
        # Store default values for each size-by option
        self.size_defaults = {
            'V magnitude': {'data_min': 19.0, 'data_max': 23.0, 'invert': True, 'unit': 'mag'},
            'H magnitude': {'data_min': 15.0, 'data_max': 28.0, 'invert': True, 'unit': 'mag'},
            'Distance': {'data_min': 0.5, 'data_max': 2.0, 'invert': True, 'unit': 'AU'},
            'Earth MOID': {'data_min': 0.0, 'data_max': 0.05, 'invert': True, 'unit': 'AU'},
            'Period': {'data_min': 0.5, 'data_max': 5.0, 'invert': True, 'unit': 'yr'},
            'Eccentricity': {'data_min': 0.0, 'data_max': 1.0, 'invert': False, 'unit': ''}
        }
    
    def reset_colorbar(self):
        """Reset colorbar to defaults (19.0 - 23.0)"""
        self.cbar_min.setValue(19.0)
        self.cbar_max.setValue(23.0)
    
    def on_color_by_changed(self, value):
        """Handle color by mode change"""
        is_cneos = (value == 'CNEOS Discovery Site')
        
        # Show/hide controls based on mode
        self.cmap_label.setVisible(not is_cneos)
        self.cmap_combo.setVisible(not is_cneos)
        self.range_min_label.setVisible(not is_cneos)
        self.cbar_min.setVisible(not is_cneos)
        self.range_max_label.setVisible(not is_cneos)
        self.cbar_max.setVisible(not is_cneos)
        self.cbar_reset.setVisible(not is_cneos)
        self.show_legend_check.setVisible(is_cneos)
        self.cneos_link.setVisible(is_cneos)
        
        self.on_settings_changed()
    
    def get_color_by(self):
        """Return what property controls symbol color"""
        return self.color_by_combo.currentText()
    
    def get_show_legend(self):
        """Return whether legend should be shown"""
        return self.show_legend_check.isChecked()
    
    def get_colorbar_range(self):
        return self.cbar_min.value(), self.cbar_max.value()
    
    def get_colormap(self):
        return self.cmap_combo.currentText()
    
    def get_size_by(self):
        """Return what property controls symbol size"""
        return self.size_combo.currentText()
    
    def get_size_settings(self):
        """Return all symbol size settings"""
        return {
            'size_by': self.size_combo.currentText(),
            'size_min': self.size_min_spin.value(),
            'size_max': self.size_max_spin.value(),
            'data_min': self.data_min_spin.value(),
            'data_max': self.data_max_spin.value(),
            'invert': self.size_invert_check.isChecked()
        }
    
    def on_size_by_changed(self, value):
        """Update defaults when size-by option changes"""
        if value in self.size_defaults:
            defaults = self.size_defaults[value]
            # Block signals to avoid multiple redraws
            self.data_min_spin.blockSignals(True)
            self.data_max_spin.blockSignals(True)
            self.size_invert_check.blockSignals(True)
            
            self.data_min_spin.setValue(defaults['data_min'])
            self.data_max_spin.setValue(defaults['data_max'])
            self.size_invert_check.setChecked(defaults['invert'])
            self.data_unit_label.setText(defaults['unit'])
            
            self.data_min_spin.blockSignals(False)
            self.data_max_spin.blockSignals(False)
            self.size_invert_check.blockSignals(False)
        
        self.on_settings_changed()
    
    def reset_size_defaults(self):
        """Reset size controls to defaults for current property"""
        value = self.size_combo.currentText()
        if value in self.size_defaults:
            defaults = self.size_defaults[value]
            self.size_min_spin.setValue(10)
            self.size_max_spin.setValue(150)
            self.data_min_spin.setValue(defaults['data_min'])
            self.data_max_spin.setValue(defaults['data_max'])
            self.size_invert_check.setChecked(defaults['invert'])
            self.data_unit_label.setText(defaults['unit'])
    
    def reset_defaults(self):
        """Reset all to default values"""
        self.color_by_combo.setCurrentText('V magnitude')
        self.cmap_combo.setCurrentText('viridis_r')
        self.cbar_min.setValue(19.0)
        self.cbar_max.setValue(23.0)
        self.show_legend_check.setChecked(True)
        self.size_combo.setCurrentText('V magnitude')
        self.size_min_spin.setValue(10)
        self.size_max_spin.setValue(150)
        self.data_min_spin.setValue(19.0)
        self.data_max_spin.setValue(23.0)
        self.size_invert_check.setChecked(True)
        self.data_unit_label.setText('mag')
        # Trigger visibility update
        self.on_color_by_changed('V magnitude')
    
    def on_settings_changed(self):
        self.settings_changed.emit()


class SettingsDialog(QDialog):
    """Dialog for orbital elements and projection options"""
    
    # Default colors for planes
    DEFAULT_COLORS = {
        'equator': '#00FFFF',   # cyan
        'ecliptic': '#4169E1',  # royal blue
        'galaxy': '#FF00FF'     # magenta
    }
    
    def __init__(self, orbital_panel, proj_panel, colorbar_panel, parent=None):
        super().__init__(parent)
        self.orbital_panel = orbital_panel
        self.proj_panel = proj_panel  # Reference for coord changes
        self.colorbar_panel = colorbar_panel
        self._layout = None
        self.setup_ui()
        
        # Connect coordinate system changes to update defaults
        if hasattr(proj_panel, 'coord_combo'):
            proj_panel.coord_combo.currentTextChanged.connect(self.update_plane_defaults)
    
    def setup_ui(self):
        self.setWindowTitle("Settings")
        self.setMinimumWidth(450)
        self.setMinimumHeight(400)
        # No maximum height - allow dialog to expand as needed
        
        # Main layout with scroll area
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)
        
        # Scroll area for all the options
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        if PYQT_VERSION == 6:
            scroll.setFrameShape(QFrame.Shape.NoFrame)
        else:
            scroll.setFrameShape(QFrame.NoFrame)
        
        # Content widget inside scroll area
        content = QWidget()
        self._layout = QVBoxLayout()
        self._layout.setContentsMargins(0, 0, 5, 0)  # Right margin for scrollbar
        self._layout.setSpacing(5)
        
        # Track all collapsible sections
        self.collapsible_sections = []
        
        # Orbital elements panel (wrapped in collapsible)
        orbital_wrapper = CollapsibleGroupBox("Orbital Elements")
        orbital_wrapper.content_layout.addWidget(self.orbital_panel)
        self._layout.addWidget(orbital_wrapper)
        self.collapsible_sections.append(orbital_wrapper)
        
        # Colorbar and Symbol Size panel (wrapped in collapsible)
        colorbar_wrapper = CollapsibleGroupBox("Colorbar and Symbol Size")
        colorbar_wrapper.content_layout.addWidget(self.colorbar_panel)
        self._layout.addWidget(colorbar_wrapper)
        self.collapsible_sections.append(colorbar_wrapper)
        
        # Planes and Poles section
        planes_group = CollapsibleGroupBox("Planes and Poles")
        planes_layout = QVBoxLayout()
        planes_layout.setSpacing(3)
        
        # Store references to controls
        self.plane_controls = {}
        
        for plane_name, display_name in [('equator', 'Equator'), ('ecliptic', 'Ecliptic'), ('galaxy', 'Galaxy')]:
            row = QHBoxLayout()
            row.setSpacing(5)
            
            # Plane checkbox
            plane_cb = QCheckBox(display_name)
            plane_cb.setMinimumWidth(70)
            plane_cb.stateChanged.connect(self.on_plane_changed)
            row.addWidget(plane_cb)
            
            # Color entry
            color_edit = QLineEdit(self.DEFAULT_COLORS[plane_name])
            color_edit.setMaximumWidth(70)
            color_edit.setToolTip("Color (name or hex, e.g., 'yellow' or '#FFFF00'). Press Enter to apply.")
            color_edit.editingFinished.connect(self.on_plane_changed)
            row.addWidget(color_edit)
            
            # Reset button
            reset_btn = QPushButton("reset")
            reset_btn.setMaximumWidth(45)
            reset_btn.clicked.connect(lambda checked, p=plane_name: self.reset_plane_color(p))
            row.addWidget(reset_btn)
            
            # Pole checkbox
            pole_cb = QCheckBox("pole")
            pole_cb.stateChanged.connect(self.on_plane_changed)
            row.addWidget(pole_cb)
            
            row.addStretch()
            planes_layout.addLayout(row)
            
            # Store references
            self.plane_controls[plane_name] = {
                'plane_cb': plane_cb,
                'color_edit': color_edit,
                'pole_cb': pole_cb
            }
        
        planes_group.setLayout(planes_layout)
        self._layout.addWidget(planes_group)
        self.collapsible_sections.append(planes_group)
        
        # Set initial defaults based on coordinate system
        self.update_plane_defaults()
        
        # Sun and Moon section
        sunmoon_group = CollapsibleGroupBox("Sun and Moon")
        sunmoon_layout = QVBoxLayout()
        sunmoon_layout.setSpacing(3)
        
        # Sun checkbox
        sun_row = QHBoxLayout()
        self.show_sun_check = QCheckBox("Show Sun")
        self.show_sun_check.setChecked(True)
        self.show_sun_check.setToolTip("Display the Sun's position on the sky map")
        self.show_sun_check.stateChanged.connect(self.on_sunmoon_changed)
        sun_row.addWidget(self.show_sun_check)
        sun_row.addStretch()
        sunmoon_layout.addLayout(sun_row)
        
        # Moon checkbox
        moon_row = QHBoxLayout()
        self.show_moon_check = QCheckBox("Show Moon")
        self.show_moon_check.setChecked(True)  # Default checked
        self.show_moon_check.setToolTip("Display the Moon's position on the sky map")
        self.show_moon_check.stateChanged.connect(self.on_sunmoon_changed)
        moon_row.addWidget(self.show_moon_check)
        moon_row.addStretch()
        sunmoon_layout.addLayout(moon_row)
        
        # Lunar exclusion penalty enable (indented)
        lunar_excl_row = QHBoxLayout()
        lunar_excl_row.addSpacing(20)
        self.lunar_exclusion_check = QCheckBox("Enable lunar exclusion penalty")
        self.lunar_exclusion_check.setChecked(False)
        self.lunar_exclusion_check.setToolTip("Apply magnitude penalty to NEOs near the Moon\nPenalty scales with lunar phase (max at full moon)")
        self.lunar_exclusion_check.stateChanged.connect(self.on_sunmoon_changed)
        lunar_excl_row.addWidget(self.lunar_exclusion_check)
        lunar_excl_row.addStretch()
        sunmoon_layout.addLayout(lunar_excl_row)
        
        # Full Moon radius row (indented)
        lunar_radius_row = QHBoxLayout()
        lunar_radius_row.addSpacing(20)
        lunar_radius_row.addWidget(QLabel("Full Moon radius:"))
        self.lunar_radius_spin = QDoubleSpinBox()
        self.lunar_radius_spin.setRange(5.0, 90.0)
        self.lunar_radius_spin.setValue(30.0)
        self.lunar_radius_spin.setSingleStep(5.0)
        self.lunar_radius_spin.setSuffix("°")
        self.lunar_radius_spin.setToolTip("Exclusion radius at full moon (scales to 0 at new moon)")
        self.lunar_radius_spin.valueChanged.connect(self.on_sunmoon_changed)
        lunar_radius_row.addWidget(self.lunar_radius_spin)
        lunar_radius_row.addWidget(QLabel("Color:"))
        self.lunar_color_edit = QLineEdit("#228B22")  # Forest green (same as moon border)
        self.lunar_color_edit.setMaximumWidth(70)
        self.lunar_color_edit.setToolTip("Color for lunar exclusion circle (hex or name)")
        self.lunar_color_edit.editingFinished.connect(self.on_sunmoon_changed)
        lunar_radius_row.addWidget(self.lunar_color_edit)
        self.lunar_show_bounds = QCheckBox("Show boundary")
        self.lunar_show_bounds.setChecked(True)
        self.lunar_show_bounds.setToolTip("Draw dashed circle showing exclusion zone")
        self.lunar_show_bounds.stateChanged.connect(self.on_sunmoon_changed)
        lunar_radius_row.addWidget(self.lunar_show_bounds)
        lunar_radius_row.addStretch()
        sunmoon_layout.addLayout(lunar_radius_row)
        
        # Full Moon penalty row (indented)
        lunar_penalty_row = QHBoxLayout()
        lunar_penalty_row.addSpacing(20)
        lunar_penalty_row.addWidget(QLabel("Full Moon penalty:"))
        self.lunar_penalty_spin = QDoubleSpinBox()
        self.lunar_penalty_spin.setRange(0.0, 10.0)
        self.lunar_penalty_spin.setValue(3.0)
        self.lunar_penalty_spin.setSingleStep(0.25)
        self.lunar_penalty_spin.setSuffix(" mag")
        self.lunar_penalty_spin.setToolTip("Magnitude penalty at full moon (scales to 0 at new moon)")
        self.lunar_penalty_spin.valueChanged.connect(self.on_sunmoon_changed)
        lunar_penalty_row.addWidget(self.lunar_penalty_spin)
        lunar_penalty_row.addStretch()
        sunmoon_layout.addLayout(lunar_penalty_row)
        
        # Moon phases checkbox (indented)
        phase_row = QHBoxLayout()
        phase_row.addSpacing(20)
        self.show_moon_phases_check = QCheckBox("Show lunar phases")
        self.show_moon_phases_check.setChecked(False)  # Default unchecked
        self.show_moon_phases_check.setToolTip("Display realistic phase shape (crescent, quarter, etc.)\nIf unchecked, shows a plain circle")
        self.show_moon_phases_check.stateChanged.connect(self.on_sunmoon_changed)
        phase_row.addWidget(self.show_moon_phases_check)
        phase_row.addStretch()
        sunmoon_layout.addLayout(phase_row)
        
        sunmoon_group.setLayout(sunmoon_layout)
        self._layout.addWidget(sunmoon_group)
        self.collapsible_sections.append(sunmoon_group)
        
        # Night divides the day section (horizon/twilight)
        horizon_group = CollapsibleGroupBox("Night divides the day")
        horizon_layout = QVBoxLayout()
        horizon_layout.setSpacing(3)
        
        # Enable checkbox
        self.horizon_enable_check = QCheckBox("Show horizon and twilight boundaries")
        self.horizon_enable_check.setChecked(False)
        self.horizon_enable_check.setToolTip("Draw local horizon and twilight lines on the sky map\nThese show where the sky is dark for a ground-based observer")
        self.horizon_enable_check.stateChanged.connect(self.on_horizon_changed)
        horizon_layout.addWidget(self.horizon_enable_check)
        
        # Observer location
        loc_label = QLabel("Observer location:")
        loc_label.setStyleSheet("font-weight: bold;")
        horizon_layout.addWidget(loc_label)
        
        lat_row = QHBoxLayout()
        lat_row.addWidget(QLabel("Latitude:"))
        self.observer_lat_spin = QDoubleSpinBox()
        self.observer_lat_spin.setRange(-90.0, 90.0)
        self.observer_lat_spin.setValue(32.2226)  # Tucson, Arizona
        self.observer_lat_spin.setDecimals(4)
        self.observer_lat_spin.setSingleStep(1.0)
        self.observer_lat_spin.setSuffix("°")
        self.observer_lat_spin.setToolTip("Observer latitude (positive = North)")
        self.observer_lat_spin.valueChanged.connect(self.on_horizon_changed)
        lat_row.addWidget(self.observer_lat_spin)
        lat_row.addWidget(QLabel("Longitude:"))
        self.observer_lon_spin = QDoubleSpinBox()
        self.observer_lon_spin.setRange(-180.0, 180.0)
        self.observer_lon_spin.setValue(-110.9747)  # Tucson, Arizona (West is negative)
        self.observer_lon_spin.setDecimals(4)
        self.observer_lon_spin.setSingleStep(1.0)
        self.observer_lon_spin.setSuffix("°")
        self.observer_lon_spin.setToolTip("Observer longitude (negative = West)")
        self.observer_lon_spin.valueChanged.connect(self.on_horizon_changed)
        lat_row.addWidget(self.observer_lon_spin)
        lat_row.addStretch()
        horizon_layout.addLayout(lat_row)
        
        # Preset locations
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Presets:"))
        self.location_preset_combo = QComboBox()
        self.location_preset_combo.addItems([
            "Tucson, AZ (CSS)",
            "Haleakala, HI (ATLAS/PS)",
            "Siding Spring, AU (CSS-S)",
            "La Palma, ES",
            "Cerro Pachón, CL",
            "Custom"
        ])
        self.location_preset_combo.setToolTip("Select a preset observatory location")
        self.location_preset_combo.currentTextChanged.connect(self._on_location_preset_changed)
        preset_row.addWidget(self.location_preset_combo)
        preset_row.addStretch()
        horizon_layout.addLayout(preset_row)
        
        # Timezone display row
        tz_row = QHBoxLayout()
        tz_row.addWidget(QLabel("Timezone:"))
        self.timezone_combo = QComboBox()
        self.timezone_combo.setEditable(True)
        self.timezone_combo.addItems([
            "UTC-7 (MST)",
            "UTC-10 (HST)",
            "UTC+10 (AEST)",
            "UTC+0 (WET)",
            "UTC-3 (CLT)",
            "Custom"
        ])
        self.timezone_combo.setToolTip("Observer timezone (informational - horizon calculation uses longitude)")
        self.timezone_combo.setMaximumWidth(120)
        tz_row.addWidget(self.timezone_combo)
        tz_row.addStretch()
        horizon_layout.addLayout(tz_row)
        
        # Boundary lines section
        lines_label = QLabel("Boundary lines:")
        lines_label.setStyleSheet("font-weight: bold;")
        horizon_layout.addWidget(lines_label)
        
        # Horizon (alt=0°)
        horizon_row = QHBoxLayout()
        self.show_horizon_check = QCheckBox("Horizon (0°)")
        self.show_horizon_check.setChecked(True)
        self.show_horizon_check.setToolTip("Show the local horizon (altitude = 0°)")
        self.show_horizon_check.stateChanged.connect(self.on_horizon_changed)
        horizon_row.addWidget(self.show_horizon_check)
        self.horizon_color_edit = QLineEdit("#FF6600")
        self.horizon_color_edit.setMaximumWidth(70)
        self.horizon_color_edit.setToolTip("Horizon line color")
        self.horizon_color_edit.editingFinished.connect(self.on_horizon_changed)
        horizon_row.addWidget(self.horizon_color_edit)
        horizon_row.addStretch()
        horizon_layout.addLayout(horizon_row)
        
        # Civil twilight (alt=-6°)
        civil_row = QHBoxLayout()
        self.show_civil_check = QCheckBox("Civil twilight (-6°)")
        self.show_civil_check.setChecked(False)
        self.show_civil_check.setToolTip("Civil twilight boundary (Sun 6° below horizon)")
        self.show_civil_check.stateChanged.connect(self.on_horizon_changed)
        civil_row.addWidget(self.show_civil_check)
        self.civil_color_edit = QLineEdit("#FF9933")
        self.civil_color_edit.setMaximumWidth(70)
        self.civil_color_edit.setToolTip("Civil twilight line color")
        self.civil_color_edit.editingFinished.connect(self.on_horizon_changed)
        civil_row.addWidget(self.civil_color_edit)
        civil_row.addStretch()
        horizon_layout.addLayout(civil_row)
        
        # Nautical twilight (alt=-12°)
        nautical_row = QHBoxLayout()
        self.show_nautical_check = QCheckBox("Nautical twilight (-12°)")
        self.show_nautical_check.setChecked(False)
        self.show_nautical_check.setToolTip("Nautical twilight boundary (Sun 12° below horizon)")
        self.show_nautical_check.stateChanged.connect(self.on_horizon_changed)
        nautical_row.addWidget(self.show_nautical_check)
        self.nautical_color_edit = QLineEdit("#CC66FF")
        self.nautical_color_edit.setMaximumWidth(70)
        self.nautical_color_edit.setToolTip("Nautical twilight line color")
        self.nautical_color_edit.editingFinished.connect(self.on_horizon_changed)
        nautical_row.addWidget(self.nautical_color_edit)
        nautical_row.addStretch()
        horizon_layout.addLayout(nautical_row)
        
        # Astronomical twilight (alt=-18°)
        astro_row = QHBoxLayout()
        self.show_astro_check = QCheckBox("Astronomical twilight (-18°)")
        self.show_astro_check.setChecked(True)
        self.show_astro_check.setToolTip("Astronomical twilight boundary (Sun 18° below horizon)\nBeyond this line, the sky is fully dark")
        self.show_astro_check.stateChanged.connect(self.on_horizon_changed)
        astro_row.addWidget(self.show_astro_check)
        self.astro_color_edit = QLineEdit("#6666FF")
        self.astro_color_edit.setMaximumWidth(70)
        self.astro_color_edit.setToolTip("Astronomical twilight line color")
        self.astro_color_edit.editingFinished.connect(self.on_horizon_changed)
        astro_row.addWidget(self.astro_color_edit)
        astro_row.addStretch()
        horizon_layout.addLayout(astro_row)
        
        # Line style options
        style_row = QHBoxLayout()
        style_row.addWidget(QLabel("Line style:"))
        self.horizon_style_combo = QComboBox()
        self.horizon_style_combo.addItems(["solid", "dashed", "dotted"])
        self.horizon_style_combo.setCurrentText("solid")
        self.horizon_style_combo.setToolTip("Line style for horizon/twilight boundaries")
        self.horizon_style_combo.currentTextChanged.connect(self.on_horizon_changed)
        style_row.addWidget(self.horizon_style_combo)
        style_row.addWidget(QLabel("Weight:"))
        self.horizon_weight_spin = QDoubleSpinBox()
        self.horizon_weight_spin.setRange(0.5, 5.0)
        self.horizon_weight_spin.setValue(1.5)
        self.horizon_weight_spin.setSingleStep(0.5)
        self.horizon_weight_spin.setToolTip("Line thickness")
        self.horizon_weight_spin.valueChanged.connect(self.on_horizon_changed)
        style_row.addWidget(self.horizon_weight_spin)
        style_row.addStretch()
        horizon_layout.addLayout(style_row)
        
        horizon_group.setLayout(horizon_layout)
        self._layout.addWidget(horizon_group)
        self.collapsible_sections.append(horizon_group)
        
        # Galactic Exclusion section
        galactic_group = CollapsibleGroupBox("Galactic Exclusion")
        galactic_layout = QVBoxLayout()
        galactic_layout.setSpacing(3)
        
        # Enable checkbox
        self.galactic_enable_check = QCheckBox("Enable galactic exclusion penalty")
        self.galactic_enable_check.setChecked(False)
        self.galactic_enable_check.setToolTip("Apply magnitude penalty to NEOs near galactic plane")
        self.galactic_enable_check.stateChanged.connect(self.on_galactic_changed)
        galactic_layout.addWidget(self.galactic_enable_check)
        
        # Offset row with color
        offset_row = QHBoxLayout()
        offset_row.addWidget(QLabel("Band width:"))
        self.galactic_offset_spin = QDoubleSpinBox()
        self.galactic_offset_spin.setRange(1.0, 30.0)
        self.galactic_offset_spin.setValue(15.0)
        self.galactic_offset_spin.setSingleStep(5.0)
        self.galactic_offset_spin.setSuffix("°")
        self.galactic_offset_spin.setToolTip("Degrees from galactic plane on each side")
        self.galactic_offset_spin.valueChanged.connect(self.on_galactic_changed)
        offset_row.addWidget(self.galactic_offset_spin)
        offset_row.addWidget(QLabel("Color:"))
        self.galactic_color_edit = QLineEdit("#FF99FF")
        self.galactic_color_edit.setMaximumWidth(70)
        self.galactic_color_edit.setToolTip("Color for galactic band boundary lines (hex or name)")
        self.galactic_color_edit.editingFinished.connect(self.on_galactic_changed)
        offset_row.addWidget(self.galactic_color_edit)
        self.galactic_show_bounds = QCheckBox("Show boundaries")
        self.galactic_show_bounds.setChecked(True)
        self.galactic_show_bounds.setToolTip("Draw lines at the band edges")
        self.galactic_show_bounds.stateChanged.connect(self.on_galactic_changed)
        offset_row.addWidget(self.galactic_show_bounds)
        offset_row.addStretch()
        galactic_layout.addLayout(offset_row)
        
        # Penalty row
        penalty_row = QHBoxLayout()
        penalty_row.addWidget(QLabel("Mag penalty:"))
        self.galactic_penalty_spin = QDoubleSpinBox()
        self.galactic_penalty_spin.setRange(0.0, 10.0)
        self.galactic_penalty_spin.setValue(2.0)
        self.galactic_penalty_spin.setSingleStep(0.25)
        self.galactic_penalty_spin.setSuffix(" mag")
        self.galactic_penalty_spin.setToolTip("Magnitude penalty for NEOs in exclusion zone")
        self.galactic_penalty_spin.valueChanged.connect(self.on_galactic_changed)
        penalty_row.addWidget(self.galactic_penalty_spin)
        penalty_row.addStretch()
        galactic_layout.addLayout(penalty_row)
        
        galactic_group.setLayout(galactic_layout)
        self._layout.addWidget(galactic_group)
        self.collapsible_sections.append(galactic_group)
        
        # Opposition Benefit section
        opposition_group = CollapsibleGroupBox("Opposition Benefit")
        opposition_layout = QVBoxLayout()
        opposition_layout.setSpacing(3)
        
        # Enable checkbox
        self.opposition_enable_check = QCheckBox("Enable opposition benefit")
        self.opposition_enable_check.setChecked(False)
        self.opposition_enable_check.setToolTip("Apply magnitude benefit to NEOs near opposition")
        self.opposition_enable_check.stateChanged.connect(self.on_opposition_changed)
        opposition_layout.addWidget(self.opposition_enable_check)
        
        # Radius row with color
        radius_row = QHBoxLayout()
        radius_row.addWidget(QLabel("Radius:"))
        self.opposition_radius_spin = QDoubleSpinBox()
        self.opposition_radius_spin.setRange(1.0, 30.0)
        self.opposition_radius_spin.setValue(5.0)
        self.opposition_radius_spin.setSingleStep(1.0)
        self.opposition_radius_spin.setSuffix("°")
        self.opposition_radius_spin.setToolTip("Angular radius from opposition point")
        self.opposition_radius_spin.valueChanged.connect(self.on_opposition_changed)
        radius_row.addWidget(self.opposition_radius_spin)
        radius_row.addWidget(QLabel("Color:"))
        self.opposition_color_edit = QLineEdit("#90EE90")
        self.opposition_color_edit.setMaximumWidth(70)
        self.opposition_color_edit.setToolTip("Color for opposition circle (hex or name)")
        self.opposition_color_edit.editingFinished.connect(self.on_opposition_changed)
        radius_row.addWidget(self.opposition_color_edit)
        self.opposition_show_bounds = QCheckBox("Show boundaries")
        self.opposition_show_bounds.setChecked(True)
        self.opposition_show_bounds.setToolTip("Draw circle at the benefit radius")
        self.opposition_show_bounds.stateChanged.connect(self.on_opposition_changed)
        radius_row.addWidget(self.opposition_show_bounds)
        radius_row.addStretch()
        opposition_layout.addLayout(radius_row)
        
        # Benefit row
        benefit_row = QHBoxLayout()
        benefit_row.addWidget(QLabel("Mag benefit:"))
        self.opposition_benefit_spin = QDoubleSpinBox()
        self.opposition_benefit_spin.setRange(0.0, 5.0)
        self.opposition_benefit_spin.setValue(2.0)
        self.opposition_benefit_spin.setSingleStep(0.25)
        self.opposition_benefit_spin.setSuffix(" mag")
        self.opposition_benefit_spin.setToolTip("Magnitude benefit for NEOs near opposition (brighter)")
        self.opposition_benefit_spin.valueChanged.connect(self.on_opposition_changed)
        benefit_row.addWidget(self.opposition_benefit_spin)
        benefit_row.addStretch()
        opposition_layout.addLayout(benefit_row)
        
        opposition_group.setLayout(opposition_layout)
        self._layout.addWidget(opposition_group)
        self.collapsible_sections.append(opposition_group)
        
        # Site Filtering section
        site_group = CollapsibleGroupBox("Site Filtering")
        site_layout = QVBoxLayout()
        site_layout.setSpacing(3)
        
        # Whitelist row
        whitelist_row = QHBoxLayout()
        self.site_whitelist_check = QCheckBox("Only show sites:")
        self.site_whitelist_check.setChecked(False)
        self.site_whitelist_check.setToolTip("Only show NEOs discovered by these sites (comma-separated codes)")
        self.site_whitelist_check.stateChanged.connect(self.on_site_filter_changed)
        whitelist_row.addWidget(self.site_whitelist_check)
        
        self.site_whitelist_edit = QLineEdit()
        self.site_whitelist_edit.setPlaceholderText("e.g., 703, G96, F51, T05")
        self.site_whitelist_edit.setToolTip("Comma-separated site codes (e.g., 703, G96, F51)")
        self.site_whitelist_edit.editingFinished.connect(self.on_site_filter_changed)
        whitelist_row.addWidget(self.site_whitelist_edit)
        site_layout.addLayout(whitelist_row)
        
        # Blacklist row
        blacklist_row = QHBoxLayout()
        self.site_blacklist_check = QCheckBox("Hide sites:")
        self.site_blacklist_check.setChecked(False)
        self.site_blacklist_check.setToolTip("Hide NEOs discovered by these sites (comma-separated codes)")
        self.site_blacklist_check.stateChanged.connect(self.on_site_filter_changed)
        blacklist_row.addWidget(self.site_blacklist_check)
        
        self.site_blacklist_edit = QLineEdit()
        self.site_blacklist_edit.setPlaceholderText("e.g., C51")
        self.site_blacklist_edit.setToolTip("Comma-separated site codes to exclude")
        self.site_blacklist_edit.editingFinished.connect(self.on_site_filter_changed)
        blacklist_row.addWidget(self.site_blacklist_edit)
        site_layout.addLayout(blacklist_row)
        
        site_group.setLayout(site_layout)
        self._layout.addWidget(site_group)
        self.collapsible_sections.append(site_group)
        
        # Advanced Controls section (keyboard navigation)
        advanced_group = CollapsibleGroupBox("Advanced Controls")
        advanced_layout = QVBoxLayout()
        advanced_layout.setSpacing(3)
        
        # Trailing section (at top of Advanced Controls)
        trail_label = QLabel("Trailing (during animation):")
        trail_label.setStyleSheet("font-weight: bold;")
        advanced_layout.addWidget(trail_label)
        
        # Enable trailing checkbox
        self.enable_trailing_check = QCheckBox("Enable trails")
        self.enable_trailing_check.setChecked(False)
        self.enable_trailing_check.setToolTip("Show motion trails during animation")
        self.enable_trailing_check.stateChanged.connect(self.on_trailing_changed)
        advanced_layout.addWidget(self.enable_trailing_check)
        
        # Trail length
        trail_len_row = QHBoxLayout()
        trail_len_row.addWidget(QLabel("Trail length:"))
        self.trail_length_spin = QSpinBox()
        self.trail_length_spin.setRange(5, 10000)  # Allow very long trails
        self.trail_length_spin.setValue(50)
        self.trail_length_spin.setMaximumWidth(70)
        self.trail_length_spin.setToolTip("Number of animation steps to retain in trail history")
        self.trail_length_spin.valueChanged.connect(self.on_trailing_changed)
        trail_len_row.addWidget(self.trail_length_spin)
        trail_len_row.addWidget(QLabel("steps"))
        trail_len_row.addStretch()
        advanced_layout.addLayout(trail_len_row)
        
        # Trail line weight
        weight_row = QHBoxLayout()
        weight_row.addWidget(QLabel("Line weight:"))
        self.trail_weight_spin = QDoubleSpinBox()
        self.trail_weight_spin.setRange(0.5, 5.0)
        self.trail_weight_spin.setValue(1.0)
        self.trail_weight_spin.setSingleStep(0.5)
        self.trail_weight_spin.setMaximumWidth(60)
        self.trail_weight_spin.setToolTip("Trail line thickness in pixels")
        self.trail_weight_spin.valueChanged.connect(self.on_trailing_changed)
        weight_row.addWidget(self.trail_weight_spin)
        weight_row.addWidget(QLabel("px"))
        weight_row.addStretch()
        advanced_layout.addLayout(weight_row)
        
        # Trail color
        color_row = QHBoxLayout()
        color_row.addWidget(QLabel("Trail color:"))
        self.trail_color = QLineEdit("#00AA00")
        self.trail_color.setMaximumWidth(70)
        self.trail_color.setToolTip("Trail line color (green default)")
        color_row.addWidget(self.trail_color)
        color_row.addStretch()
        advanced_layout.addLayout(color_row)
        
        # Clear trails button
        self.clear_trails_btn = QPushButton("Clear Trails")
        self.clear_trails_btn.setToolTip("Remove all accumulated trail history")
        self.clear_trails_btn.clicked.connect(self.on_clear_trails)
        self.clear_trails_btn.setMaximumWidth(100)
        advanced_layout.addWidget(self.clear_trails_btn)

        # Fast animation section
        fast_sep = QFrame()
        if PYQT_VERSION == 6:
            fast_sep.setFrameShape(QFrame.Shape.HLine)
        else:
            fast_sep.setFrameShape(QFrame.HLine)
        fast_sep.setStyleSheet("color: #ccc;")
        advanced_layout.addWidget(fast_sep)

        fast_label = QLabel("Animation Performance:")
        fast_label.setStyleSheet("font-weight: bold;")
        advanced_layout.addWidget(fast_label)

        # Fast animation checkbox
        self.fast_animation_check = QCheckBox("Fast animation mode")
        self.fast_animation_check.setChecked(False)
        self.fast_animation_check.setToolTip("Show fewer points during animation for smoother playback")
        advanced_layout.addWidget(self.fast_animation_check)

        # Decimation factor
        decimate_row = QHBoxLayout()
        decimate_row.addWidget(QLabel("Show every"))
        self.decimate_spin = QSpinBox()
        self.decimate_spin.setRange(2, 20)
        self.decimate_spin.setValue(4)
        self.decimate_spin.setMaximumWidth(50)
        self.decimate_spin.setToolTip("Show every Nth point during animation (4 = 25% of points)")
        decimate_row.addWidget(self.decimate_spin)
        decimate_row.addWidget(QLabel("points"))
        decimate_row.addStretch()
        advanced_layout.addLayout(decimate_row)

        # Hidden controls for backward compatibility with factory reset
        self.trails_before_discovery_check = QCheckBox()
        self.trails_before_discovery_check.hide()
        self.trail_before_color = QLineEdit()
        self.trail_before_color.hide()
        self.trail_after_color = QLineEdit()
        self.trail_after_color.hide()
        
        # Hidden controls for keyboard navigation compatibility (used by factory reset)
        self.ud_increment_spin = QSpinBox()
        self.ud_increment_spin.setValue(1)
        self.ud_increment_spin.hide()
        self.ud_unit_combo = QComboBox()
        self.ud_unit_combo.addItems(["lunation", "month", "year"])
        self.ud_unit_combo.setCurrentText("lunation")
        self.ud_unit_combo.hide()
        
        # Separator before misc options
        sep1 = QFrame()
        if PYQT_VERSION == 6:
            sep1.setFrameShape(QFrame.Shape.HLine)
        else:
            sep1.setFrameShape(QFrame.HLine)
        sep1.setStyleSheet("color: #ccc;")
        advanced_layout.addWidget(sep1)
        
        # Hide missing tracklets (moved from Discovery Circumstances)
        self.hide_missing_discovery_check = QCheckBox("Hide NEOs with missing discovery tracklets")
        self.hide_missing_discovery_check.setChecked(False)
        self.hide_missing_discovery_check.setToolTip("Hide NEOs that don't have discovery tracklet data")
        self.hide_missing_discovery_check.stateChanged.connect(self.on_discovery_changed)
        advanced_layout.addWidget(self.hide_missing_discovery_check)
        
        # Collapsible control panels
        self.kiosk_mode_check = QCheckBox("Collapsible control panels")
        self.kiosk_mode_check.setChecked(False)
        self.kiosk_mode_check.setToolTip("Click top bar to collapse/expand all control panels as a drawer")
        self.kiosk_mode_check.stateChanged.connect(self.on_appearance_changed)
        advanced_layout.addWidget(self.kiosk_mode_check)
        
        # [ and ] keys for time navigation
        lr_row = QHBoxLayout()
        lr_row.addWidget(QLabel("[ / ] keys:"))
        self.lr_increment_spin = QSpinBox()
        self.lr_increment_spin.setRange(1, 100)
        self.lr_increment_spin.setValue(1)
        self.lr_increment_spin.setMaximumWidth(60)
        lr_row.addWidget(self.lr_increment_spin)
        
        self.lr_unit_combo = QComboBox()
        self.lr_unit_combo.addItems(["hour", "solar day", "sidereal day", "lunation", "month", "year"])
        self.lr_unit_combo.setCurrentText("hour")
        self.lr_unit_combo.setMaximumWidth(100)
        lr_row.addWidget(self.lr_unit_combo)
        lr_row.addStretch()
        advanced_layout.addLayout(lr_row)
        
        # Help text
        help_label = QLabel("Press [ to go back, ] to go forward in time")
        help_label.setStyleSheet("color: gray; font-style: italic;")
        advanced_layout.addWidget(help_label)
        advanced_group.setLayout(advanced_layout)
        self._layout.addWidget(advanced_group)
        self.collapsible_sections.append(advanced_group)
        
        # Set content layout and add to scroll area
        content.setLayout(self._layout)
        scroll.setWidget(content)
        main_layout.addWidget(scroll, 1)  # Scroll area gets stretch
        self.scroll_area = scroll  # Store reference for resizing
        
        # Button row with Open tabs, Close tabs, and Close
        button_row = QHBoxLayout()
        
        open_all_btn = QPushButton("Open tabs")
        open_all_btn.clicked.connect(self.open_all_sections)
        button_row.addWidget(open_all_btn)
        
        close_all_btn = QPushButton("Close tabs")
        close_all_btn.clicked.connect(self.close_all_sections)
        button_row.addWidget(close_all_btn)
        
        button_row.addStretch()
        
        close_btn = QPushButton("Close")
        close_btn.setAutoDefault(False)  # Prevent Enter from activating this button
        close_btn.clicked.connect(self.hide)  # Hide instead of accept to persist settings
        button_row.addWidget(close_btn)
        
        main_layout.addLayout(button_row)
        
        self.setLayout(main_layout)
        self.resize(450, 600)  # Default size
        
        # Initialize control enabled/disabled states based on default checkbox values
        self._initialize_control_states()
    
    def open_all_sections(self):
        """Expand all collapsible sections and resize dialog to fit"""
        for section in self.collapsible_sections:
            if hasattr(section, 'collapsed') and section.collapsed:
                section.toggle()
        
        # Resize dialog to fit all expanded content (up to screen height)
        QApplication.processEvents()  # Let layout update
        content_height = self.scroll_area.widget().sizeHint().height()
        button_height = 50  # Approximate button row height
        margins = 20  # Dialog margins
        desired_height = content_height + button_height + margins
        
        # Limit to screen height minus some margin for taskbar etc.
        screen = QApplication.primaryScreen()
        if screen:
            available_height = screen.availableGeometry().height() - 50
            desired_height = min(desired_height, available_height)
        
        self.resize(self.width(), desired_height)
    
    def close_all_sections(self):
        """Collapse all collapsible sections"""
        for section in self.collapsible_sections:
            if hasattr(section, 'collapsed') and not section.collapsed:
                section.toggle()
    
    def _initialize_control_states(self):
        """Set initial enabled/disabled states for all dependent controls"""
        # Sun/Moon controls
        moon_enabled = self.show_moon_check.isChecked()
        self.lunar_exclusion_check.setEnabled(moon_enabled)
        self.show_moon_phases_check.setEnabled(moon_enabled)
        
        lunar_excl_enabled = moon_enabled and self.lunar_exclusion_check.isChecked()
        self.lunar_radius_spin.setEnabled(lunar_excl_enabled)
        self.lunar_penalty_spin.setEnabled(lunar_excl_enabled)
        self.lunar_color_edit.setEnabled(lunar_excl_enabled)
        self.lunar_show_bounds.setEnabled(lunar_excl_enabled)
        
        # Horizon/twilight controls
        horizon_enabled = self.horizon_enable_check.isChecked()
        self.observer_lat_spin.setEnabled(horizon_enabled)
        self.observer_lon_spin.setEnabled(horizon_enabled)
        self.location_preset_combo.setEnabled(horizon_enabled)
        self.timezone_combo.setEnabled(horizon_enabled)
        self.show_horizon_check.setEnabled(horizon_enabled)
        self.horizon_color_edit.setEnabled(horizon_enabled and self.show_horizon_check.isChecked())
        self.show_civil_check.setEnabled(horizon_enabled)
        self.civil_color_edit.setEnabled(horizon_enabled and self.show_civil_check.isChecked())
        self.show_nautical_check.setEnabled(horizon_enabled)
        self.nautical_color_edit.setEnabled(horizon_enabled and self.show_nautical_check.isChecked())
        self.show_astro_check.setEnabled(horizon_enabled)
        self.astro_color_edit.setEnabled(horizon_enabled and self.show_astro_check.isChecked())
        self.horizon_style_combo.setEnabled(horizon_enabled)
        self.horizon_weight_spin.setEnabled(horizon_enabled)
        
        # Trailing controls
        trailing_enabled = self.enable_trailing_check.isChecked()
        self.trail_length_spin.setEnabled(trailing_enabled)
        self.trail_weight_spin.setEnabled(trailing_enabled)
        self.trail_color.setEnabled(trailing_enabled)
        self.clear_trails_btn.setEnabled(trailing_enabled)
    
    def closeEvent(self, event):
        """Override close event to hide instead of close (preserves settings)"""
        event.ignore()
        self.hide()
    
    def keyPressEvent(self, event):
        """Override key press to prevent Enter from dismissing dialog"""
        # Ignore Enter/Return key - don't dismiss dialog
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) if PYQT_VERSION == 6 else (Qt.Key_Return, Qt.Key_Enter):
            event.ignore()
            return
        super().keyPressEvent(event)
    
    def update_plane_defaults(self):
        """Set default plane visibility based on coordinate system"""
        coord_sys = self.proj_panel.get_coordinate_system() if hasattr(self.proj_panel, 'get_coordinate_system') else 'equatorial'
        
        # Block signals during default setting
        for controls in self.plane_controls.values():
            controls['plane_cb'].blockSignals(True)
            controls['pole_cb'].blockSignals(True)
        
        # Set defaults: show the two planes that are NOT the current coordinate system
        # Opposition uses ecliptic latitude, so don't show ecliptic plane
        self.plane_controls['equator']['plane_cb'].setChecked(coord_sys not in ['equatorial'])
        self.plane_controls['ecliptic']['plane_cb'].setChecked(coord_sys not in ['ecliptic', 'opposition'])
        self.plane_controls['galaxy']['plane_cb'].setChecked(coord_sys not in ['galactic'])
        
        # Poles enabled when planes are enabled
        for name, controls in self.plane_controls.items():
            controls['pole_cb'].setChecked(controls['plane_cb'].isChecked())
        
        # Unblock signals
        for controls in self.plane_controls.values():
            controls['plane_cb'].blockSignals(False)
            controls['pole_cb'].blockSignals(False)
        
        self.on_plane_changed()
    
    def reset_plane_color(self, plane_name):
        """Reset a plane's color to default"""
        self.plane_controls[plane_name]['color_edit'].setText(self.DEFAULT_COLORS[plane_name])
    
    def get_plane_settings(self):
        """Return current plane and pole settings"""
        settings = {}
        for name, controls in self.plane_controls.items():
            settings[name] = {
                'enabled': controls['plane_cb'].isChecked(),
                'color': controls['color_edit'].text(),
                'pole': controls['pole_cb'].isChecked()
            }
        return settings
    
    def get_galactic_settings(self):
        """Return current galactic exclusion settings"""
        return {
            'enabled': self.galactic_enable_check.isChecked(),
            'offset': self.galactic_offset_spin.value(),
            'penalty': self.galactic_penalty_spin.value(),
            'color': self.galactic_color_edit.text(),
            'show_bounds': self.galactic_show_bounds.isChecked()
        }
    
    def on_galactic_changed(self):
        """Emit signal when galactic exclusion settings change"""
        parent = self.parent()
        if parent and hasattr(parent, 'on_galactic_changed'):
            parent.on_galactic_changed()
    
    def get_opposition_settings(self):
        """Return current opposition benefit settings"""
        return {
            'enabled': self.opposition_enable_check.isChecked(),
            'radius': self.opposition_radius_spin.value(),
            'benefit': self.opposition_benefit_spin.value(),
            'color': self.opposition_color_edit.text(),
            'show_bounds': self.opposition_show_bounds.isChecked()
        }
    
    def on_opposition_changed(self):
        """Emit signal when opposition benefit settings change"""
        parent = self.parent()
        if parent and hasattr(parent, 'on_opposition_changed'):
            parent.on_opposition_changed()
    
    def get_discovery_settings(self):
        """Return current discovery circumstance settings"""
        return {
            'hide_missing': self.hide_missing_discovery_check.isChecked()
        }
    
    def on_discovery_changed(self):
        """Emit signal when discovery settings change"""
        parent = self.parent()
        if parent and hasattr(parent, 'on_discovery_changed'):
            parent.on_discovery_changed()
    
    def get_site_filter_settings(self):
        """Return current site filter settings"""
        # Parse whitelist
        whitelist = []
        if self.site_whitelist_check.isChecked():
            text = self.site_whitelist_edit.text().strip()
            if text:
                whitelist = [s.strip() for s in text.split(',') if s.strip()]
        
        # Parse blacklist
        blacklist = []
        if self.site_blacklist_check.isChecked():
            text = self.site_blacklist_edit.text().strip()
            if text:
                blacklist = [s.strip() for s in text.split(',') if s.strip()]
        
        return {
            'whitelist_enabled': self.site_whitelist_check.isChecked(),
            'whitelist': whitelist,
            'blacklist_enabled': self.site_blacklist_check.isChecked(),
            'blacklist': blacklist
        }
    
    def on_site_filter_changed(self):
        """Emit signal when site filter settings change"""
        parent = self.parent()
        if parent and hasattr(parent, 'on_site_filter_changed'):
            parent.on_site_filter_changed()
    
    def get_sunmoon_settings(self):
        """Return current sun and moon display settings"""
        return {
            'show_sun': self.show_sun_check.isChecked(),
            'show_moon': self.show_moon_check.isChecked(),
            'show_phases': self.show_moon_phases_check.isChecked(),
            'lunar_exclusion_enabled': self.lunar_exclusion_check.isChecked(),
            'lunar_radius': self.lunar_radius_spin.value(),
            'lunar_penalty': self.lunar_penalty_spin.value(),
            'lunar_color': self.lunar_color_edit.text(),
            'lunar_show_bounds': self.lunar_show_bounds.isChecked()
        }
    
    def on_sunmoon_changed(self):
        """Emit signal when sun/moon settings change"""
        # Enable/disable moon-related controls based on moon checkbox
        moon_enabled = self.show_moon_check.isChecked()
        self.lunar_exclusion_check.setEnabled(moon_enabled)
        self.show_moon_phases_check.setEnabled(moon_enabled)
        
        # Enable/disable lunar exclusion controls based on both moon and exclusion checkboxes
        lunar_excl_enabled = moon_enabled and self.lunar_exclusion_check.isChecked()
        self.lunar_radius_spin.setEnabled(lunar_excl_enabled)
        self.lunar_penalty_spin.setEnabled(lunar_excl_enabled)
        self.lunar_color_edit.setEnabled(lunar_excl_enabled)
        self.lunar_show_bounds.setEnabled(lunar_excl_enabled)
        
        parent = self.parent()
        if parent and hasattr(parent, 'on_sunmoon_changed'):
            parent.on_sunmoon_changed()
    
    def get_horizon_settings(self):
        """Return current horizon/twilight display settings"""
        return {
            'enabled': self.horizon_enable_check.isChecked(),
            'observer_lat': self.observer_lat_spin.value(),
            'observer_lon': self.observer_lon_spin.value(),
            'timezone': self.timezone_combo.currentText(),
            'show_horizon': self.show_horizon_check.isChecked(),
            'horizon_color': self.horizon_color_edit.text(),
            'show_civil': self.show_civil_check.isChecked(),
            'civil_color': self.civil_color_edit.text(),
            'show_nautical': self.show_nautical_check.isChecked(),
            'nautical_color': self.nautical_color_edit.text(),
            'show_astro': self.show_astro_check.isChecked(),
            'astro_color': self.astro_color_edit.text(),
            'line_style': self.horizon_style_combo.currentText(),
            'line_weight': self.horizon_weight_spin.value()
        }
    
    def on_horizon_changed(self):
        """Emit signal when horizon/twilight settings change"""
        # Enable/disable dependent controls based on main checkbox
        enabled = self.horizon_enable_check.isChecked()
        self.observer_lat_spin.setEnabled(enabled)
        self.observer_lon_spin.setEnabled(enabled)
        self.location_preset_combo.setEnabled(enabled)
        self.timezone_combo.setEnabled(enabled)
        self.show_horizon_check.setEnabled(enabled)
        self.horizon_color_edit.setEnabled(enabled and self.show_horizon_check.isChecked())
        self.show_civil_check.setEnabled(enabled)
        self.civil_color_edit.setEnabled(enabled and self.show_civil_check.isChecked())
        self.show_nautical_check.setEnabled(enabled)
        self.nautical_color_edit.setEnabled(enabled and self.show_nautical_check.isChecked())
        self.show_astro_check.setEnabled(enabled)
        self.astro_color_edit.setEnabled(enabled and self.show_astro_check.isChecked())
        self.horizon_style_combo.setEnabled(enabled)
        self.horizon_weight_spin.setEnabled(enabled)
        
        parent = self.parent()
        if parent and hasattr(parent, 'on_horizon_changed'):
            parent.on_horizon_changed()
    
    def _on_location_preset_changed(self, preset_name):
        """Handle location preset selection"""
        # Observatory locations (lat, lon, timezone)
        presets = {
            "Tucson, AZ (CSS)": (32.2226, -110.9747, "UTC-7 (MST)"),
            "Haleakala, HI (ATLAS/PS)": (20.7084, -156.2570, "UTC-10 (HST)"),
            "Siding Spring, AU (CSS-S)": (-31.2733, 149.0617, "UTC+10 (AEST)"),
            "La Palma, ES": (28.7606, -17.8816, "UTC+0 (WET)"),
            "Cerro Pachón, CL": (-30.2407, -70.7369, "UTC-3 (CLT)"),
        }
        
        if preset_name in presets:
            lat, lon, tz = presets[preset_name]
            # Block signals to avoid triggering change events during update
            self.observer_lat_spin.blockSignals(True)
            self.observer_lon_spin.blockSignals(True)
            self.timezone_combo.blockSignals(True)
            self.observer_lat_spin.setValue(lat)
            self.observer_lon_spin.setValue(lon)
            self.timezone_combo.setCurrentText(tz)
            self.observer_lat_spin.blockSignals(False)
            self.observer_lon_spin.blockSignals(False)
            self.timezone_combo.blockSignals(False)
            # Now trigger update
            self.on_horizon_changed()
        # "Custom" just leaves current values alone
    
    def get_appearance_settings(self):
        """Return current appearance settings"""
        return {
            'kiosk_mode': self.kiosk_mode_check.isChecked()
        }
    
    def on_appearance_changed(self):
        """Emit signal when appearance settings change"""
        parent = self.parent()
        if parent and hasattr(parent, 'on_appearance_changed'):
            parent.on_appearance_changed()
    
    def get_advanced_settings(self):
        """Return current advanced control settings"""
        return {
            'lr_increment': self.lr_increment_spin.value(),
            'lr_unit': self.lr_unit_combo.currentText(),
            'ud_increment': self.ud_increment_spin.value(),
            'ud_unit': self.ud_unit_combo.currentText()
        }
    
    def get_trailing_settings(self):
        """Return current trailing settings"""
        return {
            'enabled': self.enable_trailing_check.isChecked(),
            'length': self.trail_length_spin.value(),
            'weight': self.trail_weight_spin.value(),
            'color': self.trail_color.text()
        }

    def get_fast_animation_settings(self):
        """Return fast animation settings"""
        return {
            'enabled': self.fast_animation_check.isChecked(),
            'decimation': self.decimate_spin.value()
        }

    def on_trailing_changed(self):
        """Handle changes to trailing settings"""
        # Enable/disable dependent controls
        enabled = self.enable_trailing_check.isChecked()
        self.trail_length_spin.setEnabled(enabled)
        self.trail_weight_spin.setEnabled(enabled)
        self.trail_color.setEnabled(enabled)
        self.clear_trails_btn.setEnabled(enabled)
        
        # Notify parent
        parent = self.parent()
        if parent and hasattr(parent, 'on_trailing_changed'):
            parent.on_trailing_changed()
    
    def on_clear_trails(self):
        """Handle clear trails button click"""
        parent = self.parent()
        if parent and hasattr(parent, 'clear_trails'):
            parent.clear_trails()
    
    def on_plane_changed(self):
        """Emit signal when any plane setting changes"""
        # Notify parent to redraw
        parent = self.parent()
        if parent and hasattr(parent, 'on_planes_changed'):
            parent.on_planes_changed()
    
    def done(self, result):
        """Override done to remove widgets before dialog is destroyed"""
        # Remove widgets from layout and unparent them to prevent deletion
        if self._layout:
            self._layout.removeWidget(self.orbital_panel)
            self._layout.removeWidget(self.colorbar_panel)
            self.orbital_panel.setParent(None)
            self.colorbar_panel.setParent(None)
        super().done(result)


class NEOVisualizer(QMainWindow):
    """NEOlyzer - Professional NEO visualization and analysis"""
    
    def __init__(self, use_cache=True, show_fps=False):
        super().__init__()

        self.db = None
        self.cache = None
        self.calculator = None
        self.use_cache = use_cache
        self.show_fps = show_fps
        self.asteroid_classes = {}
        
        # Performance optimization: cache asteroid lists
        self.cached_asteroids = None
        self.cached_filter_state = None  # Track filter state
        
        # Performance: track last overlay JD to avoid unnecessary redraws
        self.last_overlay_jd = None
        
        # Table dialog (non-modal, persists)
        self.table_dialog = None
        
        self.setWindowTitle("NEOlyzer v3.02")
        
        # Dynamically size window to fit screen
        screen = QApplication.primaryScreen().geometry()
        width = min(1600, screen.width() - 100)
        height = min(1000, screen.height() - 100)
        self.setGeometry(50, 50, width, height)
        
        self.setup_ui()
        self.setup_statusbar()
        self.setup_keyboard_shortcuts()
        
        QTimer.singleShot(100, self.initialize_data)
    
    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        
        # Track dialog states for toggle behavior
        self.settings_dialog = None
        self.help_dialog = None
        
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(5, 5, 5, 2)  # Main margins
        main_layout.setSpacing(1)
        
        # TOP SECTION - Logo and Control Panel side-by-side (COMPACT!)
        top_section = QHBoxLayout()
        top_section.setSpacing(8)  # Space between logo and controls
        top_section.setContentsMargins(0, 0, 0, 0)
        top_section.setAlignment(QtCompat.AlignTop)  # Align all panels at top
        
        # LEFT: CSS Logo (25% bigger - 100x100) with extra margin
        self.logo_path = os.path.join(os.path.dirname(__file__), '..', 'assets', 'CSS_logo_transparent.png')
        self.logo_container = None
        
        if os.path.exists(self.logo_path):
            # Container widget for logo with extra margins
            self.logo_container = QWidget()
            logo_container_layout = QVBoxLayout()
            logo_container_layout.setContentsMargins(10, 8, 6, 0)  # Balanced margins
            
            logo_label = QLabel()
            logo_label.setCursor(QCursor(QtCompat.PointingHandCursor))
            pixmap = QPixmap(self.logo_path)
            # 10% bigger: 100 -> 110
            scaled_pixmap = pixmap.scaled(110, 110, QtCompat.KeepAspectRatio, 
                                          QtCompat.SmoothTransformation)
            logo_label.setPixmap(scaled_pixmap)
            logo_label.setToolTip("Click to visit Catalina Sky Survey website")
            logo_label.mousePressEvent = lambda event: self.open_css_website()
            logo_label.setAlignment(QtCompat.AlignTop | QtCompat.AlignLeft)
            logo_container_layout.addWidget(logo_label)
            self.logo_container.setLayout(logo_container_layout)
            self.logo_container.setMaximumSize(130, 130)  # Larger container
            top_section.addWidget(self.logo_container, 0, QtCompat.AlignTop)
        
        # RIGHT: Control Panel (compact horizontal layout)
        control_section = QWidget()
        control_section.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        control_layout = QHBoxLayout()
        control_layout.setContentsMargins(0, 0, 0, 0)
        control_layout.setSpacing(3)
        control_layout.setAlignment(QtCompat.AlignTop)  # Align panels at top
        
        # Date/Time panel
        self.time_panel = TimeControlPanel()
        self.time_panel.time_changed.connect(self.update_display)
        control_layout.addWidget(self.time_panel, 0, QtCompat.AlignTop)
        
        # Controls panel (includes animation + buttons)
        self.controls_panel = ControlsPanel(self.time_panel, self)
        self.time_panel.animation_timer.timeout.connect(self.controls_panel.advance_time)
        control_layout.addWidget(self.controls_panel, 0, QtCompat.AlignTop)
        
        # Projection panel (between Controls and Magnitude Ranges)
        self.proj_panel = ProjectionPanel()
        self.proj_panel.settings_changed.connect(self.update_display)
        control_layout.addWidget(self.proj_panel, 0, QtCompat.AlignTop)
        
        # Magnitude Ranges panel
        self.magnitude_panel = MagnitudeRangesPanel()
        self.magnitude_panel.filters_changed.connect(self.update_display)
        control_layout.addWidget(self.magnitude_panel, 0, QtCompat.AlignTop)
        
        # NEO Classes panel
        self.neo_classes_panel = NEOClassesPanel()
        self.neo_classes_panel.filters_changed.connect(self.update_display)
        self.neo_classes_panel.show_hollow_check.stateChanged.connect(self.update_display)
        self.neo_classes_panel.hide_before_discovery_check.stateChanged.connect(self.update_display)
        control_layout.addWidget(self.neo_classes_panel, 0, QtCompat.AlignTop)
        
        # Store all control panels for drawer toggle
        self.control_panels = [
            self.time_panel,
            self.controls_panel,
            self.proj_panel,
            self.magnitude_panel,
            self.neo_classes_panel
        ]
        
        control_section.setLayout(control_layout)
        self.control_section = control_section  # Store for drawer toggle
        top_section.addWidget(control_section, 1, QtCompat.AlignTop)
        
        # Create drawer toggle bar (hidden by default, shown in collapsible mode)
        self.drawer_bar = QPushButton("▼ Controls")
        self.drawer_bar.setStyleSheet("""
            QPushButton {
                text-align: left;
                padding: 6px 12px;
                background: #404040;
                color: white;
                border: none;
                font-weight: bold;
                font-size: 12pt;
            }
            QPushButton:hover {
                background: #505050;
            }
        """)
        self.drawer_bar.clicked.connect(self.toggle_drawer)
        self.drawer_bar.hide()  # Hidden by default
        self.drawer_collapsed = False
        
        # Store top section widget for drawer control
        self.top_content = QWidget()
        self.top_content.setLayout(top_section)
        self.top_content.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        
        # Wrapper that contains drawer bar + content
        top_wrapper = QVBoxLayout()
        top_wrapper.setContentsMargins(0, 0, 0, 0)
        top_wrapper.setSpacing(0)
        top_wrapper.addWidget(self.drawer_bar)
        top_wrapper.addWidget(self.top_content)
        
        top_widget = QWidget()
        top_widget.setLayout(top_wrapper)
        top_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        main_layout.addWidget(top_widget)
        
        # BOTTOM SECTION - Plot Area (full width, gets all remaining space)
        plot_widget = QWidget()
        plot_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        plot_layout = QVBoxLayout()
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(0)
        
        self.canvas = SkyMapCanvas(show_fps=self.show_fps)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        # Connect canvas click to toggle animation
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)
        
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas, 1)
        plot_widget.setLayout(plot_layout)
        
        main_layout.addWidget(plot_widget, 1)  # Give plot area stretch factor
        
        # Create orbital panel for Settings dialog
        self.orbital_panel = OrbitalElementsPanel()
        self.orbital_panel.filters_changed.connect(self.update_display)
        
        # Create colorbar panel for Settings dialog
        self.colorbar_panel = ColorbarPanel()
        self.colorbar_panel.settings_changed.connect(self.update_display)
        self.colorbar_panel.cbar_min.valueChanged.connect(self.update_colorbar)
        self.colorbar_panel.cbar_max.valueChanged.connect(self.update_colorbar)
        
        # Connect projection panel signals (proj_panel already created above)
        self.proj_panel.settings_changed.connect(self.update_projection)
        
        central.setLayout(main_layout)
    
    def show_settings(self):
        """Toggle Settings dialog (orbital elements, colorbar, symbol size)"""
        # Clear any NEO highlight when opening this dialog
        self.canvas.clear_highlight()
        if hasattr(self.canvas, 'current_info_dialog') and self.canvas.current_info_dialog is not None:
            try:
                self.canvas.current_info_dialog.canvas = None  # Prevent double clear
                self.canvas.current_info_dialog.close()
                self.canvas.current_info_dialog = None
            except:
                pass
        
        # Create dialog only once, then show/hide
        if self.settings_dialog is None:
            self.settings_dialog = SettingsDialog(self.orbital_panel, self.proj_panel, self.colorbar_panel, self)
        
        if self.settings_dialog.isVisible():
            self.settings_dialog.hide()
        else:
            self.settings_dialog.show()
            self.settings_dialog.raise_()  # Bring to front
    
    def on_planes_changed(self):
        """Handle changes to plane/pole settings"""
        if self.settings_dialog and hasattr(self.settings_dialog, 'get_plane_settings'):
            settings = self.settings_dialog.get_plane_settings()
            self.canvas.plane_settings = settings
            # Force redraw of overlays
            self.canvas.last_overlay_jd = None
            if hasattr(self.time_panel, 'current_jd') and self.time_panel.current_jd:
                self.canvas.draw_celestial_overlays(self.time_panel.current_jd)
                self.canvas.draw()
    
    def on_galactic_changed(self):
        """Handle changes to galactic exclusion settings"""
        # Force overlay redraw
        if hasattr(self.canvas, 'last_jd'):
            self.canvas.last_jd = None
        self.update_display()
    
    def on_opposition_changed(self):
        """Handle changes to opposition benefit settings"""
        # Force overlay redraw
        if hasattr(self.canvas, 'last_jd'):
            self.canvas.last_jd = None
        self.update_display()
    
    def on_discovery_changed(self):
        """Handle changes to discovery settings"""
        self.update_display()
    
    def on_site_filter_changed(self):
        """Handle changes to site filter settings"""
        self.update_display()
    
    def on_sunmoon_changed(self):
        """Handle changes to sun/moon display settings"""
        # Force overlay redraw
        if hasattr(self.canvas, 'last_jd'):
            self.canvas.last_jd = None
        self.update_display()
    
    def on_horizon_changed(self):
        """Handle changes to horizon/twilight display settings"""
        # Force overlay redraw
        if hasattr(self.canvas, 'last_jd'):
            self.canvas.last_jd = None
        self.update_display()
    
    def on_appearance_changed(self):
        """Handle changes to appearance settings (kiosk mode)"""
        if self.settings_dialog and hasattr(self.settings_dialog, 'get_appearance_settings'):
            settings = self.settings_dialog.get_appearance_settings()
            kiosk_mode = settings.get('kiosk_mode', False)
            
            if kiosk_mode:
                # Show drawer bar, content stays visible until user clicks
                self.drawer_bar.show()
            else:
                # Hide drawer bar and ensure content is visible
                self.drawer_bar.hide()
                self.top_content.show()
                self.drawer_collapsed = False
                self.drawer_bar.setText("▼ Controls")
    
    def on_trailing_changed(self):
        """Handle changes to trailing settings"""
        if self.settings_dialog and hasattr(self.settings_dialog, 'get_trailing_settings'):
            settings = self.settings_dialog.get_trailing_settings()
            # Update canvas with new trailing settings
            if hasattr(self.canvas, 'set_trailing_settings'):
                self.canvas.set_trailing_settings(settings)
    
    def clear_trails(self):
        """Clear all trail history and visual trail lines"""
        if hasattr(self.canvas, 'trail_history'):
            self.canvas.trail_history.clear()
            logger.debug("TRAIL: Cleared trail_history")
        if hasattr(self.canvas, '_clear_trails'):
            self.canvas._clear_trails()
            logger.debug("TRAIL: Cleared trail_lines")
        # Reset last trail JD to prevent immediate re-clearing
        self._last_trail_jd = getattr(self.time_panel, 'current_jd', 0) if hasattr(self, 'time_panel') else 0
        # Force redraw to show cleared trails
        if hasattr(self.canvas, 'draw'):
            self.canvas.draw()
    
    def toggle_drawer(self):
        """Toggle the control panel drawer open/closed"""
        self.drawer_collapsed = not self.drawer_collapsed
        self.top_content.setVisible(not self.drawer_collapsed)
        self.drawer_bar.setText("▶ Controls" if self.drawer_collapsed else "▼ Controls")
    
    def toggle_table(self):
        """Toggle NEO table dialog visibility"""
        if self.table_dialog is None:
            self.table_dialog = NEOTableDialog(self)
            # Position to the right of main window
            main_geom = self.geometry()
            self.table_dialog.move(main_geom.right() + 10, main_geom.top())
        
        if self.table_dialog.isVisible():
            self.table_dialog.hide()
        else:
            self.table_dialog.show()
            self.update_table()
    
    def update_table(self):
        """Update table dialog with current visible NEOs"""
        if self.table_dialog is None or not self.table_dialog.isVisible():
            return
        
        # Get current visible data from canvas
        if hasattr(self.canvas, 'visible_data') and hasattr(self.canvas, 'current_asteroids'):
            self.table_dialog.update_table(
                self.canvas.visible_data,
                self.canvas.current_asteroids,
                self.canvas.current_jd
            )
    
    def show_table_with_selection(self, selected_data):
        """Show table dialog with only selected objects"""
        if self.table_dialog is None:
            self.table_dialog = NEOTableDialog(self)
            main_geom = self.geometry()
            self.table_dialog.move(main_geom.right() + 10, main_geom.top())
        
        # Update table with selected data only
        if hasattr(self.canvas, 'current_asteroids'):
            self.table_dialog.update_table(
                selected_data,
                self.canvas.current_asteroids,
                self.canvas.current_jd
            )
        
        self.table_dialog.show()
    
    def on_canvas_click(self, event):
        """Stop animation on canvas click (but don't start it - use Play button)"""
        if event.inaxes is not None:
            # Only stop animation, don't start it from canvas click
            # This allows click-to-identify to work when not animating
            if self.time_panel.animation_timer.isActive():
                self.controls_panel.toggle_play()  # This will stop it
    
    def setup_statusbar(self):
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.status_label = QLabel("Initializing...")
        self.statusbar.addWidget(self.status_label, 1)  # Stretch factor 1
        
        # Search field (left side of permanent widgets)
        search_label = QLabel("Search:")
        self.statusbar.addPermanentWidget(search_label)
        
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search: name, number, or designation (e.g., Apophis, 99942, K22C01Q)")
        self.search_input.setMaximumWidth(220)
        self.search_input.returnPressed.connect(self.search_object)
        self.statusbar.addPermanentWidget(self.search_input)
        
        search_btn = QPushButton("Find")
        search_btn.setMaximumWidth(45)
        search_btn.clicked.connect(self.search_object)
        self.statusbar.addPermanentWidget(search_btn)
        
        clear_btn = QPushButton("Clear")
        clear_btn.setMaximumWidth(45)
        clear_btn.clicked.connect(self.clear_search)
        self.statusbar.addPermanentWidget(clear_btn)
        
        # Spacer
        spacer = QLabel("  │  ")
        spacer.setStyleSheet("color: gray;")
        self.statusbar.addPermanentWidget(spacer)
        
        # Help, Settings, Save/Reset, Exit buttons (far right)
        # Play/Pause button (duplicate of controls panel button for convenience)
        self.statusbar_play_btn = QPushButton("▶ Play")
        self.statusbar_play_btn.setMaximumWidth(70)
        self.statusbar_play_btn.clicked.connect(self.toggle_play_from_statusbar)
        self.statusbar.addPermanentWidget(self.statusbar_play_btn)
        
        # Now button (sets time to current date/time)
        statusbar_now_btn = QPushButton("Now")
        statusbar_now_btn.setMaximumWidth(45)
        statusbar_now_btn.setToolTip("Set to current date and time")
        statusbar_now_btn.clicked.connect(self.set_to_now_from_statusbar)
        self.statusbar.addPermanentWidget(statusbar_now_btn)
        
        help_btn = QPushButton("❓ Help")
        help_btn.setMaximumWidth(60)
        help_btn.clicked.connect(self.show_help)
        self.statusbar.addPermanentWidget(help_btn)
        
        settings_btn = QPushButton("⚙️ Settings")
        settings_btn.setMaximumWidth(80)
        settings_btn.clicked.connect(self.show_settings)
        self.statusbar.addPermanentWidget(settings_btn)
        
        # Save button
        save_btn = QPushButton("💾 Save")
        save_btn.setMaximumWidth(60)
        save_btn.setToolTip("Save current settings to file")
        save_btn.clicked.connect(self.save_settings)
        self.statusbar.addPermanentWidget(save_btn)
        
        # Reset dropdown menu
        self.reset_btn = QPushButton("🔄 Reset ▾")
        self.reset_btn.setMaximumWidth(75)
        reset_menu = QMenu(self.reset_btn)
        reset_menu.addAction("Restore saved settings", self.restore_settings)
        reset_menu.addAction("Factory defaults", self.reset_all)
        self.reset_btn.setMenu(reset_menu)
        self.statusbar.addPermanentWidget(self.reset_btn)
        
        exit_btn = QPushButton("❌ Exit")
        exit_btn.setMaximumWidth(55)
        exit_btn.clicked.connect(self.close)
        exit_btn.setStyleSheet("background: #ffcccc;")
        self.statusbar.addPermanentWidget(exit_btn)
    
    def search_object(self):
        """Search for an object by designation and show its info"""
        search_text = self.search_input.text().strip()
        if not search_text:
            return
        
        # Use the canvas search method
        result = self.canvas.find_and_highlight_designation(search_text)
        ast, ra, dec, dist, mag, visible = result
        
        if ast is None:
            self.status_label.setText(f"Object '{search_text}' not found in current dataset")
            return
        
        # Show info dialog
        if ra is not None:
            # Get mag_max from magnitude panel
            _, mag_max = self.magnitude_panel.get_magnitude_limits()
            dialog = NEOInfoDialog(ast, ra, dec, dist, mag, self.time_panel.current_jd, self, 
                                  canvas=self.canvas, mag_max=mag_max, calculator=self.calculator)
            if not visible:
                self.status_label.setText(f"Found {ast['designation']} - not currently visible (filtered out)")
            else:
                self.status_label.setText(f"Found {ast['designation']} - highlighted on plot")
            dialog.exec()
        else:
            self.status_label.setText(f"Found {ast['designation']} but position data unavailable")
    
    def clear_search(self):
        """Clear the search input and highlight"""
        self.search_input.clear()
        self.canvas.clear_highlight()
        self.status_label.setText("Highlight cleared")
    
    def toggle_play_from_statusbar(self):
        """Toggle animation from statusbar button"""
        self.controls_panel.toggle_play()
        # Sync statusbar button text with controls panel button
        self.statusbar_play_btn.setText(self.controls_panel.play_btn.text())
    
    def set_to_now_from_statusbar(self):
        """Set time to now from statusbar button"""
        self.time_panel.set_to_now()
    
    def setup_keyboard_shortcuts(self):
        """Set up keyboard shortcuts for time navigation.
        
        Uses [ and ] keys which work reliably across all platforms.
        Ctrl+Arrow keys are often captured by OS for desktop/space navigation.
        """
        # Bracket keys - work universally
        shortcut_back = QShortcut(QKeySequence("["), self)
        shortcut_back.activated.connect(self.navigate_time_backward)
        
        shortcut_fwd = QShortcut(QKeySequence("]"), self)
        shortcut_fwd.activated.connect(self.navigate_time_forward)
        
        # Also try Ctrl+brackets for consistency
        shortcut_ctrl_back = QShortcut(QKeySequence("Ctrl+["), self)
        shortcut_ctrl_back.activated.connect(self.navigate_time_backward)
        
        shortcut_ctrl_fwd = QShortcut(QKeySequence("Ctrl+]"), self)
        shortcut_ctrl_fwd.activated.connect(self.navigate_time_forward)
        
        # Store shortcuts to prevent garbage collection
        self._shortcuts = [shortcut_back, shortcut_fwd, shortcut_ctrl_back, shortcut_ctrl_fwd]
    
    def navigate_time_backward(self):
        """Navigate time backward by configured increment"""
        delta = self._get_time_delta()
        if delta:
            # Clear trails if jump is > 1 day
            if delta > 1.0:
                self.clear_trails()
            current_jd = self.time_panel.current_jd
            self.time_panel.set_jd(current_jd - delta)
    
    def navigate_time_forward(self):
        """Navigate time forward by configured increment"""
        delta = self._get_time_delta()
        if delta:
            # Clear trails if jump is > 1 day
            if delta > 1.0:
                self.clear_trails()
            current_jd = self.time_panel.current_jd
            self.time_panel.set_jd(current_jd + delta)
    
    def _get_time_delta(self):
        """Get time delta in days based on settings"""
        # Get settings
        if self.settings_dialog and hasattr(self.settings_dialog, 'get_advanced_settings'):
            settings = self.settings_dialog.get_advanced_settings()
        else:
            settings = {'lr_increment': 1, 'lr_unit': 'solar day'}
        
        # Constants
        SIDEREAL_DAY = 0.99726957
        LUNATION = 29.530588853
        
        increment = settings.get('lr_increment', 1)
        unit = settings.get('lr_unit', 'solar day')
        
        if unit == 'hour':
            return increment / 24.0
        elif unit == 'solar day':
            return increment
        elif unit == 'sidereal day':
            return increment * SIDEREAL_DAY
        elif unit == 'lunation':
            return increment * LUNATION
        elif unit == 'month':
            return increment * 30.4375
        elif unit == 'year':
            return increment * 365.25
        return 1.0  # default
    
    def initialize_data(self):
        try:
            logger.info("Initializing...")
            self.db = DatabaseManager(use_sqlite=True)
            self.cache = PositionCache() if self.use_cache else None
            self.calculator = FastOrbitCalculator()
            
            stats = self.db.get_statistics()
            if stats['total'] == 0:
                self.status_label.setText("No data")
                return
            
            # Build class lookup
            all_ast = self.db.get_asteroids()
            for ast in all_ast:
                self.asteroid_classes[int(ast['id'])] = ast.get('orbit_class', 'Other')
            
            # Calculate orbital element ranges from data
            if all_ast:
                periods = [ast['a'] ** 1.5 for ast in all_ast if ast['a'] > 0]
                eccentricities = [ast['e'] for ast in all_ast]
                inclinations = [ast['i'] for ast in all_ast]
                
                if periods:
                    period_range = (min(periods), max(periods))
                else:
                    period_range = None
                
                ecc_range = (min(eccentricities), max(eccentricities)) if eccentricities else None
                inc_range = (min(inclinations), max(inclinations)) if inclinations else None
                
                # Update orbital panel with actual ranges
                self.orbital_panel.set_ranges_from_data(period_range, ecc_range, inc_range)
            
            logger.info(f"Loaded {len(self.asteroid_classes)} asteroids")
            self.status_label.setText(f"Ready: {stats['total']} asteroids")
            self.update_display()
            
        except Exception as e:
            self.status_label.setText(f"Error: {e}")
            logger.error(f"Init error: {e}", exc_info=True)
    
    def update_projection(self):
        """Update projection, coordinate system, and colormap"""
        proj = self.proj_panel.get_projection()
        coord = self.proj_panel.get_coordinate_system()
        cmap = self.colorbar_panel.get_colormap()
        h_res, v_res = self.proj_panel.get_resolution()
        display_mode = self.proj_panel.get_display_mode()
        
        self.canvas.set_projection(proj)
        self.canvas.set_coordinate_system(coord)
        self.canvas.set_colormap(cmap)
        self.canvas.set_resolution(h_res, v_res)
        self.canvas.set_display_mode(display_mode)
        
        # Force overlay redraw on projection/coordinate change
        if hasattr(self.canvas, 'last_jd'):
            self.canvas.last_jd = None
        
        self.update_display()
    
    def update_colorbar(self):
        vmin, vmax = self.colorbar_panel.get_colorbar_range()
        self.canvas.set_colorbar_range(vmin, vmax)
    
    def update_display(self):
        """Update display - OPTIMIZED with caching"""
        if not self.calculator or not self.db:
            return
        
        try:
            # Update display mode and settings from projection panel
            if hasattr(self, 'proj_panel'):
                display_mode = self.proj_panel.get_display_mode()
                self.canvas.set_display_mode(display_mode)
                display_settings = self.proj_panel.get_display_settings()
                self.canvas.display_settings = display_settings
            
            jd = self.time_panel.current_jd
            
            # Clear trails if time jump > 1 day (non-animation navigation)
            # This handles CLN changes, date picker changes, etc.
            if hasattr(self.canvas, 'trail_history') and hasattr(self, '_last_trail_jd'):
                time_jump = abs(jd - self._last_trail_jd)
                # Only clear for large jumps when NOT animating
                # Animation uses small steps and should keep trails
                is_animating = self.time_panel.animation_timer.isActive()
                if time_jump > 1.0 and not is_animating:
                    logger.debug(f"TRAIL: Clearing trails due to large time jump: {time_jump:.1f} days")
                    self.canvas.trail_history.clear()
                    self.canvas._clear_trails()
            self._last_trail_jd = jd
            
            mag_min, mag_max = self.magnitude_panel.get_magnitude_limits()
            h_min, h_max = self.magnitude_panel.get_h_limits()
            show_all_neos = self.magnitude_panel.get_show_all_neos()
            
            # Override V/H limits if showing all NEOs
            if show_all_neos:
                mag_min, mag_max = -100.0, 200.0  # Extreme range to include H=99.99 objects
                h_min, h_max = None, None         # None bypasses H filter in database query
            
            moid_enabled, moid_min, moid_max = self.neo_classes_panel.get_moid_filter()
            selected_classes = self.neo_classes_panel.get_selected_classes()
            
            # Get orbital element filters
            orb_filters = self.orbital_panel.get_orbital_filters()
            
            # Get symbol size settings
            size_settings = self.colorbar_panel.get_size_settings()
            
            # Get galactic exclusion settings
            if self.settings_dialog and hasattr(self.settings_dialog, 'get_galactic_settings'):
                galactic_settings = self.settings_dialog.get_galactic_settings()
            else:
                # Defaults if dialog not available
                galactic_settings = {'enabled': False, 'offset': 15.0, 'penalty': 2.0, 'color': '#FF99FF', 'show_bounds': True}
            
            # Get opposition benefit settings
            if self.settings_dialog and hasattr(self.settings_dialog, 'get_opposition_settings'):
                opposition_settings = self.settings_dialog.get_opposition_settings()
            else:
                # Defaults if dialog not available
                opposition_settings = {'enabled': False, 'radius': 5.0, 'benefit': 2.0, 'color': '#90EE90', 'show_bounds': True}
            
            # Get discovery settings
            if self.settings_dialog and hasattr(self.settings_dialog, 'get_discovery_settings'):
                discovery_settings = self.settings_dialog.get_discovery_settings()
            else:
                discovery_settings = {'hide_missing': False}
            
            # Get hide before discovery setting
            hide_before_discovery = self.neo_classes_panel.hide_before_discovery_check.isChecked()
            hide_missing_discovery = discovery_settings.get('hide_missing', False)
            
            # Get color by settings
            color_by = self.colorbar_panel.get_color_by()
            show_legend = self.colorbar_panel.get_show_legend()
            
            # Get site filter settings
            if self.settings_dialog and hasattr(self.settings_dialog, 'get_site_filter_settings'):
                site_filter = self.settings_dialog.get_site_filter_settings()
            else:
                site_filter = {'whitelist_enabled': False, 'whitelist': [], 'blacklist_enabled': False, 'blacklist': []}
            
            # Get sun/moon display settings
            if self.settings_dialog and hasattr(self.settings_dialog, 'get_sunmoon_settings'):
                sunmoon_settings = self.settings_dialog.get_sunmoon_settings()
            else:
                sunmoon_settings = {'show_sun': True, 'show_moon': True, 'show_phases': False,
                                    'lunar_exclusion_enabled': False, 'lunar_radius': 30.0, 
                                    'lunar_penalty': 3.0, 'lunar_color': '#228B22', 'lunar_show_bounds': True}
            
            # Store sunmoon settings on canvas for draw_celestial_overlays
            self.canvas.sunmoon_settings = sunmoon_settings
            
            # Get horizon/twilight display settings
            if self.settings_dialog and hasattr(self.settings_dialog, 'get_horizon_settings'):
                horizon_settings = self.settings_dialog.get_horizon_settings()
            else:
                horizon_settings = {'enabled': False, 'observer_lat': 32.2226, 'observer_lon': -110.9747,
                                   'show_horizon': True, 'horizon_color': '#FF6600',
                                   'show_civil': False, 'civil_color': '#FF9933',
                                   'show_nautical': False, 'nautical_color': '#CC66FF',
                                   'show_astro': True, 'astro_color': '#6666FF',
                                   'line_style': 'solid', 'line_weight': 1.5}
            
            # Store horizon settings on canvas for draw_celestial_overlays
            self.canvas.horizon_settings = horizon_settings
            
            # FIX: If no classes selected, show nothing
            if selected_classes is None:
                self.canvas.update_plot(None, mag_min, mag_max, jd, self.neo_classes_panel.show_hollow_check.isChecked(),
                                       h_min, h_max, selected_classes, moid_enabled, moid_min, moid_max, orb_filters, 
                                       asteroids=None, size_settings=size_settings,
                                       galactic_settings=galactic_settings, opposition_settings=opposition_settings,
                                       hide_before_discovery=hide_before_discovery, hide_missing_discovery=hide_missing_discovery,
                                       color_by=color_by, show_legend=show_legend, site_filter=site_filter)
                self.status_label.setText("No orbit classes selected")
                return
            
            # Performance optimization: Check if filters changed
            # Include MOID and orbital element filters in cache key
            current_filter_state = (tuple(sorted(selected_classes)), h_min, h_max, moid_enabled, moid_min, moid_max,
                                  orb_filters['period_enabled'], orb_filters['period_min'], orb_filters['period_max'],
                                  orb_filters['ecc_enabled'], orb_filters['ecc_min'], orb_filters['ecc_max'],
                                  orb_filters['inc_enabled'], orb_filters['inc_min'], orb_filters['inc_max'])
            filters_changed = (self.cached_filter_state != current_filter_state)
            
            # Determine moid parameters for database query
            moid_min_param = moid_min if moid_enabled else None
            moid_max_param = moid_max if moid_enabled else None
            
            # Only query database if filters changed
            if filters_changed or self.cached_asteroids is None:
                logger.debug(f"Filters changed, querying database...")
                # Get asteroids for selected classes with SQL filtering
                asteroids = []
                for cls in selected_classes:
                    if cls == 'Amor, q≤1.15':
                        # Get all Amors with H and MOID filters, then filter by q in Python
                        all_amors = self.db.get_asteroids(orbit_class='Amor', h_min=h_min, h_max=h_max, 
                                                          moid_min=moid_min_param, moid_max=moid_max_param)
                        filtered = [ast for ast in all_amors if ast['a'] * (1 - ast['e']) <= 1.15]
                        asteroids.extend(filtered)
                    elif cls == 'Amor, q>1.15':
                        # Get all Amors with H and MOID filters, then filter by q in Python
                        all_amors = self.db.get_asteroids(orbit_class='Amor', h_min=h_min, h_max=h_max,
                                                          moid_min=moid_min_param, moid_max=moid_max_param)
                        filtered = [ast for ast in all_amors if ast['a'] * (1 - ast['e']) > 1.15]
                        asteroids.extend(filtered)
                    else:
                        # Regular class - use SQL H and MOID filtering
                        asteroids.extend(self.db.get_asteroids(orbit_class=cls, h_min=h_min, h_max=h_max,
                                                               moid_min=moid_min_param, moid_max=moid_max_param))
                
                # Cache the results
                self.cached_asteroids = asteroids
                self.cached_filter_state = current_filter_state
                logger.debug(f"Cached {len(asteroids)} asteroids")
            else:
                # Use cached list
                asteroids = self.cached_asteroids
            
            # Apply orbital element filters (Python-side filtering)
            if orb_filters['period_enabled'] or orb_filters['ecc_enabled'] or orb_filters['inc_enabled']:
                filtered_asteroids = []
                for ast in asteroids:
                    # Check period (calculate from semi-major axis: P = a^1.5)
                    if orb_filters['period_enabled']:
                        period = ast['a'] ** 1.5
                        if period < orb_filters['period_min'] or period > orb_filters['period_max']:
                            continue
                    
                    # Check eccentricity
                    if orb_filters['ecc_enabled']:
                        if ast['e'] < orb_filters['ecc_min'] or ast['e'] > orb_filters['ecc_max']:
                            continue
                    
                    # Check inclination
                    if orb_filters['inc_enabled']:
                        if ast['i'] < orb_filters['inc_min'] or ast['i'] > orb_filters['inc_max']:
                            continue
                    
                    filtered_asteroids.append(ast)
                
                asteroids = filtered_asteroids
            
            # Apply lunation discovery filter (show only NEOs discovered during current CLN)
            lunation_discoveries_only = self.magnitude_panel.get_lunation_discoveries_only()
            if lunation_discoveries_only:
                try:
                    # Use precise CLN calculation (changes at actual Full Moon)
                    current_cln, _ = jd_to_cln(jd)
                    
                    # Fast filter using precomputed discovery_cln
                    # Note: discovery_cln uses average method, so there may be ~1 day
                    # discrepancy near Full Moon boundaries
                    filtered_asteroids = [ast for ast in asteroids 
                                         if ast.get('discovery_cln') == current_cln]
                    
                    n_with_cln = sum(1 for ast in asteroids if ast.get('discovery_cln') is not None)
                    logger.debug(f"Lunation filter: CLN {current_cln}, {n_with_cln}/{len(asteroids)} have discovery_cln, {len(filtered_asteroids)} match")
                    asteroids = filtered_asteroids
                except Exception as e:
                    logger.error(f"Lunation filter error: {e}")
                    pass  # Keep all if CLN calculation fails
            
            if not asteroids:
                self.canvas.update_plot(None, mag_min, mag_max, jd, self.neo_classes_panel.show_hollow_check.isChecked(),
                                       h_min, h_max, selected_classes, moid_enabled, moid_min, moid_max, orb_filters,
                                       asteroids=None, size_settings=size_settings,
                                       galactic_settings=galactic_settings, opposition_settings=opposition_settings,
                                       hide_before_discovery=hide_before_discovery, hide_missing_discovery=hide_missing_discovery,
                                       color_by=color_by, show_legend=show_legend, site_filter=site_filter)
                if lunation_discoveries_only:
                    try:
                        current_cln, _ = jd_to_cln(jd)
                        self.status_label.setText(f"No discoveries in CLN {current_cln}")
                    except:
                        self.status_label.setText("No objects (lunation filter)")
                else:
                    self.status_label.setText("No objects")
                return
            
            # ALWAYS compute on-the-fly for smooth animation
            if self.show_fps:
                t0 = time.time()

            # Apply decimation for fast animation mode
            is_animating = self.time_panel.animation_timer.isActive()
            display_asteroids = asteroids
            if is_animating and self.settings_dialog and hasattr(self.settings_dialog, 'get_fast_animation_settings'):
                fast_settings = self.settings_dialog.get_fast_animation_settings()
                if fast_settings.get('enabled', False):
                    decimation = fast_settings.get('decimation', 4)
                    display_asteroids = asteroids[::decimation]

            positions = self.calculator.calculate_batch(display_asteroids, jd)
            if self.show_fps:
                t1 = time.time()

            # Update plot with hollow symbol visibility setting and all filter info
            # Pass asteroids list for click-to-identify feature
            self.canvas.update_plot(positions, mag_min, mag_max, jd, self.neo_classes_panel.show_hollow_check.isChecked(),
                                   h_min, h_max, selected_classes, moid_enabled, moid_min, moid_max, orb_filters,
                                   asteroids=display_asteroids, size_settings=size_settings,
                                   galactic_settings=galactic_settings, opposition_settings=opposition_settings,
                                   hide_before_discovery=hide_before_discovery, hide_missing_discovery=hide_missing_discovery,
                                   color_by=color_by, show_legend=show_legend, site_filter=site_filter)

            if self.show_fps:
                t2 = time.time()
                print(f"  calc: {(t1-t0)*1000:.0f}ms  plot: {(t2-t1)*1000:.0f}ms  total: {(t2-t0)*1000:.0f}ms")

            # Status
            if positions is not None and len(positions) > 0:
                mag = positions[:, 4]
                mask = (mag >= mag_min) & (mag < mag_max)
                n_shown = np.sum(mask)
                dt = self.time_panel.datetime_edit.dateTime().toString("yyyy-MM-dd HH:mm")
                if lunation_discoveries_only:
                    try:
                        current_cln, _ = jd_to_cln(jd)
                        self.status_label.setText(
                            f"{dt} | {n_shown} discoveries in CLN {current_cln}"
                        )
                    except:
                        self.status_label.setText(
                            f"{dt} | {n_shown} discoveries (lunation filter)"
                        )
                elif show_all_neos:
                    self.status_label.setText(
                        f"{dt} | {n_shown}/{len(positions)} objects (ALL NEOs)"
                    )
                else:
                    self.status_label.setText(
                        f"{dt} | {n_shown}/{len(positions)} objects ({mag_min:.1f} < V < {mag_max:.1f})"
                    )
            
            # Update table if open
            self.update_table()
            
        except Exception as e:
            self.status_label.setText(f"Error: {e}")
            logger.error(f"Update error: {e}", exc_info=True)
    
    def open_css_website(self):
        """Open Catalina Sky Survey website in default browser"""
        import webbrowser
        webbrowser.open('https://catalina.lpl.arizona.edu')
    
    def show_help(self):
        """Toggle help dialog with feature summary"""
        # Clear any NEO highlight when opening this dialog
        self.canvas.clear_highlight()
        if hasattr(self.canvas, 'current_info_dialog') and self.canvas.current_info_dialog is not None:
            try:
                self.canvas.current_info_dialog.canvas = None  # Prevent double clear
                self.canvas.current_info_dialog.close()
                self.canvas.current_info_dialog = None
            except:
                pass
        
        if self.help_dialog is not None and self.help_dialog.isVisible():
            self.help_dialog.close()
            self.help_dialog = None
            return
        
        self.help_dialog = QDialog(self)
        self.help_dialog.setWindowTitle("NEOlyzer - Help")
        self.help_dialog.setMinimumWidth(600)
        self.help_dialog.setMinimumHeight(500)
        
        layout = QVBoxLayout()
        
        # Create text browser for scrollable content
        text = QTextBrowser()
        text.setOpenExternalLinks(True)  # Enable clicking on mailto: and http: links
        
        help_text = """
        <h2>NEOlyzer</h2>
        <p><b>Version 2.03</b> - Near-Earth Object sky position visualization and analysis tool</p>
        
        <h3>Key Features</h3>
        <ul>
        <li><b>40,509 NEOs</b> with orbital elements and physical parameters</li>
        <li><b>Real-time position calculation</b> using JPL ephemerides</li>
        <li><b>Multiple projections:</b> Hammer, Aitoff, Mollweide, Rectangular</li>
        <li><b>Coordinate systems:</b> Equatorial, Ecliptic, Galactic, Opposition (ecliptic with opposition centered)</li>
        <li><b>NEO classification:</b> Atira, Aten, Apollo, Amor (near/far)</li>
        <li><b>Orbital element filters:</b> Period, Eccentricity, Inclination</li>
        <li><b>MOID filtering:</b> Show only PHAs or specific MOID ranges</li>
        <li><b>Time controls:</b> Animate, jump by day/month/year</li>
        <li><b>Hollow symbols:</b> Show NEOs behind the Sun</li>
        </ul>
        
        <h3>Map Elements</h3>
        <ul>
        <li><b>Filled circles:</b> NEOs on Earth's side of Sun</li>
        <li><b>Hollow circles:</b> NEOs behind Sun (optional)</li>
        <li><b>Color:</b> Visual magnitude (brightness)</li>
        <li><b>Yellow circle:</b> Sun position (red border)</li>
        <li><b>White reticle:</b> Solar opposition point</li>
        <li><b>Yellow line:</b> Ecliptic plane</li>
        <li><b>Cyan line:</b> Celestial equator (in ecliptic view)</li>
        <li><b>Magenta line:</b> Galactic plane</li>
        </ul>
        
        <h3>Caching Strategy</h3>
        <ul>
        <li><b>High precision (±6 months):</b> Daily positions</li>
        <li><b>Medium precision (±5 years):</b> Weekly positions</li>
        <li><b>Low precision (±50 years):</b> Monthly positions</li>
        </ul>
        
        <h3>Filters & Controls</h3>
        <ul>
        <li><b>Magnitude Ranges:</b> Filter by visual (V) and absolute (H) magnitude</li>
        <li><b>NEO/PHA Classes:</b> Select orbit types to display</li>
        <li><b>Orbital Elements:</b> Filter by period, eccentricity, and inclination</li>
        <li><b>MOID Filter:</b> Distance-based potentially hazardous asteroid selection</li>
        <li><b>Map Projection:</b> Choose projection type and coordinate system</li>
        <li><b>Date and Time:</b> Set specific dates or animate through time</li>
        <li><b>Animation:</b> Control speed and time step for animation</li>
        <li><b>Click on plot:</b> Toggle animation play/pause</li>
        </ul>
        
        <h3>Technology Suite</h3>
        <ul>
        <li><b>Operating Systems:</b> Cross-platform (Linux, macOS, Windows)</li>
        <li><b>Programming Language:</b> Python 3.9+</li>
        <li><b>GUI Framework:</b> PyQt6 for modern, native user interface</li>
        <li><b>Visualization:</b> Matplotlib with custom projections and overlays</li>
        <li><b>Database:</b> SQLite (lightweight) or PostgreSQL (full-scale) via SQLAlchemy ORM</li>
        <li><b>Orbital Mechanics:</b> Skyfield library (Brandon Rhodes) for high-precision ephemerides</li>
        <li><b>Ephemerides:</b> JPL DE421 planetary positions</li>
        <li><b>Coordinate Transforms:</b> Astropy for equatorial/ecliptic/galactic conversions</li>
        <li><b>Performance:</b> NumPy for vectorized calculations, multi-level caching system</li>
        <li><b>Data Sources:</b> MPC NEA.txt, JPL Small-Body Database, automated updates</li>
        </ul>
        
        <h3>Tips</h3>
        <ul>
        <li>Use <b>animation</b> to watch NEO motion over time</li>
        <li><b>Click on plot</b> to start/stop animation</li>
        <li>Toggle <b>"Show behind sun"</b> to see which NEOs are behind the Sun</li>
        <li>Enable <b>MOID filter</b> to focus on PHAs (0.000-0.050 AU)</li>
        <li>Use <b>Galactic Exclusion</b> in Settings to apply a magnitude penalty to NEOs near the galactic plane</li>
        <li>Use <b>Opposition Benefit</b> in Settings to brighten NEOs near the opposition point</li>
        <li>Try different <b>projections</b> for different perspectives</li>
        <li>The <b>stats box</b> shows current filter settings and counts</li>
        <li>Click <b>More</b> or <b>Help</b> again to close the popup</li>
        </ul>
        
        <hr>
        <p><b>Contact:</b> Rob Seaman, Catalina Sky Survey, <a href="mailto:rseaman@arizona.edu">rseaman@arizona.edu</a></p>
        
        <p style="font-size: small; color: #666;">
        <i>This visualization tool was developed with Claude (Anthropic). 
        Claude is an AI assistant created by Anthropic. Users own outputs 
        generated through Claude. For more information, see Anthropic's 
        terms of service at anthropic.com.</i>
        </p>
        """
        
        text.setHtml(help_text)
        layout.addWidget(text)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.help_dialog.accept)
        layout.addWidget(close_btn)
        
        self.help_dialog.setLayout(layout)
        self.help_dialog.show()  # Use show() not exec() for non-blocking
    
    def reset_all(self):
        try:
            # Time
            self.time_panel.set_to_now()
            
            # Controls panel
            self.controls_panel.rate_spin.setValue(1.0)
            self.controls_panel.rate_unit.setCurrentText('days/sec')
            self.controls_panel.fps_spin.setValue(10)
            self.controls_panel.stop_animation()
            
            # Magnitude panel
            self.magnitude_panel.mag_min_spin.setValue(19.0)
            self.magnitude_panel.mag_max_spin.setValue(22.0)
            self.magnitude_panel.h_min_spin.setValue(9.0)
            self.magnitude_panel.h_max_spin.setValue(22.0)
            self.magnitude_panel.show_all_neos_check.setChecked(False)
            self.magnitude_panel.lunation_discoveries_check.setChecked(False)
            
            # NEO classes panel
            self.neo_classes_panel.select_all()
            self.neo_classes_panel.moid_enabled_check.setChecked(False)
            self.neo_classes_panel.moid_min_spin.setValue(0.0)
            self.neo_classes_panel.moid_max_spin.setValue(0.05)
            self.neo_classes_panel.show_hollow_check.setChecked(False)
            self.neo_classes_panel.hide_before_discovery_check.setChecked(False)
            
            # Orbital elements panel
            self.orbital_panel.clear_period()
            self.orbital_panel.clear_ecc()
            self.orbital_panel.clear_inc()
            
            # Projection panel (map settings only)
            self.proj_panel.reset_defaults()
            
            # Colorbar panel (colorbar and symbol size)
            self.colorbar_panel.reset_defaults()
            
            # Settings dialog settings (if dialog exists)
            if self.settings_dialog:
                # Appearance
                self.settings_dialog.kiosk_mode_check.setChecked(False)
                
                # Planes and Poles - reset to defaults based on coord system
                for plane_name, controls in self.settings_dialog.plane_controls.items():
                    controls['plane_cb'].setChecked(False)
                    controls['pole_cb'].setChecked(False)
                    controls['color_edit'].setText(self.settings_dialog.DEFAULT_COLORS[plane_name])
                
                # Sun and Moon
                self.settings_dialog.show_sun_check.setChecked(True)
                self.settings_dialog.show_moon_check.setChecked(True)
                self.settings_dialog.lunar_exclusion_check.setChecked(False)
                self.settings_dialog.lunar_radius_spin.setValue(30.0)
                self.settings_dialog.lunar_penalty_spin.setValue(3.0)
                self.settings_dialog.lunar_color_edit.setText("#228B22")
                self.settings_dialog.lunar_show_bounds.setChecked(True)
                self.settings_dialog.show_moon_phases_check.setChecked(False)
                
                # Galactic Exclusion
                self.settings_dialog.galactic_enable_check.setChecked(False)
                self.settings_dialog.galactic_offset_spin.setValue(15.0)
                self.settings_dialog.galactic_penalty_spin.setValue(2.0)
                self.settings_dialog.galactic_color_edit.setText("#FF99FF")
                self.settings_dialog.galactic_show_bounds.setChecked(True)
                
                # Opposition Benefit
                self.settings_dialog.opposition_enable_check.setChecked(False)
                self.settings_dialog.opposition_radius_spin.setValue(5.0)
                self.settings_dialog.opposition_benefit_spin.setValue(2.0)
                self.settings_dialog.opposition_color_edit.setText("#90EE90")
                self.settings_dialog.opposition_show_bounds.setChecked(True)
                
                # Discovery Circumstances
                self.settings_dialog.hide_missing_discovery_check.setChecked(False)
                
                # Site Filtering
                self.settings_dialog.site_whitelist_check.setChecked(False)
                self.settings_dialog.site_whitelist_edit.clear()
                self.settings_dialog.site_blacklist_check.setChecked(False)
                self.settings_dialog.site_blacklist_edit.clear()
                
                # Advanced Controls
                self.settings_dialog.lr_increment_spin.setValue(1)
                self.settings_dialog.lr_unit_combo.setCurrentText("hour")
                self.settings_dialog.ud_increment_spin.setValue(1)
                self.settings_dialog.ud_unit_combo.setCurrentText("lunation")
                
                # Trailing
                self.settings_dialog.enable_trailing_check.setChecked(False)
                self.settings_dialog.trail_length_spin.setValue(50)
                self.settings_dialog.trail_weight_spin.setValue(1.0)
                self.settings_dialog.trail_color.setText("#00AA00")
                
                # Horizon/Twilight
                self.settings_dialog.horizon_enable_check.setChecked(False)
                self.settings_dialog.observer_lat_spin.setValue(32.2226)  # Tucson
                self.settings_dialog.observer_lon_spin.setValue(-110.9747)
                self.settings_dialog.location_preset_combo.setCurrentText("Tucson, AZ (CSS)")
                self.settings_dialog.timezone_combo.setCurrentText("UTC-7 (MST)")
                self.settings_dialog.show_horizon_check.setChecked(True)
                self.settings_dialog.horizon_color_edit.setText("#FF6600")
                self.settings_dialog.show_civil_check.setChecked(False)
                self.settings_dialog.civil_color_edit.setText("#FF9933")
                self.settings_dialog.show_nautical_check.setChecked(False)
                self.settings_dialog.nautical_color_edit.setText("#CC66FF")
                self.settings_dialog.show_astro_check.setChecked(True)
                self.settings_dialog.astro_color_edit.setText("#6666FF")
                self.settings_dialog.horizon_style_combo.setCurrentText("solid")
                self.settings_dialog.horizon_weight_spin.setValue(1.5)
                
                # Clear any existing trails
                self.clear_trails()
                
                # Update control states
                self.settings_dialog._initialize_control_states()
            
            # Disable drawer mode
            self.drawer_bar.hide()
            self.top_content.show()
            self.drawer_collapsed = False
            self.drawer_bar.setText("▼ Controls")
            
            self.status_label.setText("Reset to factory defaults")
            self.update_display()
        except Exception as e:
            logger.error(f"Reset error: {e}")
    
    def _get_settings_path(self):
        """Get path to settings file"""
        return os.path.join(os.path.expanduser("~"), ".neolyzer_settings.json")
    
    def save_settings(self):
        """Save current settings to JSON file"""
        import json
        
        try:
            settings = {
                'version': '3.02',
                'magnitude': {
                    'v_min': self.magnitude_panel.mag_min_spin.value(),
                    'v_max': self.magnitude_panel.mag_max_spin.value(),
                    'h_min': self.magnitude_panel.h_min_spin.value(),
                    'h_max': self.magnitude_panel.h_max_spin.value(),
                    'show_all': self.magnitude_panel.show_all_neos_check.isChecked()
                },
                'animation': {
                    'rate': self.controls_panel.rate_spin.value(),
                    'unit': self.controls_panel.rate_unit.currentText(),
                    'fps': self.controls_panel.fps_spin.value()
                },
                'neo_classes': {
                    'moid_enabled': self.neo_classes_panel.moid_enabled_check.isChecked(),
                    'moid_min': self.neo_classes_panel.moid_min_spin.value(),
                    'moid_max': self.neo_classes_panel.moid_max_spin.value(),
                    'show_hollow': self.neo_classes_panel.show_hollow_check.isChecked(),
                    'hide_before_discovery': self.neo_classes_panel.hide_before_discovery_check.isChecked(),
                    'selected': [cls for cls, cb in self.neo_classes_panel.class_checks.items() if cb.isChecked()]
                },
                'projection': {
                    'type': self.proj_panel.proj_combo.currentText(),
                    'coord_system': self.proj_panel.coord_combo.currentText()
                },
                'colorbar': {
                    'color_by': self.colorbar_panel.color_by_combo.currentText(),
                    'colormap': self.colorbar_panel.cmap_combo.currentText(),
                    'min': self.colorbar_panel.cbar_min.value(),
                    'max': self.colorbar_panel.cbar_max.value(),
                    'show_legend': self.colorbar_panel.show_legend_check.isChecked(),
                    'size_by': self.colorbar_panel.size_combo.currentText(),
                    'size_min': self.colorbar_panel.size_min_spin.value(),
                    'size_max': self.colorbar_panel.size_max_spin.value(),
                    'data_min': self.colorbar_panel.data_min_spin.value(),
                    'data_max': self.colorbar_panel.data_max_spin.value(),
                    'size_invert': self.colorbar_panel.size_invert_check.isChecked()
                }
            }
            
            # Add all Settings dialog settings if it exists
            if self.settings_dialog:
                # Appearance
                settings['appearance'] = {
                    'kiosk_mode': self.settings_dialog.kiosk_mode_check.isChecked()
                }
                
                # Planes and Poles - save all plane settings
                settings['planes'] = {}
                for plane_name, controls in self.settings_dialog.plane_controls.items():
                    settings['planes'][plane_name] = {
                        'enabled': controls['plane_cb'].isChecked(),
                        'color': controls['color_edit'].text(),
                        'pole': controls['pole_cb'].isChecked()
                    }
                
                # Sun and Moon - all parameters
                settings['sunmoon'] = {
                    'show_sun': self.settings_dialog.show_sun_check.isChecked(),
                    'show_moon': self.settings_dialog.show_moon_check.isChecked(),
                    'show_phases': self.settings_dialog.show_moon_phases_check.isChecked(),
                    'lunar_exclusion_enabled': self.settings_dialog.lunar_exclusion_check.isChecked(),
                    'lunar_radius': self.settings_dialog.lunar_radius_spin.value(),
                    'lunar_penalty': self.settings_dialog.lunar_penalty_spin.value(),
                    'lunar_color': self.settings_dialog.lunar_color_edit.text(),
                    'lunar_show_bounds': self.settings_dialog.lunar_show_bounds.isChecked()
                }
                
                # Galactic Exclusion - all parameters
                settings['galactic'] = {
                    'enabled': self.settings_dialog.galactic_enable_check.isChecked(),
                    'offset': self.settings_dialog.galactic_offset_spin.value(),
                    'penalty': self.settings_dialog.galactic_penalty_spin.value(),
                    'color': self.settings_dialog.galactic_color_edit.text(),
                    'show_bounds': self.settings_dialog.galactic_show_bounds.isChecked()
                }
                
                # Opposition Benefit - all parameters
                settings['opposition'] = {
                    'enabled': self.settings_dialog.opposition_enable_check.isChecked(),
                    'radius': self.settings_dialog.opposition_radius_spin.value(),
                    'benefit': self.settings_dialog.opposition_benefit_spin.value(),
                    'color': self.settings_dialog.opposition_color_edit.text(),
                    'show_bounds': self.settings_dialog.opposition_show_bounds.isChecked()
                }
                
                # Discovery Circumstances
                settings['discovery'] = {
                    'hide_missing': self.settings_dialog.hide_missing_discovery_check.isChecked()
                }
                
                # Site Filtering
                settings['site_filter'] = {
                    'whitelist_enabled': self.settings_dialog.site_whitelist_check.isChecked(),
                    'whitelist': self.settings_dialog.site_whitelist_edit.text(),
                    'blacklist_enabled': self.settings_dialog.site_blacklist_check.isChecked(),
                    'blacklist': self.settings_dialog.site_blacklist_edit.text()
                }
                
                # Advanced Controls
                settings['advanced'] = {
                    'lr_increment': self.settings_dialog.lr_increment_spin.value(),
                    'lr_unit': self.settings_dialog.lr_unit_combo.currentText()
                }
                
                # Trailing
                settings['trailing'] = {
                    'enabled': self.settings_dialog.enable_trailing_check.isChecked(),
                    'length': self.settings_dialog.trail_length_spin.value(),
                    'weight': self.settings_dialog.trail_weight_spin.value(),
                    'color': self.settings_dialog.trail_color.text()
                }
                
                # Horizon/Twilight
                settings['horizon'] = {
                    'enabled': self.settings_dialog.horizon_enable_check.isChecked(),
                    'observer_lat': self.settings_dialog.observer_lat_spin.value(),
                    'observer_lon': self.settings_dialog.observer_lon_spin.value(),
                    'timezone': self.settings_dialog.timezone_combo.currentText(),
                    'show_horizon': self.settings_dialog.show_horizon_check.isChecked(),
                    'horizon_color': self.settings_dialog.horizon_color_edit.text(),
                    'show_civil': self.settings_dialog.show_civil_check.isChecked(),
                    'civil_color': self.settings_dialog.civil_color_edit.text(),
                    'show_nautical': self.settings_dialog.show_nautical_check.isChecked(),
                    'nautical_color': self.settings_dialog.nautical_color_edit.text(),
                    'show_astro': self.settings_dialog.show_astro_check.isChecked(),
                    'astro_color': self.settings_dialog.astro_color_edit.text(),
                    'line_style': self.settings_dialog.horizon_style_combo.currentText(),
                    'line_weight': self.settings_dialog.horizon_weight_spin.value()
                }
            
            with open(self._get_settings_path(), 'w') as f:
                json.dump(settings, f, indent=2)
            
            self.status_label.setText("Settings saved")
        except Exception as e:
            logger.error(f"Save settings error: {e}")
            self.status_label.setText(f"Error saving settings: {e}")
    
    def restore_settings(self):
        """Restore settings from JSON file"""
        import json
        
        settings_path = self._get_settings_path()
        if not os.path.exists(settings_path):
            self.status_label.setText("No saved settings found")
            return
        
        try:
            with open(settings_path, 'r') as f:
                settings = json.load(f)
            
            # Magnitude
            if 'magnitude' in settings:
                m = settings['magnitude']
                self.magnitude_panel.mag_min_spin.setValue(m.get('v_min', 19.0))
                self.magnitude_panel.mag_max_spin.setValue(m.get('v_max', 22.0))
                self.magnitude_panel.h_min_spin.setValue(m.get('h_min', 9.0))
                self.magnitude_panel.h_max_spin.setValue(m.get('h_max', 22.0))
                self.magnitude_panel.show_all_neos_check.setChecked(m.get('show_all', False))
            
            # Animation
            if 'animation' in settings:
                a = settings['animation']
                self.controls_panel.rate_spin.setValue(a.get('rate', 1.0))
                self.controls_panel.rate_unit.setCurrentText(a.get('unit', 'days/sec'))
                self.controls_panel.fps_spin.setValue(a.get('fps', 10))
            
            # NEO classes
            if 'neo_classes' in settings:
                n = settings['neo_classes']
                self.neo_classes_panel.moid_enabled_check.setChecked(n.get('moid_enabled', False))
                self.neo_classes_panel.moid_min_spin.setValue(n.get('moid_min', 0.0))
                self.neo_classes_panel.moid_max_spin.setValue(n.get('moid_max', 0.05))
                self.neo_classes_panel.show_hollow_check.setChecked(n.get('show_hollow', False))
                self.neo_classes_panel.hide_before_discovery_check.setChecked(n.get('hide_before_discovery', False))
                if 'selected' in n:
                    for cls, cb in self.neo_classes_panel.class_checks.items():
                        cb.setChecked(cls in n['selected'])
            
            # Projection
            if 'projection' in settings:
                p = settings['projection']
                self.proj_panel.proj_combo.setCurrentText(p.get('type', 'Aitoff'))
                self.proj_panel.coord_combo.setCurrentText(p.get('coord_system', 'Equatorial'))
            
            # Colorbar
            if 'colorbar' in settings:
                c = settings['colorbar']
                self.colorbar_panel.color_by_combo.setCurrentText(c.get('color_by', 'V magnitude'))
                self.colorbar_panel.cmap_combo.setCurrentText(c.get('colormap', 'viridis_r'))
                self.colorbar_panel.cbar_min.setValue(c.get('min', 19.0))
                self.colorbar_panel.cbar_max.setValue(c.get('max', 23.0))
                self.colorbar_panel.show_legend_check.setChecked(c.get('show_legend', True))
                self.colorbar_panel.size_combo.setCurrentText(c.get('size_by', 'V magnitude'))
                self.colorbar_panel.size_min_spin.setValue(c.get('size_min', 10))
                self.colorbar_panel.size_max_spin.setValue(c.get('size_max', 150))
                self.colorbar_panel.data_min_spin.setValue(c.get('data_min', 19.0))
                self.colorbar_panel.data_max_spin.setValue(c.get('data_max', 23.0))
                self.colorbar_panel.size_invert_check.setChecked(c.get('size_invert', True))
            
            # Settings dialog settings
            if self.settings_dialog:
                # Appearance
                if 'appearance' in settings:
                    self.settings_dialog.kiosk_mode_check.setChecked(settings['appearance'].get('kiosk_mode', False))
                
                # Planes and Poles
                if 'planes' in settings:
                    for plane_name, plane_settings in settings['planes'].items():
                        if plane_name in self.settings_dialog.plane_controls:
                            controls = self.settings_dialog.plane_controls[plane_name]
                            controls['plane_cb'].setChecked(plane_settings.get('enabled', False))
                            controls['color_edit'].setText(plane_settings.get('color', self.settings_dialog.DEFAULT_COLORS.get(plane_name, '#FFFFFF')))
                            controls['pole_cb'].setChecked(plane_settings.get('pole', False))
                
                # Sun and Moon - all parameters
                if 'sunmoon' in settings:
                    s = settings['sunmoon']
                    self.settings_dialog.show_sun_check.setChecked(s.get('show_sun', True))
                    self.settings_dialog.show_moon_check.setChecked(s.get('show_moon', True))
                    self.settings_dialog.show_moon_phases_check.setChecked(s.get('show_phases', False))
                    self.settings_dialog.lunar_exclusion_check.setChecked(s.get('lunar_exclusion_enabled', False))
                    self.settings_dialog.lunar_radius_spin.setValue(s.get('lunar_radius', 30.0))
                    self.settings_dialog.lunar_penalty_spin.setValue(s.get('lunar_penalty', 3.0))
                    self.settings_dialog.lunar_color_edit.setText(s.get('lunar_color', '#228B22'))
                    self.settings_dialog.lunar_show_bounds.setChecked(s.get('lunar_show_bounds', True))
                
                # Galactic Exclusion - all parameters
                if 'galactic' in settings:
                    g = settings['galactic']
                    self.settings_dialog.galactic_enable_check.setChecked(g.get('enabled', False))
                    self.settings_dialog.galactic_offset_spin.setValue(g.get('offset', 15.0))
                    self.settings_dialog.galactic_penalty_spin.setValue(g.get('penalty', 2.0))
                    self.settings_dialog.galactic_color_edit.setText(g.get('color', '#FF99FF'))
                    self.settings_dialog.galactic_show_bounds.setChecked(g.get('show_bounds', True))
                
                # Opposition Benefit - all parameters
                if 'opposition' in settings:
                    o = settings['opposition']
                    self.settings_dialog.opposition_enable_check.setChecked(o.get('enabled', False))
                    self.settings_dialog.opposition_radius_spin.setValue(o.get('radius', 5.0))
                    self.settings_dialog.opposition_benefit_spin.setValue(o.get('benefit', 2.0))
                    self.settings_dialog.opposition_color_edit.setText(o.get('color', '#90EE90'))
                    self.settings_dialog.opposition_show_bounds.setChecked(o.get('show_bounds', True))
                
                # Discovery Circumstances
                if 'discovery' in settings:
                    d = settings['discovery']
                    self.settings_dialog.hide_missing_discovery_check.setChecked(d.get('hide_missing', False))
                
                # Site Filtering
                if 'site_filter' in settings:
                    sf = settings['site_filter']
                    self.settings_dialog.site_whitelist_check.setChecked(sf.get('whitelist_enabled', False))
                    self.settings_dialog.site_whitelist_edit.setText(sf.get('whitelist', ''))
                    self.settings_dialog.site_blacklist_check.setChecked(sf.get('blacklist_enabled', False))
                    self.settings_dialog.site_blacklist_edit.setText(sf.get('blacklist', ''))
                
                # Advanced Controls
                if 'advanced' in settings:
                    a = settings['advanced']
                    self.settings_dialog.lr_increment_spin.setValue(a.get('lr_increment', 1))
                    self.settings_dialog.lr_unit_combo.setCurrentText(a.get('lr_unit', 'hour'))
                
                # Trailing
                if 'trailing' in settings:
                    t = settings['trailing']
                    self.settings_dialog.enable_trailing_check.setChecked(t.get('enabled', False))
                    self.settings_dialog.trail_length_spin.setValue(t.get('length', 50))
                    self.settings_dialog.trail_weight_spin.setValue(t.get('weight', 1.0))
                    self.settings_dialog.trail_color.setText(t.get('color', '#00AA00'))
                
                # Horizon/Twilight
                if 'horizon' in settings:
                    h = settings['horizon']
                    self.settings_dialog.horizon_enable_check.setChecked(h.get('enabled', False))
                    self.settings_dialog.observer_lat_spin.setValue(h.get('observer_lat', 32.2226))
                    self.settings_dialog.observer_lon_spin.setValue(h.get('observer_lon', -110.9747))
                    self.settings_dialog.timezone_combo.setCurrentText(h.get('timezone', 'UTC-7 (MST)'))
                    self.settings_dialog.show_horizon_check.setChecked(h.get('show_horizon', True))
                    self.settings_dialog.horizon_color_edit.setText(h.get('horizon_color', '#FF6600'))
                    self.settings_dialog.show_civil_check.setChecked(h.get('show_civil', False))
                    self.settings_dialog.civil_color_edit.setText(h.get('civil_color', '#FF9933'))
                    self.settings_dialog.show_nautical_check.setChecked(h.get('show_nautical', False))
                    self.settings_dialog.nautical_color_edit.setText(h.get('nautical_color', '#CC66FF'))
                    self.settings_dialog.show_astro_check.setChecked(h.get('show_astro', True))
                    self.settings_dialog.astro_color_edit.setText(h.get('astro_color', '#6666FF'))
                    self.settings_dialog.horizon_style_combo.setCurrentText(h.get('line_style', 'solid'))
                    self.settings_dialog.horizon_weight_spin.setValue(h.get('line_weight', 1.5))
                
                # Update control states
                self.settings_dialog._initialize_control_states()
            
            self.status_label.setText("Settings restored")
            self.update_display()
        except Exception as e:
            logger.error(f"Restore settings error: {e}")
            self.status_label.setText(f"Error restoring settings: {e}")
    
    def keyPressEvent(self, event):
        """Handle keyboard events"""
        from PyQt6.QtCore import Qt
        
        if event.key() == Qt.Key.Key_Escape:
            # Clear selection mode
            if hasattr(self, 'canvas') and self.canvas.selection_mode is not None:
                self.canvas.clear_selection()
                self.status_label.setText("Selection cancelled.")
                return
        
        # Pass to parent class
        super().keyPressEvent(event)


def main():
    # Parse command line
    parser = argparse.ArgumentParser(description='NEOlyzer - NEO Sky Position Visualizer')
    parser.add_argument('--quiet', action='store_true', help='Quiet mode (warnings only)')
    parser.add_argument('--verbose', action='store_true', help='Verbose mode (info)')
    parser.add_argument('--debug', action='store_true', help='Debug mode (debug)')
    parser.add_argument('--no-cache', action='store_true', help='Disable cache (always compute)')
    parser.add_argument('--fps', action='store_true', help='Show FPS during animation')
    args = parser.parse_args()
    
    # Set logging level
    if args.debug:
        level = logging.DEBUG
    elif args.verbose:
        level = logging.INFO
    elif args.quiet:
        level = logging.WARNING
    else:
        level = logging.WARNING  # Default: quiet
    
    setup_logging(level)
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Show PyQt version info
    logger.info(f"Using PyQt{PYQT_VERSION}")
    
    window = NEOVisualizer(use_cache=not args.no_cache, show_fps=args.fps)
    window.show()
    
    # Clear focus from datetime widget (prevents year field being highlighted on startup)
    window.time_panel.datetime_edit.clearFocus()
    # Set focus to canvas for immediate keyboard interaction
    window.canvas.setFocus()
    
    try:
        # PyQt5 uses exec_(), PyQt6 uses exec()
        if PYQT_VERSION == 5:
            sys.exit(app.exec_())
        else:
            sys.exit(app.exec())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        sys.exit(0)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0)
