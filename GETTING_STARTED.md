# Getting Started with NEOlyzer

## What is NEOlyzer?

NEOlyzer visualizes the orbits of Near-Earth Objects (NEOs) -- asteroids and comets that approach within 1.3 AU of the Sun. It displays 40,000+ objects on an interactive sky map, letting you animate their motion, filter by orbital properties, and explore the history and growth of the NEO catalog.

NEOlyzer is developed at Catalina Sky Survey, Lunar and Planetary Laboratory, University of Arizona.

---

## System Requirements

- **Python 3.10+** (3.12+ recommended)
- **Operating system:** macOS, Linux, or Windows (via WSL)
- **Disk space:** ~500 MB minimum (ephemeris + database + cache)
- **RAM:** 4 GB minimum, 8 GB recommended
- **Display:** 1280x800 minimum resolution

To check your Python version:

```bash
python3 --version
```

If you need to install or upgrade Python, see [PLATFORM_NOTES.txt](PLATFORM_NOTES.txt) for platform-specific instructions.

---

## Installation

### 1. Get the code

```bash
git clone https://github.com/rlseaman/neolyzer.git
cd neolyzer
```

Or download from the [Releases](https://github.com/rlseaman/neolyzer/releases) page.

### 2. Run the installer

```bash
./install.sh
```

This creates a Python virtual environment, installs dependencies, and launches the interactive setup wizard. The setup will:

- Download the NEO catalog from the Minor Planet Center (~8 MB)
- Download a JPL planetary ephemeris (~115 MB for the default DE440)
- Fetch Earth MOID values from JPL (~2-3 minutes)
- Build the position cache (~5-30 minutes depending on options)

**If installation fails partway through**, you can safely re-run `./install.sh`. If you see a "lock file" error, the previous install may have crashed -- run `./install.sh --force-unlock` to clear it and try again.

### 3. Launch

```bash
./run_neolyzer.sh
```

---

## First Run: What You're Looking At

When NEOlyzer starts, you'll see a sky map showing every known NEO at the current date and time.

### The sky map

- Each **dot** is a Near-Earth Object, colored by visual magnitude (brightness)
- The **Sun** (yellow dot) and **Moon** (with phase) are marked
- **Brighter** (lower magnitude) objects appear larger
- The default projection is **Rectangular** in **Equatorial** coordinates (RA/Dec)

### Key controls (top panels)

- **Date/Time:** Shows the current display date. Click the calendar or type a date to jump to any time between 1550 and 2650.
- **Animation:** Press Play to watch NEOs move. Adjust the rate spinner to control speed (hours, days, or months per second).
- **NEO Classes:** Toggle Atira, Aten, Apollo, and Amor object classes on/off.
- **Magnitude:** Adjust the brightness range to show fainter or brighter objects.

### Toolbar (top)

- **Projection:** Switch between Rectangular, Hammer, Aitoff, and Mollweide map projections.
- **Coordinates:** Switch between Equatorial, Ecliptic, Galactic, and Opposition coordinate systems.
- **Search:** Find a specific object by designation (e.g., "Apophis" or "99942").

### Status bar (bottom)

Shows the current date/time, number of visible objects, Julian Date, and Moon phase (CLN = Catalina Lunation Number).

---

## Common Tasks

### Animate NEO motion

1. Click **Play** in the Animation panel
2. Adjust the **rate** spinner (e.g., "24" hours/sec shows one day per second)
3. Use the **rate unit** dropdown to switch between hours, days, months per second
4. Click **Play** again to pause
5. Use negative rates for reverse playback

### Jump to a specific date

- Type a date in the date field (YYYY-MM-DD format) and press Enter
- Or use Shift+[ and Shift+] to step backward/forward by a configurable increment

### Filter by NEO class

- The **NEO Classes** panel has buttons for each orbital class (Atira, Aten, Apollo, Amor)
- Click to toggle classes on/off
- Near/Far subdivisions available for Apollo and Amor

### Click on an object

- Click any dot on the sky map to identify it
- A red ring highlights the selected object
- The status bar shows the object's designation, magnitude, and distance
- Right-click for more options including JPL SBDB lookup

### Change the look

- **Settings** button opens detailed configuration: colors, symbol sizes, overlays, horizon/twilight boundaries, constellation lines, and more

---

## Updating the Catalog

The NEO catalog grows daily. To update:

```bash
# Quick daily update (~5 minutes)
./venv/bin/python scripts/update_catalog.py --fetch-moid --mark-stale --clear-cache

# Full update with cache rebuild (~35 minutes)
./venv/bin/python scripts/update_catalog.py --fetch-moid --mark-stale --rebuild-cache
```

---

## Getting Help

- **In-app:** Click the **Help** button for feature documentation
- **Troubleshooting:** See the [README.md](README.md) troubleshooting table
- **Platform issues:** See [PLATFORM_NOTES.txt](PLATFORM_NOTES.txt)
- **Contact:** Rob Seaman, [rseaman@arizona.edu](mailto:rseaman@arizona.edu)

---

## Next Steps

- Explore the **Settings** dialog for advanced features (horizon overlays, trailing, discovery filters)
- Try **Scripted Playback** to record and replay animated sequences
- Load **alternate catalogs** to compare catalog versions over time (see [README.md](README.md))
- Read [ASTRONOMY.txt](ASTRONOMY.txt) for background on orbital mechanics and NEO science
