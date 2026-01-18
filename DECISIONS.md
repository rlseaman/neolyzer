# NEOlyzer Design Decisions

## Mission Statement

NEOlyzer is the result of my experiences as a software engineer and data scientist in the astronomical community, and for the past decade in the planetary defense community. There are two main goals to the NEOlyzer project:

1. Create a tool that provides detailed, deep access to the near-Earth object catalog for visualization and query.

2. Explore the utility of AI coding practices for building robust facilities for scientific and explanatory purposes.

— Rob Seaman, Catalina Sky Survey, Lunar and Planetary Laboratory, University of Arizona

---

## On Capturing Design Thinking

This document exists to preserve the rationale behind NEOlyzer's design choices. The code shows *what*; this document captures *why*.

NEOlyzer was developed through an iterative collaboration between a human designer (Rob) and AI coding assistants (Claude). Hundreds of prompts shaped the architecture, features, and implementation details. The AI has no persistent memory across sessions—each conversation starts fresh. This creates a challenge: how to maintain continuity of design intent across a long-term development effort?

The answer is artifacts that persist in the repository:

- **CLAUDE.md**: Instructions for how to work on the project
- **ASTRONOMY.txt**: Documentation of astronomical dependencies and integration
- **DECISIONS.md**: This file—rationale for non-obvious choices
- **Commit messages**: A searchable log of what changed and why
- **Code comments**: Intent and constraints, not just description

The community of astronomy domain experts stretching back centuries, and software engineers over decades, built the robust foundation (Skyfield, JPL ephemerides, Python ecosystem) that supports NEOlyzer. This tool builds on that foundation to serve the planetary defense mission.

---

## Design Decisions Log

### 2025-01-18: Geocentric Positions

**Decision**: All NEO positions are geocentric (Earth-centered), not topocentric (observer-centered).

**Rationale**: Simplifies computation and is appropriate for catalog-level visualization. Topocentric corrections are small for distant objects and would add complexity without proportional benefit for the tool's purpose.

### 2025-01-18: Skyfield as Primary Astronomical Library

**Decision**: Use Skyfield (>=1.45) for astronomical computations rather than Astropy or other alternatives.

**Rationale**: Skyfield provides a modern Python API, uses JPL ephemerides directly for high accuracy, has a pure Python core for cross-platform compatibility, and is actively maintained. Astropy is used only for galactic coordinate transformations where Skyfield's coverage is limited. See ASTRONOMY.txt for full discussion.

### 2025-01-18: Planet Display Feature

**Decision**: Add optional planet display using Unicode symbols (☿♀♂♃♄♅♆♇) rather than graphical markers or labeled points.

**Rationale**: Symbols are compact, universally recognized, require no additional assets, and are embedded directly in source code (no runtime download). Pluto included as a planet for this feature. Default color is dark slate blue (#2D4A6B) to distinguish from NEO markers without being distracting.

### 2025-01-18: Symbol Size Inversion Defaults

**Decision**: Only V magnitude and H magnitude have "Invert" enabled by default for symbol sizing. Distance, MOID, Period, and Eccentricity do not.

**Rationale**: For magnitudes, lower values mean brighter objects, so inversion maps brighter→larger symbols intuitively. For other quantities, the natural mapping (larger value→larger symbol) is more intuitive without inversion.

### 2025-01-18: No Network Required After Setup

**Decision**: Core visualization functionality requires no network access after initial setup.

**Rationale**: Supports use in environments with limited connectivity (field sites, aircraft, isolated networks). All ephemeris data, NEO catalogs, and MOID data are cached locally. Only optional features (web links to external databases) require network access.

---

## Future Decisions

*(Add entries as design choices are made)*
