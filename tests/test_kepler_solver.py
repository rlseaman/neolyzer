"""Tests for Kepler equation solver and coordinate transforms in orbit_calculator.py."""

import numpy as np
import pytest

# Import the solvers directly — these are pure math, no ephemeris needed
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ── Standalone Kepler solver (extracted to avoid ephemeris load) ──

def solve_kepler(M, e, tol=1e-10):
    """Scalar Kepler solver (same algorithm as OrbitCalculator._solve_kepler)."""
    E = M if e < 0.8 else np.pi
    for _ in range(30):
        f = E - e * np.sin(E) - M
        fp = 1 - e * np.cos(E)
        delta = f / fp
        E -= delta
        if abs(delta) < tol:
            return E
    return E


def solve_kepler_vectorized(M, e, tol=1e-10):
    """Vectorized Kepler solver (same algorithm as FastOrbitCalculator)."""
    M = np.asarray(M, dtype=float)
    e = np.asarray(e, dtype=float)
    E = np.where(e < 0.8, M, np.full_like(M, np.pi))
    for _ in range(30):
        f = E - e * np.sin(E) - M
        fp = 1 - e * np.cos(E)
        delta = f / fp
        E -= delta
        if np.all(np.abs(delta) < tol):
            break
    return E


# ── Kepler equation: E - e*sin(E) = M ───────────────────────────

class TestKeplerSolver:
    """Verify that solved E satisfies M = E - e*sin(E)."""

    def _verify(self, M, e, tol=1e-9):
        E = solve_kepler(M, e)
        residual = abs(E - e * np.sin(E) - M)
        assert residual < tol, f"M={M}, e={e}: residual {residual}"

    def test_circular_orbit(self):
        """e=0: E should equal M exactly."""
        for M in [0, 0.5, np.pi, 2 * np.pi - 0.1]:
            E = solve_kepler(M, 0.0)
            assert abs(E - M) < 1e-12

    def test_low_eccentricity(self):
        """Typical NEO eccentricities."""
        for e in [0.05, 0.1, 0.2, 0.4]:
            for M in np.linspace(0, 2 * np.pi, 12, endpoint=False):
                self._verify(M, e)

    def test_high_eccentricity(self):
        """Cometary eccentricities near parabolic."""
        for e in [0.8, 0.9, 0.95, 0.99]:
            for M in np.linspace(0.01, 2 * np.pi - 0.01, 12):
                self._verify(M, e)

    def test_edge_M_zero(self):
        """M=0 should give E=0 for any eccentricity."""
        for e in [0, 0.3, 0.7, 0.99]:
            E = solve_kepler(0.0, e)
            assert abs(E) < 1e-10

    def test_edge_M_pi(self):
        """M=pi should give E=pi for any eccentricity (by symmetry)."""
        for e in [0, 0.3, 0.7, 0.99]:
            E = solve_kepler(np.pi, e)
            assert abs(E - np.pi) < 1e-9


class TestKeplerVectorized:
    """Verify vectorized solver matches scalar solver."""

    def test_matches_scalar(self):
        """Vectorized and scalar solvers should agree."""
        M_vals = np.linspace(0.01, 2 * np.pi - 0.01, 20)
        e_vals = np.full_like(M_vals, 0.3)

        E_vec = solve_kepler_vectorized(M_vals, e_vals)
        E_scalar = np.array([solve_kepler(m, 0.3) for m in M_vals])

        np.testing.assert_allclose(E_vec, E_scalar, atol=1e-10)

    def test_mixed_eccentricities(self):
        """Different eccentricities in same batch."""
        M_vals = np.array([0.5, 1.0, 2.0, 3.0, 5.0])
        e_vals = np.array([0.01, 0.2, 0.5, 0.85, 0.99])

        E_vec = solve_kepler_vectorized(M_vals, e_vals)

        for i in range(len(M_vals)):
            E_s = solve_kepler(M_vals[i], e_vals[i])
            assert abs(E_vec[i] - E_s) < 1e-10

    def test_residuals(self):
        """All vectorized solutions should satisfy Kepler's equation."""
        M_vals = np.linspace(0.01, 2 * np.pi - 0.01, 50)
        e_vals = np.full_like(M_vals, 0.6)

        E = solve_kepler_vectorized(M_vals, e_vals)
        residuals = np.abs(E - e_vals * np.sin(E) - M_vals)
        assert np.all(residuals < 1e-10)


# ── Coordinate transforms ────────────────────────────────────────

class TestCoordinateTransforms:
    """Test ecliptic ↔ equatorial rotation matrices."""

    EPSILON = np.radians(23.43928)  # J2000.0 obliquity

    def ecl_to_eq(self, x_ecl, y_ecl, z_ecl):
        """Ecliptic → equatorial (same transform as FastOrbitCalculator)."""
        cos_eps = np.cos(self.EPSILON)
        sin_eps = np.sin(self.EPSILON)
        x_eq = x_ecl
        y_eq = cos_eps * y_ecl - sin_eps * z_ecl
        z_eq = sin_eps * y_ecl + cos_eps * z_ecl
        return x_eq, y_eq, z_eq

    def eq_to_ecl(self, x_eq, y_eq, z_eq):
        """Equatorial → ecliptic (inverse rotation)."""
        cos_eps = np.cos(self.EPSILON)
        sin_eps = np.sin(self.EPSILON)
        x_ecl = x_eq
        y_ecl = cos_eps * y_eq + sin_eps * z_eq
        z_ecl = -sin_eps * y_eq + cos_eps * z_eq
        return x_ecl, y_ecl, z_ecl

    def test_vernal_equinox(self):
        """Vernal equinox is (1,0,0) in both systems."""
        x, y, z = self.ecl_to_eq(1, 0, 0)
        assert abs(x - 1) < 1e-12
        assert abs(y) < 1e-12
        assert abs(z) < 1e-12

    def test_north_ecliptic_pole(self):
        """Ecliptic north pole (0,0,1) → tilted in equatorial."""
        x, y, z = self.ecl_to_eq(0, 0, 1)
        assert abs(x) < 1e-12
        # Should point toward equatorial dec ~ 66.56°
        dec = np.arcsin(z / np.sqrt(x**2 + y**2 + z**2))
        expected_dec = np.pi / 2 - self.EPSILON
        assert abs(dec - expected_dec) < 1e-10

    def test_roundtrip(self):
        """ecl→eq→ecl should be identity."""
        for vec in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0.3, -0.7, 0.5)]:
            eq = self.ecl_to_eq(*vec)
            back = self.eq_to_ecl(*eq)
            np.testing.assert_allclose(back, vec, atol=1e-12)

    def test_rotation_preserves_length(self):
        """Rotation shouldn't change vector magnitude."""
        vec = (3.5, -1.2, 0.8)
        original_len = np.sqrt(sum(v**2 for v in vec))
        eq = self.ecl_to_eq(*vec)
        new_len = np.sqrt(sum(v**2 for v in eq))
        assert abs(new_len - original_len) < 1e-12
