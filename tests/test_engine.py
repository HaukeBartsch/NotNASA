"""Tests for sim.engine (explicit FEM/FVM transport core).

Run with the project venv:
    .venv/bin/python -m pytest tests/test_engine.py -q
or without pytest:
    .venv/bin/python tests/test_engine.py
"""
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sim import config as C
import sim.wind as wind
from sim.mesh import Mesh
from sim import sources as srcmod
from sim.engine import Engine


class NullSources:
    """Zero-emission source stub for conservation tests."""

    def __init__(self, n):
        self.n = n

    def emission(self, day, hour_utc):
        return np.zeros(self.n)


# Shared base (uniform) mesh -- the engine-math invariants below are mesh-
# independent and calibrated on the uniform mesh; the land-refined mesh is
# covered by test_refined_land.py.
MESH = Mesh(250.0, refine_land=False)
N, K = MESH.N, C.N_VLEVELS
M = MESH.M


def _nearest(lat, lon):
    return int(np.argmin((MESH.lat - lat) ** 2 + (MESH.lon - lon) ** 2))


def _patch_wind(u=0.0, w=0.0):
    """Patch the wind module to a constant zonal wind / vertical velocity."""
    orig = (wind.wind_uv, wind.w_z)

    def uv(phi, lam, h, d):
        phi = np.asarray(phi, dtype=float)
        return np.full_like(phi, u), np.zeros_like(phi)

    wind.wind_uv = uv
    wind.w_z = lambda phi, h, d: np.full_like(np.asarray(phi, dtype=float), w)
    return orig


def _restore(orig):
    wind.wind_uv, wind.w_z = orig


def _no_sink():
    old = C.TAU_S
    C.TAU_S = math.inf
    return old


def _restore_tau(old):
    C.TAU_S = old


# ---------------------------------------------------------------------------
# 1. Conservation invariants
# ---------------------------------------------------------------------------

def test_uniform_field_total_mass_conserved():
    """No sources, no sink, w=0 => advection+diffusion conserve mass exactly."""
    orig = _patch_wind(u=12.0, w=0.0)
    old_tau = _no_sink()
    try:
        eng = Engine(MESH, NullSources(N))
        c = np.ones((N, K))
        m0 = eng.mass(c)
        for _ in range(3):
            eng.step(c, 200.0, 14.0)
        m1 = eng.mass(c)
        assert np.all(np.isfinite(c)), "NaN/inf appeared"
        rel = abs(m1 - m0) / m0
        assert rel <= 1e-9, f"total mass not conserved: {m0} -> {m1} (rel {rel:.3e})"
    finally:
        _restore(orig)
        _restore_tau(old_tau)


def test_mass_balance_with_sources():
    """Per step: m1 = (m0 + DT*DZ[0]*sum(M_i S_i) + transport) * (1 - DT/TAU).

    The source adds DT*S ppm to level 0, whose cell volume is M_i*DZ[0].
    Transport is (near-)conservative; the residual is the small global vertical
    flux of the idealized w field (zero zonal mean, so only mesh-quadrature
    level). Tolerance 1% covers it comfortably; a 1/M unit bug would be ~60%.
    """
    eng = Engine(MESH, srcmod.Sources(MESH))
    c = np.zeros((N, K))
    for step in range(4):
        day, hour = 200.0, 14.0 + 0.5 * step
        em = eng.sources.emission(day, hour)
        m0 = eng.mass(c)
        expected = (m0 + C.DT_S * eng.DZ[0] * float((em * M).sum())) \
                   * (1.0 - C.DT_S / C.TAU_S)
        eng.step(c, day, hour)
        m1 = eng.mass(c)
        assert np.isfinite(m1), "mass diverged"
        assert c.max() < C.C_CAP, f"field hit the hard cap: {c.max()}"
        tol = 0.01 * max(1.0, abs(expected))
        assert abs(m1 - expected) <= tol, (
            f"mass balance broken at step {step}: got {m1}, expected {expected}")


# ---------------------------------------------------------------------------
# 2. Advection
# ---------------------------------------------------------------------------

def test_advection_drifts_downwind():
    """Uniform 10 m/s easterly (eastward) wind: a smooth blob drifts at wind speed.

    (A single-node delta is not a good tracer for vertex-upwind: its mass
    leaps to downwind edge vertices, biasing the centroid. A smooth blob
    advects at the wind speed.)
    """
    orig = _patch_wind(u=10.0, w=0.0)
    old_tau = _no_sink()
    try:
        eng = Engine(MESH, NullSources(N))
        # Gaussian blob, sigma ~ 3 cells (~7 deg), centered near (45N, 0E)
        center = np.array([math.cos(math.radians(45.0)), 0.0, math.sin(math.radians(45.0))])
        d = np.arccos(np.clip(MESH.V @ center, -1.0, 1.0))
        c = np.zeros((N, K))
        c[:, 8] = np.exp(-0.5 * (d / math.radians(7.0)) ** 2)
        m0 = eng.mass(c)

        def centroid_y():
            col = (c * M[:, None] * eng.DZ[None, :]).sum(axis=1)  # column mass
            return float((col * MESH.V[:, 1]).sum() / col.sum())

        y0 = centroid_y()
        for _ in range(20):
            eng.step(c, 200.0, 14.0)
        y1 = centroid_y()
        # 10 m/s * 36000 s = 360 km east. At (45N, 0E) east is pure +y, so the
        # unit-sphere y-shift is just dist/R (no cos(lat) factor).
        dy_expected = (10.0 * 20 * C.DT_S) / (C.R_KM * 1000.0)
        assert np.all(np.isfinite(c)), "NaN/inf appeared"
        assert c.min() >= 0.0, "advection created negative values"
        assert abs(eng.mass(c) - m0) <= 1e-9 * m0, "advection not mass-conserving"
        dy = y1 - y0
        assert 0.7 * dy_expected < dy < 1.3 * dy_expected, (
            f"centroid drifted {dy} in unit-sphere y, expected ~{dy_expected} "
            f"({dy/dy_expected:.2f}x wind speed)")
        assert c.max() > 0.5, f"blob amplitude vanished: {c.max()}"
    finally:
        _restore(orig)
        _restore_tau(old_tau)


# ---------------------------------------------------------------------------
# 3. Vertical transport
# ---------------------------------------------------------------------------

def test_vertical_advection_upward():
    """Uniform 1 m/s upwelling: a ground-level bump rises; no negative values.

    (With uniform w the column legitimately gains mass at the closed lid,
    since w is damped to 0 at the surface — so no strict conservation claim.)
    """
    orig = _patch_wind(u=0.0, w=1.0)
    old_tau = _no_sink()
    try:
        eng = Engine(MESH, NullSources(N))
        i = _nearest(10.0, 20.0)
        c = np.zeros((N, K))
        c[i, 0] = 1.0
        m0 = eng.mass(c)
        for _ in range(60):
            eng.step(c, 200.0, 14.0)
        assert np.all(np.isfinite(c)), "NaN/inf appeared"
        lvl = int(np.argmax(c[i]))
        assert lvl >= 3, f"bump did not rise: argmax level {lvl}"
        assert c.min() >= 0.0, "vertical transport created negative values"
        assert eng.mass(c) >= m0 * (1.0 - 1e-9), "column lost mass with net upwelling"
        assert c.max() <= C.C_CAP
    finally:
        _restore(orig)
        _restore_tau(old_tau)


# ---------------------------------------------------------------------------
# 4. Diffusion
# ---------------------------------------------------------------------------

def test_diffusion_smooths_and_preserves_mass():
    """Positive-definite P1 diffusion smooths a bump, stays positive, conserves mass."""
    orig = _patch_wind(u=0.0, w=0.0)
    old_tau = _no_sink()
    old_KE = C.K_H
    C.K_H = 1.0e6  # 10^4 stronger so the effect is visible at this resolution
    try:
        eng = Engine(MESH, NullSources(N))
        i = _nearest(10.0, 20.0)
        c = np.zeros((N, K))
        c[i, 8] = 1.0
        m0 = eng.mass(c)
        for _ in range(100):
            eng.step(c, 200.0, 14.0)
        assert np.all(np.isfinite(c)), "NaN/inf appeared"
        assert c.min() >= 0.0, "diffusion created negative values"
        assert c.max() < 1.0, f"diffusion should smooth the bump, max={c.max()}"
        assert abs(eng.mass(c) - m0) <= 1e-9 * m0, "diffusion not conservative"
    finally:
        _restore(orig)
        _restore_tau(old_tau)
        C.K_H = old_KE


# ---------------------------------------------------------------------------
# 5. Sources / steady behaviour
# ---------------------------------------------------------------------------

def test_source_localization():
    """Emissions act at ground level, near a source, at a sane magnitude."""
    eng = Engine(MESH, srcmod.Sources(MESH))
    c = np.zeros((N, K))
    eng.step(c, 200.0, 14.0)
    assert c.max() > 0.0, "sources produced nothing"
    i, k = np.unravel_index(int(np.argmax(c)), c.shape)
    assert k == 0, f"source should act at ground level, got level {k}"
    centers = [(r[1], r[2]) for r in srcmod.FIRE_REGIONS] + \
              [(p[1], p[2]) for p in srcmod.POP_CENTERS]
    best = min(
        math.degrees(math.acos(np.clip(
            MESH.V[i] @ np.array([np.cos(np.radians(la)) * np.cos(np.radians(lo)),
                                 np.cos(np.radians(la)) * np.sin(np.radians(lo)),
                                 np.sin(np.radians(la))]), -1.0, 1.0)))
        for (la, lo) in centers)
    assert best <= 3.5, f"max not near any source (closest {best:.2f} deg)"
    assert c.max() < 1.0e4, f"first-step source magnitude implausible: {c.max()}"


def test_sustained_run_stable():
    """500 real steps (10 days) from zero: bounded, finite, positive, nontrivial."""
    eng = Engine(MESH, srcmod.Sources(MESH))
    c = np.zeros((N, K))
    t0 = time.time()
    for s in range(500):
        eng.step(c, day=200.0 + (s * 0.5) / 24.0 % 365.0,
                 hour_utc=(s * 0.5) % 24.0)
    dt = time.time() - t0
    assert np.all(np.isfinite(c)), "NaN/inf appeared in sustained run"
    assert c.min() >= 0.0
    assert c.max() <= C.C_CAP
    assert c.max() > 1.0, "field never grew from sources"
    # far field (away from sources) must stay at a sane scale:
    # 90th percentile of the per-node column maxima
    assert float(np.sort(c.max(axis=1))[int(0.9 * N)]) < 100.0, "far field blew up"
    print(f"  [info] 500 steps in {dt:.1f}s ({500/dt:.1f} steps/s)")


# ---------------------------------------------------------------------------
# 6. Numerical guards
# ---------------------------------------------------------------------------

def test_cfl_numbers_below_one():
    eng = Engine(MESH, NullSources(N))
    for name, v in eng.cfl.items():
        assert 0.0 < v < 1.0, f"CFL[{name}] = {v}"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
