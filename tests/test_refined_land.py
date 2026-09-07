"""Tests for the land-aware mesh refinement (one step over land, none over ocean).

The refinement densifies the point set over land and re-triangulates to the
spherical Delaunay, so the deliverable is a CONFORMING simplicial complex the
existing P1 engine can consume unmodified. These tests pin that: the mesh stays
conforming, the fixed timestep stays inside the advection CFL limit, and mass is
still conserved with no new extrema.

Run with the project venv:
    .venv/bin/python -m pytest tests/test_refined_land.py -q
or without pytest:
    .venv/bin/python tests/test_refined_land.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sim import config as C            # noqa: E402
import sim.wind as wind                # noqa: E402
from sim.mesh import Mesh              # noqa: E402
from sim.engine import Engine          # noqa: E402


class NullSources:
    def __init__(self, n):
        self.n = n
    def emission(self, day, hour):
        return np.zeros(self.n)


def _refined_mesh(res=1000.0):
    """Build a land-refined mesh (one step over land) via the explicit param."""
    return Mesh(res, refine_land=True)


def test_refined_mesh_is_conforming():
    """Every shared edge has exactly two incident triangles (no hanging nodes)."""
    mesh = _refined_mesh()
    assert mesh.refinement is not None, "refinement did not run"
    # it actually refined something (more nodes/tris than the base)
    assert mesh.refinement["nodes_after"] > mesh.refinement["nodes_before"]
    lens = np.fromiter((len(v) for v in mesh.edge_map.values()), dtype=int)
    assert np.all(lens == 2), f"{int((lens != 2).sum())} non-conforming edges"


def test_refined_mesh_cfl_stays_below_one():
    """Refining halves h_min, but the fixed 900 s step must remain advection-stable."""
    mesh = _refined_mesh()
    eng = Engine(mesh, NullSources(mesh.N))
    assert eng.cfl["advection"] < 1.0, f"advection CFL {eng.cfl['advection']:.3f} >= 1"
    assert eng.cfl["diffusion"] < 1.0, f"diffusion CFL {eng.cfl['diffusion']:.3f} >= 1"


def test_refined_mesh_conserves_mass_and_stays_positive():
    """No sources/sink, constant wind: mass conserved to roundoff, no new minima."""
    orig = (wind.wind_uv, wind.w_z)
    wind.wind_uv = lambda phi, lam, h, d: (
        np.full_like(np.asarray(phi, float), 12.0), np.zeros_like(np.asarray(phi, float)))
    wind.w_z = lambda phi, h, d: np.zeros_like(np.asarray(phi, float))
    old_tau = C.TAU_S
    C.TAU_S = math.inf
    try:
        mesh = _refined_mesh()
        eng = Engine(mesh, NullSources(mesh.N))
        c = np.ones((mesh.N, C.N_VLEVELS))
        m0 = eng.mass(c)
        for _ in range(6):
            eng.step(c, 200.0, 14.0)
        assert np.all(np.isfinite(c)), "NaN/inf appeared"
        assert c.min() >= 0.0, f"advection created negative values (min {c.min():.3e})"
        assert abs(eng.mass(c) - m0) <= 1e-9 * m0, "total mass not conserved"
    finally:
        wind.wind_uv, wind.w_z = orig
        C.TAU_S = old_tau


if __name__ == "__main__":
    test_refined_mesh_is_conforming()
    test_refined_mesh_cfl_stays_below_one()
    test_refined_mesh_conserves_mass_and_stays_positive()
    print("test_refined_land: all passed")
