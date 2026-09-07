"""Central configuration for the CO2 weather simulation.

Everything the physics needs that is a tunable knob lives here. The single
"scale factor" is the resolution knob `resolution_km` (land cell spacing, 100-1000).
"""
from __future__ import annotations

import math

# --- Planet / geometry -----------------------------------------------------
R_KM = 6371.0            # Earth radius (km)
TWO_PI = 2.0 * math.pi
DEG = math.pi / 180.0

# --- Resolution knob -------------------------------------------------------
# `resolution_km` = target LAND cell spacing (km). Ocean spacing is coarser.
resolution_km: float = 250.0
ocean_spacing_km: float = min(2.0 * resolution_km, 2000.0)

# --- Mesh relaxation (Lloyd's algorithm / centroidal Voronoi) --------------
# The raw icosahedral subdivision has irregular edge lengths and mixed valence
# (4-8). Lloyd's relaxation (move each node toward its Voronoi-cell centroid,
# then re-triangulate to stay Delaunay) equalizes edge lengths and yields a
# clean 12-pentagon + hexagon icosphere. That smooths out the radial "spoke"
# artifact at the poles (README, "Pole issue in rendering").
# LLOYD_ITERATIONS=0 disables it (use the raw subdivision). Cost is a one-time
# mesh build (~0.9 s per iteration at r=100 / n=67; ~0.15 s at the r=250 default).
LLOYD_ITERATIONS: int = 20
LLOYD_ALPHA: float = 0.8            # step size (<=1 keeps the move monotone)

# --- Land-aware mesh refinement --------------------------------------------
# When enabled, base triangles whose centroid is over land are refined by
# `LAND_REFINE_STEPS` subdivision step(s) (edge midpoints inserted) while ocean
# triangles keep the base spacing. The mixed-density point set is then
# re-triangulated to the spherical Delaunay so the result stays a CONFORMING
# simplicial complex (every shared edge has the same two endpoints on both
# sides, no hanging nodes) -- that is what lets the P1 engine and its edge-based
# mass/stiffness assembly keep working. ON by default (the shipped run uses the
# land-finer mesh); pass `Mesh(..., refine_land=False)` or set this to False for
# the uniform base mesh (what the base-icosphere mesh tests pin).
REFINE_LAND: bool = True
LAND_REFINE_STEPS: int = 1          # 1 = one subdivision step over land

# --- Physics constants (visual-calibration knobs) --------------------------
K_H = 100.0              # horizontal eddy diffusivity, m^2/s
K_V = 50.0               # vertical eddy diffusivity, m^2/s
TAU_S = 20.0 * 86400.0   # background lifetime of excess CO2, seconds (20 days)
C_CAP = 1.0e6            # hard cap on concentration (excess-ppm), insurance

# --- Time ------------------------------------------------------------------
DT_S = 900.0             # 15-minute explicit step
N_DAYS = 365
N_STEPS_PER_DAY = int(86400 / DT_S)      # 96
N_STEPS = N_DAYS * N_STEPS_PER_DAY       # 35,040
SAMPLE_EVERY = int(3600 / DT_S)          # sample hourly -> every 4 steps

# --- Vertical discretization ------------------------------------------------
# Geometric levels: dense at the ground, coarsest at the top, top cell at 100 km.
N_VLEVELS = 16           # number of cells (17 level interfaces)
TOP_KM = 100.0
V_RATIO = 1.4            # geometric growth ratio of cell thickness


def make_vertical_levels(n_cells: int = N_VLEVELS, top_km: float = TOP_KM,
                         ratio: float = V_RATIO):
    """Return (z_levels_km, first_thickness_km).

    z_levels has n_cells+1 entries: 0 .. top_km. Cell thicknesses grow
    geometrically by `ratio`; the first thickness is chosen so the cells sum to
    exactly `top_km`. Dense near the ground, coarsest at the top (as required).
    """
    d0 = top_km * (ratio - 1.0) / (ratio ** n_cells - 1.0)
    z = [0.0]
    acc = 0.0
    dz = d0
    for k in range(n_cells):
        acc += dz
        if k == n_cells - 1:
            acc = top_km            # stretch the top cell to land exactly on `top`
        z.append(acc)
        dz *= ratio
    return z, d0


V_LEVELS_KM, V_DZ0_KM = make_vertical_levels()

# --- Sources ---------------------------------------------------------------
# Emission strengths are in "excess-ppm per second" units applied to the
# ground level; they are visual-calibration knobs, not real-world fluxes.
# Sized so that, against the 20-day background sink, the far field stays at a
# sane scale (tens of excess-ppm, not the ~10^3-10^5 the old 4.0/2.5 ppm/s
# produced) while source regions remain visibly elevated.
FIRE_PEAK_SRC = 0.15       # daytime peak fire source strength
# Per-city population source strength; each city emits POP_PEAK_SRC * weight,
# so the total is POP_PEAK_SRC * sum(POP_CENTERS weights). Re-tuned down from
# 0.10 when POP_CENTERS grew from 23 to 100 cities (weight-sum 13.5 -> 29.2)
# so the *total* population budget — and the far-field scale it sets — is
# unchanged, and the extra cities merely redistribute that same budget.
POP_PEAK_SRC = 0.0463
FIRE_DIURNAL = 0.9         # fraction of fire strength that is diurnally modulated
# City diurnal cycle is bimodal (README "Daily CO2 cycle in cities"); its
# shape and amplitudes live in sources.py (POP_DIURNAL_TERMS), normalized
# to a daily mean of 1 so the budget set by POP_PEAK_SRC is unchanged.

# --- Rendering normalization -------------------------------------------------
# `col_ref` (runner.py) is the haze reference: the field value at which the
# viewer's alpha saturates. It is set to a percentile of the captured field.
# The excess-CO2 field is BROAD -- most of the globe's mass sits in a diffuse
# downwind plume, not a few peaks -- so anchor to a high-but-common level, NOT
# the extreme peak. Anchoring to the peak (the old p99.9) made only the sharp
# source dots clear the viewer's alpha gate and wiped out the large-scale
# plume structure, leaving isolated "pumping dots" at grid resolution.
COL_REF_PERCENTILE = 96.0


def subsolar_lat(day: float) -> float:
    """Subsolar latitude in degrees, day in 1..365. (matches wind.py)"""
    return 23.44 * math.sin(TWO_PI * (day - 80.0) / 365.0)
