# `sim/` — CO₂ transport simulation engine

The physics core. It advects and diffuses a field of **excess CO₂** (in arbitrary
"excess-ppm" units) over a finite-element mesh of the sphere, driven by two
source classes (diurnal forest fires and diurnal dense-population emissions) and
an idealized global wind field, then hands the time series to the server.

It is a **qualitative visualization model**, not a climate model: the transport
*math* is correct and invariant-checked (mass conservation, monotonicity, CFL
stability — see `../tests/`), but the *magnitudes* are visual-calibration knobs,
not physical fluxes. That distinction is important and is called out in
[Accuracy](#accuracy).

## Files

| File          | Role                                                                 |
|---------------|----------------------------------------------------------------------|
| `config.py`   | Every tunable: resolution, physics constants, timesteps, vertical grid, source strengths, the land-refinement flag, the render reference |
| `mesh.py`     | Conforming unstructured spherical triangle mesh (icosphere + Lloyd/CVT relaxation + optional land-aware refinement) |
| `wind.py`     | Idealized wind: zonal `u`/meridional `v` belts + a mass-balanced vertical `w` |
| `land.py`     | Rasterized land mask from Natural Earth GeoJSON (used by sources & the globe texture) |
| `sources.py`  | Fire + population source footprints, diurnal and seasonal modulation |
| `engine.py`   | Explicit time-stepping core: upwind advection + P1 diffusion + vertical FVM + sink |
| `runner.py`   | Runs a time window and packs each frame into `[col, hfrac]` for the viewer |

The data file `../data/ne_110m_land.geojson` (Natural Earth, public domain) is
consumed by `land.py`.

## Pipeline

```
config.py  ── knobs ──┐
                      ▼
   mesh.py (nodes + triangles) ──► engine.py  ◄── wind.py  (u, v, w)
                      │                ▲
                      │                │ emission(N,) at ground level
                      │          sources.py ◄── land.py
                      ▼
                  runner.py   (M frames × [col, hfrac])  ──►  server/ ──►  viewer/
```

A "frame" is a snapshot of the (N, K) concentration field at one (day, hour),
reduced to the two scalars the viewer renders ([`runner.py:1-14`](runner.py)):

- **`col`** — column-integrated excess CO₂ (sum over the K vertical cells): how
  much CO₂ sits above a location.
- **`hfrac`** — vertical centroid of that column, 0..1: where in the 0–100 km
  column the mass sits (drives cloud height).

## Core ideas

### 1. The mesh: icosphere → centroidal Voronoi (Lloyd) → spherical Delaunay

The base mesh is an **icosahedral subdivision** — split each edge of a 12-vertex
icosahedron into `n` segments. It is conforming by construction, has (near)
uniform cell size, and has no lat/lon pole slivers (which would wreck the
advection CFL at high latitude). Counts follow the icosphere formulas:
`20·n²` triangles, `10·n² + 2` nodes.

| land spacing `resolution_km` | `n` | triangles | nodes |
|---|---|---|---|
| 100  | 67 | 89,780 | 44,892 |
| **250** (default) | **27** | **14,580** | **7,292** |
| 1000 | 7  | 980    | 492    |

(The base icosahedron edge on Earth is `2/√(1+φ²)·R ≈ 6699 km`; `n = round(6699/r)`.)

The raw subdivision has irregular edges and mixed valence (4–8). It is then
relaxed with **Lloyd's algorithm** ([Lloyd 1977](https://ieeexplore.ieee.org/document/1053285);
see also Du, Faber & Warburton 1999, "Centroidal Voronoi-Tessellations"): move
each node toward the centroid of its Voronoi cell, then re-triangulate to the
spherical Delaunay. This equalizes edge lengths and regularizes the 12
pentagonal points — which also **removes the radial "spoke" artifact at the
poles** in the render (the README's "Pole issue").

The re-triangulation step is exact: the 3-D convex hull of a point set
surrounding the origin *is* the spherical Delaunay triangulation, so a
`ConvexHull` call is the right move ([`mesh.py:145-163`](mesh.py)):

```python
def _spherical_delaunay(V):
    try:
        T = ConvexHull(V).simplices.astype(np.int64)
    except Exception:                      # near-coplanar: retry with jitter
        T = ConvexHull(V, qhull_options="QJ").simplices.astype(np.int64)
    ...
```

and one Lloyd iteration is a move + re-project + re-triangulate
([`mesh.py:177-181`](mesh.py)):

```python
for _ in range(int(n_iter)):
    V = V + alpha * (_incident_centroid(V, T) - V)   # toward Voronoi centroid
    V /= np.linalg.norm(V, axis=1, keepdims=True)    # back onto the unit sphere
    T = _spherical_delaunay(V)                        # stay Delaunay
```

Node count and Euler characteristic (`V − E + F = 2`) are preserved; a clean
icosphere has exactly **12 pentagonal points and hexagons elsewhere** (Euler-
forced), which `test_mesh.py` verifies.

**Optional: land-aware refinement.** With `config.REFINE_LAND = True`
(`LAND_REFINE_STEPS`, default 1), the *uniform* base mesh is refined so land is
finer than ocean: each triangle whose centroid is over land (queried from
`land.py`) gets one extra subdivision step (edge midpoints inserted) while ocean
triangles are left at the base spacing. Rather than subdivide-and-keep — which
leaves a **hanging node** on every land/ocean boundary edge — the denser point
set is re-triangulated to the spherical Delaunay, so the result is still a
conforming simplicial complex (every shared edge has two triangles, no hanging
nodes). That is the key property: it lets the P1 engine and its edge-based mass/
stiffness assembly run **unmodified** on the non-uniform mesh. It is applied
after Lloyd (so the base is clean) and not re-relaxed afterward (Lloyd equalizes
cell size, which would erase the land/ocean contrast). Off by default; the
12-pentagon statement above describes the base mesh. `test_refined_land.py`
pins conformingness, CFL < 1, and mass conservation on the refined mesh.

### 2. Vertical grid: geometric levels, dense at the ground

The 0–100 km column is split into `N_VLEVELS = 16` cells whose thicknesses grow
geometrically by `V_RATIO = 1.4`, densest at the surface ([`config.py:51-69`](config.py)):

```
0, 0.18, 0.44, 0.80, 1.31, 2.02, 3.01, 4.40, 6.35, 9.07, 12.88,
18.22, 25.69, 36.15, 50.79, 71.30, 100   (km)
```

First cell ≈ **185 m** (resolves the boundary layer where emissions land),
top cell ≈ **28.7 km**. The top cell is stretched so the last interface lands on
exactly 100 km.

### 3. Wind: idealized zonal belts + a mass-balanced vertical field

`wind.py` is the single source of truth for wind, shared by the sim (and ported
verbatim into the viewer). The dominant signal is the **zonal belt structure**
so plumes drift **west→east in the mid-latitudes and east→west in the tropics**,
in both hemispheres, as the top-level README requires
([`wind.py:51-59`](wind.py)):

```python
u = (
    -8.0 * (_G(phi,  8.0 + s, 10.0) + _G(phi, -8.0 - s, 10.0))   # easterly trades
    +12.0 * (_G(phi, 45.0 + s, 12.0) + _G(phi,-45.0 - s, 12.0))  # westerlies
    - 6.0 * (_G(phi, 75.0 + s, 12.0) + _G(phi,-75.0 - s, 12.0))  # polar easterlies
    ...
)
```

Belt centers shift with the **subsolar latitude**
`φ_sun = 23.44·sin(2π(day−80)/365)` — Earth's obliquity (23.44°), zero at the
~day-80 equinox — so the whole circulation migrates seasonally
([`wind.py:31-33`](wind.py)). The vertical `w` field encodes a Hadley/Ferrel
pattern (ITCZ upwelling, subtropical subsidence, mid-latitude ascent, polar
subsidence) and **has its area-weighted latitude mean removed** so there is zero
net vertical mass flux on the closed sphere ([`wind.py:130-141`](wind.py)).

### 4. Sources: fires (diurnal + seasonal) and cities (bimodal diurnal)

Each source is a **Gaussian footprint on the sphere**, normalized to sum to 1 so
a source's time-dependent strength `s(t)` distributes as `s(t)·w_i`
([`sources.py:230-239`](sources.py)):

```python
cos_d = np.clip(node_pos @ center, -1.0, 1.0)
d = np.arccos(cos_d)                       # angular distance
w = np.exp(-0.5 * (d / sig) ** 2)
w /= w.sum()
```

- **Fires** (18 regions): a small year-round base + a circular-Gaussian season
  peaking at each region's dry season, modulated by an afternoon diurnal cosine
  (flare by day, decay by night).
- **Cities** (100 centers): a **bimodal urban cycle** implementing the top-level
  README's "Daily CO₂ cycle in cities" — sharp morning peak (~8:00), midday
  dip, broader evening peak (~18:30), overnight trough (~3:00)
  ([`sources.py:170-174`](sources.py)):

```python
POP_DIURNAL_TERMS = (
    (8.0,  1.1, +0.60),    # morning peak: sharpest spike of the day
    (18.5, 2.0, +0.45),    # evening peak: second, broader
    (3.0,  2.2, -0.40),    # nighttime trough: lowest point, ~3 AM
)
```

Both are normalized so the **daily mean is exactly 1** (a truncated-Gaussian
mean, [`sources.py:178-193`](sources.py)), so the annual budget set by
`FIRE_PEAK_SRC` / `POP_PEAK_SRC` is unchanged by the diurnal shape. Modulation
uses **local solar time** `h_local = (hour_utc + lon/15) % 24`
([`sources.py:292-301`](sources.py)) — the same relation the viewer's sun uses.

### 5. Engine: upwind advection + P1 diffusion + vertical FVM + sink

Explicit Euler on the (N, K) field. The mesh lives on the **unit sphere**, so
wind is scaled by `1/R` and diffusivity by `1/R²`; the vertical column keeps SI
units ([`engine.py:18-19`](engine.py)).

**Horizontal advection** — first-order **upwind** edge flux: for each directed
edge, take the concentration at the upwind node, move `max(U·n,0)·c_up` out, and
deposit it on the two downwind edge nodes in proportion to their lumped masses.
This is **monotone** (no new extrema) and **exactly mass-conserving**
([`engine.py:222-238`](engine.py)):

```python
q = np.einsum("ij,ij->i", We, self.EN)                 # signed outflow speed
k_up = self.TRI_NODES[np.arange(q.shape[0]), D.argmin(axis=1)]  # upwind node
F = np.where(q > 0, self.ADV_SPEED * q * self.L * c[k_up], 0.0) # mass moved out
# deposit on downwind edge nodes ∝ their lumped masses; remove from upwind node
```

**Horizontal diffusion** — exact P1 stiffness (the "cofactor" off-diagonal),
row-sum-zero so it is conservative
([`engine.py:155-157`](engine.py)):

```python
K01 = -Keff * (l02**2 + l12**2 - l01**2) / (8.0 * A)   # off-diag for edge (0,1)
```

**Vertical transport** — finite volume on the 16 levels with a **closed lid at
100 km**: upwinded advection by the idealized `w` + central Fick diffusion
(high→low). Stability is enforced by **one shared explicit-CFL budget** — the
advection and diffusion CFL numbers *add* (both drain a cell), so they are
split as diffusion 0.5 + advection 0.4 = 0.9, keeping the self-coefficient
non-negative (no spurious negative values)
([`engine.py:71-104`](engine.py)).

**Sink + update** — a background first-order loss over a 20-day lifetime, a
positivity clip, and a hard cap ([`engine.py:267-270`](engine.py)):

```python
c += self.DT * rate / self.mesh.M[:, None]
c *= 1.0 - self.DT / C.TAU_S          # background sink (20 days)
np.clip(c, 0.0, C.C_CAP, out=c)
```

### 6. Runner: packing frames

`runner.py` steps the engine across the window and, at each sample time, writes
`col = c.sum(axis=1)` and the normalized vertical centroid `hfrac` into one
contiguous `(M, 2, N)` float32 buffer, plus a JSON-safe `meta` (node positions,
land mask, `col_ref`, source list) ([`runner.py:79-116`](runner.py)). The haze
reference `col_ref` is the field value at which the viewer's alpha saturates — set to a
high-but-common percentile of the captured field (`COL_REF_PERCENTILE = 96`),
*not* the extreme peak. The excess-CO₂ field is broad: most of the globe's mass
sits in a diffuse downwind plume, not a few hot spots. Anchoring the color ramp
to the peak (the old p99.9) made only the sharp source dots clear the viewer's
alpha gate and wiped out the large-scale plume structure, leaving isolated
"pumping dots" at grid resolution.

## Accuracy

The opening line makes the split, and the rest of this section is what backs it.

**What is correct** is the *transport mathematics*, and it is invariant-checked
in `../tests/`:

- **Mass conservation** — upwind advection moves mass node-to-node and the P1
  diffusion matrix is row-sum-zero, so the field total is preserved to roundoff.
- **Monotonicity** — first-order upwinding and the shared explicit-CFL split keep
  every cell's self-coefficient non-negative, so the step introduces no new extrema.
- **Stability** — the horizontal and vertical CFL numbers are budgeted so the
  explicit update stays non-oscillatory (see the vertical split in
  [`engine.py:71-104`](engine.py)).
- **Mesh validity** — a clean icosphere has exactly 12 pentagonal points with
  hexagons elsewhere, and the Euler characteristic `V − E + F = 2` holds
  (`test_mesh.py`).

**What is *not*** are the magnitudes. Concentrations are in arbitrary
"excess-ppm" units, and every strength is a visual-calibration knob, not a
measured flux: source peaks (`FIRE_PEAK_SRC`, `POP_PEAK_SRC`), the 20-day
background lifetime (`TAU_S`), the horizontal diffusivity, and the idealized
belt wind (`wind.py`) are all tuned to produce a readable, well-behaved plume —
not to reproduce real CO₂. Treat the output as a **qualitative behavior model**
(how a broad plume drifts, mixes, and settles under diurnal and seasonal
forcing), not a quantitative forecast.
