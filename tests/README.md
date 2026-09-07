# `tests/` — invariant checks

These are not feature tests; they pin the **invariants** the sim claims to hold
(see [`../sim/README.md`](../sim/README.md) → *Accuracy*). Each file is also a
standalone script you can run directly.

## Run

With pytest (from the repo root):

```sh
.venv/bin/python -m pytest tests/ -q
```

Or run a single file without pytest — it prints and exits non-zero on failure:

```sh
.venv/bin/python tests/test_engine.py
```

## What's checked

| File               | Invariants pinned                                                                                                                        |
|--------------------|------------------------------------------------------------------------------------------------------------------------------------------|
| `test_mesh.py`     | topology invariants hold (`V − E + F = 2`); a clean icosphere has exactly 12 pentagonal points with hexagons elsewhere; Lloyd/CVT relaxation improves edge uniformity, is deterministic, and reproduces the raw subdivision when disabled; the spherical Delaunay re-triangulation reprojects onto the unit sphere |
| `test_engine.py`   | total mass is conserved (uniform field, and with sources); advection drifts the field downwind; vertical advection moves mass upward under upwelling; diffusion smooths and preserves mass; sources stay localized; a sustained multi-step run stays stable; and the CFL numbers stay below 1 |
| `test_sources.py`  | the urban diurnal cycle matches the top-level README's "Daily CO₂ cycle in cities" — sharp morning peak, midday dip, broader evening peak, ~3 AM trough — with the daily mean exactly 1 so the annual budget is unchanged, and the fire source unaffected |
| `test_refined_land.py` | the optional land-aware refinement (one step over land, none over ocean) yields a conforming mesh (no hanging nodes), keeps the fixed timestep inside the advection CFL limit, and conserves mass with no new extrema — i.e. the P1 engine runs unmodified on the land-finer mesh |
