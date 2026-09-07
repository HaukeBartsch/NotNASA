# `data/` — input data

| File                    | What it is                                                        | Size      |
|-------------------------|-------------------------------------------------------------------|-----------|
| `ne_110m_land.geojson`  | Natural Earth 1:110m land polygons — the world's coastlines        | ~138 KB, 127 polygons |

## Provenance & license

Natural Earth (<https://www.naturalearthdata.com>), **public domain** — free
for commercial and non-commercial use, no attribution required.

The **1:110m** scale is the coarsest (most generalized) land outline. That's all
the sim needs — a coarse mask, not high-resolution coastlines.

## How it's used

`sim/land.py` reads the GeoJSON and rasterizes it into a boolean land mask
(`nlat × nlon`, `True` = land). The mask is consumed in two places:

- `sim/runner.py` ships it in the viewer's `meta` so the globe can render land
  vs. ocean.
- `sim/mesh.py` (optionally) uses it to **refine the mesh over land**: when
  `config.REFINE_LAND` is on, base triangles whose centroid is over land get one
  extra subdivision step (ocean stays at the base spacing), producing a
  land-finer / ocean-coarser conforming mesh. **Off by default** — the shipped
  run uses the uniform mesh.

It does **not** place sources (those use their own hard-coded lat/lon footprints
in `sim/sources.py`).
