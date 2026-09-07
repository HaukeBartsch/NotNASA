# `viewer/` — three.js renderer

Renders a frame of the simulated excess-CO₂ field as semi-transparent wildfire
smoke over a globe, with a day/night sun that sweeps the terminator across the
globe as the movie's clock runs (pale thin haze → tan → amber → thick brown-gray,
orange over active fire, a faint fresnel rim at the limb).

## Files

| File          | Role                                                                                  |
|---------------|---------------------------------------------------------------------------------------|
| `index.html`  | the page: a full-viewport `<canvas>` that loads `/app.js`                            |
| `app.js`      | the renderer (three.js): haze, sun/day-night, sky, camera, and the capture contract  |
| `earth.jpg`   | the equirectangular surface texture (the base globe)                                 |

three.js itself is loaded from `../node_modules/three`, served at `/three/...`
by `../server/`.

## How the field is drawn

Two haze renderers, switched with `?haze=sprites|shell`:

- **`shell`** (default): the field splatted onto a 1024×512 equirectangular grid
  and drawn as an alpha-blended shell. A continuous field (no concentric Moiré
  rings) and a smooth polar cap (no radial "starburst").
- **`sprites`**: one soft point-sprite per node, at the node positions.
  Pole-clean, but large sprites alias into "lens" rings and small ones expose the
  dot lattice — the fallback, not the default.

The field is peaky (a few hot nodes, most of the globe near zero), so a tight
Gaussian splat keeps plumes detailed instead of smearing them into blobs; alpha
(not additive) blending over the terrain reads as smoke over the surface rather
than a glow.

## Day/night sun

The sun's direction is a **pure function of the frame's `(day, hour_utc)`** — no
wall clock, no `Math.random` — and it matches the sim exactly: subsolar latitude
`23.44·sin(2π(day−80)/365)` (from `../sim/wind.py`) and subsolar longitude
`(12 − hour_utc)·15°` (local solar time, from `../sim/sources.py`). That
determinism is what lets `../capture/` screenshot reproducible frames.

## Coordinate frame

Matches the sim's node positions: `x = coslat·coslon`, `y = coslat·sinlon`,
`z = sinlat` — north pole at `+z`, longitude 0 at `+x`.

## Tuning (URL query params)

| Param         | Default | Meaning                                                        |
|---------------|---------|----------------------------------------------------------------|
| `haze`        | `shell` | `shell` or `sprites`                                           |
| `fresnel`     | `0.30`  | atmosphere rim gain (`0` disables)                              |
| `fresnelPow`  | `3.5`   | rim exponent                                                    |
| `fresnelRad`  | `1.012` | rim radius (× globe radius)                                     |
| `ref`         | `col_ref` | haze reference override (else the sim's `meta.col_ref`)       |
| `spriteR`     | `5.0`   | smoke sprite angular radius, degrees (sprites mode)            |
| `spriteOp`    | `0.55`  | per-sprite alpha (sprites mode)                                 |
| `poleband`    | `16`    | pole rows fully zonal-meaned (shell mode)                       |
| `daynight`    | on      | `0` → the old flat always-lit look (A/B)                        |
| `smokeSun`    | `1`     | `0` → don't sun-shade the CO₂ smoke                             |
| `nightLight`  | `0.30`  | night-side "moonlight" fill (`0` → near-black)                  |
| `sky`         | on      | `0` → plain dark background (no Milky Way)                      |
| `sunDay`/`sunHour` | —   | pin the sun to a static time (A/B)                              |

> Several knobs also have a matching constant at the top of `app.js`
> (`SHELL_REF_F`, `SHELL_EXPOSURE`, `CAM_DIST`, `FOOV`, `FOCUS_LAT` / `FOCUS_LON`)
> — tune those directly and re-render.

## Capture contract

`../capture/capture.py` relies on two globals `app.js` exposes:

- `window.__ready` — `true` once meta + field are loaded and frame 0 is drawn.
- `window.__settle(i)` — async; draws frame *i*, waits for the GPU, resolves `true`.

Because capture loads the plain URL (no `?live`), no auto-advance runs and every
frame is deterministic.
