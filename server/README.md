# `server/` — FastAPI bridge

Sits between the Python simulation and the browser viewer. It runs the sim
**once at startup** (the `lifespan` blocks until the frame buffer exists, so the
server is only "ready" once data is available) and then serves the frames plus
the static `../viewer/` assets and the three.js library.

## Files

| File          | Role                                                                                     |
|---------------|------------------------------------------------------------------------------------------|
| `__init__.py` | marks `server` a package                                                                 |
| `app.py`      | the FastAPI app: startup sim, API endpoints, static serving of `../viewer/` + three.js  |

## Endpoints

| Route            | Returns                                                                                                     |
|------------------|-------------------------------------------------------------------------------------------------------------|
| `GET /api/meta`  | JSON metadata: `N, K, M`, `vlevels_km`, `col_ref`, `col_max`, node positions, land mask, `sources`, plus render params |
| `GET /api/field` | the whole `(M, 2, N)` float32 buffer `[col, hfrac]` as raw little-endian bytes (`X-Shape` header)          |
| `GET /api/frame/{i}` | one frame, `(2, N)` float32 `[col, hfrac]` — the viewer pulls frames one at a time from here          |
| `GET /api/sources` | the source list (fire regions + population centers)                                                       |
| `GET /`, `/app.js`, `/earth.jpg` | the static viewer (`../viewer/`)                                                 |
| `GET /three/...` | the three.js library, served from `../node_modules/three`                                                 |

`/api/field` can be large — a full year of frames can exceed the browser's
~2 GB single-ArrayBuffer cap, which is why the viewer prefers `/api/frame/{i}`
(one frame at a time) instead of the whole buffer up front.

## Configuration (environment variables)

The sim is configured by env vars (set by `../capture/capture.py`, or these
defaults for a quick demo):

| Var                 | Meaning                                       | Default     |
|---------------------|-----------------------------------------------|-------------|
| `RES_KM`            | mesh land-cell spacing, km                    | `250`       |
| `START_DAY`         | first day of the window (1..365)              | `150`       |
| `DAYS`              | window length, days                           | `3`         |
| `SPD`               | samples / frames per day                      | `48`        |
| `WIDTH` / `HEIGHT`  | render size (informational for the viewer)    | `1280/720`  |
| `FPS`               | movie playback fps (informational)            | `8`         |
| `OUT_DIR`           | output folder (informational, used by capture)| `output`    |

## Run

```sh
.venv/bin/python -m uvicorn server.app:app --port 8000 --host 127.0.0.1
# or, from the repo root:
npm run serve
```

The server blocks on the simulation until it's done, then serves. Point a
browser at `http://127.0.0.1:8000` to see the viewer.
