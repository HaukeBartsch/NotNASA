# `capture/` — headless movie capture

Drives the whole pipeline end-to-end to produce a movie: it spawns the FastAPI
server (which runs the simulation), drives the three.js viewer headlessly with
Playwright, and screenshots one PNG per simulation frame.

## What it does

1. Picks a free port and starts the server (`uvicorn server.app:app`) with the
   chosen sim params passed as env vars.
2. Waits for `GET /api/meta` — the server only becomes ready once the sim is
   finished — and fetches the metadata.
3. Launches headless Chromium (default GPU, then a software-WebGL/SwiftShader
   fallback if init fails) and loads the viewer, waiting on `window.__ready`.
4. For each frame: calls `window.__settle(i)` (draw frame *i* and wait for the
   GPU) and screenshots it to `output/frames/frame_%06d.png`.
5. Writes two post-processing helpers into the output dir:
   - `meta.json` — the render + sim parameters of this run.
   - `make_movie.sh` — the ffmpeg one-liner that assembles the frames into
     `co2_movie.mp4` (H.264).

## Usage

```sh
.venv/bin/python capture/capture.py                     # 3-day demo, 720p
.venv/bin/python capture/capture.py --days 1 --spd 48   # 1 day, every 30 min
.venv/bin/python capture/capture.py --max-frames 6      # quick smoke test
```

Key flags: `--res` (mesh spacing, km), `--start-day`, `--days`, `--spd`
(frames/day), `--width` / `--height`, `--fps`, `--out`, `--port`, `--max-frames`,
`--keep-server`.

## Output

The primary deliverable is the **`frames/` folder** — re-process it freely
(`make_movie.sh`, or any ffmpeg / ImageMagick pipeline). Alongside it the script
writes `meta.json`, `make_movie.sh`, and `server.log` (the server's stdout,
auto-printed on error).

## Requirements

Playwright's Chromium (`playwright install chromium`) and a working `ffmpeg`
with libx264. See the top-level README for the verified environment.
