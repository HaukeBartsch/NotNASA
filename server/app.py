"""FastAPI bridge: runs the CO2 simulation and serves its frames to the
three.js viewer.

The simulation is run once at startup (blocking the lifespan) so the server is
only "ready" once the frame buffer exists. Configuration comes from the
environment (set by capture/capture.py, or defaults for a quick 1-day demo):

    RES_KM        mesh land-cell spacing, km        (default 250)
    START_DAY     first day of the window, 1..365   (default 150)
    DAYS          length of the window, days        (default 3)
    SPD           samples per day (frame rate of the sim)  (default 48)
    WIDTH/HEIGHT  render size (informational for the viewer)  (1280/720)
    FPS           movie playback fps (informational)          (default 8)
    OUT_DIR       output folder (informational, used by capture)
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from sim import runner

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIEWER_DIR = os.path.join(ROOT, "viewer")
THREE_DIR = os.path.join(ROOT, "node_modules", "three")

STATE: dict = {}


def _cfg() -> dict:
    return {
        "resolution_km": float(os.environ.get("RES_KM", "250")),
        "start_day": float(os.environ.get("START_DAY", "150")),
        "days": float(os.environ.get("DAYS", "3")),
        "samples_per_day": int(os.environ.get("SPD", "48")),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = _cfg()
    print(f"[server] sim config: {cfg}", flush=True)
    result = runner.run_simulation(**cfg, progress=True)
    STATE.update(result)
    m = result["meta"]
    print(f"[server] ready: M={m['M']} N={m['N']} K={m['K']} "
          f"col_ref={m['col_ref']:.1f}", flush=True)
    yield
    print("[server] shutting down", flush=True)


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


def _render_params() -> dict:
    return {
        "width": int(os.environ.get("WIDTH", "1280")),
        "height": int(os.environ.get("HEIGHT", "720")),
        "fps": int(os.environ.get("FPS", "8")),
        "out_dir": os.environ.get("OUT_DIR", os.path.join(ROOT, "output")),
    }


@app.get("/api/meta")
def meta():
    out = dict(STATE["meta"])
    out["render"] = _render_params()
    return out


@app.get("/api/field")
def field():
    """Raw little-endian float32 buffer of shape (M, 2, N): [col, hfrac]."""
    f = np.ascontiguousarray(STATE["field"], dtype="<f4")
    M, two, N = f.shape
    return Response(
        content=f.tobytes(),
        media_type="application/octet-stream",
        headers={"X-Shape": f"{M},{two},{N}"},
    )


@app.get("/api/frame/{i}")
def frame(i: int):
    """Single frame, little-endian float32 of shape (2, N): [col, hfrac].

    A full year of frames can exceed the browser's ~2GB single-ArrayBuffer cap,
    so the viewer fetches frames one at a time from here instead of the whole
    (M, 2, N) buffer up front.
    """
    M = STATE["field"].shape[0]
    if i < 0 or i >= M:
        return Response(status_code=404, content=f"frame {i} out of range [0,{M})")
    f = np.ascontiguousarray(STATE["field"][i], dtype="<f4")
    return Response(content=f.tobytes(), media_type="application/octet-stream")


@app.get("/api/sources")
def sources():
    return STATE["meta"]["sources"]


# --- static viewer + three.js ---------------------------------------------
@app.get("/")
def index():
    return FileResponse(os.path.join(VIEWER_DIR, "index.html"))


@app.get("/app.js")
def app_js():
    return FileResponse(os.path.join(VIEWER_DIR, "app.js"))


@app.get("/earth.jpg")
def earth_texture():
    return FileResponse(os.path.join(VIEWER_DIR, "earth.jpg"), media_type="image/jpeg")


app.mount("/three", StaticFiles(directory=THREE_DIR, check_dir=False), name="three")
