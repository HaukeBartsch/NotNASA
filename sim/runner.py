"""Run the CO2 simulation over a time window and extract per-frame fields
for the viewer.

A "frame" is a snapshot of the concentration field at one (day, hour) pair.
Each frame is reduced to the two quantities the viewer actually renders:

    col    : (N,) column-integrated excess CO2  (sum over the K vertical cells)
             -- how much CO2 sits above a location; what a satellite "sees".
    hfrac  : (N,) vertical centroid of that column, normalized to 0..1
             -- where in the 0..100 km column the mass sits; drives cloud height.

Both are packed into one (M, 2, N) float32 array so the whole movie is a single
contiguous buffer the server can ship in one request.
"""
from __future__ import annotations

import time

import numpy as np

from . import config as C
from .engine import Engine
from .mesh import Mesh
from .sources import Sources


def sample_times(start_day: float, days: float, samples_per_day: int):
    """Yield (day, hour_utc, global_step_index) for each frame of the window.

    The sim steps at N_STEPS_PER_DAY (96, i.e. 15 min) per day. We pick
    `samples_per_day` snapshot times per day; the global step index is where
    in the step loop we capture the field.
    """
    spb = C.N_STEPS_PER_DAY
    total = int(round(days * samples_per_day))
    for m in range(total):
        day_frac = m / samples_per_day
        day = int(start_day + day_frac)
        frac = day_frac - int(day_frac)
        hour = frac * 24.0
        step = int(round(day_frac * spb))
        yield float(day), float(hour), step


def run_simulation(resolution_km: float = 250.0,
                   start_day: float = 150.0,
                   days: float = 3.0,
                   samples_per_day: int = 24,
                   progress: bool = False) -> dict:
    """Simulate `days` (starting at `start_day`) and return the frame buffer.

    Returns a dict with keys:
        meta   : JSON-safe metadata (node positions, land mask, palette ref, ...)
        field  : (M, 2, N) float32 array  [col, hfrac] per frame
    (The mesh/sources/engine are not returned; they are sim-internal.)
    """
    t0 = time.time()
    mesh = Mesh(resolution_km)
    sources = Sources(mesh)
    engine = Engine(mesh, sources)
    N, K = mesh.N, engine.K
    levels = np.arange(K, dtype=np.float64)
    print(f"[runner] mesh N={N} K={K} built in {time.time()-t0:.1f}s")

    times = list(sample_times(start_day, days, samples_per_day))
    M = len(times)
    field = np.empty((M, 2, N), dtype=np.float32)
    stamp = {step: i for i, (_, _, step) in enumerate(times)}
    max_step = max(step for (_, _, step) in times)
    print(f"[runner] simulating {max_step + 1} steps "
          f"({max_step / C.N_STEPS_PER_DAY:.1f} days), capturing {M} frames")

    c = np.zeros((N, K))
    t1 = time.time()
    for s in range(max_step + 1):
        day = start_day + s / C.N_STEPS_PER_DAY
        hour = (s % C.N_STEPS_PER_DAY) / C.N_STEPS_PER_DAY * 24.0
        engine.step(c, day, hour)
        if s in stamp:
            i = stamp[s]
            col = c.sum(axis=1)                       # (N,)
            cent = c @ levels                          # (N,) weighted level index
            safe = np.where(col > 0.0, col, 1.0)
            hfrac = cent / (safe * (K - 1))
            field[i, 0] = col
            field[i, 1] = hfrac
            if progress and (i % max(1, M // 20) == 0 or i == M - 1):
                el = time.time() - t1
                rate = (s + 1) / el
                print(f"  frame {i + 1:4d}/{M}  day={day:6.2f} hour={hour:4.1f}  "
                      f"{rate:.0f} steps/s")
    dt = time.time() - t1
    print(f"[runner] sim done in {dt:.1f}s ({(max_step + 1) / dt:.0f} steps/s)")

    col_all = field[:, 0, :]
    # Anchor the haze reference to a high-but-common level of the (broad) field,
    # not the extreme peak -- see C.COL_REF_PERCENTILE for why.
    col_ref = float(np.percentile(col_all, C.COL_REF_PERCENTILE))
    land_mask = _land_mask()

    meta = {
        "N": int(N), "K": int(K), "M": int(M),
        "vlevels_km": [float(z) for z in C.V_LEVELS_KM],
        "resolution_km": float(resolution_km),
        "start_day": float(start_day),
        "days": float(days),
        "samples_per_day": int(samples_per_day),
        "col_ref": col_ref,
        "col_max": float(col_all.max()),
        "nodes": mesh.V.astype(np.float32).reshape(-1).tolist(),   # (N*3,)
        "land_mask": _b64(land_mask), "land_nlat": land_mask.shape[0],
        "land_nlon": land_mask.shape[1],
        "frames": [{"day": float(d), "hour": float(h)} for d, h, _ in times],
        "sources": sources.to_dict(),
    }
    return {"meta": meta, "field": field}


def _land_mask() -> np.ndarray:
    """Rasterized land mask (nlat, nlon) uint8, for the globe vertex colors."""
    from .land import rasterize
    return (rasterize() * 255).astype(np.uint8)


def _b64(arr: np.ndarray) -> str:
    import base64
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Run the CO2 sim and print a summary")
    p.add_argument("--res", type=float, default=250.0)
    p.add_argument("--start-day", type=float, default=150.0)
    p.add_argument("--days", type=float, default=3.0)
    p.add_argument("--spd", type=int, default=48)
    a = p.parse_args()
    r = run_simulation(a.res, a.start_day, a.days, a.spd, progress=True)
    m = r["meta"]
    print(f"\nframes={m['M']} N={m['N']} K={m['K']} "
          f"col_ref={m['col_ref']:.1f} col_max={m['col_max']:.1f}")
