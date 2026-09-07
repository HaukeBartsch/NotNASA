"""Capture the CO2 weather movie as a folder of PNG frames.

Spawns the FastAPI server (which runs the simulation), drives the three.js
viewer headlessly with Playwright, and screenshots one PNG per simulation
frame into OUT_DIR/frames/. Also writes:

    OUT_DIR/meta.json       -- render + sim parameters (for post-processing)
    OUT_DIR/make_movie.sh   -- one-liner to assemble the frames into an MP4

Usage:
    .venv/bin/python capture/capture.py                    # 3-day demo, 720p
    .venv/bin/python capture/capture.py --days 1 --spd 48  # 1 day, every 30 min
    .venv/bin/python capture/capture.py --max-frames 6     # quick test

The primary output is the frames/ folder (post-process / convert freely).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV_PY = ROOT / ".venv" / "bin" / "python"
sys.path.insert(0, str(ROOT))

from playwright.sync_api import sync_playwright  # noqa: E402


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_http(url: str, deadline_s: float, per_try: float = 20.0) -> None:
    """Block until GET url returns 200 (the server finishes its sim)."""
    t0 = time.time()
    while time.time() < t0 + deadline_s:
        try:
            with urllib.request.urlopen(url, timeout=per_try) as r:
                if r.status == 200:
                    return
        except Exception:  # noqa: BLE001 - not ready yet, retry
            pass
        print("  waiting for server to finish the simulation...")
    raise RuntimeError(f"server did not become ready within {deadline_s:.0f}s")


def _browser_args():
    # Two tiers: default GPU, then a software-WebGL fallback.
    return [
        ["--enable-unsafe-swiftshader", "--hide-scrollbars"],
        ["--enable-unsafe-swiftshader", "--use-angle=swiftshader",
         "--use-gl=angle", "--disable-gpu", "--hide-scrollbars"],
    ]


def _new_page(browser, args, width, height):
    ctx = browser.new_context(
        viewport={"width": width, "height": height}, device_scale_factor=1,
    )
    page = ctx.new_page()
    page.set_default_timeout(120000)
    return ctx, page


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--res", type=float, default=250.0, help="mesh land-cell spacing, km")
    p.add_argument("--start-day", type=float, default=150.0, help="first day, 1..365")
    p.add_argument("--days", type=float, default=3.0, help="window length in days")
    p.add_argument("--spd", type=int, default=48, help="samples (frames) per day")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=8, help="movie playback fps")
    p.add_argument("--out", type=str, default=str(ROOT / "output"))
    p.add_argument("--port", type=int, default=0, help="0 = pick a free port")
    p.add_argument("--max-frames", type=int, default=0, help="0 = all (test cap)")
    p.add_argument("--keep-server", action="store_true", help="leave the server running")
    a = p.parse_args()

    out_dir = Path(a.out).resolve()
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    # clean previous frames so the folder matches this run
    for old in frames_dir.glob("frame_*.png"):
        old.unlink()

    port = a.port or _free_port()
    base = f"http://127.0.0.1:{port}"
    env = dict(os.environ,
               RES_KM=str(a.res), START_DAY=str(a.start_day), DAYS=str(a.days),
               SPD=str(a.spd), WIDTH=str(a.width), HEIGHT=str(a.height),
               FPS=str(a.fps), OUT_DIR=str(out_dir))

    log = out_dir / "server.log"
    server = subprocess.Popen(
        [str(VENV_PY), "-m", "uvicorn", "server.app:app",
         "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        cwd=str(ROOT), env=env,
        stdout=open(log, "w"), stderr=subprocess.STDOUT,
    )

    meta = None
    exit_code = 1
    try:
        print(f"[capture] starting server on {base} (sim: {a.days}d, {a.spd}/day, res={a.res})")
        _wait_http(f"{base}/api/meta", deadline_s=1800)
        with urllib.request.urlopen(f"{base}/api/meta", timeout=30) as r:
            meta = json.load(r)
        M = meta["M"]
        nframes = M if a.max_frames <= 0 else min(a.max_frames, M)
        print(f"[capture] server ready: {M} frames, N={meta['N']}  -> capturing {nframes}")

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True, args=_browser_args()[0])
            last_err = None
            for tier, args in enumerate(_browser_args()):
                ctx, page = _new_page(browser, args, a.width, a.height)
                try:
                    page.goto(base, wait_until="load", timeout=60000)
                    try:
                        page.wait_for_function("window.__ready === true", timeout=60000)
                    except Exception:
                        err = page.evaluate("window.__initError || 'no __ready'")
                        raise RuntimeError(f"viewer init failed: {err}")
                    last_err = None
                    break
                except RuntimeError as e:
                    last_err = e
                    print(f"[capture] tier {tier} failed ({e}); trying next")
                    page.close(); ctx.close()
            if last_err:
                raise last_err

            for i in range(nframes):
                page.evaluate(f"window.__settle({i})")
                path = frames_dir / f"frame_{i:06d}.png"
                page.screenshot(path=str(path), type="png")
                if i % max(1, nframes // 10) == 0 or i == nframes - 1:
                    print(f"  frame {i + 1:4d}/{nframes}  "
                          f"(day {meta['frames'][i]['day']:.2f}, "
                          f"hour {meta['frames'][i]['hour']:.1f})")
            page.close(); ctx.close(); browser.close()

        # --- write post-processing helpers ---------------------------------
        out_meta = {
            "width": a.width, "height": a.height, "fps": a.fps,
            "frames": nframes, "frame_pattern": "frames/frame_%06d.png",
            "sim": {
                "resolution_km": a.res, "start_day": a.start_day,
                "days": a.days, "samples_per_day": a.spd,
                "col_ref": meta["col_ref"], "col_max": meta["col_max"],
                "N": meta["N"], "K": meta["K"],
            },
            "note": ("Each PNG is one simulation snapshot. Assemble with "
                     "make_movie.sh, or any ffmpeg/imagemagick pipeline."),
        }
        (out_dir / "meta.json").write_text(json.dumps(out_meta, indent=2))

        script = (
            "#!/bin/sh\n"
            "# Assemble the frame folder into an MP4 (H.264).\n"
            f'cd "$(dirname "$0")"\n'
            f"ffmpeg -y -framerate {a.fps} -i frames/frame_%06d.png "
            "-c:v libx264 -pix_fmt yuv420p -crf 18 co2_movie.mp4\n"
        )
        (out_dir / "make_movie.sh").write_text(script)
        (out_dir / "make_movie.sh").chmod(0o755)

        print(f"\n[capture] DONE  {nframes} frames -> {frames_dir}")
        print(f"[capture] meta: {out_dir / 'meta.json'}")
        print(f"[capture] make a movie:  sh {out_dir / 'make_movie.sh'}")
        exit_code = 0
    except Exception as e:  # noqa: BLE001
        print(f"\n[capture] ERROR: {e}", file=sys.stderr)
        print(f"[capture] server log: {log}", file=sys.stderr)
        if log.exists() and log.stat().st_size < 4000:
            print("---- server.log ----", file=sys.stderr)
            print(log.read_text(), file=sys.stderr)
    finally:
        if not a.keep_server:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
