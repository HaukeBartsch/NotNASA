"""Render the default (shell) viewer at the FULL-YEAR movie col_ref (1124.8),
at a couple of (day,hour) near the Sahel fire peak, to confirm West Sahara and
the continuous field hold up under the real normalization."""
import os, sys, json, time, socket, subprocess, urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from playwright.sync_api import sync_playwright

def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p
def wait_http(url, deadline=300):
    t0 = time.time()
    while time.time() < t0 + deadline:
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                if r.status == 200: return
        except Exception: pass
        time.sleep(1)
    raise RuntimeError("server not ready")

MOVIE_REF = 1124.837158203125

def main():
    port = free_port(); base = f"http://127.0.0.1:{port}"
    env = dict(os.environ, RES_KM="250", START_DAY="170", DAYS="42", SPD="48",
               WIDTH="1280", HEIGHT="720", OUT_DIR=str(ROOT/"output"))
    srv = subprocess.Popen([str(ROOT/".venv/bin/python"), "-m", "uvicorn", "server.app:app",
                            "--host","127.0.0.1","--port",str(port),"--log-level","warning"],
                           cwd=str(ROOT), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    try:
        wait_http(f"{base}/api/meta")
        with urllib.request.urlopen(f"{base}/api/meta", timeout=30) as r:
            meta = json.load(r)
        M = meta["M"]; spd = meta["samples_per_day"]
        def idx(day, hour):
            return max(0, min(M-1, int(round((day - meta["start_day"])*spd + hour/24*spd))))
        shots = [("peak_d210_h15", 210, 15), ("peak_d210_h6", 210, 6),
                 ("peak_d205_h18", 205, 18), ("peak_d215_h9", 215, 9)]
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True, args=["--enable-unsafe-swiftshader","--hide-scrollbars"])
            ctx = browser.new_context(viewport={"width":1280,"height":720}, device_scale_factor=1)
            page = ctx.new_page(); page.set_default_timeout(120000)
            page.goto(base + f"?ref={MOVIE_REF}", wait_until="load", timeout=60000)   # default = shell now
            page.wait_for_function("window.__ready === true", timeout=60000)
            for name, day, hour in shots:
                i = idx(day, hour)
                page.evaluate(f"window.__settle({i})")
                out = ROOT/"output"/f"final_{name}.png"
                page.screenshot(path=str(out), type="png")
                print(f"  {name:16s} day={day} h={hour:2d} idx={i} -> {out.name}")
            page.close(); ctx.close(); browser.close()
    finally:
        srv.terminate()
        try: srv.wait(timeout=10)
        except Exception: srv.kill()

if __name__ == "__main__":
    main()
