"""Tests for sim.sources diurnal modulation.

Verifies the urban diurnal cycle matches the README "Daily CO2 cycle in
cities" section: sharp morning peak (7-9), midday plateau/dip (11-15),
broader evening peak (17-20), overnight trough (23-5), with the daily mean
staying exactly 1 (so the annual population budget is unchanged).

Run with the project venv:
    .venv/bin/python -m pytest tests/test_sources.py -q
or without pytest:
    .venv/bin/python tests/test_sources.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sim import config as C
from sim import sources as srcmod


def _grid(n: int = 24 * 60):
    """(hours 0..24 exclusive, curve values) on a fine uniform grid."""
    h = np.linspace(0.0, 24.0, n, endpoint=False)
    return h, np.array([srcmod._pop_diurnal(x) for x in h])


# ---------------------------------------------------------------------------
# 1. Budget: daily mean is exactly 1
# ---------------------------------------------------------------------------

def test_pop_diurnal_mean_is_one():
    """24-h mean of the urban diurnal factor is 1 (budget preserved)."""
    h, f = _grid()
    mean = float(f.mean())
    assert abs(mean - 1.0) < 1e-9, f"diurnal mean {mean} != 1"


def test_pop_diurnal_positive_everywhere():
    h, f = _grid()
    assert f.min() > 0.0, f"diurnal factor goes non-positive: min {f.min()}"


# ---------------------------------------------------------------------------
# 2. Shape: the README's four features
# ---------------------------------------------------------------------------

def test_morning_peak_is_highest_and_sharpest():
    """Morning peak (~8:00) is the max of the day and sharper than evening."""
    h, f = _grid()
    argmax = h[int(np.argmax(f))]
    assert abs(argmax - 8.0) < 0.25, f"morning peak at {argmax}, expected ~8.0"

    def width_at(center):
        """Full width of the peak above (day-max + baseline)/2, in hours."""
        level = (float(f.max()) + srcmod.POP_DIURNAL_BASE) / 2.0
        # scan outward from the center to where the curve drops below `level`
        n = len(h)
        i0 = int(np.argmin(np.abs(h - center)))
        lo = i0
        while lo > i0 - n and f[(lo - 1) % n] > level:
            lo -= 1
        hi = i0
        while hi < i0 + n and f[(hi + 1) % n] > level:
            hi += 1
        return (hi - lo) * (24.0 / n)

    w_morning = width_at(8.0)
    w_evening = width_at(18.5)
    assert w_morning < w_evening, (
        f"morning peak should be sharper: w_m={w_morning:.2f}h vs "
        f"w_e={w_evening:.2f}h")


def test_evening_peak_is_broader_secondary_peak():
    """Evening peak (~18:30, window 17-20) is a local max, below the morning's."""
    h, f = _grid()
    f_at = lambda x: srcmod._pop_diurnal(x % 24.0)
    assert f_at(18.5) > f_at(17.0) and f_at(18.5) > f_at(20.0), (
        f"evening peak not inside 17-20 window: 17h={f_at(17.0):.3f} "
        f"18.5h={f_at(18.5):.3f} 20h={f_at(20.0):.3f}")
    assert f_at(18.5) < f_at(8.0), (
        f"morning should be the sharpest/highest spike: "
        f"8h={f_at(8.0):.3f} vs 18.5h={f_at(18.5):.3f}")


def test_midday_dip_is_plateau_between_peaks():
    """Midday (11-15) is a plateau/dip: below both peaks, above the trough."""
    f_at = lambda x: srcmod._pop_diurnal(x % 24.0)
    midday = max(f_at(11.0), f_at(13.0), f_at(15.0))
    assert midday < f_at(8.0), f"midday {midday:.3f} should dip below morning peak"
    assert midday < f_at(18.5), f"midday {midday:.3f} should dip below evening peak"
    assert midday > f_at(3.0), (
        f"midday {midday:.3f} should stay above the overnight trough "
        f"{f_at(3.0):.3f}")


def test_overnight_trough_is_global_min_around_3am():
    """Nighttime trough (23-5) holds the day's minimum, near 3 AM."""
    h, f = _grid()
    argmin = h[int(np.argmin(f))]
    assert 23.0 <= (argmin + 1.0) % 24.0 or (argmin + 1.0) % 24.0 <= 5.0, \
        f"trough at {argmin}, expected inside 23-5 window"
    assert abs(argmin - 3.0) < 0.5, f"trough at {argmin}, expected ~3.0"
    f_at = lambda x: srcmod._pop_diurnal(x % 24.0)
    assert f_at(3.0) < f_at(13.0) < f_at(18.5) < f_at(8.0), (
        f"ordering trough<midday<evening<morning violated: "
        f"3h={f_at(3.0):.3f} 13h={f_at(13.0):.3f} "
        f"18.5h={f_at(18.5):.3f} 8h={f_at(8.0):.3f}")


# ---------------------------------------------------------------------------
# 3. Integration: Sources dispatches cities to the urban curve
# ---------------------------------------------------------------------------

def test_sources_dispatch_and_fire_unchanged():
    """Cities use the bimodal curve at local solar time; fires keep the
    afternoon cosine (fires flare by day, decay by night)."""
    from sim.mesh import Mesh
    mesh = Mesh(250.0, refine_land=False)
    src = srcmod.Sources(mesh)

    # first population center
    i_pop = len(srcmod.FIRE_REGIONS)
    name, la, lo, wt = srcmod.POP_CENTERS[0]
    # hour at which the city's local solar time is 8:00 (morning peak)
    h_peak = (8.0 - lo / 15.0) % 24.0
    v_peak = src._diurnal(i_pop, h_peak)
    v_trough = src._diurnal(i_pop, (3.0 - lo / 15.0) % 24.0)
    assert abs(v_peak - srcmod._pop_diurnal(8.0)) < 1e-9
    assert abs(v_trough - srcmod._pop_diurnal(3.0)) < 1e-9
    assert v_peak > v_trough

    # first fire region: cosine peaking at 15:00 local, unchanged
    i_fire = 0
    lo_f = srcmod.FIRE_REGIONS[0][2]
    h_fire_peak = (15.0 - lo_f / 15.0) % 24.0
    v = src._diurnal(i_fire, h_fire_peak)
    assert abs(v - (1.0 + C.FIRE_DIURNAL)) < 1e-12, f"fire peak {v}"
    v_low = src._diurnal(i_fire, (3.0 - lo_f / 15.0) % 24.0)
    assert v_low < v


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
