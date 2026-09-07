"""Idealized global wind field.

Single source of truth for the wind, used by both the Python sim and (ported
verbatim) the JS viewer. Returns (u, v) in m/s: u east(+), v north(+).

Symbols (from the plan):
    phi  latitude, deg, N+
    lam  longitude, deg, E+
    h    hour, 0-24
    d    day, 1-365
    G(x;c,s) = exp(-((x-c)/s)^2)
    phi_sun(d) = 23.44 * sin(2*pi*(d-80)/365)   (subsolar latitude)
All belt centers shift +0.35*phi_sun seasonally.

The dominant signal is the zonal belt structure (easterly trades, mid-latitude
westerlies, polar easterlies) so that plumes drift west->east in the mid-latitudes
and east->west in the tropics, in both hemispheres, as the README requires.
"""
from __future__ import annotations

import math
import numpy as np

TWO_PI = 2.0 * math.pi


def _G(x, c, s):
    return np.exp(-(((x - c) / s) ** 2))


def subsolar_lat(d):
    """Subsolar latitude (deg). d: day 1..365 (scalar or array)."""
    return 23.44 * np.sin(TWO_PI * (d - 80.0) / 365.0)


def wind_uv(phi, lam, h, d):
    """Zonal (u) and meridional (v) wind in m/s at points (phi, lam).

    phi, lam : deg, broadcastable arrays
    h, d     : scalar hour (0-24) and day (1-365)
    Returns (u, v) arrays of shape broadcast(phi, lam).
    """
    phi = np.asarray(phi, dtype=np.float64)
    lam = np.asarray(lam, dtype=np.float64)
    phi = np.broadcast_to(phi, np.broadcast(phi, lam).shape)
    lam = np.broadcast_to(lam, phi.shape)

    ps = subsolar_lat(d)                 # seasonal shift
    s = 0.35 * ps

    # --- zonal u (east +) ---
    u = (
        -8.0 * (_G(phi, 8.0 + s, 10.0) + _G(phi, -8.0 - s, 10.0))          # easterly trades
        + 12.0 * (_G(phi, 45.0 + s, 12.0) + _G(phi, -45.0 - s, 12.0))      # westerlies
        - 6.0 * (_G(phi, 75.0 + s, 12.0) + _G(phi, -75.0 - s, 12.0))       # polar easterlies
        + 2.0 * _G(phi, 45.0 + s, 15.0) * np.cos(2.0 * (lam - 120.0))      # wave-2 asymmetry
        + 1.5 * (_G(phi, 25.0, 20.0) + _G(phi, -25.0, 20.0))
          * np.cos(TWO_PI * (h - 14.0) / 24.0)                              # diurnal
    )

    # --- meridional v (north +) ---
    v = (
        -4.0 * (_G(phi, 25.0 + s, 12.0) + _G(phi, -25.0 - s, 12.0))        # equatorward subtropics
        + 4.0 * (_G(phi, 50.0 + s, 14.0) + _G(phi, -50.0 - s, 14.0))       # poleward mid-lat
        + 2.0 * _G(phi, 0.0, 20.0) * np.cos(2.0 * (lam - 60.0))            # equatorial wave
        + 1.0 * (_G(phi, 25.0, 20.0) + _G(phi, -25.0, 20.0))
          * np.sin(TWO_PI * (h - 14.0) / 24.0)                             # diurnal
    )
    return u, v


def east_unit(lat_rad, lon_rad):
    """Unit east vector at (lat, lon) in ENU-on-sphere coordinates (x=cos lat cos lon, ...)."""
    return np.stack([
        -np.sin(lon_rad),
         np.cos(lon_rad),
         np.zeros_like(lat_rad),
    ], axis=-1)


def north_unit(lat_rad, lon_rad):
    cl = np.cos(lat_rad)
    sl = np.sin(lat_rad)
    return np.stack([
        -sl * np.cos(lon_rad),
        -sl * np.sin(lon_rad),
         cl,
    ], axis=-1)


_easting = east_unit
_northing = north_unit


def wind_3d(pos, lat_deg, lon_deg, h, d):
    """3D tangent wind vector (Nx3) at unit-sphere node positions `pos`.

    pos, lat_deg, lon_deg : (N,3)/(N,)/(N,)
    Returns (N,3) wind vector in m/s, tangent to the sphere.
    """
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    u, v = wind_uv(lat_deg, lon_deg, h, d)
    u = np.asarray(u, dtype=np.float64).reshape(-1)
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    W = u[..., None] * _easting(lat, lon) + v[..., None] * _northing(lat, lon)
    return W


def _w_raw(phi, h, d):
    """Zonal-mean vertical velocity (m/s, up+), unbalanced. See `w_z`."""
    ps = subsolar_lat(d)
    s = 0.35 * ps
    return (
        +2.5 * _G(phi, 5.0 + 0.5 * ps, 10.0)                          # ITCZ upwelling
        - 1.5 * (_G(phi, 25.0 + s, 12.0) + _G(phi, -25.0 - s, 12.0))  # subtrop. subsidence
        + 1.5 * (_G(phi, 55.0 + s, 14.0) + _G(phi, -55.0 - s, 14.0))  # mid-lat ascent
        - 1.0 * (_G(phi, 80.0, 10.0) + _G(phi, -80.0, 10.0))         # polar subsidence
        + 1.5 * (_G(phi, 15.0, 20.0) + _G(phi, -15.0, 20.0))
          * np.cos(TWO_PI * (h - 14.0) / 24.0)                        # diurnal convection
    )


# Fixed latitude grid for the area-weighted mean (cos-weighted).
_W_MEAN_GRID = np.radians(np.linspace(-90.0, 90.0, 361))
_W_MEAN_W = np.cos(_W_MEAN_GRID)
_W_MEAN_W = _W_MEAN_W / _W_MEAN_W.sum()


def w_z(phi, h, d):
    """Idealized large-scale vertical velocity (m/s, up+), zonal-mean structure.

    Hadley/Ferrel circulation pattern (belt centers shift with the sun as in
    `wind_uv`): ITCZ upwelling, subtropical subsidence, mid-latitude ascent,
    polar subsidence, plus a diurnal convection pulse in the tropics.
    The area-weighted latitude mean is removed so the field is balanced
    (zero net vertical mass flux on the sphere).
    """
    phi = np.asarray(phi, dtype=np.float64)
    mean = _W_MEAN_W @ _w_raw(_W_MEAN_GRID, h, d)
    return _w_raw(phi, h, d) - mean
