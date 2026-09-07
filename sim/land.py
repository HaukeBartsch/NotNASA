"""Rasterized land mask from the Natural Earth GeoJSON.

Provides:
    land_mask   : 2D bool array (nlat, nlon), True = land
    is_land_at(lat, lon) : nearest-neighbor land query (deg)
    land_weight(lat, lon): smooth [0,1] land weight (Gaussian-blurred) for sources
"""
from __future__ import annotations

import json
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

HERE = os.path.dirname(os.path.abspath(__file__))
GEOJSON = os.path.join(HERE, "..", "data", "ne_110m_land.geojson")

# Raster grid resolution (lon x lat). nlon=720, nlat=360 -> 0.5 deg cells.
NLON = 720
NLAT = 360


def _load_polygons(path=GEOJSON):
    with open(path) as f:
        gj = json.load(f)
    polys = []
    for feat in gj["features"]:
        geom = feat["geometry"]
        if geom["type"] == "Polygon":
            polys.append(geom["coordinates"])
        elif geom["type"] == "MultiPolygon":
            polys.extend(geom["coordinates"])
    return polys


def _lonlat_to_px(lon, lat, nlon, nlat):
    x = (lon + 180.0) / 360.0 * nlon
    y = (90.0 - lat) / 180.0 * nlat
    return x, y


def rasterize(nlon=NLON, nlat=NLAT, path=GEOJSON):
    """Return a bool land mask of shape (nlat, nlon)."""
    img = Image.new("L", (nlon, nlat), 0)
    dr = ImageDraw.Draw(img)
    for poly in _load_polygons(path):
        outer = poly[0]
        pts = [(_lonlat_to_px(lo, la, nlon, nlat)) for (lo, la) in outer]
        dr.polygon(pts, fill=255)
        for hole in poly[1:]:
            hpts = [(_lonlat_to_px(lo, la, nlon, nlat)) for (lo, la) in hole]
            dr.polygon(hpts, fill=0)
    mask = np.array(img, dtype=np.uint8) > 0          # (nlat, nlon)
    return mask


def _latlon_to_idx(lat, lon, nlat, nlon):
    j = np.clip((lon + 180.0) / 360.0 * nlon, 0, nlon - 1).astype(int)
    i = np.clip((90.0 - lat) / 180.0 * nlat, 0, nlat - 1).astype(int)
    return i, j


def smooth_weight(mask, sigma_px=4.0):
    """Gaussian-blur the bool mask into a smooth [0,1] float weight (nlat, nlon)."""
    img = Image.fromarray((mask.astype(np.float32) * 255).astype(np.uint8))
    img = img.filter(ImageFilter.GaussianBlur(radius=float(sigma_px)))
    w = np.array(img, dtype=np.float32) / 255.0
    return w


class LandMask:
    def __init__(self, nlon=NLON, nlat=NLAT, path=GEOJSON, blur_px=4.0):
        self.nlon = nlon
        self.nlat = nlat
        self.mask = rasterize(nlon, nlat, path)          # bool (nlat, nlon)
        self.weight = smooth_weight(self.mask, blur_px)  # float (nlat, nlon)

    def is_land(self, lat, lon):
        lat = np.atleast_1d(np.asarray(lat, dtype=float))
        lon = np.atleast_1d(np.asarray(lon, dtype=float))
        i, j = _latlon_to_idx(lat, lon, self.nlat, self.nlon)
        return self.mask[i, j]

    def weight_at(self, lat, lon):
        lat = np.atleast_1d(np.asarray(lat, dtype=float))
        lon = np.atleast_1d(np.asarray(lon, dtype=float))
        i, j = _latlon_to_idx(lat, lon, self.nlat, self.nlon)
        return self.weight[i, j]


if __name__ == "__main__":
    lm = LandMask()
    print("land fraction:", float(lm.mask.mean()))
    print("is_land (London 51.5,-0.12):", lm.is_land(51.5, -0.12))
    print("is_land (Atlantic 40,-30):", lm.is_land(40.0, -30.0))
