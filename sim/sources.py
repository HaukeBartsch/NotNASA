"""CO2 sources: forest fires (diurnal + seasonal) and population centers (diurnal).

Each source is a Gaussian footprint on the sphere. Nodal weights are
precomputed once and normalized so they sum to 1 per source, so a source's
time-dependent strength `s(t)` (excess-ppm per second, total) distributes as
`s(t) * w_i` per node i.

Fire regions: a small base burn rate year-round plus a seasonal Gaussian
peaking at each region's dry/fire season (circular in day-of-year). Diurnal
factor peaks in the afternoon local solar time (fires flare by day, decay
by night).

Population centers: bimodal urban diurnal cycle in local solar time (README
"Daily CO2 cycle in cities") — a sharp morning commute peak (7-9), a midday
plateau/dip (11-15), a broader evening rush peak (17-20), and a low overnight
trough (23-5). No seasonal term.
"""
from __future__ import annotations

import json
import math

import numpy as np

from . import config as C

# (name, lat, lon, rel_weight, fire_season_peak_day, season_sigma_days)
FIRE_REGIONS = [
    ("Sahel / W. Africa savanna",  12.0,  15.0, 1.00, 210, 55),
    ("East Africa savanna",         2.0,  35.0, 0.60, 350, 45),
    ("Congo basin",                -3.0,  22.0, 0.70, 340, 50),
    ("Amazonia",                   -7.0, -58.0, 1.00, 250, 55),
    ("Cerrado (Brazil)",          -14.0, -48.0, 0.70, 240, 50),
    ("Pampas (Argentina)",        -32.0, -62.0, 0.40, 310, 45),
    ("Patagonia",                 -45.0, -70.0, 0.25,  40, 50),
    ("Indian subcontinent",       22.0,  79.0, 0.80, 355, 40),
    ("SE Asia peatlands",         -2.0, 112.0, 1.00, 250, 45),
    ("Australian bush",           -32.0, 137.0, 0.80, 320, 55),
    ("Siberian taiga",            60.0,  95.0, 0.90, 190, 45),
    ("Canadian boreal",           55.0, -105.0, 0.80, 230, 50),
    ("Western USA",               44.0, -118.0, 0.60, 240, 45),
    ("Mediterranean",             39.0,  21.0, 0.60, 210, 45),
    ("E. European steppe",        52.0,  40.0, 0.50, 180, 45),
    ("S. African Karoo",          -29.0,  25.0, 0.50, 220, 50),
    ("NE China",                  45.0, 125.0, 0.50, 170, 45),
    ("Iran / Middle East",        32.0,  54.0, 0.30, 150, 50),
]

# (name, lat, lon, rel_weight)
POP_CENTERS = [
    ("Shanghai",      31.2, 121.5, 1.0),
    ("Delhi",         28.6,  77.2, 0.9),
    ("Beijing",       39.9, 116.4, 0.8),
    ("Tokyo",         35.7, 139.7, 0.8),
    ("Mumbai",        19.1,  72.9, 0.7),
    ("Jakarta",       -6.2, 106.8, 0.7),
    ("Cairo",         30.0,  31.2, 0.6),
    ("Sao Paulo",    -23.5, -46.6, 0.6),
    ("Mexico City",  19.4, -99.1, 0.6),
    ("Seoul",         37.6, 127.0, 0.6),
    ("New York",      40.7, -74.0, 0.6),
    ("Lagos",          6.5,   3.4, 0.5),
    ("Los Angeles",   34.0, -118.2, 0.5),
    ("London",        51.5,  -0.1, 0.5),
    ("Moscow",        55.8,  37.6, 0.5),
    ("Osaka",         34.7, 135.5, 0.5),
    ("Karachi",       24.9,  67.0, 0.5),
    ("Manila",        14.6, 121.0, 0.5),
    ("Buenos Aires", -34.6, -58.4, 0.5),
    ("Istanbul",      41.0,  29.0, 0.4),
    ("Chicago",       41.9, -87.6, 0.4),
    ("Washington",    38.9, -77.0, 0.4),
    ("Paris",         48.9,   2.3, 0.4),

    # Asia / SE Asia
    ("Dhaka",              23.8,  90.4, 0.35),
    ("Chongqing",          29.6, 106.5, 0.35),
    ("Guangzhou",          23.1, 113.3, 0.35),
    ("Shenzhen",           22.5, 114.1, 0.35),
    ("Chengdu",            30.6, 104.1, 0.35),
    ("Tianjin",            39.1, 117.2, 0.35),
    ("Hangzhou",           30.3, 120.2, 0.35),
    ("Wuhan",              30.6, 114.3, 0.35),
    ("Xi'an",              34.3, 108.9, 0.35),
    ("Zhengzhou",          34.7, 113.6, 0.35),
    ("Kolkata",            22.6,  88.4, 0.35),
    ("Bengaluru",          13.0,  77.6, 0.35),
    ("Hyderabad",          17.4,  78.5, 0.35),
    ("Chennai",            13.1,  80.3, 0.35),
    ("Lahore",             31.5,  74.3, 0.35),
    ("Ho Chi Minh City",   10.8, 106.7, 0.35),
    ("Bangkok",            13.7, 100.5, 0.35),
    ("Tehran",             35.7,  51.4, 0.35),
    ("Nanjing",            32.1, 118.8, 0.33),
    ("Ahmedabad",          23.0,  72.6, 0.31),
    ("Qingdao",            36.1, 120.4, 0.32),
    ("Rawalpindi",         33.6,  73.1, 0.28),
    ("Hanoi",              21.0, 105.8, 0.28),
    ("Changsha",           28.2, 112.9, 0.28),
    ("Taipei",             25.0, 121.5, 0.25),
    ("Shenyang",           41.8, 123.4, 0.25),
    ("Pune",               18.5,  73.9, 0.24),
    ("Surat",              21.2,  72.8, 0.23),
    ("Faisalabad",         31.4,  73.1, 0.21),
    ("Kuala Lumpur",       3.1, 101.7, 0.21),
    ("Singapore",          1.35, 103.8, 0.21),
    ("Hefei",              31.8, 117.2, 0.21),
    # Middle East
    ("Riyadh",             24.7,  46.7, 0.28),
    ("Baghdad",            33.3,  44.4, 0.25),
    ("Jeddah",             21.5,  39.2, 0.18),
    ("Ankara",             39.9,  32.9,  0.2),
    ("Izmir",              38.4,  27.1, 0.16),
    ("Tel Aviv",           32.1,  34.8, 0.16),
    ("Dubai",              25.2,  55.3, 0.12),
    ("Kuwait City",        29.4,  48.0, 0.11),
    ("Amman",              31.9,  35.9, 0.14),
    ("Baku",               40.4,  50.2, 0.08),
    ("Tabriz",             38.1,  46.3, 0.07),
    ("Muscat",             23.6,  58.4, 0.07),
    ("Tashkent",           41.3,  69.3,  0.1),
    ("Doha",               25.3,  51.5, 0.05),
    # Europe
    ("Madrid",             40.4,  -3.7, 0.23),
    ("Berlin",             52.5,  13.4, 0.21),
    ("Barcelona",          41.4,   2.2, 0.19),
    ("Rome",               41.9,  12.5, 0.15),
    ("Milan",              45.5,   9.2, 0.12),
    ("Naples",             40.8,  14.3, 0.11),
    ("Athens",             38.0,  23.7, 0.11),
    ("Budapest",           47.5,  19.0,  0.1),
    ("Kyiv",               50.5,  30.5,  0.1),
    ("Lisbon",             38.7,  -9.1,  0.1),
    ("Prague",             50.1,  14.4, 0.07),
    ("Vienna",             48.2,  16.4, 0.07),
    ("Minsk",              53.9,  27.6, 0.07),
    ("Bucharest",          44.4,  26.1, 0.07),
    ("Brussels",           50.8,   4.4, 0.07),
    ("Belgrade",           44.8,  20.5, 0.06),
    ("Warsaw",             52.2,  21.0, 0.05),
    ("Rotterdam",          51.9,   4.5, 0.05),
    ("Amsterdam",          52.4,   4.9, 0.05),
    ("Copenhagen",         55.7,  12.6, 0.05),
    ("Sofia",              42.7,  23.3, 0.05),
    ("Helsinki",           60.2,  24.9, 0.05),
    ("Dublin",             53.3,  -6.3, 0.04),
    # Africa
    ("Kinshasa",           -4.3,  15.3, 0.35),
    ("Dar es Salaam",      -6.8,  39.3, 0.24),
    ("Johannesburg",       -26.2,  28.0, 0.21),
    ("Addis Ababa",        9.0,  38.7, 0.18),
    ("Nairobi",            -1.3,  36.8, 0.17),
    ("Cape Town",          -33.9,  18.4, 0.16),
    ("Kano",               12.0,   8.5, 0.16),
    ("Ibadan",             7.4,   3.9, 0.14),
]


FIRE_FOOTPRINT_DEG = 1.5    # Gaussian sigma for fire regions (wide, diffuse)
POP_FOOTPRINT_DEG = 0.5     # Gaussian sigma for population centers (compact)
FIRE_BASE_FRAC = 0.08       # fraction of peak strength that burns year-round
FIRE_DIURNAL_PEAK_H = 15.0  # local solar time of peak fire emission

# Urban diurnal cycle (README "Daily CO2 cycle in cities"), local solar time.
# Sum of circular Gaussians: a sharp morning spike (commute + households
# waking/cooking, 7-9), a broader evening peak (rush hour + residential
# energy, 17-20), and a subtracted overnight trough (the city sleeps,
# 23-5). The midday dip (11-15) is the baseline the two peaks sit on.
# Each term: (center_h, sigma_h, amplitude).
POP_DIURNAL_TERMS = (
    (8.0,  1.1, +0.60),    # morning peak: sharpest spike of the day
    (18.5, 2.0, +0.45),    # evening peak: second, broader
    (3.0,  2.2, -0.40),    # nighttime trough: lowest point, ~3 AM
)
POP_DIURNAL_PEAK_H = POP_DIURNAL_TERMS[0][0]   # morning peak hour (metadata)


def _wrapped_gauss_mean(sigma: float, period: float = 24.0) -> float:
    """Exact mean of exp(-0.5*wrap(x)^2/sigma^2) over one `period`.

    The wrap (to [-period/2, period/2)) truncates the Gaussian to that window,
    so the mean is the truncated mass: sigma*sqrt(2*pi)/period
    * erf(period/(2*sigma*sqrt(2))).
    """
    half = period / 2.0
    return (sigma * math.sqrt(2.0 * math.pi) / period
            * math.erf(half / (sigma * math.sqrt(2.0))))


# Baseline that makes the 24-h mean of the curve exactly 1, so the annual
# population budget (calibrated via C.POP_PEAK_SRC) is unchanged.
POP_DIURNAL_BASE = 1.0 - sum(amp * _wrapped_gauss_mean(sig)
                             for _, sig, amp in POP_DIURNAL_TERMS)


def _circular_diff(x: float, period: float = 365.0) -> float:
    """Signed circular difference, wrapped to [-period/2, period/2)."""
    return ((x + period / 2.0) % period) - period / 2.0


def _circular_gauss(d: float, peak: float, sigma: float) -> float:
    """Gaussian in circular day-of-year distance."""
    return math.exp(-0.5 * (_circular_diff(d - peak) / sigma) ** 2)


def _pop_diurnal(h_local: float) -> float:
    """Urban diurnal factor (daily mean exactly 1) from local solar time.

    Bimodal, per the README "Daily CO2 cycle in cities":
      ~8:00  sharp morning peak (commute, households waking)  -> max of day
      11-15  midday plateau/dip (traffic eases, parks absorb CO2)
      ~18:30 broader evening peak (evening rush, residential energy)
      ~3:00  overnight trough (lowest point)
    """
    s = 0.0
    for center, sigma, amp in POP_DIURNAL_TERMS:
        d = _circular_diff(h_local - center, 24.0)
        s += amp * math.exp(-0.5 * (d / sigma) ** 2)
    return POP_DIURNAL_BASE + s


def _unit_vec(lat_deg: float, lon_deg: float) -> np.ndarray:
    la = math.radians(lat_deg)
    lo = math.radians(lon_deg)
    return np.array([math.cos(la) * math.cos(lo),
                     math.cos(la) * math.sin(lo),
                     math.sin(la)])


def _footprint_weights(node_pos: np.ndarray, lat: float, lon: float,
                       sigma_deg: float) -> np.ndarray:
    """Normalized Gaussian weights over nodes. node_pos: (N,3) unit vectors."""
    center = _unit_vec(lat, lon)
    cos_d = np.clip(node_pos @ center, -1.0, 1.0)
    d = np.arccos(cos_d)                       # angular distance, rad
    sig = math.radians(sigma_deg)
    w = np.exp(-0.5 * (d / sig) ** 2)
    w /= w.sum()
    return w.astype(np.float32)


class Sources:
    """All CO2 sources with precomputed nodal footprints.

    Attributes:
        W          : (nsrc, N) float32 normalized footprints
        strengths  : (nsrc,) base strengths, excess-ppm/s total per source
        names, kind, lat, lon, season_peak : per-source metadata (viewer)
    """

    def __init__(self, mesh,
                 fire_peak: float = C.FIRE_PEAK_SRC,
                 pop_peak: float = C.POP_PEAK_SRC):
        self.names, self.kind = [], []
        self.lat, self.lon = [], []
        self.season_peak = []
        self.diurnal_peak_h = []
        self.seasonal = []                     # (frac_base, peak_day, sigma) or None
        self.W_list = []

        for name, la, lo, wt, peak, sig in FIRE_REGIONS:
            self.names.append(name)
            self.kind.append("fire")
            self.lat.append(la)
            self.lon.append(lo)
            self.season_peak.append(peak)
            self.diurnal_peak_h.append(FIRE_DIURNAL_PEAK_H)
            self.seasonal.append((FIRE_BASE_FRAC, peak, sig))
            self.W_list.append(_footprint_weights(mesh.V, la, lo, FIRE_FOOTPRINT_DEG))

        for name, la, lo, wt in POP_CENTERS:
            self.names.append(name)
            self.kind.append("pop")
            self.lat.append(la)
            self.lon.append(lo)
            self.season_peak.append(None)
            self.diurnal_peak_h.append(POP_DIURNAL_PEAK_H)
            self.seasonal.append(None)
            self.W_list.append(_footprint_weights(mesh.V, la, lo, POP_FOOTPRINT_DEG))

        self.W = np.stack(self.W_list, axis=0)                     # (nsrc, N)
        base = []
        for i, k in enumerate(self.kind):
            peak = fire_peak if k == "fire" else pop_peak
            base.append(peak * (FIRE_REGIONS[i][3] if k == "fire"
                                else POP_CENTERS[i - len(FIRE_REGIONS)][3]))
        self.strengths = np.asarray(base, dtype=np.float64)        # (nsrc,)
        self.nsrc = len(self.names)
        self.N = mesh.N

    # --- time-dependent modulation -----------------------------------------
    def _diurnal(self, i: int, hour_utc: float) -> float:
        """Diurnal factor (mean 1) using the source's local solar time."""
        lon = self.lon[i]
        h_local = (hour_utc + lon / 15.0) % 24.0
        if self.kind[i] == "fire":
            # max at local peak hour, min 12h later, mean 1
            return (1.0 + C.FIRE_DIURNAL * math.cos(
                2.0 * math.pi * (h_local - self.diurnal_peak_h[i]) / 24.0))
        # cities: bimodal urban cycle (morning + evening peaks, overnight trough)
        return _pop_diurnal(h_local)

    def _seasonal(self, i: int, day: float) -> float:
        """Seasonal factor (mean ~1) for fire regions; 1 for population."""
        s = self.seasonal[i]
        if s is None:
            return 1.0
        frac_base, peak, sigma = s
        return frac_base + (1.0 - frac_base) * _circular_gauss(day, peak, sigma)

    def active_strengths(self, day: float, hour_utc: float) -> np.ndarray:
        """Total emission per source (excess-ppm/s) at (day, hour_utc)."""
        out = np.empty(self.nsrc)
        for i in range(self.nsrc):
            out[i] = self.strengths[i] * self._diurnal(i, hour_utc) \
                     * self._seasonal(i, day)
        return out

    def emission(self, day: float, hour_utc: float) -> np.ndarray:
        """Per-node ground-level emission rate (N,) in excess-ppm/s.

        The engine adds this to its rate vector at level 0 (multiplied by
        the timestep there), so the units stay ppm/s.
        """
        s = self.active_strengths(day, hour_utc)
        return self.W.T @ s

    def to_dict(self) -> list[dict]:
        """JSON-serializable hotspot list for the viewer."""
        return [
            {
                "name": self.names[i],
                "kind": self.kind[i],
                "lat": self.lat[i],
                "lon": self.lon[i],
                "strength": float(self.strengths[i]),
                "season_peak_day": self.season_peak[i],
            }
            for i in range(self.nsrc)
        ]
