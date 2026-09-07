"""Explicit time-stepping core for the CO2 transport simulation.

Horizontal (P1 FEM on the unstructured spherical mesh, lumped mass):
  advection -- first-order upwind edge flux (mass-conserving, monotone):
        for each directed edge e of triangle t, with outward normal n:
          q   = w_t . n                        (signed outflow speed)
           F   = max(q, 0) * c_up(t, e)         (mass moved out of t)
          c_up = concentration at the most-upstream node of t
      The flux mass is deposited on the two edge nodes split in proportion
      to their lumped masses, so total mass is exactly conserved.
  diffusion -- exact P1 stiffness, explicit (row-sum-zero => conservative):
      K_pq = -K_eff (l_rp^2 + l_rq^2 - l_pq^2) / (8 A)   for edge (p,q) opp. r

Vertical (finite volume on the 16 geometric levels, closed lid at 100 km):
    upwinded advection with the idealized w field + central diffusion.
    Time: explicit Euler with CFL guards.

Units: the mesh is on the *unit* sphere, so wind is scaled by 1/R and
diffusivity by 1/R^2; the vertical column keeps SI units (m, m/s).
"""

from __future__ import annotations

import numpy as np
from scipy import sparse as _sp

from . import config as C
from . import wind
from .mesh import Mesh


class Engine:
    def __init__(self, mesh: Mesh, sources):
        self.mesh = mesh
        self.sources = sources
        self.N = mesh.N
        self.K = C.N_VLEVELS
        self.DT = C.DT_S
        R_M = C.R_KM * 1000.0
        self.R_M = R_M
        # The vertex-upwind advection below is monotone and mass-conserving but
        # over-advects a plume by ~1.62x on this unstructured mesh (the shared
        # downwind edge nodes are re-advected by neighbouring cells in the same
        # step). Divide the edge flux by that factor so a plume advects at the
        # physical wind speed. Resolution-independent (verified at r=100/250/500).
        self.ADV_SPEED = 1.0 / 1.62

        V = mesh.V
        T = mesh.T
        self.T = T
        # --- horizontal unit vectors at nodes (for wind -> 3D) -------------
        lat = np.deg2rad(mesh.lat)
        lon = np.deg2rad(mesh.lon)
        self.EAST = wind.east_unit(lat, lon)                    # (N,3)
        self.NORTH = wind.north_unit(lat, lon)                  # (N,3)

        # --- advection edge table ------------------------------------------
        self._build_advection()

        # --- diffusion stiffness (exact P1) --------------------------------
        self._build_diffusion(Keff=C.K_H / R_M**2)

        # --- vertical FVM coefficients -------------------------------------
        z = np.asarray(C.V_LEVELS_KM)                            # (17,) km
        self.DZ = np.diff(z) * 1000.0                            # (16,) m
        # 15 internal interfaces at the level boundaries z[1..15];
        # w is damped to 0 across the ABL (~1 km), not 40 km -- the old 40 km
        # scale suppressed w to <50% all the way up to ~28 km and stranded
        # surface emissions in the bottom cell
        self.WIF_SCALE = np.tanh(z[1:-1] / 1.0)                  # (15,)
        # --- vertical transport: ONE shared explicit-CFL budget --------------
        # The column update is c_k^{n+1} = c_k + DT/DZ_k * (inflow - outflow).
        # Monotonicity (c^{n+1} >= 0 for c >= 0) requires the SELF coefficient
        # to stay >= 0, i.e. for every cell k:
        #     DT/DZ_k * (|w_out|_k + A_IF[k] + A_IF[k-1]) <= 1
        # where |w_out|_k is the outward advection speed at the interface cell
        # k loses through. Advection (upwind) and diffusion (central) each
        # remove mass from a cell, so their CFL numbers ADD. Capping them
        # separately (old 0.9 + 0.5 = 1.4 > 1) let the sum exceed 1, creating
        # negative values that the positivity clip then "repairs" by adding
        # mass. We split one shared budget (0.9, a 10% margin under the limit):
        # diffusion gets 0.5 (the classical explicit-Euler limit), advection
        # the remaining 0.4.
        v_budget = 0.9
        v_diff = 0.5
        v_adv = v_budget - v_diff                                 # = 0.4
        # advection: |w|*DT <= v_adv * DZ of the thinner (lower) cell
        self.WIF_CAP = v_adv * self.DZ[:-1] / self.DT             # (15,) m/s
        self.IF_DZ = (self.DZ[:-1] + self.DZ[1:]) / 2.0          # (15,) m
        # diffusion: per-interface coefficient A_IF[k] = K/IF_DZ, capped so the
        # explicit scheme stays monotone. The classical explicit-Euler
        # diffusion limit is DT*(A_IF[k] + A_IF[k-1])/DZ[k] <= 0.5 (per cell);
        # a higher cap is unstable on this non-uniform grid (the bottom
        # 2-cell subsystem has spectral radius 1.14 at 0.8, <= 0.77 at 0.5):
        #   DT * (A_IF[k] + A_IF[k-1]) / DZ[k] <= v_diff   (A_IF[-1] = A_IF[15] = 0)
        n_if = self.DZ.size - 1
        vtarget = v_diff
        A_IF = np.empty(n_if)
        A_IF[0] = min(C.K_V / self.IF_DZ[0], vtarget * self.DZ[0] / self.DT)
        for k in range(1, n_if):
            cap = vtarget * self.DZ[k] / self.DT - A_IF[k - 1]
            A_IF[k] = min(C.K_V / self.IF_DZ[k], max(cap, 0.0))
        self.A_IF = A_IF                                         # (15,) m/s
        self.K_IF = A_IF * self.IF_DZ                             # (15,) m^2/s (reporting)

        # --- scatter matrices for the FIXED edge endpoints (fast sparse @) ---
        E = self.a.size
        idx = np.arange(E)
        self.S_A = _sp.csr_matrix((np.ones(E), (self.a, idx)), shape=(self.N, E)).tocsr()
        self.S_B = _sp.csr_matrix((np.ones(E), (self.b, idx)), shape=(self.N, E)).tocsr()
        Ed = self.Da.size
        idd = np.arange(Ed)
        self.S_DA = _sp.csr_matrix((np.ones(Ed), (self.Da, idd)), shape=(self.N, Ed))
        self.S_DB = _sp.csr_matrix((np.ones(Ed), (self.Db, idd)), shape=(self.N, Ed))

        # --- CFL report ------------------------------------------------------
        self.cfl = self._cfl()

    # ------------------------------------------------------------------ build
    def _build_advection(self):
        V, T = self.mesh.V, self.mesh.T
        F = T.shape[0]
        a0, a1, a2 = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
        fn = np.cross(a1 - a0, a2 - a0)                          # outward if CCW
        fn = fn / (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-30)
        cent = (a0 + a1 + a2) / 3.0
        fn *= np.sign(np.einsum("ij,ij->i", fn, cent))[:, None]  # force outward

        # 3 directed edges per triangle: (0->1), (1->2), (2->0)
        pairs = np.stack([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]], axis=1)  # (F,3,2)
        dirs = np.stack([a1 - a0, a2 - a1, a0 - a2], axis=1)           # (F,3,3)
        n = np.cross(dirs, fn[:, None, :])                 # cross(dir, fn): outward
        n = n / (np.linalg.norm(n, axis=2, keepdims=True) + 1e-30)

        E = 3 * F
        self.owner = np.repeat(np.arange(F), 3)             # (E,)
        self.a = pairs[:, :, 0].reshape(-1)                 # (E,)
        self.b = pairs[:, :, 1].reshape(-1)                 # (E,)
        self.EN = n.reshape(E, 3)                           # (E,3) outward edge normals
        self.L = np.linalg.norm(dirs, axis=2).reshape(-1)   # (E,) chord lengths
        self.P = V[T][:, :, :]                              # (F,3,3) node coords
        self.PE = self.P[self.owner]                        # (E,3,3) node coords per directed edge
        self.TRI_NODES = np.repeat(T, 3, axis=0)            # (E,3)

    def _build_diffusion(self, Keff: float):
        V, T = self.mesh.V, self.mesh.T
        F = T.shape[0]
        N = self.N
        a0, a1, a2 = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
        A = 0.5 * np.linalg.norm(np.cross(a1 - a0, a2 - a0), axis=1)
        l01 = np.linalg.norm(a1 - a0, axis=1)
        l12 = np.linalg.norm(a2 - a1, axis=1)
        l02 = np.linalg.norm(a2 - a0, axis=1)
        # off-diagonal stiffness per local edge (opposite-node formula)
        K01 = -Keff * (l02**2 + l12**2 - l01**2) / (8.0 * A)
        K12 = -Keff * (l01**2 + l02**2 - l12**2) / (8.0 * A)
        K02 = -Keff * (l01**2 + l12**2 - l02**2) / (8.0 * A)

        Da, Db, Dg = [], [], []
        for (u, v), entries in self.mesh.edge_map.items():
            g = 0.0
            for (t, _opp) in entries:
                t0, t1, t2 = T[t]
                if {t0, t1} == {u, v}:
                    g += K01[t]
                elif {t1, t2} == {u, v}:
                    g += K12[t]
                else:
                    g += K02[t]
            Da.append(u)
            Db.append(v)
            Dg.append(g)
        self.Da = np.asarray(Da, dtype=np.int64)
        self.Db = np.asarray(Db, dtype=np.int64)
        self.Dg = np.asarray(Dg, dtype=np.float64)
        # exact stiffness has zero row sums -> diagonal = -sum of off-diagonals
        diag = np.zeros(N)
        np.add.at(diag, self.Da, -self.Dg)
        np.add.at(diag, self.Db, -self.Dg)
        self.diag = diag

    def _cfl(self) -> dict:
        u_max = 20.0  # conservative bound on |wind| (m/s)
        h_min = float(self.L.min())
        cfl_adv = self.DT * (u_max / self.R_M) / h_min
        # true bound on |w_z| over a (day, hour) sample
        w_max = 0.0
        for h in (2.0, 14.0):
            for d in range(15, 366, 15):
                w_max = max(w_max, float(np.abs(wind.w_z(self.mesh.lat, h, d)).max()))
        # the actual vertical flux is capped by WIF_CAP, so use the capped value
        cfl_v_adv = float(np.max(self.DT * np.minimum(w_max * self.WIF_SCALE, self.WIF_CAP) / self.DZ[:-1]))
        # exact explicit-diffusion stability number: per-cell diagonal rate
        below = np.concatenate(([0.0], self.A_IF))                 # (16,) into cell k from below
        above = np.concatenate((self.A_IF, [0.0]))                 # (16,) into cell k from above
        cfl_v_diff = float(np.max(self.DT * (below + above) / self.DZ))
        # horizontal diffusion: max diagonal rate * DT / 2
        cfl_d = float(np.max(np.abs(self.diag) / self.mesh.M)) * self.DT / 2.0
        # vertical advection and diffusion both drain a cell, so their CFL
        # numbers ADD (that is the monotonicity limit we enforce in __init__).
        # Report the sum, not the max, so this number is the real constraint.
        return {
            "advection": cfl_adv,
            "vertical": cfl_v_adv + cfl_v_diff,
            "diffusion": cfl_d,
        }

    # -------------------------------------------------------------------- step
    def step(self, c: np.ndarray, day: float, hour_utc: float) -> None:
        """Advance c (N, K) excess-ppm field one timestep, in place."""
        N, K = c.shape

        # --- wind at nodes, then triangle centroids (unit-sphere / s) -------
        u, v = wind.wind_uv(self.mesh.lat, self.mesh.lon, hour_utc, day)
        Wn = u[:, None] * self.EAST + v[:, None] * self.NORTH          # (N,3) m/s
        Wc = (Wn[self.T[:, 0]] + Wn[self.T[:, 1]] + Wn[self.T[:, 2]]) / 3.0
        We = Wc[self.owner] / self.R_M                                 # (E,3) unit-sphere/s

        # --- advection: upwind edge flux ------------------------------------
        # `rate` is a MASS rate (concentration * area / s); the final update
        # divides by the lumped mass M, so every term below must be in mass units.
        q = np.einsum("ij,ij->i", We, self.EN)                          # (E,) signed outflow speed
        D = np.einsum("eij,ej->ei", self.PE, We)                        # (E,3)
        k_up = self.TRI_NODES[np.arange(q.shape[0]), D.argmin(axis=1)]  # (E,) upwind node
        # flux = (U.n) * edge_length * c_upwind  ->  concentration * area / s.
        # The vertex-upwind transfer below is monotone and exactly mass-
        # conserving, but on this unstructured mesh it advects a plume ~1.6x
        # faster than the wind (the shared downwind edge nodes are re-advected
        # by neighbouring cells within the same step). ADV_SPEED calibrates the
        # flux to the physical wind speed; see test_advection_drifts_downwind.
        F = np.where(q[:, None] > 0.0, self.ADV_SPEED * q[:, None] * self.L[:, None] * c[k_up], 0.0)  # (E,K)
        Mab = self.mesh.M[self.a] + self.mesh.M[self.b]                # (E,)
        # deposit the flux mass on the two downwind edge nodes in proportion
        # to their lumped masses, and remove it from the upwind node
        rate = self.S_A @ (F * (self.mesh.M[self.a] / Mab)[:, None]) \
             + self.S_B @ (F * (self.mesh.M[self.b] / Mab)[:, None])    # (N,K)
        for k in range(K):
            rate[:, k] -= np.bincount(k_up, weights=F[:, k], minlength=N)

        # --- diffusion --------------------------------------------------------
        # Dg are the (negative) P1 stiffness off-diagonals and diag = -sum of
        # off-diagonals, so the row sums are zero: (Kc)_i = diag_i c_i + sum_e Dg_e c_j
        rate += self.diag[:, None] * c
        rate += self.S_DA @ (self.Dg[:, None] * c[self.Da])
        rate += self.S_DB @ (self.Dg[:, None] * c[self.Db])

        # --- vertical: FVM between the 16 levels (closed lid at top) --------
        # interface fluxes Fup are concentration * m / s PER UNIT AREA; as a
        # mass rate they scale with the lumped area M (dc/dt = F/dz restored
        # by the final /M below).
        C0 = c[:, :-1]                                                  # (N,15)
        C1 = c[:, 1:]                                                   # (N,15)
        Wz = wind.w_z(self.mesh.lat, hour_utc, day)                     # (N,)
        wif = Wz[:, None] * self.WIF_SCALE                              # (N,15)
        wif = np.clip(wif, -self.WIF_CAP[None, :], self.WIF_CAP[None, :])  # CFL guard
        Fup = wif * np.where(wif > 0.0, C0, C1)                         # upwinded advection
        Fup -= self.A_IF[None, :] * (C1 - C0)                           # diffusion (Fick: high->low)
        num = np.zeros((N, K))
        num[:, 1:] += Fup                                               # in from below
        num[:, :-1] -= Fup                                              # out above
        rate += self.mesh.M[:, None] * num / self.DZ[None, :]

        # --- sources (ground level) ------------------------------------------
        # emission is a concentration rate (ppm/s) -> mass rate = M * emission
        rate[:, 0] += self.mesh.M * self.sources.emission(day, hour_utc)  # (N,)

        # --- update: explicit Euler + background sink + positivity clip ------
        c += self.DT * rate / self.mesh.M[:, None]
        c *= 1.0 - self.DT / C.TAU_S
        np.clip(c, 0.0, C.C_CAP, out=c)

    # ------------------------------------------------------------------ utils
    def mass(self, c: np.ndarray) -> float:
        """Total excess-CO2 mass (concentration x cell volume), for balance checks."""
        return float((c * self.mesh.M[:, None] * self.DZ[None, :]).sum())