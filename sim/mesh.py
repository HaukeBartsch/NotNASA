"""
Conforming unstructured triangular mesh of the sphere.
We use an icosahedral subdivision (each base edge split into `n` segments),
then relax it with Lloyd's algorithm (centroidal Voronoi tessellation) to
equalize edge lengths and smooth the 12 pentagonal points -- see the README's
"Pole issue in rendering". Rationale for the icosahedral base: it is conforming
by construction, has uniform cell size (no lat/lon pole slivers, which would
break the advection CFL at high latitude), and its element count 20*n^2 matches
the plan's targets (r=100 -> ~45k nodes, r=250 -> ~6.2k, r=1000 -> ~500).

The mesh stores everything the FEM engine needs precomputed:
  V     : (N,3) unit-sphere node positions
  T     : (F,3) triangle vertex indices (CCW from outside)
  lat,lon: (N,) node latitude/longitude in degrees
  relaxation: before/after stats from the Lloyd step (None if disabled)
  lumped mass M_i, diffusion conductances, and the edge table
            (a, b, k_a, k_b, normal) used by upwind advection.
"""
from __future__ import annotations

import math
import numpy as np
from scipy import sparse as _sp
from scipy.spatial import ConvexHull

from . import config as C

R_KM = C.R_KM
PHI = (1.0 + math.sqrt(5.0)) / 2.0


def _icosahedron_edge_km() -> float:
    """Return the edge length (km) of the base icosahedron on Earth."""
    # Vertices are (0, ±1, ±φ) normalised; edge length = 2/√(1+φ²)
    edge = 2.0 / math.sqrt(1.0 + PHI**2)
    return edge * R_KM


def _icosahedron():
    """Return 12 unit-sphere vertices and 20 outward-oriented faces."""
    # Standard icosahedron: (0, ±1, ±φ), (±1, ±φ, 0), (±φ, 0, ±φ),
    # with all 12 sign combinations so every vertex lies on the unit sphere.
    t = PHI
    raw = [
        ( 0,  1,  t), ( 0, -1,  t), ( 0,  1, -t), ( 0, -1, -t),
        ( 1,  t,  0), (-1,  t,  0), ( 1, -t,  0), (-1, -t,  0),
        ( t,  0,  1), ( t,  0, -1), (-t,  0,  1), (-t,  0, -1),
    ]
    v = np.array(raw, dtype=np.float64)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)

    # adjacency via the (uniform) edge length = the minimum pairwise distance
    d = np.linalg.norm(v[:, None, :] - v[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    edge_len = d.min()
    adj = d < edge_len * 1.001

    faces = []
    idx = np.arange(12)
    for i in idx:
        for j in idx:
            for k in idx:
                if i < j < k and adj[i, j] and adj[i, k] and adj[j, k]:
                    faces.append([int(i), int(j), int(k)])
    assert len(faces) == 20, f"icosahedron should have 20 faces, got {len(faces)}"

    # Orient each face so its normal points outward (away from center).
    out = []
    for f in faces:
        a, b, c = v[f[0]], v[f[1]], v[f[2]]
        n = np.cross(b - a, c - a)
        centroid = (a + b + c) / 3.0
        if np.dot(n, centroid) < 0:
            f = [f[0], f[2], f[1]]
        out.append(f)
    return v, out


def _subdivide(v, faces, n):
    """Subdivide each base edge into n segments; return (nodes, faces).

    Uses a global position-keyed map so nodes at shared edges / vertices
    are merged across base-face boundaries, producing a conforming mesh.
    """
    global_map: dict[tuple[float, float, float], int] = {}
    nodes: list[np.ndarray] = []

    def get(i, j, k, A, B, C):
        """Return global node index for barycentric coords (i, j, k) on face (A,B,C)."""
        p = (i * A + j * B + k * C) / n
        p = p / np.linalg.norm(p)          # project onto the sphere
        key = (round(p[0], 9), round(p[1], 9), round(p[2], 9))
        if key not in global_map:
            global_map[key] = len(nodes)
            nodes.append(p)
        return global_map[key]

    new_faces: list[tuple[int, int, int]] = []
    for (ia, ib, ic) in faces:
        A, B, C = v[ia], v[ib], v[ic]

        for i in range(n):
            for j in range(n - i):
                a = get(i, j, n - i - j, A, B, C)
                b = get(i + 1, j, n - i - 1 - j, A, B, C)
                c = get(i, j + 1, n - i - j - 1, A, B, C)
                if i + j < n - 1:
                    d = get(i + 1, j + 1, n - i - 1 - j - 1, A, B, C)
                    if (i + j) % 2 == 0:
                        new_faces.append((a, b, d))
                        new_faces.append((a, d, c))
                    else:
                        new_faces.append((a, b, c))
                        new_faces.append((b, d, c))
                else:
                    new_faces.append((a, b, c))
    return np.array(nodes, dtype=np.float64), np.array(new_faces, dtype=np.int64)


# --- Lloyd's relaxation (centroidal Voronoi tessellation) ------------------
def _centroids_areas(V: np.ndarray, T: np.ndarray):
    """Triangle (chord) areas (F,) and centroids (F,3)."""
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    return area, (a + b + c) / 3.0


def _incident_centroid(V: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Per-node target = area-weighted mean of its incident-triangle centroids.

    This is the standard (monotone) approximation of the Voronoi-cell centroid
    that Lloyd's algorithm moves each site toward, on a triangular mesh.
    """
    N, F = V.shape[0], T.shape[0]
    area, cent = _centroids_areas(V, T)
    # sparse incidence: S[i, t] = 1 iff node i is a vertex of triangle t (3/F)
    S = _sp.csr_matrix(
        (np.ones(3 * F), (T.reshape(-1), np.repeat(np.arange(F), 3))),
        shape=(N, F))
    wsum = S @ (area[:, None] * cent)
    w = S @ area
    return wsum / (w[:, None] + 1e-30)


def _spherical_delaunay(V: np.ndarray) -> np.ndarray:
    """Outward-oriented spherical Delaunay triangulation of `V` (unit sphere).

    The 3-D convex hull of a point set that surrounds the origin IS the
    spherical Delaunay triangulation, so a convex-hull call is exactly the
    re-triangulation step that lets Lloyd's algorithm converge (moving nodes to
    Voronoi centroids with a fixed, non-Delaunay topology diverges instead).
    Node indices are preserved (only connectivity changes).
    """
    try:
        T = ConvexHull(V).simplices.astype(np.int64)
    except Exception as exc:  # near-coplanar 4-tuples: retry with Qhull jitter
        T = ConvexHull(V, qhull_options="QJ").simplices.astype(np.int64)
        del exc
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    bad = np.einsum("ij,ij->i", np.cross(b - a, c - a), (a + b + c) / 3.0) < 0
    if bad.any():
        T[bad, 1], T[bad, 2] = T[bad, 2], T[bad, 1]
    return T


def _lloyd_relax(V: np.ndarray, T: np.ndarray, n_iter: int, alpha: float):
    """Relax (V, T) toward a centroidal Voronoi tessellation.

    Each iteration: move every node toward the centroid of its Voronoi cell
    (scaled by `alpha`, then re-project to the unit sphere), and re-triangulate
    to the spherical Delaunay. Node count and Euler characteristic are preserved;
    edge lengths and cell areas converge to uniform and the 12 pentagonal points
    regularize into clean pentagons (removing the polar "spoke" artifact).
    """
    V = V.astype(np.float64).copy()
    T = T.astype(np.int64).copy()
    for _ in range(int(n_iter)):
        V = V + alpha * (_incident_centroid(V, T) - V)
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        T = _spherical_delaunay(V)
    return V, T


def _refine_land(V: np.ndarray, T: np.ndarray,
                 land_at, steps: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Refine the triangles over land by `steps` subdivision step(s); ocean untouched.

    `land_at(lat_deg, lon_deg)` -> bool array, applied to each triangle centroid.

    Why densify-the-point-set instead of subdivide-and-keep: subdivide-and-keep
    would leave a hanging node on every land/ocean boundary edge (the land
    triangle's new midpoint sitting in the middle of the unrefined ocean
    triangle's edge). A hanging node is NOT a simplicial complex, and the P1
    engine's edge-based assembly (`Mesh._build_topology` / `Engine._build_diffusion`)
    assumes every edge has exactly two incident triangles -- it would mis-handle
    those edges as boundary edges. Re-triangulating the denser point set to the
    spherical Delaunay instead yields a conforming mesh (fine over land, coarse
    over ocean) with no hanging nodes, so the engine is unchanged.

    Node positions are preserved (we only ADD midpoints) and the result is
    always a valid spherical Delaunay triangulation.
    """
    V = V.astype(np.float64).copy()
    T = T.astype(np.int64).copy()
    for _ in range(int(steps)):
        # which base triangles are over land? sample the land mask at each centroid
        a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
        cent = (a + b + c) / 3.0
        cent = cent / np.linalg.norm(cent, axis=1, keepdims=True)
        lat = np.degrees(np.arcsin(np.clip(cent[:, 2], -1.0, 1.0)))
        lon = np.degrees(np.arctan2(cent[:, 1], cent[:, 0]))
        land_tri = np.asarray(land_at(lat, lon), dtype=bool)
        n_land = int(land_tri.sum())
        if n_land == 0:
            break
        # add the 3 edge midpoints of each land triangle (re-project to the
        # sphere), deduping against existing nodes and against each other so no
        # duplicate positions reach the Delaunay call.
        keys = {(round(float(p[0]), 9), round(float(p[1]), 9), round(float(p[2]), 9))
                for p in V}
        added: list[np.ndarray] = []
        for t in np.where(land_tri)[0]:
            i, j, k = int(T[t, 0]), int(T[t, 1]), int(T[t, 2])
            for x, y in ((i, j), (j, k), (k, i)):
                p = (V[x] + V[y]) / 2.0
                p = p / np.linalg.norm(p)
                key = (round(float(p[0]), 9), round(float(p[1]), 9), round(float(p[2]), 9))
                if key not in keys:
                    keys.add(key)
                    added.append(p)
        if added:
            V = np.concatenate([V, np.asarray(added, dtype=np.float64)], axis=0)
        # re-triangulate the (now denser over land) point set -- stays Delaunay
        T = _spherical_delaunay(V)
    return V, T


def _mesh_stats(V: np.ndarray, T: np.ndarray) -> dict:
    """Mesh-quality metrics (all edges counted once via the two-triangle set)."""
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    L = np.concatenate([np.linalg.norm(a - b, axis=1),
                        np.linalg.norm(b - c, axis=1),
                        np.linalg.norm(a - c, axis=1)])
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    # per-node valence = number of incident edges
    edge_set: set[tuple[int, int]] = set()
    for t in T:
        for e in range(3):
            x, y = int(t[e]), int(t[(e + 1) % 3])
            edge_set.add((min(x, y), max(x, y)))
    nbr = np.zeros(len(V), dtype=np.int64)
    for (x, y) in edge_set:
        nbr[x] += 1
        nbr[y] += 1
    vals, counts = np.unique(nbr, return_counts=True)
    return {
        "edge_cv": float(L.std() / L.mean()),
        "edge_min_over_max": float(L.min() / L.max()),
        "area_ratio_max_min": float(area.max() / area.min()),
        "valence": {int(v): int(k) for v, k in zip(vals, counts)},
    }


class Mesh:
    def __init__(self, resolution_km: float, relax: bool | None = None,
                 refine_land: bool | None = None):
        self.resolution_km = resolution_km
        # target edge length on the sphere
        base_edge_km = _icosahedron_edge_km()
        n = max(1, int(round(base_edge_km / resolution_km)))
        self.n = n
        base_v, base_faces = _icosahedron()
        self.V, self.T = _subdivide(base_v, base_faces, n)

        # Lloyd / CVT relaxation (README "Pole issue in rendering"): equalize
        # edge lengths and regularize the 12 pentagonal points. `relax=False`
        # forces the raw subdivision (e.g. for A/B comparison); otherwise the
        # config knob governs (0 iterations disables it).
        n_iter = 0 if relax is False else int(C.LLOYD_ITERATIONS)
        self.relaxation: dict | None = None
        if n_iter > 0:
            before = _mesh_stats(self.V, self.T)
            self.V, self.T = _lloyd_relax(self.V, self.T, n_iter, C.LLOYD_ALPHA)
            self.relaxation = {
                "iterations": n_iter, "alpha": float(C.LLOYD_ALPHA),
                "before": before, "after": _mesh_stats(self.V, self.T),
            }

        # Land-aware refinement: one subdivision step over land, none over ocean.
        # Done AFTER Lloyd (so the base is clean) and NOT re-relaxed afterward
        # (Lloyd equalizes cell size, which would erase the land/ocean contrast).
        # Gated by config (ON by default); pass refine_land=False for the uniform
        # base mesh (what the base-icosphere mesh tests pin).
        refine = C.REFINE_LAND if refine_land is None else bool(refine_land)
        self.refinement: dict | None = None
        if refine:
            from .land import LandMask
            lm = LandMask()
            before = _mesh_stats(self.V, self.T)
            n0, f0 = self.V.shape[0], self.T.shape[0]
            self.V, self.T = _refine_land(self.V, self.T, lm.is_land,
                                          C.LAND_REFINE_STEPS)
            self.refinement = {
                "steps": int(C.LAND_REFINE_STEPS),
                "nodes_before": int(n0), "nodes_after": int(self.V.shape[0]),
                "tris_before": int(f0), "tris_after": int(self.T.shape[0]),
                "before": before, "after": _mesh_stats(self.V, self.T),
            }

        self.N = self.V.shape[0]
        self.F = self.T.shape[0]

        self.lat = np.degrees(np.arcsin(np.clip(self.V[:, 2], -1, 1)))
        self.lon = np.degrees(np.arctan2(self.V[:, 1], self.V[:, 0]))

        self._build_topology()
        self._build_mass()
        self._build_diffusion(C.K_H)
        self._build_edges()

    # --- topology helpers --------------------------------------------------
    def _build_topology(self):
        """Build face↔neighbour map and the global edge table."""
        # face -> list of (neighbour_face, edge_index 0..2)
        self.face_neighbours: list[list[tuple[int, int]]] = [[] for _ in range(self.F)]

        # edge -> list of (triangle_idx, opposite_node)
        edge_map: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for t in range(self.F):
            tri = self.T[t]
            for e in range(3):
                a, b, c = tri[e], tri[(e + 1) % 3], tri[(e + 2) % 3]
                key = (min(a, b), max(a, b))
                edge_map.setdefault(key, []).append((t, c))

        self.edge_map = edge_map

        # Link neighbouring faces through shared edges
        for key, entries in edge_map.items():
            if len(entries) == 2:
                t0, opp0 = entries[0]
                t1, opp1 = entries[1]
                # find which edge index each face uses for this shared edge
                e0 = self._edge_index(self.T[t0], key[0], key[1])
                e1 = self._edge_index(self.T[t1], key[0], key[1])
                self.face_neighbours[t0].append((t1, e0))
                self.face_neighbours[t1].append((t0, e1))

    @staticmethod
    def _edge_index(tri: np.ndarray, a: int, b: int) -> int:
        """Return the edge index (0, 1, or 2) containing nodes a, b."""
        for e in range(3):
            if {tri[e], tri[(e + 1) % 3]} == {a, b}:
                return e
        raise ValueError(f"edge ({a},{b}) not found in triangle {tri}")

    # --- mass lumping ------------------------------------------------------
    def _build_mass(self):
        """Compute lumped diagonal mass matrix M_i = area_i / 3 per node."""
        M = np.zeros(self.N, dtype=np.float64)
        for t in range(self.F):
            tri = self.V[self.T[t]]
            area = 0.5 * np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0]))
            for node in self.T[t]:
                M[node] += area / 3.0
        self.M = M

    # --- diffusion ---------------------------------------------------------
    def _build_diffusion(self, K_h: float):
        """Build diffusion conductances per face edge.

        For each triangle face f and local edge e, store (a, b, conductance).
        conductance = K_h * |edge| / cell_distance  (harmonic mean of two cells).
        """
        self.diff_edges: list[tuple[int, int, float]] = []
        for key, entries in self.edge_map.items():
            a, b = key
            if len(entries) == 2:
                t0, _ = entries[0]
                t1, _ = entries[1]
                edge_vec = self.V[b] - self.V[a]
                edge_len = np.linalg.norm(edge_vec)
                centroid0 = self.V[self.T[t0]].mean(axis=0)
                centroid1 = self.V[self.T[t1]].mean(axis=0)
                cell_dist = np.linalg.norm(centroid1 - centroid0)
                if cell_dist > 0:
                    cond = K_h * edge_len / cell_dist
                else:
                    cond = 0.0
                self.diff_edges.append((a, b, cond))
            else:
                # boundary face — still add with same formula using one cell
                t0, _ = entries[0]
                edge_vec = self.V[b] - self.V[a]
                edge_len = np.linalg.norm(edge_vec)
                centroid0 = self.V[self.T[t0]].mean(axis=0)
                self.diff_edges.append((a, b, 0.0))

    # --- advection edge table -----------------------------------------------
    def _build_edges(self):
        """Build the edge table used by upwind advection.

        For every face, for each local edge (a→b), compute the outward normal
        and store (a, b, k_a, k_b, normal) where k_a, k_b are the nodal
        areas (face_area / 3).
        """
        self.adv_edges: list[tuple[int, int, float, float, np.ndarray]] = []
        for t in range(self.F):
            tri = self.V[self.T[t]]
            area = 0.5 * np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0]))
            tri_nodes = self.T[t]
            for e in range(3):
                a = tri_nodes[e]
                b = tri_nodes[(e + 1) % 3]
                edge_vec = self.V[b] - self.V[a]
                edge_len = np.linalg.norm(edge_vec)
                # outward-facing normal in the plane of the face
                face_normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
                fn_norm = np.linalg.norm(face_normal)
                face_normal = face_normal / (fn_norm + 1e-30)
                # edge normal = face_normal × edge_direction, pointing outward
                edge_dir = edge_vec / edge_len
                enormal = np.cross(face_normal, edge_dir)
                k = area / 3.0
                self.adv_edges.append((a, b, k, k, enormal))
