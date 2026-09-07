"""Tests for the icosphere mesh and its Lloyd / CVT relaxation.

Run with the project venv:
    .venv/bin/python -m pytest tests/test_mesh.py -q
or without pytest:
    .venv/bin/python tests/test_mesh.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sim.mesh import Mesh, _spherical_delaunay, _mesh_stats  # noqa: E402


def _edge_set(T):
    edges = set()
    for t in T:
        for e in range(3):
            x, y = int(t[e]), int(t[(e + 1) % 3])
            edges.add((min(x, y), max(x, y)))
    return edges


def test_topology_invariants_held():
    """Relaxation preserves node count, Euler char, sphere + outward orientation."""
    m = Mesh(250.0, refine_land=False)
    V, T = m.V, m.T
    N, F = m.N, m.T.shape[0]
    # nodes on the unit sphere
    assert np.abs(np.linalg.norm(V, axis=1) - 1.0).max() < 1e-12, "nodes off the sphere"
    # outward orientation (normal . centroid > 0 for every face)
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    inward = int((np.einsum("ij,ij->i", np.cross(b - a, c - a), (a + b + c) / 3.0) < 0).sum())
    assert inward == 0, f"{inward} inward-facing triangles after relaxation"
    # closed triangulation: V - E + F = 2 (sphere), F = 2V - 4
    E = len(_edge_set(T))
    assert N - E + F == 2, f"Euler char = {N - E + F} != 2"
    assert F == 2 * N - 4, f"faces {F} != 2N-4 = {2 * N - 4}"
    # no degenerate triangles
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    assert area.min() > 0.0, "degenerate (zero-area) triangle present"


def test_relaxation_improves_edge_uniformity():
    """Lloyd's step must equalize edge lengths (the README's stated goal)."""
    raw = Mesh(250.0, relax=False, refine_land=False)
    rel = Mesh(250.0, refine_land=False)
    b, a = _mesh_stats(raw.V, raw.T), _mesh_stats(rel.V, rel.T)
    # edge-length coefficient of variance drops substantially
    assert a["edge_cv"] < 0.5 * b["edge_cv"], (
        f"edge CV not improved: raw {b['edge_cv']:.3f} -> relaxed {a['edge_cv']:.3f}")
    # the min/max edge-length ratio moves toward 1
    assert a["edge_min_over_max"] > b["edge_min_over_max"], "edges less uniform than raw"


def test_relaxation_yields_clean_icosphere_valence():
    """A relaxed icosphere has exactly 12 pentagonal points and hexagons elsewhere.

    The raw subdivision has mixed valence (4-8); the re-triangulation to the
    spherical Delaunay cleans it to the Euler-forced {5: 12, 6: N-12}.
    """
    raw = Mesh(250.0, relax=False, refine_land=False)
    rel = Mesh(250.0, refine_land=False)
    raw_val = _mesh_stats(raw.V, raw.T)["valence"]
    rel_val = _mesh_stats(rel.V, rel.T)["valence"]
    assert raw_val != rel_val, "valence histogram unchanged by relaxation"
    val = rel_val
    assert val.get(5, 0) == 12, f"expected 12 pentagons, got {val.get(5, 0)}: {val}"
    assert 4 not in val and 7 not in val and 8 not in val, (
        f"irregular valence remains after relaxation: {val}")


def test_relaxation_is_deterministic():
    """Two builds from the same seed mesh produce identical node positions."""
    m1 = Mesh(250.0, refine_land=False)
    m2 = Mesh(250.0, refine_land=False)
    assert np.allclose(m1.V, m2.V, atol=0.0), "relaxation is not deterministic"
    assert np.array_equal(m1.T, m2.T), "relaxation triangulation is not deterministic"


def test_relax_disabled_matches_raw_subdivision():
    """relax=False leaves the subdivision untouched (N/F identical, stats = raw)."""
    raw = Mesh(250.0, relax=False, refine_land=False)
    assert raw.relaxation is None, "relaxation should be off with relax=False"
    # a fresh relaxed build has the SAME node count (Lloyd moves, never adds, nodes)
    rel = Mesh(250.0, refine_land=False)
    assert rel.N == raw.N and rel.F == raw.F, "relaxation changed node/face count"


def test_spherical_delaunay_reprojects_to_sphere():
    """The re-triangulation helper is a closed, outward triangulation of the inputs."""
    import math
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    raw = [
        (0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi),
        (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0),
        (phi, 0, 1), (phi, 0, -1), (-phi, 0, 1), (-phi, 0, -1),
    ]
    V = np.array(raw, dtype=np.float64)
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    T = _spherical_delaunay(V)
    assert T.shape[0] == 20, f"icosahedron hull should have 20 faces, got {T.shape[0]}"
    assert T.min() >= 0 and T.max() < len(V), "face references an out-of-range node"
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    inward = int((np.einsum("ij,ij->i", np.cross(b - a, c - a), (a + b + c) / 3.0) < 0).sum())
    assert inward == 0, "spherical Delaunay returned inward-facing faces"


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
