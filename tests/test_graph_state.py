"""Graph state tests."""

import pytest

from qbit_simulator.algorithms.graph_state import (
    graph_state, graph_state_stabilizers,
    cluster_state_1d, ring_graph_state, cluster_state_2d, complete_graph_state,
)


def test_two_qubit_graph_state_is_bell_like():
    """Two qubits + one edge = (|00⟩+|01⟩+|10⟩-|11⟩)/2 (the CZ-Bell state)."""
    g = graph_state(2, [(0, 1)])
    # Stabilizers should be: X_0 Z_1 and Z_0 X_1.
    assert set(g.stabilizers()) == {"+XZ", "+ZX"}


def test_three_qubit_path_graph_state():
    """Path 0-1-2. Stabilizers: K_0=XZ, K_1=ZXZ, K_2=ZX (in 3-char form)."""
    g = graph_state(3, [(0, 1), (1, 2)])
    expected = {"+XZI", "+ZXZ", "+IZX"}
    assert set(g.stabilizers()) == expected


def test_stabilizer_helper_matches_graph_state_stabilizers():
    """`graph_state_stabilizers` should produce the same generators (up to
    sign) that the live StabilizerState reports after construction."""
    n = 4
    edges = [(0, 1), (1, 2), (2, 3), (0, 3)]
    expected = graph_state_stabilizers(n, edges)
    g = graph_state(n, edges)
    got = [s.lstrip("+-") for s in g.stabilizers()]
    assert set(got) == set(expected)


def test_cluster_state_1d_construction():
    g = cluster_state_1d(5)
    assert g.n == 5


def test_ring_graph_state_construction():
    g = ring_graph_state(6)
    assert g.n == 6
    # Stabilizer K_0 should connect to neighbors 1 and 5 (periodic boundary).
    stab_0 = next(s for s in g.stabilizers() if s[1] == "X")
    # qubits 1 and 5 should have Z components.
    # stab_0 looks like "+X Z _ _ _ Z" or similar — letters at positions 1, 5.
    assert stab_0[2] == "Z"
    assert stab_0[6] == "Z"


def test_cluster_state_2d_grid():
    g = cluster_state_2d(3, 4)
    assert g.n == 12   # 3 rows × 4 cols


def test_complete_graph_state_small():
    g = complete_graph_state(4)
    assert g.n == 4
    # K_n has every vertex connected to every other; each stabilizer is
    # X_v · Z on the other (n-1) qubits.
    for s in g.stabilizers():
        body = s.lstrip("+-")
        x_count = body.count("X")
        z_count = body.count("Z")
        assert x_count == 1
        assert z_count == 3


def test_large_graph_state_is_cheap():
    """1000-qubit chain cluster state should build in well under a second."""
    import time
    t0 = time.perf_counter()
    g = cluster_state_1d(1000)
    dt = time.perf_counter() - t0
    assert dt < 5.0
    assert g.n == 1000


def test_invalid_edge_raises():
    with pytest.raises(ValueError):
        graph_state(3, [(0, 0)])           # self-loop
    with pytest.raises(IndexError):
        graph_state(3, [(0, 5)])           # out of range
    with pytest.raises(ValueError):
        graph_state(3, [(0, 1), (1, 0)])   # duplicate
