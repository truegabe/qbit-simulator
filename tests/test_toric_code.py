"""Toric code (Kitaev 2003) tests."""

import pytest

from qbit_simulator.algorithms.toric_code import (
    edge_id, vertex_neighbors, plaquette_edges,
    build_toric_code, syndrome, logical_error,
    count_independent_stabilizers, n_logical_qubits,
    stabilizers_commute, logicals_commute_with_stabilizers,
    logical_x_z_anticommute,
)


# ---- Geometry ----

def test_edge_id_distinct_for_distinct_edges():
    L = 4
    ids = set()
    for r in range(L):
        for c in range(L):
            for o in ("h", "v"):
                ids.add(edge_id(r, c, o, L))
    assert len(ids) == 2 * L * L


def test_edge_id_periodic():
    """Index wraps around: (0, L) ≡ (0, 0)."""
    L = 3
    assert edge_id(0, L, "h", L) == edge_id(0, 0, "h", L)
    assert edge_id(L, 0, "v", L) == edge_id(0, 0, "v", L)


def test_edge_id_rejects_bad_orientation():
    with pytest.raises(ValueError):
        edge_id(0, 0, "x", 3)


def test_vertex_has_four_distinct_neighbors():
    """At each vertex, the 4 edges are distinct."""
    L = 3
    for r in range(L):
        for c in range(L):
            edges = vertex_neighbors(r, c, L)
            assert len(set(edges)) == 4


def test_plaquette_has_four_distinct_edges():
    L = 3
    for r in range(L):
        for c in range(L):
            edges = plaquette_edges(r, c, L)
            assert len(set(edges)) == 4


# ---- Code construction ----

@pytest.mark.parametrize("L", [2, 3, 4])
def test_n_qubits_2L_squared(L):
    code = build_toric_code(L)
    assert code.n_qubits == 2 * L * L


@pytest.mark.parametrize("L", [2, 3, 4])
def test_correct_number_of_stabilizers(L):
    code = build_toric_code(L)
    assert len(code.vertex_stabs) == L * L
    assert len(code.plaquette_stabs) == L * L


def test_rejects_L_below_2():
    with pytest.raises(ValueError):
        build_toric_code(1)


def test_two_logicals_each():
    code = build_toric_code(3)
    assert len(code.logical_x) == 2
    assert len(code.logical_z) == 2


# ---- Commutation properties ----

@pytest.mark.parametrize("L", [2, 3, 4])
def test_stabilizers_commute_pairwise(L):
    """Every A_v commutes with every B_p."""
    assert stabilizers_commute(build_toric_code(L))


@pytest.mark.parametrize("L", [2, 3, 4])
def test_logicals_commute_with_stabilizers(L):
    """Every logical operator commutes with every stabilizer."""
    assert logicals_commute_with_stabilizers(build_toric_code(L))


@pytest.mark.parametrize("L", [2, 3, 4])
def test_logical_X_Z_anticommute_pairwise(L):
    """X-logical-i ↔ Z-logical-i anticommute; X-i ↔ Z-j (i≠j) commute."""
    assert logical_x_z_anticommute(build_toric_code(L))


# ---- Stabilizer counts ----

def test_count_independent_stabilizers():
    """2 L² − 2 independent stabilizers."""
    for L in (2, 3, 4):
        assert count_independent_stabilizers(L) == 2 * L * L - 2


def test_n_logical_qubits_is_2():
    """The torus topology gives 2 logical qubits."""
    for L in (2, 3, 4):
        assert n_logical_qubits(L) == 2


# ---- Syndromes ----

def test_no_error_no_syndrome():
    code = build_toric_code(3)
    syn = syndrome(code, set(), set())
    assert not any(syn["vertex"])
    assert not any(syn["plaquette"])


def test_single_z_error_lights_two_vertices():
    """A Z error on an edge anticommutes with the 2 A_v's at its endpoints."""
    code = build_toric_code(3)
    syn = syndrome(code, set(), {0})
    assert sum(syn["vertex"]) == 2
    assert sum(syn["plaquette"]) == 0


def test_single_x_error_lights_two_plaquettes():
    code = build_toric_code(3)
    syn = syndrome(code, {0}, set())
    assert sum(syn["plaquette"]) == 2
    assert sum(syn["vertex"]) == 0


# ---- Logical-operator detection ----

def test_z_logical_no_syndrome_but_flips_x_partner():
    """Applying a Z-logical should leave all syndromes silent but flip
    the conjugate X-logical."""
    code = build_toric_code(3)
    z_loop = set(code.logical_z[0])
    syn = syndrome(code, set(), z_loop)
    err = logical_error(code, set(), z_loop)
    assert not any(syn["vertex"]) and not any(syn["plaquette"])
    assert err["logical_x"] == [1, 0]
    assert err["logical_z"] == [0, 0]


def test_x_logical_no_syndrome_but_flips_z_partner():
    code = build_toric_code(3)
    x_loop = set(code.logical_x[1])
    syn = syndrome(code, x_loop, set())
    err = logical_error(code, x_loop, set())
    assert not any(syn["vertex"]) and not any(syn["plaquette"])
    assert err["logical_z"] == [0, 1]


def test_no_error_no_logical_flip():
    code = build_toric_code(3)
    err = logical_error(code, set(), set())
    assert err["logical_x"] == [0, 0]
    assert err["logical_z"] == [0, 0]
