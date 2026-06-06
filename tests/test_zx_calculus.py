"""ZX calculus tests."""

import numpy as np
import pytest

from qbit_simulator.zx_calculus import (
    ZXDiagram, Spider, Edge,
    to_unitary, fuse_spiders, simplify,
    _spider_matrix,
)


# ---- Spider matrix ----

def test_z_spider_identity_no_phase():
    """A 1-in, 1-out Z-spider with phase 0 should be identity."""
    M = _spider_matrix("Z", phase=0.0, n_in=1, n_out=1)
    assert np.allclose(M, np.eye(2), atol=1e-12)


def test_z_spider_phase_rotation():
    """Z-spider phi=pi → diag(1, e^{i pi}) = diag(1, -1) = Z."""
    M = _spider_matrix("Z", phase=np.pi, n_in=1, n_out=1)
    assert np.allclose(M, np.diag([1, -1]), atol=1e-12)


def test_z_spider_one_in_two_out():
    """Z-spider with 1 in, 2 out: |0⟩→|00⟩, |1⟩→e^{iφ}|11⟩ — a "copy"."""
    M = _spider_matrix("Z", phase=0.0, n_in=1, n_out=2)
    assert M.shape == (4, 2)
    # |0⟩ → |00⟩.
    assert M[0, 0] == 1.0
    # |1⟩ → |11⟩.
    assert M[3, 1] == 1.0


def test_x_spider_pi_is_x_gate():
    """X-spider with phi=pi is the X gate (up to phase)."""
    M = _spider_matrix("X", phase=np.pi, n_in=1, n_out=1)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    # M and X should agree up to a global phase.
    overlap = np.abs(np.trace(M.conj().T @ X)) / 2
    assert abs(overlap - 1.0) < 1e-9


# ---- Diagram construction ----

def test_empty_diagram():
    d = ZXDiagram()
    assert len(d.spiders) == 0
    assert len(d.edges) == 0


def test_add_input_marks_boundary():
    d = ZXDiagram()
    sid = d.add_input()
    assert d.spiders[sid].is_input_boundary
    assert sid in d.inputs


def test_add_output_marks_boundary():
    d = ZXDiagram()
    sid = d.add_output()
    assert d.spiders[sid].is_output_boundary
    assert sid in d.outputs


# ---- Contraction ----

def test_identity_wire():
    d = ZXDiagram()
    sin = d.add_input()
    sout = d.add_output()
    d.add_edge(sin, sout)
    M = to_unitary(d)
    assert np.allclose(M, np.eye(2), atol=1e-12)


@pytest.mark.parametrize("phi", [0.0, 0.3, np.pi / 2, np.pi])
def test_z_rotation_wire(phi):
    d = ZXDiagram()
    sin = d.add_input(); sout = d.add_output()
    mid = d.add_spider("Z", phase=phi)
    d.add_edge(sin, mid); d.add_edge(mid, sout)
    M = to_unitary(d)
    expected = np.array([[1, 0], [0, np.exp(1j * phi)]], dtype=complex)
    assert np.allclose(M, expected, atol=1e-12)


def test_x_spider_wire_phi_pi_gives_x_up_to_phase():
    d = ZXDiagram()
    sin = d.add_input(); sout = d.add_output()
    mid = d.add_spider("X", phase=np.pi)
    d.add_edge(sin, mid); d.add_edge(mid, sout)
    M = to_unitary(d)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    overlap = np.abs(np.trace(M.conj().T @ X)) / 2
    assert abs(overlap - 1.0) < 1e-9


# ---- Fusion rule (S1) ----

def test_fuse_two_z_spiders_preserves_unitary():
    d = ZXDiagram()
    sin = d.add_input(); sout = d.add_output()
    m1 = d.add_spider("Z", phase=0.3)
    m2 = d.add_spider("Z", phase=0.4)
    d.add_edge(sin, m1); d.add_edge(m1, m2); d.add_edge(m2, sout)
    M_before = to_unitary(d)
    fused = fuse_spiders(d)
    M_after = to_unitary(d)
    assert fused
    assert np.allclose(M_before, M_after, atol=1e-12)


def test_fuse_two_z_spiders_sums_phases():
    d = ZXDiagram()
    sin = d.add_input(); sout = d.add_output()
    m1 = d.add_spider("Z", phase=0.3)
    m2 = d.add_spider("Z", phase=0.4)
    d.add_edge(sin, m1); d.add_edge(m1, m2); d.add_edge(m2, sout)
    fuse_spiders(d)
    # m1 should now have phase 0.7.
    assert abs(d.spiders[m1].phase - 0.7) < 1e-9
    # m2 should be marked inactive.
    assert d.spiders[m2].is_inactive


def test_fuse_different_colors_does_nothing():
    """A Z spider connected to an X spider should NOT fuse."""
    d = ZXDiagram()
    sin = d.add_input(); sout = d.add_output()
    m1 = d.add_spider("Z", phase=0.3)
    m2 = d.add_spider("X", phase=0.4)
    d.add_edge(sin, m1); d.add_edge(m1, m2); d.add_edge(m2, sout)
    fused = fuse_spiders(d)
    assert not fused


def test_simplify_chain_of_three():
    """3 Z-spiders in a chain → 2 fusions, final 1 spider with summed phase."""
    d = ZXDiagram()
    sin = d.add_input(); sout = d.add_output()
    spiders = [d.add_spider("Z", phase=0.1 * (k + 1)) for k in range(3)]
    d.add_edge(sin, spiders[0])
    for k in range(2):
        d.add_edge(spiders[k], spiders[k + 1])
    d.add_edge(spiders[-1], sout)
    M_before = to_unitary(d)
    n = simplify(d)
    M_after = to_unitary(d)
    assert n == 2
    assert np.allclose(M_before, M_after, atol=1e-12)


def test_simplify_terminates_for_disjoint_diagram():
    """Two disjoint Z-spiders shouldn't fuse with each other."""
    d = ZXDiagram()
    sin = d.add_input(); sout = d.add_output()
    m1 = d.add_spider("Z", phase=0.5)
    d.add_edge(sin, m1); d.add_edge(m1, sout)
    n = simplify(d)
    assert n == 0


# ---- Diagram with multiple input/output wires ----

def test_two_qubit_identity():
    """Two parallel identity wires → I ⊗ I."""
    d = ZXDiagram()
    in0 = d.add_input(); in1 = d.add_input()
    out0 = d.add_output(); out1 = d.add_output()
    d.add_edge(in0, out0); d.add_edge(in1, out1)
    M = to_unitary(d)
    assert M.shape == (4, 4)
    assert np.allclose(M, np.eye(4), atol=1e-12)
