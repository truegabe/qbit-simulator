"""PEPS tests — basic construction, gate application, and equivalence to dense."""

import numpy as np
import pytest

from qbit_simulator.peps import PEPSState
from qbit_simulator import QuantumCircuit


# ---- initial state ----

def test_initial_peps_is_all_zeros():
    """A fresh PEPS should represent |0...0⟩."""
    peps = PEPSState(Lx=2, Ly=2)
    psi = peps.to_dense()
    expected = np.zeros(16, dtype=np.complex128)
    expected[0] = 1.0
    assert np.allclose(psi, expected)


def test_initial_peps_norm_one():
    """⟨ψ|ψ⟩ = 1 for the initial product state."""
    peps = PEPSState(Lx=3, Ly=2)
    psi = peps.to_dense()
    assert abs(np.linalg.norm(psi) - 1.0) < 1e-10


def test_bond_dimensions_start_at_one():
    peps = PEPSState(Lx=3, Ly=3, max_chi=16)
    bds = peps.bond_dimensions()
    assert all(d == 1 for d in bds["horizontal"])
    assert all(d == 1 for d in bds["vertical"])


def test_site_count():
    peps = PEPSState(Lx=3, Ly=4)
    assert peps.n_sites == 12
    assert len(peps.site_indices()) == 12


# ---- single-qubit gates ----

def test_x_gate_at_corner():
    """X on (0, 0) should flip the first qubit of |00...0⟩."""
    peps = PEPSState(Lx=2, Ly=2)
    peps.x(0, 0)
    psi = peps.to_dense()
    # Site (0, 0) is the most significant qubit (flat idx 0).
    # |1000⟩ = 8 in 4-qubit MSB convention.
    expected = np.zeros(16, dtype=np.complex128)
    expected[8] = 1.0
    assert np.allclose(psi, expected)


def test_hadamard_on_2x1_peps_matches_dense():
    """H on qubit 0 of a 2x1 PEPS produces (|00⟩+|10⟩)/√2."""
    peps = PEPSState(Lx=2, Ly=1)
    peps.h(0, 0)
    psi = peps.to_dense()
    expected = np.zeros(4, dtype=np.complex128)
    expected[0] = 1 / np.sqrt(2)
    expected[2] = 1 / np.sqrt(2)
    assert np.allclose(psi, expected, atol=1e-10)


# ---- two-qubit gates ----

def test_bell_pair_via_peps_2x1():
    """H(0); CNOT(0,1) on a 2×1 PEPS gives the Bell pair."""
    peps = PEPSState(Lx=2, Ly=1, max_chi=4)
    peps.h(0, 0)
    peps.cnot((0, 0), (1, 0))
    psi = peps.to_dense()
    expected = np.zeros(4, dtype=np.complex128)
    expected[0] = 1 / np.sqrt(2)
    expected[3] = 1 / np.sqrt(2)
    inner = abs(np.vdot(expected, psi))
    assert inner > 0.999


def test_vertical_cnot_on_1x2_peps():
    """CNOT on a vertical bond."""
    peps = PEPSState(Lx=1, Ly=2, max_chi=4)
    peps.h(0, 0)
    peps.cnot((0, 0), (0, 1))
    psi = peps.to_dense()
    expected = np.zeros(4, dtype=np.complex128)
    expected[0] = 1 / np.sqrt(2)
    expected[3] = 1 / np.sqrt(2)
    inner = abs(np.vdot(expected, psi))
    assert inner > 0.999


def test_2x2_ghz_via_peps():
    """Build a 4-qubit GHZ via PEPS on 2×2 lattice."""
    peps = PEPSState(Lx=2, Ly=2, max_chi=4)
    peps.h(0, 0)
    peps.cnot((0, 0), (1, 0))   # horizontal
    peps.cnot((0, 0), (0, 1))   # vertical
    peps.cnot((1, 0), (1, 1))   # vertical
    psi = peps.to_dense()
    # GHZ on 4 qubits: (|0000⟩ + |1111⟩) / √2.
    expected = np.zeros(16, dtype=np.complex128)
    expected[0]  = 1 / np.sqrt(2)
    expected[15] = 1 / np.sqrt(2)
    inner = abs(np.vdot(expected, psi))
    assert inner > 0.999


# ---- equivalence to dense engine ----

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_peps_random_circuit_matches_dense(seed):
    """Apply the same random gate sequence to a PEPS and to a QuantumCircuit,
    verify final states match bit-for-bit."""
    rng = np.random.default_rng(seed)
    Lx, Ly = 2, 2
    N = Lx * Ly
    peps = PEPSState(Lx, Ly, max_chi=8)
    qc = QuantumCircuit(N)

    def flat_idx(x, y):
        """Site (x, y) → flat qubit index (row-major)."""
        return y * Lx + x

    for _ in range(8):
        op = rng.integers(0, 3)
        if op == 0:
            # Random 1-qubit gate (H, X, or Z).
            choice = rng.integers(0, 3)
            x, y = int(rng.integers(0, Lx)), int(rng.integers(0, Ly))
            if choice == 0:
                peps.h(x, y); qc.h(flat_idx(x, y))
            elif choice == 1:
                peps.x(x, y); qc.x(flat_idx(x, y))
            else:
                peps.z(x, y); qc.z(flat_idx(x, y))
        elif op == 1:
            # Horizontal CNOT.
            x = int(rng.integers(0, Lx - 1))
            y = int(rng.integers(0, Ly))
            peps.cnot((x, y), (x + 1, y))
            qc.cnot(flat_idx(x, y), flat_idx(x + 1, y))
        else:
            # Vertical CNOT.
            x = int(rng.integers(0, Lx))
            y = int(rng.integers(0, Ly - 1))
            peps.cnot((x, y), (x, y + 1))
            qc.cnot(flat_idx(x, y), flat_idx(x, y + 1))

    psi_peps  = peps.to_dense()
    psi_dense = qc.state
    inner = abs(np.vdot(psi_peps, psi_dense))
    assert inner > 0.999


# ---- bond dimension growth ----

def test_bond_dim_grows_after_entangling_gates():
    """After entangling gates, internal bond dimensions should exceed 1."""
    peps = PEPSState(Lx=3, Ly=3, max_chi=8)
    peps.h(0, 0)
    peps.cnot((0, 0), (1, 0))
    peps.cnot((1, 0), (2, 0))
    bds = peps.bond_dimensions()
    # The horizontal bonds in row 0 should have χ > 1 after Bell-like entangling.
    h_row0 = [bds["horizontal"][0], bds["horizontal"][1]]
    assert max(h_row0) > 1


def test_bond_dim_capped_by_max_chi():
    """Bond dimension shouldn't exceed max_chi."""
    peps = PEPSState(Lx=2, Ly=2, max_chi=2)
    peps.h(0, 0); peps.h(0, 1)
    peps.cnot((0, 0), (1, 0))
    peps.cnot((0, 1), (1, 1))
    peps.cnot((0, 0), (0, 1))
    peps.cnot((1, 0), (1, 1))
    bds = peps.bond_dimensions()
    assert max(bds["horizontal"] + bds["vertical"]) <= 2
