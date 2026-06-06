"""Tests for the Trotter-Suzuki Hamiltonian-simulation primitive."""

import numpy as np
import pytest
from scipy.linalg import expm

from qbit_simulator.algorithms.trotter import (
    trotter_evolve, apply_pauli_rotation, trotter_step,
)
from qbit_simulator.circuit import QuantumCircuit
from qbit_simulator.pauli import PauliOp


# ---- single Pauli rotation correctness ----

def test_pauli_rotation_z_matches_rz():
    """exp(-iθZ) applied as a Pauli rotation should match standard Rz."""
    theta = 0.7
    qc1 = QuantumCircuit(1)
    apply_pauli_rotation(qc1, "Z", theta)
    qc2 = QuantumCircuit(1)
    qc2.rz(2 * theta, 0)
    assert np.allclose(qc1.state, qc2.state, atol=1e-10)


def test_pauli_rotation_x_matches_rx():
    theta = 0.5
    qc1 = QuantumCircuit(1)
    apply_pauli_rotation(qc1, "X", theta)
    qc2 = QuantumCircuit(1)
    qc2.rx(2 * theta, 0)
    assert np.allclose(qc1.state, qc2.state, atol=1e-9)


def test_pauli_rotation_multi_qubit():
    """exp(-i θ ZZ) on |++⟩ = (|0⟩+|1⟩)/√2 ⊗ (|0⟩+|1⟩)/√2."""
    theta = 0.3
    qc = QuantumCircuit(2).h(0).h(1)
    apply_pauli_rotation(qc, "ZZ", theta)
    # Compute reference: exp(-i θ Z⊗Z) on |++⟩.
    from qbit_simulator.gates import Z, I2
    H = np.kron(Z, Z)
    U = expm(-1j * theta * H)
    psi0 = (np.kron(np.array([1, 1]) / np.sqrt(2),
                     np.array([1, 1]) / np.sqrt(2))).astype(np.complex128)
    expected = U @ psi0
    assert np.allclose(qc.state, expected, atol=1e-9)


# ---- Trotter evolution matches exact for short times ----

def test_trotter_matches_exact_evolution_short_time():
    """For a small Hamiltonian, Trotter should match exact evolution
    to high accuracy with enough steps."""
    H = PauliOp([(1.0 + 0j, "ZZ"), (0.5 + 0j, "XI"), (0.5 + 0j, "IX")])
    psi0 = np.array([1, 0, 0, 0], dtype=np.complex128)
    T = 0.3
    psi_trotter = trotter_evolve(H, psi0, total_time=T, n_steps=200, order=2)
    psi_exact = expm(-1j * T * H.matrix()) @ psi0
    # Order-2 Trotter at 200 steps should match exact to ~5 decimal places.
    inner = abs(np.vdot(psi_exact, psi_trotter))
    assert inner > 0.999


def test_trotter_higher_order_more_accurate():
    """Order 2 should beat order 1 for the same step count."""
    H = PauliOp([(1.0 + 0j, "ZZ"), (0.5 + 0j, "XI"), (0.5 + 0j, "IX")])
    psi0 = np.array([1, 0, 0, 0], dtype=np.complex128)
    T = 0.5
    psi_exact = expm(-1j * T * H.matrix()) @ psi0
    psi_o1 = trotter_evolve(H, psi0, total_time=T, n_steps=20, order=1)
    psi_o2 = trotter_evolve(H, psi0, total_time=T, n_steps=20, order=2)
    f1 = abs(np.vdot(psi_exact, psi_o1))
    f2 = abs(np.vdot(psi_exact, psi_o2))
    assert f2 > f1


def test_trotter_preserves_norm():
    """Trotter evolution is unitary; the final state must have unit norm."""
    H = PauliOp([(0.7 + 0j, "Z"), (0.3 + 0j, "X")])
    psi0 = np.array([1, 0], dtype=np.complex128)
    psi_t = trotter_evolve(H, psi0, total_time=0.5, n_steps=50)
    assert abs(np.linalg.norm(psi_t) - 1.0) < 1e-9


# ---- error handling ----

def test_trotter_rejects_complex_coefficient():
    H = PauliOp([(0.5 + 0.5j, "Z")])
    psi0 = np.array([1, 0], dtype=np.complex128)
    with pytest.raises(ValueError):
        trotter_evolve(H, psi0, total_time=0.1, n_steps=10, order=1)
