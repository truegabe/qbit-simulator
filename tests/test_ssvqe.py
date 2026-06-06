"""SSVQE (subspace-search VQE) tests."""

import numpy as np
import pytest

from qbit_simulator.algorithms.ssvqe import (
    ssvqe, pauli_op_to_matrix, hardware_efficient_ansatz_apply,
    _apply_single_qubit_gate, _apply_cnot, _ry,
)
from qbit_simulator.algorithms.h2_sto3g import h2_sto3g_hamiltonian


# ---- Ansatz primitives ----

def test_ry_at_zero_is_identity():
    R = _ry(0.0)
    assert np.allclose(R, np.eye(2), atol=1e-12)


def test_ry_at_pi_is_flip_with_sign():
    """Ry(π) sends |0⟩ → |1⟩ (and |1⟩ → -|0⟩)."""
    R = _ry(np.pi)
    zero = np.array([1, 0], dtype=complex)
    out = R @ zero
    assert abs(out[1] - 1) < 1e-9
    assert abs(out[0]) < 1e-9


def test_apply_single_qubit_gate_preserves_norm():
    rng = np.random.default_rng(0)
    psi = rng.normal(size=8) + 1j * rng.normal(size=8)
    psi = psi / np.linalg.norm(psi)
    gate = _ry(0.7)
    out = _apply_single_qubit_gate(psi, gate, q=1, n=3)
    assert abs(np.linalg.norm(out) - 1.0) < 1e-12


def test_apply_single_qubit_gate_targets_correct_qubit():
    """Regression: a previous version applied gate to qubit (n-1-q)
    instead of qubit q (axis-indexing off by reversal)."""
    # 3 qubits. Ry(pi) on qubit 0 of |000⟩ → |100⟩ (idx 4) MSB-first.
    Ry_pi = _ry(np.pi)
    psi = np.zeros(8, dtype=complex); psi[0] = 1.0
    out = _apply_single_qubit_gate(psi, Ry_pi, q=0, n=3)
    # Ry(pi)|0⟩ = |1⟩; on qubit 0, the result is |100⟩ at idx 4.
    assert abs(abs(out[4]) - 1.0) < 1e-9


def test_apply_cnot_correct_on_basis_state():
    # 2 qubits, |10⟩ → CNOT(0, 1) → |11⟩.
    # MSB-first: |10⟩ corresponds to qubit 0 = 1, qubit 1 = 0 → index 10 in binary = 2.
    psi = np.zeros(4, dtype=complex)
    psi[2] = 1.0  # |10⟩
    out = _apply_cnot(psi, control=0, target=1, n=2)
    # Should map to |11⟩ = index 3.
    assert abs(out[3] - 1.0) < 1e-12


def test_hardware_efficient_ansatz_returns_callable():
    apply, n_p = hardware_efficient_ansatz_apply(n_qubits=3, depth=2)
    # depth=2 → 3 layers of Ry (depth+1) on 3 qubits = 9 params.
    assert n_p == 9
    assert callable(apply)


def test_hardware_efficient_ansatz_preserves_norm():
    apply, n_p = hardware_efficient_ansatz_apply(n_qubits=2, depth=2)
    rng = np.random.default_rng(0)
    params = rng.uniform(-1, 1, size=n_p)
    ref = np.array([1, 0, 0, 0], dtype=complex)
    out = apply(params, ref)
    assert abs(np.linalg.norm(out) - 1.0) < 1e-10


# ---- pauli_op_to_matrix ----

def test_pauli_op_to_matrix_h2():
    H_pauli = h2_sto3g_hamiltonian(0.74)
    H_mat = pauli_op_to_matrix(H_pauli, 2)
    assert H_mat.shape == (4, 4)
    assert np.allclose(H_mat, H_mat.conj().T, atol=1e-12)


# ---- SSVQE ----

def test_ssvqe_rejects_non_orthogonal_refs():
    H_mat = pauli_op_to_matrix(h2_sto3g_hamiltonian(0.74), 2)
    apply, n_p = hardware_efficient_ansatz_apply(n_qubits=2, depth=2)
    refs = [np.array([1, 0, 0, 0], dtype=complex),
            np.array([1, 1, 0, 0], dtype=complex) / np.sqrt(2)]
    init = np.zeros(n_p)
    with pytest.raises(ValueError):
        ssvqe(H_mat, apply, refs, weights=[1.0, 0.5], init_params=init)


def test_ssvqe_rejects_non_decreasing_weights():
    H_mat = pauli_op_to_matrix(h2_sto3g_hamiltonian(0.74), 2)
    apply, n_p = hardware_efficient_ansatz_apply(n_qubits=2, depth=2)
    refs = [np.eye(4, dtype=complex)[:, i] for i in range(2)]
    init = np.zeros(n_p)
    with pytest.raises(ValueError):
        ssvqe(H_mat, apply, refs, weights=[0.5, 0.5], init_params=init)
    with pytest.raises(ValueError):
        ssvqe(H_mat, apply, refs, weights=[0.5, 1.0], init_params=init)


def test_ssvqe_finds_h2_spectrum():
    """SSVQE on the 2-qubit H₂ STO-3G should find all 4 eigenvalues."""
    H_pauli = h2_sto3g_hamiltonian(0.74)
    H_mat = pauli_op_to_matrix(H_pauli, 2)
    true_eigs = np.linalg.eigvalsh(H_mat)

    apply, n_p = hardware_efficient_ansatz_apply(n_qubits=2, depth=3)
    refs = [np.eye(4, dtype=complex)[:, i] for i in range(4)]
    rng = np.random.default_rng(0)
    init = rng.uniform(-1, 1, size=n_p)
    result = ssvqe(H_mat, apply, refs, weights=[1.0, 0.7, 0.4, 0.2],
                    init_params=init, max_iter=500)
    energies = sorted(result["energies"])
    for k, (E_ss, E_true) in enumerate(zip(energies, true_eigs)):
        assert abs(E_ss - E_true) < 1e-4, f"eigenstate {k}: {E_ss} vs {E_true}"


def test_ssvqe_returns_orthogonal_states():
    """Output |ψ_i⟩ should be mutually orthogonal (since U is unitary
    and references are orthogonal)."""
    H_mat = pauli_op_to_matrix(h2_sto3g_hamiltonian(0.74), 2)
    apply, n_p = hardware_efficient_ansatz_apply(n_qubits=2, depth=2)
    refs = [np.eye(4, dtype=complex)[:, i] for i in range(2)]
    rng = np.random.default_rng(0)
    init = rng.uniform(-1, 1, size=n_p)
    result = ssvqe(H_mat, apply, refs, weights=[1.0, 0.5],
                    init_params=init, max_iter=200)
    states = result["states"]
    overlap = abs(np.vdot(states[0], states[1]))
    assert overlap < 1e-10


def test_ssvqe_energies_at_least_as_high_as_true():
    """Each output energy ≥ corresponding true eigenvalue (variational)."""
    H_mat = pauli_op_to_matrix(h2_sto3g_hamiltonian(0.74), 2)
    true_eigs = sorted(np.linalg.eigvalsh(H_mat))
    apply, n_p = hardware_efficient_ansatz_apply(n_qubits=2, depth=3)
    refs = [np.eye(4, dtype=complex)[:, i] for i in range(3)]
    rng = np.random.default_rng(0)
    init = rng.uniform(-1, 1, size=n_p)
    result = ssvqe(H_mat, apply, refs, weights=[1.0, 0.5, 0.3],
                    init_params=init, max_iter=300)
    energies = sorted(result["energies"])
    for E_ss, E_true in zip(energies, true_eigs):
        assert E_ss >= E_true - 1e-7
