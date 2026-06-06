"""State tomography tests."""

import numpy as np
import pytest

from qbit_simulator.tomography import (
    state_tomography, reconstruct_density_matrix, exact_pauli_expectations,
    state_fidelity, project_to_psd, all_pauli_strings,
)


# ---- pauli string enumeration ----

def test_all_pauli_strings_count():
    assert len(all_pauli_strings(1)) == 4
    assert len(all_pauli_strings(2)) == 16
    assert len(all_pauli_strings(3)) == 64


# ---- exact tomography is perfect ----

def test_exact_tomography_pure_state():
    """With shots=None we should recover the exact state."""
    psi = np.array([1, 1], dtype=np.complex128) / np.sqrt(2)
    result = state_tomography(psi, shots=None)
    expected = np.outer(psi, psi.conj())
    assert np.allclose(result["rho_estimated"], expected, atol=1e-10)
    assert result["fidelity"] > 0.999


def test_exact_tomography_bell_pair():
    """Bell pair: |00⟩ + |11⟩, normalized."""
    psi = np.zeros(4, dtype=np.complex128)
    psi[0] = 1 / np.sqrt(2); psi[3] = 1 / np.sqrt(2)
    result = state_tomography(psi, shots=None)
    expected = np.outer(psi, psi.conj())
    assert np.allclose(result["rho_estimated"], expected, atol=1e-10)


def test_exact_tomography_mixed_state():
    """Maximally mixed state: ρ = I/2."""
    rho = np.eye(2, dtype=np.complex128) / 2
    result = state_tomography(rho, shots=None)
    assert np.allclose(result["rho_estimated"], rho, atol=1e-10)


# ---- expectation values match ----

def test_pauli_expectations_for_zero_state():
    """|0⟩: ⟨Z⟩ = 1, ⟨X⟩ = ⟨Y⟩ = 0, ⟨I⟩ = 1."""
    psi = np.array([1, 0], dtype=np.complex128)
    rho = np.outer(psi, psi.conj())
    exp = exact_pauli_expectations(rho)
    assert exp["I"] == pytest.approx(1.0, abs=1e-12)
    assert exp["Z"] == pytest.approx(1.0, abs=1e-12)
    assert exp["X"] == pytest.approx(0.0, abs=1e-12)
    assert exp["Y"] == pytest.approx(0.0, abs=1e-12)


def test_pauli_expectations_for_plus_state():
    """|+⟩: ⟨X⟩ = 1, ⟨Z⟩ = ⟨Y⟩ = 0."""
    psi = np.array([1, 1], dtype=np.complex128) / np.sqrt(2)
    rho = np.outer(psi, psi.conj())
    exp = exact_pauli_expectations(rho)
    assert exp["X"] == pytest.approx(1.0, abs=1e-12)
    assert exp["Z"] == pytest.approx(0.0, abs=1e-12)


# ---- finite-shot tomography converges with more shots ----

def test_finite_shots_approaches_exact():
    """Increasing shot count should improve fidelity."""
    psi = np.array([1, 1j], dtype=np.complex128) / np.sqrt(2)
    rng = np.random.default_rng(0)
    f_low  = state_tomography(psi, shots=20,   rng=rng)["fidelity"]
    rng = np.random.default_rng(0)
    f_high = state_tomography(psi, shots=2000, rng=rng)["fidelity"]
    # High-shot fidelity should be near 1.
    assert f_high > 0.95
    # And higher than low-shot.
    assert f_high > f_low - 0.1   # allow some stochasticity


# ---- reconstruction is Hermitian and trace-1 ----

def test_reconstructed_density_matrix_is_hermitian():
    psi = np.array([0.6, 0.8], dtype=np.complex128)
    result = state_tomography(psi, shots=None)
    rho = result["rho_estimated"]
    assert np.allclose(rho, rho.conj().T, atol=1e-10)
    assert abs(np.trace(rho) - 1.0) < 1e-10


# ---- PSD projection ----

def test_psd_projection_preserves_psd_matrix():
    """A valid density matrix should be unchanged by the projection."""
    rho = np.array([[0.7, 0.2], [0.2, 0.3]], dtype=np.complex128)
    projected = project_to_psd(rho)
    assert np.allclose(rho, projected, atol=1e-9)


def test_psd_projection_fixes_negative_eigenvalues():
    """A matrix with negative eigenvalues should be projected."""
    # Construct a Hermitian but not-PSD matrix.
    bad = np.array([[1.5, 0.5], [0.5, -0.5]], dtype=np.complex128)
    fixed = project_to_psd(bad)
    eigs = np.linalg.eigvalsh(fixed)
    assert all(e >= -1e-12 for e in eigs)
    assert abs(np.trace(fixed) - 1.0) < 1e-10


# ---- state_fidelity ----

def test_state_fidelity_identical():
    psi = np.array([0.6, 0.8], dtype=np.complex128)
    rho = np.outer(psi, psi.conj())
    assert state_fidelity(rho, rho) == pytest.approx(1.0, abs=1e-9)


def test_choi_matrix_for_identity_channel():
    """Identity channel: Choi = |Φ+⟩⟨Φ+|."""
    from qbit_simulator.tomography import choi_matrix_from_kraus
    J = choi_matrix_from_kraus([np.eye(2, dtype=np.complex128)])
    expected = np.zeros((4, 4), dtype=np.complex128)
    # |Φ+⟩ = (|00⟩+|11⟩)/√2 has support on indices 0 and 3.
    expected[0, 0] = 0.5; expected[0, 3] = 0.5
    expected[3, 0] = 0.5; expected[3, 3] = 0.5
    assert np.allclose(J, expected, atol=1e-10)


def test_choi_matrix_for_x_channel():
    """X channel: Choi = (I ⊗ X)|Φ+⟩⟨Φ+|(I ⊗ X)†."""
    from qbit_simulator.tomography import choi_matrix_from_kraus
    from qbit_simulator.gates import X
    J = choi_matrix_from_kraus([X])
    # (I⊗X)|Φ+⟩ = (|01⟩+|10⟩)/√2 = |Ψ+⟩
    expected = np.zeros((4, 4), dtype=np.complex128)
    expected[1, 1] = 0.5; expected[1, 2] = 0.5
    expected[2, 1] = 0.5; expected[2, 2] = 0.5
    assert np.allclose(J, expected, atol=1e-10)


def test_process_tomography_identity_channel():
    """Tomographically characterize the identity channel."""
    from qbit_simulator.tomography import process_tomography_single_qubit
    # Identity channel: returns input ρ unchanged.
    result = process_tomography_single_qubit(
        channel_fn=lambda rho: rho, shots=None,
    )
    # Output states should match inputs.
    assert np.allclose(result["output_states"]["0"],
                       np.array([[1, 0], [0, 0]], dtype=np.complex128),
                       atol=1e-10)
    assert np.allclose(result["output_states"]["1"],
                       np.array([[0, 0], [0, 1]], dtype=np.complex128),
                       atol=1e-10)


def test_process_tomography_bit_flip_channel():
    """X channel: every output state should be the X-conjugated input."""
    from qbit_simulator.tomography import process_tomography_single_qubit
    from qbit_simulator.gates import X
    result = process_tomography_single_qubit(
        channel_fn=lambda rho: X @ rho @ X.conj().T,
        shots=None,
    )
    # X|0⟩⟨0|X = |1⟩⟨1|
    assert np.allclose(result["output_states"]["0"],
                       np.array([[0, 0], [0, 1]], dtype=np.complex128),
                       atol=1e-10)
    # X|1⟩⟨1|X = |0⟩⟨0|
    assert np.allclose(result["output_states"]["1"],
                       np.array([[1, 0], [0, 0]], dtype=np.complex128),
                       atol=1e-10)


def test_state_fidelity_orthogonal():
    rho_a = np.outer(np.array([1, 0], dtype=np.complex128),
                     np.array([1, 0], dtype=np.complex128).conj())
    rho_b = np.outer(np.array([0, 1], dtype=np.complex128),
                     np.array([0, 1], dtype=np.complex128).conj())
    assert state_fidelity(rho_a, rho_b) < 1e-10
