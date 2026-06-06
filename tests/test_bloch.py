"""Bloch sphere tests."""

import numpy as np
import pytest

from qbit_simulator.bloch import (
    bloch_vector, bloch_vector_from_multiqubit,
    ascii_bloch, purity, state_purity_from_bloch,
)


# ---- Pure state Bloch vectors ----

def test_bloch_zero_state():
    psi = np.array([1, 0], dtype=complex)
    r = bloch_vector(psi)
    assert np.allclose(r, [0, 0, 1])


def test_bloch_one_state():
    psi = np.array([0, 1], dtype=complex)
    r = bloch_vector(psi)
    assert np.allclose(r, [0, 0, -1])


def test_bloch_plus_state():
    psi = np.array([1, 1], dtype=complex) / np.sqrt(2)
    r = bloch_vector(psi)
    assert np.allclose(r, [1, 0, 0])


def test_bloch_minus_state():
    psi = np.array([1, -1], dtype=complex) / np.sqrt(2)
    r = bloch_vector(psi)
    assert np.allclose(r, [-1, 0, 0])


def test_bloch_right_circular():
    """|R⟩ = (|0⟩ + i|1⟩) / √2 maps to +Y."""
    psi = np.array([1, 1j], dtype=complex) / np.sqrt(2)
    r = bloch_vector(psi)
    assert np.allclose(r, [0, 1, 0])


def test_pure_states_on_surface():
    rng = np.random.default_rng(0)
    for _ in range(10):
        z = rng.normal(size=2) + 1j * rng.normal(size=2)
        psi = z / np.linalg.norm(z)
        r = bloch_vector(psi)
        assert abs(np.linalg.norm(r) - 1.0) < 1e-12


def test_mixed_state_inside_ball():
    rho = np.eye(2, dtype=complex) / 2
    r = bloch_vector(rho)
    assert np.allclose(r, [0, 0, 0])


# ---- Density matrix input ----

def test_bloch_density_matrix_z_state():
    rho = np.array([[1, 0], [0, 0]], dtype=complex)
    r = bloch_vector(rho)
    assert np.allclose(r, [0, 0, 1])


def test_bloch_rejects_bad_shape():
    with pytest.raises(ValueError):
        bloch_vector(np.zeros(3))


# ---- Multi-qubit reduction ----

def test_bell_state_reduces_to_maximally_mixed():
    """For |Phi+⟩ = (|00⟩ + |11⟩)/√2, each qubit alone is I/2."""
    psi = np.zeros(4, dtype=complex)
    psi[0b00] = 1 / np.sqrt(2)
    psi[0b11] = 1 / np.sqrt(2)
    for q in [0, 1]:
        r = bloch_vector_from_multiqubit(psi, 2, q)
        assert np.allclose(r, [0, 0, 0], atol=1e-12)


def test_product_state_recovers_factor():
    """For kron(plus, zero) (MSB-first): qubit 0 = |+⟩, qubit 1 = |0⟩."""
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    zero = np.array([1, 0], dtype=complex)
    # MSB-first convention: in kron(plus, zero), qubit 0 is `plus` and
    # qubit 1 is `zero` (matches circuit.py).
    psi = np.kron(plus, zero)
    r0 = bloch_vector_from_multiqubit(psi, 2, 0)
    r1 = bloch_vector_from_multiqubit(psi, 2, 1)
    assert np.allclose(r0, [1, 0, 0])    # plus state
    assert np.allclose(r1, [0, 0, 1])    # zero state


# ---- Purity ----

def test_purity_pure_state_is_one():
    psi = np.array([1, 0], dtype=complex)
    rho = np.outer(psi, psi.conj())
    assert abs(purity(rho) - 1.0) < 1e-12


def test_purity_maximally_mixed_is_half():
    rho = np.eye(2, dtype=complex) / 2
    assert abs(purity(rho) - 0.5) < 1e-12


def test_state_purity_from_bloch_matches():
    rng = np.random.default_rng(0)
    for _ in range(5):
        z = rng.normal(size=2) + 1j * rng.normal(size=2)
        psi = z / np.linalg.norm(z)
        rho = np.outer(psi, psi.conj())
        p_direct = purity(rho)
        r = bloch_vector(psi)
        p_bloch = state_purity_from_bloch(r)
        assert abs(p_direct - p_bloch) < 1e-12


# ---- ASCII rendering ----

def test_ascii_bloch_returns_string():
    r = np.array([1, 0, 0])
    out = ascii_bloch(r)
    assert isinstance(out, str)
    assert "Bloch" in out
    assert "|r| = 1.000" in out


def test_ascii_bloch_handles_zero_vector():
    r = np.array([0, 0, 0])
    out = ascii_bloch(r)
    assert "|r| = 0.000" in out
