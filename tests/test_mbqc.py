"""Measurement-based quantum computation tests."""

import numpy as np
import pytest

from qbit_simulator.algorithms.mbqc import (
    cluster_state, chain_cluster_state,
    measure_in_xy_basis,
    mbqc_single_qubit_rotation, mbqc_fidelity,
    _apply_cz, _apply_1q,
)


# ---- Cluster state ----

def test_cluster_state_normalized():
    psi = cluster_state(2, 3)
    assert abs(np.linalg.norm(psi) - 1.0) < 1e-10


def test_cluster_state_correct_size():
    psi = cluster_state(2, 3)
    assert psi.shape == (64,)


def test_chain_cluster_state_eq_one_row():
    psi_chain = chain_cluster_state(3)
    psi_grid = cluster_state(1, 3)
    assert np.allclose(psi_chain, psi_grid, atol=1e-12)


def test_two_qubit_cluster_state_is_bell():
    """The 2-qubit cluster state equals the Bell-like state
    (|0+⟩ + |1−⟩)/√2 (which is (|00⟩+|01⟩+|10⟩−|11⟩)/2)."""
    psi = chain_cluster_state(2)
    expected = np.array([1, 1, 1, -1], dtype=complex) / 2.0
    assert np.allclose(psi, expected, atol=1e-12)


# ---- Measurement primitives ----

def test_measure_in_xy_returns_bit():
    rng = np.random.default_rng(0)
    psi = chain_cluster_state(2)
    final, outcome = measure_in_xy_basis(psi, qubit=0, angle=0.5, rng=rng)
    assert outcome in (0, 1)
    assert abs(np.linalg.norm(final) - 1.0) < 1e-9


def test_measure_x_basis_on_plus_always_zero():
    """|+⟩ in X basis (angle=0) always gives outcome 0."""
    rng = np.random.default_rng(0)
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    for _ in range(20):
        _, outcome = measure_in_xy_basis(plus, qubit=0, angle=0.0, rng=rng)
        assert outcome == 0


def test_measure_x_basis_on_minus_always_one():
    """|−⟩ in X basis always gives outcome 1."""
    rng = np.random.default_rng(0)
    minus = np.array([1, -1], dtype=complex) / np.sqrt(2)
    for _ in range(20):
        _, outcome = measure_in_xy_basis(minus, qubit=0, angle=0.0, rng=rng)
        assert outcome == 1


# ---- Single-qubit rotation ----

def test_mbqc_rotation_zero_angle_is_identity():
    """Rz(0) = I."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        r = mbqc_single_qubit_rotation(
            0.0, np.array([1, 0], dtype=complex), rng=rng,
        )
        fid = abs(np.vdot(r["ideal_state"], r["output_state"])) ** 2
        assert fid > 0.99


def test_mbqc_rotation_fidelity_one_on_basis_state():
    """For input |0⟩, Rz(θ)|0⟩ = e^{-iθ/2}|0⟩, fidelity = 1."""
    rng = np.random.default_rng(0)
    fid = mbqc_fidelity(np.pi / 3, np.array([1, 0], dtype=complex),
                          n_trials=20, rng=rng)
    assert fid > 0.99


@pytest.mark.parametrize("theta", [0.0, 0.5, np.pi / 4, np.pi / 2, np.pi])
def test_mbqc_rotation_fidelity_on_plus_state(theta):
    """For input |+⟩, Rz(θ) gives a nontrivial state; check fidelity."""
    rng = np.random.default_rng(0)
    fid = mbqc_fidelity(theta, np.array([1, 1], dtype=complex) / np.sqrt(2),
                          n_trials=20, rng=rng)
    assert fid > 0.99


# ---- CZ helper ----

def test_cz_acts_on_11_with_phase():
    """CZ |11⟩ = -|11⟩, |00⟩→|00⟩."""
    psi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    out = _apply_cz(psi, 0, 1, n=2)
    # (|00⟩ + |11⟩)/√2 → (|00⟩ - |11⟩)/√2.
    expected = np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2)
    assert np.allclose(out, expected, atol=1e-12)


def test_cz_preserves_norm():
    rng = np.random.default_rng(0)
    psi = rng.normal(size=4) + 1j * rng.normal(size=4)
    psi /= np.linalg.norm(psi)
    out = _apply_cz(psi, 0, 1, n=2)
    assert abs(np.linalg.norm(out) - 1.0) < 1e-12
