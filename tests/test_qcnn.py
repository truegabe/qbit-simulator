"""Quantum convolutional NN tests."""

import numpy as np
import pytest

from qbit_simulator.algorithms.qcnn import (
    convolution_unitary, pool_unitary,
    apply_qcnn, qcnn_output, qcnn_predict,
    train_qcnn_classifier,
)


# ---- Building blocks ----

def test_convolution_unitary_shape():
    U = convolution_unitary(np.array([0.5, 0.3, -0.1]))
    assert U.shape == (4, 4)


def test_convolution_unitary_is_unitary():
    rng = np.random.default_rng(0)
    for _ in range(5):
        theta = rng.uniform(-1, 1, size=3)
        U = convolution_unitary(theta)
        assert np.allclose(U @ U.conj().T, np.eye(4), atol=1e-9)


def test_convolution_unitary_rejects_wrong_params():
    with pytest.raises(ValueError):
        convolution_unitary(np.array([0.5, 0.3]))


def test_pool_unitary_shape():
    U = pool_unitary(np.array([0.1, 0.2]))
    assert U.shape == (4, 4)


def test_pool_unitary_is_unitary():
    rng = np.random.default_rng(0)
    for _ in range(5):
        theta = rng.uniform(-1, 1, size=2)
        U = pool_unitary(theta)
        assert np.allclose(U @ U.conj().T, np.eye(4), atol=1e-9)


def test_pool_unitary_rejects_wrong_params():
    with pytest.raises(ValueError):
        pool_unitary(np.array([0.5]))


# ---- QCNN forward ----

def test_apply_qcnn_preserves_norm():
    rng = np.random.default_rng(0)
    n_qubits = 4; n_layers = 2
    psi = rng.normal(size=2**n_qubits) + 1j * rng.normal(size=2**n_qubits)
    psi /= np.linalg.norm(psi)
    theta = rng.uniform(-0.5, 0.5, size=5 * n_layers)
    out = apply_qcnn(theta, psi, n_qubits, n_layers)
    assert abs(np.linalg.norm(out) - 1.0) < 1e-9


def test_apply_qcnn_rejects_wrong_theta_length():
    psi = np.zeros(16, dtype=complex); psi[0] = 1.0
    with pytest.raises(ValueError):
        apply_qcnn(np.zeros(7), psi, n_qubits=4, n_layers=2)


def test_apply_qcnn_rejects_too_few_qubits():
    psi = np.zeros(4, dtype=complex); psi[0] = 1.0
    # n_layers = 2 → need ≥ 4 qubits; with 2 qubits this fails.
    with pytest.raises(ValueError):
        apply_qcnn(np.zeros(10), psi, n_qubits=2, n_layers=2)


def test_qcnn_output_in_range():
    """⟨Z⟩ must be in [-1, +1]."""
    rng = np.random.default_rng(0)
    psi = np.zeros(16, dtype=complex); psi[0] = 1.0
    theta = rng.uniform(-1, 1, size=10)
    val = qcnn_output(theta, psi, n_qubits=4, n_layers=2)
    assert -1.0 - 1e-9 <= val <= 1.0 + 1e-9


def test_qcnn_predict_returns_pm_one():
    rng = np.random.default_rng(0)
    psi = np.zeros(16, dtype=complex); psi[0] = 1.0
    theta = rng.uniform(-1, 1, size=10)
    p = qcnn_predict(theta, psi, n_qubits=4, n_layers=2)
    assert p in (-1, 1)


# ---- Training ----

def test_qcnn_trains_on_binary_basis_states():
    """Distinguish |0000⟩ from |1111⟩."""
    rng = np.random.default_rng(0)
    states = [np.zeros(16, dtype=complex) for _ in range(2)]
    states[0][0] = 1.0
    states[1][15] = 1.0
    y = np.array([+1, -1])
    result = train_qcnn_classifier(states, y, n_qubits=4, n_layers=2,
                                      n_iter=30, lr=0.3, rng=rng)
    assert result["final_loss"] < 0.1
    for s, yi in zip(states, y):
        assert qcnn_predict(result["params"], s, 4, 2) == yi


def test_qcnn_training_returns_structure():
    rng = np.random.default_rng(0)
    psi = np.zeros(16, dtype=complex); psi[0] = 1.0
    result = train_qcnn_classifier([psi], np.array([+1]),
                                      n_qubits=4, n_layers=2,
                                      n_iter=3, rng=rng)
    assert "params" in result
    assert "loss_history" in result
    assert "final_loss" in result


def test_qcnn_loss_history_decreases():
    rng = np.random.default_rng(0)
    states = [np.zeros(16, dtype=complex) for _ in range(2)]
    states[0][0] = 1.0; states[1][15] = 1.0
    y = np.array([+1, -1])
    result = train_qcnn_classifier(states, y, n_qubits=4, n_layers=2,
                                      n_iter=20, lr=0.2, rng=rng)
    assert result["loss_history"][-1] <= result["loss_history"][0]
