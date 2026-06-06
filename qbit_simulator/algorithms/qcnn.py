"""Quantum convolutional neural networks (QCNN).

QCNNs (Cong-Choi-Lukin 2019) mirror the classical CNN architecture:

  * **Convolution layer**: a parameterized 2-qubit unitary applied to
    nearest-neighbor pairs across the qubit register, weight-shared.
  * **Pooling layer**: a controlled rotation that contracts pairs of
    qubits into one (we implement this by tracing out half the qubits
    while applying a conditional gate).
  * Iterate conv → pool until one qubit remains; measure that qubit
    to produce a classification output.

QCNNs are interesting because:

  * They have NO barren plateaus (Pesah-Cerezo-Wang-Volkoff-Sornborger-
    Coles 2021): the variance of gradients does not vanish with system
    size, unlike generic VQE ansätze.
  * They have efficient training scaling, similar to classical CNNs.
  * They naturally suit translationally-symmetric inputs (e.g.
    classifying SPT phases of a 1D quantum state).

This module provides a small QCNN implementation:

  - `convolution_unitary(theta)`: a 2-qubit gate (3 parameters).
  - `pool_unitary(theta)`: 2-qubit controlled rotation, then trace.
  - `apply_qcnn(theta, state, n_layers)`: full QCNN circuit.
  - `train_qcnn_classifier(X_states, y, ...)`: train on labeled quantum
    states.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


# ----------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------

def _ry(theta: float) -> np.ndarray:
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def _rz(theta: float) -> np.ndarray:
    return np.array([
        [np.exp(-1j * theta / 2), 0],
        [0, np.exp(1j * theta / 2)],
    ], dtype=np.complex128)


def convolution_unitary(theta: np.ndarray) -> np.ndarray:
    """A 3-parameter 2-qubit unitary used as the conv kernel.

        U(θ) = (Ry(θ_0) ⊗ Ry(θ_1)) · CNOT(0→1) · (I ⊗ Rz(θ_2))

    Same structure as Cong-Choi-Lukin's "U_conv" up to parameter
    relabeling. 4×4 matrix.
    """
    if len(theta) != 3:
        raise ValueError("conv unitary takes 3 parameters")
    cnot = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ], dtype=np.complex128)
    return (np.kron(_ry(theta[0]), _ry(theta[1])) @ cnot
            @ np.kron(np.eye(2, dtype=np.complex128), _rz(theta[2])))


def pool_unitary(theta: np.ndarray) -> np.ndarray:
    """The QCNN "pool" gate: controlled rotation on (control, target).

        U_pool(θ) =  |0⟩⟨0|_ctrl ⊗ Ry(θ_0)  +  |1⟩⟨1|_ctrl ⊗ Ry(θ_1)

    After applying, we (mentally) trace out the control qubit; in our
    state-vector simulator we simply ignore it for downstream conv layers.
    """
    if len(theta) != 2:
        raise ValueError("pool unitary takes 2 parameters")
    return np.kron(np.diag([1, 0]).astype(complex), _ry(theta[0])) \
            + np.kron(np.diag([0, 1]).astype(complex), _ry(theta[1]))


# ----------------------------------------------------------------------------
# State-vector helpers (re-used from ssvqe-style ansatz)
# ----------------------------------------------------------------------------

def _apply_2q_gate(psi: np.ndarray, gate: np.ndarray, q0: int, q1: int,
                    n: int) -> np.ndarray:
    """Apply a 4x4 gate on qubits (q0, q1) (MSB-first: axis q = qubit q)."""
    if q1 < q0:
        swap = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ], dtype=np.complex128)
        gate = swap @ gate @ swap
        q0, q1 = q1, q0
    shape = [2] * n
    arr = psi.reshape(shape)
    arr = np.moveaxis(arr, [q0, q1], [0, 1])
    arr = arr.reshape(4, -1)
    arr = gate @ arr
    arr = arr.reshape([2, 2] + [2] * (n - 2))
    arr = np.moveaxis(arr, [0, 1], [q0, q1])
    return arr.reshape(2 ** n)


# ----------------------------------------------------------------------------
# QCNN forward pass
# ----------------------------------------------------------------------------

def apply_qcnn(theta: np.ndarray, state: np.ndarray, n_qubits: int,
                n_layers: int) -> np.ndarray:
    """Apply a QCNN of `n_layers` (conv + pool) to the input state.

    Each conv layer has weight-sharing: ALL nearest-neighbor pairs
    use the same 3-parameter conv unitary. Each pool layer has its
    own 2-parameter pool unitary, applied to (q, q+1) for q = 0, 2, 4, …
    keeping the qubits at positions 0, 2, 4, … as the "downsampled"
    output for the next layer.

    Parameter count: n_layers · (3 + 2) = 5 · n_layers.

    Args:
        theta:    parameter vector (length 5 · n_layers).
        state:    input state vector on n_qubits.
        n_qubits: total qubits (must be ≥ 2^n_layers).
        n_layers: number of (conv, pool) iterations.

    Returns:
        the transformed state vector.
    """
    if len(theta) != 5 * n_layers:
        raise ValueError(f"theta must be 5 · n_layers = {5 * n_layers}")
    if n_qubits < 2 ** n_layers:
        raise ValueError(
            f"need n_qubits ≥ 2^n_layers = {2 ** n_layers}"
        )
    psi = state.copy()
    active = list(range(n_qubits))   # qubits still in play
    for L in range(n_layers):
        conv_p = theta[5 * L:5 * L + 3]
        pool_p = theta[5 * L + 3:5 * L + 5]
        U_conv = convolution_unitary(conv_p)
        U_pool = pool_unitary(pool_p)
        # Convolution: apply conv to each nearest-neighbor pair in `active`.
        for i in range(len(active) - 1):
            psi = _apply_2q_gate(psi, U_conv, active[i], active[i + 1], n_qubits)
        # Pooling: apply pool to (a[2k], a[2k+1]); drop the second.
        new_active = []
        for k in range(len(active) // 2):
            psi = _apply_2q_gate(
                psi, U_pool, active[2 * k], active[2 * k + 1], n_qubits,
            )
            new_active.append(active[2 * k])
        active = new_active
    return psi


def qcnn_output(theta: np.ndarray, state: np.ndarray, n_qubits: int,
                 n_layers: int) -> float:
    """Classification output: ⟨Z⟩ on the final (highest-MSB) surviving
    qubit after the QCNN."""
    out_state = apply_qcnn(theta, state, n_qubits, n_layers)
    # Z on qubit 0 (the only one left in `active`).
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    # Build Z⊗I⊗...⊗I.
    M = Z
    for _ in range(n_qubits - 1):
        M = np.kron(M, np.eye(2, dtype=np.complex128))
    return float(np.real(out_state.conj() @ M @ out_state))


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------

def train_qcnn_classifier(
    X_states: list[np.ndarray],
    y: np.ndarray,
    n_qubits: int,
    n_layers: int = 2,
    n_iter: int = 50,
    lr: float = 0.2,
    rng: np.random.Generator | None = None,
) -> dict:
    """Train a QCNN classifier on labeled quantum-state inputs.

    Args:
        X_states: list of state vectors (each on n_qubits).
        y:        labels ∈ {-1, +1}.
        n_qubits: total qubits.
        n_layers: number of conv+pool iterations.
        n_iter:   training iterations.
        lr:       gradient-descent step size.
        rng:      generator.

    Loss: mean-squared error vs y over the dataset.

    Returns:
        dict with trained params, loss history, final loss.
    """
    rng = rng or np.random.default_rng()
    n_params = 5 * n_layers
    theta = rng.uniform(-0.3, 0.3, size=n_params)

    def loss(t):
        total = 0.0
        for psi, yi in zip(X_states, y):
            out = qcnn_output(t, psi, n_qubits, n_layers)
            total += (out - yi) ** 2
        return total / len(X_states)

    eps = np.pi / 6
    history = [loss(theta)]
    for it in range(n_iter):
        grad = np.zeros_like(theta)
        for k in range(n_params):
            t_plus = theta.copy(); t_plus[k] += eps
            t_minus = theta.copy(); t_minus[k] -= eps
            grad[k] = (loss(t_plus) - loss(t_minus)) / (2 * eps)
        theta = theta - lr * grad
        history.append(loss(theta))

    return {
        "params":         theta,
        "loss_history":   history,
        "final_loss":     history[-1],
    }


def qcnn_predict(theta: np.ndarray, state: np.ndarray,
                  n_qubits: int, n_layers: int) -> int:
    """Predict ±1 label from the sign of the QCNN output."""
    out = qcnn_output(theta, state, n_qubits, n_layers)
    return 1 if out > 0 else -1
