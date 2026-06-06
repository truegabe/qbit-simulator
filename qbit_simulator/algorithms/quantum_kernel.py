"""Quantum kernel methods — encode classical data into quantum states, then
use state overlaps as kernel values for classical ML algorithms.

A quantum feature map is a parameterized circuit U_φ(x) that prepares a
state |φ(x)⟩ = U_φ(x)|0...0⟩. The quantum kernel is

    K(x, x') = |⟨φ(x) | φ(x')⟩|²

For specific feature-map families, K is believed to be classically hard to
estimate (the basis for "quantum-kernel-method" speedup claims). For other
choices, K reduces to a classical RBF or polynomial kernel — useful for
benchmarking.

We implement the standard ZZ-feature-map (Havlíček et al., Nature 2019):

    U_φ(x) = exp(i Σ_{i<j} (π-x_i)(π-x_j) Z_i Z_j) · exp(i Σ_i x_i Z_i)
             · H^⊗N · exp(i Σ_{i<j} (π-x_i)(π-x_j) Z_i Z_j) · exp(i Σ_i x_i Z_i)
             · H^⊗N

This module provides the feature map, the kernel computation, and a tiny
end-to-end demonstration (kernel matrix on a labeled toy dataset).
"""

from __future__ import annotations

import numpy as np

from ..circuit import QuantumCircuit


def zz_feature_map(x: np.ndarray, reps: int = 2) -> QuantumCircuit:
    """Build the ZZ-feature-map circuit for input vector x ∈ ℝⁿ.

    Args:
        x:    1-D feature vector. The circuit acts on len(x) qubits.
        reps: number of feature-map layers (depth knob).

    Returns:
        QuantumCircuit such that qc.state = |φ(x)⟩.
    """
    n = len(x)
    qc = QuantumCircuit(n)
    for _ in range(reps):
        # Layer of Hadamards.
        for q in range(n):
            qc.h(q)
        # Single-qubit phase rotations: exp(i x_q Z_q).
        for q in range(n):
            qc.rz(2.0 * float(x[q]), q)
        # Two-qubit ZZ entanglers: exp(i (π-x_i)(π-x_j) Z_i Z_j).
        for i in range(n - 1):
            for j in range(i + 1, n):
                theta = 2.0 * (np.pi - float(x[i])) * (np.pi - float(x[j]))
                # exp(iθ Z_i Z_j) = CNOT(i,j) · RZ(2θ, j) · CNOT(i,j)
                qc.cnot(i, j)
                qc.rz(theta, j)
                qc.cnot(i, j)
    return qc


def quantum_kernel_value(x: np.ndarray, y: np.ndarray, reps: int = 2) -> float:
    """K(x, y) = |⟨φ(x) | φ(y)⟩|² for the ZZ feature map."""
    if len(x) != len(y):
        raise ValueError("inputs must have same dimension")
    psi_x = zz_feature_map(x, reps=reps).state
    psi_y = zz_feature_map(y, reps=reps).state
    return float(abs(np.vdot(psi_x, psi_y)) ** 2)


def quantum_kernel_matrix(
    X: np.ndarray,
    Y: np.ndarray | None = None,
    reps: int = 2,
) -> np.ndarray:
    """Pairwise kernel matrix K[i, j] = K(X[i], Y[j]).

    If Y is None, compute the symmetric matrix K(X, X).
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be 2-D: (n_samples, n_features)")
    n_x = X.shape[0]
    if Y is None:
        # Cache feature states.
        states = [zz_feature_map(X[i], reps=reps).state for i in range(n_x)]
        K = np.zeros((n_x, n_x), dtype=np.float64)
        for i in range(n_x):
            K[i, i] = 1.0
            for j in range(i + 1, n_x):
                K[i, j] = float(abs(np.vdot(states[i], states[j])) ** 2)
                K[j, i] = K[i, j]
        return K
    Y = np.asarray(Y, dtype=np.float64)
    n_y = Y.shape[0]
    states_x = [zz_feature_map(X[i], reps=reps).state for i in range(n_x)]
    states_y = [zz_feature_map(Y[j], reps=reps).state for j in range(n_y)]
    K = np.zeros((n_x, n_y), dtype=np.float64)
    for i in range(n_x):
        for j in range(n_y):
            K[i, j] = float(abs(np.vdot(states_x[i], states_y[j])) ** 2)
    return K
