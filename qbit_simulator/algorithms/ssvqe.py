"""SSVQE: Subspace-Search Variational Quantum Eigensolver.

Standard VQE only finds the ground state by minimizing ⟨ψ| H |ψ⟩.
SSVQE (Nakanishi-Mitarai-Fujii 2019) finds the k LOWEST eigenstates by
optimizing a WEIGHTED sum of energies on k mutually-orthogonal trial
states:

    L(θ) = sum_{i=0}^{k-1} w_i · ⟨ψ_i(θ) | H | ψ_i(θ)⟩

with weights w_0 > w_1 > … > w_{k-1} > 0 and the trial states obtained
from k orthogonal reference states |φ_i⟩ followed by the SAME
parameterized ansatz U(θ):

    |ψ_i(θ)⟩ = U(θ) |φ_i⟩

Because U(θ) is unitary and the |φ_i⟩ are orthogonal, the |ψ_i(θ)⟩
remain orthogonal for all θ. At the optimum, ψ_0 → ground state,
ψ_1 → first excited state, etc.

This module:

  - `ssvqe(H, ansatz_apply, k, weights, init_params)`: run the SSVQE
    optimization.
  - `pauli_op_to_matrix(op, n)`: helper for evaluating ⟨H⟩ on a state.

We pass the user-supplied `ansatz_apply(params, ref_state)` so this is
flexible: any parameterized circuit acting on a state vector works.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..pauli import PauliOp


def _pauli_string_matrix(s: str) -> np.ndarray:
    _I = np.eye(2, dtype=np.complex128)
    _X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    _Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    _Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    table = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}
    M = np.array([[1.0 + 0j]])
    for ch in s:
        M = np.kron(M, table[ch])
    return M


def pauli_op_to_matrix(op: PauliOp, n: int) -> np.ndarray:
    """Build the dense matrix of a PauliOp on n qubits."""
    dim = 2 ** n
    M = np.zeros((dim, dim), dtype=np.complex128)
    for coef, s in op.terms:
        M = M + coef * _pauli_string_matrix(s)
    return M


# ----------------------------------------------------------------------------
# SSVQE
# ----------------------------------------------------------------------------

def ssvqe(
    hamiltonian: np.ndarray,
    ansatz_apply: Callable[[np.ndarray, np.ndarray], np.ndarray],
    references: list[np.ndarray],
    weights: list[float],
    init_params: np.ndarray,
    optimizer: str = "BFGS",
    max_iter: int = 500,
) -> dict:
    """Run the SSVQE optimization.

    Args:
        hamiltonian:   dense matrix of H (2^n × 2^n).
        ansatz_apply:  callable (params, ref) → ψ. Must apply the same
                       unitary for all refs to preserve orthogonality.
        references:    list of k orthogonal reference state vectors.
        weights:       list of k positive weights, w_0 > w_1 > ….
        init_params:   initial guess for the variational parameters.
        optimizer:     scipy method name.
        max_iter:      maximum optimizer iterations.

    Returns:
        dict with "energies" (k optimized eigenvalues, in order),
        "states" (k state vectors), "params" (optimal θ),
        "loss_history" (list of loss values per iter).
    """
    from scipy.optimize import minimize

    k = len(references)
    if len(weights) != k:
        raise ValueError("len(weights) must equal len(references)")
    if not all(weights[i] > weights[i + 1] > 0 for i in range(k - 1)):
        raise ValueError("weights must be positive and strictly decreasing")

    # Verify orthogonality of references.
    for i in range(k):
        for j in range(i + 1, k):
            overlap = abs(np.vdot(references[i], references[j]))
            if overlap > 1e-9:
                raise ValueError(f"reference {i} and {j} are not orthogonal")

    history = []

    def loss(params):
        total = 0.0
        for w, ref in zip(weights, references):
            psi = ansatz_apply(params, ref)
            E = float(np.real(psi.conj() @ hamiltonian @ psi))
            total += w * E
        history.append(total)
        return total

    res = minimize(loss, init_params, method=optimizer,
                   options={"maxiter": max_iter})
    params_opt = res.x

    # Extract the k states and their energies.
    states = [ansatz_apply(params_opt, ref) for ref in references]
    energies = [float(np.real(psi.conj() @ hamiltonian @ psi))
                for psi in states]

    return {
        "energies":      energies,
        "states":        states,
        "params":        params_opt,
        "loss_history":  history,
        "loss_final":    float(res.fun),
        "success":       res.success,
    }


# ----------------------------------------------------------------------------
# Convenience: simple parameterized ansatz on n qubits
# ----------------------------------------------------------------------------

def hardware_efficient_ansatz_apply(n_qubits: int, depth: int = 2):
    """Return a callable (params, ref) → state that applies a
    hardware-efficient ansatz: alternating layers of Ry rotations and
    nearest-neighbor CNOT entanglers.

    The returned closure expects len(params) = depth · n_qubits + n_qubits
    (one Ry angle per qubit per layer, plus a final Ry layer).
    """
    n_params_per_layer = n_qubits
    n_layers = depth + 1   # +1 for the final Ry layer

    def apply(params: np.ndarray, ref: np.ndarray) -> np.ndarray:
        if len(params) != n_layers * n_params_per_layer:
            raise ValueError(
                f"expected {n_layers * n_params_per_layer} params, "
                f"got {len(params)}"
            )
        psi = ref.copy()
        for L in range(n_layers):
            # Ry layer on each qubit.
            for q in range(n_qubits):
                theta = params[L * n_qubits + q]
                psi = _apply_single_qubit_gate(psi, _ry(theta), q, n_qubits)
            # CNOT layer (skip on the final layer).
            if L < n_layers - 1:
                for q in range(n_qubits - 1):
                    psi = _apply_cnot(psi, q, q + 1, n_qubits)
        return psi

    return apply, n_layers * n_params_per_layer


def _ry(theta: float) -> np.ndarray:
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def _apply_single_qubit_gate(psi: np.ndarray, gate: np.ndarray, q: int,
                              n: int) -> np.ndarray:
    """Apply a 2x2 gate on qubit q (MSB-first: axis q = qubit q)."""
    shape = [2] * n
    psi_t = psi.reshape(shape)
    psi_t = np.moveaxis(psi_t, q, 0)
    psi_t = psi_t.reshape(2, -1)
    psi_t = gate @ psi_t
    psi_t = psi_t.reshape([2] + [2] * (n - 1))
    psi_t = np.moveaxis(psi_t, 0, q)
    return psi_t.reshape(2 ** n)


def _apply_cnot(psi: np.ndarray, control: int, target: int,
                 n: int) -> np.ndarray:
    """Apply CNOT (control, target) — bit-flip on target when control=1."""
    new_psi = psi.copy()
    for idx in range(2 ** n):
        # Bit positions: qubit k = bit (n-1-k) in MSB-first.
        c_bit = (idx >> (n - 1 - control)) & 1
        if c_bit == 1:
            flipped = idx ^ (1 << (n - 1 - target))
            new_psi[idx] = psi[flipped]
    return new_psi
