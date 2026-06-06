"""Quantum autoencoders.

A quantum autoencoder (Romero-Olson-Aspuru-Guzik 2017) compresses an
n-qubit input state into k qubits (with k < n) — analogous to a
classical autoencoder. Procedure:

  * Inputs: a set of n-qubit "training" states {|ψ_i⟩}.
  * Architecture: an encoder U(θ) is a parameterized n-qubit unitary.
    After applying U, we trace out the last (n − k) qubits, calling
    them the "trash" register. The remaining k qubits form the
    compressed code.
  * Training objective: maximize the OVERLAP of the trash register
    with the all-|0⟩ state — equivalent to forcing all useful
    information into the k retained qubits, since unitary evolution
    is information-preserving.
  * Decoder: U†(θ) reconstructs the original.

Loss function: average over training samples of

    L(θ) = 1 − ⟨0...0 | Tr_code(U(θ) |ψ_i⟩⟨ψ_i| U†(θ)) | 0...0⟩

If L=0 → perfect compression for that batch.

Provides:

  - `train_autoencoder(states, n_keep, ...)`: train an encoder.
  - `encode(state, encoder_params, ansatz)`: apply U(θ).
  - `decode(compressed_state, encoder_params, ansatz)`: apply U(θ)†.
  - `compression_fidelity(state, encoder_params, ansatz, n_keep)`:
    fidelity of round-tripped reconstruction.

We use the hardware-efficient ansatz from `ssvqe.py`.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .ssvqe import hardware_efficient_ansatz_apply


# ----------------------------------------------------------------------------
# Trash-state fidelity (loss building block)
# ----------------------------------------------------------------------------

def _reduced_density_matrix(psi: np.ndarray, n_total: int,
                              keep_qubits: list[int]) -> np.ndarray:
    """Trace out all qubits not in `keep_qubits`; return reduced ρ.

    keep_qubits are indexed MSB-first (qubit 0 = leftmost).
    """
    n_keep = len(keep_qubits)
    if 2 ** n_total != len(psi):
        raise ValueError("state length must be 2^n_total")
    # Move kept qubits to the front, then reshape to (d_keep, d_rest).
    arr = psi.reshape([2] * n_total)
    # Bring kept axes to the front in order.
    perm = list(keep_qubits) + [q for q in range(n_total) if q not in keep_qubits]
    arr = np.transpose(arr, perm)
    arr = arr.reshape(2 ** n_keep, 2 ** (n_total - n_keep))
    return arr @ arr.conj().T


def trash_overlap(
    state: np.ndarray,
    encoder_params: np.ndarray,
    ansatz: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_qubits: int,
    n_keep: int,
) -> float:
    """Fidelity that the trash register lands on |0...0⟩ after encoding.

    Args:
        state:           input state vector on n_qubits.
        encoder_params:  ansatz parameters.
        ansatz:          encoder unitary (params, ref) → state.
        n_qubits:        total qubits.
        n_keep:          number of "code" qubits (the first n_keep MSB-
                         indexed qubits are kept).
    """
    encoded = ansatz(encoder_params, state)
    # Trash register: the last n_qubits − n_keep qubits.
    trash_qubits = list(range(n_keep, n_qubits))
    rho_trash = _reduced_density_matrix(encoded, n_qubits, trash_qubits)
    # ⟨0...0 | rho_trash | 0...0⟩ = top-left entry.
    return float(np.real(rho_trash[0, 0]))


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------

def train_autoencoder(
    states: list[np.ndarray],
    n_qubits: int,
    n_keep: int,
    depth: int = 3,
    n_iter: int = 100,
    lr: float = 0.2,
    rng: np.random.Generator | None = None,
) -> dict:
    """Train a quantum autoencoder.

    Args:
        states:    list of input state vectors (all on n_qubits).
        n_qubits:  total qubit count.
        n_keep:    code-register size.
        depth:     ansatz depth.
        n_iter:    optimizer iterations.
        lr:        learning rate (gradient descent).
        rng:       generator.

    Returns:
        dict with optimized params, training loss history, final
        compression fidelity.
    """
    rng = rng or np.random.default_rng()
    ansatz, n_params = hardware_efficient_ansatz_apply(n_qubits, depth=depth)
    params = rng.uniform(-0.5, 0.5, size=n_params)

    def loss(theta):
        # Average 1 − ⟨0...0|trash|0...0⟩ over training set.
        L = 0.0
        for psi in states:
            L += 1.0 - trash_overlap(psi, theta, ansatz, n_qubits, n_keep)
        return L / len(states)

    # Gradient descent (parameter-shift style).
    history = [loss(params)]
    eps = np.pi / 2
    for it in range(n_iter):
        grad = np.zeros_like(params)
        for k in range(n_params):
            p_plus = params.copy(); p_plus[k] += eps
            p_minus = params.copy(); p_minus[k] -= eps
            grad[k] = (loss(p_plus) - loss(p_minus)) / 2.0
        params = params - lr * grad
        history.append(loss(params))

    return {
        "params":          params,
        "loss_history":    history,
        "final_loss":      history[-1],
        "n_qubits":        n_qubits,
        "n_keep":          n_keep,
        "n_trash":         n_qubits - n_keep,
        "ansatz":          ansatz,
    }


# ----------------------------------------------------------------------------
# Encode / decode / reconstruct
# ----------------------------------------------------------------------------

def encode(state: np.ndarray, encoder_params: np.ndarray,
            ansatz: Callable[[np.ndarray, np.ndarray], np.ndarray]
            ) -> np.ndarray:
    """Apply U(θ) — the encoder — to a state."""
    return ansatz(encoder_params, state)


def compression_fidelity(
    state: np.ndarray,
    encoder_params: np.ndarray,
    ansatz: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_qubits: int,
    n_keep: int,
) -> float:
    """Reconstruction fidelity: encode, project trash → |0⟩, decode,
    compare to original.

    For a well-trained autoencoder on training data, this should be ≈ 1.
    """
    encoded = ansatz(encoder_params, state)
    # Project trash to |0⟩: zero out amplitudes whose trash bits are nonzero.
    n_trash = n_qubits - n_keep
    projected = np.zeros_like(encoded)
    for idx in range(2 ** n_qubits):
        trash_bits = idx & ((1 << n_trash) - 1)
        if trash_bits == 0:
            projected[idx] = encoded[idx]
    norm = np.linalg.norm(projected)
    if norm < 1e-12:
        return 0.0
    projected = projected / norm
    # Decode by applying U†. We approximate U†(ψ) by inverse-parameter
    # negation — for Ry/Rz rotations alone, this is exact.
    # We perform full decoding via numerical inversion of the encoder
    # action (since the ansatz is unitary).
    # Build U as a matrix? Expensive. Use the fact that U is unitary
    # so U†ψ = numerical least-squares on rows... simpler: apply with
    # negated parameters in reverse order. We approximate by full
    # numerical inverse from encoded basis-state outputs.
    U_matrix = _build_ansatz_matrix(ansatz, encoder_params, n_qubits)
    U_dag = U_matrix.conj().T
    reconstructed = U_dag @ projected
    return float(abs(np.vdot(state, reconstructed)) ** 2)


def _build_ansatz_matrix(
    ansatz: Callable[[np.ndarray, np.ndarray], np.ndarray],
    params: np.ndarray,
    n_qubits: int,
) -> np.ndarray:
    """Build the dense unitary of an ansatz by applying it to each
    basis state."""
    d = 2 ** n_qubits
    U = np.zeros((d, d), dtype=np.complex128)
    for k in range(d):
        basis = np.zeros(d, dtype=np.complex128)
        basis[k] = 1.0
        U[:, k] = ansatz(params, basis)
    return U
