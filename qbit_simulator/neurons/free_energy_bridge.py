"""Free-energy / variational-quantum bridge.

The free-energy principle (Friston 2006) proposes that the brain
minimizes a variational free-energy functional

    F  =  E_q[log q(z) − log p(x, z)]
        =  −log p(x)  +  KL(q(z) || p(z|x))

over a recognition density q(z) parameterized by neural activity. This
is mathematically variational inference.

Quantum variational algorithms — VQE, VarQITE, QNG — minimize an
analogous quantum free-energy on a parameterized quantum state. When
both layers exist (as in this project), the connection becomes literal:

  - Treat the predictive-coding network as the "generative model"
    p(x, z).
  - Use a parameterized quantum circuit |ψ(θ)⟩ as the "recognition
    distribution" q(z).
  - Minimize the **quantum** free-energy

        F(θ) = −⟨ψ(θ) | log p̂ | ψ(θ)⟩  +  ⟨log q_θ⟩

    using a quantum-natural-gradient optimizer.

This module provides:

  - `pc_to_potential(pc_network, x)`: convert a predictive-coding
    network's prediction-error landscape into a Hermitian "potential"
    matrix that a quantum recognition state can be matched against.
  - `quantum_free_energy(psi, potential)`: ⟨ψ | V | ψ⟩.
  - `match_pc_with_quantum(pc_network, x, ansatz, ...)`: train a
    quantum ansatz to minimize the PC free energy.

This is a **toy demonstration** — the dimensions don't match real
neuroscience or real quantum chemistry. The point is to show the
**variational machinery is the same**: classical predictive coding
and quantum VQE are two implementations of the same Bayesian-inference
framework.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .predictive_coding import PredictiveCodingNetwork


# ----------------------------------------------------------------------------
# Convert PC network into a quantum potential
# ----------------------------------------------------------------------------

def pc_to_potential(
    network: PredictiveCodingNetwork, x_sensory: np.ndarray,
) -> np.ndarray:
    """Build a Hermitian matrix V whose eigenvectors correspond to
    "states" that the predictive-coding model considers plausible.

    Strategy: for each computational-basis state |k⟩ on n qubits with
    n = log₂(layer_sizes[-1]), interpret k as an integer index into a
    set of "candidate top-level states", evaluate the PC free energy
    for that candidate, and set V[k, k] = F(candidate_k).

    This makes V diagonal (no off-diagonals — the PC model is
    classically symmetric). The minimum-eigenvalue eigenvector is the
    computational-basis state with the lowest classical PC free energy
    — i.e. the "best explanation" of the sensory data.
    """
    top_size = network.layer_sizes[-1]
    n_qubits = int(np.ceil(np.log2(max(2, top_size))))
    dim = 2 ** n_qubits
    # For each candidate top state, run inference and read final F.
    energies = np.zeros(dim)
    for k in range(dim):
        # Encode k as an integer in [0, top_size) wrapped.
        top_idx = k % top_size
        candidate = np.zeros(top_size)
        candidate[top_idx] = 1.0
        # Clamp top, run downward generation, compute residual.
        # We use the network's predict_top_down to get expected sensory.
        predicted = network.predict_top_down(candidate)
        residual = x_sensory - predicted
        energies[k] = 0.5 * float(np.dot(residual, residual))
    # Return as a diagonal Hermitian matrix.
    return np.diag(energies).astype(np.complex128)


# ----------------------------------------------------------------------------
# Quantum free energy
# ----------------------------------------------------------------------------

def quantum_free_energy(psi: np.ndarray, potential: np.ndarray) -> float:
    """⟨ψ | V | ψ⟩ — the variational free energy."""
    return float(np.real(psi.conj() @ potential @ psi))


# ----------------------------------------------------------------------------
# Match-quantum-to-PC training
# ----------------------------------------------------------------------------

def match_pc_with_quantum(
    pc_network: PredictiveCodingNetwork,
    x_sensory: np.ndarray,
    ansatz: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_qubits: int,
    init_params: np.ndarray,
    use_qng: bool = True,
    n_iter: int = 50,
    lr: float = 0.1,
) -> dict:
    """Train a parameterized quantum circuit to minimize the
    free-energy potential induced by a PC network.

    Args:
        pc_network:  a predictive-coding network.
        x_sensory:   observed sensory data.
        ansatz:      callable (params, ref_state) → state vector.
        n_qubits:    qubit count.
        init_params: starting parameters.
        use_qng:     if True, use Quantum Natural Gradient; else
                     plain parameter-shift gradient descent.
        n_iter:      training iterations.
        lr:          learning rate.

    Returns:
        dict with trained params, free-energy history, final state.
    """
    V = pc_to_potential(pc_network, x_sensory)
    if V.shape[0] != 2 ** n_qubits:
        raise ValueError(
            f"potential dimension {V.shape[0]} != 2^{n_qubits}"
        )
    ref = np.zeros(2 ** n_qubits, dtype=np.complex128)
    ref[0] = 1.0

    def loss(params):
        psi = ansatz(params, ref)
        return quantum_free_energy(psi, V)

    history = [loss(init_params)]
    params = init_params.copy()
    if use_qng:
        from ..algorithms.quantum_natural_gradient import (
            quantum_natural_gradient_step,
        )
        for it in range(n_iter):
            params, info = quantum_natural_gradient_step(
                loss, ansatz, params, ref, lr=lr, lambda_reg=1e-3,
            )
            history.append(loss(params))
    else:
        from ..algorithms.quantum_natural_gradient import (
            parameter_shift_gradient,
        )
        for it in range(n_iter):
            grad = parameter_shift_gradient(loss, params)
            params = params - lr * grad
            history.append(loss(params))

    final_state = ansatz(params, ref)
    return {
        "params":         params,
        "free_energy":    history,
        "final_state":    final_state,
        "n_iter":         n_iter,
        "used_qng":       use_qng,
        "potential":      V,
    }


# ----------------------------------------------------------------------------
# Direct inspection: which candidate has lowest classical PC energy?
# ----------------------------------------------------------------------------

def best_classical_explanation(
    pc_network: PredictiveCodingNetwork, x_sensory: np.ndarray,
) -> dict:
    """Brute-force: which top-level state minimizes the PC free energy
    for the given sensory observation? Used as a ground truth for
    comparison with the quantum-variational result."""
    V = pc_to_potential(pc_network, x_sensory)
    energies = np.diag(V).real
    best_k = int(np.argmin(energies))
    return {
        "best_index":  best_k,
        "best_energy": float(energies[best_k]),
        "all_energies": energies,
    }
