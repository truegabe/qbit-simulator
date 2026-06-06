"""Noise-aware VQE — combine the existing VQE loop with realistic gate noise.

For each parameter setting θ, we run K trajectories of the noisy circuit:
each trajectory applies the ansatz gate-by-gate, and after every gate the
noise Kraus channel is stochastically applied to the affected qubits. We
then estimate the energy as the average ⟨ψ_traj|H|ψ_traj⟩ across
trajectories.

The optimization loop is the same scipy minimize as the noise-free VQE; the
only difference is that each cost-function evaluation now goes through
trajectories. This is the standard NISQ-realistic VQE workflow.

Demonstrates:
    - How decoherence degrades ground-state estimation
    - Why low-depth ansätze (fewer gates -> less noise) matter on real hardware
    - The transition from "noise-free quantum advantage" to "noisy NISQ regime"
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import minimize

from ..circuit import QuantumCircuit
from ..pauli import PauliOp
from ..noise import apply_channel_trajectory


def apply_noise_to_qubits(
    state: np.ndarray,
    n_qubits: int,
    target_qubits: list[int],
    kraus: list[np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply a single-qubit Kraus channel to each of `target_qubits`."""
    for q in target_qubits:
        state = apply_channel_trajectory(state, kraus, q, n_qubits, rng)
    return state


def noisy_circuit_state(
    ansatz: Callable[[float | np.ndarray], QuantumCircuit],
    theta: float | np.ndarray,
    kraus_per_gate: list[np.ndarray] | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build the ansatz state with noise applied after each gate.

    The `ansatz(theta)` callable returns a noise-free QuantumCircuit. We
    replay its `_ops` list, applying the Kraus channel to the affected
    qubits after each gate.
    """
    qc = ansatz(theta)
    if kraus_per_gate is None:
        return qc.state
    # Replay ops on a fresh circuit, injecting noise after each gate.
    fresh = QuantumCircuit(qc.n)
    for kind, matrix, targets in qc._ops:
        if kind == "1q":
            fresh._apply_1q(matrix, targets[0])
        elif kind == "2q":
            fresh._apply_2q(matrix, targets[0], targets[1])
        elif kind == "kq":
            fresh.apply_unitary(matrix, targets, check_unitary=False)
        # Apply noise to all qubits this gate touched.
        fresh.state = apply_noise_to_qubits(
            fresh.state, fresh.n, list(targets), kraus_per_gate, rng,
        )
    return fresh.state


def noisy_energy(
    state: np.ndarray,
    hamiltonian: PauliOp,
) -> float:
    """⟨ψ|H|ψ⟩ for a possibly-unnormalized trajectory state."""
    norm = np.linalg.norm(state)
    if norm < 1e-12:
        return 0.0
    psi = state / norm
    H = hamiltonian.matrix()
    return float(np.real(psi.conj() @ H @ psi))


def noisy_vqe(
    hamiltonian: PauliOp,
    ansatz: Callable,
    theta0: float | np.ndarray,
    kraus_per_gate: list[np.ndarray] | None,
    n_trajectories: int = 50,
    max_iter: int = 100,
    seed: int = 0,
    bounds: tuple | None = None,
) -> dict:
    """Run VQE with stochastic noise applied after each gate.

    Args:
        hamiltonian: the target Hamiltonian (PauliOp).
        ansatz: callable theta -> QuantumCircuit.
        theta0: initial parameter (float or array).
        kraus_per_gate: 1-qubit Kraus operators applied after each gate.
                        None for noise-free baseline.
        n_trajectories: shots averaged per energy evaluation.
        max_iter: optimizer iteration cap.
        seed: RNG seed.
        bounds: optimizer parameter bounds.

    Returns:
        dict with theta_opt, e_opt, n_evals, history.
    """
    rng = np.random.default_rng(seed)
    history: list[float] = []
    is_scalar = isinstance(theta0, (int, float))

    def cost(theta_arr):
        theta = float(theta_arr[0]) if is_scalar else theta_arr
        energies = []
        for _ in range(n_trajectories):
            psi = noisy_circuit_state(ansatz, theta, kraus_per_gate, rng)
            energies.append(noisy_energy(psi, hamiltonian))
        avg = float(np.mean(energies))
        history.append(avg)
        return avg

    x0 = np.array([theta0]) if is_scalar else np.asarray(theta0, dtype=float)
    method = "Nelder-Mead"
    res = minimize(cost, x0, method=method,
                   options={"maxiter": max_iter, "xatol": 1e-4, "fatol": 1e-4})
    theta_opt = float(res.x[0]) if is_scalar else res.x
    return {
        "theta_opt": theta_opt,
        "e_opt": float(res.fun),
        "n_evals": len(history),
        "history": history,
        "success": bool(res.success),
    }
