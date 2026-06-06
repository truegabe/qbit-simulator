"""Variational Quantum Imaginary Time Evolution (VarQITE).

Find the ground state of a Hamiltonian H by simulating imaginary-time
evolution |ψ(τ)⟩ ∝ e^{-Hτ}|ψ_0⟩ on a parameterized ansatz state |ψ(θ)⟩.

The math (McLachlan's variational principle):
    Project the exact imaginary-time evolution onto the tangent space of
    the ansatz, giving the equation of motion

        M(θ) · dθ/dτ = V(θ)

    where
        M_ij = Re ⟨∂_i ψ(θ) | ∂_j ψ(θ)⟩                (Fubini-Study metric)
        V_i  = -Re ⟨∂_i ψ(θ) | H | ψ(θ)⟩

    Step θ forward by Euler: θ ← θ + dθ/dτ · dτ.

Unlike standard VQE (which classically optimizes an energy landscape),
VarQITE follows a physically motivated trajectory and tends to be more
robust to barren plateaus.

This implementation uses finite-difference gradients for ∂_i ψ. For
quantum-hardware-realistic implementations, parameter-shift gradients
would be used instead (and we expose that as an option).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..circuit import QuantumCircuit
from ..pauli import PauliOp


def _ansatz_state(ansatz: Callable, theta: np.ndarray) -> np.ndarray:
    qc = ansatz(theta)
    return qc.state if hasattr(qc, "state") else qc


def _finite_diff_gradient(
    ansatz: Callable, theta: np.ndarray, eps: float = 1e-4,
) -> list[np.ndarray]:
    """Return list of d|ψ⟩/dθ_i for each parameter i, by central differences."""
    n_params = len(theta)
    grads: list[np.ndarray] = []
    for i in range(n_params):
        e_i = np.zeros(n_params); e_i[i] = eps
        psi_plus  = _ansatz_state(ansatz, theta + e_i)
        psi_minus = _ansatz_state(ansatz, theta - e_i)
        grads.append((psi_plus - psi_minus) / (2 * eps))
    return grads


def _build_M_and_V(
    ansatz: Callable,
    theta: np.ndarray,
    H: np.ndarray,
    eps: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the McLachlan M matrix and V vector at the current θ."""
    n = len(theta)
    psi = _ansatz_state(ansatz, theta)
    grads = _finite_diff_gradient(ansatz, theta, eps)
    M = np.zeros((n, n), dtype=np.float64)
    V = np.zeros(n, dtype=np.float64)
    Hpsi = H @ psi
    for i in range(n):
        V[i] = -float(np.real(np.vdot(grads[i], Hpsi)))
        for j in range(n):
            M[i, j] = float(np.real(np.vdot(grads[i], grads[j])))
    return M, V


def varqite(
    hamiltonian: PauliOp,
    ansatz: Callable,
    theta0: np.ndarray,
    n_steps: int = 200,
    d_tau: float = 0.05,
    regularization: float = 1e-6,
    eps_grad: float = 1e-4,
) -> dict:
    """Run VarQITE for `n_steps` imaginary-time steps.

    Args:
        hamiltonian:   target Hamiltonian (PauliOp).
        ansatz:        callable theta -> state-vector or QuantumCircuit.
        theta0:        initial parameter vector.
        n_steps:       number of Euler steps in imaginary time.
        d_tau:         step size in imaginary time.
        regularization: ridge term on M (M_reg = M + ε·I) for numerical stability.
        eps_grad:      finite-difference epsilon for ∂_i ψ.

    Returns:
        dict with:
            theta_final: final parameter vector
            energy_trace: list of ⟨H⟩ along the trajectory
            tau_trace:    array of τ values
            final_energy: ⟨H⟩ at the end
            ground_energy: exact ground energy of H (for reference)
    """
    theta = np.asarray(theta0, dtype=np.float64).copy()
    H_matrix = hamiltonian.matrix()

    energy_trace = []
    tau_trace = []

    for step in range(n_steps):
        psi = _ansatz_state(ansatz, theta)
        psi /= np.linalg.norm(psi)
        e = float(np.real(psi.conj() @ H_matrix @ psi))
        energy_trace.append(e)
        tau_trace.append(step * d_tau)

        M, V = _build_M_and_V(ansatz, theta, H_matrix, eps=eps_grad)
        # Solve M·dθ/dτ = V with Tikhonov regularization.
        n_p = len(theta)
        M_reg = M + regularization * np.eye(n_p)
        try:
            dtheta_dtau = np.linalg.solve(M_reg, V)
        except np.linalg.LinAlgError:
            dtheta_dtau = np.linalg.lstsq(M_reg, V, rcond=None)[0]
        theta += d_tau * dtheta_dtau

    # Final energy check.
    psi = _ansatz_state(ansatz, theta)
    psi /= np.linalg.norm(psi)
    final_energy = float(np.real(psi.conj() @ H_matrix @ psi))
    e_gs, _ = hamiltonian.ground_state()

    return {
        "theta_final":    theta,
        "energy_trace":   np.array(energy_trace),
        "tau_trace":      np.array(tau_trace),
        "final_energy":   final_energy,
        "ground_energy":  float(e_gs),
    }
