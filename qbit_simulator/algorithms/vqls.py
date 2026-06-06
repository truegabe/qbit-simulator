"""Variational Quantum Linear Solver (VQLS) — Bravo-Prieto et al. 2019.

Solves a linear system A|x⟩ = |b⟩ by variationally minimizing the cost

    C(θ) = || A|x(θ)⟩ − ⟨b|A|x(θ)⟩ · |b⟩ / |⟨b|A|x(θ)⟩| ||²

i.e. the squared overlap of A|x(θ)⟩ with the orthogonal complement of |b⟩.
When C(θ) = 0, |x(θ)⟩ is proportional to A⁻¹|b⟩.

Unlike HHL, VQLS is hardware-friendly: no QPE, no controlled rotations on
huge angular ranges, no post-selection. Trade-off: classical optimization
in a non-convex landscape (the usual VQE caveats).

This implementation:
    - Takes A as a dense Hermitian matrix.
    - Takes |b⟩ as a state vector.
    - Uses a brickwall ansatz from `vqe_mps` (importing it directly).
    - Cost function uses the dense-state expression; in a real-QC setting
      this would be estimated from Hadamard tests.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import minimize

from ..vqe_mps import brickwall_ansatz, n_params
from ..circuit import QuantumCircuit


def _hardware_efficient_ansatz(
    params: np.ndarray, n_qubits: int, n_layers: int,
) -> QuantumCircuit:
    """Hardware-efficient ansatz: per layer apply Ry to every qubit, then
    a ladder of CNOTs.

    Total parameters: n_qubits * n_layers.
    """
    qc = QuantumCircuit(n_qubits)
    idx = 0
    for layer in range(n_layers):
        for q in range(n_qubits):
            qc.ry(float(params[idx]), q)
            idx += 1
        for q in range(n_qubits - 1):
            qc.cnot(q, q + 1)
    return qc


def _hwe_n_params(n_qubits: int, n_layers: int) -> int:
    return n_qubits * n_layers


def _cost_function(
    theta: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    n_qubits: int,
    n_layers: int,
) -> float:
    """C(θ) = 1 - |⟨b|A|x(θ)⟩|² / (⟨x|A†A|x⟩ · ⟨b|b⟩)

    Equivalently: 1 - cos²(angle between A|x⟩ and |b⟩) — minimized to 0
    when A|x⟩ is collinear with |b⟩, i.e. when |x⟩ ∝ A⁻¹|b⟩.
    """
    qc = _hardware_efficient_ansatz(theta, n_qubits, n_layers)
    x = qc.state
    Ax = A @ x
    num = abs(np.vdot(b, Ax)) ** 2
    den = float(np.real(np.vdot(Ax, Ax))) * float(np.real(np.vdot(b, b)))
    if den < 1e-14:
        return 1.0
    return float(1.0 - num / den)


def vqls(
    A: np.ndarray,
    b: np.ndarray,
    n_layers: int = 3,
    max_chi: int = 16,
    max_iter: int = 300,
    seed: int = 0,
    init_scale: float = 0.5,
) -> dict:
    """Solve A|x⟩ ∝ |b⟩ variationally.

    Args:
        A:          d × d Hermitian matrix (d = 2^n).
        b:          length-d right-hand side vector (need not be unit).
        n_layers:   ansatz depth.
        max_chi:    MPS bond dimension cap.
        max_iter:   optimizer iteration cap.
        seed:       RNG seed for parameter initialization.

    Returns:
        dict with:
            x_quantum:    estimated normalized solution
            x_classical:  np.linalg.solve(A, b) for reference
            fidelity:     |⟨x_q | x_c⟩|²
            cost_final:   final value of the VQLS cost function
    """
    d = A.shape[0]
    n_qubits = int(np.log2(d))
    if 2**n_qubits != d:
        raise ValueError(f"A's dimension {d} must be a power of 2")
    b = np.asarray(b, dtype=np.complex128)
    if b.shape != (d,):
        raise ValueError(f"b must have shape ({d},)")
    b = b / np.linalg.norm(b)

    n_p = _hwe_n_params(n_qubits, n_layers)
    rng = np.random.default_rng(seed)
    theta0 = rng.uniform(-init_scale, init_scale, size=n_p)

    cost = lambda t: _cost_function(t, A, b, n_qubits, n_layers)
    result = minimize(cost, theta0, method="L-BFGS-B",
                       options={"maxiter": max_iter, "ftol": 1e-9})

    qc = _hardware_efficient_ansatz(result.x, n_qubits, n_layers)
    x_quantum = qc.state
    x_quantum /= np.linalg.norm(x_quantum)

    x_classical_unnorm = np.linalg.solve(A, b)
    x_classical = x_classical_unnorm / np.linalg.norm(x_classical_unnorm)
    fidelity = abs(np.vdot(x_classical, x_quantum)) ** 2

    return {
        "x_quantum":    x_quantum,
        "x_classical":  x_classical,
        "fidelity":     float(fidelity),
        "cost_final":   float(result.fun),
        "n_params":     n_p,
        "n_iters":      result.nit,
    }
