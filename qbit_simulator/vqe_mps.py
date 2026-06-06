"""Variational Quantum Eigensolver on a Matrix Product State (MPS-VQE).

Parameterize an N-qubit state via a brickwall ansatz: alternating layers
of 2-qubit blocks (each a small parameterized unitary) applied to |0...0>
in MPS form. The bond dimension is capped at `max_chi`, so the ansatz
naturally lives in the same low-entanglement regime DMRG exploits.

For each parameter vector θ we evaluate

    E(θ) = ⟨ψ(θ) | H | ψ(θ)⟩

via MPS/MPO environment contraction (O(N · χ³ · D²) where D is the MPO
bond dim — bounded for local Hamiltonians). Then we minimize over θ with
scipy.optimize.minimize.

Benchmarks:
  - TFIM(N=4..10): comparison to exact diagonalization
  - Heisenberg(N=4..8): comparison to exact
  - TFIM(N=20+): comparison to DMRG ground energy

Why this is interesting beyond a real QC:
  - We compute E(θ) exactly via tensor contraction. A real QC samples
    Pauli-string measurements, accumulating shot noise — needs ~10⁵
    shots per parameter for chemistry precision. We get exact E with
    one forward pass.
  - Gradients are cheap (numerical finite differences via L-BFGS-B).
  - No decoherence, no compilation, no calibration drift.

What this is NOT:
  - A real chemistry calculation. To do LiH or H₂O we'd need a chemistry
    Hamiltonian as an MPO, which requires either hand-coded Pauli
    decompositions or a PySCF bridge. The VQE machinery in this module
    is ready for that the moment such an MPO exists. For now we
    benchmark on TFIM/Heisenberg, where we have ground-truth MPOs.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import minimize

from .gates import I2, X, Ry, CNOT
from .mps import MPSState
from .mpo import MPO
from .dmrg import _contract_left_env


# ---- brickwall ansatz ----

def _two_qubit_block(params: np.ndarray) -> np.ndarray:
    """Standard 4-parameter hardware-efficient 2-qubit block:
        Ry(p0) ⊗ Ry(p1) → CNOT → Ry(p2) ⊗ Ry(p3)
    Returns a 4×4 real unitary.
    """
    p0, p1, p2, p3 = params
    left  = np.kron(Ry(p0), Ry(p1))
    right = np.kron(Ry(p2), Ry(p3))
    return right @ CNOT @ left


def n_params(n_qubits: int, n_layers: int) -> int:
    """Number of variational parameters for a brickwall ansatz with
    `n_layers` layers on `n_qubits` qubits."""
    # Each layer covers either even or odd bonds alternately.
    n_per_layer = 0
    for layer in range(n_layers):
        start = layer % 2
        n_per_layer += len(range(start, n_qubits - 1, 2))
    return n_per_layer * 4


def brickwall_ansatz(
    params: np.ndarray,
    n_qubits: int,
    n_layers: int,
    max_chi: int = 32,
) -> MPSState:
    """Build the MPS state |ψ(θ)⟩ from a flat parameter vector.

    Layout: layer 0 acts on bonds (0,1), (2,3), ...; layer 1 acts on
    bonds (1,2), (3,4), ...; and so on. Each bond block consumes 4 params.
    """
    expected = n_params(n_qubits, n_layers)
    if len(params) != expected:
        raise ValueError(f"got {len(params)} params, expected {expected}")
    mps = MPSState(n_qubits, max_chi=max_chi)
    idx = 0
    for layer in range(n_layers):
        start = layer % 2
        for q in range(start, n_qubits - 1, 2):
            U = _two_qubit_block(params[idx:idx + 4])
            mps.apply_2q_adjacent(U, q)
            idx += 4
    return mps


# ---- MPO expectation value on an MPS ----

def mps_energy(mps: MPSState, mpo: MPO) -> float:
    """⟨ψ | H | ψ⟩ where |ψ⟩ is the MPS and H is the MPO.

    Contract the chain left-to-right, reusing the environment contraction
    DMRG uses. Cost: O(N · χ² · D² · 2) per site, where D is the MPO bond.
    """
    if mps.n != mpo.n:
        raise ValueError(f"MPS has {mps.n} sites but MPO has {mpo.n}")
    env = np.ones((1, 1, 1), dtype=np.complex128)        # (B, L_mpo, K)
    for q in range(mps.n):
        env = _contract_left_env(
            env, mpo.tensors[q], mps.tensors[q], np.conj(mps.tensors[q])
        )
    return float(np.real(env[0, 0, 0]))


# ---- the VQE loop ----

def vqe_mps(
    mpo: MPO,
    n_layers: int = 3,
    max_chi: int = 16,
    seed: int = 0,
    max_iter: int = 300,
    tol: float = 1e-8,
    init_scale: float = 0.5,
    verbose: bool = False,
    callback: Callable | None = None,
) -> dict:
    """Minimize ⟨H⟩ over a brickwall MPS ansatz.

    Args:
        mpo: Hamiltonian as a Matrix Product Operator.
        n_layers: ansatz depth (more layers = more expressive, more params).
        max_chi: MPS bond cap.
        seed: RNG seed for parameter init.
        max_iter: optimizer iteration cap.
        tol: convergence tolerance on energy.
        init_scale: range of initial random parameters (uniform [-s, s]).
        verbose: print per-iteration energies.
        callback: optional callable invoked as callback(theta) each iter.

    Returns:
        dict with 'energy', 'params', 'n_params', 'history', 'success', 'mps'.
    """
    n = mpo.n
    n_p = n_params(n, n_layers)
    rng = np.random.default_rng(seed)
    theta0 = rng.uniform(-init_scale, init_scale, size=n_p)
    history: list[float] = []

    def cost(theta):
        mps = brickwall_ansatz(theta, n, n_layers, max_chi=max_chi)
        e = mps_energy(mps, mpo)
        history.append(e)
        if verbose and (len(history) % 10 == 0):
            print(f"  iter {len(history):>4}: E = {e:.8f}")
        if callback is not None:
            callback(theta)
        return e

    result = minimize(
        cost, theta0, method="L-BFGS-B",
        options={"maxiter": max_iter, "ftol": tol, "gtol": tol * 10},
    )
    final_mps = brickwall_ansatz(result.x, n, n_layers, max_chi=max_chi)
    return {
        "energy":   float(result.fun),
        "params":   result.x,
        "n_params": n_p,
        "history":  history,
        "success":  bool(result.success),
        "n_iters":  len(history),
        "mps":      final_mps,
    }
