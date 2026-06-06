"""Lindblad master equation: open-quantum-system evolution.

For a closed system, |ψ⟩ evolves under the Schrödinger equation
i ∂|ψ⟩/∂t = H|ψ⟩. For an OPEN system coupled to an environment, the
density matrix ρ instead obeys the Lindblad master equation:

    dρ/dt = -i [H, ρ]  +  sum_k γ_k (L_k ρ L_k† − 1/2 {L_k† L_k, ρ})

where {L_k} are "jump operators" describing the various decoherence
channels (T1 relaxation, T2 dephasing, two-qubit cross-talk, etc.)
and γ_k are the corresponding rates.

We solve this numerically using two methods:

  1. **Direct integration** of the Lindbladian super-operator. Stack
     the (d×d) density matrix into a d²-vector; the Lindbladian acts
     as a d²×d² matrix L. Then ρ(t) = exp(L·t) · ρ(0).
     - Exact (no Trotter error).
     - O(d⁴) memory, O(d⁶) for one matrix exponential.
     - Tractable for n ≤ 4 qubits (d² ≤ 256).

  2. **Quantum-trajectory unraveling** (Carmichael / Dum-Parkins-
     Zoller-Gardiner). Sample stochastic state-vector trajectories;
     average over realizations to recover ρ.
     - O(d²) memory per trajectory.
     - Trotter-style time stepping.
     - Tractable for larger n.

Provides:

  - `lindblad_superoperator(H, jump_ops, rates)`: assemble L.
  - `evolve_density_matrix(rho0, L, t)`: exact density-matrix evolution.
  - `quantum_trajectory_step(psi, H, jump_ops, dt, rng)`: one Monte
    Carlo step.
  - `simulate_trajectory(psi0, H, jump_ops, t, n_steps, rng)`: a single
    full trajectory.
  - `simulate_ensemble(psi0, H, jump_ops, t, n_steps, n_traj, rng)`:
    average over many trajectories.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.linalg import expm


# ----------------------------------------------------------------------------
# Lindbladian super-operator
# ----------------------------------------------------------------------------

def lindblad_superoperator(
    H: np.ndarray, jump_ops: Sequence[np.ndarray],
    rates: Sequence[float] | None = None,
) -> np.ndarray:
    """Assemble the Lindbladian L as a d²×d² matrix acting on vec(ρ).

    Using the vectorization identity vec(A · X · B) = (B^T ⊗ A) · vec(X):

      vec(-i[H, ρ])     = -i (I⊗H − H^T⊗I) · vec(ρ)
      vec(L ρ L†)       = (L*⊗L) · vec(ρ)        (since (L†)^T = L*)
      vec({L†L, ρ}/2)   = (1/2)(I⊗(L†L) + (L†L)^T⊗I) · vec(ρ)

    Args:
        H:         Hamiltonian (d×d Hermitian).
        jump_ops:  list of jump operators L_k (d×d).
        rates:     list of corresponding γ_k (default: all 1).

    Returns:
        d²×d² complex matrix L.
    """
    d = H.shape[0]
    if rates is None:
        rates = [1.0] * len(jump_ops)
    I = np.eye(d, dtype=np.complex128)
    # Hamiltonian part.
    L_super = -1j * (np.kron(I, H) - np.kron(H.T, I))
    # Jump-operator part.
    for gamma, L_k in zip(rates, jump_ops):
        L_k = np.asarray(L_k, dtype=np.complex128)
        L_dag_L = L_k.conj().T @ L_k
        L_super += gamma * (
            np.kron(L_k.conj(), L_k)
            - 0.5 * np.kron(I, L_dag_L)
            - 0.5 * np.kron(L_dag_L.T, I)
        )
    return L_super


def evolve_density_matrix(
    rho0: np.ndarray, L_super: np.ndarray, t: float,
) -> np.ndarray:
    """Exact density-matrix evolution: ρ(t) = unvec(exp(L · t) · vec(ρ_0))."""
    d = rho0.shape[0]
    vec_rho = rho0.flatten(order="F")
    vec_rho_t = expm(L_super * t) @ vec_rho
    return vec_rho_t.reshape((d, d), order="F")


# ----------------------------------------------------------------------------
# Quantum-trajectory (Monte Carlo) method
# ----------------------------------------------------------------------------

def _effective_hamiltonian(H: np.ndarray,
                            jump_ops: Sequence[np.ndarray],
                            rates: Sequence[float]) -> np.ndarray:
    """Build H_eff = H − (i/2) sum_k γ_k L_k† L_k."""
    H_eff = H.astype(np.complex128).copy()
    for gamma, L_k in zip(rates, jump_ops):
        H_eff -= 0.5j * gamma * (L_k.conj().T @ L_k)
    return H_eff


def quantum_trajectory_step(
    psi: np.ndarray,
    H: np.ndarray,
    jump_ops: Sequence[np.ndarray],
    rates: Sequence[float],
    dt: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """One step of the wave-function Monte Carlo (Carmichael) unraveling.

    Algorithm:
      1. Evolve |ψ⟩ under the effective non-Hermitian H_eff for dt;
         the norm of the resulting state DECAYS.
      2. The norm-squared decay  δp = 1 − |⟨ψ|ψ⟩|² is the jump probability.
      3. With prob δp, sample a jump channel k weighted by ⟨ψ|L_k†L_k|ψ⟩
         and apply L_k / sqrt(⟨ψ|L_k†L_k|ψ⟩).
      4. Otherwise, just renormalize |ψ⟩.
    """
    H_eff = _effective_hamiltonian(H, jump_ops, rates)
    U_eff = expm(-1j * H_eff * dt)
    psi_new = U_eff @ psi
    norm_sq = float(np.real(psi_new.conj() @ psi_new))
    delta_p = 1.0 - norm_sq
    if delta_p > 1e-15 and rng.uniform() < delta_p:
        # Jump occurred. Pick which channel.
        weights = []
        for gamma, L_k in zip(rates, jump_ops):
            p_k = gamma * float(np.real(psi.conj() @ L_k.conj().T @ L_k @ psi))
            weights.append(p_k)
        total = sum(weights)
        if total < 1e-15:
            return psi_new / np.sqrt(norm_sq) if norm_sq > 0 else psi
        probs = np.array(weights) / total
        k = int(rng.choice(len(jump_ops), p=probs))
        jumped = jump_ops[k] @ psi
        jumped_norm = float(np.linalg.norm(jumped))
        return jumped / jumped_norm if jumped_norm > 0 else jumped
    return psi_new / np.sqrt(norm_sq) if norm_sq > 0 else psi


def simulate_trajectory(
    psi0: np.ndarray,
    H: np.ndarray,
    jump_ops: Sequence[np.ndarray],
    rates: Sequence[float],
    t: float,
    n_steps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """One stochastic trajectory: returns the final state vector."""
    psi = psi0.copy()
    dt = t / n_steps
    for _ in range(n_steps):
        psi = quantum_trajectory_step(psi, H, jump_ops, rates, dt, rng)
    return psi


def simulate_ensemble(
    psi0: np.ndarray,
    H: np.ndarray,
    jump_ops: Sequence[np.ndarray],
    rates: Sequence[float],
    t: float,
    n_steps: int = 100,
    n_traj: int = 200,
    rng: np.random.Generator | None = None,
) -> dict:
    """Average over n_traj quantum trajectories to estimate ρ(t).

    Returns:
        dict with rho_estimate (d×d), trajectory_states (list).
    """
    rng = rng or np.random.default_rng()
    d = len(psi0)
    rho = np.zeros((d, d), dtype=np.complex128)
    states = []
    for _ in range(n_traj):
        psi = simulate_trajectory(psi0, H, jump_ops, rates, t, n_steps, rng)
        rho += np.outer(psi, psi.conj())
        states.append(psi)
    rho /= n_traj
    return {"rho": rho, "states": states, "n_traj": n_traj}


# ----------------------------------------------------------------------------
# Common jump operators
# ----------------------------------------------------------------------------

def amplitude_damping_jump_single_qubit() -> np.ndarray:
    """L = σ_-= |0⟩⟨1| (single-qubit relaxation jump)."""
    return np.array([[0, 1], [0, 0]], dtype=np.complex128)


def dephasing_jump_single_qubit() -> np.ndarray:
    """L = σ_z (single-qubit pure dephasing jump)."""
    return np.array([[1, 0], [0, -1]], dtype=np.complex128)


def embed_single_qubit_op(op_1q: np.ndarray, qubit: int,
                           n_qubits: int) -> np.ndarray:
    """Embed a 2x2 operator on `qubit` (MSB-first) into the n-qubit Hilbert space."""
    I = np.eye(2, dtype=np.complex128)
    result = np.array([[1.0 + 0j]])
    for k in range(n_qubits):
        result = np.kron(result, op_1q if k == qubit else I)
    return result
