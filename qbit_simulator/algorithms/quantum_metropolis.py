"""Quantum Metropolis / quantum simulated annealing.

Two related ideas for sampling thermal / ground states of quantum
Hamiltonians:

  1. **Classical Metropolis on a quantum cost function**:
        - State = computational-basis bit string.
        - Energy = ⟨bit string | H | bit string⟩ (diagonal part of H, or
          a Pauli-Z energy function).
        - Standard MCMC with Boltzmann acceptance.
        - Useful for QUBO / Ising / MaxCut problems.

  2. **Quantum simulated annealing** (QSA) / adiabatic optimization:
        - Sweep H(s) = (1-s) · H_init + s · H_problem from s=0 to s=1.
        - Track the ground state via slow evolution (small dt, small ds).
        - At s=1 the state is (approximately) the H_problem ground state.

This module provides:

  - `metropolis_sample(H_diag, n_steps, beta, rng)`: classical MCMC over
    bit strings using a precomputed diagonal Hamiltonian.
  - `metropolis_estimate_ground_state(H_diag, ...)`: cool down to the
    lowest-energy bit string.
  - `quantum_annealing_evolution(H_init, H_problem, n_steps, dt)`:
    apply Trotterized adiabatic evolution, return final state.
  - `quantum_annealing_ground_state(H_problem, ...)`: convenience wrapper
    using the transverse-field Ising H_init = -sum_i X_i.

For our purposes, both approaches converge to the same answer on
Ising-style Hamiltonians; the quantum version showcases the adiabatic
theorem.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


# ----------------------------------------------------------------------------
# Classical Metropolis on a diagonal Hamiltonian
# ----------------------------------------------------------------------------

def metropolis_sample(
    energy_fn: Callable[[int], float],
    n_bits: int,
    n_steps: int = 10_000,
    beta: float = 1.0,
    initial_state: int | None = None,
    rng: np.random.Generator | None = None,
) -> dict:
    """Classical Metropolis sampling of bit strings ∈ {0, 1}^n_bits.

    Proposal: flip a single random bit. Accept with min(1, exp(-β·ΔE)).

    Args:
        energy_fn:     callable: int (bit string as integer) → float energy.
        n_bits:        number of bits.
        n_steps:       MCMC steps.
        beta:          inverse temperature.
        initial_state: starting bit string (default: random).
        rng:           generator.

    Returns:
        dict with "samples" (list of states visited), "energies",
        "min_state", "min_energy", "n_accepts".
    """
    rng = rng or np.random.default_rng()
    state = initial_state if initial_state is not None \
        else int(rng.integers(0, 2 ** n_bits))
    E = energy_fn(state)
    samples = [state]
    energies = [E]
    min_state, min_E = state, E
    n_accepts = 0

    for _ in range(n_steps):
        flip_bit = int(rng.integers(0, n_bits))
        proposal = state ^ (1 << flip_bit)
        E_new = energy_fn(proposal)
        dE = E_new - E
        if dE < 0 or rng.uniform() < np.exp(-beta * dE):
            state, E = proposal, E_new
            n_accepts += 1
            if E < min_E:
                min_state, min_E = state, E
        samples.append(state)
        energies.append(E)

    return {
        "samples":      samples,
        "energies":     energies,
        "min_state":    min_state,
        "min_energy":   min_E,
        "n_accepts":    n_accepts,
        "acceptance":   n_accepts / n_steps,
    }


def simulated_annealing(
    energy_fn: Callable[[int], float],
    n_bits: int,
    n_steps: int = 10_000,
    beta_schedule: Callable[[float], float] | None = None,
    initial_state: int | None = None,
    rng: np.random.Generator | None = None,
) -> dict:
    """Simulated annealing: ramp β from low to high.

    Default schedule: linear β(t/n_steps) from 0.1 to 10.
    """
    rng = rng or np.random.default_rng()
    if beta_schedule is None:
        beta_schedule = lambda s: 0.1 + 9.9 * s

    state = initial_state if initial_state is not None \
        else int(rng.integers(0, 2 ** n_bits))
    E = energy_fn(state)
    min_state, min_E = state, E
    energies_history = [E]

    for step in range(n_steps):
        s = step / n_steps
        beta = beta_schedule(s)
        flip_bit = int(rng.integers(0, n_bits))
        proposal = state ^ (1 << flip_bit)
        E_new = energy_fn(proposal)
        dE = E_new - E
        if dE < 0 or rng.uniform() < np.exp(-beta * dE):
            state, E = proposal, E_new
            if E < min_E:
                min_state, min_E = state, E
        energies_history.append(E)

    return {
        "min_state":    min_state,
        "min_energy":   min_E,
        "final_state":  state,
        "energies":     energies_history,
    }


# ----------------------------------------------------------------------------
# Quantum (adiabatic) annealing
# ----------------------------------------------------------------------------

# Pauli matrices
_I = np.eye(2, dtype=np.complex128)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)


def _pauli_kron(N: int, q: int, op: np.ndarray) -> np.ndarray:
    """Embed a 2x2 op on qubit q (MSB-first) into the full 2^N space."""
    out = np.array([[1.0 + 0j]])
    for k in range(N):
        out = np.kron(out, op if k == q else _I)
    return out


def transverse_field_initial_hamiltonian(N: int) -> np.ndarray:
    """H_init = -sum_i X_i  (the standard adiabatic mixer).

    Its ground state is the uniform superposition |+...+⟩.
    """
    dim = 2 ** N
    H = np.zeros((dim, dim), dtype=np.complex128)
    for q in range(N):
        H -= _pauli_kron(N, q, _X)
    return H


def quantum_annealing_evolution(
    H_init: np.ndarray,
    H_problem: np.ndarray,
    psi_0: np.ndarray,
    n_steps: int = 200,
    total_time: float = 10.0,
) -> dict:
    """Trotterized adiabatic evolution.

    The state follows H(s) = (1-s) H_init + s H_problem from s=0 to s=1,
    with s = t/T. We discretize into n_steps and exponentiate the
    instantaneous Hamiltonian at each step.

    Returns:
        dict with "final_state", "energies" (⟨H(s)⟩ over time),
        "ground_state_energies" (true GS of H(s) — for diagnostics).
    """
    psi = psi_0.copy()
    dt = total_time / n_steps
    energies = []
    gs_energies = []
    for k in range(n_steps):
        s = (k + 0.5) / n_steps   # midpoint of step
        H_s = (1.0 - s) * H_init + s * H_problem
        # Exponentiate via eigendecomp (small problems).
        eigs, V = np.linalg.eigh(H_s)
        U = V @ np.diag(np.exp(-1j * eigs * dt)) @ V.conj().T
        psi = U @ psi
        energies.append(float(np.real(psi.conj() @ H_s @ psi)))
        gs_energies.append(float(eigs[0]))
    return {
        "final_state":           psi,
        "energies":              np.array(energies),
        "ground_state_energies": np.array(gs_energies),
    }


def quantum_annealing_ground_state(
    H_problem: np.ndarray,
    n_steps: int = 200,
    total_time: float = 10.0,
) -> dict:
    """Convenience wrapper: anneal from the |+...+⟩ state using the
    transverse-field mixer to find H_problem's ground state.

    Returns the final state plus its energy + the true GS energy
    for comparison.
    """
    dim = H_problem.shape[0]
    N = int(np.log2(dim))
    if 2 ** N != dim:
        raise ValueError(f"H_problem dimension {dim} is not a power of 2")

    H_init = transverse_field_initial_hamiltonian(N)
    # Initial state: |+...+⟩.
    psi_0 = np.ones(dim, dtype=np.complex128) / np.sqrt(dim)
    result = quantum_annealing_evolution(
        H_init, H_problem, psi_0, n_steps=n_steps, total_time=total_time,
    )
    psi_final = result["final_state"]
    E_final = float(np.real(psi_final.conj() @ H_problem @ psi_final))
    E_true = float(np.linalg.eigvalsh(H_problem)[0])
    result.update({
        "energy":       E_final,
        "true_gs":      E_true,
        "gap_to_true":  E_final - E_true,
    })
    return result
