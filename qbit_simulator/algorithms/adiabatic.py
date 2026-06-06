"""Adiabatic quantum computing — Farhi, Goldstone, Gutmann, Sipser 2000.

Start in the ground state of an easy Hamiltonian H_0, slowly interpolate to
a target Hamiltonian H_target, and end (approximately) in the ground state
of H_target. The adiabatic theorem guarantees this works if the schedule is
slow compared to the inverse-square of the minimum spectral gap.

This module implements the discretized version: time-step through the
interpolation, applying Trotterized evolution at each step. Built on top of
the existing MPSState + TEBD infrastructure for low-entanglement problems
(MaxCut on small graphs is a canonical example).

Schedule:
    H(s) = (1 - s) H_0  +  s H_target,   s ∈ [0, 1]
    s_k = k / K                          (linear schedule by default)
    apply exp(-i H(s_k) dt) for k = 1..K

For our test cases (small N), we use a dense state vector rather than MPS
because the problems are too small to need compression, and the Trotter
ordering across non-commuting H_0 and H_target terms gets cleaner.

Returns the final state vector and a trace of energies along the schedule.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from ..pauli import PauliOp


def adiabatic_evolve(
    H0: PauliOp,
    H_target: PauliOp,
    initial_state: np.ndarray,
    n_steps: int = 100,
    total_time: float = 10.0,
    schedule: str | callable = "linear",
) -> dict:
    """Adiabatic evolution from H_0 to H_target.

    Args:
        H0: starting Hamiltonian (ground state of which is `initial_state`).
        H_target: target Hamiltonian (we want its ground state).
        initial_state: 2^N state vector at s=0.
        n_steps: number of Trotter steps.
        total_time: total evolution time T. The dt is T / n_steps.
        schedule: "linear" or a callable s(t/T) returning a float in [0, 1].

    Returns:
        dict with:
            final_state:        2^N state vector at s=1
            energy_trace:       list of ⟨H_target⟩ at each step
            schedule_trace:     list of s values at each step
            overlap_with_gs:    |⟨ground state of H_target | final state⟩|²
            ground_energy:      exact ground energy of H_target
    """
    n_qubits = int(np.log2(initial_state.size))
    if 2**n_qubits != initial_state.size:
        raise ValueError("initial state must have size 2^N")
    if not callable(schedule):
        if schedule == "linear":
            s_fn = lambda x: x
        else:
            raise ValueError(f"unknown schedule {schedule!r}")
    else:
        s_fn = schedule

    H0_dense  = H0.matrix()
    Ht_dense  = H_target.matrix()

    state = initial_state.astype(np.complex128).copy()
    state /= np.linalg.norm(state)

    dt = total_time / n_steps
    energies: list[float] = []
    s_values: list[float] = []

    for k in range(1, n_steps + 1):
        s = float(s_fn(k / n_steps))
        H_s = (1 - s) * H0_dense + s * Ht_dense
        U_step = expm(-1j * dt * H_s)
        state = U_step @ state
        # Track ⟨H_target⟩ along the way (the quantity we're optimizing).
        e = float(np.real(state.conj() @ Ht_dense @ state))
        energies.append(e)
        s_values.append(s)

    # Compare to exact ground state of H_target.
    e_gs, gs = H_target.ground_state()
    overlap = abs(np.vdot(gs, state)) ** 2

    return {
        "final_state":      state,
        "energy_trace":     np.array(energies),
        "schedule_trace":   np.array(s_values),
        "ground_energy":    float(e_gs),
        "final_energy":     energies[-1],
        "overlap_with_gs":  float(overlap),
        "n_qubits":         n_qubits,
    }


def maxcut_hamiltonian(n: int, edges: list[tuple[int, int]]) -> PauliOp:
    """Build the MaxCut Hamiltonian for a graph on n vertices.

    H_target = (1/2) Σ_{(i,j) ∈ E} (Z_i Z_j - I)

    Minimizing this is equivalent to maximizing the number of edges cut by a
    Z-basis partition. Ground energy = -(number of cuttable edges).
    """
    if not edges:
        raise ValueError("MaxCut needs at least one edge")
    terms: list[tuple[complex, str]] = []
    for (i, j) in edges:
        if not (0 <= i < n and 0 <= j < n):
            raise IndexError(f"edge ({i},{j}) out of range [0,{n})")
        if i == j:
            raise ValueError(f"self-loop at vertex {i}")
        s = ["I"] * n
        s[i] = "Z"; s[j] = "Z"
        terms.append((0.5 + 0j, "".join(s)))
        terms.append((-0.5 + 0j, "I" * n))
    return PauliOp(terms)


def transverse_field_driver(n: int) -> PauliOp:
    """The standard adiabatic driver H_0 = -Σ X_i. Its ground state is |+⟩^⊗N."""
    terms: list[tuple[complex, str]] = []
    for i in range(n):
        s = ["I"] * n
        s[i] = "X"
        terms.append((-1.0 + 0j, "".join(s)))
    return PauliOp(terms)


def plus_state(n: int) -> np.ndarray:
    """Uniform superposition |+⟩^⊗N, the ground state of -Σ X_i."""
    return np.full(2**n, 1.0 / np.sqrt(2**n), dtype=np.complex128)
