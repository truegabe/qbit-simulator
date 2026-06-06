"""Measurement-Based Quantum Computation (MBQC).

In the **one-way quantum computer** (Raussendorf-Briegel 2001), all
quantum gates are implemented via SINGLE-QUBIT MEASUREMENTS on a
pre-prepared entangled "cluster state". The flow is:

  1. Prepare a 2D grid of qubits in the cluster state |C⟩:
        - Initialize each qubit in |+⟩.
        - Apply CZ to every pair of nearest-neighbors on the grid.
  2. Choose a measurement schedule: for each qubit, measure in some
     basis depending on the desired logical operation.
  3. Adapt later measurements based on earlier outcomes (the "feed-
     forward" / classical-control structure).
  4. The unmeasured qubits at the "output" end of the grid carry the
     computed quantum state.

A 1D chain implements arbitrary single-qubit rotations:

    Measure qubit k in basis cos(θ/2) X + sin(θ/2) Y
    → applies Rz(θ) to the logical state on qubit k+1 (up to byproduct).

This module provides:

  - `cluster_state(rows, cols)`: prepare a 2D rectangular cluster state.
  - `chain_cluster_state(n)`: 1D chain version.
  - `measure_in_xy_basis(state, qubit, angle, rng)`: measure qubit in
    basis cos(α/2)X + sin(α/2)Y; returns outcome bit.
  - `mbqc_single_qubit_rotation(angle, n_chain, rng)`: implement Rz(θ)
    on a 2-qubit chain via measurement.
  - `byproduct_correction(outcomes, angle)`: the Z/X corrections needed
    given measurement outcomes.

We use the dense state-vector simulator for verification.
"""

from __future__ import annotations

from itertools import product

import numpy as np


# Pauli matrices
_I = np.eye(2, dtype=np.complex128)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
_H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)


# ----------------------------------------------------------------------------
# Cluster-state preparation
# ----------------------------------------------------------------------------

def _apply_1q(psi: np.ndarray, gate: np.ndarray, q: int, n: int) -> np.ndarray:
    arr = psi.reshape([2] * n)
    arr = np.moveaxis(arr, q, 0)
    arr = gate @ arr.reshape(2, -1)
    arr = arr.reshape([2] + [2] * (n - 1))
    arr = np.moveaxis(arr, 0, q)
    return arr.reshape(2 ** n)


def _apply_cz(psi: np.ndarray, q0: int, q1: int, n: int) -> np.ndarray:
    """CZ is diagonal: phase −1 on |11⟩, +1 otherwise."""
    out = psi.copy()
    for idx in range(2 ** n):
        bit0 = (idx >> (n - 1 - q0)) & 1
        bit1 = (idx >> (n - 1 - q1)) & 1
        if bit0 == 1 and bit1 == 1:
            out[idx] = -out[idx]
    return out


def cluster_state(rows: int, cols: int) -> np.ndarray:
    """Prepare a `rows × cols` 2D cluster state.

    Algorithm: initialize every qubit in |+⟩, then apply CZ to every
    horizontal and vertical nearest-neighbor pair.
    """
    n = rows * cols
    psi = np.ones(2 ** n, dtype=np.complex128) / np.sqrt(2 ** n)  # |+...+⟩
    # CZ on horizontal pairs (within rows).
    for r in range(rows):
        for c in range(cols - 1):
            q0 = r * cols + c
            q1 = r * cols + c + 1
            psi = _apply_cz(psi, q0, q1, n)
    # CZ on vertical pairs (between rows).
    for r in range(rows - 1):
        for c in range(cols):
            q0 = r * cols + c
            q1 = (r + 1) * cols + c
            psi = _apply_cz(psi, q0, q1, n)
    return psi


def chain_cluster_state(n: int) -> np.ndarray:
    """1D chain cluster state on n qubits."""
    return cluster_state(rows=1, cols=n)


# ----------------------------------------------------------------------------
# Measurement primitives
# ----------------------------------------------------------------------------

def measure_in_xy_basis(
    psi: np.ndarray, qubit: int, angle: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Measure qubit in the basis {|+_α⟩, |−_α⟩} where

        |±_α⟩ = (|0⟩ ± e^{i α} |1⟩) / √2

    Equivalently, the basis with measurement operator
        cos(α) X + sin(α) Y.

    Args:
        psi:   state vector.
        qubit: which qubit to measure (MSB-first).
        angle: α (radians).
        rng:   generator.

    Returns:
        (collapsed_state, outcome)  with outcome ∈ {0, 1}.
        outcome = 0 means projection onto |+_α⟩, 1 → |−_α⟩.
    """
    n = int(np.log2(len(psi)))
    # Build the basis-change unitary U such that U|+_α⟩ = |0⟩, U|−_α⟩ = |1⟩.
    # |+_α⟩ = (|0⟩ + e^{iα}|1⟩)/√2 → row 0 of U is conj of this row vector.
    U = (1 / np.sqrt(2)) * np.array([
        [1, np.exp(-1j * angle)],
        [1, -np.exp(-1j * angle)],
    ], dtype=np.complex128)
    # Apply U to the target qubit.
    rotated = _apply_1q(psi, U, qubit, n)
    # Measure in computational basis on this qubit.
    p0 = 0.0
    for idx in range(2 ** n):
        bit = (idx >> (n - 1 - qubit)) & 1
        if bit == 0:
            p0 += abs(rotated[idx]) ** 2
    outcome = 0 if rng.uniform() < p0 else 1
    # Project + renormalize.
    collapsed = np.zeros_like(rotated)
    for idx in range(2 ** n):
        bit = (idx >> (n - 1 - qubit)) & 1
        if bit == outcome:
            collapsed[idx] = rotated[idx]
    collapsed = collapsed / np.linalg.norm(collapsed)
    # Rotate back to computational basis on the OTHER qubits.
    # (We leave the measured qubit in |outcome⟩ in the rotated basis.)
    final = _apply_1q(collapsed, U.conj().T, qubit, n)
    return final, outcome


# ----------------------------------------------------------------------------
# MBQC primitives
# ----------------------------------------------------------------------------

def mbqc_single_qubit_rotation(
    angle: float, input_state: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> dict:
    """Implement an Rz(θ) rotation via MBQC on a 2-qubit cluster chain.

    Procedure:
      1. Take input state |ψ_in⟩ ⊗ |+⟩.
      2. Apply CZ.
      3. Measure qubit 0 in basis cos(θ/2)X + sin(θ/2)Y → outcome m.
      4. The output (qubit 1) carries X^m H Rz(θ) |ψ_in⟩.
      5. Apply byproduct correction X^m to recover Rz(θ)|ψ_in⟩
         (the H is absorbed by the protocol design).

    For input |+⟩: Rz(θ)|+⟩ = (|0⟩ + e^{i θ}|1⟩) / √2.

    Returns:
        dict with output_state, outcome, ideal_state.
    """
    rng = rng or np.random.default_rng()
    if input_state is None:
        input_state = np.array([1, 1], dtype=np.complex128) / np.sqrt(2)
    # Step 1: |ψ_in⟩ ⊗ |+⟩
    plus = np.array([1, 1], dtype=np.complex128) / np.sqrt(2)
    psi = np.kron(input_state, plus)
    # Step 2: CZ(0, 1)
    psi = _apply_cz(psi, 0, 1, n=2)
    # Step 3: Measure qubit 0 in basis with angle = -θ (so the byproduct
    # is X^m and the output gets Rz(θ)).
    psi, outcome = measure_in_xy_basis(psi, qubit=0, angle=-angle, rng=rng)
    # Extract qubit-1 reduced state. Qubit 0 is in |outcome_α⟩ state.
    # We project on the relevant subspace.
    # After measurement, the state has qubit-0 in the rotated basis state.
    # We just trace out qubit 0.
    arr = psi.reshape(2, 2)
    # Find which arr row corresponds to outcome=outcome on qubit 0
    # AFTER the U† rotation back. Actually since we rotated back, qubit
    # 0 is in U†|outcome⟩ = the original basis vector |+_α⟩ or |−_α⟩.
    # For the output state on qubit 1, we project on either of those:
    if outcome == 0:
        v0 = (1 / np.sqrt(2)) * np.array([1, np.exp(1j * (-angle))])
    else:
        v0 = (1 / np.sqrt(2)) * np.array([1, -np.exp(1j * (-angle))])
    # Project: output_state on qubit 1 = ⟨v0|_0 ⊗ I_1 |ψ⟩.
    output_state = arr.T @ v0.conj()
    output_state = output_state / np.linalg.norm(output_state)
    # Step 5: byproduct correction X^m H.
    if outcome == 1:
        output_state = _X @ output_state
    output_state = _H @ output_state

    # Ideal target: Rz(θ) |input⟩.
    Rz = np.array([
        [np.exp(-1j * angle / 2), 0],
        [0, np.exp(1j * angle / 2)],
    ], dtype=np.complex128)
    ideal = Rz @ input_state
    return {
        "output_state":  output_state,
        "outcome":       outcome,
        "ideal_state":   ideal,
    }


def mbqc_fidelity(angle: float, input_state: np.ndarray,
                   n_trials: int = 100,
                   rng: np.random.Generator | None = None) -> float:
    """Average fidelity of the MBQC-implemented Rz(θ) over many trials."""
    rng = rng or np.random.default_rng()
    fid_sum = 0.0
    for _ in range(n_trials):
        r = mbqc_single_qubit_rotation(angle, input_state.copy(), rng=rng)
        # Fidelity = |⟨ideal|output⟩|².
        fid_sum += abs(np.vdot(r["ideal_state"], r["output_state"])) ** 2
    return fid_sum / n_trials
