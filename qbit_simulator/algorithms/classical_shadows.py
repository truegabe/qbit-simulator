"""Classical shadows: a sample-efficient measurement scheme.

Classical shadows (Huang-Kueng-Preskill 2020) let you estimate MANY
observables ⟨O_i⟩ from a SINGLE pool of measurement data, with sample
complexity scaling as O(log(M) / ε²) for M observables and target
error ε. This is exponentially better than naively measuring each
observable separately.

Procedure (for the **random-Clifford** version, used here as the
classical-Pauli sub-case):

  1. For each shot s = 1, …, N:
     a. Sample a random single-qubit Clifford C_s (we use the
        Pauli-group sub-case: random basis ∈ {X, Y, Z}).
     b. Apply C_s and measure in the computational basis → b_s.
     c. Form the "shadow"
            σ̂_s  =  ⊗_i  M⁻¹( C_s_i† |b_s_i⟩⟨b_s_i| C_s_i )
        where M⁻¹ is the inverse depolarizing-style map. For random
        single-qubit Pauli measurements, M⁻¹(ρ) = 3 ρ - I.

  2. Estimate any observable O via
            ⟨O⟩  ≈  median-of-means_{s} Tr(O · σ̂_s).

For local Pauli observables of weight k, the per-shot estimator
variance is O(4^k), so the total number of shots is O(4^k log(M) / ε²).

This module provides:

  - `random_pauli_basis(n_qubits, rng)`: sample a single random basis
    string (e.g. "XZYZ").
  - `apply_basis_measurement(psi, basis, rng)`: measure psi in `basis`,
    return outcome bits.
  - `single_shadow(psi, rng)`: produce one (basis, outcome) shadow.
  - `collect_shadows(psi, n_shots, rng)`: many shadows at once.
  - `shadow_estimate(shadows, pauli_string)`: ⟨P⟩ estimate from
    shadow data.
  - `shadow_estimate_observable(shadows, observable_paulis)`: estimate
    a Pauli-sum observable.
"""

from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------
# Random Pauli basis sampling and measurement
# ----------------------------------------------------------------------------

_BASIS_ROTATION = {
    "X": np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2),    # H: Z → X
    "Y": np.array([[1, -1j], [1, 1j]], dtype=np.complex128) / np.sqrt(2),  # S†H rotation
    "Z": np.eye(2, dtype=np.complex128),                                    # I
}


def random_pauli_basis(n_qubits: int, rng: np.random.Generator) -> str:
    """Return a length-n string with each character ∈ {X, Y, Z}."""
    return "".join(rng.choice(["X", "Y", "Z"], size=n_qubits))


def _rotate_to_basis(psi: np.ndarray, basis: str) -> np.ndarray:
    """Apply per-qubit basis-change rotations so measurement in Z
    corresponds to measurement in `basis` on the original state."""
    n = len(basis)
    arr = psi.reshape([2] * n)
    for q, b in enumerate(basis):
        if b == "Z":
            continue
        R = _BASIS_ROTATION[b]
        # Move axis q to front, apply R, move back. Qubit q = axis q
        # (MSB-first; q=0 is leftmost).
        arr = np.moveaxis(arr, q, 0)
        arr = R @ arr.reshape(2, -1)
        arr = arr.reshape([2] + [2] * (n - 1))
        arr = np.moveaxis(arr, 0, q)
    return arr.reshape(2 ** n)


def apply_basis_measurement(psi: np.ndarray, basis: str,
                              rng: np.random.Generator
                              ) -> list[int]:
    """Sample a computational-basis outcome from psi rotated into `basis`."""
    n = len(basis)
    rotated = _rotate_to_basis(psi, basis)
    probs = np.abs(rotated) ** 2
    probs = probs / probs.sum()
    idx = int(rng.choice(2 ** n, p=probs))
    # Decode bit pattern (MSB-first).
    bits = [(idx >> (n - 1 - q)) & 1 for q in range(n)]
    return bits


# ----------------------------------------------------------------------------
# Shadow construction
# ----------------------------------------------------------------------------

def single_shadow(
    psi: np.ndarray, rng: np.random.Generator,
) -> tuple[str, list[int]]:
    """Produce one classical shadow: (basis, outcome bits)."""
    n = int(np.log2(len(psi)))
    basis = random_pauli_basis(n, rng)
    outcome = apply_basis_measurement(psi, basis, rng)
    return basis, outcome


def collect_shadows(
    psi: np.ndarray, n_shots: int, rng: np.random.Generator,
) -> list[tuple[str, list[int]]]:
    """Sample n_shots shadows from psi."""
    return [single_shadow(psi, rng) for _ in range(n_shots)]


# ----------------------------------------------------------------------------
# Pauli-string estimation
# ----------------------------------------------------------------------------

def _single_shadow_pauli_estimate(
    basis: str, outcome: list[int], pauli_string: str,
) -> float:
    """Per-shot estimator of ⟨P⟩ from one shadow.

    For a Pauli string P with support on qubits S = {i : P_i ≠ I}, the
    classical-shadows formula for random single-qubit Pauli bases is:

        T̂_P  =  prod_{i ∈ S}  3 · δ(basis_i, P_i) · (1 - 2 · outcome_i)

    The δ(basis_i, P_i) is 1 if the random basis matched P at that
    qubit, else 0; if any unmatched qubit lies in S, the shot
    contributes 0 to this Pauli.
    """
    if len(basis) != len(pauli_string):
        raise ValueError("basis and Pauli string must have equal length")
    val = 1.0
    for i, p in enumerate(pauli_string):
        if p == "I":
            continue
        if basis[i] != p:
            return 0.0
        # Each matched non-identity Pauli contributes 3 · (±1).
        val *= 3.0 * (1 - 2 * outcome[i])
    return val


def shadow_estimate(
    shadows: list[tuple[str, list[int]]], pauli_string: str,
) -> float:
    """Estimate ⟨P⟩ as the mean of per-shot estimators."""
    if not shadows:
        return 0.0
    vals = [_single_shadow_pauli_estimate(b, o, pauli_string)
            for (b, o) in shadows]
    return float(np.mean(vals))


def shadow_estimate_observable(
    shadows: list[tuple[str, list[int]]],
    observable_paulis: list[tuple[complex, str]],
) -> complex:
    """Estimate ⟨O⟩ = Σ_k coef_k ⟨P_k⟩ using the same shadow pool."""
    total = 0 + 0j
    for coef, s in observable_paulis:
        total += coef * shadow_estimate(shadows, s)
    return complex(total)


# ----------------------------------------------------------------------------
# Median-of-means (variance-reducing wrapper)
# ----------------------------------------------------------------------------

def shadow_estimate_median_of_means(
    shadows: list[tuple[str, list[int]]],
    pauli_string: str,
    n_chunks: int = 5,
) -> float:
    """Median-of-means variant: split shadows into n_chunks, compute the
    mean over each, return the median. More robust to heavy-tailed
    estimators."""
    n = len(shadows)
    if n_chunks <= 0 or n < n_chunks:
        return shadow_estimate(shadows, pauli_string)
    chunk_size = n // n_chunks
    means = []
    for c in range(n_chunks):
        chunk = shadows[c * chunk_size:(c + 1) * chunk_size]
        means.append(shadow_estimate(chunk, pauli_string))
    return float(np.median(means))
