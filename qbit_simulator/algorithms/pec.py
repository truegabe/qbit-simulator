"""Probabilistic Error Cancellation (PEC).

PEC (Temme-Bravyi-Gambetta 2017) is a near-term error-mitigation
technique that produces an UNBIASED estimate of a noiseless expectation
value ⟨O⟩_ideal from a noisy device, at the cost of increased sample
variance.

Idea: express the inverse N⁻¹ of a noise channel N as a linear (real)
combination of EXPERIMENTALLY-IMPLEMENTABLE operations {ε_k}:

    N⁻¹  =  sum_k η_k · ε_k

where η_k are real numbers (possibly negative — that's where the
"probabilistic" comes in). Then:

    ⟨O⟩_ideal  =  Tr(O · N⁻¹(N(ρ_ideal)))
                =  sum_k η_k · Tr(O · ε_k(ρ_noisy))

We sample operation ε_k with probability p_k = |η_k| / γ (where
γ = sum_k |η_k|), multiply each measurement by sign(η_k) · γ, and
average.

The "cost" of mitigation is γ², which inflates the sample variance:

    Var[estimator]  =  γ² · σ²_shot

For depolarizing noise N_p(ρ) = (1-p)ρ + (p/2^n) I, the inverse has
γ = (1 + 3p/(2^n - p)) ~ 1 + O(p) — so PEC is practical only for small
p.

This module provides:

  - `depolarizing_inverse_quasiprobs(p, n)`: the η-vector for inverting
    a single-qubit depolarizing channel.
  - `pec_estimator(noisy_circuit_fn, observable, p_noise, n_shots, rng)`:
    run PEC on a single-qubit noisy circuit and return a mitigated
    estimate.
  - `pec_sampling_cost(eta)`: the variance-inflation factor γ².
"""

from __future__ import annotations

from typing import Callable

import numpy as np


# ----------------------------------------------------------------------------
# Quasi-probability decomposition for depolarizing noise
# ----------------------------------------------------------------------------

def depolarizing_inverse_quasiprobs(p: float, n_qubits: int = 1) -> dict:
    """Quasi-probability decomposition of N⁻¹ for the depolarizing
    channel on n_qubits.

    The depolarizing channel:
        N_p(ρ) = (1 − p) ρ + (p / 2^n) · I · Tr(ρ)
               = (1 − p + p / 4^n) ρ + (p / 4^n) · sum_{P ≠ I} P ρ P

    Inverse (in the Pauli basis):
        N_p⁻¹(ρ) = (1 − p_inv) ρ + (p_inv / (4^n − 1)) · sum_{P ≠ I} P ρ P
    with p_inv chosen so that N · N⁻¹ = I.

    Returns:
        dict with η (the coefficient list) and a list of "operations"
        (Pauli strings to apply).
    """
    d = 2 ** n_qubits
    # Effective depolarizing strength: N_p = (1 - p_eff) ρ + (p_eff/d²) Σ PρP.
    # In the standard form: N_p(ρ) = (1 - p) ρ + p · I/d.
    # Pauli-basis form: (1 - p + p/d²) I + (p/d²)(d² - 1) "mix" component.
    # Identity-superoperator coefficient: λ = 1 - p · (d² - 1) / d².
    # Then N_p⁻¹: η_I = (1 - 1/λ)·(1 - 1/d²) + 1/λ ; η_other = (1 - 1/λ) / d².
    # We use the practical form:
    eta_I = (1 - p / (d ** 2)) / (1 - p)
    eta_other = -p / (d ** 2 * (1 - p))
    # Build list of (η, operation_label).
    eta_list = [(eta_I, "I" * n_qubits)]
    # Non-identity n-Pauli strings.
    from itertools import product
    for combo in product("IXYZ", repeat=n_qubits):
        s = "".join(combo)
        if s == "I" * n_qubits:
            continue
        eta_list.append((eta_other, s))
    return {
        "eta":              [e for e, _ in eta_list],
        "operations":       [s for _, s in eta_list],
        "gamma":            sum(abs(e) for e, _ in eta_list),
        "n_qubits":         n_qubits,
        "p":                p,
    }


def pec_sampling_cost(eta: list[float]) -> float:
    """Sampling-overhead factor γ²: variance multiplied compared to
    unmitigated sampling."""
    gamma = sum(abs(e) for e in eta)
    return float(gamma ** 2)


# ----------------------------------------------------------------------------
# Applying a Pauli string
# ----------------------------------------------------------------------------

_PAULIS = {
    "I": np.eye(2, dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
}


def _pauli_string_matrix(s: str) -> np.ndarray:
    M = np.array([[1.0 + 0j]])
    for ch in s:
        M = np.kron(M, _PAULIS[ch])
    return M


def apply_pauli_to_state(psi: np.ndarray, pauli_str: str) -> np.ndarray:
    """Apply a Pauli string to a state vector."""
    return _pauli_string_matrix(pauli_str) @ psi


# ----------------------------------------------------------------------------
# PEC estimator
# ----------------------------------------------------------------------------

def pec_estimator(
    ideal_state_fn: Callable[[], np.ndarray],
    noisy_state_fn: Callable[[np.ndarray], np.ndarray],
    observable: np.ndarray,
    p_noise: float,
    n_shots: int = 1000,
    rng: np.random.Generator | None = None,
) -> dict:
    """Run PEC to estimate ⟨O⟩_ideal from a noisy implementation.

    Args:
        ideal_state_fn:  callable returning the IDEAL state vector for
                         the circuit (the target preparation).
        noisy_state_fn:  callable(psi_in) → psi_out applying the noisy
                         channel.
        observable:      Hermitian matrix O.
        p_noise:         depolarizing rate of the noise.
        n_shots:         number of PEC samples.
        rng:             generator.

    Returns:
        dict with mitigated_estimate, unmitigated_estimate, ideal_value,
        variance.
    """
    rng = rng or np.random.default_rng()
    n_qubits = int(np.log2(observable.shape[0]))
    qp = depolarizing_inverse_quasiprobs(p_noise, n_qubits)
    etas = np.array(qp["eta"])
    ops = qp["operations"]
    gamma = qp["gamma"]
    probs = np.abs(etas) / gamma

    # Ideal expectation (for reference).
    psi_ideal = ideal_state_fn()
    O_ideal = float(np.real(psi_ideal.conj() @ observable @ psi_ideal))

    # Unmitigated noisy estimate: average over many noise realizations.
    n_baseline = 200
    unmit_vals = []
    for _ in range(n_baseline):
        psi_noisy = noisy_state_fn(psi_ideal)
        unmit_vals.append(float(np.real(psi_noisy.conj() @ observable @ psi_noisy)))
    O_unmit = float(np.mean(unmit_vals))

    # PEC sampling: each shot draws a FRESH noise realization, picks a
    # PEC operation, and reports a weighted measurement.
    samples = np.zeros(n_shots)
    for s in range(n_shots):
        psi_noisy = noisy_state_fn(psi_ideal)
        k = int(rng.choice(len(etas), p=probs))
        psi_modified = apply_pauli_to_state(psi_noisy, ops[k])
        meas = float(np.real(psi_modified.conj() @ observable @ psi_modified))
        samples[s] = meas * gamma * np.sign(etas[k])

    mitigated = float(np.mean(samples))
    variance = float(np.var(samples) / n_shots)
    return {
        "mitigated_estimate":   mitigated,
        "unmitigated_estimate": O_unmit,
        "ideal_value":          O_ideal,
        "variance":             variance,
        "n_shots":              n_shots,
        "gamma":                gamma,
        "sampling_overhead":    gamma ** 2,
    }
