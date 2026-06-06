"""Quantum Volume (QV): IBM's holistic benchmark.

Quantum volume (Cross-Bishop-Sheldon-Nation-Gambetta 2019) is a SINGLE
NUMBER characterizing the largest "square" circuit a device can run
successfully. Specifically:

    QV  =  2^d_max

where d_max is the largest d such that the device runs depth-d, width-d
random "Model Circuits" with success probability > 2/3.

A Model Circuit on d qubits:

  1. For each layer t = 1, …, d:
     a. Pick a random permutation π_t of the d qubits.
     b. For each pair (π_t(2k), π_t(2k+1)), apply a Haar-random U(4)
        gate (a "random 2-qubit Clifford+entangling block").
  2. The depth = width = d.

The success criterion (heavy-output generation, HOG):

    Run the circuit on the device, measure once, get a bitstring b.
    Compute the IDEAL probability p_ideal(b) for that bitstring.
    Sort all 2^d bitstring probabilities; the "heavy" half is the top.
    The device "succeeds" if  Pr[b is heavy] > 2/3 + ε on average.

The threshold 2/3 comes from the Porter-Thomas distribution: a perfect
random circuit has mean heavy-output probability ≈ 0.8474.

This module provides:

  - `model_circuit(d, rng)`: build a single QV model circuit + return
    its final state vector (the ideal output).
  - `heavy_output_set(probs)`: top-half-probability bitstrings.
  - `quantum_volume_trial(d, noise_level, rng)`: simulate one trial
    with optional depolarizing noise.
  - `quantum_volume_estimate(d, n_trials, noise_level, rng)`: many
    trials, compute the empirical HOG probability.
  - `find_quantum_volume(max_d, n_trials_per_d, ...)`: sweep d to find
    the largest passing depth.
"""

from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------
# Random 2-qubit unitary
# ----------------------------------------------------------------------------

def _haar_u4(rng: np.random.Generator) -> np.ndarray:
    """Sample a Haar-random unitary in U(4)."""
    A = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    Q, R = np.linalg.qr(A)
    D = np.diag(np.diag(R) / np.abs(np.diag(R)))
    return Q @ D


# ----------------------------------------------------------------------------
# Model circuit
# ----------------------------------------------------------------------------

def _apply_2q_on_pair(psi: np.ndarray, U: np.ndarray,
                       q0: int, q1: int, n: int) -> np.ndarray:
    """Apply a 4x4 gate on qubits (q0, q1) of an n-qubit state vector."""
    if q1 < q0:
        swap = np.array([
            [1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1],
        ], dtype=np.complex128)
        U = swap @ U @ swap
        q0, q1 = q1, q0
    arr = psi.reshape([2] * n)
    arr = np.moveaxis(arr, [q0, q1], [0, 1])
    arr = arr.reshape(4, -1)
    arr = U @ arr
    arr = arr.reshape([2, 2] + [2] * (n - 2))
    arr = np.moveaxis(arr, [0, 1], [q0, q1])
    return arr.reshape(2 ** n)


def model_circuit(d: int, rng: np.random.Generator) -> dict:
    """Build a single Quantum Volume model circuit on d qubits.

    Each layer:
      1. Random permutation of d qubits.
      2. For each pair (0,1), (2,3), ..., apply a Haar-random U(4).

    Args:
        d:   qubits AND depth (width=depth, the "square" structure).
        rng: generator.

    Returns:
        dict with final_state, n_layers, permutations, gates.
    """
    n = d
    psi = np.zeros(2 ** n, dtype=np.complex128)
    psi[0] = 1.0
    permutations = []
    gates_per_layer = []
    for layer in range(d):
        perm = rng.permutation(n)
        permutations.append(perm.copy())
        gates = []
        for pair_idx in range(n // 2):
            q0 = int(perm[2 * pair_idx])
            q1 = int(perm[2 * pair_idx + 1])
            U = _haar_u4(rng)
            gates.append((q0, q1, U))
            psi = _apply_2q_on_pair(psi, U, q0, q1, n)
        gates_per_layer.append(gates)
    return {
        "final_state":  psi,
        "n_qubits":     n,
        "depth":        d,
        "permutations": permutations,
        "gates":        gates_per_layer,
    }


# ----------------------------------------------------------------------------
# Heavy output set
# ----------------------------------------------------------------------------

def heavy_output_set(probs: np.ndarray) -> set[int]:
    """Set of bit-string indices whose probability is ABOVE the median."""
    sorted_p = np.sort(probs)
    # Median: the d-th smallest, where d = len/2.
    median = sorted_p[len(sorted_p) // 2]
    return set(i for i, p in enumerate(probs) if p > median)


def heavy_output_probability(probs: np.ndarray) -> float:
    """Sum of probabilities of heavy outputs (HOG metric for one circuit).

    For ideal Porter-Thomas circuits, this converges to (1 + ln 2) / 2 ≈ 0.847.
    """
    heavy = heavy_output_set(probs)
    return float(sum(probs[i] for i in heavy))


# ----------------------------------------------------------------------------
# Noise model + trial
# ----------------------------------------------------------------------------

def _apply_depolarizing(probs: np.ndarray, p_noise: float) -> np.ndarray:
    """Mix the ideal probability distribution with uniform noise.

    p_noisy = (1 − p_noise) * p_ideal + p_noise * (1/D).

    Models the cumulative effect of incoherent errors during the circuit.
    """
    D = len(probs)
    return (1 - p_noise) * probs + p_noise / D


def quantum_volume_trial(
    d: int, noise_level: float = 0.0,
    rng: np.random.Generator | None = None,
) -> dict:
    """Run a single QV trial: build the circuit, sample a bitstring
    (according to the noisy distribution), check if it's heavy.

    Args:
        d:           qubits = depth.
        noise_level: 0 = ideal, 1 = fully depolarized.
        rng:         generator.

    Returns:
        dict with the ideal heavy probability, sampled bitstring, was-heavy flag.
    """
    rng = rng or np.random.default_rng()
    mc = model_circuit(d, rng)
    psi = mc["final_state"]
    p_ideal = np.abs(psi) ** 2
    p_noisy = _apply_depolarizing(p_ideal, noise_level)
    p_noisy = p_noisy / p_noisy.sum()
    sampled = int(rng.choice(len(p_noisy), p=p_noisy))
    heavy = heavy_output_set(p_ideal)
    return {
        "d":                  d,
        "was_heavy":          sampled in heavy,
        "heavy_prob_ideal":   heavy_output_probability(p_ideal),
        "sampled_bitstring":  sampled,
    }


def quantum_volume_estimate(
    d: int, n_trials: int = 100,
    noise_level: float = 0.0,
    rng: np.random.Generator | None = None,
) -> dict:
    """Run n_trials QV trials at depth d; report the heavy-output rate."""
    rng = rng or np.random.default_rng()
    n_heavy = 0
    heavy_probs = []
    for _ in range(n_trials):
        t = quantum_volume_trial(d, noise_level, rng)
        if t["was_heavy"]:
            n_heavy += 1
        heavy_probs.append(t["heavy_prob_ideal"])
    return {
        "d":              d,
        "n_trials":       n_trials,
        "heavy_rate":     n_heavy / n_trials,
        "mean_heavy_prob_ideal":  float(np.mean(heavy_probs)),
        "passes_2/3_threshold":   (n_heavy / n_trials) > 2 / 3,
    }


# ----------------------------------------------------------------------------
# Find QV
# ----------------------------------------------------------------------------

def find_quantum_volume(
    max_d: int = 6, n_trials_per_d: int = 50,
    noise_level: float = 0.0,
    rng: np.random.Generator | None = None,
) -> dict:
    """Sweep d from 2 upward until the 2/3 heavy-rate threshold fails.

    Returns:
        dict with the largest passing d and the corresponding 2^d.
    """
    rng = rng or np.random.default_rng()
    results = []
    largest_pass = 0
    for d in range(2, max_d + 1):
        r = quantum_volume_estimate(d, n_trials_per_d, noise_level, rng)
        results.append(r)
        if r["passes_2/3_threshold"]:
            largest_pass = max(largest_pass, d)
    return {
        "quantum_volume": 2 ** largest_pass if largest_pass > 0 else 1,
        "largest_d":      largest_pass,
        "per_d_results":  results,
    }
