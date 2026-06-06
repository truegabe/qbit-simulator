"""Random Circuit Sampling (RCS) and Cross-Entropy Benchmarking (XEB).

Random circuit sampling was Google's quantum-supremacy benchmark (2019).
The setup:

  1. Build a random circuit on n qubits of moderate depth (e.g. 20).
     - At each layer: apply random single-qubit gates from a fixed set
       (e.g. {√X, √Y, T}) to every qubit, then apply 2-qubit entangling
       gates (e.g. iSWAP-like) on a chosen pattern of qubit pairs.
  2. Sample bit-strings from |⟨b | ψ_circ⟩|².
  3. The output distribution is Porter-Thomas: exp(-D·p) where
     D = 2^n. Heavily peaked on a few bit strings.
  4. **Linear Cross-Entropy Benchmark (XEB)**: a fidelity-style metric

         F_XEB = D · ⟨ p_circ(b_sample) ⟩  −  1

     With true samples from the circuit, ⟨F_XEB⟩ = 1; uniform samples
     give 0. Any noise reduces F_XEB linearly.

This module provides:

  - `random_layer_unitary(n_qubits, rng)`: build a random single-qubit
    layer (independent Haar-random U(2) per qubit).
  - `entangling_layer_unitary(n_qubits, pattern, rng)`: apply iSWAP /
    CZ to a list of qubit pairs.
  - `build_random_circuit(n_qubits, depth, rng)`: full circuit
    (returns the resulting state vector + the unitary).
  - `linear_xeb(p_circ, samples)`: compute F_XEB.
  - `porter_thomas_kolmogorov(p_circ)`: distance from p to the
    Porter-Thomas distribution.
"""

from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------
# Random gates
# ----------------------------------------------------------------------------

def _haar_su2(rng: np.random.Generator) -> np.ndarray:
    """Sample a Haar-random SU(2) matrix."""
    A = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    Q, R = np.linalg.qr(A)
    # Phase-fix for Haar measure.
    D = np.diag(np.diag(R) / np.abs(np.diag(R)))
    Q = Q @ D
    Q = Q / np.linalg.det(Q) ** 0.5
    return Q


_ISWAP = np.array([
    [1, 0, 0, 0],
    [0, 0, 1j, 0],
    [0, 1j, 0, 0],
    [0, 0, 0, 1],
], dtype=np.complex128)

_CZ = np.diag([1, 1, 1, -1]).astype(np.complex128)


def random_layer_unitary(n_qubits: int,
                          rng: np.random.Generator) -> list[np.ndarray]:
    """A list of n_qubits 2×2 Haar-random unitaries to apply per qubit."""
    return [_haar_su2(rng) for _ in range(n_qubits)]


# ----------------------------------------------------------------------------
# Gate application helpers
# ----------------------------------------------------------------------------

def _apply_1q(psi: np.ndarray, gate: np.ndarray, q: int, n: int) -> np.ndarray:
    """Apply 1q gate on qubit q (MSB-first convention)."""
    arr = psi.reshape([2] * n)
    arr = np.moveaxis(arr, q, 0)
    arr = gate @ arr.reshape(2, -1)
    arr = arr.reshape([2] + [2] * (n - 1))
    arr = np.moveaxis(arr, 0, q)
    return arr.reshape(2 ** n)


def _apply_2q(psi: np.ndarray, gate: np.ndarray, q0: int, q1: int,
              n: int) -> np.ndarray:
    if q1 < q0:
        swap = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=complex)
        gate = swap @ gate @ swap
        q0, q1 = q1, q0
    arr = psi.reshape([2] * n)
    arr = np.moveaxis(arr, [q0, q1], [0, 1])
    arr = gate @ arr.reshape(4, -1)
    arr = arr.reshape([2, 2] + [2] * (n - 2))
    arr = np.moveaxis(arr, [0, 1], [q0, q1])
    return arr.reshape(2 ** n)


# ----------------------------------------------------------------------------
# Build a random circuit
# ----------------------------------------------------------------------------

def build_random_circuit(
    n_qubits: int, depth: int,
    entangling_gate: str = "iswap",
    rng: np.random.Generator | None = None,
) -> dict:
    """Build a random circuit of `depth` (1q + 2q) layers.

    Each layer:
      - apply independent Haar-random SU(2) gates to each qubit
      - then apply `entangling_gate` (iSWAP or CZ) to all
        nearest-neighbor pairs in an alternating pattern.

    Args:
        n_qubits:       qubit count.
        depth:          number of layers.
        entangling_gate: "iswap" or "cz".
        rng:            generator.

    Returns:
        dict with final_state, layers (list of single-qubit gate lists).
    """
    rng = rng or np.random.default_rng()
    if entangling_gate == "iswap":
        ENT = _ISWAP
    elif entangling_gate == "cz":
        ENT = _CZ
    else:
        raise ValueError(f"unknown gate {entangling_gate}")

    psi = np.zeros(2 ** n_qubits, dtype=np.complex128)
    psi[0] = 1.0
    layers = []
    for layer in range(depth):
        gates_1q = random_layer_unitary(n_qubits, rng)
        layers.append(gates_1q)
        for q in range(n_qubits):
            psi = _apply_1q(psi, gates_1q[q], q, n_qubits)
        # Entangling layer: alternating even / odd pairs.
        parity = layer % 2
        for q in range(parity, n_qubits - 1, 2):
            psi = _apply_2q(psi, ENT, q, q + 1, n_qubits)
    return {
        "final_state":  psi,
        "n_qubits":     n_qubits,
        "depth":        depth,
        "layers":       layers,
    }


# ----------------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------------

def sample_circuit(psi: np.ndarray, n_samples: int,
                    rng: np.random.Generator) -> list[int]:
    """Sample bit-strings from |ψ|²."""
    probs = np.abs(psi) ** 2
    probs = probs / probs.sum()
    return list(rng.choice(len(probs), size=n_samples, p=probs))


# ----------------------------------------------------------------------------
# Cross-entropy benchmark
# ----------------------------------------------------------------------------

def linear_xeb(p_circ: np.ndarray, samples: list[int]) -> float:
    """Linear cross-entropy fidelity:

        F_XEB = D · ⟨p_circ(b)⟩_b∈samples  −  1

    Returns 1 for perfect samples, 0 for uniform, − for negative-bias.
    """
    D = len(p_circ)
    if not samples:
        return 0.0
    return float(D * np.mean([p_circ[b] for b in samples]) - 1)


def linear_xeb_uniform_baseline(p_circ: np.ndarray) -> float:
    """The expected F_XEB if samples are drawn UNIFORMLY at random."""
    D = len(p_circ)
    return float(D * np.mean(p_circ) - 1)   # always ≈ 0


# ----------------------------------------------------------------------------
# Porter-Thomas distribution
# ----------------------------------------------------------------------------

def porter_thomas_pdf(p: np.ndarray, D: int) -> np.ndarray:
    """The Porter-Thomas density f(p) = D · exp(-D · p)."""
    return D * np.exp(-D * p)


def porter_thomas_chi2_distance(p_circ: np.ndarray,
                                  n_bins: int = 20) -> float:
    """χ² distance between the empirical p-distribution of `p_circ`
    and the Porter-Thomas distribution f(p) = D·e^{−Dp}.

    Small (≪1) means the circuit is well-randomizing.
    """
    D = len(p_circ)
    # Normalize x-coordinate to D·p (unit-mean exponential expected).
    xs = D * np.asarray(p_circ)
    # Histogram with bins on [0, 6] of x = D·p.
    bins = np.linspace(0, 6.0, n_bins + 1)
    hist, _ = np.histogram(xs, bins=bins, density=True)
    centers = 0.5 * (bins[:-1] + bins[1:])
    expected = np.exp(-centers)   # PDF of Exp(1) at center
    # χ² distance: sum((hist - expected)²/expected).
    valid = expected > 1e-9
    return float(np.sum((hist[valid] - expected[valid]) ** 2
                         / expected[valid]) / valid.sum())
