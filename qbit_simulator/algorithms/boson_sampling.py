"""Boson sampling — Aaronson & Arkhipov 2011.

A non-universal quantum-computing paradigm that's nonetheless believed to
demonstrate quantum advantage: send N indistinguishable bosons (photons)
through a random linear optical network represented by an M×M unitary U,
then measure the photon counts in each mode. The output probability for a
given mode occupation pattern is proportional to the squared modulus of a
matrix permanent of a submatrix of U:

    P(output_pattern) ∝ |Per(U_{S,T})|² / (s! · t!)

where S is the multiset of input modes (typically the first N) and T is the
output multiset. Computing matrix permanents is #P-hard in general
(Valiant 1979), which underlies the classical hardness of boson sampling.

This module:
    - Implements Ryser's O(2^N · N) permanent algorithm.
    - Provides a sampler that draws output patterns according to the boson-
      sampling distribution.
    - Useful for benchmarking, didactic demos, and verifying that small
      instances of boson sampling produce the expected statistics.

For N = 6 photons in M = 12 modes, we can sample tens to hundreds of
patterns per second on a laptop. Larger scales are classical-intractable
in principle (and have been used as quantum-supremacy benchmarks).
"""

from __future__ import annotations

from itertools import combinations_with_replacement

import numpy as np


def permanent(M: np.ndarray) -> complex:
    """Compute the matrix permanent via Ryser's formula.

    Per(M) = (-1)^n · Σ_{S ⊆ [n]} (-1)^|S| · ∏_i Σ_{j ∈ S} M[i, j]

    Complexity O(2^n · n) — practical up to n ≈ 20.
    """
    n = M.shape[0]
    if M.shape != (n, n):
        raise ValueError("permanent requires a square matrix")
    if n == 0:
        return complex(1)
    total = complex(0)
    for mask in range(1 << n):
        bits = bin(mask).count("1")
        # Sum the columns in S.
        col_sum = np.zeros(n, dtype=np.complex128)
        for j in range(n):
            if (mask >> j) & 1:
                col_sum += M[:, j]
        prod = np.prod(col_sum)
        sign = (-1) ** (n - bits)
        total += sign * prod
    return total


def random_haar_unitary(m: int, rng: np.random.Generator | None = None
                         ) -> np.ndarray:
    """Sample an m×m unitary uniformly from the Haar measure."""
    rng = rng or np.random.default_rng()
    z = (rng.normal(size=(m, m)) + 1j * rng.normal(size=(m, m))) / np.sqrt(2)
    Q, R = np.linalg.qr(z)
    # Make the diagonal of R positive so Q is uniformly Haar.
    D = np.diag(R) / np.abs(np.diag(R))
    return Q * D


def boson_sampling_probability(
    U: np.ndarray,
    input_modes: list[int],
    output_modes: list[int],
) -> float:
    """Probability of detecting `output_modes` (a list of N output indices,
    possibly with repetition) given input photons in `input_modes`.

        P = |Per(U[output, input])|² / (prod_i input_mult_i! · prod_j output_mult_j!)

    Args:
        U: M×M scattering unitary.
        input_modes:  multiset of N input mode indices (e.g. [0, 1, 2]).
        output_modes: multiset of N output mode indices.

    Returns:
        Probability (real, non-negative).
    """
    if len(input_modes) != len(output_modes):
        raise ValueError("input and output must have the same photon number")
    # Build the N×N submatrix: rows indexed by output modes, cols by input modes.
    sub = U[np.array(output_modes)[:, None], np.array(input_modes)[None, :]]
    p = abs(permanent(sub)) ** 2
    # Normalize by photon-bunching factorials.
    from math import factorial
    from collections import Counter
    in_counts  = Counter(input_modes)
    out_counts = Counter(output_modes)
    norm = 1.0
    for c in in_counts.values():
        norm *= factorial(c)
    for c in out_counts.values():
        norm *= factorial(c)
    return float(p / norm)


def all_output_patterns(n_photons: int, n_modes: int) -> list[list[int]]:
    """Enumerate all bosonic occupation patterns of `n_photons` photons in
    `n_modes` modes — i.e. all length-n multisets of mode indices."""
    return [list(c) for c in combinations_with_replacement(range(n_modes), n_photons)]


def boson_sampling_distribution(
    U: np.ndarray, input_modes: list[int],
) -> dict[tuple[int, ...], float]:
    """Full probability distribution over output patterns (only for small N)."""
    n_modes = U.shape[0]
    n_photons = len(input_modes)
    out: dict[tuple[int, ...], float] = {}
    for pattern in all_output_patterns(n_photons, n_modes):
        p = boson_sampling_probability(U, input_modes, pattern)
        out[tuple(pattern)] = p
    return out


def sample_boson_pattern(
    U: np.ndarray, input_modes: list[int],
    rng: np.random.Generator | None = None,
) -> tuple[int, ...]:
    """Sample one output pattern from the boson-sampling distribution."""
    rng = rng or np.random.default_rng()
    dist = boson_sampling_distribution(U, input_modes)
    patterns = list(dist.keys())
    probs = np.array([dist[p] for p in patterns], dtype=np.float64)
    # Normalize for safety (small numerical drift possible).
    probs = np.clip(probs, 0, None)
    probs /= probs.sum()
    idx = int(rng.choice(len(patterns), p=probs))
    return patterns[idx]
