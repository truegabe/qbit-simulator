"""Shor's algorithm — quantum factoring.

For an odd composite N, find a non-trivial factor by computing the period r
of f(x) = a^x mod N for a random a coprime to N. Period finding uses QPE on
the modular-exponentiation unitary U_a |y⟩ = |ay mod N⟩.

We implement the full algorithm, but use the *exact* modular-multiplication
unitary built as a permutation matrix rather than a gate-level decomposition.
This is the standard "Shor on a simulator" approach — quantum-correct in
behavior, classical in construction.
"""

from __future__ import annotations

from math import gcd
import random

import numpy as np

from ..circuit import QuantumCircuit
from .qpe import (
    phase_estimation,
    phase_estimation_modexp,
    phase_estimation_modexp_marginal,
    estimate_phase_from_state,
)


def modular_multiplication_unitary(a: int, N: int, n_target: int) -> np.ndarray:
    """Build the 2^n_target x 2^n_target permutation U_a |y⟩ = |ay mod N⟩.

    For y ≥ N (i.e. y not in [0, N)) we leave it fixed (|y⟩ → |y⟩) so the
    matrix is a permutation and hence unitary.
    """
    dim = 2**n_target
    U = np.zeros((dim, dim), dtype=np.complex128)
    for y in range(dim):
        if y < N:
            new_y = (a * y) % N
        else:
            new_y = y
        U[new_y, y] = 1.0
    return U


def continued_fraction_period(measured: int, n_counting: int, N: int) -> int | None:
    """Recover the period r from a QPE measurement s = measured / 2^n_counting.

    The phase φ ≈ s/r for some integer s; we find the rational approximation
    to measured/2^n_counting with denominator at most N using continued fractions.
    """
    if measured == 0:
        return None
    num, den = measured, 2**n_counting
    convergents = []
    a, b = num, den
    while b:
        convergents.append(a // b)
        a, b = b, a % b
    # Reconstruct convergents
    h0, h1 = 1, convergents[0] if convergents else 1
    k0, k1 = 0, 1
    candidates = [(h1, k1)]
    for c in convergents[1:]:
        h2 = c * h1 + h0
        k2 = c * k1 + k0
        if k2 > N:
            break
        candidates.append((h2, k2))
        h0, h1 = h1, h2
        k0, k1 = k1, k2
    # Return the largest denominator (best period candidate) ≤ N.
    for _, k in reversed(candidates):
        if 1 < k <= N:
            return k
    return None


def shor(
    N: int,
    max_attempts: int = 20,
    rng: random.Random | None = None,
    counting_extra: int | None = None,
    sparse: bool = True,
) -> list[int]:
    """Factor N. Returns a list of two non-trivial factors, or [N] if it fails.

    Total qubits used: `n_target + n_counting` where
        n_target  = ceil(log2(N))
        n_counting = n_target + counting_extra   (default: scales with N)

    counting_extra:
        Number of extra precision bits beyond `n_target`. The textbook Shor
        choice is `counting_extra = n_target` (i.e. `t = 2k`), which uses
        3·k total qubits — state size 2^{3k} · 16 bytes blows past 8 GB at
        k ≥ 10. Reducing `counting_extra` to ~4 still recovers the period
        with high probability (continued fractions only needs the phase to
        be accurate to a few bits past log2(N)), at the cost of slightly
        more retries. We auto-scale: full 2k below k=9, then taper.
    """
    if N < 2:
        raise ValueError("N must be >= 2")
    if N % 2 == 0:
        return [2, N // 2]

    rng = rng or random.Random()
    n_target = int(np.ceil(np.log2(N)))
    if counting_extra is None:
        # Below k=9 use textbook 2k for maximum success per shot; above k=9
        # taper to keep state size in the 8 GB envelope.
        counting_extra = n_target if n_target <= 9 else 4
    n_counting = n_target + counting_extra

    for _ in range(max_attempts):
        a = rng.randrange(2, N)
        g = gcd(a, N)
        if g > 1:
            return [g, N // g]

        if sparse:
            # Sparse path: compute the counting-register marginal P(c)
            # directly without materializing the 2^(t+k) state vector.
            # Memory O(2^t), time O(r · t · 2^t) instead of O(2^(t+k)).
            P = phase_estimation_modexp_marginal(a, N, n_target, n_counting)
            phase_int = int(np.random.default_rng().choice(1 << n_counting, p=P))
        else:
            # Dense fast path: closed-form state init + FFT inverse-QFT.
            qc = phase_estimation_modexp(a, N, n_target, n_counting)
            phase_int = _measure_counting_register(qc, n_counting)
        r = continued_fraction_period(phase_int, n_counting, N)

        if r is None or r % 2 != 0:
            continue
        x = pow(a, r // 2, N)
        if x == N - 1:
            continue
        f1 = gcd(x - 1, N)
        f2 = gcd(x + 1, N)
        for f in (f1, f2):
            if 1 < f < N:
                return sorted([f, N // f])

    return [N]


def _measure_counting_register(qc: QuantumCircuit, n_counting: int) -> int:
    """Sample the counting register, returning the integer outcome.

    Vectorized: reshape probs into (2^n_counting, 2^n_eig) and sum the
    eigenstate-register axis. ~30× faster than the Python-loop version at
    18-qubit total state size.
    """
    probs = qc.probabilities()
    n_eig = qc.n - n_counting
    marginal = probs.reshape(1 << n_counting, 1 << n_eig).sum(axis=1)
    marginal = np.clip(marginal, 0, None)
    marginal /= marginal.sum()
    return int(np.random.default_rng().choice(1 << n_counting, p=marginal))
