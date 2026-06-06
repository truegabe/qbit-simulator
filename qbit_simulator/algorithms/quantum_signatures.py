"""Gottesman-Chuang quantum digital signatures.

A digital signature lets a Signer produce a SIGNED MESSAGE that any
Receiver can verify came from the Signer and was not tampered with.
Classically, this requires public-key cryptography (RSA, ECDSA) whose
security relies on computational hardness assumptions (factoring,
discrete log) — both of which are broken by Shor's algorithm.

Gottesman-Chuang (2001) introduce a QUANTUM signature scheme with
information-theoretic security:

  * Setup: Signer picks a secret bit b ∈ {0, 1} and a "public key" —
    M copies of |f_b⟩, distributed to all potential receivers, where
    {|f_0⟩, |f_1⟩} are non-orthogonal quantum states (e.g. |0⟩ and |+⟩).
  * Sign: To sign bit b, Signer sends the SECRET KEY (a description of
    |f_b⟩, e.g. b itself).
  * Verify: Receiver measures their copies of |f_b⟩ in the basis that
    deterministically distinguishes |f_b⟩ from |f_{1-b}⟩. If too many
    mismatches occur, reject.

Security: a forger must produce a key that matches the public state.
Producing a single forged copy succeeds with probability < 1 (related
to the overlap ⟨f_0|f_1⟩). Producing M consistent forgeries succeeds
with probability ≤ ⟨f_0|f_1⟩^M, exponentially small in M.

This module implements:

  - `gc_setup(M, rng)`: produce a public/private key pair.
  - `gc_sign(message, private_key)`: produce a signature.
  - `gc_verify(message, signature, public_key)`: verify.
  - `gc_forge_attempt(message, public_key, rng)`: simulate a forgery
    attempt and report success probability.

For simplicity we use single-qubit states {|0⟩, |+⟩} with overlap
|⟨0|+⟩|² = 1/2. The protocol generalizes to longer messages via
per-bit signing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Public commit states.
_STATE_0 = np.array([1, 0], dtype=np.complex128)
_STATE_1 = np.array([1, 1], dtype=np.complex128) / np.sqrt(2)


# ----------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------

@dataclass
class GCKeyPair:
    """Gottesman-Chuang key pair.

    Attributes:
        secret_bit:    the Signer's choice of b ∈ {0, 1}.
        public_copies: M copies of |f_b⟩, distributed to receivers.
    """
    secret_bit: int
    public_copies: list[np.ndarray]
    M: int


def gc_setup(M: int = 50, secret_bit: int | None = None,
              rng: np.random.Generator | None = None) -> GCKeyPair:
    """Generate a fresh Gottesman-Chuang key pair.

    Args:
        M:           number of public copies (security parameter).
        secret_bit:  Signer's choice (random if None).
        rng:         generator (for default secret_bit).

    Returns:
        a GCKeyPair.
    """
    rng = rng or np.random.default_rng()
    if secret_bit is None:
        secret_bit = int(rng.uniform() < 0.5)
    if secret_bit not in (0, 1):
        raise ValueError("secret_bit must be 0 or 1")
    state = _STATE_0 if secret_bit == 0 else _STATE_1
    return GCKeyPair(
        secret_bit=secret_bit,
        public_copies=[state.copy() for _ in range(M)],
        M=M,
    )


# ----------------------------------------------------------------------------
# Signing
# ----------------------------------------------------------------------------

def gc_sign(claimed_bit: int, keypair: GCKeyPair) -> int:
    """Signer produces a signature: just reveals the bit value.

    (For a real Gottesman-Chuang scheme, the signature is the
    classical DESCRIPTION of the public state, here represented
    by the bit value itself.)

    Returns:
        the claimed bit (which a verifier will check against the
        distributed public copies).
    """
    return claimed_bit


# ----------------------------------------------------------------------------
# Verification
# ----------------------------------------------------------------------------

def _measure_in_basis(psi: np.ndarray, basis: str,
                       rng: np.random.Generator) -> int:
    """Project psi in Z or X basis, return outcome 0 or 1."""
    if basis == "Z":
        p0 = abs(psi[0]) ** 2
    elif basis == "X":
        p0 = abs((psi[0] + psi[1]) / np.sqrt(2)) ** 2
    else:
        raise ValueError(basis)
    return 0 if rng.uniform() < p0 else 1


def gc_verify(claimed_bit: int, public_copies: list[np.ndarray],
                threshold: float = 0.1, rng: np.random.Generator | None = None
                ) -> dict:
    """Verify a signature against the distributed public copies.

    Procedure:
      - For each public copy, the receiver measures in the basis that
        deterministically identifies |f_{claimed_bit}⟩.
          * claimed = 0 → measure Z (|0⟩ gives 0 deterministically).
          * claimed = 1 → measure X (|+⟩ gives X=0 deterministically).
      - Count "mismatches" — outcomes that should be impossible for
        |f_{claimed_bit}⟩.
      - Accept if mismatches/M < threshold.

    Args:
        claimed_bit:    the signature value to verify.
        public_copies:  M copies of the public state (distributed by
                        the Signer at setup time).
        threshold:      reject if mismatch rate exceeds this fraction.
        rng:            generator.

    Returns:
        dict with "accept", "mismatch_rate", "n_mismatches".
    """
    rng = rng or np.random.default_rng()
    basis = "Z" if claimed_bit == 0 else "X"
    n_mismatch = 0
    for psi in public_copies:
        out = _measure_in_basis(psi, basis, rng)
        if out != 0:
            n_mismatch += 1
    rate = n_mismatch / len(public_copies)
    return {
        "accept":         rate <= threshold,
        "mismatch_rate":  rate,
        "n_mismatches":   n_mismatch,
        "M":              len(public_copies),
    }


# ----------------------------------------------------------------------------
# Forgery attempt
# ----------------------------------------------------------------------------

def gc_forge_attempt(claimed_bit: int, true_bit: int, M: int = 50,
                       threshold: float = 0.1,
                       rng: np.random.Generator | None = None) -> dict:
    """Simulate a forgery: a forger claims `claimed_bit` but the public
    copies are actually |f_{true_bit}⟩.

    The forger SUCCEEDS if verification accepts the wrong claim.

    For Gottesman-Chuang with states {|0⟩, |+⟩}: a single mismatched
    measurement gives outcome 1 with prob 1/2. The forger needs to keep
    the mismatch rate below threshold.

    Returns:
        dict with verification outcome.
    """
    rng = rng or np.random.default_rng()
    state = _STATE_0 if true_bit == 0 else _STATE_1
    public_copies = [state.copy() for _ in range(M)]
    return gc_verify(claimed_bit, public_copies, threshold=threshold, rng=rng)


def gc_forge_failure_probability(M: int = 50, threshold: float = 0.1
                                    ) -> float:
    """Probability that a forger's wrong claim is rejected.

    Each measurement on a wrong-state |f_{1-claimed}⟩ gives a mismatch
    with prob 1/2 (= 1 - |⟨f_0|f_1⟩|²). For verification to ACCEPT, the
    forger needs < threshold·M mismatches in M trials. Binomial tail.
    """
    from math import comb
    threshold_count = int(threshold * M)
    p_mismatch = 0.5
    # Probability of NUMBER of mismatches ≤ threshold_count.
    p_accept = sum(comb(M, k) * p_mismatch ** k * (1 - p_mismatch) ** (M - k)
                    for k in range(threshold_count + 1))
    return 1.0 - p_accept   # rejection probability


# ----------------------------------------------------------------------------
# Honesty check
# ----------------------------------------------------------------------------

def gc_correctness_demo(M: int = 50, threshold: float = 0.1,
                          rng: np.random.Generator | None = None) -> dict:
    """End-to-end honest signing + verification.

    Returns dict with success indicators.
    """
    rng = rng or np.random.default_rng()
    keypair = gc_setup(M=M, secret_bit=0, rng=rng)
    sig = gc_sign(0, keypair)
    return gc_verify(sig, keypair.public_copies, threshold=threshold, rng=rng)
