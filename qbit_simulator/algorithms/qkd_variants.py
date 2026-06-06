"""B92 and SARG04: alternative QKD protocols.

We already have BB84 (4 states, 2 bases) and E91 (entanglement-based).
This module adds two more important variants:

  * **B92** (Bennett 1992): uses only TWO non-orthogonal states. Simpler
    encoding (no basis announcement) but lower key rate.
        |0⟩       represents classical bit 0
        |+⟩       represents classical bit 1
    Bob measures in {Z, X}; only conclusive (unambiguous-discrimination)
    outcomes are kept.

  * **SARG04** (Scarani-Acín-Ribordy-Gisin 2004): same FOUR states as
    BB84 but a different sifting protocol. Robust against the photon-
    number-splitting (PNS) attack on weak coherent pulses.

Both protocols are simulated with the same trajectory style as our
existing `qkd.py`: Alice prepares states, Bob measures, classical
post-processing extracts a shared key.

Provides:

  - `b92_run(n_bits, eve_attack, rng)`: simulate B92 with optional
    eavesdropping; returns the sifted key, error rate, and detection
    statistics.
  - `sarg04_run(n_bits, eve_attack, rng)`: same for SARG04.
  - `b92_security_threshold()`: QBER bound above which B92 is insecure.
"""

from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------
# B92 protocol
# ----------------------------------------------------------------------------
#
# State preparation:
#   bit 0 → |0⟩
#   bit 1 → |+⟩ = (|0⟩ + |1⟩) / √2
#
# Bob's measurement:
#   Randomly chooses Z or X basis.
#   If he measures Z and gets |1⟩:  this can ONLY come from |+⟩ → bit 1.
#   If he measures X and gets |−⟩:  this can ONLY come from |0⟩ → bit 0.
#   Other outcomes are inconclusive; the bit is discarded.
#
# Eve attack: intercept-resend in random basis (basic IR attack).

def _measure_z(state_label: str, rng: np.random.Generator) -> int:
    """Measure the qubit in the Z basis, return 0 or 1."""
    if state_label == "0":
        return 0
    if state_label == "1":
        return 1
    if state_label in ("+", "-"):
        return int(rng.uniform() < 0.5)
    raise ValueError(state_label)


def _measure_x(state_label: str, rng: np.random.Generator) -> int:
    """Measure in X basis. Returns 0 (|+⟩) or 1 (|-⟩)."""
    if state_label == "+":
        return 0
    if state_label == "-":
        return 1
    if state_label in ("0", "1"):
        return int(rng.uniform() < 0.5)
    raise ValueError(state_label)


def b92_run(n_bits: int = 100,
             eve_attack: bool = False,
             rng: np.random.Generator | None = None) -> dict:
    """Run B92 once and report the sifted key.

    Args:
        n_bits:     number of qubits Alice sends.
        eve_attack: if True, Eve intercepts each qubit and resends in a
                    random basis (Z or X) — the classic IR attack.
        rng:        numpy generator (for reproducibility).

    Returns:
        dict with alice_key, bob_key, qber, sift_rate, eve_attack.
    """
    rng = rng or np.random.default_rng()
    alice_bits = []
    bob_bits = []
    eve_intercepted = 0
    for _ in range(n_bits):
        # Alice picks a random classical bit.
        a_bit = int(rng.uniform() < 0.5)
        a_state = "0" if a_bit == 0 else "+"

        # Eve (optional).
        if eve_attack:
            # Eve measures in random basis, gets some outcome, resends the
            # corresponding eigenstate.
            eve_basis = rng.choice(["Z", "X"])
            if eve_basis == "Z":
                e_meas = _measure_z(a_state, rng)
                a_state_after = "0" if e_meas == 0 else "1"
            else:
                e_meas = _measure_x(a_state, rng)
                a_state_after = "+" if e_meas == 0 else "-"
            eve_intercepted += 1
        else:
            a_state_after = a_state

        # Bob picks a random measurement basis.
        b_basis = rng.choice(["Z", "X"])

        # Bob's measurement outcome:
        b_outcome = _measure_b92_state(a_state_after, b_basis, rng)
        if b_outcome is None:
            continue   # Bob's measurement was inconclusive — discard.

        # Bob accepts the bit:
        #   - Z measurement giving |1⟩ ⇒ Bob infers bit = 1
        #   - X measurement giving |−⟩ ⇒ Bob infers bit = 0
        if b_basis == "Z" and b_outcome == 1:
            bob_bits.append(1)
            alice_bits.append(a_bit)
        elif b_basis == "X" and b_outcome == 1:   # |−⟩ outcome
            bob_bits.append(0)
            alice_bits.append(a_bit)

    if alice_bits:
        errors = sum(a != b for a, b in zip(alice_bits, bob_bits))
        qber = errors / len(alice_bits)
    else:
        qber = 0.0
    return {
        "alice_key":  alice_bits,
        "bob_key":    bob_bits,
        "n_sifted":   len(alice_bits),
        "qber":       qber,
        "sift_rate":  len(alice_bits) / n_bits if n_bits else 0.0,
        "eve_attack": eve_attack,
    }


def _measure_b92_state(state_label: str, basis: str,
                        rng: np.random.Generator) -> int | None:
    """Bob's measurement: returns 0/1 outcome or None if 'inconclusive'.

    In B92, the only conclusive outcomes are:
      - Z=1 (occurs only from |+⟩)
      - X=1 (i.e., |−⟩; occurs only from |0⟩)
    Otherwise we mark "inconclusive" (50% probability for the typical
    states), since both alice's bits could have produced it.
    """
    if basis == "Z":
        if state_label == "0":
            out = 0
        elif state_label == "1":
            out = 1
        elif state_label == "+":
            out = int(rng.uniform() < 0.5)
        elif state_label == "-":
            out = int(rng.uniform() < 0.5)
        # Only |1⟩ outcome is conclusive (rules out alice = 0).
        return out if out == 1 else None
    elif basis == "X":
        if state_label == "0":
            out = int(rng.uniform() < 0.5)
        elif state_label == "1":
            out = int(rng.uniform() < 0.5)
        elif state_label == "+":
            out = 0
        elif state_label == "-":
            out = 1
        # Only |−⟩ outcome (=1) is conclusive.
        return out if out == 1 else None
    else:
        raise ValueError(basis)


def b92_security_threshold() -> float:
    """The QBER threshold for B92 — above this, an eavesdropper could have
    extracted the key. Standard result: ~7%."""
    return 0.07


# ----------------------------------------------------------------------------
# SARG04 protocol
# ----------------------------------------------------------------------------
#
# Same 4 states as BB84:  |0⟩, |1⟩, |+⟩, |−⟩
#
# Encoding: same as BB84.
# Difference is in the SIFTING:
#   After Bob measures, Alice publicly announces a PAIR of two non-
#   orthogonal states (one from Z-basis, one from X-basis) that her
#   state belongs to. Bob accepts if his measurement RULES OUT one
#   of the pair, and infers Alice's state from the other.
#
# Why this matters: it shifts the burden onto Eve. In a PNS attack,
# Eve splits multi-photon pulses; BB84 leaks the basis after sifting,
# so Eve can measure in the right basis. SARG04 doesn't leak the basis
# directly, so PNS is harder.

def sarg04_run(n_bits: int = 100,
                eve_attack: bool = False,
                rng: np.random.Generator | None = None) -> dict:
    """Run SARG04 once."""
    rng = rng or np.random.default_rng()
    states_list = ["0", "1", "+", "-"]
    alice_bits = []
    bob_bits = []
    for _ in range(n_bits):
        a_state = rng.choice(states_list)
        # Map state → bit (per Alice's convention).
        a_bit = 0 if a_state in ("0", "+") else 1

        if eve_attack:
            eve_basis = rng.choice(["Z", "X"])
            if eve_basis == "Z":
                e_meas = _measure_z(a_state, rng) if a_state in ("0", "+") \
                          else (1 if a_state == "1" else int(rng.uniform() < 0.5))
                a_state = "0" if e_meas == 0 else "1"
            else:
                e_meas = _measure_x(a_state, rng) if a_state in ("0", "+") \
                          else (1 if a_state == "-" else int(rng.uniform() < 0.5))
                a_state = "+" if e_meas == 0 else "-"

        b_basis = rng.choice(["Z", "X"])
        b_outcome = _measure_z(a_state, rng) if b_basis == "Z" \
                    else _measure_x(a_state, rng)

        # Sifting: Alice announces a pair of non-orthogonal states
        # containing her state. SARG04 chooses the PARTNER from the other
        # basis encoding the OPPOSITE bit — this is what makes ruling
        # out the partner equivalent to determining the bit.
        partners = {
            "0": "-",   # bits 0 vs 1
            "1": "+",   # bits 1 vs 0
            "+": "1",   # bits 0 vs 1
            "-": "0",
        }
        partner_state = partners[a_state]

        # Bob decides which of the pair is consistent with his outcome.
        # In B basis B and outcome m:
        #   If state == a_state, then m is what we computed.
        #   The "rules out" criterion: Bob's outcome must be impossible
        #   from partner_state.
        # For the four orthogonal pairs (0/-), (+/1) etc., the rule is:
        partner_basis = "X" if partner_state in ("+", "-") else "Z"
        if b_basis == partner_basis:
            # Bob can measure partner_state deterministically.
            # If his outcome corresponds to the OPPOSITE eigenstate of
            # partner_state, then partner_state is ruled out → Bob infers
            # a_state.
            partner_bit = 0 if partner_state in ("0", "+") else 1
            if b_outcome != partner_bit:
                # partner_state ruled out — Bob accepts a_state.
                bob_bits.append(a_bit)
                alice_bits.append(a_bit)
            # else: outcome consistent with both → inconclusive, discard.
        # else: Bob measured in same basis as alice → 100% correlated but
        # cannot rule out partner; discard.

    if alice_bits:
        errors = sum(a != b for a, b in zip(alice_bits, bob_bits))
        qber = errors / len(alice_bits)
    else:
        qber = 0.0
    return {
        "alice_key":  alice_bits,
        "bob_key":    bob_bits,
        "n_sifted":   len(alice_bits),
        "qber":       qber,
        "sift_rate":  len(alice_bits) / n_bits if n_bits else 0.0,
        "eve_attack": eve_attack,
    }
