"""Quantum Key Distribution: BB84 + E91.

Two protocols that let Alice and Bob establish a shared secret key whose
security is guaranteed by quantum mechanics — not by computational
hardness. Eavesdropping is detectable because measurement disturbs the
quantum state.

**BB84** (Bennett-Brassard 1984) — the first QKD protocol:
    Alice sends Bob random qubits, each prepared in a randomly chosen
    basis (Z or X). Bob measures in random bases. After transmission,
    they publicly compare basis choices and keep bits where they
    matched. Sample a small subset for error-rate estimation; if Eve
    intercepted, those will be wrong ~25% of the time.

**E91** (Ekert 1991) — entanglement-based QKD:
    A source distributes Bell pairs to Alice and Bob. They each measure
    randomly in one of three bases. Correlated outcomes (when their
    bases match) form the shared key. The CHSH inequality on the
    other outcomes provides a built-in eavesdropping test — any
    eavesdropper would reduce the CHSH violation below the Tsirelson
    bound.

Both protocols are end-to-end simulable on this engine. We give the
algorithmic flow and the eavesdropping statistics; we don't simulate
network noise, polarization losses, etc.
"""

from __future__ import annotations

import numpy as np

from ..circuit import QuantumCircuit


# ----------------------------------------------------------------------------
# BB84
# ----------------------------------------------------------------------------

def bb84_protocol(
    n_bits: int,
    eve_probability: float = 0.0,
    rng: np.random.Generator | None = None,
) -> dict:
    """Run BB84 with optional eavesdropping.

    Args:
        n_bits:           number of qubits Alice sends to Bob.
        eve_probability:  probability per qubit that Eve intercepts and
                          re-measures (0 = no eavesdropper).
        rng:              numpy generator.

    Returns:
        dict with:
            alice_bits:       Alice's original random bits.
            alice_bases:      Alice's encoding bases ('Z' or 'X').
            bob_bases:        Bob's measurement bases.
            bob_outcomes:     Bob's measured bits.
            matched_indices:  indices where bases agreed.
            sifted_alice:     Alice's bits at matched indices (shared key candidate).
            sifted_bob:       Bob's bits at the same indices.
            qber:             quantum bit error rate on the sifted key.
            n_sifted:         length of the sifted key.
            eve_intercepts:   how many qubits Eve actually intercepted.
    """
    rng = rng or np.random.default_rng()
    alice_bits  = rng.integers(0, 2, size=n_bits)
    alice_bases = rng.choice(["Z", "X"], size=n_bits)
    bob_bases   = rng.choice(["Z", "X"], size=n_bits)
    bob_outcomes = np.zeros(n_bits, dtype=np.int8)
    eve_intercepts = 0

    for i in range(n_bits):
        # Alice prepares the qubit.
        qc = QuantumCircuit(1)
        if alice_bits[i] == 1:
            qc.x(0)
        if alice_bases[i] == "X":
            qc.h(0)
        # Eve may intercept-and-resend.
        if rng.uniform() < eve_probability:
            eve_intercepts += 1
            eve_basis = rng.choice(["Z", "X"])
            if eve_basis == "X":
                qc.h(0)
            eve_outcome = int(qc.measure_all(shots=1, rng=rng)[0])
            # Eve re-prepares.
            qc = QuantumCircuit(1)
            if eve_outcome == 1:
                qc.x(0)
            if eve_basis == "X":
                qc.h(0)
        # Bob measures.
        if bob_bases[i] == "X":
            qc.h(0)
        bob_outcomes[i] = int(qc.measure_all(shots=1, rng=rng)[0])

    # Basis sifting: keep only bits where Alice's and Bob's bases matched.
    matched = np.where(alice_bases == bob_bases)[0]
    sifted_alice = alice_bits[matched]
    sifted_bob   = bob_outcomes[matched]
    qber = float(np.mean(sifted_alice != sifted_bob)) if len(matched) else 0.0

    return {
        "alice_bits":      alice_bits,
        "alice_bases":     alice_bases,
        "bob_bases":       bob_bases,
        "bob_outcomes":    bob_outcomes,
        "matched_indices": matched,
        "sifted_alice":    sifted_alice,
        "sifted_bob":      sifted_bob,
        "qber":            qber,
        "n_sifted":        int(len(matched)),
        "eve_intercepts":  eve_intercepts,
    }


# ----------------------------------------------------------------------------
# E91 (entanglement-based)
# ----------------------------------------------------------------------------

# Measurement-basis angles. Our rotation convention applies Ry(-2θ), so
# the effective angle is doubled in the correlation function E(α, β) =
# -cos(2(β - α)). To saturate CHSH at 2√2, we want the four difference
# angles 2(β - α) to lie on the {±π/4, ±3π/4} cross.
#
# Choose:
#   Alice angles {0, π/8, π/4}   (so 2α ∈ {0, π/4, π/2})
#   Bob   angles {π/8, π/4, 3π/8} (so 2β ∈ {π/4, π/2, 3π/4})
#
# With these, the "aligned" sifting pair is (alice=1, bob=0) [both π/8].
_ALICE_ANGLES = (0.0,         np.pi / 8, np.pi / 4)
_BOB_ANGLES   = (np.pi / 8,   np.pi / 4, 3 * np.pi / 8)


def _measure_in_rotated_z(qc: QuantumCircuit, qubit: int, angle: float,
                           rng: np.random.Generator) -> int:
    """Apply Ry(-2·angle) then measure in Z. Equivalent to measuring in the
    Z-basis rotated by `angle` in the X-Z plane."""
    qc.ry(-2 * angle, qubit)
    probs = qc.probabilities()
    # Marginal on qubit 0 (for our 2-qubit case).
    n = qc.n
    p0 = sum(probs[i] for i in range(2 ** n) if not ((i >> (n - 1 - qubit)) & 1))
    return 0 if rng.uniform() < p0 else 1


def e91_protocol(
    n_pairs: int,
    rng: np.random.Generator | None = None,
) -> dict:
    """Run E91 (Ekert) entanglement-based QKD.

    Args:
        n_pairs: number of Bell pairs distributed.
        rng:     numpy generator.

    Returns:
        dict with:
            alice_bases, bob_bases:  basis choices (0, 1, 2 indexing _ALICE/BOB_ANGLES)
            alice_outcomes, bob_outcomes
            sifted_alice, sifted_bob: outcomes where (alice_basis, bob_basis) gives
                                       perfectly anti-correlated singlet measurements
                                       (Alice basis 1 = Bob basis 0, both at π/4)
            qber:        error rate on the sifted key
            chsh_value:  CHSH expectation from the "discarded" rounds — should be
                         > 2 (classical bound) for a genuine quantum channel
    """
    rng = rng or np.random.default_rng()
    alice_choices = rng.integers(0, 3, size=n_pairs)
    bob_choices   = rng.integers(0, 3, size=n_pairs)
    alice_outcomes = np.zeros(n_pairs, dtype=np.int8)
    bob_outcomes   = np.zeros(n_pairs, dtype=np.int8)

    for i in range(n_pairs):
        # Prepare the singlet |ψ−⟩ = (|01⟩ - |10⟩) / √2.
        # Circuit: X on q0; H on q0; CNOT(0,1); X on q1.
        #   |00⟩ → |10⟩ → (|0⟩−|1⟩)/√2 ⊗ |0⟩ → (|00⟩−|11⟩)/√2 → (|01⟩−|10⟩)/√2
        # This singlet is invariant under U⊗U rotations, giving the
        # E(α, β) = −cos(2(β−α)) correlation function we want for E91.
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.h(0)
        qc.cnot(0, 1)
        qc.x(1)
        # Now measure Alice in basis alice_choices[i], Bob in bob_choices[i].
        alice_basis = _ALICE_ANGLES[alice_choices[i]]
        bob_basis   = _BOB_ANGLES  [bob_choices[i]]
        # Rotate both qubits and measure (full state measurement).
        qc.ry(-2 * alice_basis, 0)
        qc.ry(-2 * bob_basis,   1)
        outcome = int(qc.measure_all(shots=1, rng=rng)[0])
        alice_outcomes[i] = (outcome >> 1) & 1
        bob_outcomes[i]   = outcome & 1

    # Sifting: Alice basis 1 (π/4) and Bob basis 0 (π/4) — perfectly aligned;
    # outcomes are anti-correlated for the Bell-ψ+ state.
    sift_mask = (alice_choices == 1) & (bob_choices == 0)
    sifted_alice = alice_outcomes[sift_mask]
    # For ψ+ state, alice and bob give anti-correlated outcomes in any aligned basis.
    # Bob takes complement to get matching key.
    sifted_bob   = 1 - bob_outcomes[sift_mask]
    qber = float(np.mean(sifted_alice != sifted_bob)) if len(sifted_alice) else 0.0

    # CHSH on rounds with bases (a0, b0), (a0, b2), (a2, b0), (a2, b2)
    # — the standard four-setting CHSH. E(a, b) = ⟨A_a · B_b⟩.
    def correlation(a_idx, b_idx):
        mask = (alice_choices == a_idx) & (bob_choices == b_idx)
        if mask.sum() < 1:
            return 0.0
        # Anti-correlated for ψ+ → A=B should give -1, A≠B should give +1.
        # Use convention E = -⟨(2A-1)(2B-1)⟩ so that perfect anti-correlation = +1.
        a = 2 * alice_outcomes[mask].astype(np.int32) - 1
        b = 2 * bob_outcomes[mask].astype(np.int32) - 1
        return -float(np.mean(a * b))

    E_a0_b0 = correlation(0, 0)
    E_a0_b2 = correlation(0, 2)
    E_a2_b0 = correlation(2, 0)
    E_a2_b2 = correlation(2, 2)
    chsh = abs(E_a0_b0 - E_a0_b2 + E_a2_b0 + E_a2_b2)

    return {
        "alice_choices":  alice_choices,
        "bob_choices":    bob_choices,
        "alice_outcomes": alice_outcomes,
        "bob_outcomes":   bob_outcomes,
        "sifted_alice":   sifted_alice,
        "sifted_bob":     sifted_bob,
        "qber":           qber,
        "chsh_value":     chsh,
        "n_sifted":       int(sift_mask.sum()),
    }


# ----------------------------------------------------------------------------
# Wiesner's quantum money (1970, published 1983)
# ----------------------------------------------------------------------------

def wiesner_quantum_money(
    n_qubits: int,
    rng: np.random.Generator | None = None,
) -> dict:
    """Demonstrate Wiesner's quantum-money no-cloning protocol.

    The bank issues an n-qubit "banknote": each qubit is prepared randomly
    in one of {|0⟩, |1⟩, |+⟩, |−⟩}. The serial number records the basis +
    bit for each qubit; the bank can verify a banknote by measuring each
    qubit in its known basis and checking the value matches.

    A counterfeiter who doesn't know the bases must guess; each guess
    succeeds with probability 1/2, and even if it does, they only learn
    the qubit's value in that basis. They cannot reliably clone the
    banknote because the no-cloning theorem forbids it.

    Demonstrates: the bank's verification success rate is 100% on
    legitimate bills, and the counterfeiter's success rate is at most
    (3/4)^n per attempted clone.
    """
    rng = rng or np.random.default_rng()
    bases = rng.choice(["Z", "X"], size=n_qubits)
    bits  = rng.integers(0, 2, size=n_qubits)

    def prepare_banknote() -> QuantumCircuit:
        qc = QuantumCircuit(n_qubits)
        for q in range(n_qubits):
            if bits[q] == 1:
                qc.x(q)
            if bases[q] == "X":
                qc.h(q)
        return qc

    def verify(banknote: QuantumCircuit) -> bool:
        qc = banknote
        for q in range(n_qubits):
            if bases[q] == "X":
                qc.h(q)
        outcomes = qc.measure_all(shots=1, rng=rng)[0]
        for q in range(n_qubits):
            outcome_bit = (int(outcomes) >> (n_qubits - 1 - q)) & 1
            if outcome_bit != bits[q]:
                return False
        return True

    def attack_no_basis_knowledge() -> bool:
        """Counterfeiter measures each qubit in a random basis (the only
        thing they can do without knowing the encoding), tries to
        re-prepare. Returns whether the forged banknote passes verification.
        """
        legitimate = prepare_banknote()
        # Counterfeiter measures in random bases.
        forged = QuantumCircuit(n_qubits)
        forged.state = legitimate.state.copy()
        attacker_bases = rng.choice(["Z", "X"], size=n_qubits)
        attacker_outcomes = np.zeros(n_qubits, dtype=np.int8)
        for q in range(n_qubits):
            if attacker_bases[q] == "X":
                forged.h(q)
        full_outcome = int(forged.measure_all(shots=1, rng=rng)[0])
        for q in range(n_qubits):
            attacker_outcomes[q] = (full_outcome >> (n_qubits - 1 - q)) & 1
        # Re-prepare based on what they measured.
        re_prepared = QuantumCircuit(n_qubits)
        for q in range(n_qubits):
            if attacker_outcomes[q] == 1:
                re_prepared.x(q)
            if attacker_bases[q] == "X":
                re_prepared.h(q)
        return verify(re_prepared)

    # Run both tests.
    legitimate_verify = verify(prepare_banknote())
    attack_success    = attack_no_basis_knowledge()

    return {
        "bases":               bases,
        "bits":                bits,
        "legitimate_passes":   legitimate_verify,
        "attack_passes":       attack_success,
        "expected_attack_rate": (3 / 4) ** n_qubits,
    }
