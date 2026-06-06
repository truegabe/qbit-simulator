"""Stabilizer simulator tests.

Three classes of test:
  - Generator correctness: after each Clifford gate, the stabilizer list
    matches what's expected analytically.
  - Equivalence to dense: small-N states reconstructed via to_dense match
    the QuantumCircuit dense path bit-for-bit.
  - Scale: 1000-qubit GHZ in seconds, correlated measurements at scale.
"""

import time

import numpy as np
import pytest

from qbit_simulator import QuantumCircuit
from qbit_simulator.stabilizer import StabilizerState


# ---- initial state ----

def test_initial_stabilizers_are_z_operators():
    st = StabilizerState(4)
    expected = ["+IIIZ", "+IIZI", "+IZII", "+ZIII"]
    # Stabilizers are listed in order: row N -> stab 0 (Z_0), row N+1 -> Z_1, ...
    # Our convention: stab i = Z_i which is "I...Z...I" with Z at position i.
    got = st.stabilizers()
    assert got == [f"+{'I'*i}Z{'I'*(4-1-i)}" for i in range(4)]


def test_initial_destabilizers_are_x_operators():
    st = StabilizerState(4)
    got = st.destabilizers()
    assert got == [f"+{'I'*i}X{'I'*(4-1-i)}" for i in range(4)]


# ---- single-qubit gate updates ----

def test_h_swaps_x_and_z():
    st = StabilizerState(2).h(0)
    # After H(0): stab 0 was Z_0 -> X_0, stab 1 stays Z_1.
    # Destab 0 was X_0 -> Z_0, destab 1 stays X_1.
    assert "+XI" in st.stabilizers()
    assert "+IZ" in st.stabilizers()
    assert "+ZI" in st.destabilizers()
    assert "+IX" in st.destabilizers()


def test_x_flips_z_signs():
    st = StabilizerState(2).x(0)
    # X(0) anticommutes with Z_0 -> -Z_0. Z_1 unchanged.
    assert "-ZI" in st.stabilizers()
    assert "+IZ" in st.stabilizers()


def test_z_flips_x_signs():
    st = StabilizerState(2).h(0).z(0)
    # After H, stab 0 = X_0. Then Z(0) anticommutes with X_0 -> -X_0.
    assert "-XI" in st.stabilizers()


# ---- two-qubit gate updates ----

def test_cnot_produces_bell_stabilizers():
    """Bell pair: H(0), CNOT(0,1). Stabilizers should be {X_0 X_1, Z_0 Z_1}."""
    st = StabilizerState(2).h(0).cnot(0, 1)
    stabs = sorted(st.stabilizers())
    assert stabs == ["+XX", "+ZZ"]


def test_ghz_stabilizers():
    """GHZ on N=4: H(0) + CNOT cascade. Stabilizers: X_0...X_{N-1}, Z_i Z_{i+1}."""
    n = 4
    st = StabilizerState(n).h(0)
    for q in range(n - 1):
        st.cnot(q, q + 1)
    stabs = set(st.stabilizers())
    # X^N is one stabilizer.
    assert ("+" + "X" * n) in stabs
    # The Z_i Z_{i+1} family.
    for i in range(n - 1):
        word = ["I"] * n
        word[i] = "Z"; word[i + 1] = "Z"
        assert ("+" + "".join(word)) in stabs


# ---- equivalence with the dense state-vector path ----

@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_random_clifford_circuit_matches_dense(seed):
    """Apply a random sequence of Clifford gates; reconstructed dense state
    must equal the QuantumCircuit dense state (up to global phase)."""
    rng = np.random.default_rng(seed)
    n = 5
    st = StabilizerState(n)
    qc = QuantumCircuit(n)
    for _ in range(30):
        kind = rng.integers(0, 4)
        if kind == 0:
            q = int(rng.integers(0, n))
            st.h(q); qc.h(q)
        elif kind == 1:
            q = int(rng.integers(0, n))
            st.s(q); qc.s(q)
        elif kind == 2:
            q = int(rng.integers(0, n))
            st.x(q); qc.x(q)
        else:
            a = int(rng.integers(0, n))
            b = int(rng.integers(0, n))
            if a == b:
                continue
            st.cnot(a, b); qc.cnot(a, b)
    psi_stab  = st.to_dense()
    psi_dense = qc.state
    inner = np.vdot(psi_stab, psi_dense)
    assert abs(abs(inner) - 1.0) < 1e-9


# ---- measurement correctness ----

def test_deterministic_measurement_of_zero_state():
    """|0...0>: measuring any qubit must give 0 deterministically."""
    st = StabilizerState(5)
    for q in range(5):
        assert st.measure(q) == 0


def test_deterministic_measurement_after_x():
    """X(2)|0...0> = |00100>: measuring qubit 2 must give 1, others 0."""
    st = StabilizerState(5).x(2)
    for q in range(5):
        assert st.measure(q) == (1 if q == 2 else 0)


def test_bell_pair_measurement_correlation():
    """In a Bell pair, measuring qubit 0 then qubit 1 always gives equal outcomes."""
    rng = np.random.default_rng(0)
    for trial in range(100):
        st = StabilizerState(2).h(0).cnot(0, 1)
        m0 = st.measure(0, rng=rng)
        m1 = st.measure(1, rng=rng)
        assert m0 == m1


def test_ghz_measurement_all_correlated():
    """GHZ on N=8: all measurement outcomes must agree."""
    rng = np.random.default_rng(0)
    for trial in range(20):
        n = 8
        st = StabilizerState(n).h(0)
        for q in range(n - 1):
            st.cnot(q, q + 1)
        outcomes = [st.measure(q, rng=rng) for q in range(n)]
        assert len(set(outcomes)) == 1, f"GHZ measurements differ: {outcomes}"


# ---- scale ----

def test_thousand_qubit_ghz_builds_fast():
    """The whole point: any-N GHZ in polynomial time."""
    n = 1000
    t0 = time.perf_counter()
    st = StabilizerState(n).h(0)
    for q in range(n - 1):
        st.cnot(q, q + 1)
    dt = time.perf_counter() - t0
    # Should be well under 10 seconds at N=1000 -- much faster than that
    # in practice; we leave wiggle room for CI variance.
    assert dt < 10.0
    # Tableau is (2N+1)^2 = ~4 million bytes at int8 -- comfortable.
    assert st.tableau_storage_bytes() < 10 * 1024 * 1024


def test_thousand_qubit_ghz_measurements_correlated():
    """At N=1000, sample-correlate the first 30 qubits of a GHZ."""
    rng = np.random.default_rng(0)
    n = 1000
    st = StabilizerState(n).h(0)
    for q in range(n - 1):
        st.cnot(q, q + 1)
    outcomes = [st.measure(q, rng=rng) for q in range(30)]
    assert len(set(outcomes)) == 1


# ---- repetition code (the canonical Clifford error-correction toy) ----

@pytest.mark.skipif(
    not __import__("os").environ.get("RUN_BIG_RAM_TESTS"),
    reason="Allocates ~7.5 GB. Set RUN_BIG_RAM_TESTS=1 to enable.",
)
def test_stabilizer_at_45000_qubits():
    """Stress test: 45,000-qubit Clifford circuit using ~7.5 GB tableau.
    Demonstrates the polynomial-scaling regime."""
    import time
    rng = np.random.default_rng(0)
    N = 45_000

    t0 = time.perf_counter()
    st = StabilizerState(N)
    # Spread some Hadamards.
    for q in range(0, N, max(1, N // 500)):
        st.h(q)
    # GHZ cascade on the leftmost 2000 qubits.
    for q in range(2000):
        st.cnot(q, q + 1)
    build_time = time.perf_counter() - t0

    # Sanity: tableau allocated, no crash.
    assert st.tableau_storage_bytes() > 7 * 1024 ** 3
    # Untouched far qubit should be deterministic 0.
    assert st.measure(N - 100) == 0
    # Qubits inside the GHZ cascade should all measure the same value.
    outs = [st.measure(q, rng=rng) for q in range(5)]
    assert len(set(outs)) == 1, f"GHZ cascade samples disagree: {outs}"

    # The build at N=45,000 should complete in well under 2 minutes on
    # a modern laptop.
    assert build_time < 120.0


def test_three_qubit_repetition_code_detects_bit_flip():
    """Encode |0_L> = |000>. Inject an X error on qubit 1. The two Z
    syndromes (Z_0 Z_1 and Z_1 Z_2) should both fire."""
    rng = np.random.default_rng(0)
    # |0_L> is just |000> -- the stabilizers (Z_0 Z_1, Z_1 Z_2) already hold.
    st = StabilizerState(5)        # 3 data + 2 syndrome qubits
    # Inject error.
    st.x(1)
    # Measure Z_0 Z_1 into syndrome qubit 3.
    st.cnot(0, 3); st.cnot(1, 3)
    s0 = st.measure(3, rng=rng)
    # Measure Z_1 Z_2 into syndrome qubit 4.
    st.cnot(1, 4); st.cnot(2, 4)
    s1 = st.measure(4, rng=rng)
    # Both syndromes should detect the error on qubit 1.
    assert s0 == 1 and s1 == 1
