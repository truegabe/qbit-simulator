"""Matrix Product State tests — verify equivalence to the dense path
on small N, plus the 100-qubit GHZ that's only feasible via MPS."""

import numpy as np
import pytest

from qbit_simulator import MPSState, QuantumCircuit, mps_overlap


# ---- basic state preparation ----

def test_initial_state_is_all_zeros():
    mps = MPSState(5)
    psi = mps.to_dense()
    assert psi.shape == (32,)
    assert psi[0] == 1.0
    assert np.allclose(psi[1:], 0)
    assert mps.norm() == pytest.approx(1.0, abs=1e-10)


def test_bond_dimensions_start_at_one():
    mps = MPSState(8)
    assert mps.bond_dimensions() == [1] * 7


# ---- 1-qubit gates ----

def test_hadamard_on_first_qubit():
    mps = MPSState(3)
    mps.h(0)
    psi = mps.to_dense()
    # Should be (|000> + |100>) / √2.
    expected = np.zeros(8, dtype=np.complex128)
    expected[0] = 1 / np.sqrt(2)
    expected[4] = 1 / np.sqrt(2)
    assert np.allclose(psi, expected, atol=1e-10)


def test_x_flips_qubit():
    mps = MPSState(3)
    mps.x(1)
    psi = mps.to_dense()
    # |010> in our (qubit-0-is-MSB) convention.
    expected = np.zeros(8, dtype=np.complex128)
    expected[2] = 1.0    # binary 010 = 2
    assert np.allclose(psi, expected)


# ---- 2-qubit gates: Bell pair, GHZ, equivalence to QuantumCircuit ----

def test_bell_pair_matches_dense():
    mps = MPSState(2)
    mps.h(0).cnot(0, 1)
    psi_mps = mps.to_dense()

    qc = QuantumCircuit(2).h(0).cnot(0, 1)
    psi_dense = qc.state
    assert np.allclose(psi_mps, psi_dense, atol=1e-10)


def test_ghz_state_chi_stays_at_two():
    """N-qubit GHZ = (|0...0> + |1...1>) / √2 has bond dim 2 for any N."""
    n = 8
    mps = MPSState(n)
    mps.h(0)
    for q in range(n - 1):
        mps.cnot(q, q + 1)
    # All bonds should be 2.
    assert all(d <= 2 for d in mps.bond_dimensions())
    # Match dense.
    psi_mps = mps.to_dense()
    expected = np.zeros(2**n, dtype=np.complex128)
    expected[0] = 1 / np.sqrt(2)
    expected[-1] = 1 / np.sqrt(2)
    assert np.allclose(psi_mps, expected, atol=1e-10)


def test_ghz_100_qubits_fits_in_kilobytes():
    """The whole point: 100-qubit state in MPS form is tiny."""
    n = 100
    mps = MPSState(n)
    mps.h(0)
    for q in range(n - 1):
        mps.cnot(q, q + 1)
    assert all(d <= 2 for d in mps.bond_dimensions())
    # Storage should be well under 1 MB. Dense would be 2^100 * 16 bytes.
    assert mps.storage_bytes() < 1_000_000
    # Norm preserved.
    assert mps.norm() == pytest.approx(1.0, abs=1e-9)


# ---- nontrivial gates: CP, CZ, SWAP, long-range CNOT ----

def test_cnot_non_adjacent_matches_dense():
    n = 4
    mps = MPSState(n)
    mps.h(0)
    mps.cnot(0, 3)           # span = 3
    qc = QuantumCircuit(n).h(0).cnot(0, 3)
    assert np.allclose(mps.to_dense(), qc.state, atol=1e-10)


def test_cp_gate_matches_dense():
    n = 3
    phi = 0.3
    mps = MPSState(n)
    mps.h(0).h(2).cp(phi, 0, 2)
    qc = QuantumCircuit(n).h(0).h(2).cp(phi, 0, 2)
    assert np.allclose(mps.to_dense(), qc.state, atol=1e-10)


def test_swap_matches_dense():
    n = 4
    mps = MPSState(n).h(0).x(2).swap(0, 2)
    qc = QuantumCircuit(n).h(0).x(2).swap(0, 2)
    assert np.allclose(mps.to_dense(), qc.state, atol=1e-10)


# ---- random low-depth circuit equivalence ----

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_low_depth_circuit_matches_dense(seed):
    """Apply a random sequence of 1q + nearest-neighbor 2q gates and verify
    the MPS state matches the dense state-vector path bit-for-bit."""
    rng = np.random.default_rng(seed)
    n = 5
    mps = MPSState(n, max_chi=32)
    qc = QuantumCircuit(n)
    for _ in range(20):
        kind = rng.integers(0, 4)
        if kind == 0:
            q = int(rng.integers(0, n))
            mps.h(q); qc.h(q)
        elif kind == 1:
            q = int(rng.integers(0, n))
            mps.x(q); qc.x(q)
        elif kind == 2:
            q = int(rng.integers(0, n - 1))
            mps.cnot(q, q + 1); qc.cnot(q, q + 1)
        else:
            q = int(rng.integers(0, n))
            theta = float(rng.uniform(0, 2 * np.pi))
            mps.ry(theta, q); qc.ry(theta, q)
    assert np.allclose(mps.to_dense(), qc.state, atol=1e-9)


# ---- from_dense / round-trip ----

def test_from_dense_round_trip_simple():
    qc = QuantumCircuit(3).h(0).cnot(0, 1).cnot(1, 2)
    mps = MPSState.from_dense(qc.state)
    assert np.allclose(mps.to_dense(), qc.state, atol=1e-9)


# ---- dynamic qubit growth ----

def test_add_qubit_at_end():
    mps = MPSState(3).h(0)
    mps.add_qubit()
    assert mps.n == 4
    # State should be (|0> + |1>) ⊗ |0> ⊗ |0> ⊗ |0> = (|0000> + |1000>)/√2.
    psi = mps.to_dense()
    expected = np.zeros(16, dtype=np.complex128)
    expected[0]  = 1 / np.sqrt(2)
    expected[8]  = 1 / np.sqrt(2)
    assert np.allclose(psi, expected, atol=1e-10)


def test_add_qubit_then_entangle():
    mps = MPSState(2).h(0).cnot(0, 1)     # |00> + |11> Bell pair
    mps.add_qubit(at=2, in_state=0)
    mps.cnot(1, 2)                         # extend Bell to GHZ-3
    psi = mps.to_dense()
    expected = np.zeros(8, dtype=np.complex128)
    expected[0]  = 1 / np.sqrt(2)
    expected[7]  = 1 / np.sqrt(2)
    assert np.allclose(psi, expected, atol=1e-10)


# ---- overlap ----

def test_mps_overlap_orthogonal_states():
    a = MPSState(3)                # |000>
    b = MPSState(3).x(0)           # |100>
    assert abs(mps_overlap(a, b)) < 1e-10
    assert abs(mps_overlap(a, a) - 1.0) < 1e-10
