"""Tests for the six review fixes."""

import numpy as np
import pytest

from qbit_simulator import Qubit, QuantumCircuit
from qbit_simulator.measure import sample
from qbit_simulator.viz import circuit_ascii


# ---- #1: apply_unitary rejects non-unitary matrices ----

def test_apply_unitary_rejects_non_unitary():
    qc = QuantumCircuit(2)
    bad = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 1],   # not unitary
                    [0, 0, 0, 1]], dtype=np.complex128)
    with pytest.raises(ValueError, match="not unitary"):
        qc.apply_unitary(bad, [0, 1])


def test_apply_unitary_accepts_valid_unitary():
    qc = QuantumCircuit(2)
    from qbit_simulator.gates import CNOT
    qc.apply_unitary(CNOT, [0, 1])  # should not raise


def test_apply_unitary_check_can_be_disabled():
    qc = QuantumCircuit(2)
    bad = np.zeros((4, 4), dtype=np.complex128)
    qc.apply_unitary(bad, [0, 1], check_unitary=False)  # bypasses check
    assert np.allclose(qc.state, 0.0)


# ---- #2: mid-circuit measurement collapses state ----

def test_measure_collapse_zeros_other_outcomes():
    rng = np.random.default_rng(0)
    qc = QuantumCircuit(2).h(0).h(1)         # uniform over 4 states
    outcome = qc.measure_collapse(rng=rng)
    p = qc.probabilities()
    assert p[outcome] == pytest.approx(1.0)
    assert sum(p) == pytest.approx(1.0)


def test_measure_qubit_collapses_one_qubit_only():
    rng = np.random.default_rng(1)
    qc = QuantumCircuit(2).h(0).h(1)
    bit = qc.measure_qubit(0, rng=rng)
    # After measuring q0, q1 should still be in superposition.
    p = qc.probabilities()
    # All probability should be on states with q0 == bit.
    for idx in range(4):
        q0_bit = (idx >> 1) & 1
        if q0_bit != bit:
            assert p[idx] == pytest.approx(0.0)
    # q1 still 50/50 within the surviving branch.
    surviving = [p[idx] for idx in range(4) if ((idx >> 1) & 1) == bit]
    assert surviving[0] == pytest.approx(0.5)
    assert surviving[1] == pytest.approx(0.5)


def test_measure_qubit_on_bell_pair_correlates_partner():
    rng = np.random.default_rng(2)
    # Bell state: measuring either qubit forces the other to match.
    qc = QuantumCircuit(2).h(0).cnot(0, 1)
    bit = qc.measure_qubit(0, rng=rng)
    p = qc.probabilities()
    expected_index = 0b11 if bit == 1 else 0b00
    assert p[expected_index] == pytest.approx(1.0)


# ---- #3: sample clips negatives ----

def test_sample_clips_tiny_negative_drift():
    rng = np.random.default_rng(0)
    probs = np.array([0.5 + 1e-17, 0.5 - 1e-17, -1e-18, 0.0])
    # Without clipping, rng.choice raises on negative p.
    out = sample(probs, shots=100, rng=rng)
    assert set(out.tolist()).issubset({0, 1, 2, 3})


def test_sample_rejects_zero_sum():
    with pytest.raises(ValueError, match="sum to zero"):
        sample(np.zeros(4))


# ---- #4: Qubit.apply skips redundant normalize ----

def test_qubit_apply_does_not_normalize_by_default():
    from qbit_simulator.gates import X
    # Construct an intentionally un-normalized qubit by skipping the public API.
    q = Qubit.zero()
    q.neurons[:] = [2.0, 0.0, 0.0, 0.0]      # alpha = 2 (not normalized)
    q.apply(X)                                # gate doesn't renormalize
    assert q.neurons[2] == pytest.approx(2.0)


def test_qubit_apply_renormalize_flag_works():
    from qbit_simulator.gates import X
    q = Qubit.zero()
    q.neurons[:] = [2.0, 0.0, 0.0, 0.0]
    q.apply(X, renormalize=True)
    assert np.linalg.norm(q.neurons) == pytest.approx(1.0)


# ---- #5: circuit_ascii has per-column widths and nicer tags ----

def test_circuit_ascii_per_column_widths():
    qc = QuantumCircuit(2).h(0).rx(0.5, 1).cnot(0, 1)
    diagram = circuit_ascii(qc)
    lines = diagram.split("\n")
    # All rows should be the same length when properly column-aligned.
    assert len(set(len(line) for line in lines)) == 1
    assert "Rx(0.5)" in diagram


def test_circuit_ascii_measurement_tag():
    rng = np.random.default_rng(0)
    qc = QuantumCircuit(2).h(0).cnot(0, 1)
    qc.measure_qubit(0, rng=rng)
    diagram = circuit_ascii(qc)
    assert "[M=" in diagram


def test_circuit_ascii_empty_circuit():
    qc = QuantumCircuit(3)
    diagram = circuit_ascii(qc)
    assert "q0:" in diagram and "q1:" in diagram and "q2:" in diagram


# ---- #6: viz module imports cleanly (no unused import warnings) ----

def test_viz_imports_clean():
    import qbit_simulator.viz as v
    assert hasattr(v, "circuit_ascii")
    assert hasattr(v, "plot_bloch")
