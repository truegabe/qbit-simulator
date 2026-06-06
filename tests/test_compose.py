import numpy as np
import pytest

from qbit_simulator import QuantumCircuit


def test_add_concatenates_circuits():
    qc1 = QuantumCircuit(2).h(0)
    qc2 = QuantumCircuit(2).cnot(0, 1)
    qc_full = qc1 + qc2
    # Should equal a Bell pair built directly.
    bell = QuantumCircuit(2).h(0).cnot(0, 1)
    assert np.allclose(qc_full.state, bell.state)


def test_inverse_undoes_circuit():
    qc = QuantumCircuit(3).h(0).cnot(0, 1).rx(0.5, 2).cnot(1, 2)
    qc_inv = qc.inverse()
    combined = qc + qc_inv
    # Should return to |000>.
    initial = np.zeros(8, dtype=np.complex128); initial[0] = 1
    assert np.allclose(combined.state, initial)


def test_controlled_adds_qubit():
    """X.controlled(0) should act as CNOT(control=0, target=1)."""
    cx_ops = QuantumCircuit(1).x(0).controlled(control=0)._ops
    # Replay on |10>: expect |11>.
    test = QuantumCircuit(2).x(0)  # |10>
    test.replay_ops(cx_ops)
    assert test.probabilities()[0b11] == pytest.approx(1.0)
    # Replay on |00>: expect |00> (control off).
    test = QuantumCircuit(2)
    test.replay_ops(cx_ops)
    assert test.probabilities()[0b00] == pytest.approx(1.0)


def test_inverse_of_qft_is_inverse_qft():
    from qbit_simulator.algorithms.qft import qft
    qc = qft(3)
    inv = qc.inverse()
    full = qc + inv
    initial = np.zeros(8, dtype=np.complex128); initial[0] = 1
    assert np.allclose(full.state, initial, atol=1e-10)
