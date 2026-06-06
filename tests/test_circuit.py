import numpy as np
import pytest

from qbit_simulator import QuantumCircuit


def test_initial_state_is_all_zero():
    qc = QuantumCircuit(3)
    expected = np.zeros(8, dtype=np.complex128)
    expected[0] = 1.0
    assert np.allclose(qc.state, expected)


def test_x_on_single_qubit_in_register():
    qc = QuantumCircuit(3).x(1)
    probs = qc.probabilities()
    # |010> = index 2
    assert probs[0b010] == pytest.approx(1.0)


def test_h_on_qubit_0_creates_superposition():
    qc = QuantumCircuit(2).h(0)
    p = qc.probabilities()
    assert p[0b00] == pytest.approx(0.5)
    assert p[0b10] == pytest.approx(0.5)


def test_cnot_propagates_control():
    qc = QuantumCircuit(2).x(0).cnot(0, 1)
    p = qc.probabilities()
    assert p[0b11] == pytest.approx(1.0)


def test_cnot_does_nothing_when_control_zero():
    qc = QuantumCircuit(2).cnot(0, 1)
    p = qc.probabilities()
    assert p[0b00] == pytest.approx(1.0)


def test_cnot_non_adjacent():
    # 3 qubits: flip q0, then CNOT(0->2). Expect |101>.
    qc = QuantumCircuit(3).x(0).cnot(0, 2)
    p = qc.probabilities()
    assert p[0b101] == pytest.approx(1.0)


def test_state_stays_normalized():
    qc = QuantumCircuit(3).h(0).h(1).cnot(0, 2).x(1).h(2)
    assert np.linalg.norm(qc.state) == pytest.approx(1.0)
