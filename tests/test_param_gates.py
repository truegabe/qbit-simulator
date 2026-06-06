import numpy as np
import pytest

from qbit_simulator import QuantumCircuit
from qbit_simulator.gates import Rx, Ry, Rz, P, CP, SWAP, controlled, Z, is_unitary


@pytest.mark.parametrize("theta", [0.0, 0.3, np.pi / 2, np.pi, 2 * np.pi])
def test_rotation_gates_unitary(theta):
    for g in (Rx(theta), Ry(theta), Rz(theta), P(theta), CP(theta)):
        assert is_unitary(g)


def test_swap_unitary():
    assert is_unitary(SWAP)


def test_ry_pi_flips_zero_to_one():
    qc = QuantumCircuit(1).ry(np.pi, 0)
    assert qc.probabilities()[1] == pytest.approx(1.0)


def test_rx_2pi_is_minus_identity():
    # Rx(2pi) = -I, observable as global phase only, but state preserved.
    qc = QuantumCircuit(1).rx(2 * np.pi, 0)
    assert qc.probabilities()[0] == pytest.approx(1.0)


def test_swap_exchanges_qubits():
    qc = QuantumCircuit(2).x(0).swap(0, 1)
    p = qc.probabilities()
    assert p[0b01] == pytest.approx(1.0)


def test_cz_phase_only_on_11():
    qc = QuantumCircuit(2).x(0).x(1).cz(0, 1)
    # |11> picks up -1; probability unchanged.
    p = qc.probabilities()
    assert p[0b11] == pytest.approx(1.0)
    assert qc.state[0b11] == pytest.approx(-1.0)


def test_apply_unitary_matches_individual_gates():
    qc1 = QuantumCircuit(3).h(0).cnot(0, 2)
    qc2 = QuantumCircuit(3)
    # Build H on q0 and CNOT(0,2) as a 3-qubit unitary using kron in the
    # natural ordering and apply via apply_unitary.
    from qbit_simulator.gates import H, I2, CNOT
    H_on_q0 = np.kron(np.kron(H, I2), I2)
    # CNOT control=q0 target=q2: easier to build via apply with axes; reuse
    # circuit primitives for the reference state.
    qc2.apply_unitary(H_on_q0, [0, 1, 2])
    qc2.cnot(0, 2)
    assert np.allclose(qc1.state, qc2.state)
