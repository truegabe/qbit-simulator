import numpy as np
import pytest

from qbit_simulator import Qubit, H, X, Y, Z, S, T, I2, CNOT
from qbit_simulator.gates import is_unitary


@pytest.mark.parametrize("g", [H, X, Y, Z, S, T, CNOT])
def test_gates_are_unitary(g):
    assert is_unitary(g)


def test_h_squared_is_identity():
    assert np.allclose(H @ H, I2)


def test_x_flips_zero_to_one():
    q = Qubit.zero().apply(X)
    assert q.prob_one() == pytest.approx(1.0)


def test_x_flips_one_to_zero():
    q = Qubit.one().apply(X)
    assert q.prob_zero() == pytest.approx(1.0)


def test_h_creates_superposition():
    q = Qubit.zero().apply(H)
    assert q.prob_zero() == pytest.approx(0.5)
    assert q.prob_one() == pytest.approx(0.5)


def test_z_phase_flip_on_plus_yields_minus():
    q = Qubit.plus().apply(Z)
    expected = Qubit.minus().state
    # Allow global phase, compare via |<minus|psi>|^2 == 1.
    overlap = np.abs(np.vdot(expected, q.state)) ** 2
    assert overlap == pytest.approx(1.0)


def test_pauli_relations():
    # XYZ = iI
    assert np.allclose(X @ Y @ Z, 1j * I2)
