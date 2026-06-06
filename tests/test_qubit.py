import numpy as np
import pytest

from qbit_simulator import Qubit


def test_zero_state():
    q = Qubit.zero()
    assert q.prob_zero() == pytest.approx(1.0)
    assert q.prob_one() == pytest.approx(0.0)


def test_one_state():
    q = Qubit.one()
    assert q.prob_zero() == pytest.approx(0.0)
    assert q.prob_one() == pytest.approx(1.0)


def test_plus_state():
    q = Qubit.plus()
    assert q.prob_zero() == pytest.approx(0.5)
    assert q.prob_one() == pytest.approx(0.5)


def test_normalization():
    q = Qubit(2 + 0j, 0 + 0j)
    assert np.linalg.norm(q.neurons) == pytest.approx(1.0)


def test_4_neurons_layout():
    q = Qubit(0.6 + 0j, 0 + 0.8j)
    assert q.neurons.shape == (4,)
    assert q.neurons[0] == pytest.approx(0.6)
    assert q.neurons[1] == pytest.approx(0.0)
    assert q.neurons[2] == pytest.approx(0.0)
    assert q.neurons[3] == pytest.approx(0.8)
