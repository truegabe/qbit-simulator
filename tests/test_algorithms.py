import numpy as np
import pytest

from qbit_simulator.algorithms import bell_pair, deutsch, grover_2q


def test_bell_state_amplitudes():
    qc = bell_pair()
    p = qc.probabilities()
    assert p[0b00] == pytest.approx(0.5)
    assert p[0b11] == pytest.approx(0.5)
    assert p[0b01] == pytest.approx(0.0)
    assert p[0b10] == pytest.approx(0.0)


def test_bell_state_measurements_correlated():
    rng = np.random.default_rng(123)
    qc = bell_pair()
    outcomes = qc.measure_all(shots=2000, rng=rng)
    for o in outcomes:
        bits = format(int(o), "02b")
        assert bits[0] == bits[1]  # only 00 or 11


@pytest.mark.parametrize(
    "f,label",
    [
        (lambda x: 0, "constant"),
        (lambda x: 1, "constant"),
        (lambda x: x, "balanced"),
        (lambda x: 1 - x, "balanced"),
    ],
)
def test_deutsch(f, label):
    assert deutsch(f) == label


@pytest.mark.parametrize("marked", [0, 1, 2, 3])
def test_grover_finds_marked(marked):
    qc = grover_2q(marked)
    p = qc.probabilities()
    # Grover on N=4 with 1 iteration finds the marked state with probability 1.
    assert p[marked] == pytest.approx(1.0, abs=1e-9)
