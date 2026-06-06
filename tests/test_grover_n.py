import numpy as np
import pytest

from qbit_simulator.algorithms import grover, optimal_iterations


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 8, 10, 12])
def test_grover_finds_marked_with_high_probability(n):
    marked = (2**n) - 1  # something non-trivial like |11..1>
    qc = grover(n, marked)
    p = qc.probabilities()
    assert p[marked] > 0.9, f"n={n}: P(marked)={p[marked]:.4f}"


def test_grover_iteration_counts():
    assert optimal_iterations(2) == 1   # N=4
    assert optimal_iterations(3) == 2   # N=8
    assert optimal_iterations(4) == 3   # N=16
    assert optimal_iterations(6) == 6   # N=64


@pytest.mark.parametrize("n,marked", [(3, 5), (4, 9), (5, 17)])
def test_grover_each_marked_item(n, marked):
    qc = grover(n, marked)
    p = qc.probabilities()
    argmax = int(np.argmax(p))
    assert argmax == marked
