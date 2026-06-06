import numpy as np
import pytest

from qbit_simulator.algorithms.qaoa import (
    maxcut_hamiltonian, qaoa, qaoa_ansatz, sample_maxcut_solution,
)


def test_maxcut_hamiltonian_triangle():
    # Triangle: max cut = 2 (any single vertex on one side).
    edges = [(0, 1), (1, 2), (0, 2)]
    H = maxcut_hamiltonian(edges, 3)
    eigvals = np.linalg.eigvalsh(H.matrix())
    assert max(eigvals) == pytest.approx(2.0)


def test_qaoa_finds_max_cut_of_triangle():
    edges = [(0, 1), (1, 2), (0, 2)]
    theta_opt, max_cost, _ = qaoa(edges, n_qubits=3, p=2, seed=0)
    # Triangle max-cut is 2; QAOA at p=2 should approach this closely.
    assert max_cost > 1.4


def test_qaoa_samples_a_cut_partition():
    # Square graph: max cut = 4 (bipartition into alternating vertices).
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    theta_opt, max_cost, _ = qaoa(edges, n_qubits=4, p=2, seed=1)
    counts = sample_maxcut_solution(edges, 4, theta_opt, shots=2000, seed=2)
    # The two optimal cuts are "0101" and "1010"
    optimal_total = counts.get("0101", 0) + counts.get("1010", 0)
    assert optimal_total > 800   # at least 40% on optimal cuts
