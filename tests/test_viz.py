import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from qbit_simulator import Qubit, QuantumCircuit
from qbit_simulator.algorithms import bell_pair
from qbit_simulator.viz import (
    plot_probabilities, plot_counts, plot_bloch, bloch_coords, circuit_ascii,
)


def test_bloch_coords_known_states():
    assert bloch_coords(Qubit.zero()) == pytest.approx((0.0, 0.0, 1.0))
    assert bloch_coords(Qubit.one()) == pytest.approx((0.0, 0.0, -1.0))
    assert bloch_coords(Qubit.plus()) == pytest.approx((1.0, 0.0, 0.0))


def test_plot_probabilities_returns_fig():
    qc = bell_pair()
    fig = plot_probabilities(qc)
    assert fig is not None


def test_plot_counts_returns_fig():
    fig = plot_counts({"00": 500, "11": 500})
    assert fig is not None


def test_plot_bloch_returns_fig():
    fig = plot_bloch(Qubit.plus())
    assert fig is not None


def test_circuit_ascii_contains_qubits():
    qc = QuantumCircuit(3).h(0).cnot(0, 1).x(2).cnot(1, 2)
    diagram = circuit_ascii(qc)
    assert "q0:" in diagram and "q1:" in diagram and "q2:" in diagram
    assert "H" in diagram and "X" in diagram and "*" in diagram
