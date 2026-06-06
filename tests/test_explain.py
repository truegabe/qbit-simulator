"""Tests for explain_circuit."""

import numpy as np

from qbit_simulator import QuantumCircuit
from qbit_simulator.explain import (
    explain_circuit, describe_gate, _parse_op, _identify_algorithm,
)


def test_parse_op_basic():
    assert _parse_op("H(0)") == ("H", ["0"])
    assert _parse_op("CNOT(0,1)") == ("CNOT", ["0", "1"])
    assert _parse_op("Rx(0.5,2)") == ("Rx", ["0.5", "2"])


def test_describe_gate_known():
    s = describe_gate("H(0)")
    assert "Hadamard" in s and "qubit 0" in s

    s = describe_gate("CNOT(0,1)")
    assert "CNOT" in s and "0" in s and "1" in s


def test_describe_gate_unknown_returns_raw():
    s = describe_gate("MYSTERY(0,1)")
    assert s == "MYSTERY(0,1)"


def test_identify_bell_pair():
    history = ["H(0)", "CNOT(0,1)"]
    label = _identify_algorithm(history, n=2)
    assert label is not None
    assert "Bell" in label


def test_identify_qft_layer():
    history = ["H(0)", "CP(1.57,1,0)", "H(1)",
               "CP(0.78,2,0)", "CP(1.57,2,1)", "H(2)"]
    label = _identify_algorithm(history, n=3)
    assert label is not None
    assert "Fourier" in label


def test_identify_ghz_cascade():
    history = ["H(0)", "CNOT(0,1)", "CNOT(1,2)", "CNOT(2,3)"]
    label = _identify_algorithm(history, n=4)
    assert label is not None
    assert "GHZ" in label


def test_explain_empty_circuit():
    qc = QuantumCircuit(3)
    text = explain_circuit(qc)
    assert "empty" in text.lower()


def test_explain_bell_pair_contains_patterns():
    qc = QuantumCircuit(2).h(0).cnot(0, 1)
    text = explain_circuit(qc)
    assert "Bell pair" in text
    assert "Hadamard" in text
    assert "CNOT" in text
    assert "after op" in text  # step-by-step section ran


def test_explain_skips_step_through_for_big_circuits():
    qc = QuantumCircuit(8)
    for q in range(8):
        qc.h(q)
    text = explain_circuit(qc)
    # n > 6 so step-through suppressed.
    assert "Step-by-step state evolution:" not in text


def test_explain_returns_string():
    qc = QuantumCircuit(3).h(0).cnot(0, 1).h(2)
    assert isinstance(explain_circuit(qc), str)
