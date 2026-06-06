"""Circuit-diagram renderer tests."""

import pytest

from qbit_simulator.circuit import QuantumCircuit
from qbit_simulator.circuit_render import (
    text_diagram, unicode_diagram, circuit_stats,
    _parse_history, _column_assignments,
)


# ---- Parser ----

def test_parse_simple_history():
    ops = _parse_history(["H(0)", "X(1)", "CNOT(0,1)"])
    assert len(ops) == 3
    assert ops[0].name == "H"
    assert ops[0].targets == [0]
    assert ops[2].name == "CNOT"
    assert ops[2].targets == [0, 1]


def test_parse_parameterized_gate():
    ops = _parse_history(["RX(0.7,2)"])
    assert ops[0].name == "RX"
    assert ops[0].targets == [2]
    assert "0.7" in ops[0].label


def test_parse_ignores_unparseable():
    ops = _parse_history(["nonsense", "H(0)", "blah"])
    # Only the gate-like entry parses.
    assert any(op.name == "H" for op in ops)


# ---- Column assignment ----

def test_column_assignment_serializes_dependencies():
    """A gate on qubit 0 followed by a gate on qubit 0 → cols 0 and 1."""
    ops = _parse_history(["H(0)", "X(0)"])
    cols = _column_assignments(ops, 2)
    assert cols == [0, 1]


def test_column_assignment_parallelizes_independent():
    """Gates on different qubits → same column."""
    ops = _parse_history(["H(0)", "X(1)"])
    cols = _column_assignments(ops, 2)
    assert cols == [0, 0]


def test_column_assignment_multiqubit_blocks_intermediate():
    """A CNOT(0,2) should block any concurrent gate on qubit 1."""
    ops = _parse_history(["CNOT(0,2)", "X(1)"])
    cols = _column_assignments(ops, 3)
    assert cols[0] == 0
    assert cols[1] == 1     # forced to next column


# ---- Text rendering ----

def test_text_diagram_returns_multiline_string():
    qc = QuantumCircuit(2)
    qc.h(0).cnot(0, 1)
    diagram = text_diagram(qc)
    assert isinstance(diagram, str)
    assert "\n" in diagram


def test_text_diagram_shows_qubit_labels():
    qc = QuantumCircuit(3)
    qc.h(0)
    diagram = text_diagram(qc)
    assert "q0:" in diagram
    assert "q1:" in diagram
    assert "q2:" in diagram


def test_text_diagram_contains_gate_names():
    qc = QuantumCircuit(2)
    qc.h(0).cnot(0, 1).rx(0.5, 1)
    diagram = text_diagram(qc)
    assert "H" in diagram
    assert "CNOT" in diagram
    assert "RX" in diagram


def test_text_diagram_empty_circuit():
    qc = QuantumCircuit(2)
    diagram = text_diagram(qc)
    assert "q0:" in diagram


# ---- Unicode rendering ----

def test_unicode_diagram_replaces_wire_chars():
    qc = QuantumCircuit(2)
    qc.h(0).cnot(0, 1)
    diagram = unicode_diagram(qc)
    # ASCII dashes/pipes should be replaced with box characters.
    assert "─" in diagram
    # Multi-qubit connector becomes vertical box char.
    assert "│" in diagram


def test_unicode_diagram_no_ascii_wires():
    qc = QuantumCircuit(2)
    qc.h(0).cnot(0, 1)
    diagram = unicode_diagram(qc)
    assert "-" not in diagram


# ---- Stats ----

def test_circuit_stats_counts():
    qc = QuantumCircuit(3)
    qc.h(0).h(1).cnot(0, 1).cnot(1, 2).rx(0.1, 0)
    stats = circuit_stats(qc)
    assert stats["total_gates"] == 5
    assert stats["by_type"]["H"] == 2
    assert stats["by_type"]["CNOT"] == 2
    assert stats["by_type"]["RX"] == 1
    assert stats["n_qubits"] == 3


def test_circuit_stats_depth():
    """H(0) → CNOT(0,1) → X(0) — depth 3 (all forced sequential)."""
    qc = QuantumCircuit(2)
    qc.h(0).cnot(0, 1).x(0)
    assert circuit_stats(qc)["depth"] == 3


def test_circuit_stats_parallel_depth():
    """H(0) and H(1) can run in parallel → depth 1."""
    qc = QuantumCircuit(2)
    qc.h(0).h(1)
    assert circuit_stats(qc)["depth"] == 1


def test_circuit_stats_empty():
    qc = QuantumCircuit(2)
    stats = circuit_stats(qc)
    assert stats["total_gates"] == 0
    assert stats["depth"] == 0
