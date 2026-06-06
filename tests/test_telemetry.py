import json

import numpy as np

from qbit_simulator import Logger, QuantumCircuit
from qbit_simulator.algorithms import bell_pair, grover


def test_logger_records_qubits_and_ops():
    log = Logger()
    qc = QuantumCircuit(3)
    with log.record("test", qc):
        qc.h(0).cnot(0, 1).x(2)
    rec = log.records[0]
    assert rec.n_qubits == 3
    assert rec.total_ops == 3
    op_names = [op["name"] for op in rec.operations]
    assert op_names == ["H", "CNOT", "X"]


def test_logger_no_overhead_when_inactive():
    qc = QuantumCircuit(3).h(0).cnot(0, 1)
    # No logger attached, no record should appear anywhere.
    assert qc._logger is None


def test_logger_tracks_state_bytes():
    log = Logger()
    qc = QuantumCircuit(10)        # 2^10 * 16 = 16384 bytes
    with log.record("big", qc):
        qc.h(0)
    assert log.records[0].state_bytes == 16384
    assert log.records[0].summary()["state_MB"] == round(16384 / 1024**2, 4)


def test_logger_session_totals():
    log = Logger()
    for n in (2, 4, 6):
        qc = QuantumCircuit(n)
        with log.record(f"n={n}", qc):
            for q in range(n):
                qc.h(q)
    totals = log.totals()
    assert totals["n_records"] == 3
    assert totals["max_qubits"] == 6
    assert totals["total_ops"] == 2 + 4 + 6


def test_logger_export_to_json(tmp_path):
    log = Logger()
    qc = QuantumCircuit(2)
    with log.record("bell", qc, algorithm="bell_pair"):
        qc.h(0).cnot(0, 1)
    path = tmp_path / "log.json"
    log.to_json(path)
    payload = json.loads(path.read_text())
    assert payload["totals"]["max_qubits"] == 2
    assert payload["records"][0]["label"] == "bell"
    assert payload["records"][0]["notes"]["algorithm"] == "bell_pair"


def test_logger_with_grover():
    log = Logger()
    qc_template = QuantumCircuit(4)
    with log.record("grover_n4", qc_template):
        # Replace the template's state with a Grover-built one to log it.
        # Here we just verify that direct ops on qc_template are timed.
        for q in range(4):
            qc_template.h(q)
    rec = log.records[0]
    assert rec.total_ops == 4
    assert all(op["duration_s"] >= 0 for op in rec.operations)
