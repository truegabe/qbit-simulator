"""Tests for the SimLog lifetime tracker."""

import json
import time
from pathlib import Path

import pytest

from qbit_simulator.simlog import SimLog, SimEvent


def test_empty_log_reports_zero(tmp_path):
    log = SimLog(tmp_path / "log.jsonl")
    s = log.lifetime_stats()
    assert s["total_runs"] == 0
    assert s["kinds"] == {}
    assert s["total_compute_seconds"] == 0.0


def test_record_appends_event(tmp_path):
    p = tmp_path / "log.jsonl"
    log = SimLog(p)
    log.record("grover", n_qubits=10, wall_seconds=0.5, peak_heap_mb=2.0,
               result_summary={"reached": True})
    log.record("qft", n_qubits=8, wall_seconds=0.1)
    log2 = SimLog(p)  # fresh load
    events = log2.all()
    assert len(events) == 2
    assert events[0].kind == "grover"
    assert events[1].kind == "qft"
    assert events[0].result_summary == {"reached": True}


def test_measure_context_records(tmp_path):
    log = SimLog(tmp_path / "log.jsonl")
    with log.measure("test_op", n_qubits=4) as bag:
        bag["result_summary"] = {"foo": "bar"}
        time.sleep(0.01)
    events = log.all()
    assert len(events) == 1
    assert events[0].kind == "test_op"
    assert events[0].n_qubits == 4
    assert events[0].wall_seconds >= 0.01
    assert events[0].result_summary == {"foo": "bar"}


def test_kind_counts(tmp_path):
    log = SimLog(tmp_path / "log.jsonl")
    log.record("grover", 8, 0.1)
    log.record("grover", 10, 0.2)
    log.record("qft", 6, 0.05)
    counts = log.kind_counts()
    assert counts == {"grover": 2, "qft": 1}


def test_biggest_circuit(tmp_path):
    log = SimLog(tmp_path / "log.jsonl")
    log.record("grover", 5, 0.1)
    log.record("qft", 12, 0.1)
    log.record("shor", 8, 0.1)
    big = log.biggest_circuit()
    assert big is not None
    assert big.n_qubits == 12
    assert big.kind == "qft"


def test_lifetime_stats_aggregates(tmp_path):
    log = SimLog(tmp_path / "log.jsonl")
    log.record("grover", 5, 0.10, peak_heap_mb=1.0)
    log.record("qft", 10, 0.50, peak_heap_mb=4.0)
    s = log.lifetime_stats()
    assert s["total_runs"] == 2
    assert s["biggest_n_qubits"] == 10
    assert s["peak_heap_mb_ever"] == pytest.approx(4.0)
    assert s["total_compute_seconds"] == pytest.approx(0.60)


def test_report_returns_string_for_empty_log(tmp_path):
    log = SimLog(tmp_path / "log.jsonl")
    assert isinstance(log.report(), str)


def test_jsonl_format_is_one_event_per_line(tmp_path):
    p = tmp_path / "log.jsonl"
    log = SimLog(p)
    log.record("a", 1, 0.1)
    log.record("b", 2, 0.2)
    lines = p.read_text().strip().split("\n")
    assert len(lines) == 2
    for ln in lines:
        d = json.loads(ln)
        assert "kind" in d and "n_qubits" in d
