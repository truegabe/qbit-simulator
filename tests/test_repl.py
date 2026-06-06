"""Smoke tests for the unified REPL.

Drives the `handle()` function directly with strings to verify routing,
without running the interactive loop.
"""

import io
import sys
from pathlib import Path

import pytest

from qbit_simulator.repl import handle, suggest_command, KNOWN_COMMANDS
from qbit_simulator.simlog import SimLog


@pytest.fixture
def fresh_state():
    return {"qc": None, "last_result": None}


@pytest.fixture
def tmp_log(tmp_path):
    return SimLog(tmp_path / "log.jsonl")


def _drive(line, state, log, capsys=None):
    handled = handle(line, line.lower(), state, log)
    return handled


def test_help_recognized(fresh_state, tmp_log, capsys):
    assert _drive("help", fresh_state, tmp_log)
    out = capsys.readouterr().out
    assert "CIRCUIT BUILDING" in out
    assert "ALGORITHMS" in out


def test_circuit_creation(fresh_state, tmp_log):
    assert _drive("circuit 4", fresh_state, tmp_log)
    assert fresh_state["qc"] is not None
    assert fresh_state["qc"].n == 4


def test_gate_application(fresh_state, tmp_log):
    _drive("circuit 3", fresh_state, tmp_log)
    _drive("h 0", fresh_state, tmp_log)
    _drive("cnot 0 1", fresh_state, tmp_log)
    qc = fresh_state["qc"]
    assert len(qc.history) == 2


def test_reset(fresh_state, tmp_log):
    _drive("circuit 2", fresh_state, tmp_log)
    _drive("reset", fresh_state, tmp_log)
    assert fresh_state["qc"] is None


def test_bell_algorithm(fresh_state, tmp_log):
    assert _drive("bell", fresh_state, tmp_log)
    qc = fresh_state["qc"]
    assert qc.n == 2
    # Bell pair: 50/50 between |00> and |11>
    p = qc.probabilities()
    assert abs(p[0] - 0.5) < 1e-9
    assert abs(p[3] - 0.5) < 1e-9
    # Logged.
    assert tmp_log.lifetime_stats()["total_runs"] == 1


def test_grover_algorithm(fresh_state, tmp_log):
    assert _drive("grover 6", fresh_state, tmp_log)
    qc = fresh_state["qc"]
    assert qc.n == 6
    # Default marked = 2^N - 1 = 63
    assert qc.probabilities()[63] > 0.9


def test_grover_marked(fresh_state, tmp_log):
    assert _drive("grover 5 marked=10", fresh_state, tmp_log)
    qc = fresh_state["qc"]
    assert qc.probabilities()[10] > 0.9


def test_shor_15(fresh_state, tmp_log):
    """Shor on 15 should produce factors."""
    assert _drive("shor 15", fresh_state, tmp_log)
    events = tmp_log.all()
    assert any(e.kind == "shor" for e in events)


def test_unknown_command_returns_false(fresh_state, tmp_log):
    handled = handle("zzznotacommand", "zzznotacommand", fresh_state, tmp_log)
    assert handled is False


def test_typo_suggester():
    assert suggest_command("bellp") in {"bell", None}  # accept fuzzy match
    assert suggest_command("grov") == "grover"
    assert suggest_command("shorr") == "shor"
    assert suggest_command("xxxxxxx") is None


def test_known_commands_populated():
    for cmd in ["help", "circuit", "h", "cnot", "bell", "grover", "qft",
                "shor", "vqe", "teleport", "chsh", "save", "load", "stats"]:
        assert cmd in KNOWN_COMMANDS, f"missing '{cmd}' in KNOWN_COMMANDS"


def test_explain_command_runs(fresh_state, tmp_log, capsys):
    _drive("bell", fresh_state, tmp_log)
    assert _drive("explain", fresh_state, tmp_log)
    out = capsys.readouterr().out
    assert "Hadamard" in out or "Bell" in out


def test_save_load_roundtrip(fresh_state, tmp_log, tmp_path, monkeypatch):
    # Redirect SAVE_DIR to tmp.
    import qbit_simulator.repl as repl_mod
    monkeypatch.setattr(repl_mod, "SAVE_DIR", tmp_path)
    _drive("bell", fresh_state, tmp_log)
    _drive("save bell_test", fresh_state, tmp_log)
    fresh_state["qc"] = None
    _drive("load bell_test", fresh_state, tmp_log)
    assert fresh_state["qc"] is not None
    assert fresh_state["qc"].n == 2


def test_circuit_required_for_gate_ops(fresh_state, tmp_log, capsys):
    # No circuit set; gate ops should print a hint and not crash.
    _drive("h 0", fresh_state, tmp_log)
    out = capsys.readouterr().out
    assert "no circuit" in out.lower()


def test_chsh_records_event(fresh_state, tmp_log):
    _drive("chsh 100", fresh_state, tmp_log)
    events = tmp_log.by_kind("chsh")
    assert len(events) == 1
    assert "quantum" in events[0].result_summary
    assert events[0].result_summary["quantum"] > 0.0
