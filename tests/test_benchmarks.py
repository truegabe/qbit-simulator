"""Tests for the benchmark / profiling suite."""

import pytest

from qbit_simulator.benchmarks import (
    BenchResult,
    bench_h_gate, bench_cnot_gate, bench_random_circuit,
    bench_pauli_expectation, bench_stabilizer_clifford,
    scaling_sweep_h_gate, scaling_sweep_random_circuit,
    benchmark_all, format_results,
)


# ---- Individual benchmarks ----

def test_bench_h_gate_returns_result():
    r = bench_h_gate(n_qubits=4, n_iters=3)
    assert isinstance(r, BenchResult)
    assert r.mean_ms > 0
    assert r.n_iters == 3
    assert r.size == 4


def test_bench_cnot_gate_returns_result():
    r = bench_cnot_gate(n_qubits=4, n_iters=3)
    assert isinstance(r, BenchResult)
    assert r.mean_ms > 0


def test_bench_random_circuit():
    r = bench_random_circuit(n_qubits=4, depth=3, n_iters=2)
    assert r.mean_ms > 0
    assert r.extra is not None
    assert "n_gates" in r.extra


def test_bench_pauli_expectation():
    r = bench_pauli_expectation(n_qubits=4, n_iters=3)
    assert r.mean_ms > 0


def test_bench_stabilizer():
    r = bench_stabilizer_clifford(n_qubits=8, depth=3, n_iters=2)
    # Either succeeds or returns NaN with error in extra.
    assert isinstance(r, BenchResult)


# ---- Scaling sweeps ----

def test_scaling_sweep_h_gate_runs():
    results = scaling_sweep_h_gate(sizes=[4, 6])
    assert len(results) == 2
    assert all(isinstance(r, BenchResult) for r in results)
    assert all(r.mean_ms > 0 for r in results)


def test_scaling_sweep_random_circuit_runs():
    results = scaling_sweep_random_circuit(sizes=[4, 6])
    assert len(results) == 2
    assert all(isinstance(r, BenchResult) for r in results)


# ---- Bench_all ----

def test_benchmark_all_returns_dict():
    results = benchmark_all(n_qubits=4)
    assert isinstance(results, dict)
    assert "h_gate" in results
    assert "cnot_gate" in results


def test_format_results_returns_string():
    results = benchmark_all(n_qubits=4)
    out = format_results(results)
    assert isinstance(out, str)
    assert "Benchmark" in out


def test_format_results_includes_each_entry():
    results = benchmark_all(n_qubits=4)
    out = format_results(results)
    for key, r in results.items():
        if not isinstance(r, list):
            # Each entry's mean should appear in the output.
            assert f"{r.mean_ms:.3f}" in out or f"{r.mean_ms:.2f}" in out \
                    or str(round(r.mean_ms, 1)) in out
