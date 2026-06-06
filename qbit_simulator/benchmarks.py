"""Performance-profiling suite for core simulator operations.

Measures runtime and throughput for key primitives:

  * State-vector single-qubit gates
  * Two-qubit gates (CNOT, CZ)
  * Stabilizer simulation
  * MPS-based circuit application
  * Pauli expectation evaluation
  * Tomography

Use as a regression / scaling probe:

    from qbit_simulator.benchmarks import benchmark_all, format_results
    results = benchmark_all()
    print(format_results(results))

Each benchmark returns:
  * mean wall time in milliseconds
  * standard deviation
  * throughput (gates/sec or shots/sec where applicable)
  * problem size

We use `time.perf_counter` for monotonic high-resolution timing and
warm up with a small "discard" run before the measured iterations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class BenchResult:
    name:        str
    mean_ms:     float
    std_ms:      float
    n_iters:     int
    size:        int
    throughput:  float | None = None    # operations per second
    extra:       dict | None = None


def _time_function(fn: Callable[[], None], n_iters: int = 10
                    ) -> tuple[float, float]:
    """Run `fn` n_iters times, return (mean_ms, std_ms). Discards 1
    warm-up run."""
    # Warm-up.
    fn()
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    arr = np.asarray(times)
    return float(arr.mean()), float(arr.std())


# ----------------------------------------------------------------------------
# State-vector gate benchmarks
# ----------------------------------------------------------------------------

def bench_h_gate(n_qubits: int = 12, n_iters: int = 20) -> BenchResult:
    """Time a single H gate on a fresh n_qubit state vector."""
    from .circuit import QuantumCircuit

    def fn():
        qc = QuantumCircuit(n_qubits)
        qc.h(0)

    mean, std = _time_function(fn, n_iters)
    return BenchResult(
        name="H gate on n-qubit state",
        mean_ms=mean, std_ms=std, n_iters=n_iters,
        size=n_qubits,
        throughput=1000.0 / mean if mean > 0 else None,
    )


def bench_cnot_gate(n_qubits: int = 12, n_iters: int = 20) -> BenchResult:
    """Time a CNOT on a fresh n_qubit state vector."""
    from .circuit import QuantumCircuit

    def fn():
        qc = QuantumCircuit(n_qubits)
        qc.cnot(0, 1)

    mean, std = _time_function(fn, n_iters)
    return BenchResult(
        name="CNOT gate on n-qubit state",
        mean_ms=mean, std_ms=std, n_iters=n_iters,
        size=n_qubits,
        throughput=1000.0 / mean if mean > 0 else None,
    )


def bench_random_circuit(n_qubits: int = 8, depth: int = 20,
                           n_iters: int = 5) -> BenchResult:
    """Time a random circuit of `depth` layers of H + CNOTs."""
    from .circuit import QuantumCircuit
    rng = np.random.default_rng(0)
    schedule = []
    for _ in range(depth):
        for q in range(n_qubits):
            schedule.append(("h", q))
        for q in range(n_qubits - 1):
            schedule.append(("cnot", q, q + 1))

    def fn():
        qc = QuantumCircuit(n_qubits)
        for op in schedule:
            if op[0] == "h":
                qc.h(op[1])
            elif op[0] == "cnot":
                qc.cnot(op[1], op[2])

    mean, std = _time_function(fn, n_iters)
    n_gates = len(schedule)
    return BenchResult(
        name=f"Random circuit (n={n_qubits}, depth={depth})",
        mean_ms=mean, std_ms=std, n_iters=n_iters,
        size=n_qubits,
        throughput=(n_gates * 1000.0 / mean) if mean > 0 else None,
        extra={"gates_per_sec": (n_gates * 1000.0 / mean) if mean > 0 else None,
               "n_gates":       n_gates},
    )


# ----------------------------------------------------------------------------
# Stabilizer benchmark
# ----------------------------------------------------------------------------

def bench_stabilizer_clifford(n_qubits: int = 50, depth: int = 20,
                                n_iters: int = 5) -> BenchResult:
    """Time a stabilizer-Clifford circuit (Gottesman-Knill regime)."""
    try:
        from .stabilizer import StabilizerState
    except ImportError:
        return BenchResult(
            name=f"Stabilizer (n={n_qubits})",
            mean_ms=float("nan"), std_ms=float("nan"),
            n_iters=0, size=n_qubits,
            extra={"error": "stabilizer module not available"},
        )

    def fn():
        st = StabilizerState(n_qubits)
        for _ in range(depth):
            for q in range(n_qubits):
                st.h(q)
            for q in range(0, n_qubits - 1, 2):
                st.cnot(q, q + 1)

    mean, std = _time_function(fn, n_iters)
    n_gates = depth * (n_qubits + (n_qubits // 2))
    return BenchResult(
        name=f"Stabilizer Clifford (n={n_qubits}, depth={depth})",
        mean_ms=mean, std_ms=std, n_iters=n_iters,
        size=n_qubits,
        throughput=(n_gates * 1000.0 / mean) if mean > 0 else None,
    )


# ----------------------------------------------------------------------------
# Pauli expectation benchmark
# ----------------------------------------------------------------------------

def bench_pauli_expectation(n_qubits: int = 8, n_iters: int = 10
                              ) -> BenchResult:
    """Time computation of a Pauli expectation on a state vector."""
    from .circuit import QuantumCircuit
    qc = QuantumCircuit(n_qubits)
    for q in range(n_qubits):
        qc.h(q)
    psi = qc.state
    pauli_string = "XYZ" * (n_qubits // 3) + "X" * (n_qubits % 3)
    pauli_string = pauli_string[:n_qubits]
    # Build the Pauli matrix once.
    from .tomography import pauli_string_matrix
    P = pauli_string_matrix(pauli_string)

    def fn():
        _ = float(np.real(psi.conj() @ P @ psi))

    mean, std = _time_function(fn, n_iters)
    return BenchResult(
        name=f"Pauli expectation (n={n_qubits})",
        mean_ms=mean, std_ms=std, n_iters=n_iters,
        size=n_qubits,
    )


# ----------------------------------------------------------------------------
# Scaling sweep
# ----------------------------------------------------------------------------

def scaling_sweep_h_gate(sizes: list[int] | None = None) -> list[BenchResult]:
    """Run bench_h_gate across multiple problem sizes to extract scaling."""
    sizes = sizes if sizes is not None else [4, 6, 8, 10, 12]
    return [bench_h_gate(n_qubits=n, n_iters=20) for n in sizes]


def scaling_sweep_random_circuit(sizes: list[int] | None = None
                                   ) -> list[BenchResult]:
    sizes = sizes if sizes is not None else [4, 6, 8, 10]
    return [bench_random_circuit(n_qubits=n, depth=10, n_iters=3) for n in sizes]


# ----------------------------------------------------------------------------
# Run all + format
# ----------------------------------------------------------------------------

def benchmark_all(n_qubits: int = 8) -> dict:
    """Run all benchmarks at moderate problem size; return a dict.

    Cap problem size to keep total runtime under a few seconds.
    """
    results = {
        "h_gate":          bench_h_gate(n_qubits, n_iters=10),
        "cnot_gate":       bench_cnot_gate(n_qubits, n_iters=10),
        "random_circuit":  bench_random_circuit(n_qubits, depth=10, n_iters=3),
        "pauli_exp":       bench_pauli_expectation(n_qubits, n_iters=10),
        "stabilizer":      bench_stabilizer_clifford(
            n_qubits=min(50, 4 * n_qubits), depth=10, n_iters=3
        ),
    }
    return results


def format_results(results: dict) -> str:
    """Pretty-print a benchmark dict."""
    lines = [
        f"{'Benchmark':<50}  {'Size':>5}  {'Mean (ms)':>12}  {'Std (ms)':>10}",
        "-" * 84,
    ]
    for key, r in results.items():
        if isinstance(r, list):
            for entry in r:
                lines.append(
                    f"{entry.name:<50}  {entry.size:>5}  "
                    f"{entry.mean_ms:>12.3f}  {entry.std_ms:>10.3f}"
                )
        else:
            lines.append(
                f"{r.name:<50}  {r.size:>5}  "
                f"{r.mean_ms:>12.3f}  {r.std_ms:>10.3f}"
            )
    return "\n".join(lines)
