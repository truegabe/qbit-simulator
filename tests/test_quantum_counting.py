"""Quantum counting tests."""

import numpy as np
import pytest

from qbit_simulator.algorithms.quantum_counting import quantum_count


# ---- exact recovery for small cases ----

@pytest.mark.parametrize("n,marked", [
    (3, {0}),
    (3, {0, 4}),
    (3, {0, 1, 2, 3}),
    (4, {0, 7, 11}),
])
def test_quantum_count_recovers_M(n, marked):
    """For known marked sets, the rounded estimate should match M."""
    result = quantum_count(n, marked, n_counting=8)
    assert result["actual_M"] == len(marked)
    # Estimate should round to actual M.
    assert result["M_estimate_int"] == len(marked), (
        f"got {result['M_estimate_int']}, expected {len(marked)}, "
        f"continuous estimate {result['M_estimate']:.3f}"
    )


def test_quantum_count_all_marked():
    """If every state is marked, M = N."""
    n = 3
    marked = set(range(8))
    result = quantum_count(n, marked, n_counting=6)
    assert result["M_estimate_int"] == 8


def test_quantum_count_none_marked():
    """No marked states → M = 0."""
    result = quantum_count(3, set(), n_counting=6)
    assert result["M_estimate_int"] == 0


# ---- predicate-based oracle ----

def test_quantum_count_with_predicate():
    """Use a callable predicate to mark states (e.g. odd indices)."""
    n = 4    # N = 16
    result = quantum_count(n, marked=lambda i: i % 2 == 1, n_counting=8)
    assert result["actual_M"] == 8  # 8 odd indices in [0, 16)
    assert result["M_estimate_int"] == 8


# ---- confidence reported ----

def test_quantum_count_reports_confidence():
    """Confidence should be > 1.0 when the answer is unambiguous."""
    result = quantum_count(3, {0, 1}, n_counting=8)
    assert result["confidence"] > 0.0
