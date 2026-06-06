"""Mermin-GHZ inequality tests.

Verify both the quantum value (2^(N-1) for N-qubit GHZ) and the classical
LHV bound, and check the violation grows as expected.
"""

import pytest

from qbit_simulator.algorithms.mermin_ghz import (
    mermin_polynomial_terms,
    mermin_quantum_value,
    mermin_classical_bound,
    mermin_violation_report,
    make_ghz,
)


def test_term_count_for_n3():
    # For N=3: even Y counts are 0 and 2. 1 + C(3,2) = 1 + 3 = 4 terms.
    terms = mermin_polynomial_terms(3)
    assert len(terms) == 4


def test_term_count_for_n5():
    # k = 0, 2, 4. Counts: 1 + 10 + 5 = 16.
    terms = mermin_polynomial_terms(5)
    assert len(terms) == 16


def test_quantum_value_n3():
    assert abs(mermin_quantum_value(3)) == 4   # 2^(3-1)


def test_quantum_value_n4():
    assert abs(mermin_quantum_value(4)) == 8   # 2^(4-1)


@pytest.mark.parametrize("n", [3, 4, 5, 6, 7, 8])
def test_quantum_value_is_2_to_n_minus_1(n):
    """⟨M_N⟩ on GHZ has magnitude 2^(N-1)."""
    assert abs(mermin_quantum_value(n)) == 2 ** (n - 1)


def test_classical_bound_growth():
    # Sanity: classical bound is 2^(N/2), so N=2 → 2, N=4 → 4, N=6 → 8.
    assert mermin_classical_bound(2) == 2
    assert mermin_classical_bound(4) == 4
    assert mermin_classical_bound(6) == 8
    assert mermin_classical_bound(8) == 16


def test_violation_grows_exponentially():
    """Ratio quantum / classical should grow exponentially with N."""
    prev_ratio = 0
    for n in (3, 5, 7, 9):
        r = mermin_violation_report(n)
        assert r["violation"] > prev_ratio
        prev_ratio = r["violation"]


def test_full_report_n3():
    r = mermin_violation_report(3)
    assert r["n"] == 3
    assert r["terms"] == 4
    assert r["quantum"] == 4
    assert r["classical_bound"] == 2
    assert r["violation"] == 2.0
