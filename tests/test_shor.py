import random

import pytest

from qbit_simulator.algorithms.shor import (
    shor, modular_multiplication_unitary, continued_fraction_period,
)
from qbit_simulator.gates import is_unitary


def test_modular_multiplication_is_permutation():
    U = modular_multiplication_unitary(2, 15, 4)
    assert is_unitary(U)
    # Each column should have exactly one 1 (it's a permutation matrix).
    for col in range(16):
        nonzero = (U[:, col] != 0).sum()
        assert nonzero == 1


def test_continued_fraction_recovers_period():
    # If φ = 2/4 (a = 2 mod 15 has period 4), measured ≈ 2^n_counting · φ
    # For n_counting = 8, measured ≈ 128. Should recover r = 4.
    r = continued_fraction_period(128, 8, 15)
    assert r in (2, 4)


def test_shor_factors_15():
    # Shor's is probabilistic but for N=15 it succeeds reliably.
    rng = random.Random(42)
    result = shor(15, max_attempts=10, rng=rng)
    assert sorted(result) == [3, 5]


def test_shor_factors_even():
    assert sorted(shor(14)) == [2, 7]
