"""Quantum nonlocality games — demonstrations of correlations impossible
under any local hidden-variable theory.

Two classic examples:

**GHZ game** (Mermin 1990, Greenberger-Horne-Zeilinger 1989):
    Three parties (Alice, Bob, Charlie) receive inputs x, y, z ∈ {0, 1}
    with the promise that x ⊕ y ⊕ z = 0. They output bits a, b, c.
    Win condition:  a ⊕ b ⊕ c = x ∨ y ∨ z.

    Classical (local) maximum: 3/4 win rate.
    Quantum (with shared GHZ state and X/Y measurements): 100% win rate.
    Proof: the impossibility of consistent ±1 assignments to all
    measurement outcomes (Kochen-Specker-style algebraic contradiction).

**Mermin-Peres magic square** (Mermin 1990, Peres 1990):
    A 3×3 grid of variables, each ±1. Alice fills a row, Bob fills a
    column. Their cell agrees. The product along each row is +1; along
    each column is +1, except one whose product is -1. CLASSICALLY this
    is impossible to fill: the total parity argument gives a contradiction.

    Quantum: with a shared 2-qubit pair of Bell pairs, Alice and Bob
    can win the game every time. Each cell is a specific 2-qubit Pauli
    observable; the rows/columns of the square form sets of commuting
    observables whose product is ±I.

Both are pedagogical "smoking guns" of quantum nonclassicality, free
from the locality loopholes that complicate CHSH-style Bell tests.
"""

from __future__ import annotations

import numpy as np

from ..circuit import QuantumCircuit


# ----------------------------------------------------------------------------
# GHZ game
# ----------------------------------------------------------------------------

def _measure_in_x_or_y(qc: QuantumCircuit, qubit: int, basis: str,
                       rng: np.random.Generator) -> int:
    """Project qubit onto X or Y basis, return outcome 0 or 1.

    For X basis: apply H, then measure in Z.
    For Y basis: apply S†, then H, then measure in Z.
    """
    if basis == "X":
        qc.h(qubit)
    elif basis == "Y":
        # S† = S^3 (S has order 4). QuantumCircuit doesn't expose sdg directly.
        qc.s(qubit); qc.s(qubit); qc.s(qubit)
        qc.h(qubit)
    else:
        raise ValueError(f"basis must be X or Y, got {basis!r}")
    # Marginal probability of qubit being |0⟩ after the basis change.
    probs = qc.probabilities()
    n = qc.n
    p0 = sum(probs[i] for i in range(2 ** n) if not ((i >> (n - 1 - qubit)) & 1))
    p0 = float(np.clip(p0, 0, 1))
    if rng.uniform() < p0:
        outcome = 0
        # Project: zero out states where qubit q = 1.
        for i in range(2 ** n):
            if (i >> (n - 1 - qubit)) & 1:
                qc.state[i] = 0.0
    else:
        outcome = 1
        for i in range(2 ** n):
            if not ((i >> (n - 1 - qubit)) & 1):
                qc.state[i] = 0.0
    nrm = np.linalg.norm(qc.state)
    if nrm > 1e-12:
        qc.state /= nrm
    return outcome


def ghz_game_single_round(
    inputs: tuple[int, int, int],
    rng: np.random.Generator,
) -> dict:
    """Play one round of the GHZ game with the optimal quantum strategy.

    Quantum strategy:
        Each party measures their qubit in:
            X basis if input bit is 0
            Y basis if input bit is 1

    The win condition (a ⊕ b ⊕ c = x ∨ y ∨ z) is satisfied 100% of the
    time when using a shared 3-qubit GHZ state.

    Inputs satisfy the promise x ⊕ y ⊕ z = 0 (so either all 0 or two 1s).
    """
    x, y, z = inputs
    if (x ^ y ^ z) != 0:
        raise ValueError(f"inputs must satisfy x XOR y XOR z = 0, got {inputs}")
    # Prepare GHZ |Φ⟩ = (|000⟩ + |111⟩) / √2.
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cnot(0, 1)
    qc.cnot(1, 2)
    # Each party measures.
    a = _measure_in_x_or_y(qc, 0, "X" if x == 0 else "Y", rng)
    b = _measure_in_x_or_y(qc, 1, "X" if y == 0 else "Y", rng)
    c = _measure_in_x_or_y(qc, 2, "X" if z == 0 else "Y", rng)
    parity = a ^ b ^ c
    win_condition = (x | y | z)
    return {
        "inputs":   (x, y, z),
        "outputs":  (a, b, c),
        "parity":   parity,
        "win_cond": win_condition,
        "won":      (parity == win_condition),
    }


def ghz_game_full_simulation(
    n_trials_per_input: int = 200,
    rng: np.random.Generator | None = None,
) -> dict:
    """Run the GHZ game across all 4 input combinations.

    Returns the empirical win rate per input and the total win rate.
    """
    rng = rng or np.random.default_rng()
    # Valid inputs (x ⊕ y ⊕ z = 0): (0,0,0), (0,1,1), (1,0,1), (1,1,0).
    valid_inputs = [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
    win_rates: dict[tuple[int, int, int], float] = {}
    total_wins = 0
    total_games = 0
    for inputs in valid_inputs:
        wins = 0
        for _ in range(n_trials_per_input):
            r = ghz_game_single_round(inputs, rng)
            if r["won"]:
                wins += 1
        win_rates[inputs] = wins / n_trials_per_input
        total_wins += wins
        total_games += n_trials_per_input
    return {
        "per_input_win_rates": win_rates,
        "overall_win_rate":    total_wins / total_games,
        "classical_max":       0.75,
    }


# ----------------------------------------------------------------------------
# Mermin-Peres magic square
# ----------------------------------------------------------------------------
#
# The 3×3 grid of 2-qubit Pauli observables:
#
#     | I⊗Z   Z⊗I   Z⊗Z |   row products: +I, +I, +I
#     | X⊗I   I⊗X   X⊗X |
#     | X⊗Z   Z⊗X   Y⊗Y |   column products: +I, +I, +I  but row 3 = -I
#
# Wait actually the standard layout: each row's product = +I, each col's
# product = +I, EXCEPT one (typically the third row or column) whose
# product is -I. The "magic" is that classically you'd need 9 ±1
# assignments respecting these 6 product constraints, which is impossible
# (parity contradiction: rows give +1·+1·+1=+1, cols give -1·... etc).

MAGIC_SQUARE_OBSERVABLES = [
    ["IZ", "ZI", "ZZ"],
    ["XI", "IX", "XX"],
    ["XZ", "ZX", "YY"],
]


def _pauli_matrix(pauli: str) -> np.ndarray:
    """2-qubit Pauli string -> 4x4 matrix."""
    from ..gates import I2, X, Y, Z
    M = {"I": I2, "X": X, "Y": Y, "Z": Z}
    return np.kron(M[pauli[0]], M[pauli[1]])


def verify_magic_square() -> dict:
    """Verify the Mermin-Peres magic square's algebraic structure.

    For the layout above:
      - Each row's product of the three observables equals +I (3 rows).
      - Each column's product equals +I or -I.
      - Total parity: product over all 9 observables = -I (the obstruction).

    This is the algebraic contradiction that makes classical ±1
    assignments impossible.
    """
    rows = []
    cols = []
    for r in range(3):
        prod = np.eye(4, dtype=np.complex128)
        for c in range(3):
            prod = prod @ _pauli_matrix(MAGIC_SQUARE_OBSERVABLES[r][c])
        rows.append(prod)
    for c in range(3):
        prod = np.eye(4, dtype=np.complex128)
        for r in range(3):
            prod = prod @ _pauli_matrix(MAGIC_SQUARE_OBSERVABLES[r][c])
        cols.append(prod)
    # Each row should be ±I; collect signs.
    row_signs = []
    for prod in rows:
        # Compare to ±I.
        if np.allclose(prod, np.eye(4), atol=1e-9):
            row_signs.append(+1)
        elif np.allclose(prod, -np.eye(4), atol=1e-9):
            row_signs.append(-1)
        else:
            row_signs.append(None)
    col_signs = []
    for prod in cols:
        if np.allclose(prod, np.eye(4), atol=1e-9):
            col_signs.append(+1)
        elif np.allclose(prod, -np.eye(4), atol=1e-9):
            col_signs.append(-1)
        else:
            col_signs.append(None)
    return {
        "row_signs":  row_signs,
        "col_signs":  col_signs,
        "row_product_overall":  int(np.prod(row_signs)) if all(s is not None for s in row_signs) else None,
        "col_product_overall":  int(np.prod(col_signs)) if all(s is not None for s in col_signs) else None,
    }


def magic_square_classical_max_win_rate() -> float:
    """The maximum win rate under any classical (local hidden-variable)
    strategy is 8/9 ≈ 0.889 (one cell must always be wrong by parity).

    Quantum strategy achieves 1.0 deterministically.
    """
    return 8.0 / 9.0
