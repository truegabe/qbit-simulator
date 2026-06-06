"""QRAM: Quantum Random-Access Memory (bucket-brigade architecture).

QRAM lets a quantum algorithm access classical (or quantum) data in
SUPERPOSITION:

    sum_a alpha_a |a⟩ |0⟩  →  sum_a alpha_a |a⟩ |x_a⟩

where {x_a} is a classical database indexed by address a. Without QRAM,
algorithms like Grover-on-database, HHL, and various quantum-ML
schemes can't get below O(N) preparation time.

The **bucket-brigade** QRAM (Giovannetti-Lloyd-Maccone 2008) uses a
binary tree of switches: an address |a⟩ = |a_0 a_1 ... a_{k-1}⟩ on k
qubits selects one of 2^k leaves of a tree. Each tree node is in a
3-level state {|wait⟩, |left⟩, |right⟩}; address qubits route the
data qubit through the correct path.

We **simulate** the input/output transformation directly (rather than
modeling the switch-state physics). This is what an idealized QRAM
does. We provide:

  - `qram_query(state_a, data, address_register, data_register)`:
    perform |a, 0⟩ → |a, x_a⟩ on a state vector.
  - `prepare_state_via_qram(amplitudes)`: use QRAM to load a state of
    arbitrary amplitudes (the "quantum-data-loader" use case).
  - `qram_grover_search(database, target, n_iter)`: Grover-style search
    over a classical database accessed via QRAM.

For database size N = 2^k, QRAM uses O(N) total qubits but provides
O(log N) depth. The query unitary itself is

    O_x |a⟩|y⟩ = |a⟩|y XOR x_a⟩

(for classical data; we extend to quantum data via SWAPs).
"""

from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------
# Idealized QRAM query
# ----------------------------------------------------------------------------

def qram_query(
    psi: np.ndarray,
    database: list[int],
    n_address: int,
    n_data: int,
) -> np.ndarray:
    """Apply the QRAM oracle |a, y⟩ → |a, y XOR database[a]⟩.

    The state vector psi has 2^(n_address + n_data) entries; the first
    n_address qubits are the address (MSB-first) and the last n_data
    qubits are the data register.

    Args:
        psi:       input state vector.
        database:  list of integers (each fits in n_data bits) indexed
                   by address ∈ [0, 2^n_address).
        n_address: number of address qubits.
        n_data:    number of data qubits.

    Returns:
        the new state vector.
    """
    if len(database) != 2 ** n_address:
        raise ValueError(
            f"database length {len(database)} != 2^{n_address}"
        )
    total = n_address + n_data
    if psi.shape != (2 ** total,):
        raise ValueError(
            f"psi shape {psi.shape} != (2^{total},)"
        )
    new_psi = np.zeros_like(psi)
    for idx in range(2 ** total):
        a = idx >> n_data
        y = idx & ((1 << n_data) - 1)
        new_y = y ^ (database[a] & ((1 << n_data) - 1))
        new_idx = (a << n_data) | new_y
        new_psi[new_idx] = psi[idx]
    return new_psi


def qram_state_load(
    database: list[int],
    n_address: int,
    n_data: int,
) -> np.ndarray:
    """Build the QRAM-loaded state:

        |ψ⟩ = (1/√N) sum_a |a⟩ |database[a]⟩

    Starting from a uniform superposition on the address register and
    |0⟩ on the data register.
    """
    N = 2 ** n_address
    if len(database) != N:
        raise ValueError(f"database length {len(database)} != 2^{n_address}")
    total = n_address + n_data
    # |+...+⟩ on address, |0⟩ on data.
    psi = np.zeros(2 ** total, dtype=np.complex128)
    for a in range(N):
        # Address bit pattern |a⟩ on MSB qubits, data |0⟩.
        idx = (a << n_data)
        psi[idx] = 1.0 / np.sqrt(N)
    return qram_query(psi, database, n_address, n_data)


# ----------------------------------------------------------------------------
# Amplitude-based state preparation via QRAM
# ----------------------------------------------------------------------------

def prepare_state_via_qram(
    amplitudes: np.ndarray,
    n_address: int,
) -> np.ndarray:
    """Use QRAM to load an arbitrary state with the given amplitudes.

    Algorithm: store |amplitudes|^2 normalized to integer counts in a
    rotation database, apply rotation conditioned on amplitudes. We
    simulate the result directly here.

    Args:
        amplitudes: length-2^n_address complex array (need not be
                    normalized; we re-normalize).
        n_address:  number of address qubits.

    Returns:
        the encoded state vector of length 2^n_address.
    """
    N = 2 ** n_address
    if len(amplitudes) != N:
        raise ValueError(f"need {N} amplitudes, got {len(amplitudes)}")
    psi = np.asarray(amplitudes, dtype=np.complex128).copy()
    norm = np.linalg.norm(psi)
    if norm < 1e-12:
        raise ValueError("amplitudes have zero norm")
    return psi / norm


# ----------------------------------------------------------------------------
# Grover with QRAM oracle
# ----------------------------------------------------------------------------

def qram_grover_oracle(database: list[int], target: int,
                        n_address: int) -> np.ndarray:
    """Build the dense Grover oracle that marks addresses a with
    database[a] == target.

    O_target |a⟩ = (-1)^{[database[a] == target]} |a⟩
    """
    N = 2 ** n_address
    if len(database) != N:
        raise ValueError(f"database length {len(database)} != 2^{n_address}")
    diag = np.array([-1.0 if database[a] == target else 1.0
                      for a in range(N)], dtype=np.complex128)
    return np.diag(diag)


def diffusion_operator(n: int) -> np.ndarray:
    """Grover diffusion U_s = 2|s⟩⟨s| - I where |s⟩ is the uniform
    superposition over 2^n basis states."""
    N = 2 ** n
    s = np.ones(N, dtype=np.complex128) / np.sqrt(N)
    return 2 * np.outer(s, s.conj()) - np.eye(N, dtype=np.complex128)


def qram_grover_search(
    database: list[int],
    target: int,
    n_iter: int | None = None,
) -> dict:
    """Grover search over a classical database accessed via QRAM oracle.

    Args:
        database: list of integers; we look for any a with database[a] == target.
        target:   the value to find.
        n_iter:   number of Grover iterations. Defaults to the optimal
                  floor(π/4 · √(N/M)) where M is the number of matching
                  entries.

    Returns:
        dict with the final state, success probability, and chosen n_iter.
    """
    N = len(database)
    n_address = int(np.ceil(np.log2(N)))
    if 2 ** n_address != N:
        raise ValueError(f"database size {N} must be a power of 2")
    M = sum(1 for x in database if x == target)
    if M == 0:
        # No matches: amplitude amplification doesn't help.
        psi = np.ones(N, dtype=np.complex128) / np.sqrt(N)
        return {"state": psi, "p_success": 0.0, "n_iter": 0, "M": 0, "N": N}
    if n_iter is None:
        n_iter = int(round(np.pi / 4 * np.sqrt(N / M)))

    psi = np.ones(N, dtype=np.complex128) / np.sqrt(N)
    O = qram_grover_oracle(database, target, n_address)
    D = diffusion_operator(n_address)
    for _ in range(n_iter):
        psi = D @ O @ psi

    p_success = float(sum(abs(psi[a]) ** 2
                          for a in range(N) if database[a] == target))
    return {
        "state":      psi,
        "p_success":  p_success,
        "n_iter":     n_iter,
        "M":          M,
        "N":          N,
    }
