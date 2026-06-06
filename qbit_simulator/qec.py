"""Quantum error correcting codes — Clifford-only implementations.

Each code is built as:
  - An encoding circuit  (logical → physical qubits)
  - A list of stabilizer generators
  - A syndrome-measurement circuit (extracts the error syndrome to ancillas)
  - A classical decoder (syndrome → correction Pauli)

All codes implemented here are CSS codes (or close enough) and require only
Clifford gates, so they run end-to-end on the StabilizerState simulator.

Codes provided:
  - three_qubit_repetition_code: detects 1 bit flip
  - five_qubit_perfect_code:     [[5,1,3]] -- smallest single-error-correcting
  - steane_code:                 [[7,1,3]] -- textbook CSS code
  - shor_nine_qubit_code:        [[9,1,3]] -- original Shor code, concatenated
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .stabilizer import StabilizerState


# ----------------------------------------------------------------------------
# Common error-correction interface
# ----------------------------------------------------------------------------

@dataclass
class QECCode:
    """Bundle describing one quantum error-correcting code.

    Attributes:
        name:            human-readable code name
        n_data:          number of physical data qubits
        n_logical:       number of logical qubits encoded
        distance:        code distance (min weight of a logical operator)
        x_stabilizers:   list of Pauli strings (length n_data) of X-type stabilizers
        z_stabilizers:   list of Pauli strings (length n_data) of Z-type stabilizers
        logical_x:       Pauli string for the logical X operator
        logical_z:       Pauli string for the logical Z operator
        encode:          callable (state, data_qubits, ancilla_qubits) → None
                         that applies the encoding circuit in place
    """
    name: str
    n_data: int
    n_logical: int
    distance: int
    x_stabilizers: list[str]
    z_stabilizers: list[str]
    logical_x: str
    logical_z: str
    encode: callable


# ----------------------------------------------------------------------------
# 3-qubit repetition code (corrects 1 bit-flip; not phase)
# ----------------------------------------------------------------------------

def _encode_repetition(state: StabilizerState, q0: int, q1: int, q2: int) -> None:
    """|ψ⟩ on q0, |0⟩ on q1, q2 → encoded state on (q0, q1, q2)."""
    state.cnot(q0, q1)
    state.cnot(q0, q2)


three_qubit_repetition_code = QECCode(
    name="3-qubit repetition",
    n_data=3, n_logical=1, distance=3,    # detects 1 X-error; doesn't catch Z
    x_stabilizers=[],
    z_stabilizers=["ZZI", "IZZ"],
    logical_x="XXX",
    logical_z="ZII",
    encode=_encode_repetition,
)


# ----------------------------------------------------------------------------
# 5-qubit perfect code [[5,1,3]]
# ----------------------------------------------------------------------------
#
# Generators (a standard cyclic choice):
#     XZZXI
#     IXZZX
#     XIXZZ
#     ZXIXZ
# Logical X = XXXXX, Logical Z = ZZZZZ. Distance 3.
#
# Encoding circuit (one of many): see Laflamme-Miquel-Paz-Zurek 1996.
# We implement a small encoding that maps |ψ⟩ |0000⟩ → |ψ_L⟩ for ψ ∈ {|0⟩,|1⟩}.

def _encode_five_qubit(state: StabilizerState, *qubits: int) -> None:
    """Encode |0_L⟩ for the [[5,1,3]] code via stabilizer projection.

    The 5-qubit code is not a CSS code (its stabilizers mix X and Z), so a
    direct CSS-style encoding circuit doesn't exist. We use the generic
    measurement-based encoding (`prepare_codespace_state`): starting from
    |0⟩^⊗5, project onto the +1 eigenspace of each of the four stabilizers
    in turn, using one ancilla qubit as scratch.

    Needs `n_data + 1 = 6` qubits in the state (5 data + 1 ancilla).
    """
    if len(qubits) != 5:
        raise ValueError("5-qubit code needs exactly 5 data qubits")
    prepare_codespace_state(state, five_qubit_perfect_code, list(qubits))


five_qubit_perfect_code = QECCode(
    name="5-qubit perfect code [[5,1,3]]",
    n_data=5, n_logical=1, distance=3,
    x_stabilizers=[],  # mixed stabilizers — see below
    z_stabilizers=[],
    logical_x="XXXXX",
    logical_z="ZZZZZ",
    encode=_encode_five_qubit,
)
# Mixed-type stabilizers (each generator contains both X and Z) stored
# in a separate field for the 5-qubit code.
five_qubit_perfect_code.stabilizers = [
    "XZZXI", "IXZZX", "XIXZZ", "ZXIXZ",
]


# ----------------------------------------------------------------------------
# Steane code [[7,1,3]] — the textbook CSS code
# ----------------------------------------------------------------------------
#
# Parity-check matrix of the classical [7,4,3] Hamming code:
#     H = [[0,0,0,1,1,1,1],
#          [0,1,1,0,0,1,1],
#          [1,0,1,0,1,0,1]]
# Z-stabilizers: rows of H, with 1 → Z, 0 → I.
# X-stabilizers: rows of H, with 1 → X, 0 → I.
# Logical X = XXXXXXX, Logical Z = ZZZZZZZ.
#
# Steane is CSS, distance 3, corrects any single-qubit Pauli error.

_HAMMING_H = np.array([
    [0, 0, 0, 1, 1, 1, 1],
    [0, 1, 1, 0, 0, 1, 1],
    [1, 0, 1, 0, 1, 0, 1],
], dtype=np.int8)


def _hamming_row_to_pauli(row: np.ndarray, op: str) -> str:
    return "".join(op if b else "I" for b in row)


def _encode_steane_zero(state: StabilizerState, qs: list[int]) -> None:
    """Encode |0_L⟩ for Steane (assumes input |0⟩^⊗7).

    The encoded |0_L⟩ is a uniform superposition over the 8 codewords of the
    [7,3,4] dual code spanned by the rows of the Hamming parity-check matrix
    H. Encoding strategy:
       row 0 = 0001111  (leading qubit 3 -> propagate to {4, 5, 6})
       row 1 = 0110011  (leading qubit 1 -> propagate to {2, 5, 6})
       row 2 = 1010101  (leading qubit 0 -> propagate to {2, 4, 6})
    Step 1: H on each leading qubit creates a 3-bit superposition.
    Step 2: CNOTs from the leading qubit propagate the codeword structure.
    """
    # Leading qubits (one per X-stabilizer row).
    state.h(qs[0])
    state.h(qs[1])
    state.h(qs[3])
    # Row 2 (1010101): control = qs[0], targets {2, 4, 6}.
    state.cnot(qs[0], qs[2])
    state.cnot(qs[0], qs[4])
    state.cnot(qs[0], qs[6])
    # Row 1 (0110011): control = qs[1], targets {2, 5, 6}.
    state.cnot(qs[1], qs[2])
    state.cnot(qs[1], qs[5])
    state.cnot(qs[1], qs[6])
    # Row 0 (0001111): control = qs[3], targets {4, 5, 6}.
    state.cnot(qs[3], qs[4])
    state.cnot(qs[3], qs[5])
    state.cnot(qs[3], qs[6])


def _encode_steane(state: StabilizerState,
                    q0: int, q1: int, q2: int, q3: int,
                    q4: int, q5: int, q6: int) -> None:
    """Encode |0_L⟩ on the seven qubits (assumes input is |0⟩^⊗7)."""
    _encode_steane_zero(state, [q0, q1, q2, q3, q4, q5, q6])


steane_code = QECCode(
    name="Steane code [[7,1,3]]",
    n_data=7, n_logical=1, distance=3,
    x_stabilizers=[_hamming_row_to_pauli(row, "X") for row in _HAMMING_H],
    z_stabilizers=[_hamming_row_to_pauli(row, "Z") for row in _HAMMING_H],
    logical_x="XXXXXXX",
    logical_z="ZZZZZZZ",
    encode=_encode_steane,
)


# ----------------------------------------------------------------------------
# 9-qubit Shor code [[9,1,3]]
# ----------------------------------------------------------------------------
#
# Concatenated: 3-qubit phase-flip code (outer) on top of 3-qubit bit-flip
# code (inner). Logical |0⟩ = (1/√8) (|000⟩+|111⟩)(|000⟩+|111⟩)(|000⟩+|111⟩).
# 8 stabilizers: 6 Z-type (intra-block ZZI/IZZ in each of 3 blocks)
# + 2 X-type (XXXXXX III and III XXXXXX, comparing blocks).

def _encode_shor_nine(state: StabilizerState, qs: list[int]) -> None:
    """Encode |0_L⟩ on 9 qubits (assumes |0⟩^⊗9 input).

    |0_L⟩ = (1/√8)·[(|000⟩+|111⟩) ⊗ (|000⟩+|111⟩) ⊗ (|000⟩+|111⟩)]
    where the three (|000⟩+|111⟩) factors live on blocks (0,1,2), (3,4,5),
    (6,7,8) respectively. This is automatically the +1 eigenstate of every
    Shor stabilizer (intra-block ZZ's and inter-block X^6's).
    """
    if len(qs) != 9:
        raise ValueError("Shor code needs exactly 9 qubits")
    # Independently encode (|000⟩+|111⟩)/√2 in each of the three blocks.
    for block_start in (0, 3, 6):
        state.h(qs[block_start])
        state.cnot(qs[block_start], qs[block_start + 1])
        state.cnot(qs[block_start], qs[block_start + 2])


def _shor_stabs():
    """List the 8 stabilizer generators of the 9-qubit Shor code."""
    # Z-type: intra-block ZZ on (0,1), (1,2), (3,4), (4,5), (6,7), (7,8).
    z_stabs = []
    for block_start in (0, 3, 6):
        for offset in (0, 1):
            s = ["I"] * 9
            s[block_start + offset] = "Z"
            s[block_start + offset + 1] = "Z"
            z_stabs.append("".join(s))
    # X-type: XXXXXX III  and  III XXXXXX.
    s1 = ["X"] * 6 + ["I"] * 3
    s2 = ["I"] * 3 + ["X"] * 6
    x_stabs = ["".join(s1), "".join(s2)]
    return x_stabs, z_stabs


_shor_x, _shor_z = _shor_stabs()

shor_nine_qubit_code = QECCode(
    name="Shor 9-qubit code [[9,1,3]]",
    n_data=9, n_logical=1, distance=3,
    x_stabilizers=_shor_x,
    z_stabilizers=_shor_z,
    logical_x="XXXXXXXXX",     # actually X on any one of 3 outer-blocks works
    logical_z="ZZZZZZZZZ",     # any one Z per block
    encode=lambda state, *qs: _encode_shor_nine(state, list(qs)),
)


# ----------------------------------------------------------------------------
# Surface code patch, distance 3 [[9,1,3]]
# ----------------------------------------------------------------------------
#
# A 3x3 grid of data qubits with 4 X-type + 4 Z-type stabilizers in a
# rotated-surface-code-style layout. This patch encodes 1 logical qubit at
# distance 3 (correcting any single-qubit Pauli error).
#
# Layout:
#       0  1  2
#       3  4  5
#       6  7  8
#
# Z-stabilizers (4):
#       Z_0 Z_1 Z_3 Z_4   (top-left plaquette)
#       Z_4 Z_5 Z_7 Z_8   (bottom-right plaquette)
#       Z_1 Z_2           (top boundary, weight 2)
#       Z_6 Z_7           (bottom boundary, weight 2)
#
# X-stabilizers (4):
#       X_0 X_3           (left boundary, weight 2)
#       X_5 X_8           (right boundary, weight 2)
#       X_3 X_4 X_6 X_7   (bottom-left plaquette)
#       X_1 X_2 X_4 X_5   (top-right plaquette)
#
# Logical X = X_0 X_4 X_8 (main diagonal, weight 3)
# Logical Z = Z_2 Z_4 Z_6 (anti-diagonal, weight 3)
#
# All 8 stabilizers commute pairwise (verified at module import below).

_SURFACE_3_Z = ["ZZIZZIIII", "IIIIZZIZZ", "IZZIIIIII", "IIIIIIZZI"]
_SURFACE_3_X = ["XIIXIIIII", "IIIIIXIIX", "IIIXXIXXI", "IXXIXXIII"]


def _verify_surface_3_commutation():
    """Sanity check that all surface-code stabilizers commute pairwise."""
    stabs = _SURFACE_3_Z + _SURFACE_3_X
    for i, a in enumerate(stabs):
        for b in stabs[i + 1:]:
            # Pauli strings commute iff the number of positions where one has
            # X and the other has Z (mod transverse) is even.
            anti = 0
            for pa, pb in zip(a, b):
                ax = pa in "XY"; az = pa in "ZY"
                bx = pb in "XY"; bz = pb in "ZY"
                anti ^= (ax & bz) ^ (az & bx)
            if anti:
                raise RuntimeError(f"surface code stabilizers do not commute: {a} vs {b}")


_verify_surface_3_commutation()


surface_code_d3 = QECCode(
    name="Surface code patch d=3 [[9,1,3]]",
    n_data=9, n_logical=1, distance=3,
    x_stabilizers=_SURFACE_3_X,
    z_stabilizers=_SURFACE_3_Z,
    logical_x="XIIIXIIIX",
    logical_z="IIZIZIZII",
    encode=lambda *args, **kwargs: (_ for _ in ()).throw(
        NotImplementedError("surface_code_d3 encoding circuit not implemented; "
                             "use stabilizer structure for error-detection demos")
    ),
)


# ----------------------------------------------------------------------------
# Generic helpers: verify a state is in the codespace
# ----------------------------------------------------------------------------

def prepare_codespace_state(state: StabilizerState, code: QECCode,
                             data_qubits: list[int],
                             rng=None) -> None:
    """Project the data qubits onto the codespace of `code` via stabilizer
    projection.

    For each stabilizer S of the code, we measure S as a single multi-qubit
    Pauli observable: this is done by allocating one ancilla, entangling it
    with the data qubits using H + controlled Paulis, measuring the ancilla,
    and forcing the +1 eigenvalue outcome (apply a correction if the
    measurement returns -1).

    This is a measurement-based encoding: it works for any stabilizer code
    without needing a custom encoding circuit. The trade-off is that we may
    need one ancilla qubit per stabilizer (or reused).
    """
    if len(data_qubits) != code.n_data:
        raise ValueError(f"got {len(data_qubits)} data qubits, "
                         f"code uses {code.n_data}")
    import numpy as np
    rng = rng or np.random.default_rng()
    stabs = (list(code.x_stabilizers) + list(code.z_stabilizers)
             + list(getattr(code, "stabilizers", [])))
    for s in stabs:
        # Pick an ancilla. We need one fresh qubit per stabilizer; the
        # simplest contract is to require the caller to provide enough.
        # In practice we reuse the LAST available qubit in `state` as scratch.
        if state.n <= max(data_qubits):
            raise ValueError("state must have at least max(data_qubits)+2 qubits "
                             "(one extra for ancilla scratch)")
        ancilla = state.n - 1
        # Project onto +1 eigenspace of s by:
        #   1. Put ancilla in |+⟩
        #   2. For each non-I Pauli P_q in s, apply controlled-P_q from ancilla to q
        #   3. H on ancilla
        #   4. Measure ancilla; if outcome 1, apply s to data qubits (correction)
        state.h(ancilla)
        for q_local, ch in enumerate(s):
            q = data_qubits[q_local]
            if ch == "I":
                continue
            if ch == "X":
                state.cnot(ancilla, q)
            elif ch == "Z":
                state.cz(ancilla, q)
            elif ch == "Y":
                # CY = (I⊗S†) · CX · (I⊗S). Apply S first, then CX, then S†.
                state.s(q)
                state.cnot(ancilla, q)
                state.sdg(q)
        state.h(ancilla)
        # Force the measurement outcome to 0 (i.e. select the +1 eigenspace
        # of the stabilizer). For a random measurement, force is honored and
        # the state is projected onto |0⟩_ancilla, which corresponds to the
        # +1 eigenvalue of S on the data qubits.
        state.measure(ancilla, rng=rng, force=0)


def measure_syndromes(state: StabilizerState, code: QECCode,
                       data_qubits: list[int]) -> dict[str, int]:
    """Measure each stabilizer's eigenvalue, returning {stabilizer: 0_or_1}.

    A 0 means +1 eigenvalue (no error detected by this stabilizer); 1 means
    -1 eigenvalue (this stabilizer flagged an error).
    """
    out: dict[str, int] = {}
    n_total = state.n
    stabs = (list(getattr(code, "stabilizers", []))
             + list(code.x_stabilizers) + list(code.z_stabilizers))
    for s in stabs:
        full = ["I"] * n_total
        for local_q, ch in enumerate(s):
            full[data_qubits[local_q]] = ch
        exp = state.pauli_expectation("".join(full))
        out[s] = 0 if exp == 1 else 1
    return out


def build_decoder_table(code: QECCode) -> dict[tuple, str]:
    """Build a lookup decoder: syndrome tuple → correction Pauli string.

    Enumerates all single-qubit Pauli errors (X, Y, Z on each data qubit),
    computes the syndrome each would produce (by checking commutation with
    each stabilizer), and tabulates a correction = same Pauli that caused it.

    Caveat: only handles single-qubit errors. A real surface-code decoder
    uses minimum-weight perfect matching to handle multi-error patterns.
    """
    n = code.n_data
    stabs = (list(getattr(code, "stabilizers", []))
             + list(code.x_stabilizers) + list(code.z_stabilizers))
    table: dict[tuple, str] = {tuple([0] * len(stabs)): "I" * n}    # no error
    for q in range(n):
        for err_type in ("X", "Y", "Z"):
            err = ["I"] * n
            err[q] = err_type
            err_str = "".join(err)
            # Compute syndrome: for each stabilizer, anticommutation gives 1.
            syndrome = []
            for s in stabs:
                anti = 0
                for pe, ps in zip(err_str, s):
                    ex = pe in "XY"; ez = pe in "ZY"
                    sx = ps in "XY"; sz = ps in "ZY"
                    anti ^= (ex & sz) ^ (ez & sx)
                syndrome.append(anti)
            key = tuple(syndrome)
            if key not in table:
                table[key] = err_str
    return table


def decode_and_correct(state: StabilizerState, code: QECCode,
                        data_qubits: list[int],
                        decoder_table: dict[tuple, str]) -> str:
    """Measure syndromes, look up the correction, apply it. Returns the
    correction Pauli string that was applied (or "I"*n if no correction)."""
    syndromes = measure_syndromes(state, code, data_qubits)
    stabs = (list(getattr(code, "stabilizers", []))
             + list(code.x_stabilizers) + list(code.z_stabilizers))
    syndrome_tuple = tuple(syndromes[s] for s in stabs)
    correction = decoder_table.get(syndrome_tuple, "I" * code.n_data)
    for q_local, ch in enumerate(correction):
        q = data_qubits[q_local]
        if ch == "X":
            state.x(q)
        elif ch == "Y":
            state.y(q)
        elif ch == "Z":
            state.z(q)
    return correction


def verify_codespace(state: StabilizerState, code: QECCode,
                     data_qubits: list[int]) -> dict:
    """Check that the given qubits hold a +1 eigenstate of every stabilizer
    of `code`. Returns a dict of {stabilizer_str: expectation_value}.

    For a correctly-encoded state, every value should be +1. After an error,
    some values become -1 (indicating the syndrome that detected the error).
    """
    if len(data_qubits) != code.n_data:
        raise ValueError(f"code uses {code.n_data} qubits, got {len(data_qubits)}")
    out: dict[str, int] = {}
    n_total = state.n
    # Embed each stabilizer string into the full N-qubit state.
    stabs = (code.x_stabilizers + code.z_stabilizers
             + getattr(code, "stabilizers", []))
    for s in stabs:
        full = ["I"] * n_total
        for local_q, ch in enumerate(s):
            full[data_qubits[local_q]] = ch
        out[s] = state.pauli_expectation("".join(full))
    return out
