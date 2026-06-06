"""Toric code: the original topological quantum code (Kitaev 2003).

The toric code lives on a 2D L × L lattice with PERIODIC boundary
conditions. Qubits live on the EDGES (so 2L² qubits for an L × L
lattice). Stabilizers come in two flavors:

  * **Vertex (star) operators** A_v = ⊗ X on the 4 edges meeting at
    vertex v. Each detects Z-type errors.
  * **Plaquette operators** B_p = ⊗ Z on the 4 edges around plaquette
    p. Each detects X-type errors.

There are L² vertices and L² plaquettes, but their products satisfy
  prod_v A_v = I,    prod_p B_p = I
(periodic boundary makes these "trivial"). So 2L² stabilizers minus 2
constraints = 2L² − 2 independent stabilizers, leaving 2L² − (2L²−2) = 2
logical qubits. These two logical qubits are the topological "modes"
of the torus.

Logical operators are NONCONTRACTIBLE Wilson loops:

  * X-logical-1: horizontal X-string around the torus (loop on the
    primal lattice).
  * X-logical-2: vertical X-string around the torus.
  * Z-logicals: analogous Z-strings on the DUAL lattice.

Each pair anti-commutes; X-logicals commute with all stabilizers (they
are non-contractible loops, not boundaries of plaquettes).

This module provides:

  - `ToricCode(L)`: the lattice + stabilizer assignment.
  - `vertex_neighbors(v, L)`, `plaquette_edges(p, L)`: combinatorial
    helpers.
  - `syndrome(error_set, code)`: which stabilizers fire.
  - `logical_x_loops(L)`, `logical_z_loops(L)`: the 2 non-contractible
    paths along which X / Z logicals live.
  - `count_independent_stabilizers(L)`: returns 2L²-2 (sanity check).

For small L (≤ 4) we work with explicit operator sets; the full
Hilbert space is too large for direct state-vector simulation.
"""

from __future__ import annotations

from dataclasses import dataclass


# Edge labels on an L × L torus:
#   Each unit cell at (r, c) has TWO edges:
#     - horizontal: connects (r, c) → (r, c+1 mod L), index = "h", (r, c).
#     - vertical:   connects (r, c) → (r+1 mod L, c), index = "v", (r, c).
#   We number them as: (r, c, "h") gets id = 2*L*r + 2*c, vertical = +1.
# Total edges = 2 L²


def edge_id(r: int, c: int, orient: str, L: int) -> int:
    """Map (row, col, orientation) to a unique qubit index in [0, 2L²)."""
    r = r % L
    c = c % L
    base = 2 * L * r + 2 * c
    if orient == "h":
        return base
    elif orient == "v":
        return base + 1
    raise ValueError(f"orientation must be 'h' or 'v', got {orient}")


@dataclass
class ToricCode:
    """Toric code on an L × L lattice.

    Attributes:
        L:              linear size.
        n_qubits:       total edges (= 2 L²).
        vertex_stabs:   list of length L² where each entry is a tuple
                        of 4 edge indices for A_v = X_{e₁}X_{e₂}X_{e₃}X_{e₄}.
        plaquette_stabs: same shape, for B_p = Z_{e₁}Z_{e₂}Z_{e₃}Z_{e₄}.
        logical_x:      list of 2 X-logicals (horizontal & vertical loops).
        logical_z:      list of 2 Z-logicals.
    """
    L: int
    n_qubits: int
    vertex_stabs: list[tuple[int, int, int, int]]
    plaquette_stabs: list[tuple[int, int, int, int]]
    logical_x: list[tuple[int, ...]]
    logical_z: list[tuple[int, ...]]


def vertex_neighbors(v_r: int, v_c: int, L: int) -> tuple[int, int, int, int]:
    """The 4 edges meeting at vertex (v_r, v_c):
        - left edge:   horizontal at (v_r, v_c - 1)
        - right edge:  horizontal at (v_r, v_c)
        - up edge:     vertical at (v_r - 1, v_c)
        - down edge:   vertical at (v_r, v_c)
    """
    return (
        edge_id(v_r, v_c - 1, "h", L),
        edge_id(v_r, v_c, "h", L),
        edge_id(v_r - 1, v_c, "v", L),
        edge_id(v_r, v_c, "v", L),
    )


def plaquette_edges(p_r: int, p_c: int, L: int) -> tuple[int, int, int, int]:
    """The 4 edges around plaquette (p_r, p_c). Plaquette (p_r, p_c) is
    between rows p_r and p_r+1 and columns p_c and p_c+1.

    Edges:
        top:    horizontal at (p_r, p_c)
        bottom: horizontal at (p_r + 1, p_c)
        left:   vertical at (p_r, p_c)
        right:  vertical at (p_r, p_c + 1)
    """
    return (
        edge_id(p_r, p_c, "h", L),
        edge_id(p_r + 1, p_c, "h", L),
        edge_id(p_r, p_c, "v", L),
        edge_id(p_r, p_c + 1, "v", L),
    )


def build_toric_code(L: int) -> ToricCode:
    """Construct a toric code on an L × L lattice (L ≥ 2)."""
    if L < 2:
        raise ValueError(f"L must be ≥ 2, got {L}")
    n_qubits = 2 * L * L

    vertex_stabs = [
        vertex_neighbors(r, c, L) for r in range(L) for c in range(L)
    ]
    plaquette_stabs = [
        plaquette_edges(r, c, L) for r in range(L) for c in range(L)
    ]

    # Logical operators on a torus: closed loops winding the torus.
    # An X-loop must enter+exit each plaquette evenly → it's a dual-lattice
    # cycle, which lives on edges PERPENDICULAR to its travel direction.
    # A horizontal X-loop (winding around the "horizontal" direction on
    # the dual lattice) crosses one VERTICAL edge per column → X on
    # vertical edges of a row.
    logical_x1 = tuple(edge_id(0, c, "v", L) for c in range(L))   # row-0 verticals
    logical_x2 = tuple(edge_id(r, 0, "h", L) for r in range(L))   # col-0 horizontals
    # A Z-loop must enter+exit each vertex evenly → it's a primal-lattice
    # cycle. Z-1 wraps vertically (column 0): Z on column-0 verticals.
    # Z-2 wraps horizontally (row 0): Z on row-0 horizontals.
    logical_z1 = tuple(edge_id(r, 0, "v", L) for r in range(L))   # col-0 verticals
    logical_z2 = tuple(edge_id(0, c, "h", L) for c in range(L))   # row-0 horizontals

    return ToricCode(
        L=L,
        n_qubits=n_qubits,
        vertex_stabs=vertex_stabs,
        plaquette_stabs=plaquette_stabs,
        logical_x=[logical_x1, logical_x2],
        logical_z=[logical_z1, logical_z2],
    )


# ----------------------------------------------------------------------------
# Syndrome
# ----------------------------------------------------------------------------

def syndrome(
    code: ToricCode,
    x_errors: set[int], z_errors: set[int],
) -> dict:
    """Compute which stabilizers fire under the given error pattern.

    - Vertex stabilizers A_v = X⊗4 detect **Z** errors.
    - Plaquette stabilizers B_p = Z⊗4 detect **X** errors.

    Returns:
        dict with "vertex" (list of bool indicating each A_v's outcome)
        and "plaquette" similarly for B_p.
    """
    vertex_syndrome = []
    for stab in code.vertex_stabs:
        # A_v anticommutes with Z on any of its 4 edges → flips if odd
        # number of Z errors on its support.
        parity = sum(1 for e in stab if e in z_errors) % 2
        vertex_syndrome.append(bool(parity))
    plaquette_syndrome = []
    for stab in code.plaquette_stabs:
        parity = sum(1 for e in stab if e in x_errors) % 2
        plaquette_syndrome.append(bool(parity))
    return {
        "vertex":    vertex_syndrome,
        "plaquette": plaquette_syndrome,
    }


# ----------------------------------------------------------------------------
# Logical-error detection
# ----------------------------------------------------------------------------

def logical_error(code: ToricCode,
                    x_errors: set[int], z_errors: set[int]) -> dict:
    """For a residual error pattern (after applying a correction), check
    whether any logical operator was applied.

    Returns:
        dict with logical_x[i] = 0/1 for each of the 2 logical X-operators
        (flipped iff Z-error anticommutes with X-string), and similarly
        for logical_z.
    """
    out = {"logical_x": [], "logical_z": []}
    for x_str in code.logical_x:
        # X-string flips iff residual Z-errors have odd overlap with x_str.
        flipped = sum(1 for e in x_str if e in z_errors) % 2
        out["logical_x"].append(int(flipped))
    for z_str in code.logical_z:
        flipped = sum(1 for e in z_str if e in x_errors) % 2
        out["logical_z"].append(int(flipped))
    return out


# ----------------------------------------------------------------------------
# Sanity counters
# ----------------------------------------------------------------------------

def count_independent_stabilizers(L: int) -> int:
    """Number of independent stabilizers = 2 L² − 2.

    (L² vertex + L² plaquette − 2 product constraints.)
    """
    return 2 * L * L - 2


def n_logical_qubits(L: int) -> int:
    """Number of logical qubits = n_qubits − n_indep_stabs = 2."""
    return 2


# ----------------------------------------------------------------------------
# Stabilizer commutation verification
# ----------------------------------------------------------------------------

def stabilizers_commute(code: ToricCode) -> bool:
    """Sanity check: every A_v commutes with every B_p.

    A_v = X⊗4 on 4 edges; B_p = Z⊗4 on 4 edges. They anticommute iff
    they share an ODD number of edges. Since each vertex of a plaquette
    has 2 incident edges of the plaquette, the share count is always
    even — so all A_v and B_p commute.
    """
    for A in code.vertex_stabs:
        A_set = set(A)
        for B in code.plaquette_stabs:
            share = len(A_set & set(B))
            if share % 2 != 0:
                return False
    return True


def logicals_commute_with_stabilizers(code: ToricCode) -> bool:
    """Verify the logical operators commute with all stabilizers."""
    for x_log in code.logical_x:
        x_set = set(x_log)
        for stab in code.plaquette_stabs:    # B_p has Z's; X⊗Z anticommutes
            share = len(x_set & set(stab))
            if share % 2 != 0:
                return False
    for z_log in code.logical_z:
        z_set = set(z_log)
        for stab in code.vertex_stabs:
            share = len(z_set & set(stab))
            if share % 2 != 0:
                return False
    return True


def logical_x_z_anticommute(code: ToricCode) -> bool:
    """X-logical-i and Z-logical-i should anti-commute (one pair each)
    while X-logical-i and Z-logical-j (i ≠ j) commute."""
    for i, x_log in enumerate(code.logical_x):
        for j, z_log in enumerate(code.logical_z):
            share = len(set(x_log) & set(z_log)) % 2
            should_anticommute = (i == j)
            if share != int(should_anticommute):
                return False
    return True
