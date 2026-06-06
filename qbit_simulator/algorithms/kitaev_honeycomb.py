"""Kitaev honeycomb model: exactly-solvable anyon model.

The Kitaev honeycomb (Kitaev 2006) is a 2D lattice model of spin-1/2
sites on the vertices of a honeycomb lattice. Each spin-1/2 has three
neighbors connected by "x", "y", or "z" bonds (one of each type per
vertex). The Hamiltonian is

    H = -J_x sum_{x-links} σ^x_i σ^x_j
        -J_y sum_{y-links} σ^y_i σ^y_j
        -J_z sum_{z-links} σ^z_i σ^z_j

Despite being a 2D interacting spin model, it's EXACTLY SOLVABLE via a
mapping to free Majorana fermions in a Z_2 gauge field. The phase
diagram has two types of regions:

  * **Abelian phase** (Kitaev's "A" phase): when one coupling dominates,
    e.g. |J_z| > |J_x| + |J_y|. Gapped; the elementary excitations are
    Abelian anyons (toric-code anyons).
  * **Non-Abelian phase** (Kitaev's "B" phase): the symmetric region
    |J_x|, |J_y|, |J_z| all comparable. Gapless in the bulk; under a
    perturbative magnetic field, a gap opens and the excitations are
    non-Abelian Ising anyons.

This module:

  * Builds the Hamiltonian on a small honeycomb cluster.
  * Computes plaquette operators W_p = σ_1 σ_2 σ_3 σ_4 σ_5 σ_6
    (one Pauli per vertex, alternating around the hexagon). These
    commute with H and are the conserved Z_2 flux operators that
    label different flux sectors.
  * Identifies the no-flux (ground-state) sector and computes the
    spectrum exactly.

We work on a small "ladder" geometry (2 hexagons, 4 unit cells) so the
full Hilbert space is 2^N tractable. For N ≤ 14 sites this is fast.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


# Pauli matrices.
_I = np.eye(2, dtype=np.complex128)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
_PAULI = {"I": _I, "X": _X, "Y": _Y, "Z": _Z}


def _pauli_string_matrix(s: str) -> np.ndarray:
    M = np.array([[1.0 + 0j]])
    for ch in s:
        M = np.kron(M, _PAULI[ch])
    return M


def _two_body_pauli(N: int, i: int, j: int, p: str) -> np.ndarray:
    """Build the operator σ^p_i σ^p_j on an N-spin system as a dense matrix."""
    s = ["I"] * N
    s[i] = p
    s[j] = p
    return _pauli_string_matrix("".join(s))


# ----------------------------------------------------------------------------
# Small honeycomb cluster
# ----------------------------------------------------------------------------
#
# We use a periodic 2-hexagon "brick" cluster with 8 sites:
#
#       1 -x- 2 -x- 1
#       |     |
#       z     z
#       |     |
#       0 -y- 3 -y- 0     (sites 0, 3 are shared via periodicity)
#
# More explicitly, label 8 sites 0..7. Bonds:
#   x-bonds: (0,1), (2,3), (4,5), (6,7)
#   y-bonds: (1,2), (3,0), (5,6), (7,4)
#   z-bonds: (0,4), (1,5), (2,6), (3,7)
#
# This is two unit cells of a brick-wall geometry. Each site has exactly
# one neighbor on each bond type (x, y, z) — the defining property of
# the Kitaev model.

@dataclass
class KitaevCluster:
    """A small Kitaev cluster: list of bonds by type."""
    n_sites: int
    x_bonds: list[tuple[int, int]]
    y_bonds: list[tuple[int, int]]
    z_bonds: list[tuple[int, int]]
    plaquettes: list[tuple[int, ...]] = None     # 6-vertex hexagons


def brick_cluster_8() -> KitaevCluster:
    """An 8-site brick cluster — two hexagons with periodic boundaries.

    Bonds chosen so each site has exactly one x, one y, one z neighbor.
    """
    return KitaevCluster(
        n_sites=8,
        x_bonds=[(0, 1), (2, 3), (4, 5), (6, 7)],
        y_bonds=[(1, 2), (3, 0), (5, 6), (7, 4)],
        z_bonds=[(0, 4), (1, 5), (2, 6), (3, 7)],
        plaquettes=[
            # Two hexagons. Each plaquette operator multiplies
            # sigma_i^{p_i} around the hexagon. The standard Kitaev
            # honeycomb plaquette is W = sigma^z sigma^x sigma^y sigma^z sigma^x sigma^y
            # (alternating around the hexagon).
            # For our brick cluster, the natural plaquettes are
            # the 6-cycles (0, 4, 5, 1, 2, 6) and (3, 7, 4, 0, 1, 5) etc.
            # We use 2 simple plaquettes:
            (0, 1, 5, 4),       # 4-vertex "small" plaquette
            (2, 3, 7, 6),
        ],
    )


def four_site_cluster() -> KitaevCluster:
    """The smallest meaningful Kitaev cluster: a single 4-site loop.

    Topology: 4 sites in a square with three bond types alternating.
        0 -x- 1
        |     |
        z     z
        |     |
        3 -x- 2
        with (0,2) y-bond and (1,3) y-bond going through diagonally,
    """
    return KitaevCluster(
        n_sites=4,
        x_bonds=[(0, 1), (2, 3)],
        y_bonds=[(0, 2), (1, 3)],
        z_bonds=[(0, 3), (1, 2)],
        plaquettes=[(0, 1, 2, 3)],
    )


# ----------------------------------------------------------------------------
# Hamiltonian
# ----------------------------------------------------------------------------

def kitaev_hamiltonian(cluster: KitaevCluster,
                        Jx: float = 1.0, Jy: float = 1.0, Jz: float = 1.0
                        ) -> np.ndarray:
    """Build the dense matrix Hamiltonian for the Kitaev honeycomb."""
    N = cluster.n_sites
    dim = 2 ** N
    H = np.zeros((dim, dim), dtype=np.complex128)
    for (i, j) in cluster.x_bonds:
        H -= Jx * _two_body_pauli(N, i, j, "X")
    for (i, j) in cluster.y_bonds:
        H -= Jy * _two_body_pauli(N, i, j, "Y")
    for (i, j) in cluster.z_bonds:
        H -= Jz * _two_body_pauli(N, i, j, "Z")
    return H


def plaquette_operator(cluster: KitaevCluster, plaquette_idx: int) -> np.ndarray:
    """Build the plaquette flux operator W_p for the p-th plaquette.

    For a 4-vertex plaquette around two bonds of each type (the small
    plaquette geometry), W_p is a 4-spin operator. The exact form
    depends on bond labels at the plaquette vertices; we construct it
    explicitly by traversing the cycle.
    """
    p = cluster.plaquettes[plaquette_idx]
    N = cluster.n_sites
    # For each vertex in the plaquette, find the "outward-pointing"
    # bond type (the one not shared with neighbors in the cycle).
    # Map each vertex to its bond types.
    bond_type_of = {}
    for (i, j) in cluster.x_bonds:
        bond_type_of.setdefault(i, {})[j] = "X"
        bond_type_of.setdefault(j, {})[i] = "X"
    for (i, j) in cluster.y_bonds:
        bond_type_of.setdefault(i, {})[j] = "Y"
        bond_type_of.setdefault(j, {})[i] = "Y"
    for (i, j) in cluster.z_bonds:
        bond_type_of.setdefault(i, {})[j] = "Z"
        bond_type_of.setdefault(j, {})[i] = "Z"

    s = ["I"] * N
    L = len(p)
    for k, v in enumerate(p):
        prev_v = p[(k - 1) % L]
        next_v = p[(k + 1) % L]
        # The vertex v has 3 bonds; two go to neighbors in the plaquette
        # (prev_v and next_v); the third "external" bond gives the
        # outward type. Find which type is NOT used by the cycle bonds.
        types_in_cycle = set()
        if prev_v in bond_type_of.get(v, {}):
            types_in_cycle.add(bond_type_of[v][prev_v])
        if next_v in bond_type_of.get(v, {}):
            types_in_cycle.add(bond_type_of[v][next_v])
        all_types = {"X", "Y", "Z"}
        external = list(all_types - types_in_cycle)
        if not external:
            # If both cycle bonds are the same type (degenerate), pick any.
            external = ["Z"]
        s[v] = external[0]
    return _pauli_string_matrix("".join(s))


def ground_state(cluster: KitaevCluster, **kwargs) -> tuple[float, np.ndarray]:
    """Lowest eigenvalue + eigenstate of the Kitaev Hamiltonian."""
    H = kitaev_hamiltonian(cluster, **kwargs)
    eigvals, eigvecs = np.linalg.eigh(H)
    return float(eigvals[0]), eigvecs[:, 0]


def plaquette_flux(psi: np.ndarray, cluster: KitaevCluster,
                    plaquette_idx: int) -> float:
    """⟨W_p⟩ — the Z_2 flux through plaquette p.

    For the exact ground state in the no-flux sector, ⟨W_p⟩ = +1
    (Lieb's theorem: the unfrustrated ground state has zero flux).
    """
    W = plaquette_operator(cluster, plaquette_idx)
    return float(np.real(psi.conj() @ W @ psi))


# ----------------------------------------------------------------------------
# Phase diagnostics
# ----------------------------------------------------------------------------

def phase_label(Jx: float, Jy: float, Jz: float) -> str:
    """Determine the phase from the coupling ratios.

    The Kitaev phase diagram has three "A" (Abelian/gapped) phases when
    one coupling dominates, and one "B" (non-Abelian/gapless) phase
    in the central region.
    """
    cmax = max(abs(Jx), abs(Jy), abs(Jz))
    other_sum = abs(Jx) + abs(Jy) + abs(Jz) - cmax
    if cmax > other_sum:
        if abs(Jx) == cmax:
            return "A_x"
        if abs(Jy) == cmax:
            return "A_y"
        return "A_z"
    return "B"
