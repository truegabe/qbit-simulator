"""Minimum-Weight Perfect Matching (MWPM) decoder for the surface code.

The standard surface-code decoder. Given a syndrome (which stabilizers
fire), MWPM finds the minimum-weight set of data-qubit error chains
whose endpoints match the lit syndromes. This is asymptotically near-
optimal — it achieves the optimal threshold for the surface code under
depolarizing noise to within a small constant.

For a distance-d patch:
  * X-type stabilizers detect Z errors → matching graph on X-stabs.
  * Z-type stabilizers detect X errors → matching graph on Z-stabs.
  * Boundary nodes ("virtual" stabilizers) absorb chains that end at
    a boundary rather than another lit syndrome.

The matching is exact (we use brute-force enumeration of perfect
matchings) for small d — Edmonds' blossom algorithm would be needed
for larger d, but at d=3 there are at most 4 lit nodes per CSS sector,
giving ≤ (2·4-1)!! = 105 matchings to check.

This module provides:

  - `SurfaceCodeD3`: the standard 3×3 rotated surface-code patch
    (same layout as qec.surface_code_d3), with explicit X/Z syndrome
    adjacency graphs.
  - `MWPMDecoder(code)`: decode a syndrome → correction Pauli string.
  - `simulate_threshold(p_values, n_trials)`: run depolarizing noise
    Monte Carlo and report logical error rate vs physical error rate.

The threshold for the d=3 surface code under depolarizing noise is
~1% — below this, larger codes give exponential suppression; above it,
larger codes are worse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np


# ----------------------------------------------------------------------------
# Surface code patch geometry (d=3 rotated layout)
# ----------------------------------------------------------------------------
#
# Data-qubit layout:
#        0  1  2
#        3  4  5
#        6  7  8
#
# X-stabilizers (detect Z errors):
#   x0:  X_0 X_3              (left boundary)
#   x1:  X_5 X_8              (right boundary)
#   x2:  X_3 X_4 X_6 X_7      (bottom-left plaquette)
#   x3:  X_1 X_2 X_4 X_5      (top-right plaquette)
#
# Z-stabilizers (detect X errors):
#   z0:  Z_0 Z_1 Z_3 Z_4      (top-left plaquette)
#   z1:  Z_4 Z_5 Z_7 Z_8      (bottom-right plaquette)
#   z2:  Z_1 Z_2              (top boundary)
#   z3:  Z_6 Z_7              (bottom boundary)
#
# Logical X = X_0 X_4 X_8  (any X-chain from left to right boundary)
# Logical Z = Z_2 Z_4 Z_6  (any Z-chain from top to bottom boundary)


# Each stabilizer is a tuple of data-qubit indices.
SURFACE_D3_X_STABS = {
    "x0": (0, 3),
    "x1": (5, 8),
    "x2": (3, 4, 6, 7),
    "x3": (1, 2, 4, 5),
}

SURFACE_D3_Z_STABS = {
    "z0": (0, 1, 3, 4),
    "z1": (4, 5, 7, 8),
    "z2": (1, 2),
    "z3": (6, 7),
}

# Logical operator supports (as tuples of data-qubit indices).
SURFACE_D3_LOGICAL_X = (0, 4, 8)   # left → right X chain
SURFACE_D3_LOGICAL_Z = (2, 4, 6)   # top  → bottom Z chain


@dataclass
class SurfaceCodeD3:
    """Distance-3 rotated surface code patch [[9, 1, 3]].

    Bundles the data-qubit layout + stabilizer assignments + the
    precomputed error→syndrome map so the MWPM decoder can build its
    matching graphs.
    """
    n_data: int = 9
    x_stabilizers: dict[str, tuple[int, ...]] = field(
        default_factory=lambda: dict(SURFACE_D3_X_STABS))
    z_stabilizers: dict[str, tuple[int, ...]] = field(
        default_factory=lambda: dict(SURFACE_D3_Z_STABS))
    logical_x: tuple[int, ...] = SURFACE_D3_LOGICAL_X
    logical_z: tuple[int, ...] = SURFACE_D3_LOGICAL_Z

    def z_error_syndrome(self, z_error_qubits: set[int]) -> frozenset[str]:
        """Which X-stabilizers fire given a Z-error pattern on data qubits."""
        lit = set()
        for name, support in self.x_stabilizers.items():
            if len(set(support) & z_error_qubits) % 2 == 1:
                lit.add(name)
        return frozenset(lit)

    def x_error_syndrome(self, x_error_qubits: set[int]) -> frozenset[str]:
        """Which Z-stabilizers fire given an X-error pattern on data qubits."""
        lit = set()
        for name, support in self.z_stabilizers.items():
            if len(set(support) & x_error_qubits) % 2 == 1:
                lit.add(name)
        return frozenset(lit)


# ----------------------------------------------------------------------------
# Syndrome-graph distance (BFS over data qubits)
# ----------------------------------------------------------------------------

def _build_data_neighbors(code: SurfaceCodeD3,
                           stab_dict: dict[str, tuple[int, ...]]
                           ) -> dict[int, set[str]]:
    """For each data qubit, list which stabilizers contain it."""
    out: dict[int, set[str]] = {q: set() for q in range(code.n_data)}
    for name, support in stab_dict.items():
        for q in support:
            out[q].add(name)
    return out


def _syndrome_distance_table(
    code: SurfaceCodeD3,
    stab_dict: dict[str, tuple[int, ...]],
    boundary_pairs: dict[str, set[int]],
) -> tuple[dict[tuple[str, str], int], dict[tuple[str, str], tuple[int, ...]]]:
    """BFS over the syndrome graph to compute pairwise distances.

    Each stabilizer is a node. Two stabilizers are connected by an edge
    of weight 1 if a single data-qubit error flips both. Boundary nodes
    are virtual: they connect to any stabilizer that a single data
    qubit can connect to the boundary.

    Returns:
        (distance, path) dicts keyed by sorted (a, b) node pairs. Path
        is a tuple of data-qubit indices to flip.
    """
    nodes = list(stab_dict.keys()) + list(boundary_pairs.keys())
    # Build adjacency: edge between node u and node v iff some data qubit
    # is in both supports (or in u and in the boundary v).
    adj: dict[str, dict[str, tuple[int, ...]]] = {n: {} for n in nodes}
    # Stabilizer-stabilizer edges.
    stab_qs = {n: set(s) for n, s in stab_dict.items()}
    for u, v in combinations(stab_dict.keys(), 2):
        common = stab_qs[u] & stab_qs[v]
        if common:
            q = sorted(common)[0]
            adj[u][v] = (q,)
            adj[v][u] = (q,)
    # Stabilizer-boundary edges.
    for b_name, b_qs in boundary_pairs.items():
        for s_name, s_qs in stab_qs.items():
            common = b_qs & s_qs
            if common:
                q = sorted(common)[0]
                adj[s_name][b_name] = (q,)
                adj[b_name][s_name] = (q,)

    # BFS shortest path between every pair of nodes.
    distance: dict[tuple[str, str], int] = {}
    path: dict[tuple[str, str], tuple[int, ...]] = {}
    for start in nodes:
        dist = {start: 0}
        prev: dict[str, tuple[str, int]] = {}
        frontier = [start]
        while frontier:
            new_frontier = []
            for u in frontier:
                for v, edge_qs in adj[u].items():
                    if v not in dist:
                        dist[v] = dist[u] + 1
                        prev[v] = (u, edge_qs[0])
                        new_frontier.append(v)
            frontier = new_frontier
        for end, d in dist.items():
            if start == end:
                continue
            # Reconstruct path.
            qs = []
            cur = end
            while cur != start:
                u, q = prev[cur]
                qs.append(q)
                cur = u
            key = tuple(sorted((start, end)))
            distance[key] = d
            path[key] = tuple(sorted(qs))
    return distance, path


# Boundary regions for d=3 patch.
# X-type boundaries: top/bottom (no X-stab boundaries — X-stabs span left↔right).
# Z-type boundaries: left/right.
#
# For Z errors → X-syndrome matching:
#   Boundaries are top + bottom; a Z-error chain ending at top/bottom
#   doesn't violate any X-stabilizer.
#   X-stabs x0, x2 touch left boundary via data qubits 0, 3, 6.
#   X-stabs x1, x3 touch right boundary via data qubits 2, 5, 8.
SURFACE_D3_X_BOUNDARIES = {
    "X_top":    {0, 1, 2},
    "X_bottom": {6, 7, 8},
}
# For X errors → Z-syndrome matching:
SURFACE_D3_Z_BOUNDARIES = {
    "Z_left":   {0, 3, 6},
    "Z_right":  {2, 5, 8},
}


# ----------------------------------------------------------------------------
# Minimum-Weight Perfect Matching
# ----------------------------------------------------------------------------

def _enumerate_perfect_matchings(nodes: list) -> list[list[tuple]]:
    """All perfect matchings on `nodes` (must have even length)."""
    if len(nodes) == 0:
        return [[]]
    if len(nodes) % 2 == 1:
        return []
    first = nodes[0]
    matchings = []
    for j in range(1, len(nodes)):
        partner = nodes[j]
        rest = nodes[1:j] + nodes[j+1:]
        for sub in _enumerate_perfect_matchings(rest):
            matchings.append([(first, partner)] + sub)
    return matchings


def _min_weight_matching(
    lit_nodes: list[str],
    boundary_nodes: list[str],
    distance: dict[tuple[str, str], int],
) -> list[tuple[str, str]]:
    """Find the minimum-weight perfect matching.

    Algorithm:
      - For each subset of boundary nodes with the same parity as
        |lit_nodes|, augment lit_nodes with that subset (so the total
        node count is even).
      - Boundary-to-boundary pairs have weight 0 (free to ignore).
      - Enumerate all perfect matchings, return the lightest.
    """
    best_match = None
    best_weight = float("inf")
    # We pair each lit node with either another lit node or a boundary.
    # Equivalent: augment lit_nodes with up to len(lit_nodes) boundary
    # "copies" so the total is even, then enumerate perfect matchings
    # where boundary-boundary pairs cost 0.
    n_lit = len(lit_nodes)
    # We need at most n_lit boundary slots (each lit might pair to a boundary).
    # The number of boundary "slots" we add: any k such that n_lit + k is even.
    # The boundary set is small (2 nodes for d=3) so we just try all
    # multi-subsets up to n_lit.
    # To keep it simple, allow each boundary to appear multiple times by
    # creating distinct copies "<boundary>#0", "<boundary>#1", ...
    boundary_copies = []
    for b in boundary_nodes:
        for k in range(n_lit + 1):
            boundary_copies.append(f"{b}#{k}")

    def edge_weight(a: str, b: str) -> int:
        # Strip "#k" suffix from boundary nodes.
        a_real = a.split("#")[0] if "#" in a else a
        b_real = b.split("#")[0] if "#" in b else b
        # Two boundary nodes match for free.
        if a_real in boundary_nodes and b_real in boundary_nodes:
            return 0
        key = tuple(sorted((a_real, b_real)))
        return distance.get(key, 10**9)

    # Try every subset of boundary copies of the appropriate parity.
    from itertools import combinations
    for k in range(0, len(boundary_copies) + 1):
        if (n_lit + k) % 2 != 0:
            continue
        for chosen_bs in combinations(boundary_copies, k):
            all_nodes = list(lit_nodes) + list(chosen_bs)
            if len(all_nodes) == 0:
                if best_weight > 0:
                    best_match = []
                    best_weight = 0
                continue
            # Cap enumeration: (2k-1)!! grows fast.
            if len(all_nodes) > 8:
                continue
            for match in _enumerate_perfect_matchings(all_nodes):
                w = sum(edge_weight(a, b) for a, b in match)
                if w < best_weight:
                    best_weight = w
                    best_match = match
    return best_match if best_match is not None else []


# ----------------------------------------------------------------------------
# MWPM decoder
# ----------------------------------------------------------------------------

@dataclass
class MWPMDecoder:
    """MWPM decoder for the d=3 surface code patch.

    Pre-computes pairwise syndrome distances at construction; decoding
    is then a single minimum-weight perfect matching call.
    """
    code: SurfaceCodeD3
    _x_distance: dict = field(init=False, default=None)
    _x_path:     dict = field(init=False, default=None)
    _z_distance: dict = field(init=False, default=None)
    _z_path:     dict = field(init=False, default=None)

    def __post_init__(self):
        # For Z errors → X-stab syndrome, X-error boundaries:
        #   Z errors live in a graph where X-stabs are the nodes
        #   (since X-stabs detect Z errors). The boundary nodes are
        #   the TOP and BOTTOM regions where Z-chains can terminate
        #   without firing any X-stab.
        # But for the rotated surface code, X-chains end at top/bottom
        # and Z-chains end at left/right (depending on orientation).
        # In our layout, the X-stabs surround the bulk and the chains
        # for Z errors terminate at the boundaries opposite to where
        # X-stabs end.
        # We use the simpler convention: Z errors are matched against
        # the X-stabilizer graph with boundaries = SURFACE_D3_X_BOUNDARIES.
        self._x_distance, self._x_path = _syndrome_distance_table(
            self.code, self.code.x_stabilizers, SURFACE_D3_X_BOUNDARIES,
        )
        self._z_distance, self._z_path = _syndrome_distance_table(
            self.code, self.code.z_stabilizers, SURFACE_D3_Z_BOUNDARIES,
        )

    def decode_z_errors(self, x_syndrome: frozenset[str]) -> set[int]:
        """Given lit X-stabs, return the inferred Z-error qubits."""
        return self._match_and_path(
            list(x_syndrome),
            list(SURFACE_D3_X_BOUNDARIES.keys()),
            self._x_distance, self._x_path,
        )

    def decode_x_errors(self, z_syndrome: frozenset[str]) -> set[int]:
        """Given lit Z-stabs, return the inferred X-error qubits."""
        return self._match_and_path(
            list(z_syndrome),
            list(SURFACE_D3_Z_BOUNDARIES.keys()),
            self._z_distance, self._z_path,
        )

    @staticmethod
    def _match_and_path(lit: list[str], boundaries: list[str],
                         dist: dict, paths: dict) -> set[int]:
        if not lit:
            return set()
        matching = _min_weight_matching(lit, boundaries, dist)
        correction: set[int] = set()
        for a, b in matching:
            a_real = a.split("#")[0] if "#" in a else a
            b_real = b.split("#")[0] if "#" in b else b
            # Two boundary nodes: nothing to do.
            is_a_b = a_real not in lit and a_real in (b.split("#")[0] for b in boundaries) \
                     if False else (a_real not in (set(lit)))
            # Just look up the path.
            key = tuple(sorted((a_real, b_real)))
            if key in paths:
                # XOR the path qubits into the correction.
                for q in paths[key]:
                    correction ^= {q}
        return correction


# ----------------------------------------------------------------------------
# Threshold simulation
# ----------------------------------------------------------------------------

def _causes_logical_error(
    error_pattern: dict[str, set[int]],
    correction: dict[str, set[int]],
    code: SurfaceCodeD3,
) -> bool:
    """After applying correction, does the residual error commute with the
    logical operators?

    A logical X error: residual_x_errors anticommutes with logical Z = an
    X-chain crosses the patch top↔bottom (parity 1 on Z_2 Z_4 Z_6 support
    intersection).

    A logical Z error: residual_z_errors anticommutes with logical X =
    a Z-chain crosses left↔right (parity 1 on logical_x support).
    """
    res_x = error_pattern["X"] ^ correction["X"]
    res_z = error_pattern["Z"] ^ correction["Z"]
    # Anticommutation = odd overlap with logical operator support.
    logical_x_flip = len(res_z & set(code.logical_x)) % 2 == 1
    logical_z_flip = len(res_x & set(code.logical_z)) % 2 == 1
    return logical_x_flip or logical_z_flip


def simulate_logical_error_rate(
    code: SurfaceCodeD3,
    decoder: MWPMDecoder,
    p_physical: float,
    n_trials: int = 5000,
    rng: np.random.Generator | None = None,
) -> dict:
    """Monte-Carlo logical error rate under independent depolarizing noise.

    Each data qubit independently suffers an X, Y, or Z error with
    probability p_physical / 3 each.

    Returns:
        dict with p_physical, p_logical (logical error rate), and a
        breakdown by error type.
    """
    rng = rng or np.random.default_rng()
    n_logical = 0
    for _ in range(n_trials):
        # Sample errors per data qubit.
        x_errs = set()
        z_errs = set()
        for q in range(code.n_data):
            r = rng.uniform()
            if r < p_physical / 3.0:        # X
                x_errs.add(q)
            elif r < 2 * p_physical / 3.0:  # Y = XZ
                x_errs.add(q); z_errs.add(q)
            elif r < p_physical:            # Z
                z_errs.add(q)
        # Measure syndromes.
        x_synd = code.z_error_syndrome(z_errs)   # X-stabs detect Z errors
        z_synd = code.x_error_syndrome(x_errs)   # Z-stabs detect X errors
        # Decode.
        z_correction = decoder.decode_z_errors(x_synd)
        x_correction = decoder.decode_x_errors(z_synd)
        # Check for logical error.
        if _causes_logical_error(
            {"X": x_errs, "Z": z_errs},
            {"X": x_correction, "Z": z_correction},
            code,
        ):
            n_logical += 1
    return {
        "p_physical": p_physical,
        "p_logical":  n_logical / n_trials,
        "n_trials":   n_trials,
    }


def simulate_threshold(
    p_values: list[float],
    n_trials: int = 5000,
    rng: np.random.Generator | None = None,
) -> dict:
    """Sweep p_physical and report logical-error curve."""
    rng = rng or np.random.default_rng()
    code = SurfaceCodeD3()
    decoder = MWPMDecoder(code)
    p_logical = []
    for p in p_values:
        r = simulate_logical_error_rate(code, decoder, p, n_trials, rng)
        p_logical.append(r["p_logical"])
    return {
        "p_physical": np.array(p_values),
        "p_logical":  np.array(p_logical),
        "n_trials":   n_trials,
    }
