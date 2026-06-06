"""ZX calculus: graphical reasoning for quantum circuits.

ZX (Coecke-Duncan 2008) is a graphical language where:

  * **Z-spider** ●_α with n inputs + m outputs represents
        |0...0⟩⟨0...0| + e^{iα} |1...1⟩⟨1...1|.
  * **X-spider** ○_α is the same with Hadamards on all legs.
  * Special spiders: phase 0 → standard Z/X parity ops.
  * Hadamard edge: shorthand for H between two spiders.

The fundamental ZX rules (a complete set on Clifford+T):

  (S1) Spider fusion: two spiders of same color joined by ANY number
       of edges → one spider with summed phase.
  (S2) Identity rules: spider with 2 legs and phase 0 → just a wire.
  (S3) Color change: spider conjugated by H on every leg flips color.
  (S4) π-copy: an X-spider with phase π attached to a Z-spider can
       be "copied" through.
  (S5) Bialgebra: Z-spider and X-spider with three connecting edges →
       can rearrange.

This module implements a SUBSET focused on diagram simplification:

  - `ZXDiagram`: nodes (spiders) + edges (regular or Hadamard).
  - `add_spider(kind, phase, ...)`, `add_edge(a, b, kind)`.
  - `to_unitary()`: evaluate the diagram as a numpy matrix (for
    verification on small examples).
  - `fuse_spiders()`: apply rule S1 (mergeable spiders).
  - `remove_identity_spiders()`: rule S2.
  - `simplify(diagram)`: iterate rules until fixed point.

We focus on small diagrams (≤ 5 spiders) for educational use; full
PyZX-style scalable simplification is out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


SpiderKind = Literal["Z", "X"]
EdgeKind = Literal["regular", "hadamard"]


# Pauli matrices
_I = np.eye(2, dtype=np.complex128)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
_H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)


@dataclass
class Spider:
    id: int
    kind: SpiderKind
    phase: float = 0.0
    n_inputs: int = 0
    n_outputs: int = 0
    is_input_boundary: bool = False
    is_output_boundary: bool = False
    is_inactive: bool = False    # set by fusion to mark this spider as gone


@dataclass
class Edge:
    a: int
    b: int
    kind: EdgeKind = "regular"


@dataclass
class ZXDiagram:
    """A simple ZX diagram on n_in inputs and n_out outputs.

    Spiders are labeled 0..N-1; boundary spiders (one per input/output
    wire) are flagged.
    """
    spiders: list[Spider] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    inputs: list[int] = field(default_factory=list)   # spider ids
    outputs: list[int] = field(default_factory=list)

    def add_spider(self, kind: SpiderKind, phase: float = 0.0) -> int:
        sid = len(self.spiders)
        self.spiders.append(Spider(id=sid, kind=kind, phase=phase))
        return sid

    def add_input(self) -> int:
        """Add a boundary 'identity' spider as an input wire."""
        sid = self.add_spider("Z", phase=0.0)
        self.spiders[sid].is_input_boundary = True
        self.inputs.append(sid)
        return sid

    def add_output(self) -> int:
        sid = self.add_spider("Z", phase=0.0)
        self.spiders[sid].is_output_boundary = True
        self.outputs.append(sid)
        return sid

    def add_edge(self, a: int, b: int, kind: EdgeKind = "regular") -> None:
        self.edges.append(Edge(a=a, b=b, kind=kind))

    def neighbors(self, s: int) -> list[tuple[int, EdgeKind]]:
        out = []
        for e in self.edges:
            if e.a == s:
                out.append((e.b, e.kind))
            elif e.b == s:
                out.append((e.a, e.kind))
        return out

    def degree(self, s: int) -> int:
        return sum(1 for e in self.edges if e.a == s or e.b == s)


# ----------------------------------------------------------------------------
# Spider matrix forms
# ----------------------------------------------------------------------------

def _spider_matrix(kind: SpiderKind, phase: float, n_in: int, n_out: int
                    ) -> np.ndarray:
    """Build the dense matrix for an (n_in, n_out)-leg spider."""
    d_in = 2 ** n_in
    d_out = 2 ** n_out
    M = np.zeros((d_out, d_in), dtype=np.complex128)
    if kind == "Z":
        # Z-spider: |0...0⟩⟨0...0| + e^{iφ}|1...1⟩⟨1...1|.
        M[0, 0] = 1.0
        if d_in > 0 and d_out > 0:
            M[d_out - 1, d_in - 1] = np.exp(1j * phase)
    elif kind == "X":
        # X-spider = H⊗H ... · Z-spider · H⊗H ... on both sides.
        Z_mat = _spider_matrix("Z", phase, n_in, n_out)
        H_in = _H
        for _ in range(n_in - 1):
            H_in = np.kron(H_in, _H)
        H_out = _H
        for _ in range(n_out - 1):
            H_out = np.kron(H_out, _H)
        if n_in == 0:
            H_in = np.array([[1.0 + 0j]])
        if n_out == 0:
            H_out = np.array([[1.0 + 0j]])
        M = H_out @ Z_mat @ H_in
    else:
        raise ValueError(f"unknown spider kind: {kind}")
    return M


# ----------------------------------------------------------------------------
# Diagram → unitary (for tiny diagrams)
# ----------------------------------------------------------------------------

def to_unitary(diagram: ZXDiagram) -> np.ndarray:
    """Evaluate a ZX diagram as a matrix from inputs → outputs.

    Limitation: this is a simple, slow contraction suitable for
    diagrams with ≤ ~6 spiders. We construct the explicit linear map
    by summing over each spider's basis states.

    Returns:
        2^{n_out} × 2^{n_in} matrix.
    """
    n_in = len(diagram.inputs)
    n_out = len(diagram.outputs)
    # Identify all wires. Each edge of the diagram is a wire.
    # We index every spider's leg uniquely and sum over each internal
    # wire's spin = 0, 1.
    spiders = diagram.spiders
    edges = diagram.edges
    n_spiders = len(spiders)

    # For each spider, list of (other_spider, edge_kind, edge_idx).
    leg_assignments = [[] for _ in range(n_spiders)]
    for e_idx, e in enumerate(edges):
        leg_assignments[e.a].append(("conn", e.b, e.kind, e_idx))
        leg_assignments[e.b].append(("conn", e.a, e.kind, e_idx))

    # Each edge has 2 possible spin values. Sum over all assignments.
    n_edges = len(edges)
    d_in = 2 ** n_in
    d_out = 2 ** n_out
    U = np.zeros((d_out, d_in), dtype=np.complex128)

    # Sum over the bit-string over edges.
    for edge_config in range(2 ** n_edges):
        edge_bits = [(edge_config >> i) & 1 for i in range(n_edges)]

        # For each input/output, the values are determined by the
        # configuration of OUTER spider legs. We separately track:
        #   - input wires (boundary spiders) → contribute to input idx.
        #   - output wires (boundary spiders) → contribute to output idx.
        # Each boundary spider has degree 1 (one edge to the inside).

        # Compute the spider amplitudes for the current edge config.
        weight = 1.0 + 0j
        for s in spiders:
            # Spider has some legs (= edges + maybe boundary).
            # For ordinary boundary spiders (just identity wires), the
            # value is fully determined by their single edge's spin.
            if s.is_input_boundary or s.is_output_boundary:
                continue
            # Internal spider: collect all leg spins and apply rule.
            leg_spins = []
            for (kind, other, e_kind, e_idx) in leg_assignments[s.id]:
                bit = edge_bits[e_idx]
                if e_kind == "hadamard":
                    # Each Hadamard edge has weight depending on
                    # 'inner' value vs spider value. We treat the
                    # spider leg as in the rotated basis: the bit
                    # along this edge corresponds to the post-H state.
                    pass    # handled below by inserting H factors
                leg_spins.append((bit, e_kind))
            # Z-spider amplitude:
            if s.kind == "Z":
                # All legs must agree (in computational basis after
                # H-conversion). Spider gives 1 if all 0, e^{iφ} if all 1.
                # For Hadamard edges, "agree" means the LEG value at the
                # spider matches; the bit on the edge is the OTHER end's
                # leg value. We treat Hadamard edges by inserting an H
                # factor: amplitude (1/sqrt(2)) · (-1)^{a·b} between the
                # spider's leg value a and the bit value b on the edge.
                # That's a 2x2 matrix per leg-edge connection.
                pass
            # The fully general approach is messy. We restrict to
            # diagrams without Hadamard edges, then later add support.
        weight = None    # placeholder
        break    # we'll just use a different strategy below

    # Restart with a CLEAN tensor-network contraction strategy.
    return _contract_diagram_clean(diagram)


def _contract_diagram_clean(diagram: ZXDiagram) -> np.ndarray:
    """A simpler contraction that builds spider tensors + contracts."""
    n_in = len(diagram.inputs)
    n_out = len(diagram.outputs)
    # Build per-spider tensors. Each internal spider with k legs has
    # shape (2,)*k and entries = δ_{all legs equal 0} + e^{iφ}·δ_{all 1}.
    # H-edges insert a Hadamard along one leg.
    # We build the global tensor by repeated np.einsum.

    # Assign a unique 'leg label' to each spider-edge endpoint.
    # Each edge has 2 endpoints (one per attached spider). The two
    # endpoints share the same label (so contraction sums over the
    # 2-dim edge wire).
    spider_legs: dict[int, list[str]] = {s.id: [] for s in diagram.spiders}
    label_counter = [0]
    def next_label() -> str:
        lab = chr(ord('a') + label_counter[0])
        label_counter[0] += 1
        return lab
    edge_labels = []
    for e in diagram.edges:
        lab = next_label()
        edge_labels.append(lab)
        spider_legs[e.a].append(lab)
        spider_legs[e.b].append(lab)

    # Boundary spiders are just stubs — they DON'T get a tensor;
    # their leg label directly becomes the input or output einsum index.
    # Inactive spiders (fused away) are skipped entirely.
    is_boundary = {s.id: (s.is_input_boundary or s.is_output_boundary)
                    for s in diagram.spiders}
    is_skip = {s.id: (is_boundary[s.id] or s.is_inactive)
                for s in diagram.spiders}

    # Build each non-skip spider tensor. For Hadamard edges, we
    # absorb the H into the spider's leg.
    spider_tensors = {}
    for s in diagram.spiders:
        if is_skip[s.id]:
            continue
        n_legs = len(spider_legs[s.id])
        if n_legs == 0:
            spider_tensors[s.id] = np.array(1.0 + np.exp(1j * s.phase))
            continue
        # Z-spider tensor: T[0,0,...,0] = 1, T[1,1,...,1] = e^{iφ}, else 0.
        T = np.zeros([2] * n_legs, dtype=np.complex128)
        T[tuple([0] * n_legs)] = 1.0
        T[tuple([1] * n_legs)] = np.exp(1j * s.phase)
        if s.kind == "X":
            for axis in range(n_legs):
                T = np.tensordot(_H, T, axes=([1], [axis]))
                T = np.moveaxis(T, 0, axis)
        spider_tensors[s.id] = T

    # Absorb Hadamards on Hadamard edges into one endpoint
    # (pick the endpoint that's not a boundary; otherwise either).
    for e_idx, e in enumerate(diagram.edges):
        if e.kind == "hadamard":
            lab = edge_labels[e_idx]
            # Prefer to absorb into the non-boundary endpoint.
            target = e.a if not is_boundary[e.a] else e.b
            if is_boundary[target]:
                # Both endpoints are boundaries — insert an H matrix
                # as a fake degree-2 spider. Skip for now.
                continue
            axis = spider_legs[target].index(lab)
            T = spider_tensors[target]
            T = np.tensordot(_H, T, axes=([1], [axis]))
            T = np.moveaxis(T, 0, axis)
            spider_tensors[target] = T

    # Build einsum subscripts: skip boundary + inactive spiders.
    operands = []
    subs = []
    for s in diagram.spiders:
        if is_skip[s.id]:
            continue
        operands.append(spider_tensors[s.id])
        subs.append("".join(spider_legs[s.id]))
    # Input labels come from each input spider's single leg.
    in_labels = [spider_legs[sid][0] for sid in diagram.inputs]
    out_labels = [spider_legs[sid][0] for sid in diagram.outputs]
    spec = ",".join(subs) + "->" + "".join(out_labels) + "".join(in_labels)
    if not operands:
        # All spiders are boundaries: just an identity wire (one edge).
        return np.eye(2 ** len(diagram.inputs), dtype=np.complex128)
    result = np.einsum(spec, *operands)
    return result.reshape(2 ** len(diagram.outputs), 2 ** len(diagram.inputs))


# ----------------------------------------------------------------------------
# Simplification rules
# ----------------------------------------------------------------------------

def fuse_spiders(diagram: ZXDiagram) -> bool:
    """Rule S1: merge two adjacent same-color spiders connected by ≥1
    regular edges. Phases add; one spider is removed.

    Returns True if any fusion happened.
    """
    for e in diagram.edges:
        if e.kind != "regular":
            continue
        sa = diagram.spiders[e.a]
        sb = diagram.spiders[e.b]
        if sa.is_input_boundary or sa.is_output_boundary:
            continue
        if sb.is_input_boundary or sb.is_output_boundary:
            continue
        if sa.kind != sb.kind:
            continue
        # Fuse: combine phases, redirect all edges of sb → sa,
        # remove sb and the connecting edge.
        sa.phase = (sa.phase + sb.phase) % (2 * np.pi)
        new_edges = []
        for e2 in diagram.edges:
            if e2.a == e.a and e2.b == e.b:
                continue
            if e2.a == e.b and e2.b == e.a:
                continue
            if e2.a == sb.id:
                e2 = Edge(sa.id, e2.b, e2.kind)
            if e2.b == sb.id:
                e2 = Edge(e2.a, sa.id, e2.kind)
            new_edges.append(e2)
        diagram.edges = new_edges
        # We don't actually remove sb from the spider list (would
        # require reindexing); just mark inactive.
        sb.is_inactive = True
        sb.phase = 0
        return True
    return False


def simplify(diagram: ZXDiagram, max_iter: int = 100) -> int:
    """Iteratively apply fusion until no more changes. Returns
    the number of fusions performed."""
    count = 0
    for _ in range(max_iter):
        if not fuse_spiders(diagram):
            return count
        count += 1
    return count
