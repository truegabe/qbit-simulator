"""Graph states and cluster states.

A graph state is a stabilizer state defined by an undirected graph G = (V, E):
  - One qubit per vertex.
  - Each vertex starts in |+⟩ (apply H to |0⟩).
  - For each edge (u, v), apply CZ between qubits u and v.

The stabilizer generator at vertex v is:
        K_v = X_v · (∏ Z_u for u neighbor of v)

Graph states are the universal resource for **measurement-based quantum
computation (MBQC)**: any unitary can be implemented by adaptive single-qubit
measurements on a sufficiently large cluster state.

Special cases provided:
  - cluster_state_1d(n)  : 1D chain (the simplest universal resource for 1D MBQC)
  - cluster_state_2d(r, c): 2D rectangular lattice (universal for general MBQC)
  - ring_graph_state(n)  : 1D chain with periodic boundary
  - complete_graph_state(n): K_n (every qubit connected to every other)
"""

from __future__ import annotations

from ..stabilizer import StabilizerState


def graph_state(n: int, edges: list[tuple[int, int]]) -> StabilizerState:
    """Build the graph state for an arbitrary undirected graph.

    Args:
        n: number of vertices (qubits).
        edges: list of (u, v) pairs. Self-loops and duplicates are rejected.

    Returns:
        A StabilizerState representing the graph state |G⟩.
    """
    # Sanity check edges.
    seen: set[tuple[int, int]] = set()
    for (u, v) in edges:
        if u == v:
            raise ValueError(f"self-loop on vertex {u} not allowed")
        if not (0 <= u < n and 0 <= v < n):
            raise IndexError(f"edge ({u},{v}) out of range [0,{n})")
        key = (min(u, v), max(u, v))
        if key in seen:
            raise ValueError(f"duplicate edge {key}")
        seen.add(key)

    st = StabilizerState(n)
    # All qubits to |+⟩.
    for q in range(n):
        st.h(q)
    # CZ on every edge.
    for (u, v) in edges:
        st.cz(u, v)
    return st


def cluster_state_1d(n: int) -> StabilizerState:
    """1D cluster state: vertices 0..n-1, edges between consecutive vertices."""
    if n < 2:
        raise ValueError("cluster_state_1d needs N >= 2")
    edges = [(i, i + 1) for i in range(n - 1)]
    return graph_state(n, edges)


def ring_graph_state(n: int) -> StabilizerState:
    """1D cluster state with periodic boundary (ring)."""
    if n < 3:
        raise ValueError("ring_graph_state needs N >= 3")
    edges = [(i, (i + 1) % n) for i in range(n)]
    return graph_state(n, edges)


def cluster_state_2d(rows: int, cols: int) -> StabilizerState:
    """2D rectangular cluster state: rows×cols grid with horizontal and
    vertical nearest-neighbor edges."""
    if rows < 1 or cols < 1:
        raise ValueError("rows and cols must be positive")
    n = rows * cols

    def idx(r, c):
        return r * cols + c

    edges: list[tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                edges.append((idx(r, c), idx(r, c + 1)))
            if r + 1 < rows:
                edges.append((idx(r, c), idx(r + 1, c)))
    return graph_state(n, edges)


def complete_graph_state(n: int) -> StabilizerState:
    """Graph state on the complete graph K_n: every qubit connected."""
    if n < 2:
        raise ValueError("complete_graph_state needs N >= 2")
    edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    return graph_state(n, edges)


def graph_state_stabilizers(n: int, edges: list[tuple[int, int]]) -> list[str]:
    """Return the canonical stabilizer generators of a graph state as Pauli
    strings: K_v = X_v · (Z on every neighbor of v)."""
    neighbors: list[set[int]] = [set() for _ in range(n)]
    for (u, v) in edges:
        neighbors[u].add(v)
        neighbors[v].add(u)
    out: list[str] = []
    for v in range(n):
        s = ["I"] * n
        s[v] = "X"
        for u in neighbors[v]:
            s[u] = "Z"
        out.append("".join(s))
    return out
