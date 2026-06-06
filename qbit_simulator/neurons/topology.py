"""Network topologies: small-world, scale-free, modular.

Different brain regions have different connectivity statistics:
  - Local cortical circuits ≈ small-world (Watts-Strogatz):
    short path lengths + high clustering.
  - Cortical hub networks ≈ scale-free (Barabási-Albert):
    a few highly-connected hubs.
  - Whole brain ≈ hierarchically modular.

This module gives generators for each, plus topology metrics.
"""

from __future__ import annotations

import numpy as np


def watts_strogatz(n: int, k: int = 4, p: float = 0.1,
                    rng: np.random.Generator | None = None) -> np.ndarray:
    """Small-world graph: ring lattice with k neighbors, randomly rewired.

    Returns symmetric adjacency matrix.
    """
    rng = rng or np.random.default_rng(0)
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(1, k // 2 + 1):
            A[i, (i + j) % n] = 1
            A[(i + j) % n, i] = 1
    # Rewire with probability p.
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] and rng.uniform() < p:
                # Pick a new j' not equal to i and not already connected.
                new_j = int(rng.integers(n))
                while new_j == i or A[i, new_j]:
                    new_j = int(rng.integers(n))
                A[i, j] = 0; A[j, i] = 0
                A[i, new_j] = 1; A[new_j, i] = 1
    return A


def barabasi_albert(n: int, m: int = 2,
                     rng: np.random.Generator | None = None) -> np.ndarray:
    """Scale-free graph: preferential attachment.

    Start with m+1 fully connected nodes. Add one node at a time, with
    m edges to existing nodes chosen with probability proportional to
    degree.
    """
    rng = rng or np.random.default_rng(0)
    A = np.zeros((n, n), dtype=int)
    # Initial clique.
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            A[i, j] = 1; A[j, i] = 1
    for i in range(m + 1, n):
        degrees = A.sum(axis=0)[:i]
        if degrees.sum() == 0:
            targets = rng.choice(i, size=m, replace=False)
        else:
            p = degrees / degrees.sum()
            targets = rng.choice(i, size=m, replace=False, p=p)
        for t in targets:
            A[i, t] = 1; A[t, i] = 1
    return A


def modular_network(modules: int = 4, size_per_module: int = 25,
                     p_intra: float = 0.3, p_inter: float = 0.02,
                     rng: np.random.Generator | None = None) -> np.ndarray:
    """Modular block-structured network."""
    rng = rng or np.random.default_rng(0)
    n = modules * size_per_module
    A = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            same = (i // size_per_module) == (j // size_per_module)
            p = p_intra if same else p_inter
            if rng.uniform() < p:
                A[i, j] = 1; A[j, i] = 1
    return A


def clustering_coefficient(A: np.ndarray) -> float:
    """Global clustering coefficient (transitivity)."""
    n = A.shape[0]
    cc = []
    for i in range(n):
        neigh = np.where(A[i] > 0)[0]
        k = len(neigh)
        if k < 2:
            continue
        sub = A[np.ix_(neigh, neigh)]
        edges = sub.sum() / 2
        cc.append(edges / (k * (k - 1) / 2))
    return float(np.mean(cc)) if cc else 0.0


def average_path_length(A: np.ndarray) -> float:
    """BFS-based mean shortest path."""
    n = A.shape[0]
    total = 0.0; count = 0
    for src in range(n):
        dist = -np.ones(n, dtype=int); dist[src] = 0
        frontier = [src]
        while frontier:
            new = []
            for u in frontier:
                for v in np.where(A[u] > 0)[0]:
                    if dist[v] < 0:
                        dist[v] = dist[u] + 1
                        new.append(v)
            frontier = new
        for v in range(n):
            if v != src and dist[v] > 0:
                total += dist[v]; count += 1
    return float(total / count) if count else 0.0


def degree_distribution(A: np.ndarray) -> np.ndarray:
    return A.sum(axis=0)
