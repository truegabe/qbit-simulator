"""Discrete-time quantum walk on a 1D line.

A walker has two registers:
  - coin   ∈ {0, 1}              -- a single qubit
  - position ∈ {0, ..., N-1}     -- N classical positions

Each step:
  1. Apply a coin unitary C (default: Hadamard) on the coin register.
  2. Apply the conditional shift S:
        if coin=0 → position decreases by 1
        if coin=1 → position increases by 1

Boundary handling:
  - "periodic"  : positions wrap around (a ring)
  - "reflecting": amplitudes at the edges bounce back
  - "absorbing" : amplitudes that hit the edge are discarded

Why this is interesting:
  - The state has only 2*N complex amplitudes (not 2^N), so we represent
    it as a dense (2, N) array. No exponential blowup -- this is one of
    the rare quantum algorithms where the natural state is small.
  - The walker spreads BALLISTICALLY: after T steps, the standard deviation
    of position scales as T, not sqrt(T) like a classical random walk.
    That's the quantum-walk advantage in one sentence.

Used in: quantum search (Grover via walks), quantum simulation of physical
1D systems, quantum machine-learning kernels, and as a building block in
many algorithmic primitives.
"""

from __future__ import annotations

from typing import Literal

import numpy as np


# Default coin = Hadamard. Other choices give different spreading behaviors.
HADAMARD_COIN = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128) / np.sqrt(2)


def quantum_walk_1d(
    n_positions: int,
    n_steps: int,
    coin: np.ndarray | None = None,
    initial_position: int | None = None,
    initial_coin: np.ndarray | None = None,
    boundary: Literal["periodic", "reflecting", "absorbing"] = "periodic",
) -> np.ndarray:
    """Run T = n_steps of a discrete-time 1D quantum walk on N positions.

    Returns an array of shape (n_steps + 1, n_positions) giving the
    position probability distribution at each time step (row 0 is the
    initial state, row n_steps is after the final step).

    Args:
        n_positions: number of sites on the line.
        n_steps: how many walk steps to apply.
        coin: 2x2 unitary applied to the coin each step (default Hadamard).
        initial_position: where the walker starts (default: center).
        initial_coin: initial coin amplitude (default: (|0> + i|1>) / sqrt(2),
            which gives a symmetric Hadamard-walk distribution).
        boundary: how to treat the edges.
    """
    if n_positions < 2:
        raise ValueError("need at least 2 positions for a walk")
    if n_steps < 0:
        raise ValueError("n_steps must be >= 0")

    C = HADAMARD_COIN if coin is None else np.asarray(coin, dtype=np.complex128)
    if C.shape != (2, 2):
        raise ValueError("coin must be 2x2")

    # State: shape (2, N). state[c, x] = amplitude for (coin=c, position=x).
    state = np.zeros((2, n_positions), dtype=np.complex128)
    x0 = n_positions // 2 if initial_position is None else int(initial_position)
    if initial_coin is None:
        # (|0> + i|1>)/sqrt(2) -- symmetric spreading under Hadamard coin.
        state[0, x0] = 1.0 / np.sqrt(2)
        state[1, x0] = 1j / np.sqrt(2)
    else:
        ic = np.asarray(initial_coin, dtype=np.complex128).reshape(2)
        ic = ic / np.linalg.norm(ic)
        state[0, x0] = ic[0]
        state[1, x0] = ic[1]

    # Record probability over positions at each time step.
    history = np.zeros((n_steps + 1, n_positions), dtype=np.float64)
    history[0] = _position_probabilities(state)

    for t in range(1, n_steps + 1):
        # 1. Coin: matrix-multiply along the coin axis.
        state = np.einsum("ij,jx->ix", C, state)
        # 2. Conditional shift.
        state = _shift(state, boundary)
        # Record.
        history[t] = _position_probabilities(state)
    return history


def _position_probabilities(state: np.ndarray) -> np.ndarray:
    """Marginal P(position) = Σ_c |state[c, x]|^2."""
    return np.sum(state.real**2 + state.imag**2, axis=0)


def _shift(state: np.ndarray, boundary: str) -> np.ndarray:
    """Apply the conditional shift: coin=0 → x-1, coin=1 → x+1."""
    out = np.zeros_like(state)
    if boundary == "periodic":
        out[0] = np.roll(state[0], -1)   # coin=0 → x-1; np.roll(-1) shifts left
        out[1] = np.roll(state[1], +1)
    elif boundary == "reflecting":
        # Amplitude at edge bounces back with opposite coin direction.
        # Implementation: do a standard shift but redirect off-edge amplitudes.
        # coin=0 at x=0 -> wraps off the left, reflect to x=0 with coin=1.
        out[0, :-1] = state[0, 1:]
        out[1, 1:]  = state[1, :-1]
        # Reflect: amplitude that fell off the edges flips coin and stays.
        out[1, 0]      += state[0, 0]
        out[0, -1]     += state[1, -1]
    elif boundary == "absorbing":
        out[0, :-1] = state[0, 1:]
        out[1, 1:]  = state[1, :-1]
        # Amplitudes at the boundary are dropped (norm decreases over time).
    else:
        raise ValueError(f"unknown boundary {boundary!r}")
    return out


# ---- classical baseline for comparison ----

# ----------------------------------------------------------------------------
# 2D quantum walk on a rectangular grid
# ----------------------------------------------------------------------------

# Grover coin for a 4-direction walker: 2|s⟩⟨s| - I where |s⟩ = (1,1,1,1)/2.
GROVER_COIN_4 = np.array([
    [-0.5,  0.5,  0.5,  0.5],
    [ 0.5, -0.5,  0.5,  0.5],
    [ 0.5,  0.5, -0.5,  0.5],
    [ 0.5,  0.5,  0.5, -0.5],
], dtype=np.complex128)

# Hadamard-style coin (DFT-based) for a 4-direction walker.
DFT_COIN_4 = (1 / 2) * np.array([
    [1,  1,  1,  1],
    [1,  1j, -1, -1j],
    [1, -1,  1, -1],
    [1, -1j, -1, 1j],
], dtype=np.complex128)


def quantum_walk_2d(
    rows: int,
    cols: int,
    n_steps: int,
    coin: np.ndarray | None = None,
    initial_position: tuple[int, int] | None = None,
    boundary: Literal["periodic", "reflecting", "absorbing"] = "periodic",
) -> np.ndarray:
    """Discrete-time 2D quantum walk on a rectangular grid.

    The walker has 4 coin directions: 0=N (row-1), 1=S (row+1), 2=W (col-1),
    3=E (col+1). State has shape (4, rows, cols).

    Returns probability snapshots: shape (n_steps+1, rows, cols).
    """
    if rows < 2 or cols < 2:
        raise ValueError("need at least 2 rows and 2 columns")
    C = GROVER_COIN_4 if coin is None else np.asarray(coin, dtype=np.complex128)
    if C.shape != (4, 4):
        raise ValueError("2D coin must be 4×4")

    state = np.zeros((4, rows, cols), dtype=np.complex128)
    r0, c0 = (rows // 2, cols // 2) if initial_position is None else initial_position
    # Equal superposition over coin directions.
    state[:, r0, c0] = 0.5

    history = np.zeros((n_steps + 1, rows, cols), dtype=np.float64)
    history[0] = _position_probabilities_2d(state)

    for t in range(1, n_steps + 1):
        state = np.einsum("ij,jrc->irc", C, state)
        state = _shift_2d(state, boundary)
        history[t] = _position_probabilities_2d(state)
    return history


def _position_probabilities_2d(state: np.ndarray) -> np.ndarray:
    return np.sum(state.real**2 + state.imag**2, axis=0)


def _shift_2d(state: np.ndarray, boundary: str) -> np.ndarray:
    """Shift each coin component in its own direction."""
    out = np.zeros_like(state)
    # coin=0 → north (row-1)
    # coin=1 → south (row+1)
    # coin=2 → west  (col-1)
    # coin=3 → east  (col+1)
    if boundary == "periodic":
        out[0] = np.roll(state[0], -1, axis=0)
        out[1] = np.roll(state[1], +1, axis=0)
        out[2] = np.roll(state[2], -1, axis=1)
        out[3] = np.roll(state[3], +1, axis=1)
    elif boundary == "reflecting":
        # Each direction: shift, and amplitude that falls off the edge reflects
        # with coin direction flipped.
        # N: state[0, r, c] → out[0, r-1, c] for r >= 1, and r=0 reflects to S.
        out[0, :-1, :] = state[0, 1:, :]   # north drift
        out[1, 0,    :] += state[0, 0, :]  # reflection at top
        out[1, 1:,  :] = state[1, :-1, :]  # south drift
        out[0, -1,   :] += state[1, -1, :] # reflection at bottom
        out[2, :, :-1] = state[2, :, 1:]   # west drift
        out[3, :, 0]   += state[2, :, 0]   # reflect at left
        out[3, :, 1:]  = state[3, :, :-1]  # east drift
        out[2, :, -1]  += state[3, :, -1]  # reflect at right
    elif boundary == "absorbing":
        out[0, :-1, :] = state[0, 1:, :]
        out[1, 1:,  :] = state[1, :-1, :]
        out[2, :, :-1] = state[2, :, 1:]
        out[3, :, 1:]  = state[3, :, :-1]
    else:
        raise ValueError(f"unknown boundary {boundary!r}")
    return out


def classical_walk_1d(
    n_positions: int,
    n_steps: int,
    initial_position: int | None = None,
    p_right: float = 0.5,
    boundary: Literal["periodic", "reflecting", "absorbing"] = "periodic",
) -> np.ndarray:
    """Classical discrete-time random walk on a 1D line.

    Same I/O shape as `quantum_walk_1d`. Useful as the "expected diffusive
    spreading" baseline against which the quantum walk's ballistic spreading
    becomes obvious.
    """
    p = np.zeros(n_positions, dtype=np.float64)
    x0 = n_positions // 2 if initial_position is None else int(initial_position)
    p[x0] = 1.0

    history = np.zeros((n_steps + 1, n_positions), dtype=np.float64)
    history[0] = p

    for t in range(1, n_steps + 1):
        left  = (1.0 - p_right) * p
        right = p_right * p
        new = np.zeros_like(p)
        if boundary == "periodic":
            new += np.roll(left, -1)
            new += np.roll(right, +1)
        elif boundary == "reflecting":
            new[:-1] += left[1:]
            new[1:]  += right[:-1]
            new[0]   += right[0]
            new[-1]  += left[-1]
        elif boundary == "absorbing":
            new[:-1] += left[1:]
            new[1:]  += right[:-1]
        else:
            raise ValueError(f"unknown boundary {boundary!r}")
        p = new
        history[t] = p
    return history


# ---- statistics helpers ----

# ----------------------------------------------------------------------------
# Continuous-time quantum walk
# ----------------------------------------------------------------------------

def continuous_time_walk(
    adjacency: np.ndarray,
    initial_position: int,
    times: np.ndarray,
    gamma: float = 1.0,
) -> np.ndarray:
    """Continuous-time quantum walk on a graph defined by `adjacency`.

    The walker's wave function evolves under the Hamiltonian H = γ · A
    (where A is the adjacency matrix). At time t:
        |ψ(t)⟩ = exp(-i γ A t) |position⟩

    Args:
        adjacency: V×V real symmetric adjacency matrix (1 for edges, 0 else).
        initial_position: starting vertex index.
        times: array of times at which to sample the wavefunction.
        gamma: hopping rate.

    Returns:
        Array of shape (len(times), V) giving |ψ_v(t)|² for each (t, vertex).
    """
    from scipy.linalg import expm

    A = np.asarray(adjacency, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("adjacency must be a square matrix")
    if not np.allclose(A, A.T, atol=1e-9):
        raise ValueError("adjacency must be symmetric")
    n = A.shape[0]
    if not (0 <= initial_position < n):
        raise IndexError(f"initial position {initial_position} out of range")

    psi0 = np.zeros(n, dtype=np.complex128)
    psi0[initial_position] = 1.0

    times = np.asarray(times, dtype=np.float64)
    probs = np.zeros((len(times), n), dtype=np.float64)
    H = gamma * A
    # Diagonalize once for efficient evolution.
    eigvals, eigvecs = np.linalg.eigh(H)
    psi0_diag = eigvecs.conj().T @ psi0
    for ti, t in enumerate(times):
        psi_t_diag = np.exp(-1j * eigvals * t) * psi0_diag
        psi_t = eigvecs @ psi_t_diag
        probs[ti] = np.abs(psi_t) ** 2
    return probs


def line_graph_adjacency(n: int, periodic: bool = False) -> np.ndarray:
    """Adjacency matrix of a 1D line graph (path or ring)."""
    A = np.zeros((n, n), dtype=np.float64)
    for i in range(n - 1):
        A[i, i + 1] = 1
        A[i + 1, i] = 1
    if periodic and n > 2:
        A[0, n - 1] = 1
        A[n - 1, 0] = 1
    return A


def cycle_graph_adjacency(n: int) -> np.ndarray:
    """Cycle C_n adjacency matrix."""
    return line_graph_adjacency(n, periodic=True)


def complete_graph_adjacency(n: int) -> np.ndarray:
    """K_n adjacency matrix."""
    A = np.ones((n, n), dtype=np.float64) - np.eye(n)
    return A


def hypercube_adjacency(d: int) -> np.ndarray:
    """d-dimensional hypercube graph: 2^d vertices, connected by single-bit flips."""
    n = 2 ** d
    A = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for k in range(d):
            j = i ^ (1 << k)
            A[i, j] = 1.0
    return A


def spatial_search_ctqw(
    adjacency: np.ndarray,
    marked_vertex: int,
    times: np.ndarray,
    gamma: float | None = None,
) -> dict:
    """Continuous-time quantum walk search for a marked vertex (Childs-Goldstone 2004).

    Setup:
        - Hamiltonian H = -γ · A - |w⟩⟨w|  (graph Laplacian minus marked-vertex
          oracle term).
        - Start at the uniform-superposition state |s⟩ = (1/√N) Σ |i⟩.
        - Evolve under H for time t. The probability of finding the walker
          at the marked vertex |w⟩ oscillates with period ~ 2π · √N
          (for highly-connected graphs like K_N and the hypercube).

    Optimal search time: t* ≈ (π/2) · √N.
    Optimal γ: depends on graph; for K_N, γ = 1/N.

    Args:
        adjacency:      symmetric V×V adjacency matrix of the graph.
        marked_vertex:  index of the marked vertex (0..V-1).
        times:          times at which to compute P(marked).
        gamma:          hopping rate. Auto-set to 1/V for K_N if None.

    Returns:
        dict with:
            probabilities: P(marked) at each time
            optimal_time:  the time t in `times` at which P is maximal
            max_probability: that maximum P
    """
    A = np.asarray(adjacency, dtype=np.float64)
    n = A.shape[0]
    if not (0 <= marked_vertex < n):
        raise IndexError(f"marked_vertex {marked_vertex} out of range [0, {n})")
    if gamma is None:
        gamma = 1.0 / n     # good default for the complete graph

    # H = -γ A - |w⟩⟨w| (we set the sign so that |s⟩ is a low-energy state).
    H = -gamma * A
    H[marked_vertex, marked_vertex] -= 1.0

    # Initial uniform superposition.
    psi0 = np.ones(n, dtype=np.complex128) / np.sqrt(n)

    # Diagonalize H once; evolve in eigenbasis.
    eigvals, eigvecs = np.linalg.eigh(H)
    psi0_diag = eigvecs.conj().T @ psi0

    times = np.asarray(times, dtype=np.float64)
    probs = np.zeros(len(times), dtype=np.float64)
    for ti, t in enumerate(times):
        psi_t_diag = np.exp(-1j * eigvals * t) * psi0_diag
        psi_t = eigvecs @ psi_t_diag
        probs[ti] = abs(psi_t[marked_vertex]) ** 2

    opt_idx = int(np.argmax(probs))
    return {
        "probabilities":    probs,
        "optimal_time":     float(times[opt_idx]),
        "max_probability":  float(probs[opt_idx]),
        "gamma":            float(gamma),
    }


def spread_sigma(history: np.ndarray) -> np.ndarray:
    """Standard deviation of position at each time step."""
    n_pos = history.shape[1]
    x = np.arange(n_pos)
    mean = np.einsum("tx,x->t", history, x) / history.sum(axis=1)
    var  = np.einsum("tx,x->t", history, x**2) / history.sum(axis=1) - mean**2
    return np.sqrt(np.maximum(var, 0))
