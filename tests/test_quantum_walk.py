"""Tests for the 1D discrete-time quantum walk."""

import numpy as np
import pytest

from qbit_simulator.algorithms.quantum_walk import (
    quantum_walk_1d, classical_walk_1d, spread_sigma, HADAMARD_COIN,
)


# ---- norm & basic correctness ----

def test_norm_preserved_periodic():
    h = quantum_walk_1d(n_positions=20, n_steps=10, boundary="periodic")
    # Every row should sum to 1.
    sums = h.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-10)


def test_norm_preserved_reflecting():
    h = quantum_walk_1d(n_positions=20, n_steps=10, boundary="reflecting")
    sums = h.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-10)


def test_absorbing_loses_norm_over_time():
    """Absorbing boundary should monotonically lose probability."""
    h = quantum_walk_1d(n_positions=10, n_steps=20, boundary="absorbing")
    sums = h.sum(axis=1)
    # First step starts at 1, sums should be non-increasing.
    assert sums[0] == pytest.approx(1.0, abs=1e-12)
    for t in range(1, len(sums)):
        assert sums[t] <= sums[t - 1] + 1e-12


def test_initial_state_concentrated_at_start():
    h = quantum_walk_1d(n_positions=21, n_steps=5)
    # Initial row: all probability at the center (index 10).
    assert h[0, 10] == pytest.approx(1.0, abs=1e-12)
    assert np.allclose(np.delete(h[0], 10), 0.0)


# ---- ballistic vs diffusive spreading ----

def test_quantum_walk_spreads_ballistically():
    """sigma_quantum(T) should grow ~ linearly with T (not sqrt(T))."""
    n_steps = 60
    h_q = quantum_walk_1d(n_positions=301, n_steps=n_steps, boundary="periodic")
    sigma = spread_sigma(h_q)
    # By T=50 a Hadamard walk has sigma ~ T/sqrt(2). Allow generous tolerance.
    # Key claim: sigma(T=50) >> sqrt(50).
    assert sigma[50] > 3 * np.sqrt(50)


def test_classical_walk_spreads_diffusively():
    """sigma_classical(T) ~ sqrt(T)."""
    n_steps = 100
    h_c = classical_walk_1d(n_positions=401, n_steps=n_steps, boundary="periodic")
    sigma = spread_sigma(h_c)
    # Theory: sigma = sqrt(T) for symmetric walk. Allow +/- 20%.
    expected = np.sqrt(n_steps)
    assert 0.8 * expected < sigma[-1] < 1.2 * expected


def test_quantum_spreads_faster_than_classical_at_large_T():
    """The famous result: quantum walks beat classical walks asymptotically."""
    n_steps = 80
    n_pos = 301
    h_q = quantum_walk_1d(n_positions=n_pos, n_steps=n_steps, boundary="periodic")
    h_c = classical_walk_1d(n_positions=n_pos, n_steps=n_steps, boundary="periodic")
    sigma_q = spread_sigma(h_q)[-1]
    sigma_c = spread_sigma(h_c)[-1]
    # Quantum should have spread MUCH further. By T=80, ratio ≥ 4x.
    assert sigma_q > 4 * sigma_c


# ---- hand-computed step ----

def test_one_step_from_zero_coin_state():
    """Walker at position 1, coin = |0>. After one Hadamard step,
    the coin becomes (|0>+|1>)/sqrt(2), then the shift puts
    amplitude/sqrt(2) at position 0 (coin=0) and position 2 (coin=1).
    """
    state0 = np.array([1.0, 0.0], dtype=np.complex128)
    h = quantum_walk_1d(
        n_positions=5, n_steps=1, initial_position=2, initial_coin=state0,
        boundary="periodic",
    )
    # After one step, probabilities at positions 1 and 3 should each be 0.5.
    assert h[1, 1] == pytest.approx(0.5, abs=1e-10)
    assert h[1, 3] == pytest.approx(0.5, abs=1e-10)
    assert h[1, [0, 2, 4]] == pytest.approx(0.0, abs=1e-10)


# ---- different coins give different dynamics ----

# ---- 2D quantum walk ----

def test_2d_walk_norm_preserved_periodic():
    from qbit_simulator.algorithms.quantum_walk import quantum_walk_2d
    h = quantum_walk_2d(rows=10, cols=10, n_steps=10, boundary="periodic")
    for t in range(h.shape[0]):
        assert abs(h[t].sum() - 1.0) < 1e-10


def test_2d_walk_starts_at_initial_position():
    from qbit_simulator.algorithms.quantum_walk import quantum_walk_2d
    h = quantum_walk_2d(rows=10, cols=10, n_steps=5,
                         initial_position=(3, 7))
    # All initial probability at the start position.
    assert h[0, 3, 7] == pytest.approx(1.0, abs=1e-12)
    assert h[0].sum() - h[0, 3, 7] == pytest.approx(0.0, abs=1e-12)


def test_2d_walk_spreads_in_both_dimensions():
    """After many steps, probability should spread along both axes."""
    from qbit_simulator.algorithms.quantum_walk import quantum_walk_2d
    h = quantum_walk_2d(rows=21, cols=21, n_steps=20, boundary="periodic")
    final = h[20]
    # Spread in row direction.
    row_marginal = final.sum(axis=1)
    col_marginal = final.sum(axis=0)
    # Variance of row positions > 1.
    rows_idx = np.arange(21)
    r_mean = (rows_idx * row_marginal).sum()
    r_var  = (rows_idx**2 * row_marginal).sum() - r_mean**2
    assert r_var > 1
    # Same for cols.
    c_mean = (rows_idx * col_marginal).sum()
    c_var  = (rows_idx**2 * col_marginal).sum() - c_mean**2
    assert c_var > 1


# ---- Continuous-time quantum walk ----

def test_ctqw_starts_at_initial_position():
    from qbit_simulator.algorithms.quantum_walk import (
        continuous_time_walk, line_graph_adjacency,
    )
    A = line_graph_adjacency(5)
    probs = continuous_time_walk(A, initial_position=2, times=np.array([0.0]))
    assert probs[0, 2] == pytest.approx(1.0, abs=1e-12)


def test_ctqw_preserves_norm():
    from qbit_simulator.algorithms.quantum_walk import (
        continuous_time_walk, line_graph_adjacency,
    )
    A = line_graph_adjacency(10)
    probs = continuous_time_walk(
        A, initial_position=5, times=np.linspace(0, 5, 11),
    )
    for ti in range(probs.shape[0]):
        assert abs(probs[ti].sum() - 1.0) < 1e-9


def test_ctqw_on_cycle_returns_to_start():
    """On a 4-cycle, CTQW has periodic dynamics."""
    from qbit_simulator.algorithms.quantum_walk import (
        continuous_time_walk, cycle_graph_adjacency,
    )
    A = cycle_graph_adjacency(4)
    # Eigenvalues of C_4 adjacency: 2, 0, 0, -2. Period 2π in the "2" eigenmode.
    # At t = π, probability at initial position is the same as t = 0.
    probs = continuous_time_walk(A, 0, np.array([0, np.pi]))
    assert abs(probs[0, 0] - probs[1, 0]) < 1e-8


def test_ctqw_spreads_over_time():
    """On a long line graph, the walker spreads from a localized start."""
    from qbit_simulator.algorithms.quantum_walk import (
        continuous_time_walk, line_graph_adjacency,
    )
    A = line_graph_adjacency(21)
    probs = continuous_time_walk(A, 10, np.array([0.0, 5.0]))
    # Initial: localized at position 10. Final: spread.
    assert probs[0, 10] == pytest.approx(1.0, abs=1e-12)
    assert probs[1, 10] < 0.5  # spread away from start


# ---- Spatial search via CTQW ----

def test_ctqw_search_finds_marked_on_complete_graph():
    """On K_N with optimal γ=1/N, CTQW search achieves P_marked ~ 1 at
    t ≈ (π/2)√N."""
    from qbit_simulator.algorithms.quantum_walk import (
        spatial_search_ctqw, complete_graph_adjacency,
    )
    n = 16
    A = complete_graph_adjacency(n)
    t_opt = (np.pi / 2) * np.sqrt(n)
    times = np.linspace(0, 2 * t_opt, 100)
    result = spatial_search_ctqw(A, marked_vertex=5, times=times)
    # P at optimal time should be near 1.
    assert result["max_probability"] > 0.95


def test_ctqw_search_optimal_time_scales_as_sqrt_n():
    """The optimal search time scales as √N — verify the t* values for
    increasing N."""
    from qbit_simulator.algorithms.quantum_walk import (
        spatial_search_ctqw, complete_graph_adjacency,
    )
    optimal_times = []
    for n in (4, 9, 16, 25):
        A = complete_graph_adjacency(n)
        t_max = 3 * np.sqrt(n)
        times = np.linspace(0, t_max, 200)
        result = spatial_search_ctqw(A, marked_vertex=0, times=times)
        optimal_times.append(result["optimal_time"])
    # Ratio of consecutive optimal_times should be ≈ √(N_{i+1} / N_i).
    # For sqrt-scaling: t_opt(16) / t_opt(4) ≈ 2.
    ratio = optimal_times[2] / optimal_times[0]
    assert 1.6 < ratio < 2.4   # √(16/4) = 2 ± 20%


def test_ctqw_search_on_hypercube():
    """5D hypercube has 32 vertices; spatial search achieves √N speedup.

    Childs-Goldstone showed that for hypercubes with dim ≥ 5, CTQW search
    works with t* ~ √N and P_max approaching 1/2 or better. Below d=5,
    the spectral gap is too small for clean √N behavior.
    """
    from qbit_simulator.algorithms.quantum_walk import (
        spatial_search_ctqw, hypercube_adjacency,
    )
    A = hypercube_adjacency(d=5)   # 32 vertices
    n = A.shape[0]
    # Optimal gamma for hypercube: approximately 1/d (Childs-Goldstone).
    gamma_opt = 1.0 / 5
    t_max = 4 * np.sqrt(n)
    times = np.linspace(0, t_max, 200)
    result = spatial_search_ctqw(A, marked_vertex=3, times=times, gamma=gamma_opt)
    assert result["max_probability"] > 0.3   # well above 1/N = 0.03


def test_different_coin_produces_different_distribution():
    """Y-coin (rotation by pi/2) gives a different distribution than Hadamard."""
    Y_coin = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    h_had = quantum_walk_1d(n_positions=51, n_steps=20)
    h_y   = quantum_walk_1d(n_positions=51, n_steps=20, coin=Y_coin)
    # The two distributions should be visibly different.
    assert not np.allclose(h_had[-1], h_y[-1], atol=1e-3)
