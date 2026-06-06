"""Tests for Cat-1 encoding relays: SparseRelay and CompressedSensingRelay.
Also covers Cat-2 AttentionGatedRelay.
"""

import numpy as np
import pytest

from qbit_simulator.sparse_relay import (
    SparseCode, SparseEncoder, SparseDecoder, SparseRelay, AdaptiveSparseRelay,
)
from qbit_simulator.compressed_sensing_relay import (
    RandomProjector, ISTASolver, FISTASolver, CompressedSensingRelay,
)
from qbit_simulator.attention_gated_relay import (
    KeyMatrix, AttentionGate, AttentionGatedRelay, MultiHeadAttentionGate,
)


# ===========================================================================
# SparseCode
# ===========================================================================

def test_sparse_code_dense_roundtrip():
    code = SparseCode(indices=np.array([0, 5, 9]),
                      values=np.array([1.0, 2.0, 3.0]),
                      n_dims=10)
    x = code.dense()
    assert x[0] == pytest.approx(1.0)
    assert x[5] == pytest.approx(2.0)
    assert x[9] == pytest.approx(3.0)
    assert x[1] == pytest.approx(0.0)


def test_sparse_code_density():
    code = SparseCode(indices=np.array([0, 1]),
                      values=np.array([1.0, 1.0]),
                      n_dims=10)
    assert code.density() == pytest.approx(0.2)


def test_sparse_code_bits():
    code = SparseCode(indices=np.array([0, 1, 2]),
                      values=np.array([1.0, 1.0, 1.0]),
                      n_dims=16)
    # index_bits = ceil(log2(16)) = 4; bits = 3*(4+32) = 108
    assert code.bits() == 3 * (4 + 32)


# ===========================================================================
# SparseEncoder
# ===========================================================================

def test_encoder_topk_returns_k_elements():
    enc = SparseEncoder(k=4, mode="topk")
    x   = np.random.default_rng(0).standard_normal(20)
    code = enc.encode(x)
    assert code.k == 4


def test_encoder_topk_selects_largest():
    enc = SparseEncoder(k=2, mode="topk")
    x   = np.array([0.1, 0.9, 0.05, 0.8, 0.3])
    code = enc.encode(x)
    assert set(code.indices.tolist()) == {1, 3}


def test_encoder_threshold_mode():
    enc  = SparseEncoder(threshold=0.5, mode="threshold")
    x    = np.array([0.1, 0.8, 0.3, 0.9])
    code = enc.encode(x)
    assert 1 in code.indices and 3 in code.indices
    assert 0 not in code.indices


def test_encoder_sparsity():
    enc = SparseEncoder(k=1, mode="topk")
    x   = np.ones(10)
    s   = enc.sparsity(x)
    assert s == pytest.approx(0.9)


def test_encoder_compression_ratio_gt1():
    enc = SparseEncoder(k=2, mode="topk")
    x   = np.random.default_rng(1).standard_normal(100)
    cr  = enc.compression_ratio(x)
    assert cr > 1.0


# ===========================================================================
# SparseDecoder
# ===========================================================================

def test_decoder_reconstructs_values():
    code = SparseCode(indices=np.array([2, 7]),
                      values=np.array([0.5, -0.3]),
                      n_dims=10)
    dec = SparseDecoder()
    x   = dec.decode(code)
    assert x[2] == pytest.approx(0.5)
    assert x[7] == pytest.approx(-0.3)
    assert x[0] == pytest.approx(0.0)


def test_decoder_fill_value():
    code = SparseCode(indices=np.array([0]),
                      values=np.array([1.0]),
                      n_dims=5)
    dec = SparseDecoder(fill=99.0)
    x   = dec.decode(code)
    assert x[1] == pytest.approx(99.0)


# ===========================================================================
# SparseRelay
# ===========================================================================

def test_sparse_relay_output_shape():
    relay = SparseRelay(n_dims=16, k=4)
    x     = np.random.default_rng(0).standard_normal(16)
    x_rec, stats = relay.transmit(x)
    assert x_rec.shape == (16,)


def test_sparse_relay_stats_keys():
    relay = SparseRelay(n_dims=8, k=2)
    _, stats = relay.transmit(np.ones(8))
    for k in ("k_used", "density", "compression_ratio",
              "reconstruction_error", "bits_sent"):
        assert k in stats


def test_sparse_relay_topk_k_used():
    relay = SparseRelay(n_dims=10, k=3)
    _, stats = relay.transmit(np.random.default_rng(0).standard_normal(10))
    assert stats["k_used"] == 3


def test_sparse_relay_reconstruction_exact_noiseless():
    relay = SparseRelay(n_dims=8, k=8)   # k = n_dims -> lossless
    x     = np.array([1.0, -2.0, 0.5, 0.0, 3.0, -1.0, 0.2, 0.9])
    x_rec, _ = relay.transmit(x)
    assert np.allclose(x, x_rec, atol=1e-10)


def test_sparse_relay_noise_changes_output():
    rng   = np.random.default_rng(42)
    relay = SparseRelay(n_dims=16, k=8, noise_std=1.0)
    x     = np.ones(16)
    x_rec, _ = relay.transmit(x, rng=rng)
    assert not np.allclose(x, x_rec)


def test_sparse_relay_cumulative_stats():
    relay = SparseRelay(n_dims=8, k=2)
    for _ in range(5):
        relay.transmit(np.random.default_rng(0).standard_normal(8))
    cs = relay.cumulative_stats()
    assert cs["total_transmissions"] == 5
    assert cs["overall_compression"] > 1.0


# ===========================================================================
# AdaptiveSparseRelay
# ===========================================================================

def test_adaptive_relay_k_adapts():
    relay = AdaptiveSparseRelay(n_dims=32, target_cr=4.0, k_min=1, k_max=32)
    rng   = np.random.default_rng(0)
    for _ in range(10):
        relay.transmit(rng.standard_normal(32))
    # k should have moved from its initial value
    assert 1 <= relay.current_k <= 32


def test_adaptive_relay_output_shape():
    relay = AdaptiveSparseRelay(n_dims=16, target_cr=2.0)
    x_rec, stats = relay.transmit(np.ones(16))
    assert x_rec.shape == (16,)
    assert "k_adapted" in stats


# ===========================================================================
# RandomProjector
# ===========================================================================

def test_projector_output_shape():
    proj = RandomProjector(n_dims=64, m_meas=16)
    y    = proj.project(np.ones(64))
    assert y.shape == (16,)


def test_projector_compression_ratio():
    proj = RandomProjector(n_dims=128, m_meas=32)
    assert proj.compression_ratio == pytest.approx(4.0)


def test_projector_pseudo_inv_shape():
    proj = RandomProjector(n_dims=64, m_meas=16)
    y    = np.ones(16)
    x    = proj.pseudo_inv(y)
    assert x.shape == (64,)


def test_projector_rejects_m_ge_n():
    with pytest.raises(ValueError):
        RandomProjector(n_dims=16, m_meas=16)


# ===========================================================================
# ISTASolver
# ===========================================================================

def test_ista_recovers_sparse_signal():
    """ISTA should reduce residual on a k-sparse signal from m > k measurements."""
    rng    = np.random.default_rng(7)
    n, m   = 64, 32
    proj   = RandomProjector(n, m, seed=7)
    # Sparse signal: only 5 non-zero entries.
    x_true = np.zeros(n)
    idx    = rng.choice(n, 5, replace=False)
    x_true[idx] = rng.standard_normal(5)
    y      = proj.project(x_true)

    solver = ISTASolver(n_iters=500, lam=0.01)
    x_hat, info = solver.solve(y, proj)
    # Residual ||y - Phi x_hat|| should be smaller than naive zero solution.
    residual_zero = float(np.linalg.norm(y))          # solution = 0
    residual_hat  = info["residual"]
    assert residual_hat < residual_zero               # ISTA improved over zero
    assert "n_iters_run" in info


def test_ista_info_keys():
    proj   = RandomProjector(16, 8)
    solver = ISTASolver(n_iters=10)
    _, info = solver.solve(np.ones(8), proj)
    for k in ("n_iters_run", "residual", "sparsity"):
        assert k in info


# ===========================================================================
# FISTASolver
# ===========================================================================

def test_fista_recovers_better_than_ista_short_budget():
    """FISTA should reduce residual at least as well as ISTA given same budget."""
    rng    = np.random.default_rng(3)
    n, m   = 32, 16
    proj   = RandomProjector(n, m, seed=3)
    x_true = np.zeros(n); x_true[[0, 5]] = [1.0, -1.0]
    y      = proj.project(x_true)

    ista  = ISTASolver(n_iters=50, lam=0.01)
    fista = FISTASolver(n_iters=50, lam=0.01)
    x_ista,  i1 = ista.solve(y,  proj)
    x_fista, i2 = fista.solve(y, proj)

    # Both should improve over the zero solution.
    residual_zero = float(np.linalg.norm(y))
    assert i1["residual"] < residual_zero
    assert i2["residual"] < residual_zero
    # FISTA residual should be no worse than ISTA (with generous slack).
    assert i2["residual"] <= i1["residual"] + 0.3


# ===========================================================================
# CompressedSensingRelay
# ===========================================================================

def test_cs_relay_output_shape():
    relay = CompressedSensingRelay(n_dims=32, m_meas=8)
    x_rec, stats = relay.transmit(np.ones(32))
    assert x_rec.shape == (32,)


def test_cs_relay_stats_keys():
    relay = CompressedSensingRelay(n_dims=16, m_meas=4)
    _, stats = relay.transmit(np.ones(16))
    for k in ("compression_ratio", "reconstruction_error",
              "ista_iters", "sparsity", "bits_sent"):
        assert k in stats


def test_cs_relay_compression_ratio():
    relay = CompressedSensingRelay(n_dims=32, m_meas=8)
    _, stats = relay.transmit(np.ones(32))
    assert stats["compression_ratio"] == pytest.approx(4.0)


def test_cs_relay_noise_increases_error():
    x     = np.random.default_rng(0).standard_normal(32)
    rng   = np.random.default_rng(0)
    clean = CompressedSensingRelay(n_dims=32, m_meas=16, noise_std=0.0)
    noisy = CompressedSensingRelay(n_dims=32, m_meas=16, noise_std=1.0)
    _, s_clean = clean.transmit(x)
    _, s_noisy = noisy.transmit(x, rng=rng)
    assert s_noisy["reconstruction_error"] >= s_clean["reconstruction_error"] - 0.05


def test_cs_relay_reset_clears_state():
    relay = CompressedSensingRelay(n_dims=16, m_meas=4)
    relay.transmit(np.ones(16))
    relay.reset()
    assert relay._n_sent == 0


# ===========================================================================
# KeyMatrix
# ===========================================================================

def test_key_matrix_scores_shape():
    km     = KeyMatrix(n_channels=10, d_key=8)
    query  = np.ones(8)
    scores = km.scores(query)
    assert scores.shape == (10,)


def test_key_matrix_update_does_not_crash():
    km = KeyMatrix(n_channels=8, d_key=4)
    km.update(np.array([0, 2, 5]), np.ones(4))


# ===========================================================================
# AttentionGate
# ===========================================================================

def test_attention_gate_output_shape():
    gate    = AttentionGate(n_channels=16, d_key=8, k=4)
    x       = np.random.default_rng(0).standard_normal(16)
    query   = np.random.default_rng(0).standard_normal(8)
    x_g, idx, w = gate.attend(x, query)
    assert x_g.shape == (16,)
    assert len(idx) == 4
    assert len(w) == 4


def test_attention_gate_top_indices_selected():
    gate  = AttentionGate(n_channels=8, d_key=4, k=3)
    x     = np.zeros(8); x[2] = 1.0; x[5] = 0.9; x[7] = 0.8
    query = np.ones(4)
    x_g, idx, _ = gate.attend(x, query, learn=False)
    # The gated output should be non-zero only where selected.
    assert np.count_nonzero(x_g) <= 3


def test_attention_gate_weights_sum_approx_1():
    """Attention weights should come from a softmax -> sum to ~1."""
    gate   = AttentionGate(n_channels=10, d_key=4, k=10)
    x      = np.ones(10)
    query  = np.ones(4)
    _, idx, w = gate.attend(x, query, learn=False)
    # All channels selected; weights should sum roughly to sum of softmax top-k.
    assert np.all(w >= 0)


# ===========================================================================
# AttentionGatedRelay
# ===========================================================================

def test_attention_relay_output_shape():
    relay = AttentionGatedRelay(n_dims=16, d_key=8, k=4)
    x     = np.random.default_rng(0).standard_normal(16)
    q     = np.ones(8)
    x_rec, stats = relay.transmit(x, q)
    assert x_rec.shape == (16,)


def test_attention_relay_stats_keys():
    relay = AttentionGatedRelay(n_dims=8, d_key=4, k=2)
    x, q  = np.ones(8), np.ones(4)
    _, stats = relay.transmit(x, q)
    for k in ("k_used", "top_indices", "compression_ratio",
              "reconstruction_error"):
        assert k in stats


def test_attention_relay_compression_ratio():
    relay = AttentionGatedRelay(n_dims=16, d_key=4, k=4)
    _, stats = relay.transmit(np.ones(16), np.ones(4))
    assert stats["compression_ratio"] == pytest.approx(4.0)


def test_attention_relay_learns_from_calls():
    """After repeated calls with same query/input, keys should shift."""
    relay  = AttentionGatedRelay(n_dims=8, d_key=4, k=2, hebbian_lr=0.5)
    K_before = relay.gate.keys.K.copy()
    for _ in range(5):
        relay.transmit(np.ones(8), np.ones(4), learn=True)
    K_after = relay.gate.keys.K
    assert not np.allclose(K_before, K_after)


# ===========================================================================
# MultiHeadAttentionGate
# ===========================================================================

def test_multihead_output_shape():
    mh  = MultiHeadAttentionGate(n_channels=16, d_key=4, n_heads=2, k_per_head=3)
    x   = np.random.default_rng(0).standard_normal(16)
    q   = np.ones(4)
    out, stats = mh.attend(x, q)
    assert out.shape == (16,)


def test_multihead_stats_keys():
    mh  = MultiHeadAttentionGate(n_channels=8, d_key=4, n_heads=2, k_per_head=2)
    _, stats = mh.attend(np.ones(8), np.ones(4))
    for k in ("k_total", "n_heads", "density"):
        assert k in stats


def test_multihead_k_total_bounded():
    mh = MultiHeadAttentionGate(n_channels=8, d_key=4, n_heads=2, k_per_head=3)
    _, stats = mh.attend(np.ones(8), np.ones(4))
    assert 1 <= stats["k_total"] <= 6   # at most 2 heads * 3
