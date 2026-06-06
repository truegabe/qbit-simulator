"""HaPPY code tests."""

import numpy as np
import pytest
from itertools import combinations

from qbit_simulator.algorithms.happy_code import (
    perfect_tensor_5q, happy_encode_one_tile,
    logical_z_expectation, region_distinguishes_logical,
    boundary_reduced_density, happy_2_tile_encoder, is_isometry,
    _build_513_codewords,
)


# ---- Perfect tensor ----

def test_perfect_tensor_shape():
    V = perfect_tensor_5q()
    assert V.shape == (32, 2)


def test_perfect_tensor_is_isometry():
    V = perfect_tensor_5q()
    assert is_isometry(V)


def test_513_codewords_orthogonal():
    c0, c1 = _build_513_codewords()
    overlap = abs(np.vdot(c0, c1))
    assert overlap < 1e-9


def test_513_codewords_normalized():
    c0, c1 = _build_513_codewords()
    assert abs(np.linalg.norm(c0) - 1.0) < 1e-10
    assert abs(np.linalg.norm(c1) - 1.0) < 1e-10


# ---- Encoding ----

def test_encoding_normalizes():
    psi = np.array([1, 0], dtype=complex)
    encoded = happy_encode_one_tile(psi)
    assert abs(np.linalg.norm(encoded) - 1.0) < 1e-10


def test_encoding_rejects_wrong_size():
    with pytest.raises(ValueError):
        happy_encode_one_tile(np.zeros(4, dtype=complex))


def test_encoding_zero_state_is_codeword_zero():
    c0, _ = _build_513_codewords()
    encoded = happy_encode_one_tile(np.array([1, 0], dtype=complex))
    assert np.allclose(encoded, c0, atol=1e-12)


# ---- Logical Z ----

@pytest.mark.parametrize("a,b,expected_z", [
    (1.0, 0.0, +1.0),
    (0.0, 1.0, -1.0),
    (0.6, 0.8, -0.28),
    (1/np.sqrt(2), 1/np.sqrt(2), 0.0),
])
def test_logical_z_matches_bulk(a, b, expected_z):
    """Z_L on the boundary should reproduce the bulk Z expectation."""
    psi = np.array([a, b], dtype=complex)
    psi = psi / np.linalg.norm(psi)
    encoded = happy_encode_one_tile(psi)
    z = logical_z_expectation(encoded)
    assert abs(z - expected_z) < 1e-9


# ---- Region reconstruction (Ryu-Takayanagi pattern) ----

def test_small_regions_cannot_distinguish_logicals():
    """For the [[5,1,3]] code, regions of size ≤ 2 give identical
    reduced density matrices for |0_L⟩ and |1_L⟩."""
    for size in (1, 2):
        for region in combinations(range(5), size):
            r = region_distinguishes_logical(list(region))
            assert not r["can_reconstruct"], (
                f"region {region} of size {size} unexpectedly distinguishes")


def test_large_regions_can_distinguish_logicals():
    """Regions of size ≥ 3 should distinguish."""
    for size in (3, 4, 5):
        for region in combinations(range(5), size):
            r = region_distinguishes_logical(list(region))
            assert r["can_reconstruct"], (
                f"region {region} of size {size} failed to distinguish")


def test_full_boundary_perfectly_distinguishes():
    """The full 5-qubit boundary should give rho_diff = 2 (orthogonal
    pure states distance)."""
    r = region_distinguishes_logical([0, 1, 2, 3, 4])
    # |c0⟩ and |c1⟩ are orthogonal pure states → ||ρ_0 - ρ_1||_2 = √2.
    assert abs(r["rho_diff_norm"] - np.sqrt(2)) < 1e-9


# ---- Reduced density matrices ----

def test_boundary_reduced_density_trace_one():
    psi = happy_encode_one_tile(np.array([1, 0], dtype=complex))
    rho = boundary_reduced_density(psi, region=[0, 1], n_total=5)
    assert abs(np.trace(rho).real - 1.0) < 1e-10


def test_boundary_reduced_density_psd():
    psi = happy_encode_one_tile(np.array([0.6, 0.8], dtype=complex))
    rho = boundary_reduced_density(psi, region=[0, 1, 2], n_total=5)
    eigs = np.linalg.eigvalsh(rho)
    assert eigs.min() >= -1e-10


# ---- 2-tile encoder ----

def test_2_tile_encoder_shape():
    V2 = happy_2_tile_encoder()
    assert V2.shape == (256, 4)
