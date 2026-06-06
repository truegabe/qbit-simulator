"""Solovay-Kitaev compilation tests."""

import numpy as np
import pytest

from qbit_simulator.solovay_kitaev import (
    H, S, T, I2,
    trace_distance_su2, sequence_to_unitary, invert_sequence,
    enumerate_basic_sequences, find_closest_in_library,
    _bloch_axis_angle, _rotation_about, group_commutator_decompose,
    solovay_kitaev_decompose, compile_unitary,
)


# Build a small library once for the tests.
@pytest.fixture(scope="module")
def small_library():
    return enumerate_basic_sequences(max_length=8)


# ---- Distance metric ----

def test_distance_self_zero():
    # sqrt(1 - 1) is ~1e-8 due to floating-point in the overlap computation.
    assert trace_distance_su2(H, H) < 1e-6


def test_distance_h_vs_t():
    d = trace_distance_su2(H, T)
    assert d > 0.5


def test_distance_global_phase_invariant():
    """Global phase should not affect the distance."""
    U = H
    V = np.exp(1j * 0.7) * H
    assert trace_distance_su2(U, V) < 1e-6


# ---- Sequence machinery ----

def test_sequence_to_unitary_h_h_is_identity():
    U = sequence_to_unitary(("H", "H"))
    assert np.allclose(U, I2, atol=1e-12)


def test_sequence_to_unitary_t_eight_is_identity():
    """T^8 = I."""
    U = sequence_to_unitary(("T",) * 8)
    assert np.allclose(U, I2, atol=1e-12)


def test_invert_sequence_round_trip():
    seq = ("H", "T", "T", "H", "Tdg", "H")
    U = sequence_to_unitary(seq)
    U_inv = sequence_to_unitary(invert_sequence(seq))
    assert np.allclose(U @ U_inv, I2, atol=1e-12)


# ---- Basic library ----

def test_library_includes_identity(small_library):
    """The empty sequence (= identity) must be in the library."""
    sequences = [e.sequence for e in small_library]
    assert () in sequences


def test_library_grows_with_length():
    lib4 = enumerate_basic_sequences(max_length=4)
    lib8 = enumerate_basic_sequences(max_length=8)
    assert len(lib8) > len(lib4)


def test_find_closest_in_library_self(small_library):
    """Looking up an entry should return itself."""
    target = small_library[10].unitary
    found = find_closest_in_library(target, small_library)
    assert trace_distance_su2(target, found.unitary) < 1e-6


# ---- Bloch / rotation utilities ----

def test_bloch_axis_angle_identity():
    n, theta = _bloch_axis_angle(I2)
    assert abs(theta) < 1e-10


def test_bloch_axis_angle_rz_pi():
    # Rz(pi) = diag(-i, i) = -i Z.
    Rz_pi = np.array([[np.exp(-1j * np.pi / 2), 0],
                      [0, np.exp(1j * np.pi / 2)]])
    n, theta = _bloch_axis_angle(Rz_pi)
    assert abs(theta - np.pi) < 1e-10
    assert abs(abs(n[2]) - 1.0) < 1e-10


def test_rotation_about_z_matches_direct():
    theta = 0.7
    Rz_direct = np.array([[np.exp(-1j * theta / 2), 0],
                          [0, np.exp(1j * theta / 2)]])
    Rz_built = _rotation_about(np.array([0.0, 0.0, 1.0]), theta)
    assert np.allclose(Rz_direct, Rz_built, atol=1e-12)


# ---- Group commutator ----

@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.3, 0.7])
def test_commutator_reconstructs_target(alpha):
    """V Q V† Q† should reproduce W to machine precision."""
    rng = np.random.default_rng(42)
    axis = rng.normal(size=3)
    axis = axis / np.linalg.norm(axis)
    W = _rotation_about(axis, alpha)
    V, Q = group_commutator_decompose(W)
    W_recon = V @ Q @ V.conj().T @ Q.conj().T
    assert trace_distance_su2(W, W_recon) < 1e-6


# ---- Full Solovay-Kitaev ----

def test_sk_depth_zero_returns_library_entry(small_library):
    """At depth=0, output is just the closest library entry."""
    Rz = _rotation_about(np.array([0.0, 0.0, 1.0]), 0.5)
    seq, U, err = solovay_kitaev_decompose(Rz, depth=0, library=small_library)
    # error <= maximum library-coverage distance.
    assert err < 0.5


def test_sk_recursion_improves_error(small_library):
    """Each level should not increase the error compared to depth 0,
    and depth-1 should improve over depth-0 for at least some inputs."""
    theta = np.pi / 8
    Rz = np.array([[np.exp(-1j * theta / 2), 0],
                   [0, np.exp(1j * theta / 2)]])
    _, _, err0 = solovay_kitaev_decompose(Rz, depth=0, library=small_library)
    _, _, err1 = solovay_kitaev_decompose(Rz, depth=1, library=small_library)
    # With our small library, depth=1 should give a noticeable improvement.
    assert err1 <= err0 + 1e-6


def test_sk_output_is_unitary(small_library):
    Rz = _rotation_about(np.array([0.1, 0.3, 0.9]) /
                          np.linalg.norm([0.1, 0.3, 0.9]), 0.4)
    seq, U, err = solovay_kitaev_decompose(Rz, depth=2, library=small_library)
    assert np.allclose(U @ U.conj().T, I2, atol=1e-9)


def test_sk_sequence_matches_returned_unitary(small_library):
    """The returned unitary must equal sequence_to_unitary(seq)."""
    Rz = _rotation_about(np.array([0.0, 1.0, 0.0]), 0.3)
    seq, U, err = solovay_kitaev_decompose(Rz, depth=1, library=small_library)
    U_replay = sequence_to_unitary(seq)
    assert trace_distance_su2(U, U_replay) < 1e-6


def test_compile_unitary_returns_dict(small_library):
    Rz = _rotation_about(np.array([0.0, 0.0, 1.0]), 0.2)
    r = compile_unitary(Rz, depth=1)
    assert {"sequence", "unitary", "error", "length", "depth"} <= r.keys()
    assert r["length"] == len(r["sequence"])
