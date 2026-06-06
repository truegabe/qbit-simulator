"""Union-Find (greedy BFS) decoder tests."""

import numpy as np
import pytest

from qbit_simulator.surface_mwpm import SurfaceCodeD3
from qbit_simulator.algorithms.union_find_decoder import (
    UnionFindDecoder, _UnionFind, simulate_uf_logical_error_rate,
)


@pytest.fixture(scope="module")
def code():
    return SurfaceCodeD3()


@pytest.fixture(scope="module")
def decoder(code):
    return UnionFindDecoder(code)


# ---- Union-Find data structure ----

def test_union_find_init():
    uf = _UnionFind(["a", "b", "c"], lit={"a"})
    assert uf.find("a") == "a"
    assert uf.find("b") == "b"
    assert uf.parity["a"] == 1
    assert uf.parity["b"] == 0


def test_union_merges_clusters():
    uf = _UnionFind(["a", "b", "c"], lit=set())
    uf.union("a", "b")
    assert uf.find("a") == uf.find("b")


def test_union_parity_xor():
    uf = _UnionFind(["a", "b"], lit={"a"})
    # a has parity 1, b has parity 0; union has parity 1 ⊕ 0 = 1.
    uf.union("a", "b")
    r = uf.find("a")
    assert uf.parity[r] == 1


def test_cluster_is_even():
    uf = _UnionFind(["a", "b"], lit={"a", "b"})
    # Both lit, union has parity 1 ⊕ 1 = 0.
    uf.union("a", "b")
    assert uf.cluster_is_even("a")
    assert uf.cluster_is_even("b")


# ---- Decoder construction ----

def test_decoder_builds_adjacency(decoder):
    assert decoder._x_adj is not None
    assert decoder._z_adj is not None


# ---- Empty syndrome ----

def test_decode_empty_syndrome(decoder):
    assert decoder.decode_z_errors(frozenset()) == set()
    assert decoder.decode_x_errors(frozenset()) == set()


# ---- Single-qubit error correction ----

@pytest.mark.parametrize("q", list(range(9)))
def test_decode_single_z_error_no_logical(code, decoder, q):
    """Single Z error → corrected without logical flip."""
    z_err = {q}
    syndrome = code.z_error_syndrome(z_err)
    correction = decoder.decode_z_errors(syndrome)
    residual = z_err ^ correction
    # Residual must commute with logical X support (even overlap).
    overlap = len(residual & set(code.logical_x)) % 2
    assert overlap == 0


@pytest.mark.parametrize("q", list(range(9)))
def test_decode_single_x_error_no_logical(code, decoder, q):
    x_err = {q}
    syndrome = code.x_error_syndrome(x_err)
    correction = decoder.decode_x_errors(syndrome)
    residual = x_err ^ correction
    overlap = len(residual & set(code.logical_z)) % 2
    assert overlap == 0


# ---- Threshold behavior ----

def test_uf_low_physical_low_logical(code, decoder):
    rng = np.random.default_rng(0)
    r = simulate_uf_logical_error_rate(code, decoder, p_physical=0.005,
                                          n_trials=5000, rng=rng)
    assert r["p_logical"] < 0.005


def test_uf_logical_error_grows_with_p(code, decoder):
    rng = np.random.default_rng(0)
    r1 = simulate_uf_logical_error_rate(code, decoder, 0.01, 1000, rng)
    r2 = simulate_uf_logical_error_rate(code, decoder, 0.1, 1000, rng)
    assert r2["p_logical"] > r1["p_logical"]


def test_uf_matches_mwpm_qualitatively(code, decoder):
    """UF on d=3 should track MWPM closely (both achieve near-optimal
    decoding on small lattices). Allow loose agreement."""
    from qbit_simulator.surface_mwpm import (
        MWPMDecoder, simulate_logical_error_rate as mwpm_sim,
    )
    mwpm = MWPMDecoder(code)
    rng = np.random.default_rng(0)
    r_uf = simulate_uf_logical_error_rate(code, decoder, 0.05, 2000, rng)
    rng = np.random.default_rng(0)
    r_mw = mwpm_sim(code, mwpm, 0.05, 2000, rng)
    # Should be within 50% of MWPM.
    rel = abs(r_uf["p_logical"] - r_mw["p_logical"]) / max(r_mw["p_logical"], 1e-6)
    assert rel < 0.5


# ---- Two-qubit errors ----

def test_decode_two_separate_z_errors(code, decoder):
    """Two non-adjacent Z errors should still be correctable."""
    z_err = {0, 8}
    syndrome = code.z_error_syndrome(z_err)
    correction = decoder.decode_z_errors(syndrome)
    residual = z_err ^ correction
    overlap = len(residual & set(code.logical_x)) % 2
    assert overlap == 0
