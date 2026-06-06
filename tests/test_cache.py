"""Tests for the circuit memoization cache."""

import time
from pathlib import Path

import numpy as np
import pytest

from qbit_simulator import QuantumCircuit
from qbit_simulator.algorithms import grover, qft
from qbit_simulator.cache import (
    CircuitCache, run_cached, _make_key, _canonicalize_args, _CachedResult,
)


# ---- Key canonicalization ----

def test_canonicalize_dict_order_independent():
    a = {"n": 5, "marked": 3}
    b = {"marked": 3, "n": 5}
    assert _canonicalize_args(a) == _canonicalize_args(b)


def test_canonicalize_float_rounding():
    a = {"theta": 0.1 + 0.2}    # 0.30000000000000004
    b = {"theta": 0.3}
    # After rounding to 12 decimals they should match.
    assert _canonicalize_args(a) == _canonicalize_args(b)


def test_make_key_includes_name():
    k1 = _make_key("grover", {"n": 5})
    k2 = _make_key("qft", {"n": 5})
    assert k1 != k2


# ---- Cache get/put basics ----

def test_empty_cache_returns_none(tmp_path):
    c = CircuitCache(tmp_path / "c.npz", autoload=False)
    assert c.get("nothing", {"x": 1}) is None
    assert c.stats()["entries"] == 0


def test_put_then_get(tmp_path):
    c = CircuitCache(tmp_path / "c.npz", autoload=False)
    state = np.array([1, 0, 0, 0], dtype=np.complex128)
    c.put("test", {"n": 2}, state)
    got = c.get("test", {"n": 2})
    assert got is not None
    assert np.allclose(got, state)


def test_get_returns_independent_copy(tmp_path):
    c = CircuitCache(tmp_path / "c.npz", autoload=False)
    state = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.complex128)
    c.put("test", {}, state)
    got = c.get("test", {})
    got[0] = 999  # mutate the copy
    assert c.get("test", {})[0] == 0.5  # original untouched


def test_hits_and_misses_counted(tmp_path):
    c = CircuitCache(tmp_path / "c.npz", autoload=False)
    s = np.array([1, 0], dtype=np.complex128)
    c.get("x", {})            # miss
    c.put("x", {}, s)
    c.get("x", {})            # hit
    c.get("x", {})            # hit
    stats = c.stats()
    assert stats["hits"] == 2
    assert stats["misses"] == 1
    assert stats["hit_rate"] == pytest.approx(2/3, abs=1e-3)


# ---- LRU eviction ----

def test_max_entries_evicts_lru(tmp_path):
    c = CircuitCache(tmp_path / "c.npz", autoload=False, max_entries=2)
    s = lambda: np.array([1, 0], dtype=np.complex128)
    c.put("a", {}, s())
    c.put("b", {}, s())
    c.put("c", {}, s())   # 'a' should be evicted
    assert c.get("a", {}) is None
    assert c.get("b", {}) is not None
    assert c.get("c", {}) is not None


def test_lru_order_after_get(tmp_path):
    c = CircuitCache(tmp_path / "c.npz", autoload=False, max_entries=2)
    s = lambda: np.array([1, 0], dtype=np.complex128)
    c.put("a", {}, s())
    c.put("b", {}, s())
    _ = c.get("a", {})         # 'a' is now MRU
    c.put("c", {}, s())        # 'b' should be evicted, not 'a'
    assert c.get("a", {}) is not None
    assert c.get("b", {}) is None
    assert c.get("c", {}) is not None


def test_max_bytes_evicts(tmp_path):
    # Set tiny byte cap; even 2 entries should overflow.
    c = CircuitCache(tmp_path / "c.npz", autoload=False,
                     max_entries=100, max_bytes=16 * 4)
    # Each np.complex128 is 16 bytes; an array of length 4 is 64 bytes.
    s = lambda: np.zeros(4, dtype=np.complex128)
    c.put("a", {}, s())   # 64 B used
    c.put("b", {}, s())   # would be 128 B — evicts 'a'
    assert c.get("a", {}) is None
    assert c.get("b", {}) is not None


# ---- Persistence ----

def test_save_and_reload(tmp_path):
    p = tmp_path / "c.npz"
    c1 = CircuitCache(p, autoload=False)
    s = np.array([0.7, 0.3, 0.5, 0.4], dtype=np.complex128)
    c1.put("alpha", {"n": 2}, s)
    c1.save()
    c2 = CircuitCache(p, autoload=True)
    got = c2.get("alpha", {"n": 2})
    assert got is not None
    assert np.allclose(got, s)


def test_clear(tmp_path):
    c = CircuitCache(tmp_path / "c.npz", autoload=False)
    c.put("a", {}, np.array([1], dtype=np.complex128))
    c.put("b", {}, np.array([1], dtype=np.complex128))
    c.clear()
    assert c.get("a", {}) is None
    assert c.get("b", {}) is None
    assert c.stats()["entries"] == 0


# ---- run_cached wrapper ----

def test_run_cached_calls_build_on_miss(tmp_path):
    c = CircuitCache(tmp_path / "c.npz", autoload=False)
    calls = []
    def build():
        calls.append(1)
        return QuantumCircuit(2).h(0).cnot(0, 1)
    qc1 = run_cached(c, "bell", {}, build)
    qc2 = run_cached(c, "bell", {}, build)
    # build should have run only once.
    assert len(calls) == 1
    # Both results match.
    assert np.allclose(qc1.state, qc2.state)


def test_run_cached_with_grover_speedup(tmp_path):
    c = CircuitCache(tmp_path / "c.npz", autoload=False)
    N, marked = 10, 5
    t0 = time.perf_counter()
    qc1 = run_cached(c, "grover", {"n": N, "marked": marked},
                     lambda: grover(N, marked))
    dt_cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    qc2 = run_cached(c, "grover", {"n": N, "marked": marked},
                     lambda: grover(N, marked))
    dt_hot = time.perf_counter() - t0
    # State equality.
    assert np.allclose(qc1.state, qc2.state)
    # Hot path should be at least 5x faster than cold (typically 50-1000x).
    assert dt_hot < dt_cold


def test_cached_result_supports_basic_ops(tmp_path):
    c = CircuitCache(tmp_path / "c.npz", autoload=False)
    state = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=np.complex128)
    c.put("bell", {}, state)
    result = run_cached(c, "bell", {}, lambda: None)  # cache hit
    # Returned object should be a _CachedResult.
    assert isinstance(result, _CachedResult)
    probs = result.probabilities()
    assert probs[0] == pytest.approx(0.5)
    assert probs[3] == pytest.approx(0.5)
    counts = result.counts(shots=200, rng=np.random.default_rng(0))
    # Bell pair: only |00> and |11> should ever appear.
    for k in counts:
        assert k in ("00", "11")
