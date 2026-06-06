"""Circuit memoization cache — per-entry file store.

Stores `(algorithm_name, args) → final_state_vector` so repeated runs of
the same algorithm with the same parameters become instant lookups.

Layout on disk:
  qbit_simulator/data/qsim_cache/
    index.json          # ordered list of {key, file, nbytes, ts}
    <hash>.npy          # one numpy file per cached state vector

Why per-entry files (instead of one big .npz):
  * A single experiment state can be up to ~2 GB. Bundling everything
    into one archive means a 2 GB rewrite on every put().
  * Eviction = unlink a file. O(1) on disk.
  * Lazy load: on a hit, we mmap/np.load just the one entry, not the
    whole cache.

Eviction: LRU. Caps at `max_entries` (default 200) AND `max_bytes`
(default 8 GB). Whichever is hit first evicts the LRU entry.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_CACHE_DIR = Path(__file__).parent / "data" / "qsim_cache"
DEFAULT_MAX_ENTRIES = 200
DEFAULT_MAX_BYTES = 8 * 1024 ** 3   # 8 GB


def _canonicalize_args(args: dict) -> str:
    """Make a stable string key from an args dict. Floats are coerced to fixed
    precision so 0.1 + 0.2 hashes the same as 0.3."""
    def _norm(v):
        if isinstance(v, float):
            return round(v, 12)
        if isinstance(v, (list, tuple)):
            return [_norm(x) for x in v]
        return v
    return json.dumps({k: _norm(v) for k, v in sorted(args.items())},
                      sort_keys=True)


def _make_key(name: str, args: dict) -> str:
    return f"{name}::{_canonicalize_args(args)}"


def _file_for(key: str) -> str:
    """Filesystem-safe filename for a key."""
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:20]
    return f"{h}.npy"


class CircuitCache:
    """LRU-evicting state-vector cache, one file per entry on disk."""

    def __init__(
        self,
        path: Path | str | None = None,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        max_bytes: int = DEFAULT_MAX_BYTES,
        autoload: bool = True,
    ):
        # `path` can be a directory (preferred) or a .npz/file path for
        # backward compatibility — in the latter case we use its parent +
        # stem as the directory name.
        p = Path(path) if path is not None else DEFAULT_CACHE_DIR
        if p.suffix:
            # legacy file-style path — convert to directory next to it.
            p = p.with_suffix("")
        self.dir = p
        self.index_path = self.dir / "index.json"
        self.max_entries = max_entries
        self.max_bytes = max_bytes

        # In-memory: key -> (filename, nbytes, loaded_array_or_None)
        self._cache: OrderedDict[str, dict] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._bytes = 0
        self._dirty = False

        if autoload:
            self.load()
        atexit.register(self._on_exit)

    # ---- core ----

    def get(self, name: str, args: dict | None = None) -> np.ndarray | None:
        """Return the cached state vector (a copy) or None on miss."""
        key = _make_key(name, args or {})
        entry = self._cache.get(key)
        if entry is None:
            self._misses += 1
            return None
        self._hits += 1
        self._cache.move_to_end(key)
        self._dirty = True
        arr = entry.get("arr")
        if arr is None:
            # Lazy load from disk.
            fp = self.dir / entry["file"]
            try:
                arr = np.load(fp)
                entry["arr"] = arr
            except Exception:
                # File missing/corrupt — treat as miss, drop entry.
                self._bytes -= entry.get("nbytes", 0)
                del self._cache[key]
                self._misses += 1
                self._hits -= 1
                return None
        return arr.copy()

    def put(self, name: str, args: dict | None, state: np.ndarray) -> None:
        """Cache a state vector under (name, args), writing it to disk."""
        key = _make_key(name, args or {})
        arr = np.ascontiguousarray(state)

        if key in self._cache:
            # Replace existing.
            old = self._cache.pop(key)
            self._bytes -= old.get("nbytes", 0)
            try:
                (self.dir / old["file"]).unlink(missing_ok=True)
            except Exception:
                pass

        fname = _file_for(key)
        self.dir.mkdir(parents=True, exist_ok=True)
        try:
            np.save(self.dir / fname, arr)
        except Exception:
            # If we can't persist, still keep in memory.
            pass

        self._cache[key] = {"file": fname, "nbytes": int(arr.nbytes), "arr": arr}
        self._bytes += int(arr.nbytes)
        self._dirty = True
        self._evict_if_needed()
        self._write_index()

    def _evict_if_needed(self) -> None:
        while (len(self._cache) > self.max_entries
               or self._bytes > self.max_bytes):
            if not self._cache:
                break
            old_key, old_entry = self._cache.popitem(last=False)
            self._bytes -= old_entry.get("nbytes", 0)
            try:
                (self.dir / old_entry["file"]).unlink(missing_ok=True)
            except Exception:
                pass

    # ---- stats ----

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "entries": len(self._cache),
            "size_mb": round(self._bytes / 1024**2, 3),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 4) if total > 0 else 0.0,
            "max_entries": self.max_entries,
            "max_size_mb": round(self.max_bytes / 1024**2, 1),
        }

    def report(self) -> str:
        s = self.stats()
        return (f"Circuit cache: {s['entries']}/{s['max_entries']} entries, "
                f"{s['size_mb']} / {s['max_size_mb']} MB\n"
                f"  hits: {s['hits']}, misses: {s['misses']}, "
                f"hit_rate: {s['hit_rate']*100:.1f}%")

    def keys(self) -> list[str]:
        return list(self._cache.keys())

    def clear(self) -> None:
        for entry in self._cache.values():
            try:
                (self.dir / entry["file"]).unlink(missing_ok=True)
            except Exception:
                pass
        self._cache.clear()
        self._bytes = 0
        self._dirty = True
        self._write_index()

    # ---- persistence ----

    def _write_index(self) -> None:
        if not self.dir.exists():
            try:
                self.dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                return
        idx = [{"key": k, "file": e["file"], "nbytes": e["nbytes"]}
               for k, e in self._cache.items()]
        try:
            self.index_path.write_text(json.dumps(idx))
        except Exception:
            pass

    def save(self) -> None:
        """Flush the index. (State files are written on put().)"""
        if not self._dirty:
            return
        self._write_index()
        self._dirty = False

    def load(self) -> None:
        if not self.index_path.exists():
            return
        try:
            idx = json.loads(self.index_path.read_text())
        except Exception:
            return
        for e in idx:
            key = e["key"]
            self._cache[key] = {"file": e["file"],
                                "nbytes": int(e.get("nbytes", 0)),
                                "arr": None}     # lazy
            self._bytes += int(e.get("nbytes", 0))
        self._evict_if_needed()

    def _on_exit(self) -> None:
        try:
            self.save()
        except Exception:
            pass


# ---- convenience: run-with-cache helper ----

def run_cached(
    cache: CircuitCache,
    name: str,
    args: dict,
    build_fn,
):
    """If `(name, args)` is in cache, return the cached state. Otherwise call
    `build_fn()` (which must return a QuantumCircuit), cache its state, and
    return it.

    On a cache hit we return a `_CachedResult` wrapper; on a miss we return
    the actual QuantumCircuit produced by build_fn.
    """
    cached = cache.get(name, args)
    if cached is not None:
        return _CachedResult(state=cached, n=int(np.log2(len(cached))))
    qc = build_fn()
    cache.put(name, args, qc.state)
    return qc


class _CachedResult:
    """Lightweight stand-in for a QuantumCircuit when only the state matters."""

    def __init__(self, state: np.ndarray, n: int):
        self.state = state
        self.n = n
        self.history: list[str] = ["(restored from cache)"]
        self._ops: list = []

    def probabilities(self) -> np.ndarray:
        return np.abs(self.state) ** 2

    def measure_all(self, shots: int = 1, rng: np.random.Generator | None = None) -> np.ndarray:
        from .measure import sample
        return sample(self.probabilities(), shots=shots, rng=rng)

    def counts(self, shots: int = 1024, rng: np.random.Generator | None = None) -> dict[str, int]:
        outcomes = self.measure_all(shots=shots, rng=rng)
        out: dict[str, int] = {}
        for o in outcomes:
            key = format(int(o), f"0{self.n}b")
            out[key] = out.get(key, 0) + 1
        return out
