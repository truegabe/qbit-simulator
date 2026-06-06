"""Quantum-simulator lifetime log — `qsim_log.jsonl`.

Mirrors the brain's `clock.json` idea, but for computational events instead
of cognitive ones. Every algorithm run records:

    {
      "timestamp": <unix>,
      "kind": "grover" | "qft" | "shor" | "vqe" | "teleport" | "chsh" | ...,
      "n_qubits": int,
      "wall_seconds": float,
      "peak_heap_mb": float,
      "result_summary": dict,  # e.g. {"reached": True, "factors": [3, 5]}
      "notes": dict             # free-form metadata
    }

Persists in `qbit_simulator/data/qsim_log.jsonl` (append-only).

Provides aggregate views: `lifetime_stats()`, `kind_counts()`,
`biggest_circuit()`, `total_compute_seconds()`, etc.
"""

from __future__ import annotations

import json
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


DEFAULT_LOG_PATH = Path(__file__).parent / "data" / "qsim_log.jsonl"


@dataclass
class SimEvent:
    timestamp: float
    kind: str
    n_qubits: int
    wall_seconds: float
    peak_heap_mb: float
    result_summary: dict = field(default_factory=dict)
    notes: dict = field(default_factory=dict)


class SimLog:
    """Append-only event log + lifetime aggregator."""

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path is not None else DEFAULT_LOG_PATH
        self._cache: list[SimEvent] | None = None

    # ---- recording ----

    def append(self, event: SimEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(event)) + "\n")
        self._cache = None  # invalidate

    def record(
        self,
        kind: str,
        n_qubits: int,
        wall_seconds: float,
        peak_heap_mb: float = 0.0,
        result_summary: dict | None = None,
        notes: dict | None = None,
    ) -> SimEvent:
        ev = SimEvent(
            timestamp=time.time(),
            kind=kind,
            n_qubits=int(n_qubits),
            wall_seconds=float(wall_seconds),
            peak_heap_mb=float(peak_heap_mb),
            result_summary=result_summary or {},
            notes=notes or {},
        )
        self.append(ev)
        return ev

    @contextmanager
    def measure(self, kind: str, n_qubits: int, **notes):
        """Context manager: times the body, captures peak heap, records the event.

        Usage:
            with simlog.measure("grover", n_qubits=10) as ev:
                run_grover(...)
                ev["result_summary"] = {"reached": True}
        """
        t0 = time.perf_counter()
        tracemalloc.start()
        bag: dict[str, Any] = {"result_summary": {}}
        try:
            yield bag
        finally:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            dt = time.perf_counter() - t0
            self.record(
                kind=kind,
                n_qubits=n_qubits,
                wall_seconds=dt,
                peak_heap_mb=peak / 1024**2,
                result_summary=bag.get("result_summary", {}),
                notes=notes,
            )

    # ---- reading ----

    def all(self) -> list[SimEvent]:
        if self._cache is not None:
            return self._cache
        events: list[SimEvent] = []
        if not self.path.exists():
            self._cache = events
            return events
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(SimEvent(**json.loads(line)))
                except Exception:
                    pass
        self._cache = events
        return events

    def recent(self, n: int = 10) -> list[SimEvent]:
        return self.all()[-n:]

    def by_kind(self, kind: str) -> list[SimEvent]:
        return [e for e in self.all() if e.kind == kind]

    # ---- aggregates ----

    def kind_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self.all():
            out[e.kind] = out.get(e.kind, 0) + 1
        return out

    def biggest_circuit(self) -> SimEvent | None:
        evs = self.all()
        return max(evs, key=lambda e: e.n_qubits) if evs else None

    def total_compute_seconds(self) -> float:
        return sum(e.wall_seconds for e in self.all())

    def lifetime_stats(self) -> dict:
        evs = self.all()
        if not evs:
            return {
                "total_runs": 0,
                "kinds": {},
                "total_compute_seconds": 0.0,
                "biggest_n_qubits": 0,
                "peak_heap_mb_ever": 0.0,
                "first_run": None,
                "last_run": None,
            }
        return {
            "total_runs": len(evs),
            "kinds": self.kind_counts(),
            "total_compute_seconds": round(sum(e.wall_seconds for e in evs), 3),
            "biggest_n_qubits": max(e.n_qubits for e in evs),
            "peak_heap_mb_ever": round(max(e.peak_heap_mb for e in evs), 2),
            "first_run": evs[0].timestamp,
            "last_run": evs[-1].timestamp,
        }

    def report(self) -> str:
        s = self.lifetime_stats()
        if s["total_runs"] == 0:
            return "(no quantum runs logged yet)"
        lines = ["Quantum simulator lifetime stats:"]
        lines.append(f"  total runs:           {s['total_runs']}")
        lines.append(f"  total compute time:   {s['total_compute_seconds']:.2f}s")
        lines.append(f"  biggest circuit:      N = {s['biggest_n_qubits']} qubits")
        lines.append(f"  peak heap ever:       {s['peak_heap_mb_ever']:.1f} MB")
        lines.append(f"  by algorithm:")
        for k, n in sorted(s["kinds"].items(), key=lambda kv: -kv[1]):
            lines.append(f"    {k:<14} {n:>4}")
        first = time.strftime("%Y-%m-%d %H:%M:%S",
                              time.localtime(s["first_run"]))
        last = time.strftime("%Y-%m-%d %H:%M:%S",
                             time.localtime(s["last_run"]))
        lines.append(f"  first run:            {first}")
        lines.append(f"  last run:             {last}")
        return "\n".join(lines)
