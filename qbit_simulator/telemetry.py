"""Telemetry / logging for the simulator.

Tracks per-circuit and per-session resource use so you can compare runs and
spot scaling problems. Everything is opt-in: no overhead unless you create
a `Logger` and attach it to a `QuantumCircuit`.

What's recorded:
  - N qubits, state vector bytes
  - Per-operation: name, target qubits, wall-clock time
  - Total ops, circuit depth
  - Peak Python-allocated memory during the run (via `tracemalloc`)
  - Wall-clock total

Storage: in-memory dicts; can be exported to JSON.
"""

from __future__ import annotations

import json
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class CircuitRecord:
    label: str
    n_qubits: int
    state_bytes: int
    operations: list[dict] = field(default_factory=list)
    total_ops: int = 0
    wall_time_s: float = 0.0
    peak_python_mem_bytes: int = 0
    notes: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "label": self.label,
            "n_qubits": self.n_qubits,
            "state_bytes": self.state_bytes,
            "state_MB": round(self.state_bytes / 1024**2, 4),
            "total_ops": self.total_ops,
            "wall_time_s": round(self.wall_time_s, 4),
            "peak_python_mem_MB": round(self.peak_python_mem_bytes / 1024**2, 2),
            "notes": self.notes,
        }


class Logger:
    """Session-level logger holding one or more circuit records."""

    def __init__(self) -> None:
        self.records: list[CircuitRecord] = []
        self._active: CircuitRecord | None = None
        self._t_op_start: float | None = None
        self._t_run_start: float | None = None
        self._tracemalloc_started = False

    # ---- session ----

    @contextmanager
    def record(self, label: str, qc, **notes):
        """Context manager that wraps a circuit execution block.

        The QuantumCircuit `qc` should already exist (so we can read N and
        state size). Operations performed inside the block are timed.
        """
        rec = CircuitRecord(
            label=label,
            n_qubits=qc.n,
            state_bytes=qc.state.nbytes,
            notes=dict(notes),
        )
        self.records.append(rec)
        self._active = rec
        qc._logger = self  # type: ignore[attr-defined]

        started_here = not self._tracemalloc_started
        if started_here:
            tracemalloc.start()
            self._tracemalloc_started = True

        t0 = time.perf_counter()
        try:
            yield rec
        finally:
            rec.wall_time_s = time.perf_counter() - t0
            current, peak = tracemalloc.get_traced_memory()
            rec.peak_python_mem_bytes = peak
            if started_here:
                tracemalloc.stop()
                self._tracemalloc_started = False
            rec.state_bytes = qc.state.nbytes  # may have changed
            rec.total_ops = len(rec.operations)
            self._active = None
            if hasattr(qc, "_logger"):
                qc._logger = None  # type: ignore[attr-defined]

    # ---- per-op (called from QuantumCircuit) ----

    def log_op(self, name: str, targets: list[int], duration_s: float) -> None:
        if self._active is None:
            return
        self._active.operations.append({
            "name": name,
            "targets": targets,
            "duration_s": duration_s,
        })

    # ---- export ----

    def summary_table(self) -> str:
        if not self.records:
            return "(no records)"
        headers = ["label", "n", "state_MB", "ops", "time_s", "peak_MB"]
        rows = []
        for r in self.records:
            s = r.summary()
            rows.append([
                s["label"][:28],
                str(s["n_qubits"]),
                f"{s['state_MB']:.4f}",
                str(s["total_ops"]),
                f"{s['wall_time_s']:.3f}",
                f"{s['peak_python_mem_MB']:.2f}",
            ])
        widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
        sep = "  "
        out = [sep.join(h.ljust(widths[i]) for i, h in enumerate(headers))]
        out.append(sep.join("-" * w for w in widths))
        for row in rows:
            out.append(sep.join(row[i].ljust(widths[i]) for i in range(len(headers))))
        return "\n".join(out)

    def totals(self) -> dict:
        return {
            "n_records": len(self.records),
            "total_ops": sum(r.total_ops for r in self.records),
            "total_wall_time_s": round(sum(r.wall_time_s for r in self.records), 4),
            "max_qubits": max((r.n_qubits for r in self.records), default=0),
            "max_state_MB": round(
                max((r.state_bytes for r in self.records), default=0) / 1024**2, 4
            ),
            "max_peak_MB": round(
                max((r.peak_python_mem_bytes for r in self.records), default=0) / 1024**2, 2
            ),
        }

    def to_json(self, path: str | Path) -> None:
        payload = {
            "totals": self.totals(),
            "records": [
                {**asdict(r), "summary": r.summary()} for r in self.records
            ],
        }
        Path(path).write_text(json.dumps(payload, indent=2))
