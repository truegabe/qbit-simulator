"""ASCII / Unicode circuit-diagram renderer.

Render a `QuantumCircuit` as a readable diagram by parsing its
`history` (list of gate strings) and laying out gates on a per-qubit
timeline:

    q0: ──H──■──RZ(0.5)──@──M──
              │                │
    q1: ─────X────────────X───────

Supports the gate set in our circuit module: H, X, Y, Z, S, T, RX, RY,
RZ, P, CNOT, CZ, SWAP, CP, plus measurement and arbitrary single-qubit
gates. Multi-qubit gates draw vertical connectors between the involved
wires.

Two backends:
  * `text_diagram(qc, width)` — pure ASCII output for terminals.
  * `unicode_diagram(qc, width)` — uses box-drawing characters for a
    cleaner display.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


# Pattern: gate_name(arg1,arg2,...).
_GATE_RE = re.compile(r"([A-Za-z_]+)\(([^)]*)\)")


@dataclass
class _Op:
    name:    str
    targets: list[int]
    label:   str       # human-readable label like "H", "RZ(0.5)"


# ----------------------------------------------------------------------------
# History parser
# ----------------------------------------------------------------------------

def _parse_history(history: list[str]) -> list[_Op]:
    """Parse the gate-history strings into structured operations."""
    ops: list[_Op] = []
    for entry in history:
        m = _GATE_RE.match(entry)
        if not m:
            continue
        name = m.group(1).upper()
        args_str = m.group(2)
        args = [a.strip() for a in args_str.split(",") if a.strip()]
        # Single-qubit gates: G(q).
        if name in ("H", "X", "Y", "Z", "S", "T", "TDG", "SDG"):
            ops.append(_Op(name, [int(args[0])], name))
        # Parameterized single-qubit: RX(theta,q), P(theta,q), etc.
        elif name in ("RX", "RY", "RZ", "P", "PHASE", "U1"):
            theta = float(args[0])
            q = int(args[1])
            label = f"{name}({theta:.3g})"
            ops.append(_Op(name, [q], label))
        # Two-qubit non-parameterized.
        elif name in ("CNOT", "CX", "CZ", "SWAP", "CY"):
            ops.append(_Op(name, [int(args[0]), int(args[1])], name))
        # Controlled-rotations / parameterized 2q.
        elif name in ("CP", "CRZ", "CRX", "CRY"):
            theta = float(args[0])
            ops.append(_Op(name, [int(args[1]), int(args[2])],
                            f"{name}({theta:.3g})"))
        # Measurement.
        elif name in ("M", "MEASURE"):
            ops.append(_Op("M", [int(args[0])], "M"))
        else:
            # Unknown gate: best-effort label.
            try:
                targets = [int(a) for a in args]
            except ValueError:
                targets = []
            ops.append(_Op(name, targets, name))
    return ops


# ----------------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------------

def _column_assignments(ops: list[_Op], n_qubits: int) -> list[int]:
    """Greedy column placement: each op is placed in the earliest column
    where none of its qubits are already busy.
    """
    next_free = [0] * n_qubits   # earliest free column per qubit
    cols = []
    for op in ops:
        q_min = min(op.targets)
        q_max = max(op.targets)
        # Multi-qubit gates block the vertical wires q_min..q_max.
        col = max(next_free[q_min:q_max + 1])
        cols.append(col)
        for q in range(q_min, q_max + 1):
            next_free[q] = col + 1
    return cols


# ----------------------------------------------------------------------------
# Text rendering
# ----------------------------------------------------------------------------

def text_diagram(qc, width_per_col: int = 8) -> str:
    """Render a QuantumCircuit as ASCII art.

    Args:
        qc:            our QuantumCircuit (must expose `n` and `history`).
        width_per_col: column width in characters.

    Returns:
        multi-line string.
    """
    n = qc.n
    ops = _parse_history(qc.history)
    cols = _column_assignments(ops, n)
    if cols:
        n_cols = max(cols) + 1
    else:
        n_cols = 0

    total_width = max(1, n_cols) * width_per_col
    # Each row is (qubit) padded to total_width. We use 2 rows per
    # qubit (gate label + vertical connector) plus inter-qubit blanks.
    n_text_rows = 2 * n - 1
    grid = [[" "] * total_width for _ in range(n_text_rows)]
    # Wire characters on qubit rows.
    for q in range(n):
        row = 2 * q
        for k in range(total_width):
            grid[row][k] = "-"

    # Draw each op.
    for op, col in zip(ops, cols):
        col_start = col * width_per_col + 1
        label = op.label
        label = label[:width_per_col - 2]
        if len(op.targets) == 1:
            q = op.targets[0]
            row = 2 * q
            # Center the label in this column.
            mid = col_start + (width_per_col - 2 - len(label)) // 2
            for k, ch in enumerate(label):
                grid[row][mid + k] = ch
        else:
            # Multi-qubit: draw label on each qubit row + vertical lines
            # between them.
            q_lo = min(op.targets)
            q_hi = max(op.targets)
            mid = col_start + (width_per_col - 2 - len(label)) // 2
            for q in op.targets:
                row = 2 * q
                for k, ch in enumerate(label):
                    grid[row][mid + k] = ch
            # Vertical connector at column `mid_pipe` between q_lo and q_hi.
            mid_pipe = col_start + (width_per_col - 2) // 2
            for inter_row in range(2 * q_lo + 1, 2 * q_hi):
                grid[inter_row][mid_pipe] = "|"

    # Assemble.
    lines = []
    for q in range(n):
        row = 2 * q
        prefix = f"q{q}: "
        lines.append(prefix + "".join(grid[row]))
        if q < n - 1:
            inter_row = row + 1
            blank = " " * len(prefix)
            lines.append(blank + "".join(grid[inter_row]))
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Unicode rendering (cleaner)
# ----------------------------------------------------------------------------

_BOX_H = "─"
_BOX_V = "│"


def unicode_diagram(qc, width_per_col: int = 8) -> str:
    """Same as `text_diagram` but using box-drawing characters."""
    text = text_diagram(qc, width_per_col=width_per_col)
    # Replace ASCII wires with unicode box characters.
    return text.replace("-", _BOX_H).replace("|", _BOX_V)


# ----------------------------------------------------------------------------
# Summary statistics
# ----------------------------------------------------------------------------

def circuit_stats(qc) -> dict:
    """Count gates by type and overall depth (after greedy packing)."""
    ops = _parse_history(qc.history)
    by_type: dict[str, int] = {}
    for op in ops:
        by_type[op.name] = by_type.get(op.name, 0) + 1
    cols = _column_assignments(ops, qc.n)
    depth = max(cols) + 1 if cols else 0
    return {
        "total_gates":  len(ops),
        "by_type":      by_type,
        "depth":        depth,
        "n_qubits":     qc.n,
    }
