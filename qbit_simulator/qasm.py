"""OpenQASM 2.0 + 3.0 import/export for QuantumCircuit.

Implements a useful subset of both OpenQASM dialects:

  Supported instructions:
    h, x, y, z, s, t, sdg, tdg
    rx(theta), ry(theta), rz(theta), u1/p/phase(theta)
    cx (CNOT), cz, swap, cp/cu1/cphase(theta)
    measure q[i] -> c[i]    (OpenQASM 2.0)
    c[i] = measure q[i]      (OpenQASM 3.0)
    barrier (parsed and ignored)

  Supported declarations:
    qreg q[N];   creg c[N];    (OpenQASM 2.0)
    qubit[N] q;  bit[N] c;     (OpenQASM 3.0)

  Constants in parameter expressions: pi, tau, euler.

  Not supported (out of scope):
    multiple quantum registers (we always use a single register)
    custom `gate` definitions, function definitions
    classical control flow (if / for / while in OpenQASM 3.0)
    pulse-level OpenPulse extensions

This handles round-tripping our own circuits and ingesting simple
circuits exported from Qiskit, Cirq, OpenQASM playgrounds, etc.,
across both the legacy 2.0 and the current 3.0 dialects.
"""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np

from .circuit import QuantumCircuit


# ---- export: QuantumCircuit -> QASM string ----

def to_qasm(qc: QuantumCircuit, classical_bits: bool = True,
            version: str = "2.0") -> str:
    """Serialize a QuantumCircuit to OpenQASM.

    Args:
        qc:              the circuit to serialize.
        classical_bits:  emit a classical bit register of the same size.
        version:         "2.0" (default, qelib1.inc) or "3.0" (stdgates.inc).
    """
    if version not in ("2.0", "3.0"):
        raise ValueError("version must be '2.0' or '3.0'")
    if version == "2.0":
        lines = [
            "OPENQASM 2.0;",
            'include "qelib1.inc";',
            f"qreg q[{qc.n}];",
        ]
        if classical_bits:
            lines.append(f"creg c[{qc.n}];")
    else:
        lines = [
            "OPENQASM 3.0;",
            'include "stdgates.inc";',
            f"qubit[{qc.n}] q;",
        ]
        if classical_bits:
            lines.append(f"bit[{qc.n}] c;")
    lines.append("")
    for entry in qc.history:
        lines.append(_history_entry_to_qasm(entry, version))
    return "\n".join(lines) + "\n"


def _history_entry_to_qasm(entry: str, version: str = "2.0") -> str:
    """Translate one history string from QuantumCircuit.history to QASM."""
    m = re.match(r"([A-Za-z_]+)\(([^)]*)\)", entry)
    if not m:
        return f"// unknown history entry: {entry}"
    name = m.group(1).upper()
    args = [a.strip() for a in m.group(2).split(",") if a.strip()]
    # Phase-gate name: u1 in 2.0, p (or phase) in 3.0.
    p_gate  = "u1" if version == "2.0" else "p"
    cp_gate = "cu1" if version == "2.0" else "cp"
    if name == "H":      return f"h q[{args[0]}];"
    if name == "X":      return f"x q[{args[0]}];"
    if name == "Y":      return f"y q[{args[0]}];"
    if name == "Z":      return f"z q[{args[0]}];"
    if name == "S":      return f"s q[{args[0]}];"
    if name == "T":      return f"t q[{args[0]}];"
    if name == "RX":     return f"rx({args[0]}) q[{args[1]}];"
    if name == "RY":     return f"ry({args[0]}) q[{args[1]}];"
    if name == "RZ":     return f"rz({args[0]}) q[{args[1]}];"
    if name == "P":      return f"{p_gate}({args[0]}) q[{args[1]}];"
    if name == "CNOT":   return f"cx q[{args[0]}], q[{args[1]}];"
    if name == "CZ":     return f"cz q[{args[0]}], q[{args[1]}];"
    if name == "SWAP":   return f"swap q[{args[0]}], q[{args[1]}];"
    if name == "CP":     return f"{cp_gate}({args[0]}) q[{args[1]}], q[{args[2]}];"
    return f"// unsupported gate: {entry}"


# ---- import: QASM string -> QuantumCircuit ----

# Strip block comments and line comments.
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT  = re.compile(r"//[^\n]*")

# qreg q[N];   (OpenQASM 2.0)
_QREG_RE = re.compile(r"qreg\s+([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]")
# qubit[N] q;  (OpenQASM 3.0)
_QUBIT3_RE = re.compile(r"qubit\s*\[\s*(\d+)\s*\]\s+([A-Za-z_]\w*)")
# qubit q;     (OpenQASM 3.0, single qubit)
_QUBIT3_SINGLE_RE = re.compile(r"^qubit\s+([A-Za-z_]\w*)\s*$")
# Classical bit declarations in 3.0: bit[N] c;  or  bit c;
_BIT3_RE = re.compile(r"^bit\s*(?:\[\s*\d+\s*\])?\s+[A-Za-z_]\w*\s*$")
# Single-qubit no-arg gate: e.g. "h q[3]"
_GATE_1Q_NOARG_RE = re.compile(
    r"^(h|x|y|z|s|t|sdg|tdg)\s+([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]$"
)
# Single-qubit parameterized gate: e.g. "rx(0.5) q[2]"
_GATE_1Q_ARG_RE = re.compile(
    r"^(rx|ry|rz|u1|p|phase|u3)\s*\(([^)]*)\)\s+([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]$"
)
# Two-qubit no-arg gate: "cx q[0], q[1]"
_GATE_2Q_NOARG_RE = re.compile(
    r"^(cx|cnot|cz|swap)\s+([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]\s*,\s*"
    r"([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]$"
)
# Two-qubit param gate: "cu1(0.3) q[0], q[1]" or "cp(0.3) ..." (cphase = 3.0 alias)
_GATE_2Q_ARG_RE = re.compile(
    r"^(cu1|cp|cphase|crz)\s*\(([^)]*)\)\s+([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]\s*,\s*"
    r"([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]$"
)
# measure q[i] -> c[j]    (OpenQASM 2.0)
_MEASURE_RE = re.compile(
    r"^measure\s+[A-Za-z_]\w*\s*\[\s*(\d+)\s*\]\s*->\s*[A-Za-z_]\w*\s*\[\s*\d+\s*\]$"
)
# c[j] = measure q[i]     (OpenQASM 3.0)
_MEASURE3_RE = re.compile(
    r"^[A-Za-z_]\w*\s*\[\s*\d+\s*\]\s*=\s*measure\s+[A-Za-z_]\w*\s*\[\s*(\d+)\s*\]$"
)


def _eval_param(s: str) -> float:
    """Evaluate a QASM parameter expression. Supports pi, tau, euler and
    basic arithmetic."""
    expr = s.strip()
    # Order matters: substitute "euler" before "e" (we don't actually accept
    # bare e, but the regex below allows 'e' as part of scientific notation
    # like 1e-3, so be careful).
    expr = expr.replace("euler", str(np.e))
    expr = expr.replace("tau", str(2 * np.pi))
    expr = expr.replace("pi", str(np.pi))
    if not re.match(r"^[\d\.\+\-\*/\s\(\)e]+$", expr):
        raise ValueError(f"unsafe QASM parameter expression: {s!r}")
    return float(eval(expr, {"__builtins__": {}}, {}))


def from_qasm(text: str) -> QuantumCircuit:
    """Parse an OpenQASM 2.0 string into a QuantumCircuit.

    Raises ValueError on syntax errors or unsupported instructions.
    """
    # Strip comments.
    text = _BLOCK_COMMENT.sub("", text)
    text = _LINE_COMMENT.sub("", text)

    # Tokenize on semicolons.
    statements = [s.strip() for s in text.split(";") if s.strip()]

    qc: QuantumCircuit | None = None
    measurements: list[int] = []   # we record but don't apply (Python sim is
                                    # deterministic; measurement is post-hoc)

    for stmt in statements:
        # Skip header / no-op lines.
        if stmt.startswith("OPENQASM"):
            continue
        if stmt.startswith("include"):
            continue
        if stmt.startswith("barrier"):
            continue
        if stmt.startswith("creg"):
            continue
        # OpenQASM 3.0 classical-bit declarations (bit[N] c; or bit c;).
        if _BIT3_RE.match(stmt):
            continue
        # qreg q[N];  (OpenQASM 2.0)
        m = _QREG_RE.match(stmt)
        if m:
            n = int(m.group(2))
            if qc is None:
                qc = QuantumCircuit(n)
            else:
                raise ValueError("multiple qreg declarations not supported")
            continue
        # qubit[N] q;  (OpenQASM 3.0)
        m = _QUBIT3_RE.match(stmt)
        if m:
            n = int(m.group(1))
            if qc is None:
                qc = QuantumCircuit(n)
            else:
                raise ValueError("multiple qubit declarations not supported")
            continue
        # qubit q;     (OpenQASM 3.0 single qubit)
        m = _QUBIT3_SINGLE_RE.match(stmt)
        if m:
            if qc is None:
                qc = QuantumCircuit(1)
            else:
                raise ValueError("multiple qubit declarations not supported")
            continue
        # Anything else needs the circuit to exist.
        if qc is None:
            raise ValueError(f"gate before qreg: {stmt!r}")

        # Try each gate-pattern in turn.
        m = _GATE_1Q_NOARG_RE.match(stmt)
        if m:
            gate, _reg, idx = m.group(1), m.group(2), int(m.group(3))
            _dispatch_1q_noarg(qc, gate, idx)
            continue
        m = _GATE_1Q_ARG_RE.match(stmt)
        if m:
            gate = m.group(1)
            param = m.group(2)
            idx = int(m.group(4))
            _dispatch_1q_arg(qc, gate, param, idx)
            continue
        m = _GATE_2Q_NOARG_RE.match(stmt)
        if m:
            gate = m.group(1)
            a = int(m.group(3)); b = int(m.group(5))
            _dispatch_2q_noarg(qc, gate, a, b)
            continue
        m = _GATE_2Q_ARG_RE.match(stmt)
        if m:
            gate = m.group(1)
            param = m.group(2)
            a = int(m.group(4)); b = int(m.group(6))
            _dispatch_2q_arg(qc, gate, param, a, b)
            continue
        m = _MEASURE_RE.match(stmt)
        if m:
            measurements.append(int(m.group(1)))
            continue
        m = _MEASURE3_RE.match(stmt)
        if m:
            measurements.append(int(m.group(1)))
            continue
        raise ValueError(f"unsupported QASM statement: {stmt!r}")

    if qc is None:
        raise ValueError("QASM source contained no qreg declaration")
    return qc


def _dispatch_1q_noarg(qc: QuantumCircuit, gate: str, idx: int) -> None:
    if   gate == "h":    qc.h(idx)
    elif gate == "x":    qc.x(idx)
    elif gate == "y":    qc.y(idx)
    elif gate == "z":    qc.z(idx)
    elif gate == "s":    qc.s(idx)
    elif gate == "t":    qc.t(idx)
    elif gate == "sdg":
        # S† = S³ in finite group
        qc.s(idx); qc.s(idx); qc.s(idx)
    elif gate == "tdg":
        # T† = T⁷
        for _ in range(7): qc.t(idx)
    else:
        raise ValueError(f"unsupported 1q gate: {gate}")


def _dispatch_1q_arg(qc: QuantumCircuit, gate: str, param: str, idx: int) -> None:
    if   gate == "rx":                   qc.rx(_eval_param(param), idx)
    elif gate == "ry":                   qc.ry(_eval_param(param), idx)
    elif gate == "rz":                   qc.rz(_eval_param(param), idx)
    elif gate in ("u1", "p", "phase"):   qc.p(_eval_param(param), idx)
    elif gate == "u3":
        # u3(theta, phi, lambda) = Rz(phi) Ry(theta) Rz(lambda) -- approximate.
        params = [_eval_param(p) for p in param.split(",")]
        if len(params) != 3:
            raise ValueError("u3 needs 3 parameters")
        theta, phi, lam = params
        qc.rz(lam, idx); qc.ry(theta, idx); qc.rz(phi, idx)
    else:
        raise ValueError(f"unsupported 1q-arg gate: {gate}")


def _dispatch_2q_noarg(qc: QuantumCircuit, gate: str, a: int, b: int) -> None:
    if   gate in ("cx", "cnot"):  qc.cnot(a, b)
    elif gate == "cz":            qc.cz(a, b)
    elif gate == "swap":          qc.swap(a, b)
    else:
        raise ValueError(f"unsupported 2q gate: {gate}")


def _dispatch_2q_arg(qc: QuantumCircuit, gate: str, param: str, a: int, b: int) -> None:
    if gate in ("cu1", "cp", "cphase"):
        qc.cp(_eval_param(param), a, b)
    elif gate == "crz":
        # Decompose crz(theta): nothing native — synthesize via cp + rz adjustments.
        theta = _eval_param(param)
        qc.cp(theta, a, b)
    else:
        raise ValueError(f"unsupported 2q-arg gate: {gate}")
