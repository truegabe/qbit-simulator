"""Native object-level interop with Qiskit and Cirq.

Round-trip with Qiskit's `QuantumCircuit` and Cirq's `Circuit` without
going through the OpenQASM string layer. Provides:

    - from_qiskit(qiskit_qc) -> QuantumCircuit
    - to_qiskit(qc) -> qiskit.QuantumCircuit
    - from_cirq(cirq_circuit) -> QuantumCircuit
    - to_cirq(qc) -> cirq.Circuit

If qiskit / cirq aren't installed, the functions raise ImportError with
a clear "pip install qiskit" / "pip install cirq" message.

Supported gates (round-trip cleanly):
    H, X, Y, Z, S, T, Sdg, Tdg,
    Rx(theta), Ry(theta), Rz(theta), P(theta) / U1(theta),
    CNOT, CZ, SWAP, CP(theta)

Gates not in this set are passed through best-effort or raised as
NotImplementedError.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .circuit import QuantumCircuit


# ----------------------------------------------------------------------------
# Qiskit interop
# ----------------------------------------------------------------------------

def _require_qiskit() -> Any:
    try:
        import qiskit
    except ImportError:
        raise ImportError(
            "qiskit is not installed. Install with: pip install qiskit"
        )
    return qiskit


def to_qiskit(qc: QuantumCircuit) -> Any:
    """Convert our QuantumCircuit to a qiskit.QuantumCircuit by replaying
    the gate history.

    Returns a `qiskit.QuantumCircuit` object. Requires qiskit installed.
    """
    qiskit = _require_qiskit()
    out = qiskit.QuantumCircuit(qc.n, qc.n)   # n classical bits, for symmetry
    import re
    for entry in qc.history:
        m = re.match(r"([A-Za-z_]+)\(([^)]*)\)", entry)
        if not m:
            continue
        name = m.group(1).upper()
        args = [a.strip() for a in m.group(2).split(",") if a.strip()]
        if name == "H":      out.h(int(args[0]))
        elif name == "X":    out.x(int(args[0]))
        elif name == "Y":    out.y(int(args[0]))
        elif name == "Z":    out.z(int(args[0]))
        elif name == "S":    out.s(int(args[0]))
        elif name == "T":    out.t(int(args[0]))
        elif name == "RX":   out.rx(float(args[0]), int(args[1]))
        elif name == "RY":   out.ry(float(args[0]), int(args[1]))
        elif name == "RZ":   out.rz(float(args[0]), int(args[1]))
        elif name == "P":    out.p(float(args[0]), int(args[1]))
        elif name == "CNOT": out.cx(int(args[0]), int(args[1]))
        elif name == "CZ":   out.cz(int(args[0]), int(args[1]))
        elif name == "SWAP": out.swap(int(args[0]), int(args[1]))
        elif name == "CP":   out.cp(float(args[0]), int(args[1]), int(args[2]))
        else:
            raise NotImplementedError(f"to_qiskit: gate {name} not supported")
    return out


def from_qiskit(qiskit_qc: Any) -> QuantumCircuit:
    """Convert a qiskit.QuantumCircuit to our QuantumCircuit by walking its
    gate list and dispatching to native API methods."""
    qiskit = _require_qiskit()
    n = qiskit_qc.num_qubits
    out = QuantumCircuit(n)
    # Build a map from qiskit qubit objects to our integer indices.
    qubit_map = {q: i for i, q in enumerate(qiskit_qc.qubits)}
    for instruction in qiskit_qc.data:
        op = instruction.operation
        qubits = [qubit_map[q] for q in instruction.qubits]
        name = op.name.lower()
        params = list(op.params) if op.params else []
        if   name == "h":    out.h(qubits[0])
        elif name == "x":    out.x(qubits[0])
        elif name == "y":    out.y(qubits[0])
        elif name == "z":    out.z(qubits[0])
        elif name == "s":    out.s(qubits[0])
        elif name == "t":    out.t(qubits[0])
        elif name == "sdg":  out.s(qubits[0]); out.s(qubits[0]); out.s(qubits[0])
        elif name == "tdg":
            for _ in range(7): out.t(qubits[0])
        elif name == "rx":   out.rx(float(params[0]), qubits[0])
        elif name == "ry":   out.ry(float(params[0]), qubits[0])
        elif name == "rz":   out.rz(float(params[0]), qubits[0])
        elif name in ("p", "u1", "phase"):
            out.p(float(params[0]), qubits[0])
        elif name == "cx":   out.cnot(qubits[0], qubits[1])
        elif name == "cz":   out.cz(qubits[0], qubits[1])
        elif name == "swap": out.swap(qubits[0], qubits[1])
        elif name in ("cp", "cu1", "cphase"):
            out.cp(float(params[0]), qubits[0], qubits[1])
        elif name == "barrier":
            continue
        elif name == "measure":
            continue   # measurement is post-hoc on our deterministic sim
        else:
            raise NotImplementedError(
                f"from_qiskit: gate {name!r} not in the supported set"
            )
    return out


# ----------------------------------------------------------------------------
# Cirq interop
# ----------------------------------------------------------------------------

def _require_cirq() -> Any:
    try:
        import cirq
    except ImportError:
        raise ImportError(
            "cirq is not installed. Install with: pip install cirq"
        )
    return cirq


def to_cirq(qc: QuantumCircuit) -> Any:
    """Convert our QuantumCircuit to a cirq.Circuit by replaying history."""
    cirq = _require_cirq()
    qubits = [cirq.LineQubit(i) for i in range(qc.n)]
    ops = []
    import re
    for entry in qc.history:
        m = re.match(r"([A-Za-z_]+)\(([^)]*)\)", entry)
        if not m:
            continue
        name = m.group(1).upper()
        args = [a.strip() for a in m.group(2).split(",") if a.strip()]
        if   name == "H":    ops.append(cirq.H(qubits[int(args[0])]))
        elif name == "X":    ops.append(cirq.X(qubits[int(args[0])]))
        elif name == "Y":    ops.append(cirq.Y(qubits[int(args[0])]))
        elif name == "Z":    ops.append(cirq.Z(qubits[int(args[0])]))
        elif name == "S":    ops.append(cirq.S(qubits[int(args[0])]))
        elif name == "T":    ops.append(cirq.T(qubits[int(args[0])]))
        elif name == "RX":
            ops.append(cirq.rx(float(args[0]))(qubits[int(args[1])]))
        elif name == "RY":
            ops.append(cirq.ry(float(args[0]))(qubits[int(args[1])]))
        elif name == "RZ":
            ops.append(cirq.rz(float(args[0]))(qubits[int(args[1])]))
        elif name == "P":
            ops.append(cirq.ZPowGate(exponent=float(args[0]) / np.pi)
                       (qubits[int(args[1])]))
        elif name == "CNOT":
            ops.append(cirq.CNOT(qubits[int(args[0])], qubits[int(args[1])]))
        elif name == "CZ":
            ops.append(cirq.CZ(qubits[int(args[0])], qubits[int(args[1])]))
        elif name == "SWAP":
            ops.append(cirq.SWAP(qubits[int(args[0])], qubits[int(args[1])]))
        elif name == "CP":
            theta = float(args[0])
            ops.append(cirq.CZPowGate(exponent=theta / np.pi)
                       (qubits[int(args[1])], qubits[int(args[2])]))
        else:
            raise NotImplementedError(f"to_cirq: gate {name} not supported")
    return cirq.Circuit(ops)


def from_cirq(cirq_circuit: Any) -> QuantumCircuit:
    """Convert a cirq.Circuit to our QuantumCircuit."""
    cirq = _require_cirq()
    # Determine the qubit count.
    all_qubits = sorted(cirq_circuit.all_qubits(), key=lambda q: q.x
                         if hasattr(q, "x") else 0)
    qubit_map = {q: i for i, q in enumerate(all_qubits)}
    out = QuantumCircuit(len(all_qubits))
    for moment in cirq_circuit:
        for op in moment.operations:
            qubits = [qubit_map[q] for q in op.qubits]
            gate = op.gate
            if isinstance(gate, cirq.HPowGate) and gate.exponent == 1:
                out.h(qubits[0])
            elif isinstance(gate, cirq.XPowGate) and gate.exponent == 1:
                out.x(qubits[0])
            elif isinstance(gate, cirq.YPowGate) and gate.exponent == 1:
                out.y(qubits[0])
            elif isinstance(gate, cirq.ZPowGate) and gate.exponent == 1:
                out.z(qubits[0])
            elif isinstance(gate, cirq.ZPowGate) and gate.exponent == 0.5:
                out.s(qubits[0])
            elif isinstance(gate, cirq.ZPowGate) and gate.exponent == 0.25:
                out.t(qubits[0])
            elif isinstance(gate, cirq.Rx):
                out.rx(float(gate._rads), qubits[0])
            elif isinstance(gate, cirq.Ry):
                out.ry(float(gate._rads), qubits[0])
            elif isinstance(gate, cirq.Rz):
                out.rz(float(gate._rads), qubits[0])
            elif isinstance(gate, cirq.CNotPowGate) and gate.exponent == 1:
                out.cnot(qubits[0], qubits[1])
            elif isinstance(gate, cirq.CZPowGate) and gate.exponent == 1:
                out.cz(qubits[0], qubits[1])
            elif isinstance(gate, cirq.SwapPowGate) and gate.exponent == 1:
                out.swap(qubits[0], qubits[1])
            elif isinstance(gate, cirq.CZPowGate):
                # General controlled-phase.
                out.cp(float(gate.exponent * np.pi), qubits[0], qubits[1])
            elif isinstance(gate, cirq.MeasurementGate):
                continue
            else:
                raise NotImplementedError(
                    f"from_cirq: gate {gate!r} not in the supported set"
                )
    return out
