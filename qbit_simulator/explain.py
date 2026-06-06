"""Narrate a quantum circuit gate-by-gate.

Mirrors the brain's `explain_chain` idea but for quantum circuits. Given a
`QuantumCircuit` object, produces multi-line prose describing:

  * what each gate does in plain English
  * the running state's probability distribution at each step (top entries)
  * heuristic identification of likely algorithms (Bell pair? Grover step?
    QFT layer? Teleportation correction?)

Useful for teaching and debugging.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .circuit import QuantumCircuit


# Plain-English summaries for each gate kind.
GATE_DESCRIPTIONS = {
    "H":    "Hadamard — puts qubit {q} into an equal superposition",
    "X":    "Pauli-X (NOT) — flips |0> <-> |1> on qubit {q}",
    "Y":    "Pauli-Y — rotates qubit {q} about the y-axis by pi",
    "Z":    "Pauli-Z — phase flip on |1> for qubit {q}",
    "S":    "S gate — quarter-turn phase on qubit {q}",
    "T":    "T gate — eighth-turn phase on qubit {q}",
    "Rx":   "Rx({theta}) — rotation of qubit {q} about the x-axis",
    "Ry":   "Ry({theta}) — rotation of qubit {q} about the y-axis",
    "Rz":   "Rz({theta}) — rotation of qubit {q} about the z-axis",
    "P":    "Phase({phi}) — adds phase exp(i*{phi}) to |1> on qubit {q}",
    "CNOT": "CNOT — flip qubit {t} if qubit {c} is |1>  (entangling gate)",
    "CZ":   "CZ — phase-flip |11> on qubits {c}, {t}  (entangling)",
    "CP":   "Controlled phase({phi}) — phase on |11> with control={c}, target={t}",
    "SWAP": "SWAP — exchange qubits {a} and {b}",
}


def _parse_op(op_str: str) -> tuple[str, list[str]]:
    """Parse 'H(0)' / 'CNOT(0,1)' / 'Rx(0.5,2)' into (name, [args])."""
    if "(" not in op_str:
        return op_str, []
    name = op_str.split("(", 1)[0]
    inside = op_str[op_str.index("(") + 1: op_str.rindex(")")]
    args = [a.strip() for a in inside.split(",") if a.strip()]
    return name, args


def describe_gate(op_str: str) -> str:
    """One-line description of a single history entry."""
    name, args = _parse_op(op_str)
    template = GATE_DESCRIPTIONS.get(name)
    if template is None:
        return op_str  # unknown — return raw
    # Substitute placeholders based on gate kind.
    if name in ("H", "X", "Y", "Z", "S", "T"):
        return template.format(q=args[0]) if args else template
    if name in ("Rx", "Ry", "Rz"):
        return template.format(theta=args[0], q=args[1])
    if name == "P":
        return template.format(phi=args[0], q=args[1])
    if name == "CNOT":
        return template.format(c=args[0], t=args[1])
    if name == "CZ":
        return template.format(c=args[0], t=args[1])
    if name == "CP":
        return template.format(phi=args[0], c=args[1], t=args[2])
    if name == "SWAP":
        return template.format(a=args[0], b=args[1])
    return op_str


def _top_basis_states(state: np.ndarray, n_qubits: int,
                      top_k: int = 4, threshold: float = 0.005) -> str:
    """Return a string like '|00>:0.50, |11>:0.50' for the top-K most likely outcomes."""
    probs = np.abs(state) ** 2
    indices = np.argsort(probs)[::-1]
    items = []
    for idx in indices[:top_k]:
        p = float(probs[idx])
        if p < threshold:
            break
        bits = format(int(idx), f"0{n_qubits}b")
        items.append(f"|{bits}>:{p:.3f}")
    return ", ".join(items) if items else "(spread)"


def _identify_algorithm(history: list[str], n: int) -> str | None:
    """Heuristic match — returns a one-line guess at what this circuit is."""
    if not history:
        return None
    n_cp = sum(1 for op in history if op.startswith("CP("))
    n_h = sum(1 for op in history if op.startswith("H("))
    cnot_ops = [op for op in history if op.startswith("CNOT(")]
    # GHZ cascade comes BEFORE the Bell check, because Bell is a special case
    # of a 2-qubit GHZ. Cascade: H + chained CNOTs across all qubits.
    if n_h == 1 and len(cnot_ops) == n - 1 and n >= 3:
        return "looks like a GHZ entanglement cascade"
    # Bell pair: exactly H + CNOT, n == 2.
    if (n == 2 and len(history) == 2
            and history[0].startswith("H(") and history[1].startswith("CNOT(")):
        return "looks like a Bell pair construction"
    # QFT: heavy CP gates after Hadamards.
    if n_cp >= 3 and n_h >= 2 and n_cp >= n_h:
        return "looks like a Quantum Fourier Transform layer"
    # Grover-like state prep: uniform Hadamards across all qubits.
    if n_h == n and n >= 3 and not cnot_ops and not n_cp:
        return "uniform Hadamards (possibly Grover state preparation)"
    return None


def explain_circuit(
    qc: QuantumCircuit,
    show_intermediate_states: bool = True,
    max_state_steps: int = 10,
) -> str:
    """Produce a multi-paragraph English narrative of `qc`.

    Args:
        qc: the circuit to explain.
        show_intermediate_states: if True, replay the circuit step by step
            and print top basis-state probabilities after each gate (for
            small circuits only — capped at `max_state_steps`).
    """
    lines: list[str] = []
    n = qc.n
    history = list(qc.history)

    lines.append(f"Circuit on {n} qubit{'s' if n != 1 else ''}, "
                 f"{len(history)} operation{'s' if len(history) != 1 else ''}.")

    guess = _identify_algorithm(history, n)
    if guess:
        lines.append(f"Pattern: {guess}.")

    if not history:
        lines.append("(empty circuit)")
        return "\n".join(lines)

    # Gate-by-gate description.
    lines.append("")
    lines.append("Gate-by-gate:")
    for i, op in enumerate(history):
        lines.append(f"  {i+1:>3}. {op:<22}  — {describe_gate(op)}")

    # Final state summary.
    lines.append("")
    lines.append(f"Final state (top outcomes by probability):")
    lines.append(f"  {_top_basis_states(qc.state, n)}")

    # Step-through (small circuits only).
    if show_intermediate_states and len(history) <= max_state_steps and n <= 6:
        lines.append("")
        lines.append("Step-by-step state evolution:")
        replay = QuantumCircuit(n)
        # Replay each op using the captured matrices in qc._ops.
        for i, (kind, matrix, targets) in enumerate(qc._ops):
            try:
                if kind == "1q":
                    replay._apply_1q(matrix, targets[0])
                elif kind == "2q":
                    replay._apply_2q(matrix, targets[0], targets[1])
                elif kind == "kq":
                    replay.apply_unitary(matrix, targets, check_unitary=False)
            except Exception:
                continue
            lines.append(f"  after op {i+1} ({history[i] if i < len(history) else '?'}): "
                         f"{_top_basis_states(replay.state, n)}")

    return "\n".join(lines)
