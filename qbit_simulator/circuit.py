"""Multi-qubit circuit using a full state vector of length 2^N.

Gates act on the state by reshaping it to a rank-N tensor of shape (2,)*N and
contracting on the target axes. This handles non-adjacent qubits cleanly
without building permuted 2^N x 2^N matrices.
"""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from .gates import H, X, Y, Z, S, T, CNOT, SWAP, Rx, Ry, Rz, P, CP, is_unitary
from .measure import sample


class QuantumCircuit:
    def __init__(self, n_qubits: int):
        if n_qubits < 1:
            raise ValueError("Need at least 1 qubit.")
        self.n = n_qubits
        self.state = np.zeros(2**n_qubits, dtype=np.complex128)
        self.state[0] = 1.0  # |0...0>
        self.history: list[str] = []
        # Replayable operation list: each entry is (kind, matrix, targets)
        # kind is "1q", "2q", or "kq" (general apply_unitary).
        self._ops: list[tuple[str, np.ndarray, list[int]]] = []
        self._logger = None  # optional telemetry hook

    # ---- core gate application ----

    def _log(self, name: str, targets: list[int], t0: float) -> None:
        if self._logger is not None:
            self._logger.log_op(name, targets, time.perf_counter() - t0)

    def _apply_1q(self, gate: np.ndarray, target: int, _name: str = "1q") -> None:
        if not (0 <= target < self.n):
            raise IndexError(f"target {target} out of range for {self.n} qubits")
        t0 = time.perf_counter()
        tensor = self.state.reshape((2,) * self.n)
        tensor = np.moveaxis(tensor, target, 0)
        shape = tensor.shape
        tensor = gate @ tensor.reshape(2, -1)
        tensor = tensor.reshape(shape)
        tensor = np.moveaxis(tensor, 0, target)
        self.state = tensor.reshape(2**self.n)
        self._ops.append(("1q", gate.copy(), [target]))
        self._log(_name, [target], t0)

    def _apply_2q(self, gate4: np.ndarray, control: int, target: int, _name: str = "2q") -> None:
        if control == target:
            raise ValueError("control and target must differ")
        for q in (control, target):
            if not (0 <= q < self.n):
                raise IndexError(f"qubit {q} out of range")
        t0 = time.perf_counter()
        tensor = self.state.reshape((2,) * self.n)
        tensor = np.moveaxis(tensor, [control, target], [0, 1])
        shape = tensor.shape
        flat = tensor.reshape(4, -1)
        flat = gate4 @ flat
        tensor = flat.reshape(shape)
        tensor = np.moveaxis(tensor, [0, 1], [control, target])
        self.state = tensor.reshape(2**self.n)
        self._ops.append(("2q", gate4.copy(), [control, target]))
        self._log(_name, [control, target], t0)

    # ---- diagonal-gate fast paths ----
    #
    # Many common gates (Z, S, T, P(φ), CZ, CP(φ)) are diagonal in the
    # computational basis — they don't mix amplitudes, only multiply phases.
    # The generic _apply_1q / _apply_2q path builds the matrix and does a
    # reshape + matmul on the entire 2^n state. For a diagonal gate this is
    # wasteful: we just need to multiply a slice of the state by a phase.
    #
    # These helpers exploit that. Speedup at large n is substantial —
    # an inverse QFT on 14 qubits goes from O(t²) reshape+matmul ops to
    # O(t²) in-place slice multiplies, with the latter ~5-10× faster.
    #
    # We still record the equivalent matrix in `_ops` so circuit replay
    # works unchanged.

    def _apply_diag_phase_1q(self, q: int, phase_on_1: complex) -> None:
        """Multiply amplitudes where qubit `q` is |1⟩ by `phase_on_1`."""
        view = self.state.reshape((2,) * self.n)
        slc = [slice(None)] * self.n
        slc[q] = 1
        view[tuple(slc)] *= phase_on_1
        # state already updated via the view (reshape is a view, not a copy).

    def _apply_diag_phase_2q_cc(
        self, qa: int, qb: int, phase_on_11: complex,
    ) -> None:
        """Multiply amplitudes where qubits qa AND qb are both |1⟩ by phase."""
        view = self.state.reshape((2,) * self.n)
        slc = [slice(None)] * self.n
        slc[qa] = 1
        slc[qb] = 1
        view[tuple(slc)] *= phase_on_11

    def apply_unitary(
        self,
        U: np.ndarray,
        targets: Sequence[int],
        check_unitary: bool = True,
    ) -> "QuantumCircuit":
        """Apply an arbitrary 2^k x 2^k unitary to the listed target qubits (in order).

        targets[0] is the most-significant qubit of U's basis indexing.
        """
        k = len(targets)
        if U.shape != (2**k, 2**k):
            raise ValueError(f"U has shape {U.shape}, expected ({2**k},{2**k})")
        if len(set(targets)) != k:
            raise ValueError("targets must be distinct")
        for q in targets:
            if not (0 <= q < self.n):
                raise IndexError(f"qubit {q} out of range")
        if check_unitary and not is_unitary(U):
            raise ValueError("U is not unitary (U^dagger @ U != I)")

        tensor = self.state.reshape((2,) * self.n)
        tensor = np.moveaxis(tensor, list(targets), list(range(k)))
        shape = tensor.shape
        flat = tensor.reshape(2**k, -1)
        flat = U @ flat
        tensor = flat.reshape(shape)
        tensor = np.moveaxis(tensor, list(range(k)), list(targets))
        self.state = tensor.reshape(2**self.n)
        self.history.append(f"U_{k}q({list(targets)})")
        self._ops.append(("kq", U.copy(), list(targets)))
        return self

    # ---- public gate API ----

    def h(self, q): self._apply_1q(H, q, "H"); self.history.append(f"H({q})"); return self
    def x(self, q): self._apply_1q(X, q, "X"); self.history.append(f"X({q})"); return self
    def y(self, q): self._apply_1q(Y, q, "Y"); self.history.append(f"Y({q})"); return self

    def z(self, q):
        # Z is diagonal — only flip the sign of |1⟩ amplitudes.
        if not (0 <= q < self.n):
            raise IndexError(f"qubit {q} out of range")
        t0 = time.perf_counter()
        self._apply_diag_phase_1q(q, -1.0)
        self._ops.append(("1q", Z.copy(), [q]))
        self._log("Z", [q], t0)
        self.history.append(f"Z({q})"); return self

    def s(self, q):
        # S = diag(1, i)
        if not (0 <= q < self.n):
            raise IndexError(f"qubit {q} out of range")
        t0 = time.perf_counter()
        self._apply_diag_phase_1q(q, 1j)
        self._ops.append(("1q", S.copy(), [q]))
        self._log("S", [q], t0)
        self.history.append(f"S({q})"); return self

    def t(self, q):
        # T = diag(1, e^{iπ/4})
        if not (0 <= q < self.n):
            raise IndexError(f"qubit {q} out of range")
        t0 = time.perf_counter()
        self._apply_diag_phase_1q(q, np.exp(1j * np.pi / 4))
        self._ops.append(("1q", T.copy(), [q]))
        self._log("T", [q], t0)
        self.history.append(f"T({q})"); return self

    def rx(self, theta: float, q: int):
        self._apply_1q(Rx(theta), q, "Rx"); self.history.append(f"Rx({theta:.4g},{q})"); return self

    def ry(self, theta: float, q: int):
        self._apply_1q(Ry(theta), q, "Ry"); self.history.append(f"Ry({theta:.4g},{q})"); return self

    def rz(self, theta: float, q: int):
        self._apply_1q(Rz(theta), q, "Rz"); self.history.append(f"Rz({theta:.4g},{q})"); return self

    def p(self, phi: float, q: int):
        # P(φ) = diag(1, e^{iφ}) — diagonal, fast path.
        if not (0 <= q < self.n):
            raise IndexError(f"qubit {q} out of range")
        t0 = time.perf_counter()
        self._apply_diag_phase_1q(q, np.exp(1j * phi))
        self._ops.append(("1q", P(phi).copy(), [q]))
        self._log("P", [q], t0)
        self.history.append(f"P({phi:.4g},{q})"); return self

    def cnot(self, control: int, target: int):
        self._apply_2q(CNOT, control, target, "CNOT"); self.history.append(f"CNOT({control},{target})"); return self

    def swap(self, a: int, b: int):
        self._apply_2q(SWAP, a, b, "SWAP"); self.history.append(f"SWAP({a},{b})"); return self

    def cp(self, phi: float, control: int, target: int):
        # CP(φ) is diagonal: only |11⟩ gets *= e^{iφ}.
        if control == target:
            raise ValueError("control and target must differ")
        for q in (control, target):
            if not (0 <= q < self.n):
                raise IndexError(f"qubit {q} out of range")
        t0 = time.perf_counter()
        self._apply_diag_phase_2q_cc(control, target, np.exp(1j * phi))
        self._ops.append(("2q", CP(phi).copy(), [control, target]))
        self._log("CP", [control, target], t0)
        self.history.append(f"CP({phi:.4g},{control},{target})"); return self

    def cz(self, control: int, target: int):
        # CZ is diagonal: only |11⟩ gets *= -1.
        if control == target:
            raise ValueError("control and target must differ")
        for q in (control, target):
            if not (0 <= q < self.n):
                raise IndexError(f"qubit {q} out of range")
        from .gates import controlled
        t0 = time.perf_counter()
        self._apply_diag_phase_2q_cc(control, target, -1.0)
        self._ops.append(("2q", controlled(Z).copy(), [control, target]))
        self._log("CZ", [control, target], t0)
        self.history.append(f"CZ({control},{target})"); return self

    # ---- composition & I/O ----

    def copy(self) -> "QuantumCircuit":
        new = QuantumCircuit(self.n)
        new.state = self.state.copy()
        new.history = list(self.history)
        new._ops = [(k, m.copy(), list(t)) for k, m, t in self._ops]
        return new

    def replay_ops(self, ops: list) -> "QuantumCircuit":
        """Apply a list of (kind, matrix, targets) tuples to the current state."""
        for kind, matrix, targets in ops:
            if kind == "1q":
                self._apply_1q(matrix, targets[0])
            elif kind == "2q":
                self._apply_2q(matrix, targets[0], targets[1])
            elif kind == "kq":
                self.apply_unitary(matrix, targets, check_unitary=False)
            else:
                raise ValueError(f"unknown op kind {kind}")
        return self

    def __add__(self, other: "QuantumCircuit") -> "QuantumCircuit":
        """Concatenate two same-N circuits: apply other's ops on top of self."""
        if other.n != self.n:
            raise ValueError(f"Qubit count mismatch: {self.n} vs {other.n}")
        out = self.copy()
        out.replay_ops(other._ops)
        out.history.extend(other.history)
        return out

    def inverse(self) -> "QuantumCircuit":
        """Build a circuit whose unitary is the conjugate-transpose of self's.

        Acts on |0..0⟩ by default. To use as the inverse of an existing
        circuit, concatenate: `qc_inv = qc.inverse(); state_back = (qc + qc_inv).state`.
        """
        out = QuantumCircuit(self.n)
        # Reverse order and conjugate-transpose each gate.
        for kind, matrix, targets in reversed(self._ops):
            out.replay_ops([(kind, matrix.conj().T, list(targets))])
            out.history.append(f"{kind}†({targets})")
        return out

    def controlled(self, control: int) -> "QuantumCircuit":
        """Return a new circuit on (n+1) qubits implementing this one controlled
        by the new qubit at index `control`.

        Qubit indices in the new circuit: the new control qubit goes at
        `control`; existing qubits shift to make room (qubits >= control
        in the original become qubit+1 in the new circuit).
        """
        new_n = self.n + 1
        if not (0 <= control <= self.n):
            raise IndexError("control out of range")
        out = QuantumCircuit(new_n)

        def remap(q: int) -> int:
            return q if q < control else q + 1

        for kind, matrix, targets in self._ops:
            new_targets = [remap(q) for q in targets]
            k = len(new_targets)
            dim = 2**k
            # Controlled matrix: I on |0⟩_c block, original matrix on |1⟩_c block.
            CM = np.eye(2 * dim, dtype=np.complex128)
            CM[dim:, dim:] = matrix
            out.apply_unitary(CM, [control] + new_targets, check_unitary=False)
        return out

    def save(self, path) -> None:
        """Save the state vector and history to a .npz file."""
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            state=self.state,
            n=self.n,
            history=np.array(self.history, dtype=object),
        )

    @classmethod
    def load(cls, path) -> "QuantumCircuit":
        """Load a saved circuit from disk."""
        data = np.load(path, allow_pickle=True)
        n = int(data["n"])
        qc = cls(n)
        qc.state = data["state"]
        qc.history = list(data["history"])
        return qc

    def reverse_qubits(self) -> "QuantumCircuit":
        """Reverse the qubit ordering in one pass — equivalent to N/2 SWAPs."""
        tensor = self.state.reshape((2,) * self.n)
        tensor = np.transpose(tensor, axes=tuple(range(self.n - 1, -1, -1)))
        self.state = np.ascontiguousarray(tensor).reshape(2**self.n)
        self.history.append("ReverseQubits")
        return self

    # ---- inspection / measurement ----

    def probabilities(self) -> np.ndarray:
        return np.abs(self.state) ** 2

    def measure_all(self, shots: int = 1, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample basis-state outcomes WITHOUT collapsing the circuit state.

        Use this for collecting measurement statistics over many shots.
        For a destructive mid-circuit measurement that collapses the state,
        use `measure_qubit()` or `measure_collapse()`.
        """
        return sample(self.probabilities(), shots=shots, rng=rng)

    def measure_collapse(self, rng: np.random.Generator | None = None) -> int:
        """Measure the full register once and collapse the state. Returns the outcome index."""
        probs = self.probabilities()
        outcome = int(sample(probs, shots=1, rng=rng)[0])
        new_state = np.zeros_like(self.state)
        new_state[outcome] = 1.0
        self.state = new_state
        self.history.append(f"MeasureAll->{outcome}")
        return outcome

    def measure_qubit(self, q: int, rng: np.random.Generator | None = None) -> int:
        """Measure a single qubit, collapse and renormalize the state. Returns 0 or 1."""
        if not (0 <= q < self.n):
            raise IndexError(f"qubit {q} out of range")
        tensor = self.state.reshape((2,) * self.n)
        # Probability that qubit q is in |0>: sum over all other axes of |amp|^2 in slice axis_q=0.
        # Move target axis to position 0 for clean slicing.
        moved = np.moveaxis(tensor, q, 0)
        p0 = float(np.sum(np.abs(moved[0]) ** 2))
        p0 = min(max(p0, 0.0), 1.0)
        if rng is None:
            rng = np.random.default_rng()
        outcome = 0 if rng.random() < p0 else 1
        # Zero the non-selected branch, then renormalize.
        moved[1 - outcome] = 0.0
        norm = np.linalg.norm(moved)
        if norm == 0.0:
            raise RuntimeError("Post-measurement state has zero norm (shouldn't happen).")
        moved /= norm
        self.state = np.moveaxis(moved, 0, q).reshape(2**self.n)
        self.history.append(f"Measure(q{q})->{outcome}")
        return outcome

    def counts(self, shots: int = 1024, rng: np.random.Generator | None = None) -> dict[str, int]:
        outcomes = self.measure_all(shots=shots, rng=rng)
        out: dict[str, int] = {}
        for o in outcomes:
            key = format(int(o), f"0{self.n}b")
            out[key] = out.get(key, 0) + 1
        return out

    def __repr__(self) -> str:
        return f"QuantumCircuit(n={self.n}, ops={len(self.history)})"


# ---- index helpers for in-place gate application ----
# Computed on the fly — caching would dominate memory at large N.

def _indices_where_bit_zero(n: int, bit: int) -> np.ndarray:
    step = 1 << bit
    block = step << 1
    starts = np.arange(0, 1 << n, block)
    offsets = np.arange(step)
    return (starts[:, None] + offsets[None, :]).ravel()


def _indices_where_two_bits_zero(n: int, b1: int, b2: int) -> np.ndarray:
    mask = (1 << b1) | (1 << b2)
    all_idx = np.arange(1 << n)
    return all_idx[(all_idx & mask) == 0]
