"""Unified quantum simulator REPL.

A single interactive entry point that exposes every algorithm in
`qbit_simulator.algorithms` plus gate-by-gate circuit construction,
save/load, telemetry, and circuit explanation.

Run with:

    python qbit_simulator/repl.py

Commands are grouped by category and `help` shows them all. Unknown
commands get a "did you mean..." suggestion.
"""

from __future__ import annotations

import argparse
import difflib
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from qbit_simulator import QuantumCircuit
from qbit_simulator.algorithms import (
    bell_pair, deutsch, grover, qft, apply_qft,
    phase_estimation, shor, qaoa, qaoa_ansatz, maxcut_hamiltonian,
    teleport_state, fidelity,
    chsh_quantum_win_rate, chsh_classical_win_rate, tsirelson_bound,
    h2_hamiltonian, h2_sto3g_hamiltonian, vqe, h2_ansatz,
)
from qbit_simulator.simlog import SimLog
from qbit_simulator.explain import explain_circuit
from qbit_simulator.cache import CircuitCache, run_cached


SAVE_DIR = Path(__file__).parent / "data" / "circuits"


# ---- command help table ----

HELP = [
    ("CIRCUIT BUILDING", [
        ("circuit <N>",        "create a new N-qubit circuit"),
        ("h <q>",              "Hadamard on qubit q"),
        ("x|y|z|s|t <q>",      "Pauli/phase gates on qubit q"),
        ("rx|ry|rz <theta> <q>", "parameterized rotation"),
        ("cnot <c> <t>",       "controlled-NOT"),
        ("cz <c> <t>",         "controlled-Z"),
        ("cp <phi> <c> <t>",   "controlled phase"),
        ("swap <a> <b>",       "swap two qubits"),
        ("reset",              "discard the current circuit"),
    ]),
    ("INSPECTION", [
        ("show",               "ASCII diagram + final probabilities"),
        ("probs",              "full probability distribution"),
        ("counts [shots]",     "sample N measurements (default 1024)"),
        ("explain",            "narrative gate-by-gate description"),
    ]),
    ("ALGORITHMS", [
        ("bell",               "Bell pair (2 qubits)"),
        ("grover <N> [marked=k]", "Grover search on N qubits"),
        ("qft <N>",            "Quantum Fourier Transform circuit"),
        ("shor <N>",           "factor integer N with Shor's algorithm"),
        ("vqe [R]",            "VQE on the Hubbard H2 model at bond length R (Å)"),
        ("vqe_sto3g [R]",      "VQE on the literature-exact STO-3G H2 model"),
        ("qaoa",               "QAOA on a small Max-Cut example"),
        ("teleport [theta]",   "teleport a single-qubit state (parameterized)"),
        ("chsh [rounds]",      "play the CHSH game; quantum vs classical win rate"),
        ("deutsch <kind>",     "Deutsch with kind=const0|const1|x|notx"),
    ]),
    ("I/O", [
        ("save <name>",        "save current circuit to disk"),
        ("load <name>",        "load a previously saved circuit"),
        ("list",               "list saved circuits"),
    ]),
    ("TELEMETRY", [
        ("log [N]",            "last N events from the lifetime log"),
        ("stats",              "aggregate lifetime stats"),
        ("cache",              "show circuit-cache stats"),
        ("cache_clear",        "wipe the circuit cache"),
    ]),
    ("META", [
        ("help",               "show this list"),
        (":quit / :q",         "exit"),
    ]),
]

KNOWN_COMMANDS: set[str] = set()
for _, items in HELP:
    for cmd, _ in items:
        head = cmd.split()[0]
        for token in head.replace("|", " ").split():
            KNOWN_COMMANDS.add(token.lstrip(":<"))


def show_help() -> None:
    print()
    for category, items in HELP:
        print(f"  {category}")
        for cmd, desc in items:
            print(f"    {cmd:<26} {desc}")
        print()


def suggest_command(token: str) -> str | None:
    matches = difflib.get_close_matches(token, list(KNOWN_COMMANDS), n=1, cutoff=0.6)
    return matches[0] if matches else None


# ---- circuit helpers ----

def _need_circuit(state: dict) -> QuantumCircuit | None:
    qc = state.get("qc")
    if qc is None:
        print("  no circuit yet — `circuit <N>` first")
        return None
    return qc


def _print_state(qc: QuantumCircuit, top_k: int = 6) -> None:
    """Print top-K basis state probabilities."""
    probs = qc.probabilities()
    order = np.argsort(probs)[::-1]
    print(f"  state ({qc.n} qubits, top {top_k}):")
    for idx in order[:top_k]:
        if probs[idx] < 1e-4:
            break
        bits = format(int(idx), f"0{qc.n}b")
        bar = "#" * int(probs[idx] * 30)
        print(f"    |{bits}>  {probs[idx]:.4f}  {bar}")


# ---- main loop ----

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-log", action="store_true",
                        help="don't record events to qsim_log.jsonl")
    parser.add_argument("--no-cache", action="store_true",
                        help="disable the circuit memoization cache")
    args = parser.parse_args()

    log = None if args.no_log else SimLog()
    cache = None if args.no_cache else CircuitCache()

    print()
    print("=" * 70)
    print("  QBot Quantum Simulator — unified REPL")
    print("=" * 70)
    print(f"  Algorithms available: bell, deutsch, grover, qft, shor, vqe,")
    print(f"                        qaoa, teleport, chsh, phase_estimation")
    if log is not None:
        s = log.lifetime_stats()
        if s["total_runs"]:
            print(f"  Lifetime runs: {s['total_runs']}   "
                  f"biggest N: {s['biggest_n_qubits']}   "
                  f"total compute: {s['total_compute_seconds']:.1f}s")
    print()
    print("Type 'help' for command list. ':quit' to exit.\n")

    state: dict = {"qc": None, "last_result": None, "cache": cache}

    while True:
        try:
            line = input("qsim> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not line:
            continue
        low = line.lower()
        if low in {":quit", ":q", "exit", "quit"}:
            break

        try:
            handled = handle(line, low, state, log)
        except (ValueError, IndexError, KeyError) as e:
            print(f"  ! {type(e).__name__}: {e}")
            handled = True
        except Exception as e:
            print(f"  ! error: {type(e).__name__}: {e}")
            traceback.print_exc(limit=2)
            handled = True

        if not handled:
            token = low.split()[0]
            suggestion = suggest_command(token)
            if suggestion:
                print(f"  unknown command '{token}'. did you mean '{suggestion}'?  "
                      f"(or type 'help')")
            else:
                print(f"  unknown command '{token}'. type 'help' for the list.")

    print("\ngoodbye.")


def handle(line: str, low: str, state: dict, log: SimLog | None) -> bool:
    """Return True if recognized. False -> show unknown-command suggestion."""
    tokens = line.split()
    head = tokens[0].lower()

    # ---- meta ----
    if head == "help":
        show_help(); return True

    # ---- circuit building ----
    if head == "circuit":
        n = int(tokens[1])
        state["qc"] = QuantumCircuit(n)
        print(f"  new circuit with {n} qubits")
        return True
    if head == "reset":
        state["qc"] = None
        print("  circuit cleared")
        return True
    if head in {"h", "x", "y", "z", "s", "t"}:
        qc = _need_circuit(state)
        if qc is None: return True
        q = int(tokens[1])
        getattr(qc, head)(q)
        print(f"  {head.upper()} on q{q}")
        return True
    if head in {"rx", "ry", "rz"}:
        qc = _need_circuit(state)
        if qc is None: return True
        theta = float(tokens[1]); q = int(tokens[2])
        getattr(qc, head)(theta, q)
        print(f"  {head}({theta}) on q{q}")
        return True
    if head == "cnot":
        qc = _need_circuit(state)
        if qc is None: return True
        c, t = int(tokens[1]), int(tokens[2])
        qc.cnot(c, t)
        print(f"  CNOT control=q{c} target=q{t}")
        return True
    if head == "cz":
        qc = _need_circuit(state)
        if qc is None: return True
        c, t = int(tokens[1]), int(tokens[2])
        qc.cz(c, t)
        print(f"  CZ q{c} -> q{t}")
        return True
    if head == "cp":
        qc = _need_circuit(state)
        if qc is None: return True
        phi = float(tokens[1]); c = int(tokens[2]); t = int(tokens[3])
        qc.cp(phi, c, t)
        print(f"  CP({phi}) q{c} -> q{t}")
        return True
    if head == "swap":
        qc = _need_circuit(state)
        if qc is None: return True
        a, b = int(tokens[1]), int(tokens[2])
        qc.swap(a, b)
        print(f"  SWAP q{a} <-> q{b}")
        return True

    # ---- inspection ----
    if head == "show":
        qc = _need_circuit(state)
        if qc is None: return True
        from qbit_simulator.viz import circuit_ascii
        print(circuit_ascii(qc))
        _print_state(qc)
        return True
    if head == "probs":
        qc = _need_circuit(state)
        if qc is None: return True
        _print_state(qc, top_k=2**min(qc.n, 6))
        return True
    if head == "counts":
        qc = _need_circuit(state)
        if qc is None: return True
        shots = int(tokens[1]) if len(tokens) > 1 else 1024
        counts = qc.counts(shots=shots)
        print(f"  {shots} measurements:")
        for k in sorted(counts):
            bar = "#" * int(counts[k] / shots * 30)
            print(f"    |{k}>  {counts[k]:>4}  {bar}")
        return True
    if head == "explain":
        qc = _need_circuit(state)
        if qc is None: return True
        print(explain_circuit(qc))
        return True

    # ---- algorithms ----
    cache: CircuitCache | None = state.get("cache")

    if head == "bell":
        def _build():
            return QuantumCircuit(2).h(0).cnot(0, 1)
        if cache is not None:
            qc = run_cached(cache, "bell", {}, _build)
            hit = qc.history == ["(restored from cache)"]
        else:
            qc = _build()
            hit = False
        state["qc"] = qc
        _print_state(qc)
        if hit: print("  (cache hit)")
        if log and not hit:
            log.record("bell", n_qubits=2, wall_seconds=0.001,
                       result_summary={"probs": [0.5, 0.0, 0.0, 0.5]})
        return True
    if head == "grover":
        n = int(tokens[1])
        marked = 2**n - 1
        for tok in tokens[2:]:
            if tok.startswith("marked="):
                marked = int(tok.split("=", 1)[1])
        args_d = {"n": n, "marked": marked}
        def _build():
            return grover(n, marked)
        if cache is not None:
            qc = run_cached(cache, "grover", args_d, _build)
            hit = qc.history == ["(restored from cache)"]
        else:
            with (log.measure("grover", n_qubits=n, marked=marked)
                  if log else _nullctx()) as bag:
                qc = _build()
                if log:
                    bag["result_summary"] = {"marked": marked,
                                              "p_found": float(qc.probabilities()[marked])}
            hit = False
        if cache is not None and not hit and log:
            log.record("grover", n_qubits=n, wall_seconds=0.0,
                       result_summary={"marked": marked,
                                        "p_found": float(qc.probabilities()[marked])})
        p_found = float(qc.probabilities()[marked])
        state["qc"] = qc
        print(f"  grover({n}, marked={marked}): P(found) = {p_found:.4f}"
              + ("   [cache hit]" if hit else ""))
        return True
    if head == "qft":
        n = int(tokens[1])
        args_d = {"n": n}
        def _build():
            return qft(n)
        if cache is not None:
            qc = run_cached(cache, "qft", args_d, _build)
            hit = qc.history == ["(restored from cache)"]
        else:
            with (log.measure("qft", n_qubits=n) if log else _nullctx()) as bag:
                qc = _build()
            hit = False
        if cache is not None and not hit and log:
            log.record("qft", n_qubits=n, wall_seconds=0.0)
        state["qc"] = qc
        print(f"  qft({n}) built ({len(qc.history)} gates)"
              + ("   [cache hit]" if hit else ""))
        _print_state(qc, top_k=8 if n <= 4 else 4)
        return True
    if head == "shor":
        N = int(tokens[1])
        with (log.measure("shor", n_qubits=2 * int(np.ceil(np.log2(N))),
                          target=N) if log else _nullctx()) as bag:
            factors = shor(N)
            if log: bag["result_summary"] = {"target": N, "factors": factors}
        print(f"  shor({N}) -> {factors}")
        return True
    if head == "vqe":
        R = float(tokens[1]) if len(tokens) > 1 else 0.74
        H = h2_hamiltonian(R)
        with (log.measure("vqe", n_qubits=2, R=R, model="hubbard")
              if log else _nullctx()) as bag:
            theta_opt, e_opt, trace = vqe(H, h2_ansatz, theta0=0.0,
                                           bounds=(-np.pi, np.pi))
            e_exact, _ = H.ground_state()
            if log: bag["result_summary"] = {"R": R, "E_vqe": float(e_opt),
                                              "E_exact": float(e_exact)}
        print(f"  vqe(R={R} Å, Hubbard H2):")
        print(f"    optimal theta = {theta_opt:.4f}")
        print(f"    E_VQE   = {e_opt:.6f} Ha")
        print(f"    E_exact = {e_exact:.6f} Ha")
        return True
    if head == "vqe_sto3g":
        R = float(tokens[1]) if len(tokens) > 1 else 0.74
        H = h2_sto3g_hamiltonian(R)
        with (log.measure("vqe_sto3g", n_qubits=2, R=R, model="sto-3g")
              if log else _nullctx()) as bag:
            theta_opt, e_opt, _ = vqe(H, h2_ansatz, theta0=0.0,
                                       bounds=(-np.pi, np.pi))
            e_exact, _ = H.ground_state()
            if log: bag["result_summary"] = {"R": R, "E_vqe": float(e_opt),
                                              "E_exact": float(e_exact)}
        print(f"  vqe_sto3g(R={R} Å, literature STO-3G):")
        print(f"    E_VQE   = {e_opt:.6f} Ha")
        print(f"    E_exact = {e_exact:.6f} Ha  (literature ~ -1.137 at 0.74)")
        return True
    if head == "qaoa":
        # Default: triangle Max-Cut.
        edges = [(0, 1), (1, 2), (0, 2)]
        with (log.measure("qaoa", n_qubits=3) if log else _nullctx()) as bag:
            theta_opt, cost, _ = qaoa(edges, n_qubits=3, p=2, seed=0)
            if log: bag["result_summary"] = {"cost": float(cost)}
        print(f"  qaoa(triangle): max-cut estimate = {cost:.4f}  (theoretical max = 2)")
        return True
    if head == "teleport":
        theta = float(tokens[1]) if len(tokens) > 1 else 0.6
        a = np.cos(theta / 2); b = np.sin(theta / 2)
        with (log.measure("teleport", n_qubits=3, theta=theta)
              if log else _nullctx()) as bag:
            qc, ar, br, (m0, m1) = teleport_state(complex(a), complex(b))
            f = fidelity(complex(a), complex(b), ar, br)
            if log: bag["result_summary"] = {"fidelity": f, "m0": m0, "m1": m1}
        print(f"  teleport(theta={theta}):")
        print(f"    Bell measurement: m0={m0}, m1={m1}")
        print(f"    fidelity (Alice initial vs Bob final) = {f:.6f}")
        return True
    if head == "chsh":
        rounds = int(tokens[1]) if len(tokens) > 1 else 2000
        with (log.measure("chsh", n_qubits=2, rounds=rounds)
              if log else _nullctx()) as bag:
            rng = np.random.default_rng()
            q_rate = chsh_quantum_win_rate(rounds, rng=rng)
            c_rate = chsh_classical_win_rate(rounds, rng=rng)
            bound = tsirelson_bound()
            if log: bag["result_summary"] = {"quantum": q_rate, "classical": c_rate,
                                              "tsirelson": bound}
        print(f"  chsh({rounds} rounds):")
        print(f"    quantum win rate:   {q_rate:.4f}")
        print(f"    classical win rate: {c_rate:.4f}")
        print(f"    Tsirelson bound:    {bound:.4f}")
        return True
    if head == "deutsch":
        kind = tokens[1].lower()
        from qbit_simulator.algorithms.deutsch import deutsch as _deutsch
        if kind in ("const0", "constant0"):
            fn = lambda x: 0; label = "constant_0"
        elif kind in ("const1", "constant1"):
            fn = lambda x: 1; label = "constant_1"
        elif kind == "x":
            fn = lambda x: x; label = "balanced_x"
        elif kind == "notx":
            fn = lambda x: 1 - x; label = "balanced_notx"
        else:
            print(f"  unknown kind {kind!r}. choose: const0, const1, x, notx")
            return True
        result = _deutsch(fn)
        print(f"  deutsch(f={label}) -> classified as: {result}")
        if log: log.record("deutsch", n_qubits=2, wall_seconds=0.001,
                           result_summary={"f": label, "classification": result})
        return True

    # ---- I/O ----
    if head == "save":
        qc = _need_circuit(state)
        if qc is None: return True
        name = tokens[1]
        SAVE_DIR.mkdir(parents=True, exist_ok=True)
        qc.save(SAVE_DIR / f"{name}.npz")
        print(f"  saved -> {SAVE_DIR / f'{name}.npz'}")
        return True
    if head == "load":
        name = tokens[1]
        path = SAVE_DIR / f"{name}.npz"
        if not path.exists():
            print(f"  not found: {path}"); return True
        state["qc"] = QuantumCircuit.load(path)
        print(f"  loaded {state['qc'].n}-qubit circuit ({len(state['qc'].history)} gates)")
        return True
    if head == "list":
        if not SAVE_DIR.exists() or not list(SAVE_DIR.glob("*.npz")):
            print("  no saved circuits"); return True
        for p in sorted(SAVE_DIR.glob("*.npz")):
            print(f"  {p.stem}  ({p.stat().st_size} bytes)")
        return True

    # ---- telemetry ----
    if head == "log":
        if log is None:
            print("  logging disabled (--no-log)"); return True
        n = int(tokens[1]) if len(tokens) > 1 else 10
        for e in log.recent(n):
            ts = time.strftime("%H:%M:%S", time.localtime(e.timestamp))
            print(f"  [{ts}] {e.kind:<14} n={e.n_qubits:<3} "
                  f"{e.wall_seconds:>6.2f}s  peak={e.peak_heap_mb:>6.2f}MB")
        return True
    if head == "stats":
        if log is None:
            print("  logging disabled"); return True
        print(log.report())
        return True
    if head == "cache":
        if cache is None:
            print("  cache disabled (--no-cache)"); return True
        print(cache.report())
        return True
    if head == "cache_clear":
        if cache is None:
            print("  cache disabled"); return True
        n = len(cache.keys())
        cache.clear()
        cache.save()
        print(f"  cleared {n} cached entries")
        return True

    return False


class _nullctx:
    """Drop-in replacement for `log.measure` when logging is disabled."""
    def __enter__(self): return {}
    def __exit__(self, *a): pass


if __name__ == "__main__":
    main()
