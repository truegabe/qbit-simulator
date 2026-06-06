"""
GUBIT 13 — IBM Quantum Test Suite
==================================
Three pre-registered experiments based on IBM_Quantum_Research_Guide.md
and the GUBIT 12 "proposed next experiments" section.

  NEXT-1  EXP F re-run  — biased QAOA vs brain.think(), corrected bias_h=0.5
  NEXT-2  Mermin-GHZ    — quantum-classical gap at N = 3, 5, 7 qubits
  NEXT-3  Bell decay     — CHSH S value vs chip-distance on ibm_kingston

Usage
-----
  1. Fill in TOKEN and INSTANCE below.
  2. (Optional) populate REAL_QUERIES with 10 lines from your chat log.
  3. Run:  python gubit13_quantum_suite.py

All circuits are packed into ONE job (one queue slot).
Results are printed to stdout and saved to gubit13_results.json.

Package requirements (already installed per IBM_Quantum_Research_Guide.md):
  qiskit==2.4.1
  qiskit-ibm-runtime==0.47.0
"""

from __future__ import annotations
import sys, math, time, json, collections

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Credentials ───────────────────────────────────────────────────────────────
TOKEN    = "p4lFUSJGEcl1TqppM1RK9IRuS0qGksNcVzLQ5E8P5i1V"
INSTANCE = "crn:v1:bluemix:public:quantum-computing:us-east:a/3af9ce945ed64bcf9c33288507f9700f:1ed7fd17-83e5-4c8a-8dd7-40d37e3a735c::"

SHOTS = 4096   # same for all circuits — one job, one shot count

# ═════════════════════════════════════════════════════════════════════════════
# PRE-REGISTRATION  (written before QPU submission — do not edit after submit)
# ═════════════════════════════════════════════════════════════════════════════
PRE_REGISTRATION = """
╔══════════════════════════════════════════════════════════════════════════════╗
║  GUBIT 13  PRE-REGISTRATION   (written 2026-05-23, before QPU submission)  ║
╚══════════════════════════════════════════════════════════════════════════════╝

NEXT-1 — EXP F re-run (GUBIT 12 EXP F, corrected bias)
  Change from GUBIT 12: bias_h = 0.5  (was 1.5 — over-pinned seed, kill-switch)
  Queries: REAL_QUERIES if populated, else 10 synthetic from GUBIT-10 concepts
  Hypothesis: QPU top-3 states match brain.think() on >= 70% of queries
  Decision rule:
      match_rate > 0.70  →  QPU can substitute for think() on uncertain queries
      0.40 – 0.70        →  use as ensemble vote
      < 0.40             →  kill switch (distrust only if bias bug is fixed)
  Prediction: corrected bias should lift match rate above GUBIT-12's 36.67%

NEXT-2 — Mermin-GHZ scaling  (N = 3, 5, 7)
  N=3: 4 circuits measure the full Mermin polynomial M3 = XXX - XYY - YXY - YYX
  N=5,7: 4 circuits each measure GHZ fidelity + three Pauli correlators
  Classical bound on M3: |M3| <= 2.  Quantum maximum: M3 = 4.
  Predictions:
      N=3  fidelity ~0.96, M3 ~ 3.5  (violation expected, same as GUBIT-10 GHZ)
      N=5  fidelity ~0.80, M_est ~ 2.5  (violation likely)
      N=7  fidelity ~0.55, M_est ~ 1.8  (below bound — noise may dominate)
  Shot noise floor at 4096 shots, 2^N states:
      N=3 (8 states):   ~0.011
      N=5 (32 states):  ~0.005
      N=7 (128 states): ~0.003

NEXT-3 — Chip-distance Bell decay  (ibm_kingston topology)
  4 qubit-pairs at increasing topological distances (found via BFS on coupling map)
  4 CHSH measurement settings per pair = 16 circuits
  Classical bound: S <= 2  |  Tsirelson bound: S <= 2*sqrt(2) = 2.828
  Prediction: S degrades ~0.15 per doubling of hop-distance
      d=1: S ~ 2.7      d=2: S ~ 2.55
      d=4: S ~ 2.3      d=8: S ~ 2.0  (borderline)
  Key output: "Bell decay length" — max distance at which S > 2 on this chip.
"""
# PRE_REGISTRATION is printed inside main() so importing this module is side-effect-free.

# ═════════════════════════════════════════════════════════════════════════════
# BRAIN DATA  (GUBIT 10 J matrix — same as GUBIT 12)
# ═════════════════════════════════════════════════════════════════════════════
CONCEPTS_6 = ["bus", "car", "passenger", "music", "song", "concert"]   # baby excluded
CONCEPTS_7 = ["bus", "car", "passenger", "music", "song", "concert", "baby"]

J_7_RAW = {
    ("bus",       "car"):       0.65,
    ("bus",       "passenger"): 0.70,
    ("bus",       "music"):     0.10,
    ("bus",       "baby"):      0.15,
    ("car",       "passenger"): 0.55,
    ("music",     "song"):      0.90,
    ("music",     "concert"):   0.70,
    ("music",     "baby"):      0.40,
    ("song",      "concert"):   0.60,
    ("song",      "baby"):      0.35,
    ("concert",   "baby"):      0.20,
}

def subgraph_j(concepts: list[str], j_raw: dict) -> dict[tuple[int,int], float]:
    """Return index-keyed J subgraph for a list of concept names."""
    j_sub = {}
    for i, ci in enumerate(concepts):
        for k, ck in enumerate(concepts):
            if i >= k:
                continue
            for key in ((ci, ck), (ck, ci)):
                if key in j_raw:
                    j_sub[(i, k)] = j_raw[key]
                    break
    return j_sub

# Classical think() — top associations by J weight for each concept
def classical_think(seed: str, j_raw: dict, concepts: list[str], top_n: int = 5) -> list[str]:
    pairs = []
    for (a, b), w in j_raw.items():
        if a == seed:   pairs.append((b, w))
        elif b == seed: pairs.append((a, w))
    pairs.sort(key=lambda x: -x[1])
    return [seed] + [p[0] for p in pairs[:top_n - 1]]

THINK = {c: classical_think(c, J_7_RAW, CONCEPTS_7) for c in CONCEPTS_7}

# ── Real queries (fill these in from your chat log before running) ─────────────
# Format: (seed_concept, [concept_list_of_5])
# If left empty the script falls back to 10 synthetic queries.
REAL_QUERIES: list[tuple[str, list[str]]] = [
    # ("music", ["music", "song", "concert", "bus", "baby"]),   # <- example
]

SYNTHETIC_QUERIES: list[tuple[str, list[str]]] = [
    ("music",     ["music", "song",    "concert",   "bus",       "baby"]),
    ("bus",       ["bus",   "car",     "passenger",  "music",     "baby"]),
    ("song",      ["song",  "music",   "concert",    "car",       "baby"]),
    ("concert",   ["concert","music",  "song",       "bus",       "baby"]),
    ("car",       ["car",   "bus",     "passenger",  "music",     "baby"]),
    ("passenger", ["passenger","bus",  "car",        "music",     "baby"]),
    ("music",     ["music", "song",    "concert",    "passenger", "car"]),
    ("bus",       ["bus",   "car",     "passenger",  "concert",   "song"]),
    ("song",      ["song",  "music",   "concert",    "bus",       "car"]),
    ("concert",   ["concert","music",  "song",       "passenger", "car"]),
]

QUERIES = REAL_QUERIES if REAL_QUERIES else SYNTHETIC_QUERIES
# Note: if REAL_QUERIES has <10 entries the match-rate denominator stays correct
# (we just run fewer circuits and divide by len(QUERIES)).
# Message is printed inside main() to avoid output on import.

# ═════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════
from qiskit import QuantumCircuit

def get_counts(pub_result) -> dict:
    """Extract measurement counts from a SamplerV2 PubResult."""
    data = pub_result.data
    for name in ("c", "meas", "c0"):
        b = getattr(data, name, None)
        if b is not None and hasattr(b, "get_counts"):
            return b.get_counts()
    for name in vars(data):
        a = getattr(data, name, None)
        if hasattr(a, "get_counts"):
            return a.get_counts()
    raise RuntimeError(f"no counts found in PubResult — attrs: {list(vars(data))}")

def jsd(p: dict, q: dict) -> float:
    """Jensen-Shannon divergence between two count dicts (symmetric, in nats)."""
    tp = sum(p.values()) or 1
    tq = sum(q.values()) or 1
    keys = set(p) | set(q)
    js = 0.0
    for k in keys:
        pk = p.get(k, 0) / tp
        qk = q.get(k, 0) / tq
        m  = (pk + qk) / 2
        if pk > 0 and m > 0: js += pk * math.log(pk / m)
        if qk > 0 and m > 0: js += qk * math.log(qk / m)
    return js / 2

def shot_noise_floor(shots: int, n_states: int) -> float:
    """1-sigma statistical noise on a single probability bin (uniform prior)."""
    p = 1.0 / n_states
    return math.sqrt(p * (1 - p) / shots)

def parity_correlator(counts: dict, shots: int) -> float:
    """<(-1)^(sum of bits)> — the N-qubit ZZ...Z correlator in a rotated basis."""
    total = 0
    for bitstring, cnt in counts.items():
        parity = sum(int(b) for b in bitstring) % 2
        total += (1 - 2 * parity) * cnt
    return total / shots

def ghz_fidelity(counts: dict, N: int, shots: int) -> float:
    """P(0^N) + P(1^N) from a Z-basis GHZ measurement."""
    zeros = "0" * N
    ones  = "1" * N
    return (counts.get(zeros, 0) + counts.get(ones, 0)) / shots

def top_states(counts: dict, shots: int, k: int = 5) -> list[tuple[str, float]]:
    return [(bs, cnt / shots)
            for bs, cnt in sorted(counts.items(), key=lambda x: -x[1])[:k]]

def active_concepts(bitstring: str, concepts: list[str]) -> list[str]:
    """Return concepts where the corresponding bit is '1' (Qiskit: bit 0 = rightmost)."""
    return [concepts[i]
            for i, b in enumerate(reversed(bitstring))
            if b == "1" and i < len(concepts)]

def overlap_count(a: list[str], b: list[str]) -> int:
    return len(set(a) & set(b))

# ═════════════════════════════════════════════════════════════════════════════
# NEXT-1: EXP F RE-RUN  — biased QAOA per query
# ═════════════════════════════════════════════════════════════════════════════
GAMMA   = 0.6
BETA    = 0.3
P_LAYERS = 2
BIAS_H  = 0.5   # was 1.5 in GUBIT 12 — corrected here

def build_biased_qaoa(concepts: list[str], j_sub: dict,
                      gamma: float, beta: float, p: int,
                      seed_idx: int, bias_h: float) -> QuantumCircuit:
    """
    QAOA p-layer circuit with a bias field on the seed qubit.
    The bias Rz(-2*gamma*bias_h) nudges the seed concept to be active
    without overwhelming the J couplings (bias_h=0.5 << 1.5).
    """
    n = len(concepts)
    qc = QuantumCircuit(n, n)
    qc.h(range(n))                        # equal superposition

    for _ in range(p):
        # ── Cost operator: J couplings as ZZ terms ────────────────────────
        for (i, k), w in j_sub.items():
            qc.cx(i, k)
            qc.rz(2 * gamma * w, k)
            qc.cx(i, k)
        # ── Bias field on seed qubit (local h term in Ising Hamiltonian) ──
        if bias_h > 0:
            qc.rz(-2 * gamma * bias_h, seed_idx)
        # ── Mixer operator ────────────────────────────────────────────────
        qc.rx(2 * beta, range(n))

    qc.measure(range(n), range(n))
    return qc

def build_next1_circuits() -> tuple[list[QuantumCircuit], list[dict]]:
    circuits, meta = [], []
    for seed, concept_list in QUERIES:
        n        = len(concept_list)
        seed_idx = concept_list.index(seed)
        j_sub    = subgraph_j(concept_list, J_7_RAW)
        qc       = build_biased_qaoa(concept_list, j_sub,
                                     GAMMA, BETA, P_LAYERS,
                                     seed_idx, BIAS_H)
        circuits.append(qc)
        meta.append({
            "seed":           seed,
            "concepts":       concept_list,
            "seed_idx":       seed_idx,
            "classical_top5": THINK.get(seed, [seed]),
        })
    return circuits, meta

def analyse_next1(result_slice, meta: list[dict]) -> dict:
    """
    Match score: fraction of QPU top-3 bitstrings whose active concepts
    overlap by >= 3 with the classical think() top-5.
    """
    records      = []
    match_count  = 0

    for i, m in enumerate(meta):
        counts    = get_counts(result_slice[i])
        top3      = top_states(counts, SHOTS, k=3)
        classical = m["classical_top5"]
        concepts  = m["concepts"]
        matched   = False

        for bs, prob in top3:
            active = active_concepts(bs, concepts)
            if overlap_count(active, classical) >= 3:
                matched = True
                break

        if matched:
            match_count += 1

        record = {
            "seed":           m["seed"],
            "classical_top5": classical,
            "qpu_top3":       [(bs, round(p, 4)) for bs, p in top3],
            "qpu_top3_active": [active_concepts(bs, concepts) for bs, _ in top3],
            "matched":        matched,
        }
        records.append(record)
        status = "OK " if matched else "---"
        print(f"  [{status}] seed={m['seed']:10s}  "
              f"top1={top3[0][0]} ({top3[0][1]:.1%})  "
              f"classical={classical[1:]}")

    match_rate = match_count / len(meta)
    print(f"\n  Match rate: {match_count}/{len(meta)} = {match_rate:.1%}")

    if match_rate > 0.70:
        verdict = "QPU AGREES — can substitute for think() on uncertain queries"
    elif match_rate >= 0.40:
        verdict = "PARTIAL — use as ensemble vote alongside think()"
    else:
        verdict = ("KILL SWITCH — QPU disagrees with think() "
                   "(check: are queries real? is bias_h still too high?)")

    print(f"  Verdict: {verdict}\n")
    return {"match_rate": round(match_rate, 4), "verdict": verdict, "records": records}

# ═════════════════════════════════════════════════════════════════════════════
# NEXT-2: MERMIN-GHZ  — quantum-classical gap at N = 3, 5, 7
# ═════════════════════════════════════════════════════════════════════════════
MERMIN_SIZES = [3, 5, 7]

def _ghz_base(N: int) -> QuantumCircuit:
    """GHZ state preparation: H on qubit 0, then LINEAR CNOT chain.
    Linear chain (cx(i, i+1)) instead of star (cx(0, i)) — far fewer SWAPs
    on heavy-hex topology because adjacent circuit qubits map to adjacent
    physical qubits after transpilation.
    """
    qc = QuantumCircuit(N, N)
    qc.h(0)
    for i in range(N - 1):
        qc.cx(i, i + 1)   # linear, not star
    return qc

def _apply_basis(qc: QuantumCircuit, basis_str: str) -> None:
    """
    Rotate each qubit into the specified measurement basis.
    'X' -> H gate (Hadamard rotates Z-basis to X-basis)
    'Y' -> Sdg + H  (S-dagger then Hadamard rotates Z-basis to Y-basis)
    'Z' -> no rotation
    """
    for i, b in enumerate(basis_str):
        if b == "X":
            qc.h(i)
        elif b == "Y":
            qc.sdg(i)
            qc.h(i)
        # 'Z' — no rotation needed

def build_mermin_circuits() -> tuple[list[QuantumCircuit], list[dict]]:
    """
    N=3 : exact 4-term Mermin polynomial  M3 = XXX - XYY - YXY - YYX
          Classical bound: |M3| <= 2.  Quantum maximum: M3 = 4.
    N=5,7: Z-basis fidelity + X^N + X^(N-2)YY + YYX^(N-2)
          Allows estimating GHZ fidelity and partial Mermin violations.
    """
    circuits, meta = [], []

    for N in MERMIN_SIZES:
        if N == 3:
            # Full exact Mermin polynomial: 4 correlator settings
            settings = ["XXX", "XYY", "YXY", "YYX"]
        else:
            # Z fidelity + 3 correlator settings
            settings = [
                "Z" * N,
                "X" * N,
                "X" * (N - 2) + "YY",
                "YY" + "X" * (N - 2),
            ]

        for basis in settings:
            qc = _ghz_base(N)
            _apply_basis(qc, basis)
            qc.measure(range(N), range(N))
            circuits.append(qc)
            meta.append({"N": N, "basis": basis})

    return circuits, meta

def analyse_next2(result_slice, meta: list[dict]) -> dict:
    """
    Compute GHZ fidelity and Mermin violation for each N.
    Classical bound for M3 (4-term): |M3| <= 2, quantum max = 4.
    """
    # Group results by N
    by_n: dict[int, dict[str, dict]] = collections.defaultdict(dict)
    for i, m in enumerate(meta):
        counts = get_counts(result_slice[i])
        by_n[m["N"]][m["basis"]] = counts

    all_results = {}
    for N in MERMIN_SIZES:
        data   = by_n[N]
        noise  = shot_noise_floor(SHOTS, 2 ** N)
        record = {"N": N, "shot_noise_floor": round(noise, 5)}

        if N == 3:
            # Exact Mermin polynomial
            c_xxx = parity_correlator(data["XXX"], SHOTS)
            c_xyy = parity_correlator(data["XYY"], SHOTS)
            c_yxy = parity_correlator(data["YXY"], SHOTS)
            c_yyx = parity_correlator(data["YYX"], SHOTS)
            M3    = c_xxx - c_xyy - c_yxy - c_yyx

            record.update({
                "C_XXX": round(c_xxx, 4),
                "C_XYY": round(c_xyy, 4),
                "C_YXY": round(c_yxy, 4),
                "C_YYX": round(c_yyx, 4),
                "M3":    round(M3, 4),
                "classical_bound":  2,
                "quantum_max":      4,
                "violation": M3 > 2,
            })
            print(f"  N=3  M3={M3:.3f}  (classical<=2, quantum=4)  "
                  f"{'VIOLATION' if M3 > 2 else 'no violation'}")
            print(f"       C_XXX={c_xxx:.3f}  C_XYY={c_xyy:.3f}  "
                  f"C_YXY={c_yxy:.3f}  C_YYX={c_yyx:.3f}")

        else:
            # Z-basis fidelity + 3 correlators
            z_basis  = "Z" * N
            x_basis  = "X" * N
            xnyy_b   = "X" * (N - 2) + "YY"
            yyxn_b   = "YY" + "X" * (N - 2)

            fidelity  = ghz_fidelity(data[z_basis], N, SHOTS)
            c_x       = parity_correlator(data[x_basis],  SHOTS)
            c_xnyy    = parity_correlator(data[xnyy_b],   SHOTS)
            c_yyxn    = parity_correlator(data[yyxn_b],   SHOTS)

            # 3-term estimate (ideal: 1 - (-1) - (-1) = 3 on a 3-term sub-polynomial)
            # Classical bound on any 3-term {±1} sum: |M_est| <= 3 (trivial)
            # For GHZ: ideal M_est = 1 - (-1) - (-1) = 3
            # We report it as a proxy; not the full Mermin value for N>3
            M_est = c_x - c_xnyy - c_yyxn

            record.update({
                "GHZ_fidelity":    round(fidelity, 4),
                "C_X":             round(c_x, 4),
                f"C_{xnyy_b}":     round(c_xnyy, 4),
                f"C_{yyxn_b}":     round(c_yyxn, 4),
                "M_est_3term":     round(M_est, 4),
                "ideal_M_est":     3,
                "note": ("M_est is a 3-term proxy — not the full Mermin polynomial. "
                         "Ideal GHZ gives M_est=3. Classical max is also 3, so this "
                         "is NOT a Bell inequality violation test — use N=3 M3 for that."),
            })
            print(f"  N={N}  GHZ_fidelity={fidelity:.3f}  C_X={c_x:.3f}  "
                  f"M_est(3-term)={M_est:.3f}")
            print(f"       C_{xnyy_b}={c_xnyy:.3f}  C_{yyxn_b}={c_yyxn:.3f}")

        all_results[N] = record

    return all_results

# ═════════════════════════════════════════════════════════════════════════════
# NEXT-3: CHIP-DISTANCE BELL DECAY
# ═════════════════════════════════════════════════════════════════════════════
TARGET_DISTANCES = [1, 2, 4, 8]   # topological hops on ibm_kingston

def find_pairs_by_distance(backend, target_distances: list[int]) -> dict[int, tuple[int,int]]:
    """
    BFS on ibm_kingston's coupling map.
    Returns {target_distance: (q0, q1)} — the closest physical pair found.
    Prefers high-index qubits (often better calibrated on Heron R2).
    """
    coupling_map = backend.coupling_map
    n_qubits     = backend.num_qubits

    # Build undirected adjacency from the coupling map edges
    adj: dict[int, set[int]] = {i: set() for i in range(n_qubits)}
    for edge in coupling_map.get_edges():
        adj[edge[0]].add(edge[1])
        adj[edge[1]].add(edge[0])

    # BFS from each qubit — build full distance matrix lazily
    def bfs(start: int) -> dict[int, int]:
        dist  = {start: 0}
        queue = collections.deque([start])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        return dist

    # For each target distance, find the best pair
    selected: dict[int, tuple[int,int]] = {}
    # Use multiple anchors to get diverse pairs
    anchors = [0, n_qubits // 4, n_qubits // 2]

    # Cache BFS results to avoid recomputing
    bfs_cache: dict[int, dict[int, int]] = {}
    def bfs_cached(start: int) -> dict[int, int]:
        if start not in bfs_cache:
            bfs_cache[start] = bfs(start)
        return bfs_cache[start]

    used_pairs: set[frozenset] = set()   # avoid reusing same physical pair
    for target_d in target_distances:
        best_pair  = None
        best_delta = float("inf")
        for anchor in anchors:
            dists = bfs_cached(anchor)
            for q, d in sorted(dists.items()):   # sorted for determinism
                if q == anchor:
                    continue
                if frozenset((anchor, q)) in used_pairs:
                    continue
                delta = abs(d - target_d)
                if delta < best_delta:
                    best_delta = delta
                    best_pair  = (anchor, q)
        if best_pair is None:
            raise RuntimeError(f"Could not find a qubit pair near distance {target_d}")
        used_pairs.add(frozenset(best_pair))
        selected[target_d] = best_pair
        actual_d = bfs_cached(best_pair[0]).get(best_pair[1], -1)
        print(f"  [NEXT-3] target_dist={target_d}  "
              f"pair={best_pair}  actual_hops={actual_d}")

    return selected

def build_chsh_circuits(q0: int, q1: int
                        ) -> tuple[list[QuantumCircuit], list[tuple]]:
    """
    4 CHSH measurement settings for a Bell pair.
    Uses a 2-qubit circuit (logical qubits 0, 1); physical placement is set
    via initial_layout=[q0, q1] at transpile time — NOT n_qubits-wide circuits.

    Angle convention for |Phi+>: a=0, a'=pi/2, b=pi/4, b'=-pi/4.
    E(alpha, beta) = cos(alpha - beta) for |Phi+>.
    S = |E(a,b) + E(a,b') + E(a',b) - E(a',b')|
      = |cos(-pi/4) + cos(pi/4) + cos(pi/4) - cos(3pi/4)|
      = |1/sqrt2 + 1/sqrt2 + 1/sqrt2 + 1/sqrt2| = 2*sqrt(2)  [ideal]
    Classical bound: S <= 2.  Tsirelson bound: 2*sqrt(2) = 2.828.
    """
    angle_pairs = [           # (alice_angle, bob_angle)
        (0,              math.pi / 4),   # E(a,  b )
        (0,             -math.pi / 4),   # E(a,  b')
        (math.pi / 2,    math.pi / 4),   # E(a', b )
        (math.pi / 2,   -math.pi / 4),   # E(a', b')  ← subtracted in chsh_s
    ]

    circuits, settings = [], []
    for alice_a, bob_a in angle_pairs:
        qc = QuantumCircuit(2, 2)          # 2 logical qubits only
        # Bell pair on logical qubits 0 and 1
        qc.h(0)
        qc.cx(0, 1)
        # Measurement-basis rotations
        if alice_a != 0:
            qc.ry(alice_a, 0)
        if bob_a != 0:
            qc.ry(bob_a, 1)
        qc.measure(0, 0)
        qc.measure(1, 1)
        circuits.append(qc)
        settings.append((alice_a, bob_a))

    return circuits, settings

def correlator_2q(counts: dict, shots: int) -> float:
    """<Z_0 Z_1> correlator from a 2-bit measurement result.
    Qiskit bitstring: rightmost character = classical bit 0 = qubit 0.
    2-qubit CHSH circuits always produce 2-character bitstrings.
    """
    total = 0
    for bitstring, cnt in counts.items():
        b0 = int(bitstring[-1])    # qubit 0  (Alice)
        b1 = int(bitstring[-2])    # qubit 1  (Bob)
        total += (1 - 2 * b0) * (1 - 2 * b1) * cnt
    return total / shots

def chsh_s(e_ab: float, e_ab_: float, e_a_b: float, e_a_b_: float) -> float:
    # Angle convention: Alice 0/pi/2, Bob pi/4/-pi/4.
    # For |Phi+>, E(a,b)=cos(a-b): last term E(a',b')=cos(3pi/4)=-1/sqrt(2)
    # so S = |1/sqrt2 + 1/sqrt2 + 1/sqrt2 - (-1/sqrt2)| = 2sqrt2  (correct)
    # The minus is on the LAST term, not the second.
    return abs(e_ab + e_ab_ + e_a_b - e_a_b_)

def build_next3_circuits(backend) -> tuple[list[QuantumCircuit], list[dict], list]:
    """
    Find physical qubit pairs then build 4 CHSH circuits per pair.

    Returns (logical_circuits, meta, transpiled_circuits).
    NEXT-3 circuits are ALREADY TRANSPILED here because each pair needs a
    different initial_layout — we cannot use the single global pm from main().
    optimization_level=1 preserves the initial_layout reliably; level=3 may
    reroute freely and destroy the distance measurement.
    """
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    pairs = find_pairs_by_distance(backend, TARGET_DISTANCES)

    logical_circs, transpiled_circs, meta = [], [], []
    for target_d, (q0, q1) in pairs.items():
        chsh_circs, settings = build_chsh_circuits(q0, q1)

        # Per-pair pass manager pins logical qubit 0→q0, qubit 1→q1
        pm_pair = generate_preset_pass_manager(
            optimization_level=1,      # respect initial_layout
            backend=backend,
            initial_layout=[q0, q1],   # physical qubit indices
        )

        for qc, (alice_a, bob_a) in zip(chsh_circs, settings):
            logical_circs.append(qc)
            transpiled_circs.append(pm_pair.run(qc))
            meta.append({
                "target_dist": target_d,
                "q0": q0, "q1": q1,
                "alice_angle": alice_a,
                "bob_angle":   bob_a,
            })

    return logical_circs, meta, transpiled_circs

def analyse_next3(result_slice, meta: list[dict]) -> dict:
    """
    Compute CHSH S value for each qubit pair.
    Groups 4 consecutive circuits per pair.
    """
    # Group into blocks of 4
    grouped: dict[tuple[int,int], list] = collections.defaultdict(list)
    for i, m in enumerate(meta):
        counts = get_counts(result_slice[i])
        key    = (m["q0"], m["q1"])
        grouped[key].append((m, counts))

    results = {}
    for (q0, q1), items in grouped.items():
        assert len(items) == 4, f"Expected 4 CHSH circuits for ({q0},{q1}), got {len(items)}"
        E = [correlator_2q(counts, SHOTS) for _, counts in items]
        S  = chsh_s(*E)
        target_d = items[0][0]["target_dist"]

        label = f"d={target_d} ({q0},{q1})"
        results[label] = {
            "target_dist": target_d,
            "qubits":      [q0, q1],
            "correlators": [round(e, 4) for e in E],
            "S":           round(S, 4),
            "classical_bound":  2.0,
            "tsirelson_bound":  round(2 * math.sqrt(2), 4),
            "violation":   S > 2.0,
        }
        status = "VIOLATION" if S > 2.0 else "no violation"
        print(f"  d={target_d} qubits=({q0},{q1})  S={S:.3f}  [{status}]")

    return results

# ═════════════════════════════════════════════════════════════════════════════
# NOISELESS BASELINE (gatekeeper — runs locally on AerSimulator)
# ═════════════════════════════════════════════════════════════════════════════
def run_noiseless_baseline(n1_circuits, n1_meta):
    """
    Simulate the first NEXT-1 circuit on a noiseless AerSimulator.
    Returns a counts dict — used as JSD reference against hardware.
    Mirrors EXP A from GUBIT 12.
    """
    try:
        from qiskit_aer import AerSimulator
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

        sim = AerSimulator()
        pm  = generate_preset_pass_manager(optimization_level=3, backend=sim)
        t   = pm.run(n1_circuits[0])
        from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
        s   = AerSamplerV2()
        job = s.run([t], shots=SHOTS * 4)   # 4× shots for cleaner baseline
        counts_sim = get_counts(job.result()[0])
        print("[GATEKEEPER] Noiseless AerSimulator baseline complete.")
        return counts_sim
    except ImportError:
        print("[GATEKEEPER] qiskit-aer not installed — skipping noiseless baseline.")
        return None
    except Exception as e:
        print(f"[GATEKEEPER] Baseline failed: {e}")
        return None

# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    print(PRE_REGISTRATION)

    if REAL_QUERIES:
        print(f"[NEXT-1] Using {len(REAL_QUERIES)} REAL queries from chat log.")
    else:
        print("[NEXT-1] No real queries — using 10 synthetic (GUBIT-10 concepts).")
        print("         Populate REAL_QUERIES for a valid kill-switch verdict.\n")

    # ── Connect ───────────────────────────────────────────────────────────────
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    print("Connecting to IBM Quantum (ibm_cloud channel)...")
    service = QiskitRuntimeService(
        channel  = "ibm_cloud",   # NOT "ibm_quantum"
        token    = TOKEN,
        instance = INSTANCE,
    )
    backend = service.least_busy(min_num_qubits=7, simulator=False, operational=True)
    print(f"Backend: {backend.name}  ({backend.num_qubits} qubits)\n")

    # ── Build circuits ────────────────────────────────────────────────────────
    print("Building NEXT-1 circuits (biased QAOA)...")
    n1_circs, n1_meta = build_next1_circuits()

    print("Building NEXT-2 circuits (Mermin-GHZ, N=3/5/7)...")
    n2_circs, n2_meta = build_mermin_circuits()

    # NEXT-3: already transpiled inside build_next3_circuits (per-pair initial_layout)
    print("Building & transpiling NEXT-3 circuits (chip-distance Bell)...")
    n3_circs, n3_meta, n3_transpiled = build_next3_circuits(backend)

    # ── Noiseless baseline (local, free) ──────────────────────────────────────
    print("\nRunning noiseless AerSimulator baseline for gatekeeper check...")
    baseline_counts = run_noiseless_baseline(n1_circs, n1_meta)

    # ── Transpile NEXT-1 and NEXT-2 ───────────────────────────────────────────
    print("\nTranspiling NEXT-1 and NEXT-2 circuits (optimization_level=3)...")
    pm             = generate_preset_pass_manager(optimization_level=3, backend=backend)
    n1_transpiled  = [pm.run(qc) for qc in n1_circs]
    n2_transpiled  = [pm.run(qc) for qc in n2_circs]
    all_transpiled = n1_transpiled + n2_transpiled + n3_transpiled

    print("Transpile depths (pre → post):")
    for label, logical, transpiled_list in [
        ("N1", n1_circs, n1_transpiled),
        ("N2", n2_circs, n2_transpiled),
        ("N3", n3_circs, n3_transpiled),
    ]:
        for j, (qc, t) in enumerate(zip(logical, transpiled_list)):
            print(f"  {label}[{j:02d}]  {qc.depth():3d} → {t.depth():4d}")

    total = len(all_transpiled)
    print(f"\nTotal circuits: {total}  "
          f"(N1={len(n1_transpiled)}, N2={len(n2_transpiled)}, N3={len(n3_transpiled)})")
    print(f"Shots per circuit: {SHOTS}")

    # ── Submit ONE job ────────────────────────────────────────────────────────
    print(f"\nSubmitting {len(all_transpiled)} circuits as one job...")
    sampler = SamplerV2(mode=backend)
    job     = sampler.run(all_transpiled, shots=SHOTS)
    print(f"Job ID: {job.job_id()}")
    print("Waiting for results (this blocks until the job is done)...")
    t_start = time.time()
    result  = job.result()
    t_wall  = time.time() - t_start
    print(f"Done in {t_wall:.1f} s\n")

    # ── Slice results ─────────────────────────────────────────────────────────
    idx = 0
    n1_results = result[idx : idx + len(n1_transpiled)]; idx += len(n1_transpiled)
    n2_results = result[idx : idx + len(n2_transpiled)]; idx += len(n2_transpiled)
    n3_results = result[idx : idx + len(n3_transpiled)]; idx += len(n3_transpiled)

    # ── GATEKEEPER: hardware vs noiseless (first NEXT-1 circuit) ─────────────
    hw_counts_0 = get_counts(n1_results[0])
    gatekeeper  = {}
    if baseline_counts is not None:
        js = jsd(hw_counts_0, baseline_counts)
        top3_hw  = [bs for bs, _ in top_states(hw_counts_0, SHOTS, 3)]
        top3_sim = [bs for bs, _ in top_states(baseline_counts, SHOTS * 4, 3)]
        overlap  = len(set(top3_hw) & set(top3_sim))
        gatekeeper = {
            "JSD_hw_vs_noiseless":   round(js, 4),
            "top3_hw":               top3_hw,
            "top3_sim":              top3_sim,
            "top3_overlap":          overlap,
            "verdict": (
                "H0 — same attractors (blurrier on HW)" if overlap >= 2 else
                "H1 — different attractors (possible quantum effect)" if js > 0.30 else
                "INCONCLUSIVE"
            ),
        }
        print("── GATEKEEPER ──────────────────────────────────────────────────")
        print(f"  JSD(hw, noiseless) = {js:.4f}")
        print(f"  top-3 HW:   {top3_hw}")
        print(f"  top-3 SIM:  {top3_sim}")
        print(f"  overlap: {overlap}/3   verdict: {gatekeeper['verdict']}\n")

    # ── NEXT-1 analysis ───────────────────────────────────────────────────────
    print("── NEXT-1: EXP F re-run ────────────────────────────────────────────")
    r_n1 = analyse_next1(n1_results, n1_meta)

    # ── NEXT-2 analysis ───────────────────────────────────────────────────────
    print("── NEXT-2: Mermin-GHZ ──────────────────────────────────────────────")
    r_n2 = analyse_next2(n2_results, n2_meta)

    # ── NEXT-3 analysis ───────────────────────────────────────────────────────
    print("── NEXT-3: Chip-distance Bell decay ────────────────────────────────")
    r_n3 = analyse_next3(n3_results, n3_meta)

    # ── Save results ──────────────────────────────────────────────────────────
    output = {
        "date":         "2026-05-23",
        "backend":      backend.name,
        "job_id":       job.job_id(),
        "shots":        SHOTS,
        "wall_time_s":  round(t_wall, 1),
        "gatekeeper":   gatekeeper,
        "NEXT1":        r_n1,
        "NEXT2":        {str(k): v for k, v in r_n2.items()},
        "NEXT3":        r_n3,
    }

    out_path = "gubit13_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("GUBIT 13 SUMMARY")
    print("=" * 70)

    if gatekeeper:
        print(f"Gatekeeper JSD:   {gatekeeper['JSD_hw_vs_noiseless']:.4f}  "
              f"({gatekeeper['verdict']})")

    print(f"NEXT-1 match rate: {r_n1['match_rate']:.1%}  "
          f"({r_n1['verdict'][:40]}...)")

    if "3" in r_n2 or 3 in r_n2:
        m3 = r_n2.get(3, r_n2.get("3", {}))
        print(f"NEXT-2 M3 (N=3):   {m3.get('M3','?')}  "
              f"(classical<=2, quantum=4, violation={m3.get('violation','?')})")
    for N in [5, 7]:
        rec = r_n2.get(N, r_n2.get(str(N), {}))
        print(f"NEXT-2 N={N}:       fidelity={rec.get('GHZ_fidelity','?')}  "
              f"C_X={rec.get('C_X','?')}")

    violations = [(k, v["S"]) for k, v in r_n3.items() if v.get("violation")]
    non_viol   = [(k, v["S"]) for k, v in r_n3.items() if not v.get("violation")]
    print(f"NEXT-3 Bell decay: {len(violations)} pairs violate S>2, "
          f"{len(non_viol)} do not.")
    for k, s in sorted(violations + non_viol, key=lambda x: x[1], reverse=True):
        print(f"  {k}  S={s:.3f}")

    print("=" * 70)

if __name__ == "__main__":
    main()
