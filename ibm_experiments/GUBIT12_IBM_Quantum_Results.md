# GUBIT 12 — IBM Quantum Results

## Six pre-registered experiments × ibm_kingston

**Date:** 2026-05-23
**Backend:** ibm_kingston (156-qubit Heron R2)
**Job ID:** `d88c8sp789is7391vo0g`
**Wall time on QPU:** 38.9 seconds
**Shots:** 4096 per circuit
**Total circuits:** 28 (1 noiseless baseline + 27 hardware)
**Script:** `gubit12_quantum_suite.py`

---

## TL;DR

| EXP | Pre-registered question | Result | Action |
|---|---|---|---|
| **A** | Is the QPU finding different attractors than a noiseless QAOA? | **H0 — same attractors, blurrier.** Top-3 overlap 100%. JSD 0.018. | **Drop quantum-architecture claims for this QAOA.** |
| **B** | Does the quantum walk diverge from classical random walk on the KG? | **H1 — walks diverge.** Max JSD 0.461 at t=1.0. Leakage 7.5%. | CTQW dynamics confirmed on hardware. Does NOT prove SPECTRA filter is suboptimal. |
| **C** | Can biased QAOA discover bridge nodes like baby? | **Method validated weakly.** Baby ranked #2 but ALL 7 concepts cluster within shot noise. | Method works in principle. Signal too weak at this depth to find new bridges. |
| **D** | At what circuit-padding depth does QAOA signal degrade by 50%? | **λ_circuit = 7,408 ns**, R² = 0.94. Padding 0→800ns only cost ~10% signal. | Try **QAOA p=3** with halved Trotter steps. Depth headroom exists. |
| **E** | Does multi-seed interference produce phase-separated cluster states? | **H_compromise.** Both clusters always coexist (52–78%). Phase-dependent but no suppression. | No "multi-seed mode" needed in `respond()`. Classical sum suffices. |
| **F** | Does QPU output reproduce `brain.think()`? | **36.67% match — kill switch.** But result suspect (synthetic queries + every top-state is seed-only). | Re-run with real chat-log queries AND weaker `bias_h`. Do NOT trust this verdict yet. |

---

## Quick concepts primer

These show up in every experiment below. If you already know them, skip.

### Qubit
A two-level quantum system. Classical bit is either 0 or 1; a qubit can be in a **superposition** `α|0⟩ + β|1⟩` where α, β are complex amplitudes with `|α|² + |β|² = 1`. When measured, it "collapses" to 0 with probability `|α|²` or 1 with probability `|β|²`.

A bitstring like `|000111⟩` means qubit 5 = 0, qubit 4 = 0, qubit 3 = 0, qubit 2 = 1, qubit 1 = 1, qubit 0 = 1. Qiskit reads the rightmost character as qubit 0.

### Hamiltonian and the J matrix
A Hamiltonian H is a description of energies: which configurations of the system are low-energy (preferred) and which are high-energy (avoided). For our brain, the "J matrix" defines pairwise associations: J(bus, car) = 0.65 means bus and car like to be active together. The Hamiltonian built from this J matrix has its low-energy states corresponding to **clusters of co-active concepts**.

### QAOA — Quantum Approximate Optimization Algorithm
A way to find low-energy states of a Hamiltonian using a quantum circuit. Two knobs per layer:
- **γ (gamma):** strength of the "cost" pull toward low-energy states.
- **β (beta):** strength of the "mixer" that lets the state explore.

You apply these alternately for `p` layers. Larger `p` = deeper circuit, potentially better answers, but more noise on hardware. GUBIT 10 found `p=2, γ=0.6, β=0.3` is the sweet spot for this brain.

When you measure after QAOA, the most likely bitstrings should be the low-energy ones — for us, that's "transport cluster on" or "music cluster on."

### Noiseless simulator (AerSimulator)
A classical program that simulates exactly what a perfect quantum computer would output. Runs on a CPU. For small problems (~30 qubits) it's faster than the QPU. The point of comparing hardware to noiseless simulator: if they agree, the QPU is not doing anything the simulator can't.

### CTQW — Continuous-Time Quantum Walk
A walker that moves through a graph using quantum amplitudes instead of classical probabilities. Classical random walks spread *diffusively* (like ink in water); quantum walks spread *ballistically* (like a wave) and can interfere with themselves — sometimes localizing instead of spreading.

### JSD — Jensen-Shannon divergence
A symmetric measure of how different two probability distributions are. JSD = 0 means identical. JSD = log(2) ≈ 0.69 means maximally different. We use thresholds:
- JSD < 0.10: distributions are essentially the same (noise-blur level)
- JSD > 0.30: distributions are meaningfully different

### Shot count
We run each quantum circuit many times (4096 "shots") because measurement is probabilistic. Each shot gives one bitstring. The histogram of 4096 bitstrings is our estimate of the true probability distribution.

### Why this matters
Every experiment below asks a yes/no question about whether your code is actually getting any benefit from real quantum hardware vs. just a classical simulator. The answers tell you what to build next.

---

# EXP A — Quantum-advantage shootout

## What it asks (plain language)

If I run the brain's QAOA optimization on real quantum hardware vs. a perfect classical simulator of that same circuit, **do they produce different answers?** If yes, hardware is doing something the simulator can't — that's the empirical basis for calling the brain "quantum." If no, the QPU is just a slower, noisier classical sampler.

This is the **gatekeeper** experiment. Every other quantum claim in the project rests on its answer.

## The setup

One circuit: QAOA with `p=2`, `γ=0.6`, `β=0.3` on the 6-concept baby-excluded J matrix from GUBIT 10. The exact same circuit runs in two places:
1. **Noiseless `AerSimulator`** locally on the CPU (no QPU cost)
2. **`ibm_kingston` hardware** (1 circuit submitted in the main job)

Both produce a histogram over the 64 possible 6-bit measurement outcomes. We compare them.

## What we measure

- **JSD(hardware, noiseless)** — overall distribution similarity
- **Top-3 overlap** — do the same 3 bitstrings dominate both?
- **Top-5 overlap** — same question, looser

## Decision rule (pre-registered before the run)

| Condition | Verdict | Action |
|---|---|---|
| top-3 overlap ≥ 2/3 | **H0** — same attractors, just blurrier on hardware | Drop quantum-architecture claims |
| top-3 overlap ≤ 1/3 AND JSD > 0.30 | **H1** — different attractors found | Quantum effect is real, keep building |
| Anything else | INCONCLUSIVE | Re-run at 8192 shots |

## Result

| Rank | Noiseless (CPU sim) | Hardware (ibm_kingston) |
|---|---|---|
| #1 | `\|000000⟩` (817) | `\|000000⟩` (726) |
| #2 | `\|111111⟩` (809) | `\|000111⟩` (708) |
| #3 | `\|000111⟩` (805) | `\|111111⟩` (687) |
| #4 | `\|111000⟩` (752) | `\|111000⟩` (615) |
| #5 | `\|100000⟩` (100) | `\|011000⟩` (116) |

`|000111⟩` = transport cluster active. `|111000⟩` = music cluster active. `|111111⟩` = all 6 on. `|000000⟩` = nothing on.

```
JSD(hardware, noiseless)   = 0.0175
top-3 overlap              = 100%
top-5 overlap              = 80%
```

## What this means

Hardware found **exactly the same top-3 attractors** as the noiseless simulator. The 4th and 5th positions reshuffle. JSD is barely above zero — that 0.018 is from measurement noise, not from finding different optima.

**Verdict: H0.** The QPU is reproducing what a CPU simulator already produces. There is no observable quantum-interference effect adding new attractors here.

The implication is uncomfortable but clean: for this QAOA on this J matrix, the brain's "quantum walk" framing is not validated. The good QAOA tuning findings from GUBIT 10 (γ=0.6/β=0.3, baby exclusion, p=2 sweet spot) remain useful — they just aren't *quantum* findings. A classical simulator would have produced the same tuning.

---

# EXP B — Quantum walk vs classical random walk

## What it asks

A classical random walker on a graph (Markov chain) is the standard way to model "what concepts do I activate after starting from X?" — and the SPECTRA TopicParticleFilter we shipped is built on that math. A quantum walker on the same graph uses amplitude interference instead of probability addition, so it can spread differently. **Does the quantum walk produce a different distribution than the classical one?**

## The setup

4-qubit one-hot encoding on a 4-node KG subgraph:
```
KG_CONCEPTS_4 = ["Mozart", "Beethoven", "Newton", "Einstein"]
KG_ADJ_4 = {
    ("Mozart",    "Beethoven"): 0.90,    # strong: composers
    ("Newton",    "Einstein"):  0.90,    # strong: physicists
    ("Beethoven", "Newton"):    0.20,    # weak bridge
    ("Mozart",    "Einstein"):  0.15,    # weak bridge
}
```

**Quantum walk circuit:** start in `|0001⟩` (walker localized on Mozart, qubit 0 = 1, all others 0). Apply Trotterized `exp(-iHt)` where H is the hopping Hamiltonian. The hopping operator `(X_i X_j + Y_i Y_j)/2` preserves the one-hot subspace: a walker at node i hops to node j without creating extra walkers.

Three evolution times: `t ∈ {0.5, 1.0, 2.0}` → 3 circuits.

**Classical comparison:** a pure Markov walk on the same graph for matched step counts.

## What we measure

For each evolution time `t`:
- **Hardware distribution** over the 4 one-hot states
- **Classical RW distribution** at matched step count
- **JSD(hw_onehot, classical)** — only on the one-hot subspace
- **Leakage** = how much hardware probability falls OUTSIDE the one-hot subspace (= hardware noise turning the walker into a "two-walker" or "no-walker" error state)

## Decision rule

| Condition | Verdict |
|---|---|
| avg leakage > 0.5 | Circuit too deep — results unreliable |
| max JSD < 0.10 | H0 — walks indistinguishable |
| max JSD > 0.30 | H1 — walks diverge |

## Result

| t | Classical peak | Hardware peak | One-hot JSD | Leakage |
|---|---|---|---|---|
| 0.5 | Beethoven (86%) | Beethoven (54%), Mozart (35%) | 0.187 | 7.9% |
| 1.0 | Mozart (72%) | **Beethoven (73%)** ← walker moved | **0.461** | 7.4% |
| 2.0 | Mozart (59%) | Mozart (50%), Newton (24%) | 0.167 | 7.2% |

Average leakage: 7.5% (clean).

## What this means

H1 — walks diverge. At `t=1.0`, the classical walker has mostly stayed at Mozart (72%) but the quantum walker has moved to Beethoven (73%). This is the **ballistic vs. diffusive** difference: the quantum walk reaches the strongly-coupled neighbor faster than diffusion allows.

**But read this carefully:** CTQW and classical RW are *known* to differ by physics. They have different generators. This experiment confirms hardware implements CTQW correctly (with 7.5% noise leakage), but it does NOT prove that the SPECTRA TopicParticleFilter is missing structure. The filter is meant to mimic the *brain's* association dynamics, not a classical RW on a chip graph.

To answer "is my filter suboptimal?" you'd need to compare the QPU walk against the **TopicParticleFilter directly**, not against a generic Markov walk. That's the experiment to run next.

What this confirms positively: hardware can implement CTQW dynamics on small graphs at reasonable depth. The infrastructure works.

---

# EXP C — Emergent bridge-node discovery

## What it asks

In GUBIT 10 you discovered "baby" is a bridge concept (connected to both transport and music clusters) by human inspection. **Could the hardware have discovered baby is a bridge without you telling it?** If yes, the method generalizes to fresh brains where you don't know the structure yet.

## The setup

7 biased QAOAs, one per concept (bus, car, passenger, music, song, concert, baby — full 7-concept set including baby). Each circuit applies a strong `Rz(-2γh)` bias on its seed concept, then runs the J Hamiltonian.

For each seed run, we look at the top-10 most-likely output states. A state is **"mixed cluster"** if it contains at least one transport concept AND at least one music concept. We compute a **bridging score** per concept:

> bridging_score(c) = average probability that c appears in mixed-cluster states across all 7 seed runs (excluding the run where c is itself the seed)

A pure cluster member (like "bus") shouldn't score very high — it only appears in mixed states when something else is dragging it across the cluster boundary. A bridge (like "baby") should score high — by definition it sits between clusters.

## Decision rule

| Condition | Verdict |
|---|---|
| Baby ranks #1 or #2 by bridging score | Method validated |
| Baby ranks ≥ #3 | Method failed — drop EXP C |
| A non-cluster concept other than baby also scores high | **Discovered new bridge** → add to KG as bridge |

## Result

| Rank | Concept | Bridging score | Cluster |
|---|---|---|---|
| 1 | concert | 0.2400 | music |
| **2** | **baby** | **0.2393** | **bridge (known)** |
| 3 | bus | 0.2391 | transport |
| 4 | passenger | 0.2389 | transport |
| 5 | car | 0.2371 | transport |
| 6 | song | 0.2342 | music |
| 7 | music | 0.2309 | music |

## What this means

Baby ranked #2 — the pre-registered validation criterion passed. But **every concept's bridging score is between 0.231 and 0.240** — a total spread of 0.009. At 4096 shots the per-concept statistical noise floor is ≈ 0.015. **The differences between concepts are smaller than shot noise.**

So the method didn't *fail* — it placed baby high — but it didn't really *succeed* either. With these noise levels, "baby is #2" is a coin flip, and "concert is #1" is even more so.

**Read:** the bridge-discovery method works mechanically. To actually discover bridges in a new brain, you need either (a) more shots per circuit (8192 or 16384) to reduce noise, (b) a sharper bias `bias_h` to amplify the structural signal, or (c) a backend with better connectivity to keep the post-transpile depth lower. The current run is at post-transpile depths 170–207 which is near the noise ceiling for Heron.

---

# EXP D — Per-circuit noise budget

## What it asks

How deep can a QAOA circuit be before noise eats the answer? GUBIT 10 reported "p=2 is the sweet spot, p=3 was worse." **Was that true because p=3 specifically was too deep, or is there still depth headroom we're not using?**

This experiment measures the *circuit family's* decoherence directly — not a single-qubit T2.

## The setup

4 copies of the same QAOA circuit, but with **deliberate idle padding** between layers using `qc.delay(pad_ns, q, unit="ns")` on every qubit. Padding values: 0, 200, 400, 800 nanoseconds.

The idle time forces the qubits to sit decohering between layers. We watch how much QAOA signal survives as padding grows.

**Signal definition:** the fraction of probability mass that lands on the top-3 most-likely states from the zero-padding run. This is a tight measure — it tracks how much the QAOA's actual peaks decohere, not just whether *anything* lands in cluster-shaped states.

We then fit:
> signal(pad) = A · exp(-pad / λ_circuit)

`λ_circuit` is the e-fold time of signal degradation. Bigger λ = more depth headroom.

## Decision rule

| Condition | Verdict |
|---|---|
| λ_circuit < 200 ns | p=2 is the permanent ceiling; never try deeper |
| λ_circuit 200–400 ns | Marginal; stay at p=2 |
| λ_circuit > 400 ns | Depth headroom exists; try p=3 with halved Trotter angles |

## Result

| Padding (ns) | Signal (mass on ref top-3) |
|---|---|
| 0   | 0.5137 |
| 200 | 0.4875 |
| 400 | 0.4771 |
| 800 | 0.4583 |

```
Fit:   signal = A · exp(-pad / λ)
λ_circuit = 7,408 ns
R²        = 0.939
```

## What this means

Signal only dropped from 51% to 46% across 800 ns of inserted padding — about a 10% relative loss. The exponential fit extrapolates to λ ≈ 7.4 microseconds.

**Caveat:** the absolute change is small (10% over 800 ns). The fit is mathematically real (R² = 0.94) but the dynamic range is limited. Read λ = 7,408 ns as "at least several microseconds" rather than a precise number. The practical takeaway is unambiguous: **there's room to go deeper.**

GUBIT 10's "p=2 ceiling" was likely a transient — maybe a one-off bad calibration day on `ibm_marrakesh`. On `ibm_kingston`, trying QAOA `p=3` with halved Trotter angles per layer (to keep total γ similar) should be safe.

---

# EXP E — Multi-seed interference

## What it asks

If I seed the QAOA with TWO competing concepts simultaneously (music AND bus), what happens? A classical model just sums their contributions — you get a soup with both clusters partially active. A quantum model can produce **interference**: at some relative phases, the "both active" state should be suppressed by destructive interference, leaving one cluster dominant.

If the QPU shows this kind of phase-dependent cluster suppression, it's doing something a classical filter can't fake. That would justify a new code path in `respond()` for multi-topic prompts.

## The setup

Biased QAOA with TWO seeds: music (qubit index where music is) and bus. Both get `Rz(-2γh)` biases each layer. Additionally, a controlled-phase gate `qc.cp(φ, i_music, i_bus)` adds a phase `φ` precisely to the `|music=1, bus=1⟩` component (the "both active" amplitude).

Three circuits, one per phase: `φ ∈ {0, π/2, π}`.

We classify each output bitstring by which cluster(s) it activates:
- **both:** active concepts span transport AND music
- **transport_only:** all active concepts in transport cluster
- **music_only:** all active concepts in music cluster
- **empty / noise:** other

Then we compute `ratio = P(both) / (P(transport_only) + P(music_only))`. Phase-dependent variation in this ratio = real interference.

## Decision rule

| Condition | Verdict |
|---|---|
| At any phase, ratio < 0.5 (single-cluster dominates) | **H_separation** — interference suppresses "both" |
| max_ratio - min_ratio > 0.5 across phases | Phase-dependent (tunable) |
| All phases have ratio > 1 | **H_compromise** — both clusters always coexist |

## Result

| Phase φ | P(both) | P(transport only) | P(music only) | Ratio |
|---|---|---|---|---|
| 0    | **78.5%** | 10.4% |  9.8% | 3.89 |
| π/2  | 56.5%    | 20.3% | 21.7% | 1.35 |
| π    | 52.6%    | 20.9% | 24.3% | 1.17 |

## What this means

**H_compromise** — at every phase, the "both clusters active" state outnumbers the single-cluster states. The decision rule for separation (ratio < 0.5) never triggered.

**However:** the ratio varied 3.3× across phases (1.17 → 3.89). That IS a real phase-dependent interference effect. It's just not strong enough to flip the dominant state from "both" to "one."

Combined with EXP A's H0, this paints a consistent picture: the QAOA on this J matrix behaves *qualitatively* classically. There's interference (the phase dependence proves it) but it's not strong enough to give the brain a new capability classical addition can't fake.

**Action:** no "multi-seed mode" in `respond()`. When a user prompts with two competing topics, just sum the classical evidence — that reproduces what the QPU does here.

---

# EXP F — Real-query reproduction (the END-TO-END test)

## What it asks

Forget the toy concept sets. Take **real user queries** from the brain's chat history. For each query, classically compute `brain.think(q)` and capture its top-5 associations. Then run the QPU's QAOA on the same J-subgraph and ask: **does the QPU concentrate on the same concepts that the classical think() concentrates on?**

This is the only test in the suite that uses real production data end-to-end. If the QPU agrees with `think()`, it could potentially substitute for `think()` in confidence-critical paths. If it disagrees, either `think()` is wrong or the J-extraction is broken — kill switch.

## The setup

10 queries. For each:
1. Take the query's seed concept + the top-4 associations from `think(seed)`
2. Build a 5-concept set
3. Extract the J subgraph (couplings between those 5)
4. Run biased QAOA (bias on seed, p=2, γ=0.6, β=0.3)
5. Look at the QPU's top-3 output states
6. **Match score:** fraction of QPU top-3 states whose active concepts overlap by ≥3 with the classical top-5

`recent_queries.json` was not present, so we used 10 **synthetic** queries built from GUBIT 10's concept space.

## Decision rule

| Avg match rate | Verdict |
|---|---|
| > 70% | QPU reproduces think() — can substitute on high-uncertainty queries |
| 40–70% | Partial agreement — use as ensemble vote |
| < 40% | Disagrees with think() — **kill switch**, do not deploy |

## Result

| # | Seed | QPU top-3 (bitstring → concepts) | Match rate |
|---|---|---|---|
| 1 | music     | `00001` (seed only), `00010`, `11111` | 33% |
| 2 | bus       | `00001` (seed only), `00111`, `11111` | 33% |
| 3 | song      | `00001` (seed only), `11101`, `11111` | 67% |
| 4 | concert   | `00001` (seed only), `11001`, `11111` | 33% |
| 5 | car       | `00001` (seed only), `11001`, `00111` |  0% |
| 6 | passenger | `00001` (seed only), `00111`, `11011` | 33% |
| 7 | music     | `00001` (seed only), `11111`, `00000` | 33% |
| 8 | bus       | `00001` (seed only), `00111`, `11111` | 33% |
| 9 | song      | `00001` (seed only), `11101`, `11111` | 67% |
| 10| concert   | `00001` (seed only), `11001`, `11111` | 33% |

**Average match rate: 36.67%** — below threshold. Kill switch.

## What this means — but read the caveat

Two things make the kill-switch verdict **unreliable**:

1. **Every query's #1 state is `00001` (just the seed concept active, nothing else).** That means `bias_h = 1.5` is over-pinning the seed qubit. The QAOA isn't doing "spread activation" because the bias is overwhelming the J couplings. This is a calibration bug in our setup, not a brain bug.

2. **Queries are synthetic.** They're recycled GUBIT 10 concepts in arbitrary orders, not actual `brain.think()` output. Even a perfect QPU couldn't reproduce something that wasn't real to begin with.

**Conclusion:** do not act on this verdict. Re-run with `bias_h = 0.5` and 10 real queries from `~/.qbit-brain/context/<identity>.json`. Cost: ~20s of remaining QPU budget.

---

# Cross-experiment narrative

Putting all six results together:

- **EXP A** says the QAOA isn't doing anything quantum-distinct from a classical simulator.
- **EXP B** confirms the hardware can implement CTQW dynamics (different from classical RW), but the difference is physics-of-CTQW, not a brain-architecture finding.
- **EXP C** mechanically validates the bridge-discovery method but the signal is below shot noise.
- **EXP D** says there's still depth headroom; QAOA p=3 might work.
- **EXP E** shows interference exists (phase dependence is real) but it's too weak to give a qualitatively new code path.
- **EXP F** kill-switched but the result is contaminated by a calibration bug — re-run before believing it.

**The honest reading:** the brain's QAOA layer is good classical optimization run via a quantum circuit. It's not "quantum cognition." The framing should stop pretending otherwise. The science is fine; the marketing was wrong.

What remains valuable across the project:
- The QAOA tuning findings (γ, β, baby exclusion) — keep, just as classical tuning
- The CTQW dynamics — useful as a graph-spreading operator, classically simulable
- The perception engine, conversation pipeline, topic particle filter, consciousness stream — never depended on quantum
- The QRNG seed — actually useful (real quantum randomness)

---

# Proposed next experiments (~9 minutes of monthly budget remaining)

## NEXT-1 — EXP F re-run (cheapest, closes the loose end)

What changes: lower `bias_h` from 1.5 to 0.5, populate `recent_queries.json` with 10 real chat-log queries (or accept the synthetic fallback and just see what happens with corrected bias). Same 10 circuits. ~20s QPU.

**Question:** does the QPU agree with `brain.think()` on real queries when the bias isn't overwhelming the J couplings?

## NEXT-2 — Mermin-GHZ paradox at scaling sizes (pure physics, ~30s QPU)

This has **nothing to do with the brain**. It's the canonical "is this thing really quantum?" demonstration, decoupled from any application.

### What it asks
The Mermin polynomial M is a Bell-style inequality extended to N-qubit GHZ states. The classical bound is `2^((N-1)/2)`. The quantum bound is `2^(N-1)`. So at:
- N=3: classical ≤ 2, quantum can reach 4 (gap of 2×)
- N=5: classical ≤ 4, quantum can reach 16 (gap of 4×)
- N=7: classical ≤ 8, quantum can reach 64 (gap of 8×)

The quantum-classical gap grows **exponentially** with N. CHSH only shows a 1.41× gap. Mermin at large N is more dramatic.

### The setup
Prepare an N-qubit GHZ state `(|00...0⟩ + |11...1⟩)/√2`:
```
qc.h(0)
for i in range(1, N):
    qc.cx(0, i)
```
Then run the 4 Mermin measurement settings (these are specific (X, Y) measurement combinations across qubits; the math is in Mermin's 1990 paper). Compute the Mermin polynomial from the 4 expectation values.

Three sizes: N = 3, 5, 7. Per size: GHZ prep + 4 measurement bases × 8192 shots. Total: **12 circuits, ~30s QPU**.

### Pre-registered prediction
- N=3: M ≈ 3.5 (clean violation of classical bound 2)
- N=5: M ≈ 7–10 (clean violation of bound 4)
- N=7: M ≈ 8–20 (uncertain — depends on whether the 6-CNOT GHZ ladder survives noise)

### What you keep
A real Mermin-violation curve as a function of N on `ibm_kingston`. Direct measurement of "how big can I make a coherent multi-qubit superposition before noise eats it." Compare against GUBIT 10's 3-qubit GHZ fidelity numbers (97.7% / 96.6%).

This is the most genuinely *quantum* thing you can demonstrate in 30 seconds. No brain framing required.

## NEXT-3 — Chip-distance Bell test (engineering data, ~30s QPU)

This is more practical and produces a dataset.

### What it asks
On `ibm_kingston`'s 156-qubit topology, qubit pairs vary in physical distance. Adjacent qubits Bell-violate cleanly; far pairs require the transpiler to insert many SWAP gates. **At what chip-distance does Bell violation collapse?**

### The setup
Choose 6 qubit pairs at increasing topological distances on the heavy-hex lattice: `d ∈ {1, 2, 4, 8, 16, 32}` hops apart. For each pair:
1. Prepare `(|00⟩ + |11⟩)/√2` Bell state (Hadamard + CNOT) on those two qubits
2. Run the 4 CHSH measurement settings (each at a specific rotation angle pair)
3. Compute S = |E(a,b) - E(a,b') + E(a',b) + E(a',b')|

6 distances × 4 settings × 4096 shots = **24 circuits, ~30s QPU**.

### Pre-registered prediction
S degrades roughly linearly with the number of SWAP gates inserted (each SWAP ≈ 3 CNOTs, each CNOT has error ~1e-2). Expect:
- d=1: S ≈ 2.7
- d=2: S ≈ 2.5
- d=4: S ≈ 2.3
- d=8: S ≈ 2.1
- d=16: S ≈ 1.9 (below classical threshold)
- d=32: S ≈ 1.5 (no violation)

### What you keep
A hardware-specific **"Bell decay length"** — the maximum chip-distance at which entanglement still violates classical bounds on `ibm_kingston`. This is a number IBM doesn't publish. It tells you the practical maximum scale for any algorithm that wants to entangle distant qubits on this device.

Could be a small open dataset / blog post — generally useful beyond your project.

## My pick

If you have time for one: **NEXT-1** (EXP F re-run) — cheapest, closes the loose end in this report.

If you have time for two: **NEXT-1 + NEXT-2** — closes the loose end AND gives you a beautiful piece of physics on the side.

If you have time for all three: total budget ≈ 80s, well within the ~9 minutes you have left this month.

---

# Files produced

| File | Purpose |
|---|---|
| `gubit12_results.json` | Per-experiment verdicts, metrics, decision rules |
| `gubit12_raw_counts.json` | Raw shot counts for re-analysis without re-submitting |
| `gubit12_summary.md` | Auto-generated short summary |
| `gubit12_stdout.log` | Full console transcript of the run |
| `gubit12_quantum_suite.py` | The script that ran all 6 experiments |

---

# Reflection

GUBIT 10 was an honest QAOA hyperparameter search dressed in quantum framing. GUBIT 11 retrofitted unrelated quantum measurements into biological constants. GUBIT 12 was the first run that **asked falsifiable questions and got falsifiable answers** — including answers the project doesn't want.

The right response to EXP A's H0 is not to look for a different quantum claim. It's to update priors: the QAOA-on-J-matrix workflow is **classically simulable for problems of this size**, and the project's value is in everything else — the perception engine, the conversation pipeline, the persistent topic filter, the consciousness stream. Those don't depend on quantum hardware to be interesting.

The remaining budget (~9 minutes / month) is best spent on:
1. Closing EXP F's loose end (real queries + lower bias)
2. Mermin-GHZ scaling (a beautiful, unrelated, scientifically clean experiment)
3. Maybe chip-distance Bell mapping if there's time

After that, decisions about the project's quantum framing are based on actual data instead of marketing.
