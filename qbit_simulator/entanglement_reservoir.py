"""Entanglement reservoir -- pre-shared standing quantum channel.

Instead of establishing entanglement on demand (slow), regions
pre-distribute entangled qudit pairs during idle time and store them
unmeasured.  When communication is needed, pairs are *spent* -- used
by one of the three Cat-4 protocols below.

                    IDLE (sleep / consolidation)
                    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   Region A  <----  refresh(k pairs of dim d)  ---->  Region B
             [pair 0]  [pair 1]  ...  [pair k]
              all unmeasured, coherent, waiting

                    ACTIVE (waking operation)
                    ~~~~~~~~~~~~~~~~~~~~~~~~~
   Region A needs to talk?
     spend(1 pair)  ->  SuperdenseCodingChannel   -> 2*log2(d) bits/use
     spend(1 pair)  ->  TeleportationChannel      -> exact qudit state
     spend(n pairs) ->  EntangledBroadcastChannel -> sync N regions at once

Decoherence model
-----------------
Each pair has a coherence score c in [0, 1].
  c = 1.0  ->  perfect Bell state, fully usable
  c = 0.0  ->  fully decohered, must be discarded
Per time step:  c  <-  c * exp(-dt / tau)
A pair is usable when c >= coherence_threshold (default 0.5).

High-d qudits (d=10) are worth more per pair:
  qubit  (d=2):  1 ebit   ->  2 bits via superdense coding
  d=10 qudit:    3.32 ebits  -> 6.64 bits via superdense coding
  d=100 qudit:   6.64 ebits  -> 13.28 bits via superdense coding

Protocols that spend pairs
--------------------------
  SuperdenseCodingChannel
      Alice encodes integer m in {0 .. d^2-1} by applying
      X^a Z^b to her qudit (m = a*d + b), then sends it over
      the classical channel.  Bob decodes by Bell measurement.
      Cost: 1 pair + 1 qudit transmission -> 2*log2(d) classical bits.

  TeleportationChannel
      Alice holds unknown quantum state |psi>.  Using 1 entangled pair
      + a Bell measurement, she sends 2 classical correction values
      to Bob, who applies X^m2 Z^m1 to recover |psi> exactly.
      Cost: 1 pair + 2 classical correction values.

  EntangledBroadcastChannel
      Prepares a GHZ state across n+1 qudits: one for Alice,
      one each for n receivers.  Alice's measurement instantly
      correlates all receivers to the same outcome.
      Cost: n pairs from reservoir (one per receiver).

File location
-------------
  C:\\Calculatorul F 1\\F 1\\qbit_simulator\\entanglement_reservoir.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .qudit import QuditCircuit, QuditState, H_gate, X_power, Z_gate, CSUM_gate


# ---------------------------------------------------------------------------
# EntangledPair  -- one pre-shared pair, maintained unmeasured
# ---------------------------------------------------------------------------

@dataclass
class EntangledPair:
    """A single entangled qudit pair stored in the reservoir.

    The joint state is the d-dimensional Bell state:
        |Phi> = (1/sqrt(d)) sum_{j=0}^{d-1} |j>_A |j>_B

    Parameters
    ----------
    d         : qudit dimension
    coherence : current fidelity to ideal Bell state  [0, 1]
    age       : number of time steps since creation
    """
    d:         int
    coherence: float = 1.0
    age:       int   = 0
    _circuit:  Optional[QuditCircuit] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self._circuit is None:
            self._circuit = self._make_bell_pair()

    def _make_bell_pair(self) -> QuditCircuit:
        """Prepare |Phi> = (1/sqrt(d)) sum_j |j,j>."""
        qc = QuditCircuit(n=2, d=self.d)
        qc.H(0)       # QFT superposition on Alice's qudit
        qc.CSUM(0, 1) # Entangle: sum_j |j>|0> -> sum_j |j>|j>
        return qc

    def decohere(self, dt: float, tau: float) -> None:
        """Advance time and reduce coherence."""
        self.coherence *= float(np.exp(-dt / tau))
        self.age       += 1

    def is_usable(self, threshold: float = 0.5) -> bool:
        return self.coherence >= threshold

    def get_circuit(self) -> QuditCircuit:
        """Return the joint circuit (copy -- spending does not alter reservoir)."""
        return QuditCircuit(n=2, d=self.d,
                            init_state=self._circuit.amplitudes().copy())


# ---------------------------------------------------------------------------
# EntanglementReservoir  -- pool manager
# ---------------------------------------------------------------------------

@dataclass
class EntanglementReservoir:
    """Pool of pre-shared entangled qudit pairs between two brain regions.

    Parameters
    ----------
    capacity          : maximum number of pairs to maintain
    d                 : qudit dimension (d=10 recommended)
    decoherence_tau   : coherence half-life in time steps
                        (large tau = stable pairs, long shelf-life)
    coherence_threshold: pairs below this are considered spent/degraded
    rng               : random generator
    """
    capacity:            int   = 16
    d:                   int   = 10
    decoherence_tau:     float = 100.0   # time steps
    coherence_threshold: float = 0.5
    rng: np.random.Generator   = field(
        default_factory=lambda: np.random.default_rng(0), repr=False)

    _pairs: List[EntangledPair] = field(default_factory=list, repr=False)
    _t:     int                 = field(default=0, repr=False)
    _spent_total: int           = field(default=0, repr=False)
    _refreshed_total: int       = field(default=0, repr=False)

    def __post_init__(self) -> None:
        # Start fully charged.
        self.refresh()

    # ---- charge management ----

    def refresh(self, n: Optional[int] = None) -> int:
        """Re-establish n pairs (or fill to capacity).

        Returns the number of new pairs created.
        """
        target = self.capacity if n is None else n
        created = 0
        while len(self._pairs) < self.capacity and created < target:
            self._pairs.append(EntangledPair(d=self.d))
            created += 1
        self._refreshed_total += created
        return created

    def spend(self, n: int = 1) -> List[EntangledPair]:
        """Consume n usable pairs and return them for use by a protocol.

        Raises RuntimeError if insufficient usable pairs.
        """
        usable = [p for p in self._pairs if p.is_usable(self.coherence_threshold)]
        if len(usable) < n:
            raise RuntimeError(
                f"Reservoir has only {len(usable)} usable pairs "
                f"(requested {n}).  Call refresh() first.")
        spent = usable[:n]
        for p in spent:
            self._pairs.remove(p)
        self._spent_total += n
        return spent

    def try_spend(self, n: int = 1) -> List[EntangledPair]:
        """Like spend() but returns as many as available (no error)."""
        usable = [p for p in self._pairs if p.is_usable(self.coherence_threshold)]
        actual = min(n, len(usable))
        spent  = usable[:actual]
        for p in spent:
            self._pairs.remove(p)
        self._spent_total += actual
        return spent

    # ---- time evolution ----

    def step(self, dt: float = 1.0) -> None:
        """Advance time: decohere all pairs, discard degraded ones."""
        self._t += 1
        for p in self._pairs:
            p.decohere(dt, self.decoherence_tau)
        self._pairs = [p for p in self._pairs
                       if p.is_usable(self.coherence_threshold)]

    # ---- status ----

    @property
    def n_usable(self) -> int:
        return sum(1 for p in self._pairs
                   if p.is_usable(self.coherence_threshold))

    @property
    def charge_level(self) -> float:
        """Fraction of capacity that is currently usable  [0, 1]."""
        return self.n_usable / max(self.capacity, 1)

    @property
    def mean_coherence(self) -> float:
        if not self._pairs:
            return 0.0
        return float(np.mean([p.coherence for p in self._pairs]))

    def stats(self) -> dict:
        return {
            "capacity":         self.capacity,
            "n_pairs":          len(self._pairs),
            "n_usable":         self.n_usable,
            "charge_level":     self.charge_level,
            "mean_coherence":   self.mean_coherence,
            "d":                self.d,
            "bits_available":   self.n_usable * 2 * np.log2(self.d),
            "time_steps":       self._t,
            "total_spent":      self._spent_total,
            "total_refreshed":  self._refreshed_total,
        }

    def __repr__(self) -> str:
        return (f"EntanglementReservoir(d={self.d}, "
                f"{self.n_usable}/{self.capacity} usable, "
                f"charge={self.charge_level:.1%}, "
                f"mean_coh={self.mean_coherence:.3f})")


# ---------------------------------------------------------------------------
# SuperdenseCodingChannel  -- 2*log2(d) bits per pair
# ---------------------------------------------------------------------------

class SuperdenseCodingChannel:
    """Send an integer message using one pre-shared entangled pair.

    Capacity: 2 * log2(d) classical bits per pair.
    For d=10: 6.64 bits  (vs 3.32 bits for a single unentangled qudit).

    Protocol (Weinfurter / Bennett-Wiesner):
      Alice has integer m in {0, ..., d^2 - 1}
        split: m = a*d + b   (a, b in {0..d-1})
        apply: X^a then Z^b to her qudit of the shared pair
        send:  her (now modified) qudit over the channel
      Bob receives Alice's qudit + has his own half:
        apply: CSUM†(Alice->Bob), then H† (inverse QFT) to Alice's qudit
        measure both qudits -> reads back (a, b) -> m = a*d + b
    """

    def __init__(self, reservoir: EntanglementReservoir) -> None:
        self.reservoir = reservoir
        self._messages_sent = 0

    def encode(self, message: int,
               rng: Optional[np.random.Generator] = None) -> tuple[np.ndarray, int]:
        """Alice encodes `message` using one reservoir pair.

        Returns
        -------
        alice_state : np.ndarray  -- Alice's modified qudit state vector
                      (this is what gets sent over the classical channel)
        pair_d      : int         -- qudit dimension (Bob needs this)
        """
        d = self.reservoir.d
        if not (0 <= message < d * d):
            raise ValueError(f"message must be in [0, {d*d-1}], got {message}")
        pair = self.reservoir.spend(1)[0]
        qc   = pair.get_circuit()   # joint 2-qudit state
        a, b = divmod(message, d)
        # Apply X^a then Z^b to Alice's qudit (qudit 0).
        if a > 0:
            qc.state.apply_single(X_power(d, a), 0)
        if b > 0:
            qc.state.apply_single(
                np.linalg.matrix_power(Z_gate(d), b).astype(complex), 0)
        self._messages_sent += 1
        # Return Alice's half (the modified 2-qudit joint state).
        return qc.amplitudes(), d

    def decode(self, joint_state: np.ndarray, d: int,
               rng: Optional[np.random.Generator] = None) -> int:
        """Bob decodes the message from the received joint state.

        Parameters
        ----------
        joint_state : the 2-qudit state vector (Alice's modified + Bob's half)
        d           : qudit dimension
        """
        if rng is None:
            rng = np.random.default_rng()
        qc = QuditCircuit(n=2, d=d, init_state=joint_state.copy())
        # Reverse entanglement: CSUM†(0->1) then H†(inverse QFT) on qudit 0.
        qc.CSUMdg(0, 1)
        qc.Hdg(0)
        # Qudit 0 -> b,  qudit 1 -> (d - a) % d.
        b_meas    = qc.measure(0, rng)
        neg_a_meas = qc.measure(1, rng)
        a_recovered = (d - neg_a_meas) % d
        return a_recovered * d + b_meas

    def send(self, message: int,
             rng: Optional[np.random.Generator] = None) -> int:
        """Full round-trip encode + decode (single-process simulation)."""
        state, d = self.encode(message, rng)
        return self.decode(state, d, rng)

    @property
    def bits_per_use(self) -> float:
        return 2 * np.log2(self.reservoir.d)


# ---------------------------------------------------------------------------
# TeleportationChannel  -- exact qudit state transfer
# ---------------------------------------------------------------------------

class TeleportationChannel:
    """Teleport an arbitrary qudit state from Alice to Bob.

    No information is lost -- Bob recovers the EXACT state |psi>.
    Cost: 1 entangled pair + 2 classical correction values.

    Protocol (Bennett et al. 1993, generalized to qudits):
      Alice has unknown state |psi> (n_q qudits) and her half of pair.
        apply: CSUM(|psi> -> Alice's pair qudit)
        apply: H† to |psi>
        measure: |psi> qudit -> m1, Alice's pair qudit -> m2
        send: (m1, m2) to Bob over classical channel
      Bob has his half of the pair:
        apply: X^m2 to Bob's qudit
        apply: Z^m1 to Bob's qudit
        Bob's qudit is now |psi>.
    """

    def __init__(self, reservoir: EntanglementReservoir) -> None:
        self.reservoir = reservoir
        self._teleported = 0

    def send(self, psi: np.ndarray,
             rng: Optional[np.random.Generator] = None
             ) -> tuple[int, int, int]:
        """Alice's side: measure and return classical corrections.

        Parameters
        ----------
        psi : state vector of one qudit to teleport (length d)

        Returns
        -------
        m1, m2 : classical correction values (sent to Bob)
        d      : qudit dimension
        """
        if rng is None:
            rng = np.random.default_rng()
        d    = self.reservoir.d
        psi  = np.asarray(psi, dtype=complex)
        if len(psi) != d:
            raise ValueError(f"psi must have length d={d}, got {len(psi)}")
        pair = self.reservoir.spend(1)[0]
        pair_qc = pair.get_circuit()  # 2-qudit pair state

        # Build a 3-qudit circuit: [psi | Alice_pair | Bob_pair]
        # qudit 0 = |psi>, qudit 1 = Alice's pair half, qudit 2 = Bob's half
        joint = np.kron(psi, pair_qc.amplitudes())   # d^3 amplitudes
        qc    = QuditCircuit(n=3, d=d, init_state=joint)

        # Alice's operations (d-dim generalization of Bennett et al.):
        qc.CSUMdg(0, 1)   # CSUM†: |psi> control, Alice's Bell qudit target
        qc.H(0)            # QFT (forward) on |psi> qudit

        # Alice measures her two qudits.
        m1 = qc.measure(0, rng)   # from |psi> qudit
        m2 = qc.measure(1, rng)   # from Alice's pair qudit

        self._teleported += 1
        return m1, m2, d

    def receive(self, bob_pair_state: np.ndarray,
                m1: int, m2: int, d: int,
                rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Bob's side: apply corrections to recover |psi>.

        Parameters
        ----------
        bob_pair_state : Bob's half of the entangled pair state (length d)
        m1, m2         : classical corrections from Alice
        d              : qudit dimension

        Returns the recovered state vector (length d).
        """
        # Bob's corrections: X^(d-m2) then Z^(d-m1)
        qc = QuditCircuit(n=1, d=d, init_state=bob_pair_state.copy())
        if m2 > 0:
            qc.state.apply_single(X_power(d, d - m2), 0)
        if m1 > 0:
            qc.state.apply_single(
                np.linalg.matrix_power(Z_gate(d), d - m1).astype(complex), 0)
        return qc.amplitudes()

    def teleport(self, psi: np.ndarray,
                 rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Full round-trip teleportation (single-process simulation).

        Returns Bob's recovered state vector.
        """
        if rng is None:
            rng = np.random.default_rng()
        d         = self.reservoir.d
        pair      = self.reservoir.spend(1)[0]
        pair_qc   = pair.get_circuit()

        # Extract Bob's reduced state for use after Alice's measurement.
        # For a perfect Bell pair, Bob's half is a flat superposition.
        # We simulate the full 3-qudit joint state.
        psi    = np.asarray(psi, dtype=complex)
        joint  = np.kron(psi, pair_qc.amplitudes())
        qc     = QuditCircuit(n=3, d=d, init_state=joint)
        qc.CSUMdg(0, 1)
        qc.H(0)
        m1 = qc.measure(0, rng)
        m2 = qc.measure(1, rng)
        # Bob's qudit is now qudit 2 in qc.
        # Reconstruct Bob's state (pure, up to phase) from the collapsed state.
        psi_bob = qc.amplitudes().reshape([d] * 3)[m1, m2, :]  # (d,)
        norm    = np.linalg.norm(psi_bob)
        psi_bob = psi_bob / norm if norm > 1e-12 else psi_bob
        # Apply corrections: X^(d-m2) then Z^(d-m1).
        qc_bob  = QuditCircuit(n=1, d=d, init_state=psi_bob)
        if m2 > 0:
            qc_bob.state.apply_single(X_power(d, d - m2), 0)
        if m1 > 0:
            qc_bob.state.apply_single(
                np.linalg.matrix_power(Z_gate(d), d - m1).astype(complex), 0)
        return qc_bob.amplitudes()


# ---------------------------------------------------------------------------
# EntangledBroadcastChannel  -- GHZ-based multi-region sync
# ---------------------------------------------------------------------------

class EntangledBroadcastChannel:
    """Synchronize N receiver regions using a GHZ state.

    GHZ state (d-dimensional, n+1 qudits):
        |GHZ> = (1/sqrt(d)) sum_{j=0}^{d-1} |j>^(x)(n+1)

    When Alice (qudit 0) measures and gets outcome j, ALL receivers
    collapse to |j> simultaneously -- one-shot global synchronization.

    Cost: n pairs from reservoir (one shared with each receiver).
    """

    def __init__(self, reservoir: EntanglementReservoir) -> None:
        self.reservoir  = reservoir
        self._broadcasts = 0

    def broadcast(self, n_receivers: int,
                  rng: Optional[np.random.Generator] = None
                  ) -> dict:
        """Prepare GHZ state and simulate Alice's measurement.

        All receivers collapse to the same outcome as Alice.

        Returns
        -------
        dict with:
          alice_outcome : int  -- what Alice measured
          receiver_outcomes : list[int]  -- what each receiver sees (all equal)
          ghz_fidelity : float -- mean coherence of pairs used
          n_pairs_spent : int
        """
        if rng is None:
            rng = np.random.default_rng()
        d       = self.reservoir.d
        pairs   = self.reservoir.try_spend(n_receivers)
        n_used  = len(pairs)
        if n_used == 0:
            raise RuntimeError("No usable pairs in reservoir for broadcast.")

        # Build GHZ state across n_used + 1 qudits.
        n_total = n_used + 1
        qc      = QuditCircuit(n=n_total, d=d)
        # H on qudit 0 (Alice), then CSUM(0, k) for each receiver k.
        qc.H(0)
        for k in range(1, n_total):
            qc.CSUM(0, k)

        # Modulate each qudit by its pair's coherence (phase noise model).
        for k, pair in enumerate(pairs, start=1):
            if pair.coherence < 1.0:
                phase_noise = (1.0 - pair.coherence) * np.pi * 0.5
                noise_gate  = np.diag([
                    np.exp(1j * phase_noise * np.random.randn())
                    for _ in range(d)])
                qc.state.apply_single(noise_gate.astype(complex), k)

        # Alice measures.
        alice_outcome = qc.measure(0, rng)

        # All receivers collapse to correlated outcomes.
        receiver_outcomes = []
        for k in range(1, n_total):
            receiver_outcomes.append(qc.measure(k, rng))

        mean_coh = float(np.mean([p.coherence for p in pairs]))
        self._broadcasts += 1
        return {
            "alice_outcome":     alice_outcome,
            "receiver_outcomes": receiver_outcomes,
            "n_pairs_spent":     n_used,
            "mean_coherence":    mean_coh,
            "all_agree":         all(r == alice_outcome
                                    for r in receiver_outcomes),
        }

    @property
    def bits_per_broadcast(self) -> float:
        return np.log2(self.reservoir.d)
