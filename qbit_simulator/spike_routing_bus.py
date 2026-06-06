"""Spike routing bus -- Address Event Representation (AER).

Instead of broadcasting full dense activation vectors between brain
regions, only transmit EVENTS:

    SpikeEvent(neuron_id, timestamp, value)

This is how neuromorphic hardware works:
  Intel Loihi, IBM TrueNorth, SpiNNaker, BrainScaleS.

The key advantages over dense transmission:
  * Traffic = O(active neurons), not O(all neurons)
    -> On sparse cortical activations (~5% active) = 20x less data
  * Exact spike timing is preserved (temporal coding is free)
  * Natural asynchronous, event-driven computation
  * Routing table: each neuron sends to specific targets, not broadcast

Signal model:
  Dense activation vector x (N floats, threshold-based firing)
    -> AEREncoder: emit events for neurons where |x[i]| > threshold
    -> SpikeRoutingBus: route events to registered target regions
    -> AERDecoder: reconstruct dense activation at receiver

Classes
-------
  SpikeEvent        -- single event: (neuron_id, time, value)
  RoutingTable      -- maps source neuron IDs to target region sets
  SpikeRoutingBus   -- collects, routes and delivers events
  AEREncoder        -- dense activation -> spike events
  AERDecoder        -- spike events -> dense activation
  AERRelay          -- full encode -> bus -> decode pipeline
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict
from typing import NamedTuple

import numpy as np


# ---------------------------------------------------------------------------
# SpikeEvent
# ---------------------------------------------------------------------------

class SpikeEvent(NamedTuple):
    """A single address event."""
    neuron_id: int     # source neuron index
    timestamp: float   # time of emission (ms or arbitrary units)
    value:     float   # spike amplitude / rate (1.0 for binary spikes)


# ---------------------------------------------------------------------------
# RoutingTable
# ---------------------------------------------------------------------------

class RoutingTable:
    """Maps source neuron IDs to target region names.

    Example:
        rt = RoutingTable()
        rt.connect(src_ids=[0, 1, 2], target="V1")
        rt.connect(src_ids=[2, 3],    target="PFC")
        # neuron 2 projects to both V1 and PFC (divergent routing)
    """

    def __init__(self) -> None:
        self._table: dict[int, set[str]] = defaultdict(set)

    def connect(self, src_ids: list[int], target: str) -> None:
        for nid in src_ids:
            self._table[nid].add(target)

    def targets_for(self, neuron_id: int) -> set[str]:
        return self._table.get(neuron_id, set())

    def all_targets(self) -> set[str]:
        targets: set[str] = set()
        for t in self._table.values():
            targets |= t
        return targets

    def connect_all_to_all(self, n_neurons: int, target: str) -> None:
        """Connect every neuron to one target (broadcast)."""
        for i in range(n_neurons):
            self._table[i].add(target)

    def connect_topographic(self, n_neurons: int,
                             targets: list[str]) -> None:
        """Split neurons evenly across targets (topographic map)."""
        chunk = max(1, n_neurons // len(targets))
        for i, tgt in enumerate(targets):
            lo = i * chunk
            hi = lo + chunk if i < len(targets) - 1 else n_neurons
            for nid in range(lo, hi):
                self._table[nid].add(tgt)


# ---------------------------------------------------------------------------
# SpikeRoutingBus
# ---------------------------------------------------------------------------

class SpikeRoutingBus:
    """Event bus that collects, routes and delivers spike events.

    Usage
    -----
        bus = SpikeRoutingBus()
        bus.register_region("V1",  n_neurons=64)
        bus.register_region("PFC", n_neurons=32)
        bus.routing.connect_all_to_all(64, "PFC")

        bus.emit_events(events_from_retina)
        v1_events  = bus.collect("V1",  time_window=(0, 10))
        pfc_events = bus.collect("PFC", time_window=(0, 10))
    """

    def __init__(self) -> None:
        self.routing: RoutingTable        = RoutingTable()
        self._queues: dict[str, list[SpikeEvent]] = defaultdict(list)
        self._regions: dict[str, int]     = {}   # name -> n_neurons
        self._stats: dict[str, int]       = defaultdict(int)

    def register_region(self, name: str, n_neurons: int) -> None:
        self._regions[name] = n_neurons
        if name not in self._queues:
            self._queues[name] = []

    def emit_event(self, event: SpikeEvent) -> None:
        """Route a single event to all registered targets."""
        targets = self.routing.targets_for(event.neuron_id)
        for t in targets:
            self._queues[t].append(event)
            self._stats["total_routed"] += 1
        self._stats["total_emitted"] += 1

    def emit_events(self, events: list[SpikeEvent]) -> None:
        for e in events:
            self.emit_event(e)

    def collect(self, region: str,
                time_window: tuple[float, float] | None = None
                ) -> list[SpikeEvent]:
        """Collect (and remove) events for a region, optionally within a time window."""
        all_ev = self._queues.get(region, [])
        if time_window is None:
            result = list(all_ev)
            self._queues[region] = []
        else:
            t0, t1 = time_window
            result   = [e for e in all_ev if t0 <= e.timestamp <= t1]
            self._queues[region] = [e for e in all_ev
                                     if not (t0 <= e.timestamp <= t1)]
        return result

    def peek(self, region: str) -> list[SpikeEvent]:
        """Return events without removing them."""
        return list(self._queues.get(region, []))

    def clear(self, region: str = None) -> None:
        if region:
            self._queues[region] = []
        else:
            for k in self._queues:
                self._queues[k] = []

    def queue_size(self, region: str) -> int:
        return len(self._queues.get(region, []))

    def stats(self) -> dict:
        return dict(self._stats)


# ---------------------------------------------------------------------------
# AEREncoder  (dense activation -> spike events)
# ---------------------------------------------------------------------------

@dataclass
class AEREncoder:
    """Convert a dense activation vector to a list of SpikeEvents.

    Encoding strategy:
      THRESHOLD  -- emit an event for every neuron with |x[i]| > threshold
                    value = x[i].  Sparse for typical cortical activations.
      TOPK       -- emit exactly k events for the top-k most active neurons.
      RATE       -- treat x[i] in [0,1] as firing rate; emit events with
                    probability x[i].  Models Poisson spiking.
    """
    n_neurons:  int
    mode:       str   = "threshold"   # 'threshold' | 'topk' | 'rate'
    threshold:  float = 0.1
    k:          int   = 10            # used for topk mode
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0), repr=False)

    def encode(self, x: np.ndarray, timestamp: float = 0.0
               ) -> list[SpikeEvent]:
        """Convert activation x (n_neurons,) to a list of SpikeEvents."""
        x = np.asarray(x, dtype=np.float64).ravel()
        events: list[SpikeEvent] = []

        if self.mode == "threshold":
            for i, v in enumerate(x):
                if abs(v) > self.threshold:
                    events.append(SpikeEvent(i, timestamp, float(v)))

        elif self.mode == "topk":
            idx = np.argsort(np.abs(x))[::-1][:self.k]
            for i in idx:
                if x[i] != 0:
                    events.append(SpikeEvent(int(i), timestamp, float(x[i])))

        elif self.mode == "rate":
            probs = np.clip(np.abs(x), 0, 1)
            fired = self.rng.random(len(x)) < probs
            for i in np.where(fired)[0]:
                events.append(SpikeEvent(int(i), timestamp, float(x[i])))

        else:
            raise ValueError(f"unknown mode: {self.mode}")

        return events

    def sparsity(self, x: np.ndarray) -> float:
        """Fraction of neurons that would NOT fire."""
        n_events = len(self.encode(x))
        return 1.0 - n_events / max(self.n_neurons, 1)

    def compression_ratio(self, x: np.ndarray) -> float:
        """Bits saved: (dense float vector) / (event list)."""
        n_ev    = len(self.encode(x))
        dense   = self.n_neurons * 32          # 32-bit floats
        # Each event = (neuron_id int + timestamp float + value float) ~ 3 * 32 bits
        sparse  = n_ev * 3 * 32
        return dense / max(sparse, 1)


# ---------------------------------------------------------------------------
# AERDecoder  (spike events -> dense activation)
# ---------------------------------------------------------------------------

@dataclass
class AERDecoder:
    """Reconstruct a dense activation vector from a list of SpikeEvents.

    Modes:
      LATEST   -- use the most recent event value for each neuron
      SUM      -- sum all event values for each neuron in the time window
      MEAN     -- average event values
    """
    n_neurons: int
    mode:      str   = "latest"
    fill:      float = 0.0    # value for neurons with no events

    def decode(self, events: list[SpikeEvent]) -> np.ndarray:
        x = np.full(self.n_neurons, self.fill, dtype=np.float64)
        if not events:
            return x
        if self.mode == "latest":
            # Keep most recent event per neuron.
            seen: dict[int, SpikeEvent] = {}
            for e in events:
                if e.neuron_id not in seen or e.timestamp > seen[e.neuron_id].timestamp:
                    seen[e.neuron_id] = e
            for nid, e in seen.items():
                if 0 <= nid < self.n_neurons:
                    x[nid] = e.value
        elif self.mode == "sum":
            for e in events:
                if 0 <= e.neuron_id < self.n_neurons:
                    x[e.neuron_id] += e.value
        elif self.mode == "mean":
            counts = np.zeros(self.n_neurons, dtype=int)
            for e in events:
                if 0 <= e.neuron_id < self.n_neurons:
                    x[e.neuron_id] += e.value
                    counts[e.neuron_id] += 1
            mask = counts > 0
            x[mask] /= counts[mask]
        else:
            raise ValueError(f"unknown mode: {self.mode}")
        return x


# ---------------------------------------------------------------------------
# AERRelay  (full encode -> bus -> decode pipeline)
# ---------------------------------------------------------------------------

@dataclass
class AERRelay:
    """Full encode -> route -> decode pipeline.

    Parameters
    ----------
    n_source  : number of neurons in source region
    n_target  : number of neurons in target region
    enc_mode  : AEREncoder mode ('threshold' | 'topk' | 'rate')
    dec_mode  : AERDecoder mode ('latest' | 'sum' | 'mean')
    threshold : encoder threshold (for 'threshold' mode)
    k         : top-k count (for 'topk' mode)
    """
    n_source:  int
    n_target:  int
    enc_mode:  str   = "threshold"
    dec_mode:  str   = "latest"
    threshold: float = 0.1
    k:         int   = 10
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0), repr=False)

    encoder: AEREncoder     = field(default=None, repr=False)
    bus:     SpikeRoutingBus = field(default=None, repr=False)
    decoder: AERDecoder     = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.encoder = AEREncoder(self.n_source, mode=self.enc_mode,
                                   threshold=self.threshold, k=self.k,
                                   rng=self.rng)
        self.bus     = SpikeRoutingBus()
        self.bus.register_region("source", self.n_source)
        self.bus.register_region("target", self.n_target)
        self.bus.routing.connect_all_to_all(self.n_source, "target")
        self.decoder = AERDecoder(self.n_target, mode=self.dec_mode)

    def transmit(self, x: np.ndarray,
                 timestamp: float = 0.0) -> tuple[np.ndarray, dict]:
        """Encode x, route through bus, decode at target.

        Returns
        -------
        x_rec  : reconstructed activation (n_target,)
        stats  : dict with n_events, sparsity, compression_ratio,
                 reconstruction_error
        """
        x      = np.asarray(x, dtype=np.float64).ravel()
        events = self.encoder.encode(x, timestamp)
        self.bus.emit_events(events)
        recv   = self.bus.collect("target")
        # Trim to target size (in case source > target, use latest).
        x_rec  = self.decoder.decode(recv)[:self.n_target]
        if len(x_rec) < self.n_target:
            x_rec = np.pad(x_rec, (0, self.n_target - len(x_rec)))

        n_ev   = len(events)
        sparse = self.encoder.sparsity(x)
        cr     = self.encoder.compression_ratio(x)
        rec_err = float(
            np.linalg.norm(x[:self.n_target] - x_rec) /
            (np.linalg.norm(x[:self.n_target]) + 1e-12))

        stats = {
            "n_events":            n_ev,
            "sparsity":            sparse,
            "compression_ratio":   cr,
            "reconstruction_error": rec_err,
            "bus_stats":           self.bus.stats(),
        }
        return x_rec, stats
