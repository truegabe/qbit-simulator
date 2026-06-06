"""Sharp-wave ripples and memory replay (Buzsáki).

During quiet wakefulness and NREM sleep, hippocampal CA3-CA1 emits
brief (~100 ms) high-frequency (~150-250 Hz) oscillations called
sharp-wave ripples (SWRs). Cells active during prior experience replay
in compressed temporal order — believed to drive consolidation.

Simplified model:
  - A "place-cell sequence" of length L.
  - Replay: at random intervals, generate a fast (compressed) sequence
    of activations either forward or reverse.
  - Detection: spike-burst windows with above-threshold population activity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RippleReplayer:
    sequence_length: int = 20
    replay_speed: float = 10.0     # compression factor
    p_replay: float = 0.01         # per-step probability
    p_reverse: float = 0.5
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0), repr=False)

    def maybe_replay(self) -> tuple[bool, np.ndarray]:
        if self.rng.uniform() >= self.p_replay:
            return False, np.array([])
        idx = np.arange(self.sequence_length)
        if self.rng.uniform() < self.p_reverse:
            idx = idx[::-1]
        return True, idx

    def generate(self, n_steps: int = 1000) -> np.ndarray:
        """Generate population activity matrix (n_steps, sequence_length)."""
        out = np.zeros((n_steps, self.sequence_length))
        t = 0
        while t < n_steps:
            ripple, idx = self.maybe_replay()
            if not ripple:
                t += 1
                continue
            # Each cell fires at one offset in the ripple.
            dur = max(int(self.sequence_length / self.replay_speed), 1)
            for k, cell in enumerate(idx):
                offset = int(k * dur / max(self.sequence_length, 1))
                if t + offset < n_steps:
                    out[t + offset, cell] += 1
            t += dur
        return out


def detect_ripples(pop_activity: np.ndarray, threshold: float = 2.0,
                    min_duration: int = 3) -> list:
    """Detect ripple events as windows where total activity > threshold."""
    total = pop_activity.sum(axis=1)
    above = total >= threshold
    events = []
    start = None
    for t, a in enumerate(above):
        if a and start is None:
            start = t
        elif not a and start is not None:
            if t - start >= min_duration:
                events.append((start, t))
            start = None
    if start is not None and len(above) - start >= min_duration:
        events.append((start, len(above)))
    return events
