"""Vestibulo-ocular reflex (VOR) — image stabilization.

When the head rotates, the eyes counter-rotate at equal speed in the
opposite direction, keeping the retinal image stable.

Architecture:
  - Vestibular afferent: encodes head velocity Ω(t).
  - Direct pathway (brainstem) maps Ω → eye velocity (gain ≈ 1).
  - Cerebellar side-loop: learns to adapt the gain based on retinal
    slip (visual error).

This module implements adaptive VOR gain via gradient descent on
retinal-slip error.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class VORAgent:
    gain: float = 0.5      # starts low, learns toward 1
    eta: float = 0.01

    def eye_velocity(self, head_vel: float) -> float:
        return -self.gain * head_vel

    def update(self, head_vel: float, visual_motion: float = 0.0) -> float:
        """visual_motion: motion of visual world in retinal coordinates.

        Ideal: visual_motion = 0 (perfect stabilization).
        Retinal slip = visual_motion + (head_vel + eye_vel)
                     = visual_motion + head_vel * (1 - gain).
        """
        eye_vel = self.eye_velocity(head_vel)
        slip = visual_motion + (head_vel + eye_vel)
        # Gradient: dL/dgain = -slip * head_vel (slip ~ (1 - gain) head_vel).
        self.gain += self.eta * slip * head_vel
        return float(slip)


def simulate_vor(agent: VORAgent, head_vel_seq: np.ndarray,
                  visual_motion_seq: np.ndarray | None = None) -> dict:
    if visual_motion_seq is None:
        visual_motion_seq = np.zeros_like(head_vel_seq)
    slips = []; gains = []
    for hv, vm in zip(head_vel_seq, visual_motion_seq):
        slips.append(agent.update(hv, vm))
        gains.append(agent.gain)
    return {"slip": np.array(slips), "gain": np.array(gains)}
