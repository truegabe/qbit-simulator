"""Hemispheric bridge -- corpus callosum model.

The human corpus callosum is a thick white-matter tract connecting the
left and right cerebral hemispheres through ~200-300 million axons.
Despite the massive hemispheres (billions of neurons each), communication
passes through this tight bottleneck.

Key properties modelled here
-----------------------------
1. BOTTLENECK:  Hemispheres operate at high dimensionality; only a small
   fraction of information crosses (top-k or compressed projection).

2. INTERHEMISPHERIC INHIBITION:  When one hemisphere is strongly active,
   it sends inhibitory signals to homologous areas of the other hemisphere
   (winner-take-all between hemispheres for motor commands, language, etc.).

3. HANDEDNESS / SPECIALISATION:  Left hemisphere is biased toward
   language/analytic; right toward spatial/holistic.  Modelled as
   asymmetric bottleneck widths and distinct processing transforms.

4. CALLOSAL DELAY:  Signals cross with a ~10 ms delay (myelination-limited).

Classes
-------
  Hemisphere          -- encapsulates one hemisphere's state and transform
  BottleneckChannel   -- top-k sparse or random-projection crossing
  InhibitionModule    -- inter-hemispheric suppression
  HemisphericBridge   -- full left <-> right relay system
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Hemisphere
# ---------------------------------------------------------------------------

class Hemisphere:
    """One hemisphere: state vector + optional linear transform.

    The transform can represent specialised processing (e.g. left = FFT-like
    analytic, right = spatial pooling) applied BEFORE the signal is sent
    across the bridge.

    Parameters
    ----------
    n_dims     : number of neurons / channels
    name       : 'left' or 'right' (or any label)
    transform  : optional (n_out, n_dims) matrix applied on encode;
                 if None, identity is used (n_out = n_dims)
    """

    def __init__(self, n_dims: int, name: str = "left",
                 transform: Optional[np.ndarray] = None) -> None:
        self.n_dims    = n_dims
        self.name      = name
        self.transform = transform   # (n_out, n_dims) or None
        self._state    = np.zeros(n_dims)
        self._activity_history: list[float] = []

    @property
    def state(self) -> np.ndarray:
        return self._state.copy()

    def update(self, x: np.ndarray) -> None:
        self._state = np.asarray(x, dtype=np.float64).ravel()[:self.n_dims]
        self._activity_history.append(float(np.linalg.norm(self._state)))

    def encode(self) -> np.ndarray:
        """Apply hemisphere-specific transform before crossing."""
        if self.transform is not None:
            return self.transform @ self._state
        return self._state.copy()

    @property
    def mean_activity(self) -> float:
        if not self._activity_history:
            return 0.0
        return float(np.mean(self._activity_history[-20:]))

    def dominance_score(self) -> float:
        """Scalar measure of how active this hemisphere is (for inhibition)."""
        return float(np.linalg.norm(self._state))

    def __repr__(self) -> str:
        return f"Hemisphere(name='{self.name}', n_dims={self.n_dims})"


# ---------------------------------------------------------------------------
# BottleneckChannel
# ---------------------------------------------------------------------------

class BottleneckChannel:
    """Compress a signal through a callosal bottleneck.

    Two strategies:
      'topk'       -- only the k largest activations cross (sparse)
      'projection' -- random Gaussian projection to m_bottleneck dims,
                      then pseudo-inverse at the other side

    Parameters
    ----------
    n_in         : dimensionality of the sending hemisphere's encoded output
    n_out        : dimensionality expected by the receiving hemisphere
    bottleneck   : number of values that physically cross the bridge
    mode         : 'topk' | 'projection'
    delay_steps  : number of time steps the signal is buffered (callosal delay)
    noise_std    : axonal noise during crossing
    seed         : for reproducible projection matrix
    """

    def __init__(self, n_in: int, n_out: int,
                 bottleneck: int = 32,
                 mode: str = "topk",
                 delay_steps: int = 1,
                 noise_std: float = 0.0,
                 seed: int = 0) -> None:
        self.n_in        = n_in
        self.n_out       = n_out
        self.bottleneck  = min(bottleneck, n_in)
        self.mode        = mode
        self.noise_std   = noise_std

        # Delay buffer (FIFO).
        from collections import deque
        self._buf: deque = deque(
            [np.zeros(n_in)] * delay_steps,
            maxlen=max(delay_steps, 1))

        if mode == "projection":
            rng      = np.random.default_rng(seed)
            raw      = rng.standard_normal((bottleneck, n_in))
            raw     /= np.sqrt(bottleneck)
            self.Phi = raw                          # (bottleneck, n_in)
            self.PhiT = raw.T                       # (n_in, bottleneck)
        else:
            self.Phi  = None
            self.PhiT = None

        # Reconstruction matrix: project from n_in -> n_out if sizes differ.
        if n_in != n_out:
            rng2        = np.random.default_rng(seed + 99)
            self.R_out  = rng2.standard_normal((n_out, n_in))
            self.R_out /= np.maximum(
                np.linalg.norm(self.R_out, axis=1, keepdims=True), 1e-12)
        else:
            self.R_out = None

    # ------------------------------------------------------------------
    def transmit(self, x: np.ndarray,
                 rng: Optional[np.random.Generator] = None
                 ) -> tuple[np.ndarray, dict]:
        """Push x into buffer; return the delayed, compressed, reconstructed signal."""
        x = np.asarray(x, dtype=np.float64).ravel()[:self.n_in]
        if len(x) < self.n_in:
            x = np.pad(x, (0, self.n_in - len(x)))

        # Delay: push new signal, pop oldest.
        oldest = self._buf[0].copy()
        self._buf.append(x)
        x_delayed = oldest

        # Bottleneck compression.
        if self.mode == "topk":
            k       = self.bottleneck
            idx     = np.argpartition(np.abs(x_delayed), -k)[-k:]
            crossed = np.zeros(self.n_in)
            crossed[idx] = x_delayed[idx]
            if self.noise_std > 0:
                rng  = rng or np.random.default_rng()
                crossed[idx] += rng.standard_normal(k) * self.noise_std
            x_rec = crossed

        elif self.mode == "projection":
            y = self.Phi @ x_delayed
            if self.noise_std > 0:
                rng = rng or np.random.default_rng()
                y  += rng.standard_normal(len(y)) * self.noise_std
            x_rec = self.PhiT @ y   # pseudo-inverse reconstruction
        else:
            raise ValueError(f"unknown mode: {self.mode}")

        # Map to output dimensionality.
        if self.R_out is not None:
            x_rec = self.R_out @ x_rec

        cmp_len = min(self.n_in, self.n_out)
        rec_err = float(np.linalg.norm(x_delayed[:cmp_len] -
                                        x_rec[:cmp_len]) /
                         (np.linalg.norm(x_delayed[:cmp_len]) + 1e-12))

        stats = {
            "bottleneck":           self.bottleneck,
            "mode":                 self.mode,
            "compression_ratio":    self.n_in / max(self.bottleneck, 1),
            "reconstruction_error": rec_err,
        }
        return x_rec[:self.n_out], stats


# ---------------------------------------------------------------------------
# InhibitionModule
# ---------------------------------------------------------------------------

class InhibitionModule:
    """Interhemispheric inhibition.

    When hemisphere A dominates (high norm), it sends an inhibitory signal
    to hemisphere B:
        x_B_inhibited = x_B * (1 - inh_strength * sigma(dominance_A - dominance_B))

    This implements a soft winner-take-all between hemispheres.

    Parameters
    ----------
    inh_strength : maximum suppression level [0, 1]
    """

    def __init__(self, inh_strength: float = 0.5) -> None:
        self.inh_strength = inh_strength

    @staticmethod
    def _sigmoid(z: float) -> float:
        return 1.0 / (1.0 + np.exp(-z))

    def apply(self, x_recv: np.ndarray,
              dom_sender: float, dom_receiver: float) -> tuple[np.ndarray, float]:
        """Suppress x_recv based on the sender's relative dominance.

        Returns
        -------
        x_suppressed : inhibited signal
        inh_factor   : scalar in [0, 1] (1 = no suppression)
        """
        diff        = dom_sender - dom_receiver
        inh         = self.inh_strength * self._sigmoid(diff)
        inh_factor  = 1.0 - inh
        return x_recv * inh_factor, float(inh_factor)


# ---------------------------------------------------------------------------
# HemisphericBridge
# ---------------------------------------------------------------------------

class HemisphericBridge:
    """Full corpus callosum model: left <-> right bidirectional relay.

    Each time step:
      1. Both hemispheres encode their current state.
      2. Each encoded signal passes through its bottleneck channel.
      3. Interhemispheric inhibition is applied to the RECEIVED signals.
      4. Each hemisphere's state is updated with its own state + received.

    Parameters
    ----------
    n_left, n_right : neuron counts per hemisphere
    bottleneck_lr   : bottleneck width for left->right crossing
    bottleneck_rl   : bottleneck width for right->left crossing
    mode            : bottleneck compression ('topk' | 'projection')
    delay_steps     : callosal transmission delay (time steps)
    inh_strength    : inter-hemispheric inhibition strength [0, 1]
    noise_std       : channel noise
    left_transform, right_transform : optional specialisation matrices
    """

    def __init__(self,
                 n_left: int = 256, n_right: int = 256,
                 bottleneck_lr: int = 32, bottleneck_rl: int = 32,
                 mode: str = "topk",
                 delay_steps: int = 1,
                 inh_strength: float = 0.3,
                 noise_std: float = 0.0,
                 left_transform: Optional[np.ndarray] = None,
                 right_transform: Optional[np.ndarray] = None,
                 seed: int = 0) -> None:

        self.left  = Hemisphere(n_left,  "left",  left_transform)
        self.right = Hemisphere(n_right, "right", right_transform)

        n_left_enc  = left_transform.shape[0]  if left_transform  is not None else n_left
        n_right_enc = right_transform.shape[0] if right_transform is not None else n_right

        self.ch_lr = BottleneckChannel(n_left_enc,  n_right,
                                        bottleneck=bottleneck_lr,
                                        mode=mode, delay_steps=delay_steps,
                                        noise_std=noise_std, seed=seed)
        self.ch_rl = BottleneckChannel(n_right_enc, n_left,
                                        bottleneck=bottleneck_rl,
                                        mode=mode, delay_steps=delay_steps,
                                        noise_std=noise_std, seed=seed + 1)

        self.inhibition  = InhibitionModule(inh_strength)
        self._step_count = 0
        self._stats_log: list[dict] = []

    # ------------------------------------------------------------------
    def step(self, x_left: np.ndarray, x_right: np.ndarray,
             rng: Optional[np.random.Generator] = None) -> dict:
        """One simulation step: update both hemispheres and exchange signals.

        Parameters
        ----------
        x_left  : new activation for the left hemisphere
        x_right : new activation for the right hemisphere

        Returns
        -------
        stats : dict with crossing stats, inhibition factors, dominance
        """
        rng = rng or np.random.default_rng()

        # Update hemisphere states.
        self.left.update(x_left)
        self.right.update(x_right)

        # Encode.
        enc_left  = self.left.encode()
        enc_right = self.right.encode()

        # Cross bottlenecks.
        recv_right, stats_lr = self.ch_lr.transmit(enc_left,  rng)
        recv_left,  stats_rl = self.ch_rl.transmit(enc_right, rng)

        # Interhemispheric inhibition.
        dom_l = self.left.dominance_score()
        dom_r = self.right.dominance_score()
        recv_right, inh_r = self.inhibition.apply(recv_right, dom_l, dom_r)
        recv_left,  inh_l = self.inhibition.apply(recv_left,  dom_r, dom_l)

        self._step_count += 1
        stats = {
            "step":              self._step_count,
            "dominance_left":    dom_l,
            "dominance_right":   dom_r,
            "inh_factor_left":   inh_l,
            "inh_factor_right":  inh_r,
            "lr_error":          stats_lr["reconstruction_error"],
            "rl_error":          stats_rl["reconstruction_error"],
            "lr_compression":    stats_lr["compression_ratio"],
            "rl_compression":    stats_rl["compression_ratio"],
            "recv_left":         recv_left,
            "recv_right":        recv_right,
        }
        self._stats_log.append({k: v for k, v in stats.items()
                                  if not isinstance(v, np.ndarray)})
        return stats

    def dominant_hemisphere(self) -> str:
        """Return name of the currently dominant hemisphere."""
        if self.left.dominance_score() >= self.right.dominance_score():
            return "left"
        return "right"

    def history(self) -> list[dict]:
        return list(self._stats_log)

    def __repr__(self) -> str:
        return (f"HemisphericBridge("
                f"left={self.left.n_dims}, right={self.right.n_dims}, "
                f"bn_lr={self.ch_lr.bottleneck}, bn_rl={self.ch_rl.bottleneck})")
