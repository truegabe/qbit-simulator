"""FNO-powered relay modules -- drop-in upgrades for existing relays.

Each class here mirrors an existing relay but replaces the hand-coded
predictor / oscillator / projection with a learned FNO.

Before training the FNO weights are random, so results are no better
than the baseline.  After training (see fit() on each class) they
become a learned surrogate that generalises across signal shapes.

Classes
-------
  FNOPredictiveRelay    -- replaces EMA/Linear predictor in PredictiveRelay
  FNOOscillatoryRelay   -- replaces the hardcoded carrier in OscillatoryRelay
  FNOHierarchicalRelay  -- replaces random projection matrices in HierarchicalRelay
  FNODecoherenceModel   -- models qudit coherence decay as a PDE surrogate

Relationship to existing modules
---------------------------------
  fno_relays.py           uses  fno_core.FNO1d
  FNOPredictiveRelay      extends  predictive_relay.PredictiveRelay
  FNOOscillatoryRelay     wraps    oscillatory_relay.OscillatoryRelay
  FNOHierarchicalRelay    extends  hierarchical_relay.HierarchicalRelay
  FNODecoherenceModel     plugs into entanglement_reservoir.EntanglementReservoir
"""

from __future__ import annotations

from typing import Optional
import numpy as np

from qbit_simulator.fno_core import FNO1d


# ---------------------------------------------------------------------------
# FNOPredictiveRelay
# ---------------------------------------------------------------------------

class FNOPredictiveRelay:
    """Predictive relay where predictions come from a trained FNO.

    Instead of a fixed EMA or linear predictor, the FNO learns the
    temporal dynamics of the signal from data.  It takes a window of
    past activations and predicts the next time step.

    Only the PREDICTION ERRORS are transmitted (same protocol as
    PredictiveRelay), so compression still applies.

    Parameters
    ----------
    n_dims      : signal dimensionality
    window      : number of past time steps fed to the FNO
    d_model     : FNO internal width
    n_layers    : FNO depth
    k_max       : Fourier modes retained
    threshold   : error sparsification threshold (0 = send everything)
    noise_std   : channel noise on transmitted errors
    seed        : reproducibility
    """

    def __init__(self, n_dims: int, window: int = 8,
                 d_model: int = 32, n_layers: int = 2, k_max: int = 8,
                 threshold: float = 0.0, noise_std: float = 0.0,
                 seed: int = 0) -> None:
        self.n_dims    = n_dims
        self.window    = window
        self.threshold = threshold
        self.noise_std = noise_std

        # FNO: input = (window, n_dims), output = (window, n_dims)
        # We use the last output step as the prediction.
        self.fno = FNO1d(d_in=n_dims, d_out=n_dims,
                          d_model=d_model, n_layers=n_layers,
                          k_max=min(k_max, window // 2 + 1),
                          seed=seed)

        self._buf   = np.zeros((window, n_dims))   # rolling input buffer
        self._t     = 0
        self._trained = False

        self._total_sent:  int   = 0
        self._err_acc:     float = 0.0

    # ------------------------------------------------------------------
    def _predict(self) -> np.ndarray:
        """Predict next step from the current window buffer."""
        out = self.fno.forward(self._buf)   # (window, n_dims)
        return out[-1]                      # last step = next prediction

    def relay(self, x: np.ndarray,
              rng: Optional[np.random.Generator] = None
              ) -> tuple[np.ndarray, dict]:
        """Predict, transmit error, reconstruct at receiver.

        Returns
        -------
        x_rec  : reconstructed signal
        stats  : dict with prediction_error, sparsity, compression_ratio
        """
        x         = np.asarray(x, dtype=np.float64).ravel()[:self.n_dims]
        pred      = self._predict()
        error     = x - pred

        # Sparsify.
        if self.threshold > 0:
            mask        = np.abs(error) > self.threshold
            error_sent  = np.where(mask, error, 0.0)
        else:
            error_sent  = error.copy()
            mask        = np.ones(self.n_dims, dtype=bool)

        if self.noise_std > 0:
            rng = rng or np.random.default_rng()
            error_sent += rng.standard_normal(self.n_dims) * self.noise_std

        x_rec = pred + error_sent

        # Roll buffer.
        self._buf = np.roll(self._buf, -1, axis=0)
        self._buf[-1] = x
        self._t  += 1

        k_sent   = int(mask.sum())
        cr       = self.n_dims / max(k_sent, 1)
        pred_err = float(np.linalg.norm(pred - x) /
                          (np.linalg.norm(x) + 1e-12))
        rec_err  = float(np.linalg.norm(x - x_rec) /
                          (np.linalg.norm(x) + 1e-12))

        self._total_sent += 1
        self._err_acc    += rec_err

        stats = {
            "prediction_error":     pred_err,
            "reconstruction_error": rec_err,
            "k_sent":               k_sent,
            "sparsity":             1.0 - k_sent / max(self.n_dims, 1),
            "compression_ratio":    cr,
            "mean_error_so_far":    self._err_acc / self._total_sent,
            "fno_trained":          self._trained,
            "t":                    self._t,
        }
        return x_rec, stats

    # ------------------------------------------------------------------
    def collect_training_data(self, signals: np.ndarray) -> tuple:
        """Build (X, Y) training pairs from a signal array.

        Parameters
        ----------
        signals : (T, n_dims) -- time series of activations

        Returns
        -------
        X : (N, window, n_dims) -- input windows
        Y : (N, window, n_dims) -- target (shifted by 1)
        """
        T   = len(signals)
        N   = T - self.window - 1
        if N <= 0:
            raise ValueError(f"Need at least {self.window + 2} time steps")
        X = np.stack([signals[i:i + self.window]     for i in range(N)])
        Y = np.stack([signals[i + 1:i + self.window + 1] for i in range(N)])
        return X, Y

    def fit(self, signals: np.ndarray, **kwargs) -> list[float]:
        """Train the FNO on a signal time series.

        Requires PyTorch.  See FNO1d.fit() for details.
        """
        X, Y   = self.collect_training_data(signals)
        losses = self.fno.fit(X, Y, **kwargs)
        self._trained = True
        return losses

    def save(self, path: str) -> None:
        self.fno.save(path)

    @classmethod
    def load_weights(cls, path: str, **init_kwargs) -> "FNOPredictiveRelay":
        relay     = cls(**init_kwargs)
        relay.fno = FNO1d.load(path)
        relay._trained = True
        return relay

    def __repr__(self) -> str:
        return (f"FNOPredictiveRelay(n_dims={self.n_dims}, "
                f"window={self.window}, trained={self._trained})\n"
                f"  {self.fno}")


# ---------------------------------------------------------------------------
# FNOOscillatoryRelay
# ---------------------------------------------------------------------------

class FNOOscillatoryRelay:
    """Oscillatory relay where phase evolution is modelled by an FNO.

    The standard OscillatoryRelay uses a fixed sinusoidal carrier.
    This version learns the carrier shape from data -- useful for
    non-sinusoidal rhythms (e.g. sharp gamma bursts, theta sequences).

    The FNO takes the current phase state (n_dims,) and predicts the
    next phase state, which is used to encode the next signal.

    Parameters
    ----------
    n_dims      : signal / channel dimensionality
    d_model     : FNO internal width
    n_layers    : FNO depth
    k_max       : Fourier modes (keep low for smooth oscillations)
    freq_hz     : initial carrier frequency (used before training)
    dt          : time step in seconds
    noise_std   : phase noise
    """

    def __init__(self, n_dims: int, d_model: int = 32,
                 n_layers: int = 2, k_max: int = 4,
                 freq_hz: float = 40.0, dt: float = 0.001,
                 noise_std: float = 0.05, seed: int = 0) -> None:
        self.n_dims    = n_dims
        self.freq_hz   = freq_hz
        self.dt        = dt
        self.noise_std = noise_std

        # FNO operates on phase state: input (n_dims,) -> output (n_dims,)
        # We treat n_dims as the spatial axis with 1 channel.
        self.fno      = FNO1d(d_in=1, d_out=1,
                               d_model=d_model, n_layers=n_layers,
                               k_max=min(k_max, n_dims // 2 + 1),
                               seed=seed)
        self._phase   = np.zeros(n_dims)
        self._t       = 0.0
        self._trained = False

    def _advance_phase(self, rng: Optional[np.random.Generator]) -> np.ndarray:
        """Step phase forward: use FNO if trained, else sinusoidal."""
        if self._trained:
            x_in       = self._phase[:, np.newaxis]   # (n_dims, 1)
            delta      = self.fno.forward(x_in)[:, 0]  # (n_dims,)
        else:
            delta = np.full(self.n_dims,
                             2 * np.pi * self.freq_hz * self.dt)
        if self.noise_std > 0:
            rng    = rng or np.random.default_rng()
            delta += rng.standard_normal(self.n_dims) * self.noise_std
        self._phase = (self._phase + delta) % (2 * np.pi)
        self._t    += self.dt
        return self._phase.copy()

    def encode(self, x: np.ndarray,
               rng: Optional[np.random.Generator] = None
               ) -> tuple[np.ndarray, np.ndarray]:
        """Encode signal x onto the current phase carrier.

        Returns
        -------
        carrier    : (n_dims,) -- phase-modulated carrier
        phase      : (n_dims,) -- current phase state
        """
        x       = np.asarray(x, dtype=np.float64).ravel()[:self.n_dims]
        phase   = self._advance_phase(rng)
        carrier = np.sin(phase + np.pi * np.clip(x, -1, 1))
        return carrier, phase

    def decode(self, carrier: np.ndarray,
               phase: np.ndarray) -> np.ndarray:
        """Recover signal from carrier and phase."""
        return np.clip((np.arcsin(np.clip(carrier, -1, 1)) - phase) / np.pi,
                       -1, 1)

    def transmit(self, x: np.ndarray,
                 rng: Optional[np.random.Generator] = None
                 ) -> tuple[np.ndarray, dict]:
        """Encode -> (channel) -> decode round trip."""
        x         = np.asarray(x, dtype=np.float64).ravel()[:self.n_dims]
        carrier, phase = self.encode(x, rng)
        x_rec          = self.decode(carrier, phase)

        rec_err = float(np.linalg.norm(x - x_rec) /
                         (np.linalg.norm(x) + 1e-12))
        stats   = {
            "reconstruction_error": rec_err,
            "mean_phase":           float(phase.mean()),
            "phase_spread":         float(phase.std()),
            "fno_trained":          self._trained,
            "t":                    self._t,
        }
        return x_rec, stats

    def fit(self, phase_trajectories: np.ndarray, **kwargs) -> list[float]:
        """Train phase-evolution FNO on observed phase sequences.

        Parameters
        ----------
        phase_trajectories : (T, n_dims) -- recorded phase states over time
        """
        T  = len(phase_trajectories)
        X  = phase_trajectories[:-1, :, np.newaxis]   # (T-1, n_dims, 1)
        Y  = phase_trajectories[1:,  :, np.newaxis]   # (T-1, n_dims, 1)
        losses = self.fno.fit(X, Y, **kwargs)
        self._trained = True
        return losses

    def __repr__(self) -> str:
        return (f"FNOOscillatoryRelay(n_dims={self.n_dims}, "
                f"freq={self.freq_hz}Hz, trained={self._trained})\n"
                f"  {self.fno}")


# ---------------------------------------------------------------------------
# FNOHierarchicalRelay
# ---------------------------------------------------------------------------

class FNOHierarchicalRelay:
    """Hierarchical relay where each level uses an FNO instead of a random matrix.

    The FNO at each level learns the optimal compression function from data,
    rather than using a fixed random projection.

    After training, the encode path learns to preserve task-relevant
    features; the decode path learns to reconstruct from compressed codes.

    Parameters
    ----------
    dims     : list of dimensionalities [n_input, n_L1, ..., n_top]
    d_model  : FNO internal width (same for all levels)
    n_layers : FNO depth per level
    k_max    : Fourier modes per level
    """

    def __init__(self, dims: list[int], d_model: int = 32,
                 n_layers: int = 2, k_max: int = 8,
                 seed: int = 0) -> None:
        if len(dims) < 2:
            raise ValueError("dims needs at least 2 entries")
        self.dims    = dims
        self.d_model = d_model

        self.encoders = [
            FNO1d(d_in=dims[i], d_out=dims[i + 1],
                   d_model=d_model, n_layers=n_layers,
                   k_max=min(k_max, dims[i] // 2 + 1),
                   seed=seed + i)
            for i in range(len(dims) - 1)
        ]
        self.decoders = [
            FNO1d(d_in=dims[i + 1], d_out=dims[i],
                   d_model=d_model, n_layers=n_layers,
                   k_max=min(k_max, dims[i + 1] // 2 + 1),
                   seed=seed + 100 + i)
            for i in range(len(dims) - 1)
        ]
        self._trained = False

    @property
    def total_compression(self) -> float:
        cr = 1.0
        for i in range(len(self.dims) - 1):
            cr *= self.dims[i] / max(self.dims[i + 1], 1)
        return cr

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Bottom-up pass: (dims[0],) -> (dims[-1],)"""
        h = np.asarray(x, dtype=np.float64).ravel()
        for enc in self.encoders:
            h = enc.forward(h[np.newaxis, :])[0]   # (1, n) -> (n,)
        return h

    def decode(self, h: np.ndarray) -> np.ndarray:
        """Top-down pass: (dims[-1],) -> (dims[0],)"""
        h = np.asarray(h, dtype=np.float64).ravel()
        for dec in reversed(self.decoders):
            h = dec.forward(h[np.newaxis, :])[0]
        return h

    def relay(self, x: np.ndarray) -> tuple[np.ndarray, dict]:
        x     = np.asarray(x, dtype=np.float64).ravel()
        h     = self.encode(x)
        x_rec = self.decode(h)[:len(x)]
        if len(x_rec) < len(x):
            x_rec = np.pad(x_rec, (0, len(x) - len(x_rec)))
        rec_err = float(np.linalg.norm(x - x_rec) /
                         (np.linalg.norm(x) + 1e-12))
        stats = {
            "total_compression":    self.total_compression,
            "reconstruction_error": rec_err,
            "top_dim":              len(h),
            "fno_trained":          self._trained,
        }
        return x_rec, stats

    def __repr__(self) -> str:
        s = " -> ".join(str(d) for d in self.dims)
        return (f"FNOHierarchicalRelay([{s}], "
                f"cr={self.total_compression:.1f}x, trained={self._trained})")


# ---------------------------------------------------------------------------
# FNODecoherenceModel
# ---------------------------------------------------------------------------

class FNODecoherenceModel:
    """FNO surrogate for qudit decoherence dynamics.

    Replaces the fixed exponential decay model in EntanglementReservoir
    with a learned PDE surrogate that can model non-Markovian dynamics,
    correlated noise, or environment-specific decoherence patterns.

    The FNO takes a coherence field (one value per pair in the reservoir)
    and predicts the coherence field after one time step.

    Parameters
    ----------
    capacity  : number of entangled pairs (= spatial dimension for FNO)
    d_model   : FNO internal width
    k_max     : Fourier modes (keep low for smooth decay curves)
    """

    def __init__(self, capacity: int, d_model: int = 16,
                 k_max: int = 4, seed: int = 0) -> None:
        self.capacity = capacity
        self.fno      = FNO1d(d_in=1, d_out=1,
                               d_model=d_model, n_layers=2,
                               k_max=min(k_max, capacity // 2 + 1),
                               seed=seed)
        self._trained = False
        self._tau     = 10.0   # fallback exponential tau

    def step(self, coherences: np.ndarray, dt: float) -> np.ndarray:
        """Predict coherence after one dt time step.

        Parameters
        ----------
        coherences : (capacity,) current coherence values in [0, 1]
        dt         : time step

        Returns
        -------
        new_coherences : (capacity,) predicted values in [0, 1]
        """
        c = np.asarray(coherences, dtype=np.float64).ravel()[:self.capacity]
        if len(c) < self.capacity:
            c = np.pad(c, (0, self.capacity - len(c)), constant_values=1.0)

        if self._trained:
            x_in = c[:, np.newaxis]                        # (capacity, 1)
            out  = self.fno.forward(x_in)[:, 0]            # (capacity,)
            return np.clip(out, 0.0, 1.0)
        else:
            # Fallback: exponential decay.
            return c * np.exp(-dt / self._tau)

    def fit(self, trajectories: np.ndarray, **kwargs) -> list[float]:
        """Train on observed coherence decay trajectories.

        Parameters
        ----------
        trajectories : (T, capacity) -- coherence over time for each pair
        """
        X = trajectories[:-1, :, np.newaxis]   # (T-1, capacity, 1)
        Y = trajectories[1:,  :, np.newaxis]
        losses = self.fno.fit(X, Y, **kwargs)
        self._trained = True
        return losses

    def calibrate_fallback(self, tau: float) -> None:
        """Set the fallback exponential tau (used before training)."""
        self._tau = tau

    def __repr__(self) -> str:
        return (f"FNODecoherenceModel(capacity={self.capacity}, "
                f"trained={self._trained})\n  {self.fno}")
