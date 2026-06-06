"""Binding bus -- solves the binding problem via gamma-band synchrony.

The binding problem: how does the brain combine features processed in
different areas (colour in V4, motion in MT, shape in IT) into one
unified percept?  Binding by synchrony proposes that features that
belong to the same object fire in the same gamma (~40 Hz) phase.

Mechanism
---------
  1. Each region R_i produces a feature vector f_i and assigns a
     PHASE tag phi_i in [0, 2*pi) to each feature.

  2. The BindingBus measures pairwise Phase Locking Values (PLV) between
     regions to decide which feature pairs are "in sync".

  3. Features with matching phases (below phase_threshold difference)
     are BOUND into a unified percept via weighted averaging.

  4. A global gamma oscillator drives all regions with a shared reference;
     regions lock onto this reference when attending the same object.

Classes
-------
  FeatureBundle     -- feature vector + phase tags from one region
  GammaOscillator   -- shared reference oscillator driving phase assignments
  PhaseComparator   -- measures inter-region phase alignment (PLV)
  BindingBus        -- orchestrates N regions into bound percepts
  UnifiedPercept    -- output: merged feature vector + binding metadata
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# GammaOscillator
# ---------------------------------------------------------------------------

class GammaOscillator:
    """Sinusoidal reference oscillator at a given frequency.

    Provides a shared phase reference phi(t) = 2*pi*f*t.
    Regions can lock to this phase or offset it.

    Parameters
    ----------
    freq_hz   : oscillation frequency (default 40 Hz = gamma)
    dt        : time step in seconds (default 1 ms)
    """

    def __init__(self, freq_hz: float = 40.0, dt: float = 0.001) -> None:
        self.freq_hz = freq_hz
        self.dt      = dt
        self._t      = 0.0

    def step(self) -> float:
        """Advance one time step; return current phase in [0, 2*pi)."""
        self._t += self.dt
        return (2 * np.pi * self.freq_hz * self._t) % (2 * np.pi)

    @property
    def current_phase(self) -> float:
        return (2 * np.pi * self.freq_hz * self._t) % (2 * np.pi)

    def reset(self) -> None:
        self._t = 0.0


# ---------------------------------------------------------------------------
# FeatureBundle
# ---------------------------------------------------------------------------

@dataclass
class FeatureBundle:
    """Feature vector + phase assignment from one brain region.

    Parameters
    ----------
    features  : (n_features,) activation vector from the region
    phases    : (n_features,) phase tag for each feature in [0, 2*pi)
    region    : name of the source region
    timestamp : simulation time when this bundle was created
    """
    features:  np.ndarray
    phases:    np.ndarray
    region:    str   = ""
    timestamp: float = 0.0

    def __post_init__(self) -> None:
        self.features = np.asarray(self.features, dtype=np.float64).ravel()
        self.phases   = np.asarray(self.phases,   dtype=np.float64).ravel()
        if len(self.phases) != len(self.features):
            raise ValueError("features and phases must have the same length")

    @property
    def n_features(self) -> int:
        return len(self.features)

    def mean_phase(self) -> float:
        """Circular mean of all feature phases."""
        return float(np.angle(np.mean(np.exp(1j * self.phases))))

    def phase_spread(self) -> float:
        """Circular standard deviation (0 = all in phase, pi = maximally spread)."""
        R = float(np.abs(np.mean(np.exp(1j * self.phases))))
        return float(np.sqrt(-2 * np.log(max(R, 1e-12))))


# ---------------------------------------------------------------------------
# PhaseComparator
# ---------------------------------------------------------------------------

class PhaseComparator:
    """Measure phase alignment between two FeatureBundles.

    Phase Locking Value:
        PLV = |E[exp(i*(phi_a - phi_b))]|

    PLV = 1.0 -> perfectly in phase (same object)
    PLV = 0.0 -> random phase relationship (different objects)

    Parameters
    ----------
    plv_threshold : minimum PLV to consider two features as "bound"
    """

    def __init__(self, plv_threshold: float = 0.7) -> None:
        self.plv_threshold = plv_threshold

    def plv(self, bundle_a: FeatureBundle,
            bundle_b: FeatureBundle) -> float:
        """PLV between the mean phases of two bundles."""
        phi_a = bundle_a.phases
        phi_b = bundle_b.phases
        # Use shortest matching length.
        n      = min(len(phi_a), len(phi_b))
        if n == 0:
            return 0.0
        diff   = phi_a[:n] - phi_b[:n]
        return float(np.abs(np.mean(np.exp(1j * diff))))

    def are_bound(self, bundle_a: FeatureBundle,
                  bundle_b: FeatureBundle) -> bool:
        return self.plv(bundle_a, bundle_b) >= self.plv_threshold

    def pairwise_plv_matrix(self,
                             bundles: list[FeatureBundle]) -> np.ndarray:
        """Return N x N symmetric PLV matrix for a list of bundles."""
        n   = len(bundles)
        mat = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                v        = self.plv(bundles[i], bundles[j])
                mat[i,j] = v
                mat[j,i] = v
        return mat


# ---------------------------------------------------------------------------
# UnifiedPercept
# ---------------------------------------------------------------------------

@dataclass
class UnifiedPercept:
    """Output of the BindingBus.

    Attributes
    ----------
    features     : merged feature vector (weighted average of bound regions)
    binding_mask : (n_regions,) bool -- which regions contributed
    plv_matrix   : (n_regions, n_regions) pairwise phase locking
    mean_plv     : scalar global synchrony
    n_bound      : number of regions included in the percept
    region_names : list of contributing region names
    """
    features:     np.ndarray
    binding_mask: np.ndarray
    plv_matrix:   np.ndarray
    mean_plv:     float
    n_bound:      int
    region_names: list[str]

    def is_bound(self) -> bool:
        """True if at least 2 regions contributed."""
        return self.n_bound >= 2

    def dominant_region(self) -> Optional[str]:
        """Region with largest feature norm."""
        if not self.region_names:
            return None
        return self.region_names[0]


# ---------------------------------------------------------------------------
# BindingBus
# ---------------------------------------------------------------------------

class BindingBus:
    """Synchronize features from N regions into one unified percept.

    Usage
    -----
        bus = BindingBus(plv_threshold=0.7)
        bus.register("V4",  n_features=64)
        bus.register("MT",  n_features=64)
        bus.register("IT",  n_features=128)

        bundle_v4 = bus.encode("V4",  color_features, ref_phase=osc.step())
        bundle_mt = bus.encode("MT",  motion_features, ref_phase=osc.step())
        bundle_it = bus.encode("IT",  shape_features,  ref_phase=osc.step())

        percept = bus.bind([bundle_v4, bundle_mt, bundle_it])

    Parameters
    ----------
    plv_threshold    : PLV above which two regions are considered bound
    output_dim       : dimensionality of merged output (None = max region dim)
    phase_noise_std  : noise added to phase assignments (models imperfect sync)
    merge_mode       : 'mean' | 'max' | 'weighted' (weight by feature norm)
    oscillator       : shared GammaOscillator (created internally if None)
    """

    def __init__(self,
                 plv_threshold: float = 0.7,
                 output_dim: Optional[int] = None,
                 phase_noise_std: float = 0.1,
                 merge_mode: str = "weighted",
                 oscillator: Optional[GammaOscillator] = None) -> None:
        self.plv_threshold   = plv_threshold
        self.output_dim      = output_dim
        self.phase_noise_std = phase_noise_std
        self.merge_mode      = merge_mode
        self.oscillator      = oscillator or GammaOscillator()
        self.comparator      = PhaseComparator(plv_threshold)
        self._regions: dict[str, int] = {}   # name -> n_features
        self._history: list[dict]    = []

    # ------------------------------------------------------------------
    def register(self, name: str, n_features: int) -> None:
        """Register a brain region with its feature dimensionality."""
        self._regions[name] = n_features

    def encode(self, region: str, features: np.ndarray,
               ref_phase: float = 0.0,
               timestamp: float = 0.0,
               rng: Optional[np.random.Generator] = None) -> FeatureBundle:
        """Assign gamma phases to features and wrap into a FeatureBundle.

        Each feature gets phase = ref_phase + noise, so features that are
        co-activated near the same gamma cycle are assigned similar phases.

        Parameters
        ----------
        features  : (n_features,) activation vector
        ref_phase : reference phase (from shared GammaOscillator)
        timestamp : simulation time
        """
        features = np.asarray(features, dtype=np.float64).ravel()
        rng      = rng or np.random.default_rng()

        # Features with higher amplitude lock more tightly to the reference.
        amplitudes   = np.abs(features)
        max_amp      = float(amplitudes.max()) if len(amplitudes) > 0 else 1.0
        max_amp      = max(max_amp, 1e-12)
        lock_strength = amplitudes / max_amp   # in [0, 1]

        # Phase = ref ± noise scaled by (1 - lock_strength).
        noise  = rng.standard_normal(len(features)) * self.phase_noise_std
        phases = (ref_phase + noise * (1 - lock_strength)) % (2 * np.pi)

        return FeatureBundle(features=features, phases=phases,
                              region=region, timestamp=timestamp)

    # ------------------------------------------------------------------
    def bind(self, bundles: list[FeatureBundle],
             anchor: Optional[FeatureBundle] = None
             ) -> UnifiedPercept:
        """Merge features from in-phase regions into a unified percept.

        Parameters
        ----------
        bundles : list of FeatureBundle from different regions
        anchor  : if given, only bind regions in-phase with this bundle;
                  otherwise the strongest region is the anchor.

        Returns
        -------
        UnifiedPercept
        """
        if not bundles:
            empty = np.zeros(self.output_dim or 1)
            return UnifiedPercept(features=empty,
                                   binding_mask=np.zeros(0, dtype=bool),
                                   plv_matrix=np.zeros((0, 0)),
                                   mean_plv=0.0, n_bound=0,
                                   region_names=[])

        n      = len(bundles)
        plv_mat = self.comparator.pairwise_plv_matrix(bundles)

        # Choose anchor: bundle with highest feature norm; track its index.
        if anchor is None:
            norms       = [np.linalg.norm(b.features) for b in bundles]
            anchor_idx  = int(np.argmax(norms))
            anchor      = bundles[anchor_idx]
        else:
            # Find anchor by identity; fall back to 0 if not in list.
            anchor_idx = next(
                (i for i, b in enumerate(bundles) if b is anchor), 0)

        # Binding mask: in-phase with anchor.
        mask = np.array([
            self.comparator.plv(anchor, b) >= self.plv_threshold
            for b in bundles
        ])

        bound_bundles = [b for b, m in zip(bundles, mask) if m]
        if not bound_bundles:
            bound_bundles    = [anchor]
            mask[anchor_idx] = True

        # Output dimension.
        out_dim = self.output_dim or max(b.n_features for b in bound_bundles)

        # Merge.
        out = self._merge(bound_bundles, out_dim)

        mean_plv = float(plv_mat[np.triu_indices(n, k=1)].mean()) if n > 1 else 0.0

        region_names = [b.region for b in bound_bundles]
        percept = UnifiedPercept(features=out,
                                  binding_mask=mask,
                                  plv_matrix=plv_mat,
                                  mean_plv=mean_plv,
                                  n_bound=int(mask.sum()),
                                  region_names=region_names)

        self._history.append({
            "n_regions": n,
            "n_bound":   percept.n_bound,
            "mean_plv":  mean_plv,
        })
        return percept

    # ------------------------------------------------------------------
    def _merge(self, bundles: list[FeatureBundle], out_dim: int) -> np.ndarray:
        """Merge bound bundles into a single feature vector."""
        out = np.zeros(out_dim)

        if self.merge_mode == "mean":
            for b in bundles:
                f = b.features
                l = min(len(f), out_dim)
                out[:l] += f[:l]
            out /= max(len(bundles), 1)

        elif self.merge_mode == "max":
            for b in bundles:
                f = b.features
                l = min(len(f), out_dim)
                out[:l] = np.maximum(out[:l], f[:l])

        elif self.merge_mode == "weighted":
            total_w = 0.0
            for b in bundles:
                w  = np.linalg.norm(b.features) + 1e-12
                f  = b.features
                l  = min(len(f), out_dim)
                out[:l] += w * f[:l]
                total_w  += w
            out /= max(total_w, 1e-12)

        else:
            raise ValueError(f"unknown merge_mode: {self.merge_mode}")

        return out

    # ------------------------------------------------------------------
    def step_and_bind(self, region_activations: dict[str, np.ndarray],
                      rng: Optional[np.random.Generator] = None
                      ) -> UnifiedPercept:
        """Convenience: advance oscillator, encode all regions, bind.

        Parameters
        ----------
        region_activations : {region_name: activation_vector}
        """
        ref   = self.oscillator.step()
        rng   = rng or np.random.default_rng()
        bundles = []
        for name, act in region_activations.items():
            b = self.encode(name, act, ref_phase=ref,
                             timestamp=self.oscillator._t, rng=rng)
            bundles.append(b)
        return self.bind(bundles)

    def history(self) -> list[dict]:
        return list(self._history)

    def __repr__(self) -> str:
        regions = list(self._regions.keys())
        return (f"BindingBus(regions={regions}, "
                f"plv_threshold={self.plv_threshold}, "
                f"merge='{self.merge_mode}')")
