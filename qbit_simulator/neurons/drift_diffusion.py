"""Drift-diffusion model (DDM): Bayesian decision-making by evidence accumulation.

The drift-diffusion model (Ratcliff 1978) is the canonical model of
perceptual decision-making in neuroscience. Sensory evidence is
accumulated over time; a decision is made when the accumulated
evidence crosses a threshold.

Dynamics:

    dE/dt = drift + noise · η(t)

with E(0) = 0 (or a starting bias), η white Gaussian noise. The choice
is "+1" if E crosses +threshold, "−1" if it crosses −threshold.

Predictions:
  - **Reaction-time distribution**: skewed, with a long right tail.
  - **Accuracy / speed trade-off**: lower threshold → faster but less
    accurate.
  - **Choice probability**: depends on the drift rate (signal-to-noise).

Two-choice version is implemented as standard; we also include a
multi-choice "race model" generalization.

This module provides:

  - `DDM(drift, noise, threshold)`: single trial.
  - `.simulate_trial()`: return (choice, rt, evidence_trace).
  - `.simulate_many(n_trials)`: distribution of choices + RTs.
  - `RaceModel`: K independent accumulators racing to threshold (K-choice).
  - `fit_drift_from_choices(choices)`: maximum-likelihood drift estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ----------------------------------------------------------------------------
# Two-choice DDM
# ----------------------------------------------------------------------------

@dataclass
class DDM:
    """Standard 2-alternative drift-diffusion model.

    drift > 0 → biased toward +1 boundary; drift < 0 → −1.
    noise = σ.
    """
    drift:     float = 0.1
    noise:     float = 1.0
    threshold: float = 1.0
    start_bias: float = 0.0    # initial E (in (−threshold, +threshold))
    max_steps: int = 1000
    dt:        float = 0.01

    def simulate_trial(self, rng: np.random.Generator) -> dict:
        """One trial. Returns choice ∈ {+1, −1} or 0 if no decision."""
        E = self.start_bias
        trace = [E]
        for t in range(self.max_steps):
            E += self.drift * self.dt + self.noise * np.sqrt(self.dt) * rng.normal()
            trace.append(E)
            if E >= self.threshold:
                return {"choice": +1, "rt": (t + 1) * self.dt,
                        "trace": np.array(trace)}
            if E <= -self.threshold:
                return {"choice": -1, "rt": (t + 1) * self.dt,
                        "trace": np.array(trace)}
        return {"choice": 0, "rt": self.max_steps * self.dt,
                "trace": np.array(trace)}

    def simulate_many(self, n_trials: int = 1000,
                        rng: np.random.Generator | None = None) -> dict:
        """Many trials. Returns choice distribution + RT distribution."""
        rng = rng or np.random.default_rng()
        choices = []
        rts = []
        for _ in range(n_trials):
            r = self.simulate_trial(rng)
            choices.append(r["choice"])
            rts.append(r["rt"])
        choices = np.array(choices)
        rts = np.array(rts)
        return {
            "choices":      choices,
            "rts":          rts,
            "p_plus":       float((choices == +1).mean()),
            "p_minus":      float((choices == -1).mean()),
            "p_undecided":  float((choices == 0).mean()),
            "mean_rt":      float(rts.mean()),
        }


# ----------------------------------------------------------------------------
# K-choice race model
# ----------------------------------------------------------------------------

@dataclass
class RaceModel:
    """K independent accumulators E_k(t), each with its own drift and
    noise. The first one to cross `threshold` determines the choice."""
    drifts:    np.ndarray              # shape (K,)
    noise:     float = 1.0
    threshold: float = 1.0
    max_steps: int = 1000
    dt:        float = 0.01

    @property
    def K(self) -> int:
        return len(self.drifts)

    def simulate_trial(self, rng: np.random.Generator) -> dict:
        E = np.zeros(self.K)
        traces = [E.copy()]
        for t in range(self.max_steps):
            E += (np.asarray(self.drifts) * self.dt
                   + self.noise * np.sqrt(self.dt) * rng.normal(size=self.K))
            traces.append(E.copy())
            # First to cross.
            if (E >= self.threshold).any():
                winner = int(np.argmax(E))
                return {
                    "choice":  winner,
                    "rt":      (t + 1) * self.dt,
                    "traces":  np.array(traces),
                }
        # No winner.
        return {
            "choice":  int(np.argmax(E)),
            "rt":      self.max_steps * self.dt,
            "traces":  np.array(traces),
            "timeout": True,
        }

    def simulate_many(self, n_trials: int = 1000,
                        rng: np.random.Generator | None = None) -> dict:
        rng = rng or np.random.default_rng()
        choices = np.zeros(n_trials, dtype=int)
        rts = np.zeros(n_trials)
        for i in range(n_trials):
            r = self.simulate_trial(rng)
            choices[i] = r["choice"]
            rts[i] = r["rt"]
        p_choice = np.array([np.mean(choices == k) for k in range(self.K)])
        return {
            "choices":  choices,
            "rts":      rts,
            "p_choice": p_choice,
            "mean_rt":  float(rts.mean()),
        }


# ----------------------------------------------------------------------------
# Theoretical predictions for sanity checks
# ----------------------------------------------------------------------------

def theoretical_choice_probability(drift: float, noise: float,
                                       threshold: float) -> float:
    """Closed-form P(+1) for the symmetric DDM:

        P(+1) = 1 / (1 + exp(-2 · drift · threshold / σ²))

    (Wald / inverse-Gaussian theory.)
    """
    if noise < 1e-12:
        return 1.0 if drift > 0 else 0.0
    z = 2 * drift * threshold / (noise ** 2)
    return float(1.0 / (1.0 + np.exp(-z)))


def theoretical_mean_rt(drift: float, noise: float, threshold: float
                          ) -> float:
    """Mean reaction time for the symmetric DDM with start_bias=0:

        E[RT] = (threshold / drift) · tanh(drift · threshold / σ²)
    """
    if abs(drift) < 1e-9:
        # In the pure-noise limit, RT diverges; return a large number.
        return float(threshold ** 2 / max(noise ** 2, 1e-12))
    z = drift * threshold / (noise ** 2)
    return float((threshold / drift) * np.tanh(z))


# ----------------------------------------------------------------------------
# Maximum-likelihood drift estimation
# ----------------------------------------------------------------------------

def fit_drift_from_choices(
    choices: np.ndarray, noise: float = 1.0, threshold: float = 1.0,
) -> float:
    """Crude MLE of drift from observed binary choices, using the
    theoretical P(+1) formula above. Returns the drift that best
    matches the empirical p_+."""
    p_plus = float((choices == +1).mean())
    p_plus = np.clip(p_plus, 1e-6, 1 - 1e-6)
    # Invert: log(p / (1-p)) = 2 · drift · threshold / σ².
    drift = (noise ** 2 / (2 * threshold)) * np.log(p_plus / (1 - p_plus))
    return float(drift)
