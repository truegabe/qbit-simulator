"""Variational Auto-Encoder (Kingma & Welling 2013).

Encoder q(z | x) outputs (μ, log σ²) of a Gaussian; decoder p(x | z)
reconstructs x. Trained to minimize

    L = - E_q[log p(x | z)] + KL(q(z|x) || p(z))

with p(z) = N(0, I). The reparameterization trick z = μ + σ ⊙ ε
allows gradient backprop through the sampling step.

Numpy-only implementation. Encoder/decoder are 1-hidden-layer MLPs.
Optimizer: vanilla SGD with momentum.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(np.float64)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


@dataclass
class VAE:
    """One-hidden-layer Gaussian-encoder Bernoulli-decoder VAE."""
    n_in: int
    n_hidden: int = 64
    n_latent: int = 8
    eta: float = 0.005
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0), repr=False)

    W_enc1: np.ndarray = field(default=None, repr=False)
    b_enc1: np.ndarray = field(default=None, repr=False)
    W_mu:   np.ndarray = field(default=None, repr=False)
    W_logvar: np.ndarray = field(default=None, repr=False)
    b_mu:   np.ndarray = field(default=None, repr=False)
    b_logvar: np.ndarray = field(default=None, repr=False)
    W_dec1: np.ndarray = field(default=None, repr=False)
    b_dec1: np.ndarray = field(default=None, repr=False)
    W_dec_out: np.ndarray = field(default=None, repr=False)
    b_dec_out: np.ndarray = field(default=None, repr=False)

    def __post_init__(self) -> None:
        def _w(d_out, d_in):
            return self.rng.normal(0, np.sqrt(2.0 / d_in), (d_out, d_in))
        if self.W_enc1 is None:
            self.W_enc1 = _w(self.n_hidden, self.n_in)
            self.b_enc1 = np.zeros(self.n_hidden)
            self.W_mu   = _w(self.n_latent, self.n_hidden)
            self.b_mu   = np.zeros(self.n_latent)
            self.W_logvar = _w(self.n_latent, self.n_hidden)
            self.b_logvar = np.zeros(self.n_latent)
            self.W_dec1 = _w(self.n_hidden, self.n_latent)
            self.b_dec1 = np.zeros(self.n_hidden)
            self.W_dec_out = _w(self.n_in, self.n_hidden)
            self.b_dec_out = np.zeros(self.n_in)

    def encode(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h = relu(self.W_enc1 @ x + self.b_enc1)
        mu = self.W_mu @ h + self.b_mu
        logvar = self.W_logvar @ h + self.b_logvar
        return mu, logvar, h

    def reparam(self, mu: np.ndarray, logvar: np.ndarray) -> np.ndarray:
        eps = self.rng.standard_normal(mu.shape)
        return mu + np.exp(0.5 * logvar) * eps, eps

    def decode(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = relu(self.W_dec1 @ z + self.b_dec1)
        x_logits = self.W_dec_out @ h + self.b_dec_out
        return sigmoid(x_logits), h

    def loss_and_grads(self, x: np.ndarray) -> tuple[float, dict]:
        mu, logvar, h_enc = self.encode(x)
        z, eps = self.reparam(mu, logvar)
        x_hat, h_dec = self.decode(z)
        # Bernoulli reconstruction loss.
        rec = -((x * np.log(x_hat + 1e-9) + (1 - x) * np.log(1 - x_hat + 1e-9))).sum()
        # KL term.
        kl = -0.5 * (1 + logvar - mu * mu - np.exp(logvar)).sum()
        loss = rec + kl

        # Backprop (manual).
        dlogits = x_hat - x   # dL_rec/dlogits
        gW_dec_out = np.outer(dlogits, h_dec)
        gb_dec_out = dlogits
        dh_dec = self.W_dec_out.T @ dlogits
        dh_dec_pre = dh_dec * relu_grad(self.W_dec1 @ z + self.b_dec1)
        gW_dec1 = np.outer(dh_dec_pre, z)
        gb_dec1 = dh_dec_pre
        dz = self.W_dec1.T @ dh_dec_pre

        # dz/dmu = 1, dz/dlogvar = 0.5 exp(0.5 logvar) eps.
        dmu_rec = dz
        dlogvar_rec = dz * 0.5 * np.exp(0.5 * logvar) * eps
        # KL grads.
        dmu_kl = mu
        dlogvar_kl = 0.5 * (np.exp(logvar) - 1)
        dmu = dmu_rec + dmu_kl
        dlogvar = dlogvar_rec + dlogvar_kl

        gW_mu = np.outer(dmu, h_enc); gb_mu = dmu
        gW_logvar = np.outer(dlogvar, h_enc); gb_logvar = dlogvar
        dh_enc = self.W_mu.T @ dmu + self.W_logvar.T @ dlogvar
        dh_enc_pre = dh_enc * relu_grad(self.W_enc1 @ x + self.b_enc1)
        gW_enc1 = np.outer(dh_enc_pre, x)
        gb_enc1 = dh_enc_pre
        return float(loss), {
            "W_enc1": gW_enc1, "b_enc1": gb_enc1,
            "W_mu":   gW_mu,   "b_mu":   gb_mu,
            "W_logvar": gW_logvar, "b_logvar": gb_logvar,
            "W_dec1": gW_dec1, "b_dec1": gb_dec1,
            "W_dec_out": gW_dec_out, "b_dec_out": gb_dec_out,
        }

    def step(self, x: np.ndarray) -> float:
        loss, g = self.loss_and_grads(x)
        for k, gv in g.items():
            setattr(self, k, getattr(self, k) - self.eta * gv)
        return loss

    def train(self, X: np.ndarray, n_iter: int = 1000) -> list:
        losses = []
        for it in range(n_iter):
            x = X[self.rng.integers(0, X.shape[0])]
            losses.append(self.step(x))
        return losses
