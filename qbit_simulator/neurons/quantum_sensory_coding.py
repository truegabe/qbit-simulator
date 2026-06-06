"""Quantum-coded sensory representations.

Bridges the neural sensory stack (retina → V1) to the quantum layer
by encoding visual features as quantum amplitudes.

Three encoding schemes are provided:

  1. **Amplitude encoding** — flatten a 2^n-pixel image and write
     each pixel intensity into the corresponding amplitude (after
     normalization). Each n-qubit register holds an N-pixel image.

  2. **Basis encoding** — for binary input (after retinal thresholding),
     each pixel maps to one qubit, |0> or |1>.

  3. **Angle encoding** — each pixel's intensity becomes the rotation
     angle of one RY gate on its own qubit. Compact, lossy.

Once encoded, V1-like processing becomes a **unitary** that filters
features: a parameterized circuit whose layers implement
orientation-selective filtering.

We then provide a *quantum discrimination metric*: the SWAP test
between two quantum-encoded images gives |⟨ψ_A|ψ_B⟩|² in a single
overlap measurement. For images that share an orientation feature
the overlap is high; for orthogonal orientations it's low.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .retina import Retina


# ----------------------------------------------------------------------------
# Encodings
# ----------------------------------------------------------------------------

def amplitude_encode(image: np.ndarray) -> np.ndarray:
    """Encode a 2D image into a state vector with one amplitude per pixel.

    The image is padded to the next power of two and L²-normalized.
    """
    flat = image.astype(np.float64).ravel()
    flat = np.maximum(flat, 0)    # non-negative for valid probability density
    n = len(flat)
    d = 1
    while d < n:
        d *= 2
    padded = np.zeros(d)
    padded[:n] = flat
    norm = np.linalg.norm(padded)
    if norm < 1e-12:
        # Avoid zero-norm state: fallback to uniform.
        return np.ones(d, dtype=np.complex128) / np.sqrt(d)
    return (padded / norm).astype(np.complex128)


def basis_encode(binary_image: np.ndarray) -> np.ndarray:
    """Encode a binary image as a single basis state |bits⟩.

    Length of binary_image dictates the number of qubits.
    """
    bits = binary_image.astype(int).ravel()
    n = len(bits)
    idx = 0
    for b in bits:
        idx = (idx << 1) | int(b > 0)
    psi = np.zeros(2 ** n, dtype=np.complex128)
    psi[idx] = 1.0
    return psi


def angle_encode(image: np.ndarray, scale: float = np.pi) -> np.ndarray:
    """Per-pixel RY rotation on independent qubits. State factorizes."""
    flat = image.astype(np.float64).ravel()
    # Normalize to [0, 1].
    m = flat.max() if flat.max() > 0 else 1.0
    flat = flat / m
    psi = np.array([1.0], dtype=np.complex128)
    for v in flat:
        th = scale * v
        c, s = np.cos(th / 2), np.sin(th / 2)
        qubit = np.array([c, s], dtype=np.complex128)
        psi = np.kron(psi, qubit)
    return psi


# ----------------------------------------------------------------------------
# Quantum overlap test (SWAP test)
# ----------------------------------------------------------------------------

def swap_test_overlap(psi_a: np.ndarray, psi_b: np.ndarray) -> float:
    """Return |⟨ψ_a | ψ_b⟩|² — what the SWAP test measures.

    Both states must have the same dimension.
    """
    if psi_a.shape != psi_b.shape:
        raise ValueError("states must have same dimension")
    return float(np.abs(psi_a.conj() @ psi_b) ** 2)


def discrimination_matrix(states: list[np.ndarray]) -> np.ndarray:
    """Pairwise overlap matrix M[i, j] = |⟨ψ_i | ψ_j⟩|²."""
    K = len(states)
    M = np.zeros((K, K))
    for i in range(K):
        for j in range(i, K):
            M[i, j] = swap_test_overlap(states[i], states[j])
            M[j, i] = M[i, j]
    return M


# ----------------------------------------------------------------------------
# V1-style quantum filtering
# ----------------------------------------------------------------------------

@dataclass
class QuantumV1Filter:
    """A learnable unitary that filters orientation features.

    The filter is parameterized as
        U(theta) = product of single-qubit RY(θ_q) gates,
    one per qubit. Training adjusts θ to maximize overlap with a
    target "feature" state.
    """
    n_qubits: int
    theta: np.ndarray = field(default=None, repr=False)
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(0), repr=False)

    def __post_init__(self) -> None:
        if self.theta is None:
            self.theta = self.rng.uniform(0, 2 * np.pi, self.n_qubits)

    def unitary(self) -> np.ndarray:
        """Build the full 2^n × 2^n filter unitary."""
        U = np.array([[1.0]], dtype=np.complex128)
        for q in range(self.n_qubits):
            th = self.theta[q]
            c, s = np.cos(th / 2), np.sin(th / 2)
            R = np.array([[c, -s], [s, c]], dtype=np.complex128)
            U = np.kron(U, R)
        return U

    def filter_state(self, psi: np.ndarray) -> np.ndarray:
        return self.unitary() @ psi

    def fit_to_target(self, target: np.ndarray, lr: float = 0.05,
                       n_iter: int = 200) -> list:
        """Train theta to maximize |⟨target | U|0⟩|²."""
        psi0 = np.zeros(2 ** self.n_qubits, dtype=np.complex128); psi0[0] = 1.0
        losses = []
        for _ in range(n_iter):
            psi = self.filter_state(psi0)
            losses.append(1.0 - swap_test_overlap(psi, target))
            # Finite-diff gradient on θ.
            grad = np.zeros_like(self.theta)
            eps = 1e-3
            for q in range(self.n_qubits):
                self.theta[q] += eps
                f_plus = -swap_test_overlap(self.filter_state(psi0), target)
                self.theta[q] -= 2 * eps
                f_minus = -swap_test_overlap(self.filter_state(psi0), target)
                self.theta[q] += eps
                grad[q] = (f_plus - f_minus) / (2 * eps)
            self.theta -= lr * grad
        return losses


# ----------------------------------------------------------------------------
# End-to-end: retina → quantum encoding → V1 filter → discrimination
# ----------------------------------------------------------------------------

def encode_retinal_output(image: np.ndarray,
                            retina: Retina | None = None,
                            mode: str = "amplitude") -> np.ndarray:
    """Run retina on `image`, then quantum-encode the ON-channel response."""
    if retina is None:
        retina = Retina(sigma_center=1.0, sigma_surround=3.0, on_off=True)
    r_out = retina(image)
    on_channel = r_out["on"]
    if mode == "amplitude":
        return amplitude_encode(on_channel)
    if mode == "angle":
        # Smaller images only.
        return angle_encode(on_channel)
    if mode == "basis":
        return basis_encode((on_channel > on_channel.mean()).astype(int))
    raise ValueError(f"unknown mode: {mode}")
