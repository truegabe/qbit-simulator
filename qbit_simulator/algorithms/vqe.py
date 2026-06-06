"""Variational Quantum Eigensolver (VQE).

Implements the standard VQE loop:
  1. Prepare an ansatz state |ψ(θ)⟩ via a parameterized circuit.
  2. Compute energy E(θ) = ⟨ψ(θ)|H|ψ(θ)⟩.
  3. Classically minimize E(θ).

Includes:
  - Hardware-efficient H₂ ansatz (one parameter): |ψ(θ)⟩ = cos(θ/2)|01⟩ + sin(θ/2)|10⟩
  - Golden-section search for 1D minimization
  - Nelder-Mead for multi-parameter minimization
  - param_shift_gradient: exact gradient via the parameter-shift rule
  - vqe_gradient: gradient-based VQE using L-BFGS-B (faster than Nelder-Mead)
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..circuit import QuantumCircuit
from ..pauli import PauliOp


# ---- ansatz ----

def h2_ansatz(theta: float) -> QuantumCircuit:
    """One-parameter ansatz for the Hubbard-dimer H₂ model.

    Produces cos(θ/2)|00⟩ + sin(θ/2)|01⟩, which spans the {|cov⟩, |ionic_+⟩}
    subspace where the ground state lives in our basis ordering.
    """
    qc = QuantumCircuit(2)
    qc.ry(theta, 1)
    return qc


# ---- 1D minimization ----

def golden_section(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-7,
    max_iter: int = 200,
) -> tuple[float, float]:
    """Find x in [a, b] minimizing f. Returns (x, f(x))."""
    phi = (1 + 5**0.5) / 2
    invphi = 1 / phi
    c = b - (b - a) * invphi
    d = a + (b - a) * invphi
    fc, fd = f(c), f(d)
    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - (b - a) * invphi
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + (b - a) * invphi
            fd = f(d)
    x = (a + b) / 2
    return x, f(x)


# ---- multi-D minimization (Nelder-Mead) ----

def nelder_mead(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    step: float = 0.5,
    tol: float = 1e-8,
    max_iter: int = 500,
) -> tuple[np.ndarray, float]:
    x0 = np.asarray(x0, dtype=np.float64)
    n = len(x0)
    simplex = [x0.copy()]
    for i in range(n):
        v = x0.copy()
        v[i] += step
        simplex.append(v)
    fvals = [f(v) for v in simplex]

    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    for _ in range(max_iter):
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = [fvals[i] for i in order]
        if max(np.linalg.norm(simplex[i] - simplex[0]) for i in range(1, n + 1)) < tol:
            break
        centroid = np.mean(simplex[:-1], axis=0)
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = f(xr)
        if fvals[0] <= fr < fvals[-2]:
            simplex[-1], fvals[-1] = xr, fr
            continue
        if fr < fvals[0]:
            xe = centroid + gamma * (xr - centroid)
            fe = f(xe)
            simplex[-1], fvals[-1] = (xe, fe) if fe < fr else (xr, fr)
            continue
        xc = centroid + rho * (simplex[-1] - centroid)
        fc = f(xc)
        if fc < fvals[-1]:
            simplex[-1], fvals[-1] = xc, fc
            continue
        # Shrink
        for i in range(1, n + 1):
            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
            fvals[i] = f(simplex[i])
    return simplex[0], fvals[0]


# ---- Parameter-shift gradient ----

def param_shift_gradient(
    hamiltonian: PauliOp,
    ansatz: Callable[..., QuantumCircuit],
    theta: np.ndarray,
) -> np.ndarray:
    """Exact gradient of ⟨H⟩ via the parameter-shift rule.

    For each parameter θ_i:
        dE/dθ_i = [E(θ + π/2 · eᵢ) - E(θ - π/2 · eᵢ)] / 2

    This is exact (not finite-difference) for gates of the form exp(-i θ/2 P)
    where P is a Pauli operator — which includes Rx, Ry, Rz, and all
    hardware-native parameterized gates.

    Args:
        hamiltonian: target PauliOp.
        ansatz:      callable (theta_array) -> QuantumCircuit or callable
                     (*theta_scalars) -> QuantumCircuit. Both forms accepted.
        theta:       current parameter vector (1-D array of floats).

    Returns:
        grad: array of shape (len(theta),) with ∂E/∂θ_i at each index.
    """
    theta = np.asarray(theta, dtype=np.float64)
    n_params = len(theta)
    grad = np.zeros(n_params)

    def _energy(t: np.ndarray) -> float:
        try:
            qc = ansatz(t)          # array form: ansatz(theta_array)
        except TypeError:
            qc = ansatz(*t)         # unpacked form: ansatz(t0, t1, ...)
        return hamiltonian.expectation(qc.state)

    for i in range(n_params):
        shift = np.zeros(n_params)
        shift[i] = np.pi / 2
        e_plus  = _energy(theta + shift)
        e_minus = _energy(theta - shift)
        grad[i] = (e_plus - e_minus) / 2.0

    return grad


def vqe_gradient(
    hamiltonian: PauliOp,
    ansatz: Callable[..., QuantumCircuit],
    theta0: np.ndarray,
    tol: float = 1e-8,
    max_iter: int = 500,
) -> tuple[np.ndarray, float, list[float]]:
    """VQE using L-BFGS-B with exact parameter-shift gradients.

    Typically 10-50× fewer energy evaluations than Nelder-Mead for the
    same convergence tolerance.

    Args:
        hamiltonian: target PauliOp.
        ansatz:      callable (*theta_scalars or theta_array) -> QuantumCircuit.
        theta0:      initial parameter vector.
        tol:         gradient norm tolerance for convergence.
        max_iter:    maximum optimizer iterations.

    Returns:
        (optimal_theta, ground_energy, energy_trace)
    """
    from scipy.optimize import minimize

    theta0 = np.asarray(theta0, dtype=np.float64)
    trace: list[float] = []

    def objective(t: np.ndarray) -> tuple[float, np.ndarray]:
        try:
            qc = ansatz(t)          # array form
        except TypeError:
            qc = ansatz(*t)         # unpacked form
        e = hamiltonian.expectation(qc.state)
        trace.append(e)
        g = param_shift_gradient(hamiltonian, ansatz, t)
        return float(e), g

    result = minimize(
        objective,
        theta0,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "gtol": tol},
    )
    return result.x, float(result.fun), trace


# ---- VQE core ----

def vqe(
    hamiltonian: PauliOp,
    ansatz: Callable[..., QuantumCircuit],
    theta0: float | np.ndarray,
    bounds: tuple[float, float] | None = None,
) -> tuple[float | np.ndarray, float, list[float]]:
    """Run VQE on `hamiltonian` using `ansatz`.

    For 1-parameter ansatze pass a float `theta0` and (optional) `bounds=(a,b)`
    to use golden-section search; otherwise pass an array `theta0` to use
    Nelder-Mead.

    Returns (optimal_theta, ground_energy, energy_trace).
    """
    trace: list[float] = []

    def energy_scalar(theta: float) -> float:
        qc = ansatz(theta)
        e = hamiltonian.expectation(qc.state)
        trace.append(e)
        return e

    def energy_vector(theta: np.ndarray) -> float:
        qc = ansatz(*theta) if isinstance(theta, np.ndarray) else ansatz(theta)
        e = hamiltonian.expectation(qc.state)
        trace.append(e)
        return e

    if isinstance(theta0, (float, int)):
        a, b = bounds if bounds is not None else (-np.pi, np.pi)
        theta_opt, e_opt = golden_section(energy_scalar, a, b)
        return theta_opt, e_opt, trace
    else:
        theta_opt, e_opt = nelder_mead(energy_vector, np.asarray(theta0, dtype=np.float64))
        return theta_opt, e_opt, trace
