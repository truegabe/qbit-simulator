"""Quantum Phase Estimation.

Given a unitary U with eigenstate |ψ⟩ and eigenvalue e^{2πiφ}, estimate φ to
t bits of precision using t counting qubits + the eigenstate register.

Circuit:
  - Hadamard all counting qubits.
  - For k = 0 .. t-1: apply U^{2^k} controlled on counting qubit k.
  - Inverse QFT on the counting register.
  - Measure counting register; outcome / 2^t ≈ φ.
"""

from __future__ import annotations

import numpy as np

from ..circuit import QuantumCircuit


def inverse_qft(qc: QuantumCircuit, qubits: list[int]) -> None:
    """Inverse QFT on the listed qubits (in their natural ordering)."""
    n = len(qubits)
    # Reverse first
    for j in range(n // 2):
        qc.swap(qubits[j], qubits[n - 1 - j])
    for j in reversed(range(n)):
        for k in reversed(range(j + 1, n)):
            angle = -np.pi / (2 ** (k - j))
            qc.cp(angle, qubits[k], qubits[j])
        qc.h(qubits[j])


def phase_estimation(
    U: np.ndarray,
    eigenstate: np.ndarray,
    n_counting: int,
) -> QuantumCircuit:
    """Run QPE for unitary U on the given eigenstate.

    Args:
        U: 2^k x 2^k unitary matrix.
        eigenstate: length 2^k state vector for the eigenstate register.
        n_counting: number of counting qubits (precision = 1/2^n_counting).

    Returns:
        A QuantumCircuit with the counting qubits as qubits 0..n_counting-1
        and the eigenstate register on the remaining qubits.
    """
    k = int(np.log2(len(eigenstate)))
    if U.shape != (2**k, 2**k):
        raise ValueError(f"U shape {U.shape} doesn't match eigenstate length {len(eigenstate)}")

    total_qubits = n_counting + k
    qc = QuantumCircuit(total_qubits)

    # Load eigenstate into the lower register (qubits n_counting .. end).
    eig_state = np.zeros(2**total_qubits, dtype=np.complex128)
    # Place eigenstate amplitudes at indices where the counting register is 0.
    for j, amp in enumerate(eigenstate):
        eig_state[j] = amp
    qc.state = eig_state

    # Hadamard all counting qubits.
    for q in range(n_counting):
        qc.h(q)

    # Apply controlled U^{2^p} for each counting qubit.
    U_power = U.copy()
    for p in range(n_counting):
        # Controlled version of U_power: controlled on counting qubit (n_counting - 1 - p)
        # so that the binary expansion of the measured integer reads in standard order.
        control = n_counting - 1 - p
        targets = list(range(n_counting, total_qubits))
        _apply_controlled_unitary(qc, U_power, control, targets)
        U_power = U_power @ U_power  # square it: U -> U^2

    # Inverse QFT on counting register.
    inverse_qft(qc, list(range(n_counting)))
    return qc


def _apply_controlled_unitary(
    qc: QuantumCircuit,
    U: np.ndarray,
    control: int,
    targets: list[int],
) -> None:
    """Apply U to `targets` conditioned on `control` being |1⟩."""
    k = len(targets)
    dim = 2**k
    # Build the (k+1)-qubit controlled-U matrix: I on |0⟩_c block, U on |1⟩_c block.
    CU = np.eye(2 * dim, dtype=np.complex128)
    CU[dim:, dim:] = U
    qc.apply_unitary(CU, [control] + targets, check_unitary=False)


def phase_estimation_modexp(
    a: int,
    N: int,
    n_target: int,
    n_counting: int,
) -> QuantumCircuit:
    """QPE for the modular-multiplication unitary U_a |y⟩ = |ay mod N⟩.

    Instead of materializing U_a as a 2^k × 2^k matrix and repeatedly squaring
    it, then applying n_counting controlled (k+1)-qubit unitaries to a 2^total
    state vector, we use the closed form of the post-controlled-U state:

        |ψ⟩ = (1/√2^t) Σ_c |c⟩ ⊗ |a^c mod N⟩

    valid whenever the eigenstate register starts in |1⟩. We populate the
    state directly with a single O(2^t) pass, then apply the standard inverse
    QFT on the counting register.

    Speedup over the generic `phase_estimation(U_a, |1⟩, t)` path scales as
    2^k · t for the matrix-squaring elimination, plus a much larger factor
    for skipping the (k+1)-qubit controlled-unitary applications which would
    otherwise dominate at large N.
    """
    total = n_counting + n_target
    dim_target = 1 << n_target
    dim_count = 1 << n_counting

    qc = QuantumCircuit(total)
    state = np.zeros(1 << total, dtype=np.complex128)

    # a^c mod N table — built iteratively (one mul per c).
    norm = 1.0 / np.sqrt(dim_count)
    v = 1
    for c in range(dim_count):
        # idx layout: top n_counting bits are counting register (qubits 0..n_counting-1),
        # low n_target bits are eigenstate register. v = a^c mod N ∈ [0, N) ⊂ [0, dim_target).
        state[(c << n_target) | v] = norm
        v = (v * a) % N

    # Inverse QFT on the counting register, applied as an FFT along the
    # counting axis. The gate-circuit inverse QFT is mathematically
    # IQFT = fft / √2^t (the bit-reversal swaps in the circuit cancel with
    # NumPy's natural FFT ordering). This collapses O(t^2) two-qubit gates
    # on a 2^total state into a single in-place FFT.
    state_2d = state.reshape(dim_count, dim_target)
    state_2d = np.fft.fft(state_2d, axis=0) / np.sqrt(dim_count)
    qc.state = state_2d.reshape(-1)

    qc.history.append(f"QPE_modexp(a={a},N={N},t={n_counting})")
    qc.history.append("inverse_qft(fft)")
    return qc


# Optional GPU backend — used by phase_estimation_modexp_marginal when
# `backend="gpu"` (or "auto" + cupy is importable + a CUDA device exists).
try:
    import cupy as _cp                          # type: ignore
    _HAS_CUPY = _cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    _cp = None                                  # type: ignore
    _HAS_CUPY = False

# Optional multithreaded FFT — falls back to numpy's single-threaded
# pocketfft if scipy isn't importable. scipy.fft.fft(..., workers=-1) uses
# every core; on a 16-core box that's ~10× faster than np.fft on a single
# 2^16-point transform.
try:
    import scipy.fft as _spfft                  # type: ignore
    _HAS_SCIPY_FFT = True
except Exception:
    _spfft = None                               # type: ignore
    _HAS_SCIPY_FFT = False


def phase_estimation_modexp_marginal(
    a: int,
    N: int,
    n_target: int,
    n_counting: int,
    backend: str = "auto",
    batch_size: int = 64,
) -> np.ndarray:
    """Counting-register marginal P(c) for Shor QPE — sparse path.

    Mathematically equivalent to:
        qc = phase_estimation_modexp(a, N, n_target, n_counting)
        P  = qc.probabilities().reshape(2^t, 2^k).sum(axis=1)

    but it never materializes the dense 2^(t+k) state vector. Memory drops
    from O(2^(t+k)) to O(2^t · batch_size) — at N=3599 (t=16, k=12) that's
    a few hundred MB instead of 4 GB.

    Backends:
      "cpu"  — scipy.fft with workers=-1 (all cores) if scipy is importable,
               otherwise numpy.fft single-threaded.
      "gpu"  — CuPy cuFFT on the first CUDA device (requires `pip install
               cupy-cudaXXx`).
      "auto" — gpu if available, else parallel cpu.

    How it works:

    The pre-FFT state has only 2^t nonzero amplitudes, one per counting value
    c, sitting at column y = a^c mod N with amplitude 1/√2^t. Define indicator
    v_y[c'] = 1 iff a^{c'} ≡ y (mod N). Then

        P(c) = Σ_y |state_post[c, y]|²
             = (1/4^t) · Σ_{y ∈ image(f)} |FFT(v_y)[c]|²

    Instead of looping `r` times over y (one FFT per y), we batch
    `batch_size` indicator columns into a 2D array and let the FFT do `B`
    transforms in parallel along axis=0. This trades a constant amount of
    extra memory for both intra-FFT and inter-FFT parallelism.
    """
    t = n_counting
    T = 1 << t

    # Resolve backend.
    if backend == "auto":
        backend = "gpu" if _HAS_CUPY else "cpu"
    if backend == "gpu" and not _HAS_CUPY:
        raise RuntimeError("backend='gpu' but CuPy is not importable or no "
                           "CUDA device is visible")
    if backend not in ("cpu", "gpu"):
        raise ValueError(f"unknown backend {backend!r}")

    # f[c] = a^c mod N — built iteratively in pure Python (numpy modmul
    # offers no advantage at int64; the loop is ~50 ms for t=16).
    f = np.empty(T, dtype=np.int64)
    v = 1
    for c in range(T):
        f[c] = v
        v = (v * a) % N
    unique_ys = np.unique(f)

    if backend == "gpu":
        return _modexp_marginal_gpu(f, unique_ys, T, batch_size)
    return _modexp_marginal_cpu(f, unique_ys, T, batch_size)


def _modexp_marginal_cpu(f: np.ndarray, unique_ys: np.ndarray,
                         T: int, batch_size: int) -> np.ndarray:
    """CPU implementation — parallel streaming with ThreadPoolExecutor.

    Each worker handles ~1/n_threads of the unique-y values, doing one
    `np.fft.fft(v_y)` per y. NumPy releases the GIL inside FFT so threads
    truly run in parallel on different cores; this is dramatically faster
    than scipy's batched-FFT-with-workers approach at our problem sizes
    (small per-FFT, lots of FFTs) because batching forces the threading
    overhead at the wrong granularity.

    `batch_size` is reused here as "chunk per thread" — each worker pulls
    `batch_size` consecutive y's at a time to amortize Python overhead.
    """
    import os
    from concurrent.futures import ThreadPoolExecutor

    n_unique = len(unique_ys)
    n_threads = min(os.cpu_count() or 1, n_unique, 16)

    # Split unique_ys into n_threads contiguous slices.
    slice_size = (n_unique + n_threads - 1) // n_threads
    slices = [unique_ys[i * slice_size:(i + 1) * slice_size]
              for i in range(n_threads) if i * slice_size < n_unique]

    def _process_slice(ys_slice: np.ndarray) -> np.ndarray:
        local_P = np.zeros(T, dtype=np.float64)
        # Process `batch_size` y's at a time within the thread to amortize
        # Python loop overhead while keeping per-FFT memory small.
        for chunk_start in range(0, len(ys_slice), batch_size):
            chunk = ys_slice[chunk_start:chunk_start + batch_size]
            for y in chunk:
                v_y = (f == y).astype(np.complex128)
                V = np.fft.fft(v_y)
                local_P += V.real * V.real + V.imag * V.imag
        return local_P

    if n_threads == 1:
        P = _process_slice(unique_ys)
    else:
        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            partials = list(pool.map(_process_slice, slices))
        P = np.sum(partials, axis=0)

    P /= P.sum()
    return P


def _modexp_marginal_gpu(f: np.ndarray, unique_ys: np.ndarray,
                         T: int, batch_size: int) -> np.ndarray:
    """CuPy GPU implementation. Returns a NumPy array (transferred from GPU)."""
    f_g = _cp.asarray(f)
    ys_g = _cp.asarray(unique_ys)
    P = _cp.zeros(T, dtype=_cp.float64)
    n_unique = int(ys_g.size)

    for start in range(0, n_unique, batch_size):
        batch = ys_g[start:start + batch_size]
        M = (f_g[:, None] == batch[None, :]).astype(_cp.complex128)
        V = _cp.fft.fft(M, axis=0)
        P += _cp.sum(V.real * V.real + V.imag * V.imag, axis=1)
    P /= P.sum()
    return _cp.asnumpy(P)


def estimate_phase_from_state(qc: QuantumCircuit, n_counting: int) -> float:
    """Read the most likely phase from the counting register's marginal distribution."""
    probs = qc.probabilities()
    total = qc.n
    # Marginal over counting register: sum over eigenstate-register indices.
    n_eig = total - n_counting
    counting_probs = np.zeros(2**n_counting)
    for idx in range(2**total):
        # Top n_counting bits of idx are counting register (since counting qubits are 0..n_counting-1, MSB).
        c_idx = idx >> n_eig
        counting_probs[c_idx] += probs[idx]
    best = int(np.argmax(counting_probs))
    return best / (2**n_counting)
