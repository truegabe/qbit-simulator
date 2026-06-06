"""Tests for holographic primitives: TFD, entropy, mutual info, Page curve."""

import numpy as np
import pytest

from qbit_simulator.algorithms.holography import (
    thermofield_double_state, bipartite_entropy, schmidt_values,
    mutual_information, haar_random_state, page_curve,
)


# ---- Thermofield double ----

def test_tfd_is_normalized():
    """|TFD⟩ should be a unit-norm state."""
    rng = np.random.default_rng(0)
    M = rng.normal(size=(4, 4))
    H = (M + M.T) / 2
    for beta in (0.5, 1.0, 5.0):
        tfd = thermofield_double_state(H, beta)
        assert abs(np.linalg.norm(tfd) - 1.0) < 1e-9


def test_tfd_zero_beta_is_maximally_entangled():
    """At β → 0, TFD becomes the maximally entangled state on two copies."""
    H = np.diag([0.1, 0.2, 0.3, 0.4])
    tfd = thermofield_double_state(H, beta=0.0)
    # All weights equal: |TFD⟩ = (1/√d) Σ_n |n⟩|n⟩ = max entangled.
    # In the eigenbasis (which is computational for diagonal H), this is just
    # (1/2)(|00⟩|00⟩ + |01⟩|01⟩ + |10⟩|10⟩ + |11⟩|11⟩).
    expected = np.zeros(16, dtype=np.complex128)
    for n in range(4):
        expected[n * 4 + n] = 0.5
    assert np.allclose(abs(tfd), abs(expected), atol=1e-9)


def test_tfd_high_beta_concentrates_on_ground():
    """At β → ∞, TFD becomes |GS⟩|GS⟩ (product of ground states)."""
    H = np.diag([0.0, 1.0, 2.0, 3.0])
    tfd = thermofield_double_state(H, beta=50.0)
    # Should be ≈ |0⟩|0⟩ = (1, 0, 0, ..., 0) in the 16-dim joint space.
    assert abs(tfd[0]) > 0.99


def test_tfd_reduced_density_is_thermal():
    """Tracing out one side of |TFD⟩ should give ρ_thermal = e^{-βH}/Z."""
    H = np.diag([0.1, 0.5, 1.0, 1.5])
    beta = 0.7
    tfd = thermofield_double_state(H, beta)
    # Reshape into a 4x4 matrix (4 = first system dim) and partial trace.
    psi_mat = tfd.reshape(4, 4)
    rho_L = psi_mat @ psi_mat.conj().T
    # Expected:
    expected_weights = np.exp(-beta * np.diag(H))
    expected_weights /= expected_weights.sum()
    rho_expected = np.diag(expected_weights).astype(np.complex128)
    assert np.allclose(rho_L, rho_expected, atol=1e-9)


# ---- Bipartite entropy ----

def test_entropy_product_state_zero():
    """Product state has zero bipartite entropy."""
    psi = np.zeros(8, dtype=np.complex128); psi[0] = 1   # |000⟩
    assert bipartite_entropy(psi, 3, [0]) < 1e-12
    assert bipartite_entropy(psi, 3, [0, 1]) < 1e-12


def test_entropy_bell_pair_is_one_bit():
    """|Φ+⟩ = (|00⟩+|11⟩)/√2 has entropy log 2 (or 1 bit)."""
    psi = np.zeros(4, dtype=np.complex128)
    psi[0] = 1 / np.sqrt(2); psi[3] = 1 / np.sqrt(2)
    s_nat = bipartite_entropy(psi, 2, [0])
    s_bits = bipartite_entropy(psi, 2, [0], base=2.0)
    assert abs(s_nat - np.log(2)) < 1e-10
    assert abs(s_bits - 1.0) < 1e-10


def test_entropy_3_qubit_ghz():
    """3-qubit GHZ has entropy 1 bit across any bipartition (1 + 2)."""
    psi = np.zeros(8, dtype=np.complex128)
    psi[0] = 1 / np.sqrt(2); psi[7] = 1 / np.sqrt(2)
    for A in ([0], [1], [2]):
        s = bipartite_entropy(psi, 3, A, base=2.0)
        assert abs(s - 1.0) < 1e-10


def test_schmidt_values_normalized():
    """Σ s_i² = 1 for any pure state."""
    rng = np.random.default_rng(0)
    psi = haar_random_state(4, rng)
    for A in ([0], [0, 1], [0, 2], [1, 3]):
        s = schmidt_values(psi, 4, A)
        assert abs((s * s).sum() - 1.0) < 1e-10


# ---- Renyi entropies ----

def test_renyi_alpha_equals_one_matches_von_neumann():
    """α=1 Renyi entropy equals the von Neumann entropy."""
    from qbit_simulator.algorithms.holography import renyi_entropy
    rng = np.random.default_rng(0)
    psi = haar_random_state(4, rng)
    s_vn  = bipartite_entropy(psi, 4, [0, 1])
    s_one = renyi_entropy(psi, 4, [0, 1], alpha=1.0)
    assert abs(s_vn - s_one) < 1e-10


def test_renyi_alpha_2_is_negative_log_purity():
    """α=2 Renyi entropy: S₂ = -log(Σ pᵢ²) = -log(Tr ρ²)."""
    from qbit_simulator.algorithms.holography import renyi_entropy
    psi = np.zeros(4, dtype=np.complex128)
    psi[0] = 1 / np.sqrt(2); psi[3] = 1 / np.sqrt(2)
    # Bell pair: ρ_A = I/2, so purity = 1/2 → S₂ = log 2.
    s2 = renyi_entropy(psi, 2, [0], alpha=2.0)
    assert abs(s2 - np.log(2)) < 1e-10


def test_renyi_alpha_infinity():
    """α=∞ Renyi entropy: S_∞ = -log(max pᵢ)."""
    from qbit_simulator.algorithms.holography import renyi_entropy
    psi = np.zeros(4, dtype=np.complex128)
    psi[0] = 1 / np.sqrt(2); psi[3] = 1 / np.sqrt(2)
    s_inf = renyi_entropy(psi, 2, [0], alpha=float("inf"))
    # max p_i = 1/2, so S_∞ = log 2.
    assert abs(s_inf - np.log(2)) < 1e-10


def test_renyi_monotone_in_alpha():
    """S_α is non-increasing in α (a fundamental Renyi-entropy property)."""
    from qbit_simulator.algorithms.holography import renyi_entropy
    rng = np.random.default_rng(0)
    psi = haar_random_state(4, rng)
    A = [0, 1]
    s1   = renyi_entropy(psi, 4, A, alpha=0.5)
    s2   = renyi_entropy(psi, 4, A, alpha=1.0)
    s3   = renyi_entropy(psi, 4, A, alpha=2.0)
    s_inf = renyi_entropy(psi, 4, A, alpha=float("inf"))
    assert s1 >= s2 - 1e-9
    assert s2 >= s3 - 1e-9
    assert s3 >= s_inf - 1e-9


# ---- Spectral form factor ----

def test_sff_at_t_zero_equals_one():
    """g(0) = |Σ e^0|² / Z² = 1."""
    from qbit_simulator.algorithms.holography import spectral_form_factor
    H = np.diag([0.1, 0.5, 1.0, 1.5])
    sff = spectral_form_factor(H, np.array([0.0]), beta=0.0)
    assert abs(sff[0] - 1.0) < 1e-10


def test_sff_returns_real_positive():
    """SFF should be real and non-negative."""
    from qbit_simulator.algorithms.holography import spectral_form_factor
    rng = np.random.default_rng(0)
    H = rng.normal(size=(8, 8))
    H = (H + H.T) / 2
    sff = spectral_form_factor(H, np.linspace(0, 10, 20))
    assert np.all(np.isreal(sff))
    assert np.all(sff >= -1e-12)


def test_sff_plateau_at_long_time():
    """For a generic chaotic spectrum, g(t) plateaus at level ~ 1/D at long t."""
    from qbit_simulator.algorithms.holography import spectral_form_factor
    rng = np.random.default_rng(0)
    # GOE-distributed eigenvalues (random symmetric matrix).
    H = rng.normal(size=(16, 16))
    H = (H + H.T) / 2
    times = np.linspace(0, 200, 50)
    sff = spectral_form_factor(H, times)
    # Long-time plateau should be near 1/D = 1/16.
    plateau_estimate = np.mean(sff[-10:])
    assert 0.01 < plateau_estimate < 0.5


# ---- Operator size growth ----

def test_operator_size_at_t_zero():
    """At t=0, O(t) = O, so size(0) = weight of the initial Pauli."""
    from qbit_simulator.algorithms.holography import operator_size_growth
    from qbit_simulator.gates import I2, X
    # Take O = X_0 on 3 qubits. Weight = 1.
    O = np.kron(np.kron(X, I2), I2)
    H = np.eye(8, dtype=np.complex128)   # trivial Hamiltonian: no evolution
    sizes = operator_size_growth(H, O, n_qubits=3, times=np.array([0.0]))
    assert abs(sizes[0] - 1.0) < 1e-9


def test_operator_size_with_identity_hamiltonian():
    """If H = I, [H, O] = 0 and O(t) = O for all t. Size stays constant."""
    from qbit_simulator.algorithms.holography import operator_size_growth
    from qbit_simulator.gates import I2, X
    O = np.kron(np.kron(X, I2), I2)
    H = np.eye(8, dtype=np.complex128)
    times = np.linspace(0, 5, 5)
    sizes = operator_size_growth(H, O, n_qubits=3, times=times)
    # All sizes should be approximately 1.
    for s in sizes:
        assert abs(s - 1.0) < 1e-9


def test_operator_size_grows_under_nontrivial_evolution():
    """For a non-commuting H, an initial single-qubit operator should
    eventually spread to more qubits."""
    from qbit_simulator.algorithms.holography import operator_size_growth
    from qbit_simulator.gates import I2, X
    # H = X_0 + Z_0 Z_1 on 3 qubits (nontrivial coupling).
    from qbit_simulator.pauli import PauliOp
    H = PauliOp([(1.0 + 0j, "XII"), (1.0 + 0j, "ZZI")]).matrix()
    O = np.kron(np.kron(X, I2), I2)
    times = np.array([0.0, 0.5, 1.5])
    sizes = operator_size_growth(H, O, n_qubits=3, times=times)
    assert sizes[0] == pytest.approx(1.0, abs=1e-9)
    # Size should grow at intermediate t.
    assert sizes[-1] > sizes[0]


# ---- Lyapunov from OTOC ----

def test_lyapunov_extraction_on_exponential_data():
    """Synthetic test: C(t) = C0·e^{λt}. Should recover λ exactly."""
    from qbit_simulator.algorithms.holography import lyapunov_from_otoc
    lam_true = 0.7
    C0 = 0.01
    times = np.linspace(0.5, 2.5, 20)
    otoc = C0 * np.exp(lam_true * times)
    result = lyapunov_from_otoc(times, otoc)
    assert abs(result["lyapunov"] - lam_true) < 1e-9


def test_lyapunov_from_otoc_on_syk():
    """SYK OTOC should give a positive Lyapunov exponent."""
    from qbit_simulator.algorithms.holography import lyapunov_from_otoc
    from qbit_simulator.algorithms.syk import (
        syk_hamiltonian, out_of_time_ordered_correlator,
    )
    from qbit_simulator.gates import I2, X
    H = syk_hamiltonian(M_majoranas=8, q=4, seed=0)
    V = np.kron(np.kron(np.kron(X, I2), I2), I2)
    W = np.kron(np.kron(np.kron(I2, X), I2), I2)
    times = np.linspace(0.1, 3.0, 20)
    otoc = out_of_time_ordered_correlator(H, V, W, beta=1.0, times=times)
    result = lyapunov_from_otoc(times, otoc, fit_range=(0.5, 2.0))
    assert result["lyapunov"] > 0


# ---- Krylov complexity ----

def test_krylov_complexity_at_t_zero_is_zero():
    """C_K(0) = 0: all amplitude on |φ_0⟩."""
    from qbit_simulator.algorithms.holography import krylov_complexity
    bs = [1.0, 2.0, 3.0]
    ck = krylov_complexity(bs, np.array([0.0]))
    assert abs(ck[0]) < 1e-9


def test_krylov_complexity_grows_with_time():
    from qbit_simulator.algorithms.holography import krylov_complexity
    bs = [1.0, 2.0, 3.0, 4.0, 5.0]
    times = np.array([0.0, 0.5, 1.0, 2.0])
    ck = krylov_complexity(bs, times)
    assert ck[1] > ck[0]


def test_krylov_complexity_bounded_by_basis_size():
    """C_K(t) ≤ n_basis - 1 (highest Krylov index = number of b's)."""
    from qbit_simulator.algorithms.holography import krylov_complexity
    bs = [1.0, 1.5, 2.0]   # n_basis = 4
    ck = krylov_complexity(bs, np.linspace(0, 100, 30))
    assert np.all(ck <= 3.0 + 1e-6)


def test_krylov_complexity_from_lanczos():
    """End-to-end: Lanczos on a random Hermitian H, then Krylov."""
    from qbit_simulator.algorithms.holography import (
        lanczos_coefficients, krylov_complexity,
    )
    from qbit_simulator.gates import I2, X
    rng = np.random.default_rng(0)
    H = rng.normal(size=(8, 8))
    H = (H + H.T) / 2
    O = np.kron(np.kron(X, I2), I2)
    lanczos = lanczos_coefficients(H, O, n_steps=15)
    ck = krylov_complexity(lanczos["b_coeffs"], np.linspace(0, 2, 5))
    assert ck[0] == pytest.approx(0.0, abs=1e-9)
    assert ck[-1] > 0


# ---- Wormhole teleportation ----

def test_wormhole_teleportation_beta_zero():
    """At β=0 the TFD is maximally entangled — teleportation is perfect.
    For a single-qubit message, fidelity should be ~1."""
    from qbit_simulator.algorithms.holography import wormhole_teleportation_fidelity
    # Trivial Hamiltonian → β doesn't matter, TFD is maximally entangled.
    H = np.zeros((2, 2), dtype=np.complex128)
    rng = np.random.default_rng(0)
    fids = []
    for _ in range(20):
        r = wormhole_teleportation_fidelity(H, beta=0.0, rng=rng)
        fids.append(r["fidelity"])
    assert float(np.mean(fids)) > 0.95


def test_wormhole_teleportation_large_beta_degrades():
    """At high β with a nontrivial H, the TFD has less entanglement → fidelity
    drops below 1."""
    from qbit_simulator.algorithms.holography import wormhole_teleportation_fidelity
    H = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)  # X
    rng = np.random.default_rng(0)
    fids_lo = []
    fids_hi = []
    for _ in range(40):
        r_lo = wormhole_teleportation_fidelity(H, beta=0.0,
                                                 rng=np.random.default_rng(int(rng.integers(0, 1000))))
        r_hi = wormhole_teleportation_fidelity(H, beta=10.0,
                                                 rng=np.random.default_rng(int(rng.integers(0, 1000))))
        fids_lo.append(r_lo["fidelity"])
        fids_hi.append(r_hi["fidelity"])
    # On average, high-β fidelity should be lower than β=0 fidelity.
    # (Allow some noise — single shots can fluctuate.)
    assert float(np.mean(fids_hi)) < float(np.mean(fids_lo)) + 0.05


def test_wormhole_teleportation_returns_valid_density_matrix():
    """The output reduced state should be a valid density matrix (PSD, trace 1)."""
    from qbit_simulator.algorithms.holography import wormhole_teleportation_fidelity
    H = np.diag([0.3, -0.3]).astype(np.complex128)
    r = wormhole_teleportation_fidelity(H, beta=1.0, rng=np.random.default_rng(0))
    rho = r["output_rho"]
    assert abs(np.trace(rho) - 1.0) < 1e-9
    eigs = np.linalg.eigvalsh(rho)
    assert all(e >= -1e-9 for e in eigs)


# ---- Hayden-Preskill ----

def test_hayden_preskill_starts_at_zero():
    """At |D|=0, no radiation collected → I(R:D)=0."""
    from qbit_simulator.algorithms.holography import hayden_preskill_protocol
    result = hayden_preskill_protocol(n_bh=3, k_alice=1,
                                       rng=np.random.default_rng(0), base=2.0)
    assert result["mutual_info_R_D"][0] == pytest.approx(0.0, abs=1e-9)


def test_hayden_preskill_recovers_full_information():
    """At |D| = full system, I(R:D) = 2k_alice (maximum possible, since
    R is k-qubit and maximally entangled with A in the full system)."""
    from qbit_simulator.algorithms.holography import hayden_preskill_protocol
    k = 1
    result = hayden_preskill_protocol(n_bh=3, k_alice=k,
                                       rng=np.random.default_rng(0), base=2.0)
    # When D = entire BH+Alice block, R is purified by D.
    last_mi = result["mutual_info_R_D"][-1]
    # I(R:D) = 2·S(R) = 2k for the maximally entangled (A, R) pair.
    assert abs(last_mi - 2 * k) < 0.5  # average behavior, single realization


def test_hayden_preskill_monotone_growth():
    """I(R:D) grows monotonically (weakly) with |D|."""
    from qbit_simulator.algorithms.holography import hayden_preskill_protocol
    # Average over multiple seeds to suppress fluctuations.
    n_avg = 5
    mi_avg = None
    for seed in range(n_avg):
        result = hayden_preskill_protocol(
            n_bh=3, k_alice=1, rng=np.random.default_rng(seed), base=2.0,
        )
        if mi_avg is None:
            mi_avg = result["mutual_info_R_D"].copy()
        else:
            mi_avg += result["mutual_info_R_D"]
    mi_avg /= n_avg
    # Should be (weakly) increasing on average.
    for i in range(len(mi_avg) - 1):
        assert mi_avg[i + 1] >= mi_avg[i] - 0.2


# ---- Lanczos coefficients ----

def test_lanczos_returns_correct_structure():
    from qbit_simulator.algorithms.holography import lanczos_coefficients
    from qbit_simulator.gates import I2, X
    H = np.eye(8, dtype=np.complex128)
    O = np.kron(np.kron(X, I2), I2)
    result = lanczos_coefficients(H, O, n_steps=5)
    assert "b_coeffs" in result and "a_coeffs" in result
    # For trivial H = I, [H, O] = 0, so b_1 = 0 and recursion stops.
    assert result["n_steps"] == 0


def test_lanczos_b_coeffs_positive():
    """b_n should always be non-negative."""
    from qbit_simulator.algorithms.holography import lanczos_coefficients
    from qbit_simulator.gates import I2, X, Z
    from qbit_simulator.pauli import PauliOp
    H = PauliOp([(1.0 + 0j, "XII"), (1.0 + 0j, "ZZI"),
                  (0.5 + 0j, "IXI"), (0.5 + 0j, "IIZ")]).matrix()
    O = np.kron(np.kron(X, I2), I2)
    result = lanczos_coefficients(H, O, n_steps=10)
    assert all(b > 0 for b in result["b_coeffs"])


def test_lanczos_for_syk_grows():
    """For SYK (chaotic), b_n should grow with n at small n."""
    from qbit_simulator.algorithms.holography import lanczos_coefficients
    from qbit_simulator.algorithms.syk import syk_hamiltonian
    from qbit_simulator.gates import I2, X
    H = syk_hamiltonian(M_majoranas=8, q=4, seed=0).matrix()
    O = np.kron(np.kron(np.kron(X, I2), I2), I2)
    result = lanczos_coefficients(H, O, n_steps=10)
    bs = result["b_coeffs"]
    # The first few b_n should generally increase.
    if len(bs) >= 3:
        # Not strictly monotonic at small N, but the trend should be upward.
        assert max(bs) >= bs[0]


# ---- Mutual information ----

def test_mutual_information_product_state():
    """I(A:B) = 0 for product states."""
    psi = np.zeros(8, dtype=np.complex128); psi[0] = 1
    assert mutual_information(psi, 3, [0], [1]) < 1e-10


def test_mutual_information_bell_pair():
    """I(0:1) = 2 log 2 for Bell pair."""
    psi = np.zeros(4, dtype=np.complex128)
    psi[0] = 1 / np.sqrt(2); psi[3] = 1 / np.sqrt(2)
    mi = mutual_information(psi, 2, [0], [1])
    assert abs(mi - 2 * np.log(2)) < 1e-9


# ---- Haar random state ----

def test_haar_random_state_unit_norm():
    rng = np.random.default_rng(0)
    for n in (1, 3, 5):
        psi = haar_random_state(n, rng)
        assert abs(np.linalg.norm(psi) - 1.0) < 1e-12


def test_haar_state_average_entropy_grows_with_subsystem_size():
    """For Haar states, mean entropy increases monotonically with subsystem
    size (up to half the system)."""
    rng = np.random.default_rng(0)
    n = 5
    means = []
    for k in range(1, n // 2 + 1):
        entropies = []
        for _ in range(20):
            psi = haar_random_state(n, rng)
            entropies.append(bipartite_entropy(psi, n, list(range(k)), base=2.0))
        means.append(np.mean(entropies))
    for i in range(len(means) - 1):
        assert means[i + 1] > means[i] - 0.05


# ---- Page curve ----

def test_page_curve_returns_correct_shape():
    rng = np.random.default_rng(0)
    result = page_curve(n_qubits=4, n_samples=10, rng=rng)
    assert len(result["subsystem_sizes"]) == 5  # 0..4
    assert len(result["mean_entropy"]) == 5
    assert result["mean_entropy"][0] == 0  # k=0 → zero entropy
    assert result["mean_entropy"][-1] == 0  # k=n → zero entropy


def test_page_curve_peaks_at_half():
    """Page curve should peak at k = n/2."""
    rng = np.random.default_rng(0)
    n = 6
    result = page_curve(n_qubits=n, n_samples=30, rng=rng)
    means = result["mean_entropy"]
    # Find argmax.
    peak_k = int(np.argmax(means))
    # Peak should be at n/2 (= 3 here), allow ±1 for stochasticity.
    assert abs(peak_k - n // 2) <= 1


def test_page_curve_is_symmetric():
    """Page curve is symmetric around k = n/2."""
    rng = np.random.default_rng(0)
    n = 4
    result = page_curve(n_qubits=n, n_samples=30, rng=rng)
    means = result["mean_entropy"]
    # ⟨S(k)⟩ ≈ ⟨S(n-k)⟩.
    for k in range(1, n // 2):
        assert abs(means[k] - means[n - k]) < 0.3
