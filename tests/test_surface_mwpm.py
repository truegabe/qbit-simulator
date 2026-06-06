"""Surface-code MWPM decoder tests."""

import numpy as np
import pytest

from qbit_simulator.surface_mwpm import (
    SurfaceCodeD3, MWPMDecoder,
    SURFACE_D3_X_STABS, SURFACE_D3_Z_STABS,
    simulate_logical_error_rate, simulate_threshold,
)


@pytest.fixture(scope="module")
def code():
    return SurfaceCodeD3()


@pytest.fixture(scope="module")
def decoder(code):
    return MWPMDecoder(code)


# ---- Geometry ----

def test_code_has_9_data_qubits(code):
    assert code.n_data == 9


def test_code_has_4_x_and_4_z_stabilizers(code):
    assert len(code.x_stabilizers) == 4
    assert len(code.z_stabilizers) == 4


def test_x_stabs_commute_with_z_stabs(code):
    """All X-stabs must commute with all Z-stabs (CSS property)."""
    for x_sup in code.x_stabilizers.values():
        for z_sup in code.z_stabilizers.values():
            assert len(set(x_sup) & set(z_sup)) % 2 == 0


def test_logical_operators_anticommute(code):
    """Logical X · logical Z support overlap must be odd."""
    overlap = len(set(code.logical_x) & set(code.logical_z))
    assert overlap % 2 == 1


def test_logical_x_commutes_with_z_stabs(code):
    for z_sup in code.z_stabilizers.values():
        assert len(set(code.logical_x) & set(z_sup)) % 2 == 0


def test_logical_z_commutes_with_x_stabs(code):
    for x_sup in code.x_stabilizers.values():
        assert len(set(code.logical_z) & set(x_sup)) % 2 == 0


# ---- Syndrome computation ----

def test_no_error_no_syndrome(code):
    assert code.z_error_syndrome(set()) == frozenset()
    assert code.x_error_syndrome(set()) == frozenset()


def test_single_z_error_lights_correct_stabs(code):
    """Z error on q4 lights both bulk plaquettes (x2 and x3)."""
    syndrome = code.z_error_syndrome({4})
    assert syndrome == frozenset({"x2", "x3"})


def test_single_z_error_corner_lights_one_stab(code):
    """Z error on q0 lights only x0 (left boundary)."""
    assert code.z_error_syndrome({0}) == frozenset({"x0"})


# ---- Decoder correctness on single errors ----

@pytest.mark.parametrize("q", list(range(9)))
def test_decoder_corrects_single_z_error(code, decoder, q):
    """For any single Z error, the decoder should output a correction
    that, XOR'd with the error, produces a stabilizer (no logical error)."""
    z_err = {q}
    syndrome = code.z_error_syndrome(z_err)
    correction = decoder.decode_z_errors(syndrome)
    residual = z_err ^ correction
    # Residual must commute with all logical X (no logical Z applied).
    overlap = len(residual & set(code.logical_x)) % 2
    assert overlap == 0


@pytest.mark.parametrize("q", list(range(9)))
def test_decoder_corrects_single_x_error(code, decoder, q):
    x_err = {q}
    syndrome = code.x_error_syndrome(x_err)
    correction = decoder.decode_x_errors(syndrome)
    residual = x_err ^ correction
    overlap = len(residual & set(code.logical_z)) % 2
    assert overlap == 0


def test_decoder_handles_empty_syndrome(code, decoder):
    assert decoder.decode_z_errors(frozenset()) == set()
    assert decoder.decode_x_errors(frozenset()) == set()


# ---- Threshold sweep ----

def test_logical_error_rate_runs(code, decoder):
    rng = np.random.default_rng(0)
    r = simulate_logical_error_rate(code, decoder, p_physical=0.05,
                                      n_trials=200, rng=rng)
    assert 0.0 <= r["p_logical"] <= 1.0
    assert r["n_trials"] == 200


def test_low_physical_low_logical():
    """At p=0.005, logical error rate should be much lower than physical."""
    rng = np.random.default_rng(0)
    code = SurfaceCodeD3()
    decoder = MWPMDecoder(code)
    r = simulate_logical_error_rate(code, decoder, p_physical=0.005,
                                      n_trials=5000, rng=rng)
    # For d=3 at p=0.5%, expect p_L well under p_phys = 0.005.
    assert r["p_logical"] < 0.005


def test_high_physical_high_logical():
    """At p=0.2, logical error rate should be substantial."""
    rng = np.random.default_rng(0)
    code = SurfaceCodeD3()
    decoder = MWPMDecoder(code)
    r = simulate_logical_error_rate(code, decoder, p_physical=0.2,
                                      n_trials=1000, rng=rng)
    assert r["p_logical"] > 0.1


def test_threshold_curve_monotonic():
    """Logical error rate should be monotonically increasing in p_phys."""
    rng = np.random.default_rng(0)
    r = simulate_threshold([0.01, 0.05, 0.1, 0.15], n_trials=1500, rng=rng)
    # Allow small statistical noise: each step should be at most decreasing
    # by a small amount.
    diffs = np.diff(r["p_logical"])
    assert (diffs >= -0.02).all()
