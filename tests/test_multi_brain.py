"""Tests for the v2 MultiRegionBrain."""

from pathlib import Path

import numpy as np
import pytest

from quantum_brain import MultiRegionBrain


REGIONS_DIR = Path(__file__).resolve().parent.parent / "quantum_brain" / "data" / "regions"
pytestmark = pytest.mark.skipif(
    not REGIONS_DIR.exists(),
    reason="regions not built; run quantum_brain.regions.build_region_vocabs() first",
)


@pytest.fixture(scope="module")
def brain():
    return MultiRegionBrain.load_default()


def test_brain_loads_8_regions(brain):
    assert brain.n_regions == 8


def test_each_region_has_distinct_label(brain):
    assert len(set(brain.region_labels)) == brain.n_regions


def test_routing_is_semantically_appropriate(brain):
    cases = [
        ("money", ["profits", "sales", "revenue"]),
        ("football", ["finals", "playoff", "championship"]),
        ("classical", ["contemporary", "literature", "literary", "musical"]),
    ]
    for word, expected_label_terms in cases:
        routing = brain.agent.route_explain(word, top_k=1)
        assert routing, f"no routing for {word}"
        rid, label, _ = routing[0]
        assert any(t in label for t in expected_label_terms), \
            f"{word} routed to {label}, expected one of {expected_label_terms}"


def test_query_returns_input_as_top(brain):
    rng = np.random.default_rng(0)
    out = brain.think("football", theta=0.3, shots=2000, top_k_associations=5, rng=rng)
    assert out[0][0] == "football"


def test_blend_returns_both_inputs(brain):
    rng = np.random.default_rng(0)
    out = brain.think(["music", "football"], theta=0.5, shots=3000,
                      top_k_associations=10, rng=rng)
    words = {w for w, _ in out}
    assert "music" in words
    assert "football" in words


def test_trace_lists_activated_regions(brain):
    rng = np.random.default_rng(0)
    _, trace = brain.think("money", theta=0.5, shots=1000, rng=rng, return_trace=True)
    assert "assignments" in trace
    assert len(trace["assignments"]) >= 1
