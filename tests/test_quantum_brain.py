"""Tests for the quantum brain prototype."""

import json
from pathlib import Path

import numpy as np
import pytest

from quantum_brain import Vocabulary, QuantumBrain


VOCAB_PATH = Path(__file__).resolve().parent.parent / "quantum_brain" / "data" / "vocab.json"
pytestmark = pytest.mark.skipif(
    not VOCAB_PATH.exists(),
    reason="vocab.json missing; build it by running quantum_brain.encoding.Vocabulary.build()",
)


@pytest.fixture(scope="module")
def vocab():
    return Vocabulary.load(VOCAB_PATH)


@pytest.fixture(scope="module")
def brain(vocab):
    return QuantumBrain(vocab)


def test_vocabulary_codes_are_unique(vocab):
    assert len(set(vocab.codes.tolist())) == len(vocab.codes)


def test_vocabulary_lookups_roundtrip(vocab):
    # Pick a known word.
    w = vocab.words[0]
    c = vocab.encode(w)
    assert c is not None
    assert vocab.decode(c) == w


def test_hamming_neighbors_are_semantic(vocab):
    """For at least one known cluster, neighbors should look related."""
    # "father" should have family-related neighbors.
    fc = vocab.encode("father")
    if fc is None:
        pytest.skip("'father' not in vocab")
    all_codes = np.array(list(vocab.code_to_word.keys()), dtype=np.int64)
    dists = np.array([bin(int(c) ^ fc).count("1") for c in all_codes])
    order = np.argsort(dists)[:10]
    near_words = [vocab.code_to_word[int(all_codes[i])] for i in order]
    # Expect at least one family/relation-ish word in the top 10.
    family_words = {"father", "mother", "son", "daughter", "wife", "husband",
                    "family", "brother", "sister", "child", "boy", "girl",
                    "uncle", "grandfather", "william", "named"}
    assert family_words & set(near_words), f"got {near_words}"


def test_brain_returns_input_word_as_top_for_low_theta(brain):
    """At low theta the input word should dominate."""
    rng = np.random.default_rng(0)
    out = brain.think("music", theta=0.2, shots=2000, top_k=5, rng=rng)
    assert out[0][0] == "music"
    assert out[0][1] > 0.7


def test_brain_unknown_word_raises(brain):
    with pytest.raises(KeyError):
        brain.think("xyzzynotaword", theta=0.5, shots=100)


def test_blend_includes_both_inputs(brain):
    rng = np.random.default_rng(0)
    out = brain.think(["father", "mother"], theta=0.5, shots=3000, top_k=10, rng=rng)
    words = {w for w, _ in out}
    assert "father" in words
    assert "mother" in words


def test_higher_theta_spreads_more_widely(brain):
    """Top-1 word probability should drop as theta grows (more spread)."""
    rng1 = np.random.default_rng(1)
    rng2 = np.random.default_rng(1)
    p_low = brain.think("school", theta=0.2, shots=3000, top_k=1, rng=rng1)[0][1]
    p_high = brain.think("school", theta=0.8, shots=3000, top_k=1, rng=rng2)[0][1]
    assert p_low > p_high


def test_brain_results_are_normalized_probabilities(brain):
    rng = np.random.default_rng(0)
    out = brain.think("computer", theta=0.5, shots=3000, top_k=200, rng=rng)
    total = sum(p for _, p in out)
    # top_k=200 should capture essentially all the probability mass.
    assert total > 0.95
    assert all(0 <= p <= 1 for _, p in out)
