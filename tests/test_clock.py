"""Tests for the biological clock."""

import json
import time
from pathlib import Path

import pytest

from quantum_brain import BiologicalClock


def test_new_clock_starts_awake(tmp_path):
    c = BiologicalClock(tmp_path / "clock.json")
    assert c.is_awake
    assert c.wake_count == 1
    assert c.sleep_count == 0


def test_age_increases_over_time(tmp_path):
    c = BiologicalClock(tmp_path / "clock.json")
    a1 = c.age_seconds()
    time.sleep(0.1)
    a2 = c.age_seconds()
    assert a2 > a1


def test_heartbeat_accumulates_awake_time(tmp_path):
    c = BiologicalClock(tmp_path / "clock.json")
    time.sleep(0.2)
    c.heartbeat()
    assert c.total_awake_seconds >= 0.2
    assert c.total_asleep_seconds == 0


def test_sleep_wake_cycle(tmp_path):
    c = BiologicalClock(tmp_path / "clock.json")
    time.sleep(0.1)
    c.go_to_sleep()
    assert not c.is_awake
    assert c.sleep_count == 1
    time.sleep(0.15)
    c.wake_up()
    assert c.is_awake
    assert c.wake_count == 2
    assert c.last_sleep_duration >= 0.15
    assert c.total_asleep_seconds >= 0.15


def test_clock_persists_across_instances(tmp_path):
    p = tmp_path / "clock.json"
    c1 = BiologicalClock(p)
    time.sleep(0.1)
    c1.heartbeat()
    c1.save()           # explicit flush for the test
    awake_before = c1.total_awake_seconds
    birth_before = c1.birth_time

    # Simulate reload.
    c2 = BiologicalClock(p)
    assert c2.birth_time == birth_before
    assert c2.total_awake_seconds >= awake_before


def test_clock_records_offline_as_asleep(tmp_path):
    p = tmp_path / "clock.json"
    c1 = BiologicalClock(p)
    c1.heartbeat()
    initial_asleep = c1.total_asleep_seconds

    # Manually advance "last_heartbeat" backwards to simulate downtime.
    c1.last_heartbeat -= 5.0
    c1.save()

    c2 = BiologicalClock(p)
    # The 5s gap should have been counted as sleep.
    assert c2.total_asleep_seconds >= initial_asleep + 4.5


def test_clock_report_returns_string(tmp_path):
    c = BiologicalClock(tmp_path / "clock.json")
    text = c.report()
    assert "AWAKE" in text or "ASLEEP" in text
    assert "Age" in text


def test_clock_format_duration_handles_minutes(tmp_path):
    from quantum_brain.clock import _format_duration
    assert "s" in _format_duration(30)
    assert "m" in _format_duration(65)
    assert "h" in _format_duration(3700)
    assert "d" in _format_duration(86400 * 2 + 100)


# Integration with the full brain
def test_brain_has_clock(tmp_path):
    from quantum_brain import QuantumBrainV3
    from quantum_brain.hierarchical import Hierarchy
    HIERARCHY_DIR = Path(__file__).resolve().parent.parent / "quantum_brain" / "data" / "hierarchy"
    if not HIERARCHY_DIR.exists():
        pytest.skip("no hierarchy")
    h = Hierarchy.load(HIERARCHY_DIR)
    brain = QuantumBrainV3(
        h,
        glove_path=HIERARCHY_DIR.parent / "glove50.txt",
        cache_path=HIERARCHY_DIR.parent / "glove_cache.npz",
        memory_path=tmp_path / "mem.json",
        concepts_path=tmp_path / "concepts.json",
        episodes_path=tmp_path / "ep.jsonl",
        clock_path=tmp_path / "clock.json",
    )
    assert brain.clock.is_awake


def test_dream_transitions_clock_through_sleep(tmp_path):
    """After dream(), the brain should have at least 1 sleep event recorded."""
    from quantum_brain import QuantumBrainV3, dream, brainstorm
    from quantum_brain.hierarchical import Hierarchy
    HIERARCHY_DIR = Path(__file__).resolve().parent.parent / "quantum_brain" / "data" / "hierarchy"
    if not HIERARCHY_DIR.exists():
        pytest.skip("no hierarchy")
    h = Hierarchy.load(HIERARCHY_DIR)
    brain = QuantumBrainV3(
        h,
        glove_path=HIERARCHY_DIR.parent / "glove50.txt",
        cache_path=HIERARCHY_DIR.parent / "glove_cache.npz",
        memory_path=tmp_path / "mem.json",
        concepts_path=tmp_path / "concepts.json",
        episodes_path=tmp_path / "ep.jsonl",
        clock_path=tmp_path / "clock.json",
    )
    # Need at least one episode so dream has something to replay.
    brainstorm(brain, "music", n_chains=2, steps_per_chain=3,
               concept_store=brain.concepts, top_k=5)
    sleep_count_before = brain.clock.sleep_count
    dream(brain, n_replays=2)
    assert brain.clock.sleep_count == sleep_count_before + 1
    assert brain.clock.is_awake  # back awake afterwards
