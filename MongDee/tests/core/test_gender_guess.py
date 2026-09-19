"""A person with ANY gender evidence is never shown as UNKNOWN: the smoother offers a sticky best guess."""
from __future__ import annotations

from core.attributes import GlobalPersonAttributeSmoother


def test_no_evidence_no_guess():
    s = GlobalPersonAttributeSmoother()
    assert s.guess("P1", now=1.0) is None


def test_one_sample_gives_a_guess_before_the_label_is_decided():
    s = GlobalPersonAttributeSmoother()
    s.add_sample("P1", "female", 0.9, now=1.0)
    assert s.get("P1", 1.0).status != "ok"                 # not decided yet ...
    gender, conf = s.guess("P1", now=1.0)                  # ... but not unknown either
    assert gender == "female" and 0.5 <= conf <= 0.99


def test_the_guess_is_sticky_and_only_flips_on_real_opposite_evidence():
    s = GlobalPersonAttributeSmoother()
    s.add_sample("P1", "female", 0.9, now=1.0)
    assert s.guess("P1", now=1.0)[0] == "female"
    s.add_sample("P1", "male", 0.55, now=1.1)              # a weak contrary frame must not flip it
    assert s.guess("P1", now=1.1)[0] == "female"
    for i in range(6):
        s.add_sample("P1", "male", 0.95, now=1.2 + i * 0.1)
    assert s.guess("P1", now=2.0)[0] == "male"


def test_a_stale_person_has_no_guess():
    s = GlobalPersonAttributeSmoother(stale_grace_sec=5.0)
    s.add_sample("P1", "male", 0.9, now=1.0)
    assert s.guess("P1", now=100.0) is None
