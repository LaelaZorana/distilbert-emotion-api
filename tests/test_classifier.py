"""Unit tests for the offline stub classifier and the loader."""
from __future__ import annotations

from app import LABELS
from app.classifier import StubClassifier, load_classifier
from app.config import Settings


def test_stub_returns_valid_distribution():
    clf = StubClassifier()
    (dist,) = clf.predict(["i feel so happy today"])
    assert set(dist) == set(LABELS)
    assert abs(sum(dist.values()) - 1.0) < 1e-6
    assert all(0.0 <= p <= 1.0 for p in dist.values())


def test_stub_is_deterministic():
    clf = StubClassifier()
    a = clf.predict(["the same sentence, every time"])[0]
    b = clf.predict(["the same sentence, every time"])[0]
    assert a == b


def test_stub_picks_lexicon_emotion():
    clf = StubClassifier()
    cases = {
        "i am terrified and my hands are shaking": "fear",
        "how dare they, i am absolutely furious": "anger",
        "i adore you, my beloved sweetheart": "love",
        "i feel so sad and lonely and hopeless": "sadness",
        "what a wonderful, delightful, fantastic day": "joy",
        "wow i am completely shocked and astonished": "surprise",
    }
    for text, expected in cases.items():
        dist = clf.predict([text])[0]
        top = max(dist, key=dist.get)
        assert top == expected, f"{text!r} -> {top}, expected {expected}"


def test_stub_batch_matches_singles():
    clf = StubClassifier()
    texts = ["i am scared", "i am joyful", "i am angry"]
    batched = clf.predict(texts)
    singles = [clf.predict([t])[0] for t in texts]
    assert batched == singles


def test_loader_returns_stub_when_offline():
    clf = load_classifier(Settings(offline=True))
    assert clf.backend == "stub"
