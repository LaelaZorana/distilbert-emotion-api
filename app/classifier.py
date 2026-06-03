"""Model loader abstraction: one interface, two backends.

The service never imports torch/transformers directly. It asks
:func:`load_classifier` for an object exposing ``predict(texts) -> list[dict]``,
where each dict maps every label to a probability that sums to 1.0.

Two implementations satisfy that contract:

* :class:`StubClassifier` — pure-Python, deterministic, zero downloads. Scores
  text with a small hand-built emotion lexicon so the distribution is plausible
  and *stable* (the same sentence always yields the same probabilities), which
  is what makes offline demos, tests, and load tests meaningful. Selected when
  ``OFFLINE=1`` (the default).
* :class:`TransformersClassifier` — the real fine-tuned DistilBERT loaded once
  via a Hugging Face ``pipeline``. Selected when ``OFFLINE=0``.

Both are warmed up on construction so the first real request is not slow.
"""
from __future__ import annotations

import logging
import math
import re
from typing import Dict, List, Protocol

from app import ID2LABEL, LABELS
from app.config import Settings

logger = logging.getLogger(__name__)

Distribution = Dict[str, float]


class Classifier(Protocol):
    """Anything the service can serve predictions from."""

    backend: str

    def predict(self, texts: List[str]) -> List[Distribution]:
        """Return one ``{label: probability}`` dict per input text."""
        ...


def _softmax(scores: List[float]) -> List[float]:
    hi = max(scores)
    exps = [math.exp(s - hi) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


# --- Offline stub -----------------------------------------------------------

# A compact emotion lexicon. Not meant to rival the real model — it exists so the
# offline path produces a deterministic, label-aware distribution instead of a
# constant, which keeps demos and tests honest. Weights are deliberately modest
# so the softmax stays smooth rather than collapsing to a one-hot vector.
_LEXICON: Dict[str, Dict[str, float]] = {
    "sadness": {
        "sad": 2.2, "unhappy": 2.0, "cry": 2.0, "crying": 2.0, "lonely": 2.0,
        "depressed": 2.4, "miserable": 2.2, "down": 1.4, "hopeless": 2.2,
        "lost": 1.6, "hurt": 1.6, "grief": 2.4, "disappointed": 1.8, "empty": 1.8,
    },
    "joy": {
        "happy": 2.2, "joy": 2.4, "glad": 2.0, "great": 1.6, "wonderful": 2.0,
        "delighted": 2.2, "excited": 1.8, "smiling": 2.0, "grateful": 1.6,
        "love": 0.6, "amazing": 1.8, "fantastic": 2.0, "relieved": 1.6, "thrilled": 2.2,
    },
    "love": {
        "love": 2.4, "adore": 2.4, "beloved": 2.2, "affection": 2.2, "tender": 2.0,
        "caring": 1.8, "sweetheart": 2.2, "cherish": 2.2, "fond": 1.8,
        "romantic": 2.0, "devoted": 2.0,
    },
    "anger": {
        "angry": 2.4, "mad": 2.0, "furious": 2.6, "rage": 2.6, "hate": 2.2,
        "annoyed": 1.8, "irritated": 1.8, "outraged": 2.4, "resent": 2.0,
        "disgusted": 1.8, "betrayed": 2.0, "unfair": 1.6,
    },
    "fear": {
        "afraid": 2.4, "scared": 2.4, "fear": 2.4, "terrified": 2.6, "anxious": 2.2,
        "nervous": 2.0, "worried": 2.0, "panic": 2.4, "dread": 2.2, "frightened": 2.4,
        "shaking": 1.8, "uneasy": 1.8,
    },
    "surprise": {
        "surprised": 2.4, "shocked": 2.2, "amazed": 2.0, "astonished": 2.4,
        "unexpected": 2.0, "wow": 1.8, "suddenly": 1.4, "stunned": 2.2,
        "speechless": 2.0, "unbelievable": 1.8,
    },
}

_WORD_RE = re.compile(r"[a-z']+")


class StubClassifier:
    """Deterministic, network-free classifier driven by an emotion lexicon."""

    backend = "stub"

    def __init__(self) -> None:
        # Slight positive prior on the two dominant training classes (joy,
        # sadness) so empty / unknown text falls back to a realistic shape
        # rather than a flat uniform distribution.
        self._prior = {lab: 0.0 for lab in LABELS}
        self._prior["joy"] = 0.30
        self._prior["sadness"] = 0.25
        logger.info("StubClassifier ready (offline, no weights loaded)")

    def _score_one(self, text: str) -> Distribution:
        scores = dict(self._prior)
        for token in _WORD_RE.findall(text.lower()):
            for label, lex in _LEXICON.items():
                w = lex.get(token)
                if w:
                    scores[label] += w
        ordered = [scores[ID2LABEL[i]] for i in range(len(LABELS))]
        probs = _softmax(ordered)
        return {ID2LABEL[i]: probs[i] for i in range(len(LABELS))}

    def predict(self, texts: List[str]) -> List[Distribution]:
        return [self._score_one(t) for t in texts]


# --- Real model -------------------------------------------------------------

class TransformersClassifier:
    """The fine-tuned DistilBERT, loaded once via a transformers pipeline."""

    backend = "transformers"

    def __init__(self, model_id: str) -> None:
        # Imported lazily so the dependency is only required for a real run.
        from transformers import pipeline  # type: ignore

        logger.info("Loading model %s via transformers pipeline ...", model_id)
        self._pipe = pipeline(
            "text-classification",
            model=model_id,
            top_k=None,  # return the full distribution, not just the argmax
        )
        # Warm up so the first served request does not pay graph-build cost.
        self._pipe("warmup")
        logger.info("Model %s loaded and warmed up", model_id)

    def predict(self, texts: List[str]) -> List[Distribution]:
        raw = self._pipe(list(texts))
        # pipeline returns list[list[{label, score}]] when top_k=None.
        out: List[Distribution] = []
        for row in raw:
            dist = {item["label"]: float(item["score"]) for item in row}
            # Guarantee every canonical label is present and ordered.
            out.append({lab: dist.get(lab, 0.0) for lab in LABELS})
        return out


def load_classifier(settings: Settings) -> Classifier:
    """Build the classifier the settings ask for.

    Falls back to the stub if a real load is requested but the ML stack is not
    installed, so the service still boots (degraded) instead of crash-looping.
    """
    if settings.offline:
        return StubClassifier()
    try:
        return TransformersClassifier(settings.model_id)
    except Exception:  # pragma: no cover - exercised only with the real stack
        logger.exception(
            "Failed to load real model %s; falling back to offline stub. "
            "Install the 'ml' extra and ensure the weights are reachable.",
            settings.model_id,
        )
        return StubClassifier()
