"""Tests for the micro-batcher's correctness and coalescing behavior.

These drive the async batcher with ``asyncio.run`` directly so no pytest-asyncio
plugin is required (keeps the test dependency surface minimal).
"""
from __future__ import annotations

import asyncio
import threading
from typing import List

import pytest

from app.batching import MicroBatcher
from app.classifier import Distribution, StubClassifier


class _CountingClassifier:
    """Wraps the stub and records how many forward passes (batches) it ran."""

    backend = "counting"

    def __init__(self) -> None:
        self._inner = StubClassifier()
        self.calls = 0
        self.max_batch = 0
        self._lock = threading.Lock()

    def predict(self, texts: List[str]) -> List[Distribution]:
        with self._lock:
            self.calls += 1
            self.max_batch = max(self.max_batch, len(texts))
        return self._inner.predict(texts)


def test_batcher_results_match_direct():
    async def scenario():
        clf = StubClassifier()
        batcher = MicroBatcher(clf, max_batch_size=8, max_delay_ms=5)
        await batcher.start()
        try:
            text = "i feel wonderful today"
            via_batcher = await batcher.submit(text)
            direct = clf.predict([text])[0]
            assert via_batcher == direct
        finally:
            await batcher.stop()

    asyncio.run(scenario())


def test_concurrent_requests_are_coalesced():
    async def scenario():
        clf = _CountingClassifier()
        batcher = MicroBatcher(clf, max_batch_size=32, max_delay_ms=20)
        await batcher.start()
        try:
            texts = [f"sentence number {i}" for i in range(20)]
            results = await asyncio.gather(*(batcher.submit(t) for t in texts))
            assert len(results) == 20
            # 20 concurrent submissions should collapse into far fewer forward
            # passes than 20 (ideally 1-2 given the 20ms window).
            assert clf.calls < 20
            assert clf.max_batch > 1
        finally:
            await batcher.stop()

    asyncio.run(scenario())


def test_batch_respects_max_size():
    async def scenario():
        clf = _CountingClassifier()
        batcher = MicroBatcher(clf, max_batch_size=4, max_delay_ms=50)
        await batcher.start()
        try:
            texts = [f"text {i}" for i in range(12)]
            await asyncio.gather(*(batcher.submit(t) for t in texts))
            assert clf.max_batch <= 4
        finally:
            await batcher.stop()

    asyncio.run(scenario())


def test_submit_after_stop_raises():
    async def scenario():
        clf = StubClassifier()
        batcher = MicroBatcher(clf)
        await batcher.start()
        await batcher.stop()
        with pytest.raises(RuntimeError):
            await batcher.submit("anything")

    asyncio.run(scenario())
