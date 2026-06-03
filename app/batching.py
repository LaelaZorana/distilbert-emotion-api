"""Dynamic micro-batching for higher throughput under concurrency.

Many small concurrent requests are the worst case for a transformer: each pays
the fixed per-call overhead alone. The batcher fixes that. Callers ``await
submit(text)``; the batcher collects everything that arrives within a short
window (or until ``max_batch_size`` is reached), runs ONE forward pass for the
whole group, then fans the results back out to each waiter.

Design notes:
* A single background task owns the queue, so there is no lock contention and
  exactly one inference runs at a time (a CPU model has no intra-op parallelism
  to gain from concurrent forward passes anyway).
* The model is synchronous and CPU-bound, so it runs in a thread via
  ``run_in_executor`` to avoid blocking the event loop.
* A failed batch propagates the exception to every waiter in that batch and the
  loop keeps going — one bad batch never wedges the service.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List

from app.classifier import Classifier, Distribution
from app.metrics import BATCH_SIZE, INFERENCE_LATENCY

logger = logging.getLogger(__name__)


@dataclass
class _Job:
    text: str
    future: "asyncio.Future[Distribution]" = field(default_factory=lambda: asyncio.get_event_loop().create_future())


class MicroBatcher:
    """Coalesces concurrent single-text predictions into batched forward passes."""

    def __init__(
        self,
        classifier: Classifier,
        max_batch_size: int = 64,
        max_delay_ms: float = 5.0,
    ) -> None:
        self._classifier = classifier
        self._max_batch_size = max(1, max_batch_size)
        self._max_delay = max(0.0, max_delay_ms) / 1000.0
        self._queue: "asyncio.Queue[_Job]" = asyncio.Queue()
        self._worker: asyncio.Task | None = None
        self._closing = False

    async def start(self) -> None:
        if self._worker is None:
            self._closing = False
            self._worker = asyncio.create_task(self._run(), name="micro-batcher")

    async def stop(self) -> None:
        self._closing = True
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None

    async def submit(self, text: str) -> Distribution:
        """Enqueue one text and await its probability distribution."""
        if self._closing:
            raise RuntimeError("batcher is shutting down")
        job = _Job(text=text)
        await self._queue.put(job)
        return await job.future

    async def submit_many(self, texts: List[str]) -> List[Distribution]:
        """Submit several texts concurrently; they share the batcher's batches."""
        return await asyncio.gather(*(self.submit(t) for t in texts))

    async def _collect_batch(self) -> List[_Job]:
        """Block for the first job, then drain up to the window / size limit."""
        first = await self._queue.get()
        batch = [first]
        if self._max_delay > 0:
            deadline = asyncio.get_event_loop().time() + self._max_delay
            while len(batch) < self._max_batch_size:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    break
                try:
                    job = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    batch.append(job)
                except asyncio.TimeoutError:
                    break
        # Opportunistically grab anything already queued without further waiting.
        while len(batch) < self._max_batch_size and not self._queue.empty():
            batch.append(self._queue.get_nowait())
        return batch

    async def _run(self) -> None:
        loop = asyncio.get_event_loop()
        while True:
            batch = await self._collect_batch()
            texts = [job.text for job in batch]
            BATCH_SIZE.observe(len(texts))
            try:
                with INFERENCE_LATENCY.time():
                    results: List[Distribution] = await loop.run_in_executor(
                        None, self._classifier.predict, texts
                    )
                for job, dist in zip(batch, results):
                    if not job.future.done():
                        job.future.set_result(dist)
            except Exception as exc:  # one bad batch must not kill the worker
                logger.exception("batch inference failed for %d items", len(texts))
                for job in batch:
                    if not job.future.done():
                        job.future.set_exception(exc)
