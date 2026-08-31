"""
worker/queue.py

Job queue abstraction.

WORKER_MODE=thread  → LocalThreadQueue (default)
    Jobs run in a ThreadPoolExecutor inside the same uvicorn process.
    Simple, zero infrastructure. Good for local use and low concurrency.

WORKER_MODE=queue   → AzureQueueStorage
    Jobs are enqueued as messages in Azure Queue Storage.
    A separate worker container (same Docker image, different CMD) pulls
    and processes them. Horizontally scalable.

Both implement the same interface so the API layer doesn't care which
mode is active.
"""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

from config import get_settings


# ── abstract interface ────────────────────────────────────────────────────────

class AbstractQueue(ABC):

    @abstractmethod
    async def enqueue(self, job_id: str) -> None:
        """Add a job_id to the queue."""

    @abstractmethod
    async def start(self, handler: Callable[[str], None]) -> None:
        """
        Start processing jobs.
        handler(job_id) is called for each job pulled from the queue.
        For thread mode this starts the thread pool.
        For queue mode this starts a polling loop (separate process).
        """

    @abstractmethod
    async def stop(self) -> None:
        """Graceful shutdown."""


# ── local thread queue ────────────────────────────────────────────────────────

class LocalThreadQueue(AbstractQueue):
    """
    Runs jobs in a ThreadPoolExecutor within the same process.

    Enqueue puts the job_id into an asyncio.Queue.
    A background task reads from the queue and submits to the thread pool.

    Pros:  zero infrastructure, simple
    Cons:  shares memory/CPU with the API, not horizontally scalable,
           jobs are lost if the process crashes (no persistence)
    For production: replace with AzureQueueStorage.
    """

    def __init__(self, max_workers: int = 2):
        self._queue:    asyncio.Queue = asyncio.Queue()
        self._executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=max_workers)
        self._handler:  Optional[Callable] = None
        self._task:     Optional[asyncio.Task] = None

    async def enqueue(self, job_id: str) -> None:
        await self._queue.put(job_id)

    async def start(self, handler: Callable[[str], None]) -> None:
        self._handler = handler
        self._task = asyncio.create_task(self._consume())

    async def _consume(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            job_id = await self._queue.get()
            if job_id is None:   # shutdown sentinel
                break
            # Run CPU-bound inference in thread pool so event loop stays free
            loop.run_in_executor(self._executor, self._handler, job_id)
            self._queue.task_done()

    async def stop(self) -> None:
        await self._queue.put(None)   # sentinel
        if self._task:
            await self._task
        self._executor.shutdown(wait=True)


# ── Azure Queue Storage ───────────────────────────────────────────────────────

class AzureQueueStorage(AbstractQueue):
    """
    Azure Queue Storage backend.

    Enqueue: sends job_id as a JSON message to Azure Queue Storage.
    start():  starts a polling loop — intended to run in a separate
              worker container, not in the API process.

    The API enqueues; the worker polls and processes.
    Both use the same job_id — the worker looks up the Job in the DB.

    Required env vars:
        AZURE_QUEUE_CONNECTION_STRING
        AZURE_QUEUE_NAME
        QUEUE_VISIBILITY_TIMEOUT_S   (how long worker has before message reappears)

    Install:
        pip install azure-storage-queue
    """

    def __init__(
        self,
        connection_string: str,
        queue_name: str,
        visibility_timeout_s: int = 300,
    ):
        try:
            from azure.storage.queue.aio import QueueServiceClient
        except ImportError:
            raise ImportError(
                "azure-storage-queue is required for queue mode. "
                "pip install azure-storage-queue"
            )
        self._conn_str           = connection_string
        self._queue_name         = queue_name
        self._visibility_timeout = visibility_timeout_s
        self._QueueServiceClient = QueueServiceClient
        self._running            = False

    def _client(self):
        return self._QueueServiceClient.from_connection_string(self._conn_str)

    async def enqueue(self, job_id: str) -> None:
        async with self._client() as client:
            queue = client.get_queue_client(self._queue_name)
            await queue.send_message(json.dumps({"job_id": job_id}))

    async def start(self, handler: Callable[[str], None]) -> None:
        """
        Polling loop — run this in the worker container, not the API.
        Polls every 5 seconds. In production, consider long-polling or
        Azure Event Grid triggers instead.
        """
        self._running = True
        async with self._client() as client:
            queue = client.get_queue_client(self._queue_name)
            while self._running:
                messages = queue.receive_messages(
                    messages_per_page=1,
                    visibility_timeout=self._visibility_timeout,
                )
                async for msg in messages:
                    try:
                        payload = json.loads(msg.content)
                        job_id  = payload["job_id"]
                        handler(job_id)
                        await queue.delete_message(msg)
                    except Exception as e:
                        # Message becomes visible again after visibility_timeout
                        print(f"Worker error processing message: {e}")
                await asyncio.sleep(5)

    async def stop(self) -> None:
        self._running = False


# ── factory ───────────────────────────────────────────────────────────────────

def get_queue() -> AbstractQueue:
    settings = get_settings()
    if settings.worker_mode == "queue":
        if not settings.azure_queue_connection_string:
            raise ValueError(
                "AZURE_QUEUE_CONNECTION_STRING must be set when WORKER_MODE=queue"
            )
        return AzureQueueStorage(
            connection_string=settings.azure_queue_connection_string,
            queue_name=settings.azure_queue_name,
            visibility_timeout_s=settings.queue_visibility_timeout_s,
        )
    return LocalThreadQueue(max_workers=settings.worker_threads)