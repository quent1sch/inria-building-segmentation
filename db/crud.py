"""
db/crud.py

Database operations: create, read, update for Job and CachedResult.

All functions take an AsyncSession and are called from either the API
layer or the worker — they don't manage transactions themselves, letting
the caller control commit/rollback via the session context manager.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import CachedResult, Job, JobStatus


# ── hashing ───────────────────────────────────────────────────────────────────

def compute_input_hash(canonical_ref: str, params: dict) -> str:
    """
    SHA256 of the canonical input reference + sorted params JSON.

    canonical_ref:
      - upload:    SHA256 of file bytes (content-addressed)
      - from-path: absolute path + file mtime (invalidates on file change)
      - from-url:  URL string (same URL = same content assumed)

    params: dict of inference parameters (resolution, processing, etc.)
            sorted for determinism regardless of insertion order.
    """
    params_str = json.dumps(params, sort_keys=True)
    payload    = f"{canonical_ref}||{params_str}".encode()
    return hashlib.sha256(payload).hexdigest()


def hash_bytes(data: bytes) -> str:
    """SHA256 of raw bytes — used for upload input_hash."""
    return hashlib.sha256(data).hexdigest()


# ── Job CRUD ──────────────────────────────────────────────────────────────────

async def create_job(
    session: AsyncSession,
    user_id: str,
    input_mode: str,
    input_ref: str,
    params: dict,
    result_type: str,
    input_hash: Optional[str] = None,
) -> Job:
    job = Job(
        user_id=user_id,
        input_mode=input_mode,
        input_ref=input_ref,
        input_hash=input_hash,
        params=params,
        result_type=result_type,
        status=JobStatus.QUEUED,
    )
    session.add(job)
    await session.flush()   # assigns job.id without committing
    return job


async def get_job(session: AsyncSession, job_id: str) -> Optional[Job]:
    result = await session.execute(select(Job).where(Job.id == job_id))
    return result.scalar_one_or_none()


async def list_jobs(
    session: AsyncSession,
    user_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> list[Job]:
    q = select(Job).order_by(Job.created_at.desc()).limit(limit).offset(offset)
    if user_id:
        q = q.where(Job.user_id == user_id)
    result = await session.execute(q)
    return list(result.scalars().all())


async def mark_processing(session: AsyncSession, job_id: str) -> None:
    await session.execute(
        update(Job)
        .where(Job.id == job_id)
        .values(status=JobStatus.PROCESSING, started_at=datetime.now(timezone.utc))
    )


async def mark_done(
    session: AsyncSession,
    job_id: str,
    result_path: str,
    duration_s: float,
) -> None:
    now = datetime.now(timezone.utc)
    await session.execute(
        update(Job)
        .where(Job.id == job_id)
        .values(
            status=JobStatus.DONE,
            result_path=result_path,
            completed_at=now,
            duration_s=duration_s,
        )
    )


async def mark_failed(
    session: AsyncSession,
    job_id: str,
    error: str,
) -> None:
    await session.execute(
        update(Job)
        .where(Job.id == job_id)
        .values(
            status=JobStatus.FAILED,
            error=error,
            completed_at=datetime.now(timezone.utc),
        )
    )


# ── Cache CRUD ────────────────────────────────────────────────────────────────

async def get_cached_result(
    session: AsyncSession,
    input_hash: str,
) -> Optional[CachedResult]:
    result = await session.execute(
        select(CachedResult).where(CachedResult.input_hash == input_hash)
    )
    cached = result.scalar_one_or_none()
    if cached:
        # Update hit count and last_accessed in same transaction
        cached.hit_count += 1
        cached.last_accessed = datetime.now(timezone.utc)
    return cached


async def store_cached_result(
    session: AsyncSession,
    input_hash: str,
    result_path: str,
    result_type: str,
) -> CachedResult:
    """
    Store a cache entry. If the hash already exists (race condition between
    two identical concurrent requests), update the existing entry silently.
    """
    existing = await session.execute(
        select(CachedResult).where(CachedResult.input_hash == input_hash)
    )
    cached = existing.scalar_one_or_none()

    if cached:
        cached.result_path = result_path
        cached.result_type = result_type
    else:
        cached = CachedResult(
            input_hash=input_hash,
            result_path=result_path,
            result_type=result_type,
        )
        session.add(cached)

    return cached