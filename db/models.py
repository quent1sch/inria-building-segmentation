"""
db/models.py

SQLAlchemy ORM models.

Design decisions
----------------
- user_id on Job: costs nothing now, enables per-user history, rate limiting,
  and quota tracking later. Defaults to settings.default_user_id ("local"
  for single-user setups). In production, extract from JWT/API key in a
  FastAPI dependency and pass through.

- input_hash on Job: SHA256 of (input content or canonical URL) + params.
  Used for cache lookup — if the same file with the same params was already
  processed, return the cached result immediately.

- CachedResult: separate table so cache entries can outlive the job that
  created them and can be managed independently (TTL, manual invalidation).

- params stored as JSON: avoids schema migrations when new inference params
  are added. The params dict matches the API query params exactly.
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    JSON,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return str(uuid.uuid4())


class Base(DeclarativeBase):
    pass


# ── enums ─────────────────────────────────────────────────────────────────────

class JobStatus:
    QUEUED     = "queued"
    PROCESSING = "processing"
    DONE       = "done"
    FAILED     = "failed"


class InputMode:
    UPLOAD = "upload"
    PATH   = "path"
    URL    = "url"


class ResultType:
    MASK    = "mask"
    OVERLAY = "overlay"
    VECTOR  = "vector"


# ── Job ───────────────────────────────────────────────────────────────────────

class Job(Base):
    """
    One inference request = one Job row.

    Lifecycle:
      queued → processing → done
                         → failed

    result_path is null until status=done.
    error is null unless status=failed.
    """
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=_uuid
    )

    # ── who ───────────────────────────────────────────────────────────────
    user_id: Mapped[str] = mapped_column(
        String(128), nullable=False, index=True,
        comment="User identifier. 'local' for single-user setup. "
                "In production: extracted from JWT or API key.",
    )

    # ── what ──────────────────────────────────────────────────────────────
    status: Mapped[str] = mapped_column(
        Enum("queued", "processing", "done", "failed", name="job_status"),
        nullable=False, default=JobStatus.QUEUED, index=True,
    )
    input_mode: Mapped[str] = mapped_column(
        Enum("upload", "path", "url", name="input_mode"),
        nullable=False,
    )
    # Human-readable reference: original filename, path, or URL
    input_ref: Mapped[str] = mapped_column(Text, nullable=False)
    # SHA256(canonical_input + sorted_params) — used for cache lookup
    input_hash: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True, index=True
    )
    # Inference parameters as JSON — resolution, processing, simplify_tolerance, etc.
    params: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    result_type: Mapped[str] = mapped_column(
        Enum("mask", "overlay", "vector", name="result_type"),
        nullable=False, default=ResultType.MASK,
    )

    # ── result ────────────────────────────────────────────────────────────
    # Storage key (local path or Azure blob name) — null until done
    result_path: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # ── timing ────────────────────────────────────────────────────────────
    created_at:   Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    started_at:   Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_s:   Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # ── relationship ──────────────────────────────────────────────────────
    cached_result: Mapped[Optional["CachedResult"]] = relationship(
        "CachedResult",
        primaryjoin="Job.input_hash == foreign(CachedResult.input_hash)",
        viewonly=True,
    )

    def to_dict(self) -> dict:
        return {
            "job_id":       self.id,
            "user_id":      self.user_id,
            "status":       self.status,
            "input_mode":   self.input_mode,
            "input_ref":    self.input_ref,
            "params":       self.params,
            "result_type":  self.result_type,
            "result_path":  self.result_path,
            "error":        self.error,
            "created_at":   self.created_at.isoformat() if self.created_at else None,
            "started_at":   self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_s":   self.duration_s,
        }


# ── CachedResult ──────────────────────────────────────────────────────────────

class CachedResult(Base):
    """
    Stores the result storage key for a given (input_hash).

    If a new job arrives with the same input_hash, the result is served
    from cache without running inference again.

    Cache key = SHA256(canonical_input_reference + sorted_params_json)
    This means: same file + same params = cache hit.
    """
    __tablename__ = "cached_results"
    __table_args__ = (
        UniqueConstraint("input_hash", name="uq_cached_results_hash"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    input_hash:  Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    result_path: Mapped[str] = mapped_column(Text, nullable=False)
    result_type: Mapped[str] = mapped_column(String(16), nullable=False)
    hit_count:   Mapped[int] = mapped_column(Integer, default=0)
    created_at:  Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    last_accessed: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )