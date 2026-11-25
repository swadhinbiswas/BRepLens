"""SQLAlchemy storage for parts and jobs (SQLite default, PostgreSQL optional)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    select,
    update,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class PartRow(Base):
    __tablename__ = "parts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    filename: Mapped[str] = mapped_column(String(255))
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    path: Mapped[str] = mapped_column(Text)
    size_bytes: Mapped[int] = mapped_column(Integer)
    created_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    n_faces: Mapped[int | None] = mapped_column(Integer, nullable=True)
    metadata_json: Mapped[str] = mapped_column(Text, default="{}")


class JobRow(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    part_id: Mapped[str] = mapped_column(ForeignKey("parts.id"), index=True)
    mode: Mapped[str] = mapped_column(String(16), default="hybrid")
    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)
    created_utc: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_utc: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    result_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    timings_json: Mapped[str] = mapped_column(Text, default="{}")


class Storage:
    def __init__(self, database_url: str) -> None:
        connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        self.engine = create_engine(database_url, future=True, connect_args=connect_args)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(self.engine, expire_on_commit=False)

    def create_part(
        self,
        *,
        part_id: str,
        filename: str,
        sha256: str,
        path: str,
        size_bytes: int,
        metadata: dict | None = None,
    ) -> None:
        with self.Session() as s:
            s.add(
                PartRow(
                    id=part_id,
                    filename=filename,
                    sha256=sha256,
                    path=path,
                    size_bytes=size_bytes,
                    metadata_json=json.dumps(metadata or {}),
                )
            )
            s.commit()

    def get_part(self, part_id: str) -> dict[str, Any] | None:
        with self.Session() as s:
            row = s.get(PartRow, part_id)
            if row is None:
                return None
            return {
                "id": row.id,
                "filename": row.filename,
                "sha256": row.sha256,
                "path": row.path,
                "size_bytes": row.size_bytes,
                "n_faces": row.n_faces,
                "metadata": json.loads(row.metadata_json),
                "created_utc": row.created_utc.isoformat(),
            }

    def set_part_faces(self, part_id: str, n_faces: int) -> None:
        with self.Session() as s:
            s.execute(update(PartRow).where(PartRow.id == part_id).values(n_faces=n_faces))
            s.commit()

    def create_job(self, *, job_id: str, part_id: str, mode: str) -> None:
        with self.Session() as s:
            s.add(JobRow(id=job_id, part_id=part_id, mode=mode, status="queued"))
            s.commit()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self.Session() as s:
            row = s.get(JobRow, job_id)
            if row is None:
                return None
            return {
                "id": row.id,
                "part_id": row.part_id,
                "mode": row.mode,
                "status": row.status,
                "created_utc": row.created_utc.isoformat(),
                "started_utc": row.started_utc.isoformat() if row.started_utc else None,
                "finished_utc": row.finished_utc.isoformat() if row.finished_utc else None,
                "error": row.error,
                "result_path": row.result_path,
                "timings": json.loads(row.timings_json or "{}"),
            }

    def list_jobs(self, limit: int = 20) -> list[dict[str, Any]]:
        with self.Session() as s:
            rows = s.execute(
                select(JobRow).order_by(JobRow.created_utc.desc()).limit(limit)
            ).scalars()
            return [
                {
                    "id": r.id,
                    "part_id": r.part_id,
                    "mode": r.mode,
                    "status": r.status,
                    "created_utc": r.created_utc.isoformat(),
                    "error": r.error,
                }
                for r in rows
            ]

    def claim_next_job(self) -> dict[str, Any] | None:
        """Atomically claim the oldest queued job (safe for one or more workers)."""
        with self.Session() as s:
            while True:
                row = s.execute(
                    select(JobRow)
                    .where(JobRow.status == "queued")
                    .order_by(JobRow.created_utc)
                    .limit(1)
                    .with_for_update(skip_locked=True)
                ).scalar_one_or_none()
                if row is None:
                    return None
                result = s.execute(
                    update(JobRow)
                    .where(JobRow.id == row.id, JobRow.status == "queued")
                    .values(status="running", started_utc=utcnow())
                )
                s.commit()
                if int(getattr(result, "rowcount", 0) or 0) == 1:
                    return self.get_job(row.id)

    def mark_running(self, job_id: str) -> bool:
        with self.Session() as s:
            result = s.execute(
                update(JobRow)
                .where(JobRow.id == job_id, JobRow.status == "queued")
                .values(status="running", started_utc=utcnow())
            )
            s.commit()
            return int(getattr(result, "rowcount", 0) or 0) == 1

    def finish_job(
        self,
        job_id: str,
        *,
        status: str,
        result_path: str | None = None,
        error: str | None = None,
        timings: dict[str, float] | None = None,
    ) -> None:
        with self.Session() as s:
            s.execute(
                update(JobRow)
                .where(JobRow.id == job_id)
                .values(
                    status=status,
                    finished_utc=utcnow(),
                    result_path=result_path,
                    error=error,
                    timings_json=json.dumps(timings or {}),
                )
            )
            s.commit()

    def requeue_stale(self, older_than_seconds: float = 3600.0) -> int:
        """Requeue jobs left running by a crashed worker."""
        cutoff = utcnow().timestamp() - older_than_seconds
        with self.Session() as s:
            rows = s.execute(select(JobRow).where(JobRow.status == "running")).scalars()
            count = 0
            for row in rows:
                started = (row.started_utc or row.created_utc).timestamp()
                if started < cutoff:
                    row.status = "queued"
                    count += 1
            s.commit()
            return count

    def stats(self) -> dict[str, int]:
        with self.Session() as s:
            total = s.query(JobRow).count()
            failed = s.query(JobRow).filter(JobRow.status == "failed").count()
            queued = s.query(JobRow).filter(JobRow.status == "queued").count()
            running = s.query(JobRow).filter(JobRow.status == "running").count()
            done = s.query(JobRow).filter(JobRow.status == "succeeded").count()
            return {
                "jobs": total,
                "failed": failed,
                "queued": queued,
                "running": running,
                "succeeded": done,
            }


__all__ = ["JobRow", "PartRow", "Storage", "utcnow"]
