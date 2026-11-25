"""FastAPI service: upload, job queue, feature JSON, similarity, schema."""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from breplens.schema import SCHEMA_VERSION, FeatureSet
from breplens.service.config import Settings
from breplens.service.pipeline import (
    ModelRegistry,
    load_result,
    load_similarity_index,
    similarity_for_graph,
)
from breplens.service.storage import Storage

log = logging.getLogger(__name__)
Mode = Literal["rules", "gnn", "hybrid"]
STEP_MAGIC = b"ISO-10303-21"


class JobCreated(BaseModel):
    job_id: str
    part_id: str
    status: str = "queued"
    mode: str


class JobStatus(BaseModel):
    job_id: str
    part_id: str
    status: str
    mode: str
    created_utc: str
    started_utc: str | None = None
    finished_utc: str | None = None
    error: str | None = None
    timings: dict[str, float] = {}


class SimilarHit(BaseModel):
    id: str
    name: str
    family: str
    score: float
    metadata: dict[str, Any] = {}


class Health(BaseModel):
    status: str
    schema_version: str
    queue_backend: str
    model_available: bool
    jobs: dict[str, int]


def _validate_mode(mode: str) -> str:
    if mode not in ("rules", "gnn", "hybrid"):
        raise HTTPException(status_code=422, detail=f"mode must be rules/gnn/hybrid, got {mode}")
    return mode


def create_app(settings: Settings | None = None, *, storage: Storage | None = None) -> FastAPI:
    settings = settings or Settings()
    settings.ensure_dirs()
    store = storage or Storage(settings.resolved_database_url)
    registry = ModelRegistry()
    app = FastAPI(
        title="BRepLens API",
        version="1.0.0",
        description="CAD-agnostic manufacturing feature recognition with a job queue.",
    )
    app.state.settings = settings
    app.state.storage = store
    app.state.registry = registry
    app.state.index = load_similarity_index(settings.similarity_index)
    app.state.started = time.time()

    @app.middleware("http")
    async def _access_log(request: Request, call_next: Any) -> Any:
        rid = uuid.uuid4().hex[:12]
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        log.info(
            json.dumps(
                {
                    "event": "request",
                    "request_id": rid,
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "ms": round((time.perf_counter() - t0) * 1000, 2),
                }
            )
        )
        return response

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:  # pragma: no cover
        log.exception("unhandled error")
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "type": type(exc).__name__,
                    "message": "internal error",
                    "path": request.url.path,
                }
            },
        )

    def _enqueue(job_id: str) -> None:
        if settings.queue_backend != "rq":
            return
        try:  # Redis/RQ is optional; leaving the job queued is a safe fallback.
            from redis import Redis
            from rq import Queue

            queue = Queue(connection=Redis.from_url(settings.redis_url))
            queue.enqueue("breplens.service.worker.rq_process_job", job_id, job_timeout=600)
        except Exception as exc:  # pragma: no cover - depends on external service
            log.warning("RQ enqueue failed (%s); job remains queued for a SQLite worker", exc)

    @app.get("/health", response_model=Health)
    def health() -> Health:
        loaded = (
            registry.get(settings.model_path, settings.model_device)
            if settings.model_path.exists()
            else None
        )
        return Health(
            status="ok",
            schema_version=SCHEMA_VERSION,
            queue_backend=settings.queue_backend,
            model_available=loaded is not None,
            jobs=store.stats(),
        )

    @app.post("/v1/jobs", response_model=JobCreated, status_code=202)
    async def submit_job(
        file: UploadFile = File(...),
        mode: str = Form("hybrid"),
    ) -> JobCreated:
        mode = _validate_mode(mode)
        filename = file.filename or "upload.step"
        if Path(filename).suffix.lower() not in (".step", ".stp"):
            raise HTTPException(status_code=415, detail="only .step/.stp files are accepted")
        limit = settings.max_upload_mb * 1024 * 1024
        part_id = str(uuid.uuid4())
        stored = settings.data_dir / "uploads" / f"{part_id}.step"
        sha = hashlib.sha256()
        size = 0
        first = True
        try:
            with stored.open("wb") as fh:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > limit:
                        raise HTTPException(
                            status_code=413,
                            detail=f"file exceeds {settings.max_upload_mb} MiB limit",
                        )
                    if first:
                        if not chunk.lstrip().startswith(STEP_MAGIC):
                            raise HTTPException(
                                status_code=422, detail="file is not an ISO-10303-21 STEP file"
                            )
                        first = False
                    sha.update(chunk)
                    fh.write(chunk)
        except HTTPException:
            stored.unlink(missing_ok=True)
            raise
        if size == 0:
            stored.unlink(missing_ok=True)
            raise HTTPException(status_code=422, detail="empty upload")
        store.create_part(
            part_id=part_id,
            filename=filename,
            sha256=sha.hexdigest(),
            path=str(stored),
            size_bytes=size,
        )
        job_id = str(uuid.uuid4())
        store.create_job(job_id=job_id, part_id=part_id, mode=mode)
        _enqueue(job_id)
        return JobCreated(job_id=job_id, part_id=part_id, mode=mode)

    @app.get("/v1/jobs", response_model=list[JobStatus])
    def list_jobs(limit: int = 20) -> list[JobStatus]:
        return [
            JobStatus(job_id=j["id"], **{k: v for k, v in j.items() if k != "id"})
            for j in store.list_jobs(limit)
        ]

    @app.get("/v1/jobs/{job_id}", response_model=JobStatus)
    def job_status(job_id: str) -> JobStatus:
        job = store.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return JobStatus(job_id=job["id"], **{k: v for k, v in job.items() if k != "id"})

    @app.get("/v1/jobs/{job_id}/features", response_model=FeatureSet)
    def job_features(job_id: str) -> FeatureSet:
        job = store.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        if job["status"] == "failed":
            raise HTTPException(status_code=409, detail=job["error"] or "job failed")
        if job["status"] != "succeeded":
            raise HTTPException(status_code=409, detail=f"job is {job['status']}")
        return load_result(store, job)

    @app.get("/v1/jobs/{job_id}/similar", response_model=list[SimilarHit])
    def job_similar(job_id: str, k: int = 5) -> list[SimilarHit]:
        if not 1 <= k <= 20:
            raise HTTPException(status_code=422, detail="k must be between 1 and 20")
        job = store.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        if job["status"] != "succeeded":
            raise HTTPException(status_code=409, detail=f"job is {job['status']}")
        part = store.get_part(job["part_id"])
        assert part is not None
        fs = load_result(store, job)
        from breplens.adapters.occ import OccAdapter

        graph = OccAdapter().load_graph(part["path"], name=Path(part["filename"]).stem)
        hits = similarity_for_graph(graph, fs, settings.similarity_index, k=k)
        return [SimilarHit(**h) for h in hits]

    @app.get("/v1/schema")
    def schema() -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "feature_set": FeatureSet.model_json_schema(),
        }

    return app


app = create_app()
