"""Worker process: polls the SQLite queue or serves RQ jobs."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time

from breplens.service.config import Settings
from breplens.service.pipeline import ModelRegistry, process_job
from breplens.service.storage import Storage

log = logging.getLogger(__name__)
_stop = False
_registry = ModelRegistry()


def _handle_signal(signum: int, frame: object) -> None:  # pragma: no cover - signal path
    global _stop
    _stop = True


def run_once(storage: Storage, settings: Settings) -> bool:
    job = storage.claim_next_job()
    if job is None:
        return False
    log.info("job %s claimed (%s)", job["id"], job["mode"])
    try:
        process_job(storage, settings, job, _registry)
        log.info("job %s succeeded", job["id"])
    except Exception as exc:
        log.exception("job %s failed", job["id"])
        storage.finish_job(job["id"], status="failed", error=f"{type(exc).__name__}: {exc}")
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="BRepLens job worker")
    ap.add_argument("--once", action="store_true", help="process at most one job and exit")
    ap.add_argument("--recover-stale", action="store_true", help="requeue stale running jobs first")
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=os.environ.get("BREPLENS_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    settings = Settings()
    settings.ensure_dirs()
    storage = Storage(settings.resolved_database_url)
    if args.recover_stale:
        n = storage.requeue_stale()
        log.info("requeued %s stale jobs", n)
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    log.info(
        "worker started (queue=%s, db=%s)", settings.queue_backend, settings.resolved_database_url
    )
    while not _stop:
        if run_once(storage, settings):
            continue
        if args.once:
            break
        time.sleep(settings.worker_poll_seconds)
    return 0


def rq_process_job(job_id: str) -> None:  # pragma: no cover - exercised only with Redis
    """RQ entry point; job status/results still live in the shared SQL database."""
    logging.basicConfig(level=os.environ.get("BREPLENS_LOG_LEVEL", "INFO"))
    settings = Settings()
    settings.ensure_dirs()
    storage = Storage(settings.resolved_database_url)
    if not storage.mark_running(job_id):
        log.warning("RQ job %s is not queued; skipping", job_id)
        return
    job = storage.get_job(job_id)
    if job is None:
        raise RuntimeError(f"unknown job {job_id}")
    try:
        process_job(storage, settings, job, _registry)
    except Exception as exc:
        storage.finish_job(job_id, status="failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    sys.exit(main())
