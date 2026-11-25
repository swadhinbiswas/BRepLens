from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from breplens.service import worker
from breplens.service.app import create_app
from breplens.service.config import Settings
from breplens.service.storage import Storage


@pytest.fixture()
def service(tmp_path: Path, root: Path):
    settings = Settings(
        data_dir=tmp_path,
        database_url="",
        similarity_index=root / "demo" / "library.json",
        model_path=root / "models" / "gnn_synthetic.pt",
        max_upload_mb=25,
    )
    storage = Storage(settings.resolved_database_url)
    app = create_app(settings, storage=storage)
    with TestClient(app) as client:
        yield client, settings, storage


def _upload(client: TestClient, path: Path, mode: str = "rules"):
    with path.open("rb") as fh:
        return client.post(
            "/v1/jobs",
            files={"file": (path.name, fh, "application/step")},
            data={"mode": mode},
        )


def test_health_and_schema(service) -> None:
    client, _, _ = service
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["schema_version"] == "1.0.0"
    assert "jobs" in body
    s = client.get("/v1/schema")
    assert s.status_code == 200
    assert s.json()["feature_set"]["type"] == "object"


def test_end_to_end_rules(service, samples_dir: Path) -> None:
    client, settings, storage = service
    r = _upload(client, samples_dir / "sample_04_pocket.step")
    assert r.status_code == 202, r.text
    job = r.json()
    assert job["status"] == "queued"
    assert worker.run_once(storage, settings)
    status = client.get(f"/v1/jobs/{job['job_id']}").json()
    assert status["status"] == "succeeded"
    assert status["timings"]["total_ms"] > 0
    features = client.get(f"/v1/jobs/{job['job_id']}/features")
    assert features.status_code == 200
    fs = features.json()
    assert fs["mode"] == "rules"
    assert any(f["type"] == "pocket" for f in fs["features"])
    sim = client.get(f"/v1/jobs/{job['job_id']}/similar?k=3")
    assert sim.status_code == 200
    assert len(sim.json()) <= 3
    assert client.get("/v1/jobs").json()


def test_gnn_mode(service, samples_dir: Path) -> None:
    client, settings, storage = service
    r = _upload(client, samples_dir / "sample_00_plate_holes.step", mode="gnn")
    job = r.json()
    assert worker.run_once(storage, settings)
    fs = client.get(f"/v1/jobs/{job['job_id']}/features").json()
    assert fs["mode"] == "gnn"
    assert fs["model_version"]


def test_upload_validation(service, tmp_path: Path, samples_dir: Path) -> None:
    client, _, _ = service
    bad_ext = client.post("/v1/jobs", files={"file": ("x.txt", b"hello", "text/plain")})
    assert bad_ext.status_code == 415
    bad_magic = client.post("/v1/jobs", files={"file": ("x.step", b"not step", "application/step")})
    assert bad_magic.status_code == 422
    empty = client.post("/v1/jobs", files={"file": ("x.step", b"", "application/step")})
    assert empty.status_code == 422
    bad_mode = _upload(client, samples_dir / "sample_04_pocket.step", mode="magic")
    assert bad_mode.status_code == 422


def test_size_limit(tmp_path: Path, root: Path, samples_dir: Path) -> None:
    settings = Settings(data_dir=tmp_path, max_upload_mb=1, similarity_index=root / "nope.json")
    storage = Storage(settings.resolved_database_url)
    app = create_app(settings, storage=storage)
    with TestClient(app) as client:
        data = b"ISO-10303-21;\n" + b"x" * (2 * 1024 * 1024)
        r = client.post("/v1/jobs", files={"file": ("big.step", data, "application/step")})
        assert r.status_code == 413


def test_failed_job_is_reported(service, tmp_path: Path) -> None:
    client, settings, storage = service
    corrupt = tmp_path / "corrupt.step"
    corrupt.write_bytes(b"ISO-10303-21;\nnot really a step file\n")
    r = _upload(client, corrupt)
    job = r.json()
    assert worker.run_once(storage, settings)
    status = client.get(f"/v1/jobs/{job['job_id']}").json()
    assert status["status"] == "failed"
    assert status["error"]
    features = client.get(f"/v1/jobs/{job['job_id']}/features")
    assert features.status_code == 409


def test_missing_job_and_bounds(service) -> None:
    client, _, _ = service
    assert client.get("/v1/jobs/nope").status_code == 404
    assert client.get("/v1/jobs/nope/features").status_code == 404
    assert client.get("/v1/jobs/nope/similar").status_code == 404


def test_face_limit_enforced(tmp_path: Path, root: Path, samples_dir: Path) -> None:
    settings = Settings(
        data_dir=tmp_path,
        database_url="",
        similarity_index=root / "demo" / "library.json",
        max_faces=5,
    )
    storage = Storage(settings.resolved_database_url)
    app = create_app(settings, storage=storage)
    with TestClient(app) as client:
        r = _upload(client, samples_dir / "sample_04_pocket.step")
        job = r.json()
        assert worker.run_once(storage, settings)
        status = client.get(f"/v1/jobs/{job['job_id']}").json()
        assert status["status"] == "failed"
        assert "limit" in (status["error"] or "").lower()
