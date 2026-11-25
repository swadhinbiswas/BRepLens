#!/usr/bin/env python
"""Smoke-test the docker-compose API and write a provenance-bearing result."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _post_multipart(url: str, path: Path, mode: str) -> dict:
    boundary = "----breplens-smoke"
    data = path.read_bytes()
    body = (
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
            f"Content-Type: application/step\r\n\r\n"
        ).encode()
        + data
        + (
            f'\r\n--{boundary}\r\nContent-Disposition: form-data; name="mode"\r\n\r\n{mode}\r\n'
            f"--{boundary}--\r\n"
        ).encode()
    )
    req = urllib.request.Request(
        url + "/v1/jobs",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    return json.load(urllib.request.urlopen(req, timeout=60))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--part", default=str(ROOT / "samples" / "sample_09_bracket_mixed.step"))
    ap.add_argument("--mode", default="hybrid")
    ap.add_argument("--out", default=str(ROOT / "results" / "docker_service_test.json"))
    args = ap.parse_args()
    health = json.load(urllib.request.urlopen(args.base_url + "/health", timeout=20))
    t0 = time.perf_counter()
    job = _post_multipart(args.base_url, Path(args.part), args.mode)
    submitted = time.perf_counter() - t0
    deadline = time.time() + 120
    status: dict = {}
    while time.time() < deadline:
        status = json.load(
            urllib.request.urlopen(f"{args.base_url}/v1/jobs/{job['job_id']}", timeout=20)
        )
        if status["status"] in ("succeeded", "failed"):
            break
        time.sleep(0.2)
    elapsed = time.perf_counter() - t0
    out = {
        "health_ok": health.get("status") == "ok",
        "submit_seconds": round(submitted, 3),
        "job_seconds": round(elapsed, 3),
        "status": status.get("status"),
        "error": status.get("error"),
        "mode": args.mode,
        "job_timings": status.get("timings", {}),
        "provenance": {
            "script": "scripts/docker_smoke_test.py",
            "command": " ".join(sys.argv),
            "base_url": args.base_url,
            "part": str(Path(args.part).relative_to(ROOT)),
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }
    if status.get("status") == "succeeded":
        fs = json.load(
            urllib.request.urlopen(f"{args.base_url}/v1/jobs/{job['job_id']}/features", timeout=20)
        )
        sim = json.load(
            urllib.request.urlopen(
                f"{args.base_url}/v1/jobs/{job['job_id']}/similar?k=3", timeout=20
            )
        )
        out.update(
            n_features=len(fs["features"]),
            features=[f["type"] for f in fs["features"]],
            similar_top1=sim[0]["name"] if sim else None,
        )
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=2))
    return 0 if out["status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
