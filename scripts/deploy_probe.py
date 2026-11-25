#!/usr/bin/env python
"""Hosted-deployment probe: cold start, CPU latency, peak RSS, headless server.

Run with the inference-only environment:
    .deploy-venv/bin/python scripts/deploy_probe.py

Writes results/deploy_benchmark.json. Numbers come from this run only.
"""

from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

SAMPLES = sorted((ROOT / "samples").glob("*.step"))


def _rss_mb(pid: int | None = None) -> float:
    if pid is None:
        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)
    import psutil

    proc = psutil.Process(pid)
    return float(proc.memory_info().rss / (1024 * 1024))


def _apptest_probe() -> dict:
    from streamlit.testing.v1 import AppTest

    t0 = time.perf_counter()
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=300).run()
    cold = time.perf_counter() - t0
    exceptions = [str(e.value) for e in at.exception]
    if exceptions:
        raise RuntimeError(f"app initial render failed: {exceptions}")
    t0 = time.perf_counter()
    at.radio[0].set_value("hybrid").run()
    radio_rerun = time.perf_counter() - t0
    t1 = time.perf_counter()
    at.selectbox[0].select(SAMPLES[0].name).run()
    model_load_and_first_part = time.perf_counter() - t1
    exceptions = [str(e.value) for e in at.exception]
    if exceptions:
        raise RuntimeError(f"app first sample failed: {exceptions}")
    e2e = []
    for sample in SAMPLES:
        t = time.perf_counter()
        at.selectbox[0].select(sample.name).run()
        elapsed = time.perf_counter() - t
        if at.exception:
            raise RuntimeError(
                f"app failed on {sample.name}: {[str(e.value) for e in at.exception]}"
            )
        e2e.append({"sample": sample.name, "seconds": round(elapsed, 3)})
    return {
        "cold_start_seconds": round(cold, 3),
        "radio_rerun_seconds": round(radio_rerun, 3),
        "model_load_and_first_part_seconds": round(model_load_and_first_part, 3),
        "e2e_samples": e2e,
        "peak_rss_apptest_mb": round(_rss_mb(), 1),
    }


def _pipeline_probe() -> dict:
    from breplens.adapters.occ import OccAdapter
    from breplens.ml.inference import load_for_inference
    from breplens.recognition.gnn_recognizer import recognize_gnn, recognize_hybrid
    from breplens.recognition.rules import recognize_rules

    adapter = OccAdapter()
    t0 = time.perf_counter()
    model, classes, _ = load_for_inference(ROOT / "models" / "gnn_synthetic.pt", device="cpu")
    load_model_s = time.perf_counter() - t0
    timings: dict[str, list[float]] = {"rules": [], "gnn": [], "hybrid": [], "tessellate": []}
    counts: dict[str, int] = {"rules": 0, "gnn": 0, "hybrid": 0}
    for sample in SAMPLES:
        graph = adapter.load_graph(sample)
        t = time.perf_counter()
        fr = recognize_rules(graph)
        timings["rules"].append((time.perf_counter() - t) * 1000)
        counts["rules"] += len(fr.features)
        t = time.perf_counter()
        fg = recognize_gnn(graph, model, classes, device="cpu")
        timings["gnn"].append((time.perf_counter() - t) * 1000)
        counts["gnn"] += len(fg.features)
        t = time.perf_counter()
        fh = recognize_hybrid(graph, model, classes, device="cpu")
        timings["hybrid"].append((time.perf_counter() - t) * 1000)
        counts["hybrid"] += len(fh.features)
        t = time.perf_counter()
        adapter.tessellate(sample, deflection=0.25)
        timings["tessellate"].append((time.perf_counter() - t) * 1000)
    out = {}
    for k, vals in timings.items():
        out[k] = {
            "mean_ms": round(float(np.mean(vals)), 2),
            "median_ms": round(float(np.median(vals)), 2),
            "p95_ms": round(float(np.percentile(vals, 95)), 2),
            "n": len(vals),
        }
    return {
        "model_load_seconds": round(load_model_s, 3),
        "latency": out,
        "features_total_over_samples": counts,
        "peak_rss_pipeline_mb": round(_rss_mb(), 1),
    }


def _server_probe() -> dict:
    port = 8599
    env = dict(os.environ)
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        [
            str(ROOT / ".deploy-venv" / "bin" / "streamlit"),
            "run",
            str(ROOT / "streamlit_app.py"),
            "--server.headless",
            "true",
            "--server.port",
            str(port),
            "--server.address",
            "127.0.0.1",
        ],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    healthy = False
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/_stcore/health", timeout=2) as r:
                if r.status == 200:
                    healthy = True
                    break
        except Exception:
            if proc.poll() is not None:
                break
            time.sleep(0.2)
    start = time.perf_counter() - t0
    rss = _rss_mb(proc.pid) if proc.poll() is None else 0.0
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    if not healthy:
        out = proc.stdout.read().decode(errors="replace")[-2000:] if proc.stdout else ""
        raise RuntimeError(f"streamlit server did not become healthy:\n{out}")
    return {
        "server_start_seconds": round(start, 2),
        "idle_rss_mb": round(rss, 1),
        "healthy": healthy,
    }


def _install_seconds() -> int | None:
    log = ROOT / "results" / "deploy_install.txt"
    if not log.exists():
        return None
    import re

    m = re.search(r"SECONDS=(\d+)", log.read_text(errors="replace"))
    return int(m.group(1)) if m else None


def main() -> int:
    payload = {
        "python": sys.version.split()[0],
        "host": os.uname().nodename,
        "install": {
            "venv_bytes": int(
                sum(p.stat().st_size for p in (ROOT / ".deploy-venv").rglob("*") if p.is_file())
            ),
            "venv_mb": round(
                sum(p.stat().st_size for p in (ROOT / ".deploy-venv").rglob("*") if p.is_file())
                / (1024 * 1024),
                1,
            ),
            "install_seconds": _install_seconds(),
            "install_command": ".deploy-venv/bin/pip install -r requirements.txt",
        },
        "n_samples": len(SAMPLES),
        "apptest": _apptest_probe(),
        "pipeline": _pipeline_probe(),
        "server": _server_probe(),
        "limits": {
            "target_peak_mb": 1500,
            "streamlit_community_cloud_approx": "2 vCPU / 2.7 GB RAM",
        },
        "provenance": {
            "script": "scripts/deploy_probe.py",
            "command": " ".join(sys.argv),
            "venv": ".deploy-venv (Python " + sys.version.split()[0] + ")",
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out = ROOT / "results" / "deploy_benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    print(json.dumps(payload, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
