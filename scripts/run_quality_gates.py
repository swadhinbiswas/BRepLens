#!/usr/bin/env python
"""Run the local quality gates and write test/coverage provenance results."""

from __future__ import annotations

import importlib.metadata as md
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def _run(cmd: list[str], *, log: Path | None = None) -> subprocess.CompletedProcess[str]:
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if log is not None:
        log.write_text(r.stdout + r.stderr)
    return r


def main() -> int:
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    commands = {
        "ruff": [PY, "-m", "ruff", "check", "src", "tests", "scripts", "streamlit_app.py"],
        "mypy": [PY, "-m", "mypy", "src/breplens"],
        "tests": [PY, "-m", "pytest", "-o", "addopts=", "-q"],
        "coverage_full": [
            PY,
            "-m",
            "pytest",
            "-q",
            "--cov=breplens",
            "--cov-report=json:results/coverage.json",
        ],
        "coverage_core": [
            PY,
            "-m",
            "pytest",
            "-q",
            "--cov=breplens.geometry",
            "--cov=breplens.recognition",
            "--cov=breplens.schema",
            "--cov-report=json:results/coverage_core.json",
        ],
    }
    outputs = {}
    for name, cmd in commands.items():
        t0 = time.perf_counter()
        log = results / (
            "coverage.txt"
            if name == "coverage_full"
            else "coverage_core.txt"
            if name == "coverage_core"
            else f"{name}.txt"
        )
        r = _run(cmd, log=log)
        outputs[name] = {
            "returncode": r.returncode,
            "seconds": round(time.perf_counter() - t0, 2),
            "tail": (r.stdout + r.stderr).strip().splitlines()[-1:] or [""],
        }
        if r.returncode != 0:
            print(f"{name} FAILED\n{r.stdout}\n{r.stderr}")
            return r.returncode
    # remove transient logs for lint/type commands; coverage logs are evidence
    (results / "ruff.txt").unlink(missing_ok=True)
    (results / "mypy.txt").unlink(missing_ok=True)
    text = (results / "tests.txt").read_text(errors="replace")
    m = re.search(r"(\d+) passed", text)
    tests_payload = {
        "passed": int(m.group(1)) if m else None,
        "failed": 0,
        "wall_seconds": outputs["tests"]["seconds"],
        "command": "pytest -o addopts='' -q",
        "python": platform.python_version(),
        "pytest": md.version("pytest"),
        "exit_code": outputs["tests"]["returncode"],
        "raw_tail": outputs["tests"]["tail"][0],
        "provenance": {
            "script": "scripts/run_quality_gates.py",
            "command": " ".join(sys.argv),
            "generated_utc": stamp,
        },
    }
    (results / "tests.json").write_text(json.dumps(tests_payload, indent=1))
    meta = {
        "script": "scripts/run_quality_gates.py",
        "commands": {k: " ".join(v) for k, v in commands.items() if k.startswith("coverage")},
        "python": platform.python_version(),
        "pytest": md.version("pytest"),
        "generated_utc": stamp,
        "outcomes": outputs,
    }
    (results / "coverage_meta.json").write_text(json.dumps(meta, indent=1))
    print(
        json.dumps(
            {
                "tests": tests_payload["passed"],
                "gates": {k: v["returncode"] for k, v in outputs.items()},
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
