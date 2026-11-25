#!/usr/bin/env python
"""Record the environment audit used by docs/SETUP.md."""

from __future__ import annotations

import importlib.metadata as md
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as exc:
        return f"unavailable: {type(exc).__name__}"


def main() -> int:
    gpu = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ]
    )
    packages = {}
    for name in (
        "cadquery-ocp",
        "cadquery",
        "torch",
        "torch-geometric",
        "numpy",
        "networkx",
        "pydantic",
        "streamlit",
        "plotly",
        "fastapi",
        "sqlalchemy",
        "rq",
        "redis",
    ):
        try:
            packages[name] = md.version(name)
        except md.PackageNotFoundError:
            packages[name] = None
    disk = shutil.disk_usage(ROOT)
    payload = {
        "os": platform.platform(),
        "platform": sys.platform,
        "kernel": platform.release(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "cpu": platform.processor(),
        "cpu_count": os.cpu_count(),
        "memory_gb": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3, 2)
        if hasattr(os, "sysconf")
        else None,
        "gpu": gpu,
        "disk_free_gb": round(disk.free / 1024**3, 2),
        "disk_total_gb": round(disk.total / 1024**3, 2),
        "packages": packages,
        "provenance": {
            "script": "scripts/env_audit.py",
            "command": " ".join(sys.argv),
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }
    out = ROOT / "results" / "env_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    print(json.dumps(payload, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
