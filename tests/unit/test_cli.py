from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_cli_recognize_and_schema(tmp_path: Path) -> None:
    out = tmp_path / "features.json"
    r = subprocess.run(
        [
            sys.executable,
            "-m",
            "breplens.cli",
            "recognize",
            str(ROOT / "samples" / "sample_04_pocket.step"),
            "--mode",
            "rules",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert payload["n_features"] >= 1
    assert json.loads(out.read_text())["schema_version"] == "1.0.0"
    r2 = subprocess.run(
        [sys.executable, "-m", "breplens.cli", "schema", "--out", str(tmp_path / "schema.json")],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert r2.returncode == 0, r2.stderr
    assert (tmp_path / "schema.json").exists()
