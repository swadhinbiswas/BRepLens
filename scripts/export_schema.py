#!/usr/bin/env python
"""Export the versioned feature JSON Schema and a sample FeatureSet."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from breplens.schema import SCHEMA_VERSION, FeatureSet, PartGraph


def main() -> int:
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    out = {
        "schema_version": SCHEMA_VERSION,
        "feature_set": FeatureSet.model_json_schema(),
        "part_graph": PartGraph.model_json_schema(),
    }
    path = docs / "schema.v1.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
