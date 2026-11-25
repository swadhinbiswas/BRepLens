#!/usr/bin/env python
"""Evaluate the rule-based recognizer against construction ground truth.

Usage:
    python scripts/evaluate_rules.py --split test
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from breplens.geometry.synthetic import CLASS_TO_ID, SYNTHETIC_CLASSES
from breplens.ml.evaluation import (
    dataset_summary,
    evaluate_feature_sets,
)
from breplens.recognition.rules import recognize_rules
from breplens.schema import PartGraph


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "synthetic"))
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--out", default=str(ROOT / "results" / "rules_eval.json"))
    args = ap.parse_args()

    data = Path(args.data)
    manifest = json.loads((data / "manifest.json").read_text())
    names = [
        p["name"] for p in manifest["parts"] if args.split == "all" or p["split"] == args.split
    ]
    split_by_name = {p["name"]: p["split"] for p in manifest["parts"]}
    records = []
    t0 = time.perf_counter()
    graph_ms = 0.0
    for i, name in enumerate(names, 1):
        graph = PartGraph.model_validate_json((data / "graphs" / f"{name}.json").read_text())
        truth = json.loads((data / "truth" / f"{name}.json").read_text())
        tg = time.perf_counter()
        fs = recognize_rules(graph)
        graph_ms += (time.perf_counter() - tg) * 1000.0
        records.append(
            {
                "name": name,
                "family": truth["family"],
                "split": split_by_name.get(name, "unknown"),
                "graph": graph,
                "truth": truth,
                "pred": fs,
            }
        )
        if i % 20 == 0:
            print(f"[{i}/{len(names)}]", flush=True)
    wall = time.perf_counter() - t0
    metrics = evaluate_feature_sets(records, class_names=SYNTHETIC_CLASSES, class_ids=CLASS_TO_ID)
    metrics["dataset"] = dataset_summary(records)
    metrics["by_family"] = {
        fam: evaluate_feature_sets(
            [r for r in records if r["family"] == fam],
            class_names=SYNTHETIC_CLASSES,
            class_ids=CLASS_TO_ID,
        )["feature"]
        for fam in sorted({r["family"] for r in records})
    }
    metrics["timing"] = {
        "wall_seconds": round(wall, 3),
        "recognition_mean_ms": round(graph_ms / max(1, len(records)), 3),
        "recognition_total_ms": round(graph_ms, 3),
        "parts": len(records),
    }
    metrics["provenance"] = {
        "script": "scripts/evaluate_rules.py",
        "dataset": str(data.relative_to(ROOT)),
        "split": args.split,
        "seed": manifest.get("seed"),
        "python": platform.python_version(),
        "host": platform.node(),
        "command": " ".join(sys.argv),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=1))
    print(
        json.dumps(
            {
                "split": args.split,
                "n_parts": metrics["n_parts"],
                "feature_f1": metrics["feature"]["f1"],
                "precision": metrics["feature"]["precision"],
                "recall": metrics["feature"]["recall"],
                "face_accuracy": metrics["face"]["accuracy"],
                "face_macro_f1": metrics["face"]["macro_f1"],
                "exact_part_rate": metrics["feature"]["exact_part_rate"],
                "recognition_mean_ms": metrics["timing"]["recognition_mean_ms"],
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
