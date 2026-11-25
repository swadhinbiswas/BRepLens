#!/usr/bin/env python
"""Rules vs GNN vs hybrid on the same synthetic test split (features + faces)."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from breplens.geometry.synthetic import CLASS_TO_ID, SYNTHETIC_CLASSES
from breplens.ml.evaluation import evaluate_feature_sets, predicted_face_labels
from breplens.ml.inference import load_for_inference, predict_probs
from breplens.recognition.gnn_recognizer import recognize_gnn, recognize_hybrid
from breplens.recognition.rules import recognize_rules
from breplens.schema import PartGraph


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "synthetic"))
    ap.add_argument("--model", default=str(ROOT / "models" / "gnn_synthetic.pt"))
    ap.add_argument("--out", default=str(ROOT / "results" / "recognition_comparison.json"))
    args = ap.parse_args()
    data = Path(args.data)
    json.loads((data / "manifest.json").read_text())
    split = json.loads((data / "splits.json").read_text())["assignments"]
    names = [n for n, s in sorted(split.items()) if s == "test"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, class_names, meta = load_for_inference(args.model, device=device)
    print(f"model={meta.get('architecture')} classes={class_names} device={device}")
    records: dict[str, list[dict]] = {"rules": [], "gnn": [], "hybrid": []}
    times: dict[str, float] = {"rules": 0.0, "gnn": 0.0, "hybrid": 0.0}
    gnn_face_preds: dict[str, list[int]] = {}
    t0 = time.perf_counter()
    for i, name in enumerate(names, 1):
        graph = PartGraph.model_validate_json((data / "graphs" / f"{name}.json").read_text())
        truth = json.loads((data / "truth" / f"{name}.json").read_text())
        truth["split"] = split[name]
        tr = time.perf_counter()
        rules = recognize_rules(graph)
        times["rules"] += time.perf_counter() - tr
        tr = time.perf_counter()
        gnn = recognize_gnn(graph, model, class_names, device=device)
        times["gnn"] += time.perf_counter() - tr
        tr = time.perf_counter()
        hybrid = recognize_hybrid(graph, model, class_names, device=device)
        times["hybrid"] += time.perf_counter() - tr
        probs = predict_probs(model, graph, device=device)
        gnn_face_preds[name] = [int(x) for x in probs.argmax(axis=1)]
        for key, fs in (("rules", rules), ("gnn", gnn), ("hybrid", hybrid)):
            records[key].append(
                {
                    "name": name,
                    "family": truth["family"],
                    "split": "test",
                    "graph": graph,
                    "truth": truth,
                    "pred": fs,
                }
            )
        if i % 50 == 0:
            print(f"[{i}/{len(names)}]", flush=True)
    results: dict[str, dict] = {}
    for key, recs in records.items():
        metrics = evaluate_feature_sets(recs, class_names=SYNTHETIC_CLASSES, class_ids=CLASS_TO_ID)
        if key in ("gnn", "hybrid"):
            from breplens.ml.evaluation import per_class_metrics

            if key == "gnn":
                y_true = [f.label_id or 0 for r in recs for f in r["graph"].faces]
                y_pred = [cid for r in recs for cid in gnn_face_preds[r["name"]]]
            else:
                y_true = []
                y_pred = []
                for r in recs:
                    y_true.extend(f.label_id or 0 for f in r["graph"].faces)
                    y_pred.extend(
                        predicted_face_labels(
                            r["graph"],
                            r["pred"],
                            CLASS_TO_ID,
                            fallback=gnn_face_preds[r["name"]],
                        )
                    )
            metrics["face_direct"] = per_class_metrics(y_true, y_pred, SYNTHETIC_CLASSES)
        metrics["seconds_total"] = round(times[key], 3)
        metrics["ms_per_part"] = round(1000 * times[key] / max(1, len(recs)), 3)
        metrics["by_family"] = {
            fam: evaluate_feature_sets(
                [r for r in recs if r["family"] == fam],
                class_names=SYNTHETIC_CLASSES,
                class_ids=CLASS_TO_ID,
            )["feature"]
            for fam in sorted({r["family"] for r in recs})
        }
        results[key] = metrics
    payload = {
        "dataset": "breplens_synthetic",
        "split": "test",
        "n_parts": len(names),
        "class_names": SYNTHETIC_CLASSES,
        "methods": results,
        "model": {
            "checkpoint": str(Path(args.model).relative_to(ROOT)),
            "metadata": meta,
            "class_names": class_names,
        },
        "provenance": {
            "script": "scripts/evaluate_synthetic_three_ways.py",
            "command": " ".join(sys.argv),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "wall_seconds": round(time.perf_counter() - t0, 2),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    for method, m in results.items():
        print(
            method,
            {
                "feature_f1": m["feature"]["f1"],
                "feature_precision": m["feature"]["precision"],
                "feature_recall": m["feature"]["recall"],
                "face_accuracy": m["face"]["accuracy"],
                "face_macro_f1": m["face"]["macro_f1"],
                "exact_part_rate": m["feature"]["exact_part_rate"],
                "ms_per_part": m["ms_per_part"],
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
