#!/usr/bin/env python
"""Head-to-head rules vs GNN vs hybrid on the MFCAD test split (face level).

MFCAD's 16 fine classes are collapsed to 8 macro classes so that the rule
recognizer (which does not distinguish rectangular/triangular/6-sided pockets)
is compared on the same vocabulary as the GNN. This is a deliberately hard,
external dataset; the synthetic dataset is where feature instances and
parameters are scored.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from breplens.ml.evaluation import per_class_metrics, predicted_face_labels
from breplens.ml.inference import load_for_inference, predict_probs
from breplens.ml.mfcad import MFCAD_CLASSES
from breplens.recognition.rules import recognize_rules
from breplens.schema import FeatureType, PartGraph

MACRO = ["stock", "slot", "passage", "step", "pocket", "chamfer", "hole", "fillet"]
MACRO_ID = {m: i for i, m in enumerate(MACRO)}

MFCAD_MACRO = {
    "rectangular_through_slot": "slot",
    "triangular_through_slot": "slot",
    "rectangular_blind_slot": "slot",
    "rectangular_passage": "passage",
    "triangular_passage": "passage",
    "6sides_passage": "passage",
    "rectangular_through_step": "step",
    "2sides_through_step": "step",
    "slanted_through_step": "step",
    "rectangular_blind_step": "step",
    "triangular_blind_step": "step",
    "rectangular_pocket": "pocket",
    "triangular_pocket": "pocket",
    "6sides_pocket": "pocket",
    "chamfer": "chamfer",
    "stock": "stock",
}

RULE_MACRO: dict[str, str] = {
    FeatureType.SLOT.value: "slot",
    FeatureType.POCKET.value: "pocket",
    FeatureType.STEP.value: "step",
    FeatureType.CHAMFER.value: "chamfer",
    FeatureType.THROUGH_HOLE.value: "hole",
    FeatureType.BLIND_HOLE.value: "hole",
    FeatureType.COUNTERBORE.value: "hole",
    FeatureType.COUNTERSINK.value: "hole",
    FeatureType.FILLET.value: "fillet",
    FeatureType.BOSS.value: "hole",
}
RULE_CLASS_IDS = {k: MACRO_ID[v] for k, v in RULE_MACRO.items()}
RULE_CLASS_IDS["base"] = MACRO_ID["stock"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "mfcad"))
    ap.add_argument("--model", default=str(ROOT / "models" / "gnn_mfcad.pt"))
    ap.add_argument("--gnn-eval", default=str(ROOT / "results" / "mfcad_gnn_eval.json"))
    ap.add_argument("--out", default=str(ROOT / "results" / "mfcad_head_to_head.json"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    data = Path(args.data)
    truths = []
    for p in sorted((data / "truth").glob("*.json")):
        t = json.loads(p.read_text())
        if t["split"] == "test":
            truths.append(t)
    if args.limit:
        truths = truths[: args.limit]
    print(f"test parts: {len(truths)}")

    model, class_names, meta = load_for_inference(
        args.model, device="cuda" if torch.cuda.is_available() else "cpu"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"model: {meta.get('architecture')} classes={len(class_names)} device={device}")
    cls_to_macro = [MACRO_ID[MFCAD_MACRO[c]] for c in class_names]

    y_true: list[int] = []
    y_rules: list[int] = []
    y_gnn: list[int] = []
    y_hybrid_rule: list[int] = []
    y_hybrid_gnn: list[int] = []
    y_hybrid_vote: list[int] = []
    rules_seconds = 0.0
    gnn_seconds = 0.0
    n_features = Counter()
    for i, truth in enumerate(truths, 1):
        graph = PartGraph.model_validate_json(
            (data / "graphs" / f"{truth['name']}.json").read_text()
        )
        gt_ids = [int(x) for x in truth["labels"]]
        y_true.extend(MACRO_ID[MFCAD_MACRO[MFCAD_CLASSES[x]]] for x in gt_ids)

        t0 = time.perf_counter()
        fs = recognize_rules(graph)
        rules_seconds += time.perf_counter() - t0
        n_features.update(f.type.value for f in fs.features)
        rules_faces = predicted_face_labels(graph, fs, RULE_CLASS_IDS)

        t0 = time.perf_counter()
        probs = predict_probs(model, graph, device=device)
        gnn_seconds += time.perf_counter() - t0
        gnn_faces = [int(cls_to_macro[int(j)]) for j in probs.argmax(axis=1)]

        y_rules.extend(rules_faces)
        y_gnn.extend(gnn_faces)
        for r, g in zip(rules_faces, gnn_faces, strict=True):
            y_hybrid_rule.append(g if r == MACRO_ID["stock"] else r)
            y_hybrid_gnn.append(r if g == MACRO_ID["stock"] else g)
            if r == g:
                y_hybrid_vote.append(r)
            elif r == MACRO_ID["stock"]:
                y_hybrid_vote.append(g)
            elif g == MACRO_ID["stock"]:
                y_hybrid_vote.append(r)
            else:
                y_hybrid_vote.append(g)
        if i % 50 == 0:
            print(f"[{i}/{len(truths)}]", flush=True)

    methods = {
        "rules": per_class_metrics(y_true, y_rules, MACRO),
        "gnn": per_class_metrics(y_true, y_gnn, MACRO),
        "hybrid_rule_priority": per_class_metrics(y_true, y_hybrid_rule, MACRO),
        "hybrid_gnn_priority": per_class_metrics(y_true, y_hybrid_gnn, MACRO),
        "hybrid_vote": per_class_metrics(y_true, y_hybrid_vote, MACRO),
    }
    for _name, m in methods.items():
        m["n_faces"] = len(y_true)
    gnn_full = json.loads(Path(args.gnn_eval).read_text()) if Path(args.gnn_eval).exists() else {}
    payload = {
        "dataset": {
            "name": "MFCAD",
            "split": "test",
            "n_parts": len(truths),
            "n_faces": len(y_true),
            "classes": MACRO,
            "mapping": MFCAD_MACRO,
            "license": "MIT",
            "source": "https://github.com/hducg/MFCAD",
        },
        "methods": methods,
        "ranked_by_macro_f1": sorted(
            (
                {"method": k, "macro_f1": v["macro_f1"], "accuracy": v["accuracy"]}
                for k, v in methods.items()
            ),
            key=lambda x: x["macro_f1"],
            reverse=True,
        ),
        "rule_feature_counts": dict(n_features),
        "timing": {
            "rules_total_seconds": round(rules_seconds, 3),
            "rules_ms_per_part": round(1000 * rules_seconds / max(1, len(truths)), 3),
            "gnn_total_seconds": round(gnn_seconds, 3),
            "gnn_ms_per_part": round(1000 * gnn_seconds / max(1, len(truths)), 3),
        },
        "gnn_full_16_class": {
            "test_accuracy": gnn_full.get("runs", {})
            .get(gnn_full.get("selected_architecture", ""), {})
            .get("test", {})
            .get("accuracy"),
            "test_macro_f1": gnn_full.get("runs", {})
            .get(gnn_full.get("selected_architecture", ""), {})
            .get("test", {})
            .get("macro_f1"),
            "selected_architecture": gnn_full.get("selected_architecture"),
        },
        "provenance": {
            "script": "scripts/evaluate_head_to_head.py",
            "command": " ".join(sys.argv),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    print(json.dumps(payload["ranked_by_macro_f1"], indent=1))
    print(json.dumps(payload["timing"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
