#!/usr/bin/env python
"""Convert a deterministic MFCAD subset into cached BRepLens graphs.

Usage:
    python scripts/convert_mfcad.py --limit 3000 --workers 16
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from breplens.ml.mfcad import (
    MFCAD_CLASSES,
    load_face_truth,
    truth_record,
)


def _split(name: str) -> str:
    h = int(hashlib.md5(name.encode()).hexdigest()[:8], 16) % 1000
    if h < 700:
        return "train"
    if h < 850:
        return "val"
    return "test"


def _job(args: tuple[str, str]) -> dict:
    step_path, out_dir = args
    from breplens.adapters.occ import OccAdapter

    src = Path(step_path)
    out = Path(out_dir)
    name = src.stem
    graph_path = out / "graphs" / f"{name}.json"
    truth_path = out / "truth" / f"{name}.json"
    split = _split(name)
    if graph_path.exists() and truth_path.exists():
        t = json.loads(truth_path.read_text())
        t["cached"] = True
        return {"ok": True, "name": name, "split": split, "cached": True}
    try:
        labels = load_face_truth(src)
        adapter = OccAdapter()
        graph = adapter.load_graph(src, name=name, read_step_face_ids=True)
        labels_by_face: list[int] = []
        missing = 0
        for face in graph.faces:
            if face.step_face_id is None or not (0 <= face.step_face_id < len(labels)):
                missing += 1
                labels_by_face.append(-1)
                face.label = None
                face.label_id = None
                continue
            lid = int(labels[face.step_face_id])
            labels_by_face.append(lid)
            face.label = MFCAD_CLASSES[lid]
            face.label_id = lid
        if missing > 0:
            raise ValueError(f"{missing} faces without STEP label ids")
        truth = truth_record(src, labels_by_face, split)
        truth["graph"] = f"graphs/{name}.json"
        graph.metadata.generator = "MFCAD"
        graph.metadata.family = truth["family"]
        graph.metadata.n_features = sum(
            1
            for i, lid in enumerate(labels_by_face)
            if lid != 15 and (i == 0 or labels_by_face[i - 1] != lid)
        )
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        truth_path.parent.mkdir(parents=True, exist_ok=True)
        graph_path.write_text(graph.model_dump_json(indent=1))
        truth_path.write_text(json.dumps(truth, indent=1))
        return {"ok": True, "name": name, "split": split, "n_faces": len(graph.faces)}
    except Exception as exc:
        return {"ok": False, "name": name, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "data" / "raw" / "MFCAD"))
    ap.add_argument("--out", default=str(ROOT / "data" / "mfcad"))
    ap.add_argument("--limit", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=max(1, min(16, (mp.cpu_count() or 4) - 2)))
    args = ap.parse_args()

    step_dir = Path(args.root) / "dataset" / "step"
    all_steps = sorted(step_dir.glob("*.step"))
    if not all_steps:
        raise SystemExit(f"no STEP files under {step_dir}")
    if args.limit > 0 and args.limit < len(all_steps):
        n = args.limit
        idx = [round(i * (len(all_steps) - 1) / (n - 1)) for i in range(n)]
        steps = [all_steps[i] for i in idx]
    else:
        steps = all_steps
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    results = []
    with mp.Pool(processes=args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_job, [(str(p), str(out)) for p in steps]), 1):
            results.append(res)
            if i % 50 == 0 or i == len(steps):
                print(f"[{i}/{len(steps)}] {time.perf_counter() - t0:.1f}s", flush=True)

    ok = [r for r in results if r["ok"]]
    failures = [r for r in results if not r["ok"]]
    label_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    for t in (out / "truth").glob("*.json"):
        data = json.loads(t.read_text())
        split_counts[data["split"]] = split_counts.get(data["split"], 0) + 1
        for k, v in data["label_counts"].items():
            label_counts[k] = label_counts.get(k, 0) + int(v)
    manifest = {
        "dataset": "MFCAD",
        "source": "https://github.com/hducg/MFCAD",
        "license": "MIT",
        "n_available": len(all_steps),
        "n_selected": len(steps),
        "n_converted": len(ok),
        "n_failed": len(failures),
        "split_counts": split_counts,
        "face_label_counts": label_counts,
        "class_names": list(MFCAD_CLASSES),
        "selection": "evenly spaced over sorted STEP filenames",
        "conversion_wall_seconds": round(time.perf_counter() - t0, 2),
        "failures": failures[:50],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in (
                    "n_selected",
                    "n_converted",
                    "n_failed",
                    "split_counts",
                    "conversion_wall_seconds",
                )
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
