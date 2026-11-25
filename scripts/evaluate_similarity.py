#!/usr/bin/env python
"""Build indexes and evaluate retrieval precision@k on synthetic and MFCAD data."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from breplens.recognition.rules import recognize_rules
from breplens.schema import PartGraph
from breplens.similarity.signature import (
    SignatureIndex,
    part_signature,
    precision_at_k,
)


def _load_split(data: Path) -> tuple[dict[str, str], list[dict]]:
    split: dict[str, str] = {}
    if (data / "splits.json").exists():
        split = json.loads((data / "splits.json").read_text())["assignments"]
    truths: list[dict] = []
    for p in sorted((data / "truth").glob("*.json")):
        t = json.loads(p.read_text())
        if t.get("dataset") == "MFCAD" and "labels" in t:
            from breplens.ml.mfcad import macro_family

            t = dict(t)
            t["family"] = macro_family([int(x) for x in t["labels"]])
        truths.append(t)
        split.setdefault(t["name"], t.get("split", "train"))
    return split, truths


def _evaluate(name: str, data: Path, *, max_library: int = 0, max_queries: int = 0) -> dict:
    split, truths = _load_split(data)
    train = [t for t in truths if split.get(t["name"]) in ("train", "val")]
    test = [t for t in truths if split.get(t["name"]) == "test"]
    if max_library:
        train = train[:max_library]
    if max_queries:
        test = test[:max_queries]
    index = SignatureIndex()
    build_times = []
    for t in train:
        graph = PartGraph.model_validate_json((data / "graphs" / f"{t['name']}.json").read_text())
        tr = time.perf_counter()
        fs = recognize_rules(graph)
        index.add_graph(
            graph,
            name=t["name"],
            family=t.get("family", ""),
            features=fs,
            metadata={"split": "library"},
        )
        build_times.append(time.perf_counter() - tr)
    q_times = []
    vectors: list[np.ndarray] = []
    families: list[str] = []
    ids: list[str] = []
    for t in test:
        graph = PartGraph.model_validate_json((data / "graphs" / f"{t['name']}.json").read_text())
        tr = time.perf_counter()
        fs = recognize_rules(graph)
        vectors.append(part_signature(graph, fs))
        q_times.append(time.perf_counter() - tr)
        families.append(t.get("family", ""))
        ids.append(t["name"])
    metrics: dict[str, float] = {}
    for k in (1, 5, 10):
        metrics.update(precision_at_k(vectors, families, ids, index, k))
    for k in (1, 5, 10):
        metrics.update(precision_at_k(vectors, families, ids, index, k, min_jaccard=0.5))
    # cosine separation between relevant and irrelevant neighbours
    relevant, irrelevant = [], []
    for q, fam in zip(vectors, families, strict=True):
        for e in index.entries:
            s = float(np.dot(q, e.vector))
            (relevant if e.family == fam else irrelevant).append(s)
    metrics["mean_cosine_same_family"] = round(float(np.mean(relevant)), 4) if relevant else 0.0
    metrics["mean_cosine_other_family"] = (
        round(float(np.mean(irrelevant)), 4) if irrelevant else 0.0
    )
    # one worked example for docs
    example = None
    if vectors:
        hits = index.search(vectors[0], k=5, exclude_id=ids[0])
        example = {
            "query": ids[0],
            "query_family": families[0],
            "hits": [
                {"name": h.name, "family": h.family, "score": round(h.score, 4)} for h in hits
            ],
        }
    return {
        "dataset": name,
        "n_library": len(index.entries),
        "n_queries": len(vectors),
        "dim": int(index.dim or 0),
        "metrics": metrics,
        "timing": {
            "index_build_seconds": round(float(np.sum(build_times)), 3),
            "index_build_ms_per_part": round(1000 * float(np.mean(build_times)), 3)
            if build_times
            else 0.0,
            "query_ms_mean": round(1000 * float(np.mean(q_times)), 3) if q_times else 0.0,
        },
        "example": example,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "similarity_eval.json"))
    ap.add_argument("--max-library", type=int, default=0)
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--demo-out", default=str(ROOT / "demo" / "library.json"))
    ap.add_argument("--demo-parts", type=int, default=60)
    args = ap.parse_args()
    t0 = time.perf_counter()
    synthetic = _evaluate(
        "breplens_synthetic",
        ROOT / "data" / "synthetic",
        max_library=args.max_library,
        max_queries=args.max_queries,
    )
    mfcad = _evaluate(
        "MFCAD",
        ROOT / "data" / "mfcad",
        max_library=args.max_library,
        max_queries=args.max_queries,
    )
    payload = {
        "synthetic": synthetic,
        "mfcad": mfcad,
        "config": {"signature_dim": synthetic["dim"], "k": [1, 5, 10]},
        "provenance": {
            "script": "scripts/evaluate_similarity.py",
            "command": " ".join(sys.argv),
            "python": platform.python_version(),
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "wall_seconds": round(time.perf_counter() - t0, 2),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    # Demo library: deterministic spread across synthetic families, small JSON.
    demo_index = SignatureIndex()
    split, truths = _load_split(ROOT / "data" / "synthetic")
    limit = args.demo_parts
    candidates = [t for t in truths if split.get(t["name"]) in ("train", "val")]
    candidates = candidates[: max(1, min(limit, len(candidates)))]
    for t in candidates:
        graph = PartGraph.model_validate_json(
            (ROOT / "data" / "synthetic" / "graphs" / f"{t['name']}.json").read_text()
        )
        demo_index.add_graph(
            graph, name=t["name"], family=t.get("family", ""), features=recognize_rules(graph)
        )
    demo_path = Path(args.demo_out)
    demo_index.save(demo_path)
    print(
        json.dumps(
            {
                "synthetic": synthetic["metrics"],
                "mfcad": mfcad["metrics"],
                "demo_library": len(demo_index.entries),
                "demo_bytes": demo_path.stat().st_size,
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
