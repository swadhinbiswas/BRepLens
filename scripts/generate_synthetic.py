#!/usr/bin/env python
"""Generate the synthetic BRepLens dataset with ground truth (resumable).

Example:
    python scripts/generate_synthetic.py --per-family 30 --workers 16
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from breplens.geometry.synthetic import FAMILIES, build_part, make_spec


def _split_for(family: str, index: int) -> str:
    # Deterministic stratified split: 70% train, 15% val, 15% test.
    # hashlib (not hash()) keeps the assignment stable across processes/runs.
    import hashlib

    digest = hashlib.md5(f"{family}:{index}".encode()).hexdigest()
    bucket = int(digest[:8], 16) % 1000
    r = bucket / 1000.0
    if r < 0.70:
        return "train"
    if r < 0.85:
        return "val"
    return "test"


def _job(args: tuple[str, int, str, str]) -> dict:
    family, index, out_dir, seed = args
    out = Path(out_dir)
    name = f"{family}_{index:05d}"
    truth_path = out / "truth" / f"{name}.json"
    graph_path = out / "graphs" / f"{name}.json"
    step_path = out / "step" / f"{name}.step"
    if truth_path.exists() and graph_path.exists() and step_path.exists():
        data = json.loads(truth_path.read_text())
        data["split"] = _split_for(family, index)
        data["cached"] = True
        return data
    rng = random.Random(seed)
    spec = make_spec(family, rng, index)
    spec.seed = seed
    truth = build_part(spec, out)
    truth["split"] = _split_for(family, index)
    truth["cached"] = False
    return truth


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-family", type=int, default=30)
    ap.add_argument("--out", default=str(ROOT / "data" / "synthetic"))
    ap.add_argument("--workers", type=int, default=max(1, min(16, (mp.cpu_count() or 4) - 2)))
    ap.add_argument("--seed", type=int, default=20261124)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for fam_i, family in enumerate(FAMILIES):
        for j in range(args.per_family):
            jobs.append((family, j, str(out), args.seed + fam_i * 1000 + j))
    t0 = time.perf_counter()
    results: list[dict] = []
    with mp.Pool(processes=args.workers) as pool:
        for i, truth in enumerate(pool.imap_unordered(_job, jobs), 1):
            results.append(truth)
            if i % 10 == 0 or i == len(jobs):
                elapsed = time.perf_counter() - t0
                print(f"[{i}/{len(jobs)}] {elapsed:.1f}s", flush=True)
    results.sort(key=lambda r: r["name"])
    manifest = {
        "dataset": "breplens_synthetic",
        "seed": args.seed,
        "per_family": args.per_family,
        "families": list(FAMILIES),
        "n_parts": len(results),
        "generation_wall_seconds": round(time.perf_counter() - t0, 2),
        "parts": [
            {
                "name": r["name"],
                "family": r["family"],
                "split": r["split"],
                "n_faces": r["n_faces"],
                "n_edges": r["n_edges"],
                "n_features": len(r["features"]),
                "step": r["step"],
            }
            for r in results
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    by_split: dict[str, int] = {}
    for r in results:
        by_split[r["split"]] = by_split.get(r["split"], 0) + 1
    (out / "splits.json").write_text(
        json.dumps(
            {
                "split_counts": by_split,
                "assignments": {r["name"]: r["split"] for r in results},
            },
            indent=1,
        )
    )
    print(
        json.dumps(
            {
                "n_parts": len(results),
                "split_counts": by_split,
                "seconds": manifest["generation_wall_seconds"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
