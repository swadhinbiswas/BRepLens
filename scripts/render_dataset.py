#!/usr/bin/env python
"""Render multi-view images for the synthetic part-family dataset (resumable)."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from breplens.ml.views import VIEW_NAMES


def _job(args: tuple[str, str, int, float]) -> dict:
    step_path, out_dir, size, deflection = args
    import numpy as np

    from breplens.adapters.occ import OccAdapter
    from breplens.ml.views import render_views

    src = Path(step_path)
    out = Path(out_dir) / f"{src.stem}.npz"
    if out.exists():
        return {"ok": True, "name": src.stem, "cached": True}
    try:
        mesh = OccAdapter().tessellate(src, deflection=deflection)
        views = render_views(mesh, size=size)
        if views.shape != (len(VIEW_NAMES), size, size):
            raise ValueError(f"unexpected view shape {views.shape}")
        if int(views.max()) == 0:
            raise ValueError("empty render")
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, views=views)
        return {"ok": True, "name": src.stem, "cached": False}
    except Exception as exc:
        return {"ok": False, "name": src.stem, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "synthetic"))
    ap.add_argument("--out", default=str(ROOT / "data" / "renders"))
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--deflection", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=max(1, min(16, (mp.cpu_count() or 4) - 2)))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    data = Path(args.data)
    source = json.loads((data / "manifest.json").read_text())
    parts = source["parts"]
    if args.limit:
        parts = parts[: args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(str(data / p["step"]), str(out), args.size, args.deflection) for p in parts]
    t0 = time.perf_counter()
    results = []
    with mp.Pool(processes=args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_job, jobs), 1):
            results.append(res)
            if i % 50 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] {time.perf_counter() - t0:.1f}s", flush=True)
    ok = [r for r in results if r["ok"]]
    failures = [r for r in results if not r["ok"]]
    split_counts: dict[str, int] = {}
    families: dict[str, int] = {}
    for p in parts:
        split_counts[p["split"]] = split_counts.get(p["split"], 0) + 1
        families[p["family"]] = families.get(p["family"], 0) + 1
    manifest = {
        "dataset": source["dataset"],
        "view_names": list(VIEW_NAMES),
        "size": args.size,
        "deflection": args.deflection,
        "workers": args.workers,
        "n_parts": len(parts),
        "n_rendered": len(ok),
        "n_failed": len(failures),
        "split_counts": split_counts,
        "families": families,
        "failures": failures[:20],
        "render_wall_seconds": round(time.perf_counter() - t0, 2),
        "provenance": {
            "script": "scripts/render_dataset.py",
            "command": " ".join(sys.argv),
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in (
                    "n_parts",
                    "n_rendered",
                    "n_failed",
                    "render_wall_seconds",
                    "split_counts",
                )
            }
        )
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
