#!/usr/bin/env python
"""Train/evaluate the optional multi-view CNN vs the graph family classifier."""

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
import torch

from breplens.ml.family import (
    FAMILY_NAMES,
    load_graph_splits,
    load_image_splits,
    save_cnn,
    save_gnn,
    train_cnn,
    train_gnn,
)
from breplens.ml.train import set_seed


def _cpu_cnn_ms(model, x: torch.Tensor, n: int = 30) -> float:
    model.cpu().eval()
    times = []
    with torch.no_grad():
        for _ in range(n):
            t = time.perf_counter()
            model(x[:1])
            times.append((time.perf_counter() - t) * 1000)
    return round(float(np.median(times)), 3)


def _cpu_gnn_ms(model, graphs, n: int = 30) -> float:
    from torch_geometric.loader import DataLoader

    model.cpu().eval()
    times = []
    with torch.no_grad():
        for i in range(n):
            batch = next(iter(DataLoader([graphs[i % len(graphs)]], batch_size=1)))
            t = time.perf_counter()
            model(batch)
            times.append((time.perf_counter() - t) * 1000)
    return round(float(np.median(times)), 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "synthetic"))
    ap.add_argument("--renders", default=str(ROOT / "data" / "renders"))
    ap.add_argument("--epochs-cnn", type=int, default=40)
    ap.add_argument("--epochs-gnn", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "results" / "cnn_eval.json"))
    ap.add_argument("--cnn-model", default=str(ROOT / "models" / "family_cnn.pt"))
    ap.add_argument("--gnn-model", default=str(ROOT / "models" / "family_gnn.pt"))
    args = ap.parse_args()
    set_seed(args.seed)
    image_splits, image_meta = load_image_splits(args.data, args.renders)
    graph_splits = load_graph_splits(args.data)
    print(
        json.dumps(
            {
                "images": {k: int(v[1].numel()) for k, v in image_splits.items()},
                "graphs": {k: len(v) for k, v in graph_splits.items()},
                "classes": list(FAMILY_NAMES),
            },
            indent=1,
        )
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()
    cnn = train_cnn(image_splits, seed=args.seed, epochs=args.epochs_cnn, device=device)
    gnn = train_gnn(graph_splits, seed=args.seed, epochs=args.epochs_gnn, device=device)
    total_train = time.perf_counter() - t0
    cnn_model = cnn.pop("model")
    gnn_model = gnn.pop("model")
    cnn_meta = {
        "dataset": "breplens_synthetic",
        "method": "multi-view CNN",
        "test_accuracy": cnn["test"]["accuracy"],
        "test_macro_f1": cnn["test"]["macro_f1"],
        "seed": args.seed,
        "trained_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "image_norm": {k: image_meta[k] for k in ("mean", "std")},
        "view_names": image_meta["render_manifest"]["view_names"],
    }
    gnn_meta = {
        "dataset": "breplens_synthetic",
        "method": "edge-aware GNN + global pooling",
        "test_accuracy": gnn["test"]["accuracy"],
        "test_macro_f1": gnn["test"]["macro_f1"],
        "seed": args.seed,
        "trained_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    cnn_path = save_cnn(args.cnn_model, cnn_model, metadata=cnn_meta)
    gnn_path = save_gnn(args.gnn_model, gnn_model, metadata=gnn_meta)
    cnn_cpu_ms = _cpu_cnn_ms(cnn_model, image_splits["test"][0])
    gnn_cpu_ms = _cpu_gnn_ms(gnn_model, graph_splits["test"])
    methods = {
        "cnn_multiview": {
            **cnn,
            "model_mb": round(cnn_path.stat().st_size / 1024**2, 4),
            "cpu_ms_per_part": cnn_cpu_ms,
            "views": list(image_meta["render_manifest"]["view_names"]),
        },
        "graph_gnn": {
            **gnn,
            "model_mb": round(gnn_path.stat().st_size / 1024**2, 4),
            "cpu_ms_per_part": gnn_cpu_ms,
        },
    }
    ranked = sorted(
        (
            {"method": k, "accuracy": v["test"]["accuracy"], "macro_f1": v["test"]["macro_f1"]}
            for k, v in methods.items()
        ),
        key=lambda x: (x["macro_f1"], x["accuracy"]),
        reverse=True,
    )
    payload = {
        "dataset": {
            "name": "breplens_synthetic",
            "n_train": image_meta["n_train"],
            "n_val": image_meta["n_val"],
            "n_test": image_meta["n_test"],
            "classes": list(FAMILY_NAMES),
            "rendering": image_meta["render_manifest"],
        },
        "methods": methods,
        "comparison": {
            "ranked": ranked,
            "winner": ranked[0]["method"],
            "note": "Same parts, split and seed; graph method uses no images.",
        },
        "total_training_seconds": round(total_train, 3),
        "provenance": {
            "script": "scripts/train_family_models.py",
            "command": " ".join(sys.argv),
            "seed": args.seed,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_geometric": __import__("torch_geometric").__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "host": platform.node(),
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    print(
        json.dumps(
            {
                "cnn_test_acc": cnn["test"]["accuracy"],
                "cnn_macro_f1": cnn["test"]["macro_f1"],
                "gnn_test_acc": gnn["test"]["accuracy"],
                "gnn_macro_f1": gnn["test"]["macro_f1"],
                "winner": ranked[0]["method"],
                "cnn_ms": cnn_cpu_ms,
                "gnn_ms": gnn_cpu_ms,
                "out": str(out.relative_to(ROOT)),
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
