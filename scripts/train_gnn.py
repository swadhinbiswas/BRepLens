#!/usr/bin/env python
"""Train face-classification GNNs on synthetic or MFCAD data.

Examples:
    python scripts/train_gnn.py --dataset synthetic --epochs 40
    python scripts/train_gnn.py --dataset mfcad --epochs 25
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

import torch

from breplens.geometry.synthetic import CLASS_TO_ID, SYNTHETIC_CLASSES
from breplens.ml.mfcad import MFCAD_CLASSES
from breplens.ml.model import GNNConfig, save_checkpoint
from breplens.ml.train import load_dataset, train_model


def _model_size_mb(path: Path) -> float:
    return round(path.stat().st_size / (1024 * 1024), 4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["synthetic", "mfcad"], default="synthetic")
    ap.add_argument("--data", default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--arch", choices=["both", "gcn", "edge_gnn"], default="both")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--model-out", default=None)
    args = ap.parse_args()

    if args.dataset == "synthetic":
        data_dir = Path(args.data or ROOT / "data" / "synthetic")
        class_names = list(SYNTHETIC_CLASSES)
        dict(CLASS_TO_ID)
        out = Path(args.out or ROOT / "results" / "gnn_eval.json")
        model_out = Path(args.model_out or ROOT / "models" / "gnn_synthetic.pt")
    else:
        data_dir = Path(args.data or ROOT / "data" / "mfcad")
        class_names = list(MFCAD_CLASSES)
        {n: i for i, n in enumerate(class_names)}
        out = Path(args.out or ROOT / "results" / "mfcad_gnn_eval.json")
        model_out = Path(args.model_out or ROOT / "models" / "gnn_mfcad.pt")

    t0 = time.perf_counter()
    bundle = load_dataset(data_dir, class_names=class_names, limit=args.limit)
    load_seconds = time.perf_counter() - t0
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "train": len(bundle.train),
                "val": len(bundle.val),
                "test": len(bundle.test),
                "classes": len(class_names),
                "load_seconds": round(load_seconds, 2),
                "cuda": torch.cuda.is_available(),
                "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            },
            indent=1,
        )
    )
    architectures = ["gcn", "edge_gnn"] if args.arch == "both" else [args.arch]
    runs: dict[str, dict] = {}
    best: dict | None = None
    for arch in architectures:
        cfg = GNNConfig(architecture=arch, hidden=64, layers=3, dropout=0.15)
        print(f"=== training {arch} ===", flush=True)
        result = train_model(
            bundle,
            config=cfg,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
        )
        model = result.pop("model")
        y_true = result.pop("y_true")
        y_pred = result.pop("y_pred")
        result["y_true"] = y_true
        result["y_pred"] = y_pred
        runs[arch] = result
        if best is None or result["val"]["macro_f1"] > best[0]["val"]["macro_f1"]:
            best = (result, model, cfg)
        print(
            f"{arch}: test acc={result['test']['accuracy']:.4f} macro_f1={result['test']['macro_f1']:.4f}",
            flush=True,
        )
    assert best is not None
    best_result, best_model, best_cfg = best
    checkpoint = save_checkpoint(
        model_out,
        best_model.cpu(),
        class_names=class_names,
        metadata={
            "dataset": args.dataset,
            "architecture": best_cfg.architecture,
            "test_accuracy": best_result["test"]["accuracy"],
            "test_macro_f1": best_result["test"]["macro_f1"],
            "seed": args.seed,
            "trained_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    # CPU inference latency per graph (deployment-relevant)
    import numpy as np
    from torch_geometric.loader import DataLoader

    from breplens.ml.model import load_checkpoint

    cpu_model, _ = load_checkpoint(checkpoint, device="cpu")
    sample = bundle.test[: min(80, len(bundle.test))]
    loader = DataLoader(sample, batch_size=1, shuffle=False)
    latencies = []
    with torch.no_grad():
        for batch in loader:
            t = time.perf_counter()
            cpu_model(batch)
            latencies.append((time.perf_counter() - t) * 1000.0)
    payload = {
        "provenance": {
            "script": "scripts/train_gnn.py",
            "command": " ".join(sys.argv),
            "dataset": args.dataset,
            "data_dir": str(data_dir.relative_to(ROOT)),
            "seed": args.seed,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "host": platform.node(),
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "dataset": {
            "name": args.dataset,
            "n_train": len(bundle.train),
            "n_val": len(bundle.val),
            "n_test": len(bundle.test),
            "class_names": class_names,
            "load_seconds": round(load_seconds, 3),
        },
        "runs": runs,
        "selected_architecture": best_cfg.architecture,
        "checkpoint": {
            "path": str(checkpoint.relative_to(ROOT)),
            "size_mb": _model_size_mb(checkpoint),
            "n_parameters": best_result["n_parameters"],
        },
        "cpu_inference_ms_per_graph": {
            "mean": round(float(np.mean(latencies)), 3),
            "median": round(float(np.median(latencies)), 3),
            "p95": round(float(np.percentile(latencies, 95)), 3),
            "n": len(latencies),
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1))
    print(
        json.dumps(
            {
                "selected": best_cfg.architecture,
                "test_accuracy": best_result["test"]["accuracy"],
                "test_macro_f1": best_result["test"]["macro_f1"],
                "train_seconds": best_result["train_seconds"],
                "cpu_ms_per_graph": payload["cpu_inference_ms_per_graph"]["median"],
                "checkpoint_mb": payload["checkpoint"]["size_mb"],
                "out": str(out.relative_to(ROOT)),
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
