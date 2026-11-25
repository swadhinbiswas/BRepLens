"""Dataset loading, batching and training utilities for face-classification GNNs."""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from breplens.geometry.aag import to_pyg_data
from breplens.ml.model import FaceGNN, GNNConfig
from breplens.schema import PartGraph


@dataclass
class DatasetBundle:
    train: list[Any]
    val: list[Any]
    test: list[Any]
    class_names: list[str]
    class_ids: dict[str, int]
    families: dict[str, list[str]]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def load_dataset(
    data_dir: str | Path,
    *,
    class_names: Sequence[str],
    split_key: str = "split",
    limit: int | None = None,
) -> DatasetBundle:
    """Load cached graph JSONs and build PyG Data objects with labels."""
    from breplens.geometry.synthetic import CLASS_TO_ID as SYNTHETIC_IDS

    _ = SYNTHETIC_IDS  # kept for callers that import through this module
    data = Path(data_dir)
    manifest = json.loads((data / "manifest.json").read_text())
    split_by_name: dict[str, str] = {}
    splits_path = data / "splits.json"
    if splits_path.exists():
        split_by_name = json.loads(splits_path.read_text()).get("assignments", {})
    elif "parts" in manifest:
        split_by_name = {p["name"]: p["split"] for p in manifest["parts"]}
    if not split_by_name:
        for tpath in sorted((data / "truth").glob("*.json")):
            rec = json.loads(tpath.read_text())
            split_by_name[rec["name"]] = rec.get("split", "train")
    names = sorted(split_by_name)
    if limit is not None:
        names = names[:limit]
    class_ids = {name: i for i, name in enumerate(class_names)}
    bundle = DatasetBundle([], [], [], list(class_names), class_ids, {"train": [], "val": [], "test": []})
    for name in names:
        graph_path = data / "graphs" / f"{name}.json"
        if not graph_path.exists():
            continue
        graph = PartGraph.model_validate_json(graph_path.read_text())
        for face in graph.faces:
            if face.label is None:
                face.label_id = class_ids.get("stock", class_ids.get("base", 0))
            elif face.label in class_ids:
                face.label_id = class_ids[face.label]
            else:
                face.label_id = class_ids.get("stock", class_ids.get("base", 0))
        data_obj = to_pyg_data(graph)
        split = split_by_name.get(name, "train")
        getattr(bundle, split if split in ("train", "val", "test") else "train").append(data_obj)
        bundle.families.setdefault(split, []).append(graph.metadata.family or "")
    return bundle


def class_weights(dataset: Sequence[Any], num_classes: int) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.float64)
    for data in dataset:
        for y in data.y.tolist():
            if 0 <= y < num_classes:
                counts[y] += 1
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / np.sqrt(counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def evaluate_model(
    model: FaceGNN,
    loader: Any,
    device: torch.device,
    num_classes: int,
    class_names: Sequence[str] | None = None,
) -> tuple[dict[str, float], list[int], list[int], list[int]]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[float] = []
    losses = []
    crit = nn.CrossEntropyLoss()
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        loss = crit(logits, batch.y)
        losses.append(float(loss.detach().cpu()))
        probs = torch.softmax(logits, dim=-1)
        pred = probs.argmax(dim=-1)
        conf = probs.max(dim=-1).values
        y_true.extend(batch.y.cpu().tolist())
        y_pred.extend(pred.cpu().tolist())
        y_prob.extend((conf * 1000).round().to(torch.int64).cpu().tolist())
    from breplens.ml.evaluation import per_class_metrics

    names = list(class_names) if class_names is not None else [str(i) for i in range(num_classes)]
    metrics = per_class_metrics(y_true, y_pred, names)
    metrics["loss"] = round(float(np.mean(losses)), 5) if losses else 0.0
    return metrics, y_true, y_pred, y_prob


def train_model(
    bundle: DatasetBundle,
    *,
    config: GNNConfig,
    seed: int = 42,
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 5e-3,
    weight_decay: float = 5e-4,
    patience: int = 8,
    device: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train one GNN configuration with early stopping on validation macro-F1."""
    from torch_geometric.loader import DataLoader

    set_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    num_classes = len(bundle.class_names)
    model = FaceGNN(num_classes, config).to(dev)
    train_loader = DataLoader(bundle.train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(bundle.val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(bundle.test, batch_size=batch_size, shuffle=False)
    weights = class_weights(bundle.train, num_classes).to(dev)
    crit = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    use_amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_state: dict[str, torch.Tensor] | None = None
    best_val = -1.0
    best_epoch = 0
    history: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    epoch_times: list[float] = []
    if use_amp:
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(1, epochs + 1):
        te = time.perf_counter()
        model.train()
        losses = []
        for batch in train_loader:
            batch = batch.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(batch)
                loss = crit(logits, batch.y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        sched.step()
        val_metrics, _, _, _ = evaluate_model(
            model, val_loader, dev, num_classes, bundle.class_names
        )
        epoch_times.append(time.perf_counter() - te)
        rec = {
            "epoch": epoch,
            "train_loss": round(float(np.mean(losses)), 5),
            "val_macro_f1": val_metrics["macro_f1"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(rec)
        if verbose:
            print(
                f"epoch {epoch:03d} loss={rec['train_loss']:.4f} "
                f"val_acc={rec['val_accuracy']:.4f} val_f1={rec['val_macro_f1']:.4f}",
                flush=True,
            )
        if val_metrics["macro_f1"] > best_val:
            best_val = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            if verbose:
                print(f"early stop at epoch {epoch} (best {best_epoch})", flush=True)
            break
    train_seconds = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    val_metrics, _, _, _ = evaluate_model(
        model, val_loader, dev, num_classes, bundle.class_names
    )
    test_metrics, y_true, y_pred, _ = evaluate_model(
        model, test_loader, dev, num_classes, bundle.class_names
    )
    peak_mem = 0.0
    if use_amp:
        peak_mem = float(torch.cuda.max_memory_allocated() / (1024**2))
    return {
        "config": config.to_dict(),
        "seed": seed,
        "device": str(dev),
        "class_names": bundle.class_names,
        "n_parameters": model.n_parameters,
        "n_train": len(bundle.train),
        "n_val": len(bundle.val),
        "n_test": len(bundle.test),
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val,
        "train_seconds": round(train_seconds, 3),
        "epoch_seconds_mean": round(float(np.mean(epoch_times)), 4) if epoch_times else 0.0,
        "peak_gpu_mb": round(peak_mem, 1),
        "history": history,
        "val": val_metrics,
        "test": test_metrics,
        "y_true": y_true,
        "y_pred": y_pred,
        "model": model,
    }


__all__ = [
    "DatasetBundle",
    "class_weights",
    "evaluate_model",
    "load_dataset",
    "set_seed",
    "train_model",
]
