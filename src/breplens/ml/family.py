"""Part-family classification: multi-view CNN vs graph classifier.

The comparison uses the same synthetic parts and the same train/val/test split.
Images are six orthographic shaded views (±X/±Y/±Z); the graph model
reuses the AAG node/edge features and edge-aware message passing with global
pooling. No labels leak between splits.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GraphLoader
from torch_geometric.nn import global_max_pool, global_mean_pool

from breplens.geometry.aag import EDGE_FEATURE_NAMES, FEATURE_NAMES, to_pyg_data
from breplens.ml.evaluation import per_class_metrics
from breplens.ml.model import EdgeAwareLayer
from breplens.ml.views import VIEW_NAMES
from breplens.schema import PartGraph

FAMILY_NAMES: tuple[str, ...] = (
    "plate_holes",
    "plate_blind",
    "counterbore",
    "countersink",
    "pocket",
    "slot",
    "stepped_block",
    "boss_plate",
    "fillet_block",
    "chamfer_block",
    "bracket_mixed",
)
FAMILY_IDS = {name: i for i, name in enumerate(FAMILY_NAMES)}


class PartFamilyCNN(nn.Module):
    """Small CNN over six stacked orthographic views."""

    def __init__(
        self, num_classes: int = len(FAMILY_NAMES), in_channels: int = len(VIEW_NAMES)
    ) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.head(self.features(x)))

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class FamilyGNN(nn.Module):
    """Edge-aware GNN with global mean+max pooling for whole-part classification."""

    def __init__(
        self, num_classes: int = len(FAMILY_NAMES), hidden: int = 64, layers: int = 3
    ) -> None:
        super().__init__()
        self.input = nn.Linear(len(FEATURE_NAMES), hidden)
        self.layers = nn.ModuleList(
            [EdgeAwareLayer(hidden, len(EDGE_FEATURE_NAMES)) for _ in range(layers)]
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x = self.input(data.x)
        for layer in self.layers:
            x = layer(x, data.edge_index, data.edge_attr)
        pooled = torch.cat([global_mean_pool(x, data.batch), global_max_pool(x, data.batch)], dim=1)
        return cast(torch.Tensor, self.head(pooled))

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def _macro_f1(y_true: Sequence[int], y_pred: Sequence[int], n: int) -> float:
    names = [str(i) for i in range(n)]
    return float(per_class_metrics(list(y_true), list(y_pred), names)["macro_f1"])


def load_image_splits(
    data_dir: str | Path, renders_dir: str | Path
) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], dict[str, Any]]:
    data = Path(data_dir)
    renders = Path(renders_dir)
    manifest = json.loads((data / "manifest.json").read_text())
    render_manifest = json.loads((renders / "manifest.json").read_text())
    split: dict[str, list[tuple[np.ndarray, int]]] = {"train": [], "val": [], "test": []}
    for part in manifest["parts"]:
        path = renders / f"{part['name']}.npz"
        if not path.exists():
            continue
        views = np.load(path)["views"].astype(np.float32) / 255.0
        split[part["split"]].append((views, FAMILY_IDS[part["family"]]))
    out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for key, items in split.items():
        x = torch.from_numpy(np.stack([v for v, _ in items])).float()
        y = torch.tensor([label for _, label in items], dtype=torch.long)
        out[key] = (x, y)
    # Normalize per channel using train statistics only.
    mean = out["train"][0].mean(dim=(0, 2, 3), keepdim=True)
    std = out["train"][0].std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-4)
    for key in out:
        out[key] = ((out[key][0] - mean) / std, out[key][1])
    meta = {
        "mean": mean.flatten().tolist(),
        "std": std.flatten().tolist(),
        "n_train": len(out["train"][1]),
        "n_val": len(out["val"][1]),
        "n_test": len(out["test"][1]),
        "render_manifest": render_manifest,
        "source_manifest": {
            "n_parts": manifest.get("n_parts"),
            "split_counts": render_manifest.get("split_counts"),
        },
    }
    return out, meta


def load_graph_splits(data_dir: str | Path) -> dict[str, list[Data]]:
    data = Path(data_dir)
    manifest = json.loads((data / "manifest.json").read_text())
    out: dict[str, list[Data]] = {"train": [], "val": [], "test": []}
    for part in manifest["parts"]:
        path = data / "graphs" / f"{part['name']}.json"
        if not path.exists():
            continue
        graph = PartGraph.model_validate_json(path.read_text())
        d = to_pyg_data(graph)
        d.y = torch.tensor([FAMILY_IDS[part["family"]]], dtype=torch.long)
        out[part["split"]].append(d)
    return out


def _class_weights(y: torch.Tensor, n: int) -> torch.Tensor:
    counts = torch.bincount(y, minlength=n).float().clamp_min(1.0)
    w = counts.rsqrt()
    return w / w.mean()


@torch.no_grad()
def evaluate_cnn(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, device: torch.device, batch_size: int = 128
) -> dict[str, Any]:
    model.eval()
    preds: list[int] = []
    for i in range(0, len(x), batch_size):
        logits = model(x[i : i + batch_size].to(device))
        preds.extend(logits.argmax(dim=-1).cpu().tolist())
    metrics = per_class_metrics(y.tolist(), preds, list(FAMILY_NAMES))
    return metrics


@torch.no_grad()
def evaluate_gnn(
    model: nn.Module, graphs: list[Data], device: torch.device, batch_size: int = 64
) -> dict[str, Any]:
    model.eval()
    loader = GraphLoader(graphs, batch_size=batch_size, shuffle=False)
    preds: list[int] = []
    true: list[int] = []
    for batch in loader:
        batch = batch.to(device)
        preds.extend(model(batch).argmax(dim=-1).cpu().tolist())
        true.extend(batch.y.view(-1).cpu().tolist())
    return per_class_metrics(true, preds, list(FAMILY_NAMES))


def train_cnn(
    splits: dict[str, tuple[torch.Tensor, torch.Tensor]],
    *,
    seed: int = 42,
    epochs: int = 40,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = PartFamilyCNN().to(dev)
    x, y = splits["train"]
    ds = TensorDataset(x, y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    weights = _class_weights(y, len(FAMILY_NAMES)).to(dev)
    crit = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val = -1.0
    best_epoch = 0
    history = []
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for bx, by in loader:
            bx, by = bx.to(dev), by.to(dev)
            if torch.rand(1).item() < 0.5:
                bx = torch.flip(bx, dims=[3])
            opt.zero_grad(set_to_none=True)
            logits = model(bx)
            loss = crit(logits, by)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        sched.step()
        val = evaluate_cnn(model, *splits["val"], dev)
        rec = {
            "epoch": epoch,
            "loss": round(float(np.mean(losses)), 5),
            "val_acc": val["accuracy"],
            "val_macro_f1": val["macro_f1"],
        }
        history.append(rec)
        if verbose:
            print(
                f"[cnn] epoch {epoch:03d} loss={rec['loss']:.4f} val_acc={rec['val_acc']:.4f} f1={rec['val_macro_f1']:.4f}",
                flush=True,
            )
        if val["macro_f1"] > best_val:
            best_val = val["macro_f1"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= 8:
            break
    model.load_state_dict(best_state)
    test = evaluate_cnn(model, *splits["test"], dev)
    return {
        "model": model,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val,
        "train_seconds": round(time.perf_counter() - t0, 3),
        "history": history,
        "test": test,
        "n_parameters": model.n_parameters,
    }


def train_gnn(
    splits: dict[str, list[Data]],
    *,
    seed: int = 42,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 3e-3,
    device: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = FamilyGNN().to(dev)
    train_loader = GraphLoader(splits["train"], batch_size=batch_size, shuffle=True)
    y_train = torch.tensor([int(d.y.item()) for d in splits["train"]], dtype=torch.long)
    weights = _class_weights(y_train, len(FAMILY_NAMES)).to(dev)
    crit = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val = -1.0
    best_epoch = 0
    history = []
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            batch = batch.to(dev)
            opt.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = crit(logits, batch.y.view(-1))
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        sched.step()
        val = evaluate_gnn(model, splits["val"], dev)
        rec = {
            "epoch": epoch,
            "loss": round(float(np.mean(losses)), 5),
            "val_acc": val["accuracy"],
            "val_macro_f1": val["macro_f1"],
        }
        history.append(rec)
        if verbose:
            print(
                f"[gnn] epoch {epoch:03d} loss={rec['loss']:.4f} val_acc={rec['val_acc']:.4f} f1={rec['val_macro_f1']:.4f}",
                flush=True,
            )
        if val["macro_f1"] > best_val:
            best_val = val["macro_f1"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= 8:
            break
    model.load_state_dict(best_state)
    test = evaluate_gnn(model, splits["test"], dev)
    return {
        "model": model,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val,
        "train_seconds": round(time.perf_counter() - t0, 3),
        "history": history,
        "test": test,
        "n_parameters": model.n_parameters,
    }


def save_cnn(path: str | Path, model: PartFamilyCNN, *, metadata: dict[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "class_names": list(FAMILY_NAMES),
            "view_names": list(VIEW_NAMES),
            "metadata": metadata,
        },
        out,
    )
    return out


def load_cnn(
    path: str | Path, *, device: str | torch.device = "cpu"
) -> tuple[PartFamilyCNN, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    model = PartFamilyCNN()
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, payload


def save_gnn(path: str | Path, model: FamilyGNN, *, metadata: dict[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "class_names": list(FAMILY_NAMES),
            "metadata": metadata,
        },
        out,
    )
    return out


def load_gnn(
    path: str | Path, *, device: str | torch.device = "cpu"
) -> tuple[FamilyGNN, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    model = FamilyGNN()
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, payload


__all__ = [
    "FAMILY_IDS",
    "FAMILY_NAMES",
    "FamilyGNN",
    "PartFamilyCNN",
    "evaluate_cnn",
    "evaluate_gnn",
    "load_cnn",
    "load_gnn",
    "load_graph_splits",
    "load_image_splits",
    "save_cnn",
    "save_gnn",
    "train_cnn",
    "train_gnn",
]
