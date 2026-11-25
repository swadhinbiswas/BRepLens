"""CPU/GPU inference helpers for the face-classification GNN."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from breplens.geometry.aag import to_pyg_data
from breplens.ml.model import FaceGNN, load_checkpoint
from breplens.schema import PartGraph


def predict_probs(
    model: FaceGNN, graph: PartGraph, *, device: str | torch.device = "cpu"
) -> np.ndarray:
    """Return an ``(n_faces, n_classes)`` probability matrix."""
    data = to_pyg_data(graph).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(data)
        probs = torch.softmax(logits, dim=-1)
    return probs.detach().cpu().numpy()


def predict_labels(
    model: FaceGNN,
    graph: PartGraph,
    class_names: Sequence[str],
    *,
    device: str | torch.device = "cpu",
) -> list[str]:
    probs = predict_probs(model, graph, device=device)
    ids = probs.argmax(axis=1)
    return [class_names[int(i)] for i in ids]


def load_for_inference(
    checkpoint: str | Path, *, device: str | torch.device = "cpu"
) -> tuple[FaceGNN, list[str], dict[str, Any]]:
    model, payload = load_checkpoint(checkpoint, device=device)
    return model, list(payload["class_names"]), payload.get("metadata", {})


__all__ = ["load_for_inference", "predict_labels", "predict_probs"]
