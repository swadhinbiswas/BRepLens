"""Face-classification GNNs for the neutral AAG (PyTorch + PyTorch Geometric).

Two architectures are provided:

* ``gcn`` -- a GCN baseline that ignores edge attributes (sanity baseline).
* ``edge_gnn`` -- an edge-aware message-passing network: messages are an MLP of
  ``[h_src, h_dst, edge_attr]``, updates use a GRU cell, with residual +
  LayerNorm. This is close in spirit to AAGNet's geometry-aware message passing
  while remaining small enough for CPU inference.

Models classify every face into a feature class; feature instances are formed by
clustering predicted faces and measuring parameters geometrically.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, MessagePassing

from breplens.geometry.aag import EDGE_FEATURE_NAMES, FEATURE_NAMES


@dataclass(frozen=True)
class GNNConfig:
    architecture: str = "edge_gnn"
    hidden: int = 64
    layers: int = 3
    dropout: float = 0.15

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EdgeAwareLayer(MessagePassing):
    """Edge-conditioned message passing with a GRU node update."""

    def __init__(self, hidden: int, edge_dim: int) -> None:
        super().__init__(aggr="sum")
        self.msg = nn.Sequential(
            nn.Linear(2 * hidden + edge_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.gru = nn.GRUCell(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(0.0)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        m = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        h = self.gru(m, x)
        return cast(torch.Tensor, self.norm(x + self.drop(h)))

    def message(
        self, x_i: torch.Tensor, x_j: torch.Tensor, edge_attr: torch.Tensor
    ) -> torch.Tensor:
        return cast(torch.Tensor, self.msg(torch.cat([x_i, x_j, edge_attr], dim=-1)))


class FaceGNN(nn.Module):
    """Small face classifier over an attributed adjacency graph."""

    def __init__(self, num_classes: int, config: GNNConfig | None = None) -> None:
        super().__init__()
        self.config = config or GNNConfig()
        cfg = self.config
        edge_dim = len(EDGE_FEATURE_NAMES)
        self.input = nn.Linear(len(FEATURE_NAMES), cfg.hidden)
        if cfg.architecture == "gcn":
            self.layers = nn.ModuleList(
                [GCNConv(cfg.hidden, cfg.hidden) for i in range(cfg.layers)]
            )
        elif cfg.architecture == "edge_gnn":
            self.layers = nn.ModuleList(
                [EdgeAwareLayer(cfg.hidden, edge_dim) for _ in range(cfg.layers)]
            )
        else:  # pragma: no cover - config validated by CLI
            raise ValueError(f"unknown architecture {cfg.architecture}")
        self.drop = nn.Dropout(cfg.dropout)
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, num_classes),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x = self.input(data.x)
        for i, layer in enumerate(self.layers):
            if self.config.architecture == "gcn":
                h = layer(x, data.edge_index)
            else:
                h = layer(x, data.edge_index, data.edge_attr)
            x = x + self.drop(torch.relu(h))
            if i == 0:
                x = torch.relu(x)
        return cast(torch.Tensor, self.head(x))

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def save_checkpoint(
    path: str | Path,
    model: FaceGNN,
    *,
    class_names: list[str],
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save a small, self-describing CPU-loadable checkpoint."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "config": model.config.to_dict(),
        "class_names": class_names,
        "feature_names": list(FEATURE_NAMES),
        "edge_feature_names": list(EDGE_FEATURE_NAMES),
        "metadata": metadata or {},
    }
    torch.save(payload, out)
    return out


def load_checkpoint(
    path: str | Path, *, device: str | torch.device = "cpu"
) -> tuple[FaceGNN, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    cfg = GNNConfig(**payload["config"])
    model = FaceGNN(len(payload["class_names"]), cfg)
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, payload


def load_metadata(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    return {
        "config": payload["config"],
        "class_names": payload["class_names"],
        "metadata": payload.get("metadata", {}),
        "n_parameters": sum(
            v.numel() for v in payload["state_dict"].values() if v.dtype.is_floating_point
        ),
    }


def checkpoint_json(path: str | Path) -> str:
    return json.dumps(load_metadata(path), indent=1)


__all__ = [
    "EdgeAwareLayer",
    "FaceGNN",
    "GNNConfig",
    "checkpoint_json",
    "load_checkpoint",
    "load_metadata",
    "save_checkpoint",
]
