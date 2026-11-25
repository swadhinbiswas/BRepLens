from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")
import torch

from breplens.ml.family import (
    FAMILY_NAMES,
    FamilyGNN,
    PartFamilyCNN,
    load_cnn,
    load_graph_splits,
    load_image_splits,
)


def test_cnn_forward() -> None:
    model = PartFamilyCNN()
    x = torch.zeros(2, 6, 96, 96)
    logits = model(x)
    assert logits.shape == (2, len(FAMILY_NAMES))
    assert 20_000 < model.n_parameters < 100_000


def test_gnn_forward() -> None:
    from breplens.geometry.aag import to_pyg_data
    from breplens.schema import PartGraph

    graph = (
        PartGraph.model_validate_json(
            (
                Path(__file__).resolve().parents[2]
                / "data"
                / "synthetic"
                / "graphs"
                / "pocket_00000.json"
            ).read_text()
        )
        if (
            Path(__file__).resolve().parents[2]
            / "data"
            / "synthetic"
            / "graphs"
            / "pocket_00000.json"
        ).exists()
        else None
    )
    if graph is None:
        pytest.skip("synthetic cache not present")
    data = to_pyg_data(graph)
    data.y = torch.tensor([0])
    from torch_geometric.loader import DataLoader

    batch = next(iter(DataLoader([data], batch_size=1)))
    model = FamilyGNN()
    logits = model(batch)
    assert logits.shape == (1, len(FAMILY_NAMES))


def test_family_checkpoints(root: Path) -> None:
    cnn_path = root / "models" / "family_cnn.pt"
    if not cnn_path.exists():
        pytest.skip("family CNN checkpoint not committed")
    _model, payload = load_cnn(cnn_path)
    assert payload["metadata"]["test_accuracy"] >= 0
    assert payload["view_names"] == ["right", "left", "front", "back", "top", "bottom"]


def test_image_dataset_splits(root: Path) -> None:
    if not (root / "data" / "renders" / "manifest.json").exists():
        pytest.skip("render cache not present")
    splits, meta = load_image_splits(root / "data" / "synthetic", root / "data" / "renders")
    assert meta["n_train"] > 0 and meta["n_test"] > 0
    x, y = splits["test"]
    assert x.shape[1] == 6 and x.shape[2] == 96
    assert y.numel() == x.shape[0]


def test_graph_dataset_splits(root: Path) -> None:
    if not (root / "data" / "synthetic" / "graphs").exists():
        pytest.skip("graph cache not present")
    splits = load_graph_splits(root / "data" / "synthetic")
    assert len(splits["train"]) > 0
    assert len(splits["test"]) > 0
