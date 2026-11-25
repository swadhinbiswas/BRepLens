from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from breplens.ml.inference import load_for_inference, predict_probs
from breplens.ml.model import FaceGNN, GNNConfig, save_checkpoint
from breplens.recognition.gnn_recognizer import recognize_gnn, recognize_hybrid
from breplens.schema import FeatureType

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def checkpoint(root: Path) -> Path:
    p = root / "models" / "gnn_synthetic.pt"
    if not p.exists():
        pytest.skip("synthetic checkpoint not committed")
    return p


def test_checkpoint_loads(checkpoint: Path) -> None:
    model, classes, meta = load_for_inference(checkpoint, device="cpu")
    assert isinstance(model, FaceGNN)
    assert "through_hole" in classes
    assert model.n_parameters < 500_000
    assert checkpoint.stat().st_size < 2 * 1024 * 1024
    assert meta.get("architecture") in {"edge_gnn", "gcn"}


def test_predict_probs(occ_adapter, samples_dir: Path, checkpoint: Path) -> None:
    model, classes, _ = load_for_inference(checkpoint, device="cpu")
    g = occ_adapter.load_graph(samples_dir / "sample_04_pocket.step")
    probs = predict_probs(model, g, device="cpu")
    assert probs.shape == (len(g.faces), len(classes))
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)
    assert (probs >= 0).all()


def test_save_load_roundtrip(checkpoint: Path, tmp_path: Path) -> None:
    model, classes, _ = load_for_inference(checkpoint, device="cpu")
    out = tmp_path / "roundtrip.pt"
    save_checkpoint(out, model, class_names=classes, metadata={"x": 1})
    model2, classes2, meta2 = load_for_inference(out, device="cpu")
    assert classes2 == classes
    assert meta2["x"] == 1
    for a, b in zip(model.state_dict().values(), model2.state_dict().values(), strict=True):
        assert torch.equal(a, b)


def test_gnn_features(occ_adapter, samples_dir: Path, checkpoint: Path) -> None:
    model, classes, _ = load_for_inference(checkpoint, device="cpu")
    g = occ_adapter.load_graph(samples_dir / "sample_09_bracket_mixed.step")
    fs = recognize_gnn(g, model, classes, device="cpu")
    types = {f.type for f in fs.features}
    assert FeatureType.THROUGH_HOLE in types
    assert FeatureType.POCKET in types
    assert all(f.source == "gnn" for f in fs.features)
    assert len(fs.features) >= 4


def test_hybrid_superset(occ_adapter, samples_dir: Path, checkpoint: Path) -> None:
    from breplens.recognition.rules import recognize_rules

    model, classes, _ = load_for_inference(checkpoint, device="cpu")
    g = occ_adapter.load_graph(samples_dir / "sample_04_pocket.step")
    base = recognize_rules(g)
    hybrid = recognize_hybrid(g, model, classes, device="cpu")
    assert hybrid.mode == "hybrid"
    assert hybrid.features
    assert len(hybrid.features) >= len(base.features)
    assert {f.type for f in base.features} <= {f.type for f in hybrid.features}
    # no duplicate face claims in a hybrid result
    seen: set[int] = set()
    for f in hybrid.features:
        assert not (set(f.face_ids) & seen)
        seen |= set(f.face_ids)


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        FaceGNN(3, GNNConfig(architecture="unknown"))


def test_unmeasurable_recess_returns_none() -> None:
    import numpy as np

    from breplens.recognition.gnn_recognizer import _measure_recess
    from breplens.schema import FaceRecord, PartGraph, PartMetadata, SurfaceType, Vec3

    graph = PartGraph(
        metadata=PartMetadata(name="flat", bbox_min=Vec3(), bbox_max=Vec3(x=10, y=10, z=0)),
        faces=[
            FaceRecord(
                id=0,
                surface_type=SurfaceType.PLANE,
                area=100.0,
                centroid=Vec3(x=5, y=5, z=0),
                bbox_min=Vec3(x=0, y=0, z=0),
                bbox_max=Vec3(x=10, y=10, z=0),
                normal=Vec3(x=0, y=0, z=1),
            )
        ],
    )
    probs = np.ones((1, 1), dtype=np.float32)
    assert (
        _measure_recess(
            graph,
            [0],
            FeatureType.POCKET,
            probs,
            0,
            __import__("breplens.recognition.params", fromlist=["RuleConfig"]).RuleConfig(),
        )
        is None
    )
