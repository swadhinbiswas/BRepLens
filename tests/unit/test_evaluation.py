from __future__ import annotations

import pytest

from breplens.ml.evaluation import (
    face_metrics_from_ids,
    match_features,
    parameter_errors,
    per_class_metrics,
    predicted_face_labels,
)
from breplens.schema import (
    Feature,
    FeatureParams,
    FeatureSet,
    FeatureType,
    PartGraph,
    Vec3,
)


def _fs(*features: Feature) -> FeatureSet:
    return FeatureSet(part="x", features=list(features))


def _feat(i: str, t: FeatureType, pos: tuple[float, float, float], **params) -> Feature:
    return Feature(
        id=i,
        type=t,
        confidence=1.0,
        face_ids=[0],
        params=FeatureParams(**params),
        position=Vec3.of(pos),
        axis=Vec3(x=0, y=0, z=1) if t in (FeatureType.THROUGH_HOLE, FeatureType.POCKET) else None,
    )


def test_perfect_match() -> None:
    f = _feat("h", FeatureType.THROUGH_HOLE, (0, 0, 10), diameter=6, depth=10)
    truth = [
        {
            "type": "through_hole",
            "position": [0, 0, 10],
            "axis": [0, 0, 1],
            "params": {"diameter": 6, "depth": 10},
        }
    ]
    pairs = match_features([f], truth)
    assert len(pairs) == 1


def test_wrong_type_does_not_match() -> None:
    f = _feat("h", FeatureType.POCKET, (0, 0, 10), length=6)
    truth = [
        {
            "type": "through_hole",
            "position": [0, 0, 10],
            "axis": [0, 0, 1],
            "params": {"diameter": 6},
        }
    ]
    assert match_features([f], truth) == []


def test_metrics_known_values() -> None:
    names = ["a", "b"]
    m = per_class_metrics([0, 0, 1, 1], [0, 1, 1, 1], names)
    assert m["accuracy"] == 0.75
    assert m["per_class"]["a"]["precision"] == 1.0
    assert m["per_class"]["a"]["recall"] == 0.5
    assert m["per_class"]["b"]["recall"] == 1.0
    assert m["macro_f1"] < 1.0
    assert len(m["confusion_matrix"]) == 2


def test_face_metrics() -> None:
    m = face_metrics_from_ids([0, 1, 1], [0, 1, 0], ["base", "hole"])
    assert m["accuracy"] == pytest.approx(2 / 3, abs=1e-4)
    assert m["per_class"]["hole"]["fn"] == 1


def test_predicted_labels_and_fallback() -> None:
    from breplens.schema import FaceRecord, PartMetadata, SurfaceType

    faces = [
        FaceRecord(
            id=i,
            surface_type=SurfaceType.PLANE,
            area=1,
            centroid=Vec3(),
            bbox_min=Vec3(),
            bbox_max=Vec3(x=1, y=1, z=1),
        )
        for i in range(3)
    ]
    g = PartGraph(metadata=PartMetadata(), faces=faces)
    fs = _fs(Feature(id="p", type=FeatureType.POCKET, confidence=1, face_ids=[1]))
    class_ids = {"base": 0, "pocket": 1}
    assert predicted_face_labels(g, fs, class_ids) == [0, 1, 0]
    assert predicted_face_labels(g, fs, class_ids, fallback=[2, 2, 2]) == [2, 1, 2]


def test_parameter_errors() -> None:
    pred = _feat("h", FeatureType.THROUGH_HOLE, (0, 0, 0), diameter=6.2, depth=10)
    truth = {
        "type": "through_hole",
        "position": [0, 0, 0],
        "axis": [0, 0, 1],
        "params": {"diameter": 6.0, "depth": 10.0},
    }
    err = parameter_errors([(pred, truth)])
    assert err["through_hole"]["diameter"]["mean_abs"] == pytest.approx(0.2, abs=1e-3)
    assert err["through_hole"]["depth"]["mean_abs"] == 0.0
