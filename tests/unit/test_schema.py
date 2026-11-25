from __future__ import annotations

import json
import math

import pytest
from pydantic import ValidationError

from breplens.schema import (
    SCHEMA_VERSION,
    EdgeConvexity,
    EdgeRecord,
    FaceRecord,
    Feature,
    FeatureParams,
    FeatureSet,
    FeatureType,
    MeshData,
    PartGraph,
    PartMetadata,
    SurfaceType,
    Vec3,
)


def test_vec3_math() -> None:
    a = Vec3(x=1, y=2, z=3)
    b = Vec3(x=-2, y=1, z=0)
    assert a + b == Vec3(x=-1, y=3, z=3)
    assert a - b == Vec3(x=3, y=1, z=3)
    assert (a * 2).x == 2
    assert a.dot(b) == 0
    assert a.cross(b) == Vec3(x=-3, y=-6, z=5)
    assert pytest.approx(1.0) == a.normalized().norm()
    assert Vec3().normalized() == Vec3()
    assert Vec3.of((1, 2, 3)) == Vec3(x=1, y=2, z=3)


def _face(fid: int = 0, **kwargs) -> FaceRecord:
    base = dict(
        id=fid,
        surface_type=SurfaceType.PLANE,
        area=10.0,
        centroid=Vec3(),
        bbox_min=Vec3(x=-1, y=-1, z=-1),
        bbox_max=Vec3(x=1, y=1, z=1),
        normal=Vec3(x=0, y=0, z=1),
    )
    base.update(kwargs)
    return FaceRecord(**base)


def test_face_normal_is_normalized() -> None:
    f = _face(normal=Vec3(x=0, y=0, z=5))
    assert f.normal is not None
    assert f.normal.norm() == pytest.approx(1.0)


def test_bbox_order_rejected() -> None:
    with pytest.raises(ValidationError):
        _face(bbox_min=Vec3(x=2, y=0, z=0), bbox_max=Vec3(x=1, y=1, z=1))


def test_material_side_vocab() -> None:
    assert _face(material_side="outside").material_side == "outside"
    with pytest.raises(ValidationError):
        _face(material_side="sideways")


def test_edge_rejects_self_loop() -> None:
    with pytest.raises(ValidationError):
        EdgeRecord(
            id=0,
            face_a=1,
            face_b=1,
            convexity=EdgeConvexity.CONCAVE,
            length=1.0,
            curve_type="line",
            midpoint=Vec3(),
        )


def test_graph_face_lookup_and_adjacency() -> None:
    g = PartGraph(
        metadata=PartMetadata(name="p", n_faces=2, n_edges=1),
        faces=[_face(0), _face(1)],
        edges=[
            EdgeRecord(
                id=0,
                face_a=0,
                face_b=1,
                convexity=EdgeConvexity.CONCAVE,
                length=2.0,
                curve_type="line",
                midpoint=Vec3(),
            )
        ],
    )
    assert g.face(1).id == 1
    with pytest.raises(KeyError):
        g.face(5)
    assert g.neighbors(0) == {1}
    assert g.edge_between(1, 0) is not None
    assert g.adjacency()[0][0][0] == 1


def test_feature_requires_faces() -> None:
    with pytest.raises(ValidationError):
        Feature(id="x", type=FeatureType.POCKET, confidence=0.5, face_ids=[])


def test_feature_and_set_roundtrip() -> None:
    feat = Feature(
        id="pocket_0",
        type=FeatureType.POCKET,
        confidence=0.9,
        face_ids=[1, 2],
        params=FeatureParams(length=10, width=5, depth=2),
        position=Vec3(),
        axis=Vec3(x=0, y=0, z=1),
    )
    fs = FeatureSet(part="p", features=[feat])
    assert fs.schema_version == SCHEMA_VERSION
    again = FeatureSet.model_validate_json(fs.model_dump_json())
    assert again.features[0].params.depth == 2
    schema = json.loads(json.dumps(FeatureSet.model_json_schema()))
    assert "FeatureSet" in schema["title"] or schema["type"] == "object"


def test_confidence_bounds() -> None:
    with pytest.raises(ValidationError):
        Feature(id="x", type=FeatureType.POCKET, confidence=1.5, face_ids=[0])


def test_mesh_lengths() -> None:
    mesh = MeshData(
        vertices=[(0, 0, 0), (1, 0, 0), (0, 1, 0)], triangles=[(0, 1, 2)], triangle_face_ids=[0]
    )
    mesh.validate_lengths()
    bad = MeshData(vertices=[(0, 0, 0)], triangles=[(0, 1, 2)], triangle_face_ids=[0])
    with pytest.raises(ValueError):
        bad.validate_lengths()
    mismatched = MeshData(vertices=[(0, 0, 0)], triangles=[], triangle_face_ids=[0])
    with pytest.raises(ValueError):
        mismatched.validate_lengths()


def test_math_angles_not_nan() -> None:
    assert math.isfinite(
        math.degrees(
            math.acos(
                max(-1.0, min(1.0, Vec3(x=1, y=0, z=0).normalized().dot(Vec3(x=1, y=0, z=0))))
            )
        )
    )
