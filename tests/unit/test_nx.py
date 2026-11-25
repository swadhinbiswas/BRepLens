from __future__ import annotations

from pathlib import Path

import pytest

from breplens.adapters.base import CadLoadError
from breplens.schema import Feature, FeatureParams, FeatureSet, FeatureType, Vec3


def test_body_summary(fake_nx, root: Path) -> None:
    from breplens.adapters.nx import NxOpenAdapter

    adapter = NxOpenAdapter()
    assert adapter.available()
    summary = adapter.body_summary()
    assert summary["bodies"] == 1
    assert summary["faces"] == 3
    assert summary["status"].startswith("UNTESTED")


def test_load_graph_via_export(fake_nx, root: Path, tmp_path: Path) -> None:
    from breplens.adapters.nx import UNTESTED_LIVE, NxOpenAdapter

    adapter = NxOpenAdapter()
    prt = tmp_path / "part.prt"
    prt.write_bytes(b"mock")
    graph = adapter.load_graph(prt)
    assert len(graph.faces) == 11
    assert graph.metadata.source_format == "nx"
    assert UNTESTED_LIVE in graph.warnings


def _feature_set() -> FeatureSet:
    return FeatureSet(
        part="mock",
        mode="rules",
        features=[
            Feature(
                id="h",
                type=FeatureType.THROUGH_HOLE,
                confidence=1,
                face_ids=[0],
                params=FeatureParams(diameter=6, depth=10, is_through=True),
                position=Vec3(x=0, y=0, z=10),
                axis=Vec3(x=0, y=0, z=1),
            ),
            Feature(
                id="p",
                type=FeatureType.POCKET,
                confidence=1,
                face_ids=[1],
                params=FeatureParams(length=20, width=10, depth=3),
                position=Vec3(x=0, y=0, z=10),
                axis=Vec3(x=0, y=0, z=1),
            ),
            Feature(
                id="s",
                type=FeatureType.SLOT,
                confidence=1,
                face_ids=[2],
                params=FeatureParams(length=20, width=5, depth=3),
                position=Vec3(x=0, y=0, z=10),
                axis=Vec3(x=0, y=0, z=1),
            ),
            Feature(
                id="b",
                type=FeatureType.BOSS,
                confidence=1,
                face_ids=[3],
                params=FeatureParams(diameter=8, height=4),
                position=Vec3(x=0, y=0, z=10),
                axis=Vec3(x=0, y=0, z=1),
            ),
            Feature(
                id="f",
                type=FeatureType.FILLET,
                confidence=1,
                face_ids=[4],
                params=FeatureParams(radius=2),
            ),
        ],
    )


def test_write_back_report(fake_nx, root: Path, tmp_path: Path) -> None:
    import json

    from breplens.adapters.nx import NxOpenAdapter

    adapter = NxOpenAdapter()
    prt = tmp_path / "part.prt"
    prt.write_bytes(b"mock")
    report = json.loads(adapter.write_features(prt, _feature_set()))
    assert report["created"] == ["h", "p", "s", "b"]
    assert report["skipped"] == ["f"]
    assert report["status"].startswith("UNTESTED")
    names = [c[0] for c in fake_nx.CALL_LOG]
    assert "HolePackageBuilder.CommitFeature" in names
    assert names.count("BlockFeatureBuilder.CommitFeature") == 2
    assert "CylinderBuilder.CommitFeature" in names


def test_non_z_hole_skipped(fake_nx, root: Path, tmp_path: Path) -> None:
    import json

    from breplens.adapters.nx import NxOpenAdapter

    fs = FeatureSet(
        part="x",
        mode="rules",
        features=[
            Feature(
                id="h",
                type=FeatureType.THROUGH_HOLE,
                confidence=1,
                face_ids=[0],
                params=FeatureParams(diameter=6, depth=10),
                position=Vec3(x=0, y=10, z=10),
                axis=Vec3(x=1, y=0, z=0),
            )
        ],
    )
    prt = tmp_path / "part.prt"
    prt.write_bytes(b"mock")
    report = json.loads(NxOpenAdapter().write_features(prt, fs))
    assert report["created"] == []
    assert report["skipped"] == ["h"]
    assert any("Z-axis" in w for w in report["warnings"])


def test_writeback_requires_nx() -> None:
    from breplens.adapters.nx import NxOpenAdapter

    # In a normal environment NXOpen is absent; in the fake fixture it is present.
    if NxOpenAdapter.available():
        pytest.skip("NXOpen mock/real package present")
    with pytest.raises(CadLoadError):
        NxOpenAdapter()._session()
