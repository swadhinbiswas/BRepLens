from __future__ import annotations

import math
from pathlib import Path

import pytest

from breplens.recognition.rules import recognize_rules
from breplens.schema import FeatureType


@pytest.fixture(scope="module")
def results(occ_adapter, samples_dir: Path) -> dict[str, object]:
    out = {}
    for p in sorted(samples_dir.glob("*.step")):
        out[p.name] = recognize_rules(occ_adapter.load_graph(p))
    return out


def _types(fs) -> list[str]:
    return [f.type.value for f in fs.features]


def test_plate_holes(results) -> None:
    fs = results["sample_00_plate_holes.step"]
    assert _types(fs) == ["through_hole"] * 4
    for f in fs.features:
        assert f.params.diameter and 5 < f.params.diameter < 8
        assert f.params.depth and f.params.depth > 5
        assert f.params.is_through is True
        assert f.position is not None


def test_blind_hole(results) -> None:
    fs = results["sample_01_plate_blind.step"]
    assert _types(fs) == ["blind_hole"]
    f = fs.features[0]
    assert f.params.is_through is False
    assert f.params.depth and f.params.depth > 2
    assert f.params.diameter and f.params.diameter > 5


def test_counterbore(results) -> None:
    fs = results["sample_02_counterbore.step"]
    assert _types(fs) == ["counterbore"]
    f = fs.features[0]
    assert f.params.outer_diameter and f.params.inner_diameter
    assert f.params.outer_diameter > f.params.inner_diameter
    assert f.params.depth and f.params.depth > 1


def test_countersink(results) -> None:
    fs = results["sample_03_countersink.step"]
    assert _types(fs) == ["countersink"]
    f = fs.features[0]
    assert f.params.angle_deg and abs(f.params.angle_deg - 90) < 2
    assert f.params.outer_diameter and f.params.inner_diameter
    assert f.params.outer_diameter > f.params.inner_diameter


def test_pocket_and_slot(results) -> None:
    pocket = results["sample_04_pocket.step"]
    assert _types(pocket) == ["pocket"]
    p = pocket.features[0]
    assert p.params.length and p.params.width and p.params.depth
    assert 0 < p.params.depth < 20
    slot = results["sample_05_slot.step"]
    assert _types(slot) == ["slot"]
    assert slot.features[0].params.length and slot.features[0].params.width


def test_step_and_boss(results) -> None:
    step = results["sample_06_stepped_block.step"]
    assert _types(step) == ["step"]
    assert step.features[0].params.height and step.features[0].params.height > 1
    boss = results["sample_07_boss_plate.step"]
    assert _types(boss) == ["boss"]
    assert boss.features[0].params.diameter and boss.features[0].params.height


def test_fillets(results) -> None:
    fs = results["sample_08_fillet_block.step"]
    assert _types(fs) == ["fillet"] * 4
    for f in fs.features:
        assert f.params.radius and f.params.radius > 1


def test_intersecting_bracket(results) -> None:
    fs = results["sample_09_bracket_mixed.step"]
    types = _types(fs)
    assert types.count("through_hole") == 3
    assert types.count("chamfer") == 4
    assert "pocket" in types
    pocket = next(f for f in fs.features if f.type == FeatureType.POCKET)
    assert pocket.params.depth and pocket.params.length and pocket.params.width
    # the hole inside the pocket is measured from the pocket floor, not the top
    inside = [
        f
        for f in fs.features
        if f.type == FeatureType.THROUGH_HOLE and f.params.diameter and f.params.diameter > 7
    ]
    assert len(inside) == 1
    assert inside[0].params.is_through is True


def test_no_overlapping_faces(results) -> None:
    for fs in results.values():
        seen: set[int] = set()
        for f in fs.features:
            assert not (set(f.face_ids) & seen)
            seen |= set(f.face_ids)


def test_parameters_finite_and_confidence(results) -> None:
    for fs in results.values():
        for f in fs.features:
            assert 0 < f.confidence <= 1
            for name, value in f.params.model_dump(exclude_none=True).items():
                if name == "extra":
                    continue
                if isinstance(value, float):
                    assert math.isfinite(value)
                    assert value >= 0


def test_negative_plain_box(occ_adapter, tmp_path: Path) -> None:
    cq = pytest.importorskip("cadquery")
    part = cq.Workplane("XY").box(40, 30, 10)
    step = tmp_path / "box.step"
    cq.exporters.export(part, str(step))
    fs = recognize_rules(occ_adapter.load_graph(step))
    assert fs.features == []


def test_determinism(occ_adapter, samples_dir: Path) -> None:
    g = occ_adapter.load_graph(samples_dir / "sample_09_bracket_mixed.step")
    a = recognize_rules(g)
    b = recognize_rules(g)
    assert [f.id for f in a.features] == [f.id for f in b.features]
    assert a.model_dump(exclude={"timings", "created_utc"}) == b.model_dump(
        exclude={"timings", "created_utc"}
    )


def test_latency_is_sane(results) -> None:
    for fs in results.values():
        assert 0.0 <= fs.timings.recognition_ms < 5000.0
