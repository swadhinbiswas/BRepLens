from __future__ import annotations

import random
from pathlib import Path

import pytest

from breplens.adapters.occ import OccAdapter
from breplens.geometry.synthetic import (
    CLASS_TO_ID,
    FAMILIES,
    SYNTHETIC_CLASSES,
    build_part,
    make_spec,
)


@pytest.mark.slow
def test_every_family_builds_with_labels(tmp_path: Path) -> None:
    adapter = OccAdapter()
    rng = random.Random(7)
    built = 0
    for i, family in enumerate(FAMILIES):
        spec = make_spec(family, rng, i)
        spec.name = f"test_{family}"
        truth = build_part(spec, tmp_path, adapter=adapter)
        assert truth["features"], family
        graph_path = tmp_path / "graphs" / f"test_{family}.json"
        assert graph_path.exists()
        from breplens.schema import PartGraph

        graph = PartGraph.model_validate_json(graph_path.read_text())
        assert graph.metadata.family == family
        assert graph.metadata.n_features == len(truth["features"])
        for f in graph.faces:
            assert f.label in SYNTHETIC_CLASSES
            assert f.label_id == CLASS_TO_ID[f.label]
        for feat in truth["features"]:
            assert feat["face_ids"], f"{family}:{feat['gid']}"
        built += 1
    assert built == len(FAMILIES)


def test_committed_samples(samples_dir: Path) -> None:
    steps = sorted(samples_dir.glob("*.step"))
    assert 5 <= len(steps) <= 10
    assert steps[0].stat().st_size > 1000


def test_ground_truth_parameters_positive() -> None:
    rng = random.Random(11)
    for family in FAMILIES:
        spec = make_spec(family, rng, 0)
        for feat in spec.features:
            assert feat.gid
            assert feat.params
            assert feat.position is not None
            assert all(
                (v >= 0 if isinstance(v, (int, float)) and not isinstance(v, bool) else True)
                for v in feat.params.values()
            )


def test_spec_is_deterministic() -> None:
    a = make_spec("bracket_mixed", random.Random(3), 0)
    b = make_spec("bracket_mixed", random.Random(3), 0)
    assert a.params == b.params
    assert [f.gid for f in a.features] == [f.gid for f in b.features]
