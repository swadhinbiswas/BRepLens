from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from breplens.similarity.signature import (
    LibraryEntry,
    SignatureIndex,
    part_signature,
    precision_at_k,
)


def test_signature_is_normalized(occ_adapter, samples_dir: Path) -> None:
    g = occ_adapter.load_graph(samples_dir / "sample_04_pocket.step")
    v = part_signature(g)
    assert v.ndim == 1
    assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-5)
    v2 = part_signature(g)
    assert np.allclose(v, v2)


def test_index_self_exclusion(occ_adapter, samples_dir: Path) -> None:
    idx = SignatureIndex()
    entries = []
    for name in (
        "sample_04_pocket.step",
        "sample_07_boss_plate.step",
        "sample_00_plate_holes.step",
    ):
        g = occ_adapter.load_graph(samples_dir / name)
        entries.append(idx.add_graph(g, name=name, family=name))
    hits = idx.search(entries[0].vector, k=3, exclude_id="sample_04_pocket.step")
    assert len(hits) == 2
    assert all(h.id != "sample_04_pocket.step" for h in hits)
    assert all(-1.001 <= h.score <= 1.001 for h in hits)


def test_save_load(tmp_path: Path) -> None:
    idx = SignatureIndex()
    idx.add(
        LibraryEntry(id="a", name="a", family="fam", vector=np.array([1.0, 0.0], dtype=np.float32))
    )
    idx.add(
        LibraryEntry(
            id="b", name="b", family="other", vector=np.array([0.0, 1.0], dtype=np.float32)
        )
    )
    path = idx.save(tmp_path / "lib.json")
    idx2 = SignatureIndex.load(path)
    assert len(idx2.entries) == 2
    hits = idx2.search(np.array([1.0, 0.0]), k=1)
    assert hits[0].id == "a"


def test_precision_at_k_toy() -> None:
    idx = SignatureIndex()
    idx.add(LibraryEntry(id="a1", name="a1", family="A", vector=np.array([1.0, 0.0])))
    idx.add(LibraryEntry(id="a2", name="a2", family="A", vector=np.array([0.9, 0.1])))
    idx.add(LibraryEntry(id="b1", name="b1", family="B", vector=np.array([0.0, 1.0])))
    m = precision_at_k([np.array([1.0, 0.0])], ["A"], ["q"], idx, 2)
    assert m["precision@2"] == 1.0
    assert m["n_queries"] == 1


def test_demo_library(root: Path) -> None:
    path = root / "demo" / "library.json"
    if not path.exists():
        pytest.skip("demo library not built")
    idx = SignatureIndex.load(path)
    assert len(idx.entries) >= 10
    assert all(e.vector.shape[0] == idx.dim for e in idx.entries)
