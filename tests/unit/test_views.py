from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from breplens.ml.views import VIEW_NAMES, render_scene_png, render_views
from breplens.schema import MeshData


@pytest.fixture()
def mesh(occ_adapter, samples_dir: Path) -> MeshData:
    return occ_adapter.tessellate(samples_dir / "sample_04_pocket.step", deflection=0.4)


def test_render_shape_and_content(mesh: MeshData, tmp_path: Path) -> None:
    views = render_views(mesh, size=64)
    assert views.shape == (len(VIEW_NAMES), 64, 64)
    assert views.dtype == np.uint8
    assert views.max() > 0
    assert all(v.max() > 0 for v in views)


def test_render_is_deterministic(mesh: MeshData) -> None:
    a = render_views(mesh, size=48)
    b = render_views(mesh, size=48)
    assert np.array_equal(a, b)


def test_render_empty_mesh() -> None:
    empty = MeshData(vertices=[], triangles=[], triangle_face_ids=[])
    views = render_views(empty, size=32)
    assert views.shape == (len(VIEW_NAMES), 32, 32)
    assert views.max() == 0


def test_render_png(mesh: MeshData, tmp_path: Path) -> None:
    path = tmp_path / "views.png"
    render_scene_png(mesh, str(path), size=64)
    assert path.exists() and path.stat().st_size > 100
