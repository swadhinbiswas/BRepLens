from __future__ import annotations

from pathlib import Path

import pytest

from breplens.adapters.base import CadLoadError
from breplens.schema import EdgeConvexity, SurfaceType


def test_load_box_plate(occ_adapter, samples_dir: Path) -> None:
    g = occ_adapter.load_graph(samples_dir / "sample_00_plate_holes.step")
    assert g.metadata.units == "mm"
    assert len(g.faces) == 10
    assert len(g.edges) > 0
    assert sum(f.surface_type == SurfaceType.CYLINDER for f in g.faces) == 4
    assert g.metadata.volume and g.metadata.volume > 0
    assert g.metadata.bbox_max.z > g.metadata.bbox_min.z
    # material-side convention: through-hole rims are convex
    assert all(e.convexity == EdgeConvexity.CONVEX for e in g.edges)


def test_hole_material_side(occ_adapter, samples_dir: Path) -> None:
    g = occ_adapter.load_graph(samples_dir / "sample_07_boss_plate.step")
    cyl = [f for f in g.faces if f.surface_type == SurfaceType.CYLINDER]
    assert len(cyl) == 1
    assert cyl[0].material_side == "inside"
    g2 = occ_adapter.load_graph(samples_dir / "sample_00_plate_holes.step")
    assert all(
        f.material_side == "outside" for f in g2.faces if f.surface_type == SurfaceType.CYLINDER
    )


def test_pocket_has_concave_edges(occ_adapter, samples_dir: Path) -> None:
    g = occ_adapter.load_graph(samples_dir / "sample_04_pocket.step")
    concave = [e for e in g.edges if e.convexity == EdgeConvexity.CONCAVE]
    assert len(concave) >= 4
    assert all(e.dihedral_angle_deg is not None for e in concave)


def test_fillet_has_smooth_edges(occ_adapter, samples_dir: Path) -> None:
    g = occ_adapter.load_graph(samples_dir / "sample_08_fillet_block.step")
    assert sum(e.convexity == EdgeConvexity.SMOOTH for e in g.edges) >= 8


def test_blind_hole_convexity(occ_adapter, samples_dir: Path) -> None:
    g = occ_adapter.load_graph(samples_dir / "sample_01_plate_blind.step")
    concave = [e for e in g.edges if e.convexity == EdgeConvexity.CONCAVE]
    assert len(concave) == 1
    cyl = next(f for f in g.faces if f.surface_type == SurfaceType.CYLINDER)
    assert cyl.material_side == "outside"


def test_smooth_angle_parameter(occ_adapter) -> None:
    from breplens.adapters.occ import OccAdapter

    assert OccAdapter(smooth_angle_deg=10.0).smooth_angle_deg == 10.0
    with pytest.raises(AssertionError):
        OccAdapter(smooth_angle_deg=90.0)


def test_tessellation_consistency(occ_adapter, samples_dir: Path) -> None:
    mesh = occ_adapter.tessellate(samples_dir / "sample_04_pocket.step", deflection=0.3)
    assert len(mesh.vertices) > 0
    assert len(mesh.triangles) > 0
    mesh.validate_lengths()
    assert all(0 <= fid < 11 for fid in mesh.triangle_face_ids)


def test_errors(occ_adapter, tmp_path: Path, samples_dir: Path) -> None:
    empty = tmp_path / "empty.step"
    empty.write_bytes(b"")
    with pytest.raises(CadLoadError):
        occ_adapter.load_graph(empty)
    junk = tmp_path / "junk.step"
    junk.write_bytes(b"hello world")
    with pytest.raises(CadLoadError):
        occ_adapter.load_graph(junk)
    wrong = tmp_path / "part.iges"
    wrong.write_bytes((samples_dir / "sample_04_pocket.step").read_bytes())
    with pytest.raises(CadLoadError):
        occ_adapter.load_graph(wrong)
    missing = tmp_path / "missing.step"
    with pytest.raises(CadLoadError):
        occ_adapter.load_graph(missing)


def test_deflection_bounds(occ_adapter, samples_dir: Path) -> None:
    with pytest.raises(CadLoadError):
        occ_adapter.tessellate(samples_dir / "sample_04_pocket.step", deflection=0.0)


def test_step_face_ids_when_mfcad_available(occ_adapter, root: Path) -> None:
    step = root / "data" / "raw" / "MFCAD" / "dataset" / "step" / "0-0-0-0-0-23.step"
    if not step.exists():
        pytest.skip("MFCAD raw data not downloaded")
    g = occ_adapter.load_graph(step, read_step_face_ids=True)
    assert len(g.faces) == 11
    assert all(f.step_face_id is not None for f in g.faces)
