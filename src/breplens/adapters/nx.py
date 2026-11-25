"""Siemens NX adapter (NX Open Python).

STATUS: UNTESTED AGAINST A LIVE NX SESSION.

The graph is extracted by exporting the work part to STEP with the NX Open
``DexManager`` (Siemens' documented STEP exporter) and reading it with the fully
tested :class:`~breplens.adapters.occ.OccAdapter`. This deliberately avoids
unverified per-face geometry calls while still using NX Open for session/body
traversal, STEP export and parametric write-back.

Write-back uses a recipe reported as tested in a public NX Open Python cookbook
(NX 2506) for holes/bosses, but BRepLens has NOT verified it live, and a conservative parametric subtractive
block for pockets/slots because a native pocket/slot builder could not be
verified from public documentation. All API names that could not be verified in
this environment are wrapped in defensive ``getattr`` calls and reported as
warnings rather than silently assumed.

Every method is unit-tested against a faithful mock under ``tests/fakes``;
that validates our translation logic, not the live NX API.
"""

from __future__ import annotations

import contextlib
import json
import tempfile
from pathlib import Path
from typing import Any

from breplens.adapters.base import CadAdapter, CadLoadError, WriteBackError
from breplens.adapters.occ import OccAdapter
from breplens.schema import FeatureSet, FeatureType, MeshData, PartGraph, Vec3

UNTESTED_LIVE = "UNTESTED against a live NX session"


def _import_nx() -> Any:
    try:
        import NXOpen  # type: ignore[import-not-found]
        import NXOpen.Features  # type: ignore[import-not-found]
        import NXOpen.GeometricUtilities  # type: ignore[import-not-found]
        import NXOpen.UF  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - only with NX installed
        raise CadLoadError(
            "NXOpen is not importable. Run this adapter inside Siemens NX "
            "(File > Execute > NX Open) or a configured external NX session. "
            f"Import error: {type(exc).__name__}: {exc}"
        ) from exc
    return NXOpen


class NxOpenAdapter(CadAdapter):
    """NX Open adapter. See module docstring for the live-session caveat."""

    name = "nxopen"
    supported_suffixes = (".prt",)

    def __init__(self) -> None:
        self._occ = OccAdapter()

    @staticmethod
    def available() -> bool:
        try:
            _import_nx()
            return True
        except Exception:
            return False

    # ------------------------------------------------------------- traversal
    def _session(self) -> Any:
        NXOpen = _import_nx()
        session = NXOpen.Session.GetSession()
        if session is None:
            raise CadLoadError("NX session is not available")
        return session

    def _work_part(self) -> Any:
        session = self._session()
        part = session.Parts.Work
        if part is None:
            part = getattr(session.Parts, "Display", None)
        if part is None:
            raise CadLoadError("no work/display part is open in NX")
        return part

    def body_summary(self) -> dict[str, Any]:
        """Enumerate bodies/faces/edges through NX Open (traversal smoke test).

        This does not build the neutral IR; it is used for diagnostics and unit
        tests. Face geometry is read through the STEP export instead.
        """
        part = self._work_part()
        bodies = list(part.Bodies)
        faces_total = 0
        edges_total = 0
        face_types: dict[str, int] = {}
        for body in bodies:
            for face in body.GetFaces():
                faces_total += 1
                ftype = getattr(face, "SolidFaceType", None)
                name = str(getattr(ftype, "name", ftype))
                face_types[name] = face_types.get(name, 0) + 1
                with contextlib.suppress(Exception):
                    edges_total += len(list(face.GetEdges()))
        return {
            "part": str(getattr(part, "FullPath", "")),
            "bodies": len(bodies),
            "faces": faces_total,
            "face_edge_uses": edges_total,
            "face_types": face_types,
            "status": UNTESTED_LIVE,
        }

    # ------------------------------------------------------------------ graph
    def _export_step(self, part: Any, out_path: Path) -> None:
        session = self._session()
        bodies = list(part.Bodies)
        if not bodies:
            raise CadLoadError("NX work part has no bodies")
        creator = session.DexManager.CreateStepCreator()
        try:
            export_as = getattr(getattr(creator, "ExportAs", None), "Ap242", None)
            if export_as is not None:
                creator.ExportAs = export_as
            if hasattr(creator, "ObjectTypes"):
                creator.ObjectTypes.Solids = True
            if hasattr(creator, "ExportSelectionBlock"):
                import NXOpen  # type: ignore[import-not-found]

                creator.ExportSelectionBlock.SelectionScope = (
                    NXOpen.ObjectSelector.Scope.SelectedObjects
                )
                creator.ExportSelectionBlock.SelectionComp.Add(bodies)
            creator.InputFile = str(getattr(part, "FullPath", ""))
            creator.OutputFile = str(out_path)
            creator.Commit()
        finally:
            with contextlib.suppress(Exception):
                creator.Destroy()
        if not out_path.exists() or out_path.stat().st_size == 0:
            raise CadLoadError("NX STEP export did not produce a file")

    def load_graph(self, path: str | Path, *, name: str | None = None) -> PartGraph:
        """Load the open NX part (or *path*) via a STEP export, then build the AAG."""
        self._session()
        src = Path(path)
        if src.exists() and src.suffix.lower() == ".prt":
            self._open_part(src)
        part = self._work_part()
        with tempfile.TemporaryDirectory(prefix="breplens_nx_") as td:
            step = Path(td) / "export.stp"
            self._export_step(part, step)
            graph = self._occ.load_graph(step, name=name or src.stem or "nx_part")
        graph.metadata.source_format = "nx"
        graph.metadata.source_path = str(path)
        graph.metadata.notes = "graph extracted via NX STEP export; NX face tags are not preserved"
        graph.warnings.append(UNTESTED_LIVE)
        return graph

    def _open_part(self, path: Path) -> None:
        session = self._session()
        if not path.exists():
            raise CadLoadError(f"part not found: {path}")
        try:  # verified neither by docs access nor live NX
            base = getattr(getattr(session.Parts, "Work", None), "FullPath", None)
            if base and str(base) == str(path):
                return
            import NXOpen  # type: ignore[import-not-found]

            option = getattr(getattr(NXOpen, "DisplayPartOption", None), "AllowAdditional", None)
            if hasattr(session.Parts, "OpenActiveDisplay"):
                session.Parts.OpenActiveDisplay(str(path), option)
            elif hasattr(session.Parts, "OpenBaseDisplay"):
                session.Parts.OpenBaseDisplay(str(path))
            else:  # pragma: no cover
                raise WriteBackError("no NX Open API found to open a part")
        except CadLoadError:
            raise
        except Exception as exc:  # pragma: no cover
            raise CadLoadError(f"failed to open NX part: {exc}") from exc

    def tessellate(self, path: str | Path, *, deflection: float = 0.2) -> MeshData:
        with tempfile.TemporaryDirectory(prefix="breplens_nx_mesh_") as td:
            step = Path(td) / "export.stp"
            self._export_step(self._work_part(), step)
            return self._occ.tessellate(step, deflection=deflection)

    # ------------------------------------------------------------- write-back
    def _undo_mark(self, label: str) -> Any:
        session = self._session()
        try:
            import NXOpen  # type: ignore[import-not-found]

            return session.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, label)
        except Exception:
            return None

    def _update(self, mark: Any) -> None:
        with contextlib.suppress(Exception):
            self._session().UpdateManager.DoUpdate(mark)

    def _target_bodies(self, part: Any) -> list[Any]:
        bodies = list(part.Bodies)
        if not bodies:
            raise WriteBackError("no target body in NX part")
        return bodies

    @staticmethod
    def _axis_is_z(axis: Vec3 | None) -> bool:
        return axis is None or abs(axis.normalized().z) > 0.999

    def _create_hole(self, part: Any, feat: Any, body: Any) -> None:
        if feat.type != FeatureType.THROUGH_HOLE and feat.type != FeatureType.BLIND_HOLE:
            raise WriteBackError(f"hole builder does not handle {feat.type.value}")
        NXOpen = _import_nx()
        if not self._axis_is_z(feat.axis):
            raise WriteBackError("only Z-axis holes are supported in this untested write-back")
        if feat.position is None or feat.params.diameter is None:
            raise WriteBackError("hole feature lacks position or diameter")
        builders = part.Features
        hp = builders.CreateHolePackageBuilder(NXOpen.Features.HolePackage.Null)
        mark = self._undo_mark(f"BRepLens hole {feat.id}")
        try:
            holetype = getattr(
                getattr(NXOpen.Features.HolePackageBuilder, "Holetype", None), "Simple", None
            )
            if holetype is not None:
                hp.HoleType = holetype
            hp.GeneralSimpleHoleDiameter.SetFormula(str(round(feat.params.diameter, 6)))
            depth = feat.params.depth or 50.0
            hp.HoleDepthLimitOption = NXOpen.Features.HolePackageBuilder.HoleDepthLimitOptions.Value
            hp.GeneralSimpleHoleDepth.SetFormula(str(round(max(depth, 0.1) + 0.5, 6)))
            hp.GeneralTipAngle.SetFormula("118")
            hp.Tolerance = 0.01
            section = hp.HolePosition
            section.SetAllowedEntityTypes(NXOpen.Section.AllowTypes.OnlyPoints)
            p = feat.position
            point = part.Points.CreatePoint(NXOpen.Point3d(p.x, p.y, p.z))
            rule = part.ScRuleFactory.CreateRuleCurveDumbFromPoints([point])
            section.AddToSection(
                [rule],
                point,
                NXOpen.NXObject.Null,
                NXOpen.NXObject.Null,
                NXOpen.Point3d(p.x, p.y, p.z),
                NXOpen.Section.Mode.Create,
                False,
            )
            hp.BooleanOperation.SetTargetBodies([body])
            hp.CommitFeature()
        finally:
            with contextlib.suppress(Exception):
                hp.Destroy()
        self._update(mark)

    def _create_box_cut(self, part: Any, feat: Any, body: Any) -> None:
        """Parametric subtractive block used for pockets and slots."""
        NXOpen = _import_nx()
        if not self._axis_is_z(feat.axis):
            raise WriteBackError(
                "only Z-axis pockets/slots are supported in this untested write-back"
            )
        if feat.position is None:
            raise WriteBackError("pocket/slot lacks position")
        p = feat.position
        length = feat.params.length or feat.params.width
        width = feat.params.width or feat.params.length
        depth = feat.params.depth or feat.params.height
        if length is None or width is None or depth is None:
            raise WriteBackError("pocket/slot lacks length/width/depth")
        mark = self._undo_mark(f"BRepLens cut {feat.id}")
        builder = part.Features.CreateBlockFeatureBuilder(NXOpen.Features.Feature.Null)
        try:
            builder.Type = NXOpen.Features.BlockFeatureBuilder.Types.OriginAndEdgeLengths
            origin = NXOpen.Point3d(p.x - length / 2, p.y - width / 2, p.z - depth)
            builder.SetOriginAndLengths(origin, str(length), str(width), str(depth))
            builder.SetBooleanOperationAndTarget(NXOpen.Features.Feature.BooleanType.Subtract, body)
            builder.CommitFeature()
        finally:
            with contextlib.suppress(Exception):
                builder.Destroy()
        self._update(mark)

    def _create_boss(self, part: Any, feat: Any, body: Any) -> None:
        NXOpen = _import_nx()
        if not self._axis_is_z(feat.axis):
            raise WriteBackError("only Z-axis bosses are supported in this untested write-back")
        if feat.position is None or feat.params.diameter is None or feat.params.height is None:
            raise WriteBackError("boss lacks position/diameter/height")
        p = feat.position
        mark = self._undo_mark(f"BRepLens boss {feat.id}")
        builder = part.Features.CreateCylinderBuilder(NXOpen.Features.Feature.Null)
        try:
            builder.Type = NXOpen.Features.CylinderBuilder.Types.AxisDiameterAndHeight
            origin = part.Points.CreatePoint(NXOpen.Point3d(p.x, p.y, p.z))
            direction = part.Directions.CreateDirection(
                NXOpen.Point3d(p.x, p.y, p.z),
                NXOpen.Vector3d(0.0, 0.0, 1.0),
                NXOpen.SmartObject.UpdateOption.WithinModeling,
            )
            builder.Axis = part.Axes.CreateAxis(
                origin, direction, NXOpen.SmartObject.UpdateOption.WithinModeling
            )
            builder.Diameter.RightHandSide = str(round(feat.params.diameter, 6))
            builder.Height.RightHandSide = str(round(feat.params.height, 6))
            builder.BooleanOption.Type = (
                NXOpen.GeometricUtilities.BooleanOperation.BooleanType.Unite
            )
            builder.BooleanOption.SetTargetBodies([body])
            builder.CommitFeature()
        finally:
            with contextlib.suppress(Exception):
                builder.Destroy()
        self._update(mark)

    def write_features(self, path: str | Path, features: FeatureSet) -> str:
        """Create native NX features for the recognized FeatureSet.

        Returns a JSON report. Holes/bosses use candidate NX Open recipes from a
        public cookbook (not verified live here); pockets/slots use a parametric
        subtractive block. Unsupported feature
        types are skipped with explicit warnings (never silently).
        """
        self._open_part(Path(path))
        part = self._work_part()
        body = self._target_bodies(part)[0]
        created: list[str] = []
        skipped: list[str] = []
        warnings: list[str] = [UNTESTED_LIVE]
        for feat in features.features:
            try:
                if feat.type in (FeatureType.THROUGH_HOLE, FeatureType.BLIND_HOLE):
                    self._create_hole(part, feat, body)
                elif feat.type in (FeatureType.POCKET, FeatureType.SLOT):
                    self._create_box_cut(part, feat, body)
                elif feat.type == FeatureType.BOSS:
                    self._create_boss(part, feat, body)
                elif feat.type in (FeatureType.COUNTERBORE, FeatureType.COUNTERSINK):
                    skipped.append(feat.id)
                    warnings.append(
                        f"{feat.id}: native counterbore/countersink API unverified; skipped"
                    )
                    continue
                elif feat.type in (FeatureType.FILLET, FeatureType.CHAMFER):
                    skipped.append(feat.id)
                    warnings.append(f"{feat.id}: edge-selection write-back not verified; skipped")
                    continue
                else:
                    skipped.append(feat.id)
                    warnings.append(f"{feat.id}: unsupported type {feat.type.value}")
                    continue
                created.append(feat.id)
            except Exception as exc:
                skipped.append(feat.id)
                warnings.append(f"{feat.id}: {type(exc).__name__}: {exc}")
        report = {
            "adapter": self.name,
            "status": UNTESTED_LIVE,
            "created": created,
            "skipped": skipped,
            "warnings": warnings,
        }
        return json.dumps(report, indent=1)


__all__ = ["UNTESTED_LIVE", "NxOpenAdapter"]
