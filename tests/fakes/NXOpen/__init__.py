"""Minimal but faithful NXOpen mock used by BRepLens unit tests.

This is NOT the Siemens API. It reproduces the class/method shapes that the
BRepLens adapter uses (verified recipes plus documented traversal) so that the
adapter's translation logic is testable without NX. Live-session correctness is
explicitly out of scope.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

CALL_LOG: list[tuple[str, dict]] = []
_REPO = Path(__file__).resolve().parents[3]
_MOCK_STEP = _REPO / "samples" / "sample_04_pocket.step"


class NXException(Exception):
    pass


class _Null:
    @classmethod
    def Null(cls):
        return cls()


class NXObject(_Null):
    pass


class Point3d:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        self.X, self.Y, self.Z = float(x), float(y), float(z)


class Vector3d:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        self.X, self.Y, self.Z = float(x), float(y), float(z)


class Expression:
    def __init__(self) -> None:
        self.RightHandSide = "0"

    def SetFormula(self, value: str) -> None:
        self.RightHandSide = value


class FaceType:
    Planar = "Planar"
    Cylindrical = "Cylindrical"
    Conical = "Conical"


class Edge:
    def __init__(self, tag: int = 1) -> None:
        self.Tag = tag


class Face:
    def __init__(self, index: int, face_type: str = FaceType.Planar) -> None:
        self.Tag = 100 + index
        self.SolidFaceType = face_type

    def GetEdges(self):
        return [Edge(self.Tag * 10 + i) for i in range(2)]


class Body:
    def __init__(self, n_faces: int = 3) -> None:
        self._faces = [Face(i) for i in range(n_faces)]

    def GetFaces(self):
        return list(self._faces)

    def SetName(self, name: str) -> None:
        CALL_LOG.append(("Body.SetName", {"name": name}))


class Session:
    MarkVisibility = SimpleNamespace(Visible=1)

    @classmethod
    def GetSession(cls):
        return _SESSION

    def SetUndoMark(self, visibility, label):
        CALL_LOG.append(("SetUndoMark", {"label": label}))
        return {"mark": label}

    @property
    def UpdateManager(self):
        return SimpleNamespace(DoUpdate=self._update)

    def _update(self, mark):
        CALL_LOG.append(("UpdateManager.DoUpdate", {"mark": str(mark)}))

    @property
    def Parts(self):
        return _PARTS

    @property
    def DexManager(self):
        return _DEX


class _Parts:
    def __init__(self) -> None:
        self.Work = None
        self.Display = None

    def OpenActiveDisplay(self, path, option=None):
        CALL_LOG.append(("Parts.OpenActiveDisplay", {"path": str(path)}))
        self.Work = Part(str(path))
        self.Display = self.Work
        return self.Work


class _Points:
    def CreatePoint(self, p):
        CALL_LOG.append(("Points.CreatePoint", {"p": (p.X, p.Y, p.Z)}))
        return p


class _Directions:
    def CreateDirection(self, origin, vector, option):
        CALL_LOG.append(("Directions.CreateDirection", {"origin": (origin.X, origin.Y, origin.Z)}))
        return (origin, vector)


class _Axes:
    def CreateAxis(self, origin, direction, option):
        CALL_LOG.append(("Axes.CreateAxis", {}))
        return (origin, direction)


class _Collectors:
    def CreateCollector(self):
        return _Collector()


class _Collector:
    def ReplaceRules(self, rules, delete):
        CALL_LOG.append(("ScCollector.ReplaceRules", {"rules": len(rules)}))


class _RuleFactory:
    def CreateRuleCurveDumbFromPoints(self, points):
        return ("curve_points", points)

    def CreateRuleEdgeDumb(self, edges):
        return ("edges", len(edges))

    def CreateRuleFaceDumb(self, faces):
        return ("faces", len(faces))


class _SelectionComp:
    def Add(self, objects):
        CALL_LOG.append(("StepSelection.Add", {"n": len(objects)}))


class _StepCreator:
    ExportAs = SimpleNamespace(Ap242="Ap242")
    ObjectTypes = SimpleNamespace(Solids=False)

    def __init__(self) -> None:
        self.ExportSelectionBlock = SimpleNamespace(
            SelectionScope=None,
            SelectionComp=_SelectionComp(),
        )
        self.InputFile = ""
        self.OutputFile = ""

    def Commit(self):
        CALL_LOG.append(("StepCreator.Commit", {"output": self.OutputFile}))
        if _MOCK_STEP.exists():
            shutil.copyfile(_MOCK_STEP, self.OutputFile)
        else:  # pragma: no cover
            Path(self.OutputFile).write_text("ISO-10303-21;\nENDSEC;\n")

    def Destroy(self):
        pass


class _Dex:
    def CreateStepCreator(self):
        return _StepCreator()


class _BooleanOperation:
    def __init__(self) -> None:
        self.TargetBodies = []

    def SetTargetBodies(self, bodies):
        self.TargetBodies = list(bodies)
        CALL_LOG.append(("BooleanOperation.SetTargetBodies", {"n": len(self.TargetBodies)}))


class _BooleanOption:
    def __init__(self) -> None:
        self.Type = None
        self.TargetBodies = []

    def SetTargetBodies(self, bodies):
        self.TargetBodies = list(bodies)


class _BaseBuilder:
    name = "builder"

    def __init__(self) -> None:
        CALL_LOG.append((f"{self.name}.create", {}))

    def CommitFeature(self):
        CALL_LOG.append((f"{self.name}.CommitFeature", {}))
        return SimpleNamespace(GetBodies=lambda: [Body()])

    def Destroy(self):
        pass


class _HoleBuilder(_BaseBuilder):
    name = "HolePackageBuilder"
    Holetype = SimpleNamespace(Simple="Simple")
    HoleDepthLimitOptions = SimpleNamespace(Value="Value", ThroughBody="ThroughBody")

    def __init__(self) -> None:
        super().__init__()
        self.HoleType = None
        self.GeneralSimpleHoleDiameter = Expression()
        self.GeneralSimpleHoleDepth = Expression()
        self.GeneralTipAngle = Expression()
        self.HoleDepthLimitOption = None
        self.Tolerance = 0.0
        self.HolePosition = _Section()
        self.BooleanOperation = _BooleanOperation()


class _Section:
    AllowTypes = SimpleNamespace(OnlyPoints="OnlyPoints")
    Mode = SimpleNamespace(Create="Create")

    def SetAllowedEntityTypes(self, value):
        self.allowed = value

    def AddToSection(self, rules, point, a, b, p, mode, flag):
        CALL_LOG.append(("HolePosition.AddToSection", {"p": (p.X, p.Y, p.Z)}))


class _BlockBuilder(_BaseBuilder):
    name = "BlockFeatureBuilder"
    Types = SimpleNamespace(OriginAndEdgeLengths="OriginAndEdgeLengths")

    def __init__(self) -> None:
        super().__init__()
        self.Type = None
        self.OriginAndLengths = None
        self.BooleanOperation = _BooleanOperation()
        self.BooleanType = None

    def SetOriginAndLengths(self, origin, length, width, height):
        self.OriginAndLengths = (origin, length, width, height)
        CALL_LOG.append(("Block.SetOriginAndLengths", {"l": length, "w": width, "h": height}))

    def SetBooleanOperationAndTarget(self, kind, target):
        self.BooleanType = kind
        CALL_LOG.append(("Block.SetBooleanOperationAndTarget", {"kind": str(kind)}))


class _CylinderBuilder(_BaseBuilder):
    name = "CylinderBuilder"
    Types = SimpleNamespace(AxisDiameterAndHeight="AxisDiameterAndHeight")

    def __init__(self) -> None:
        super().__init__()
        self.Type = None
        self.Axis = None
        self.Diameter = Expression()
        self.Height = Expression()
        self.BooleanOption = _BooleanOption()


class _FeatureCollection:
    class Maps:
        pass

    def __init__(self, part) -> None:
        self.part = part

    def CreateHolePackageBuilder(self, null):
        return _HoleBuilder()

    def CreateBlockFeatureBuilder(self, null):
        return _BlockBuilder()

    def CreateCylinderBuilder(self, null):
        return _CylinderBuilder()

    def CreateEdgeBlendBuilder(self, null):
        return _BaseBuilder()

    def CreateChamferBuilder(self, null):
        return _BaseBuilder()


class Part:
    def __init__(self, path: str = "/tmp/mock.prt") -> None:
        self.FullPath = path
        self.Bodies = [Body(3)]
        self.Features = _FeatureCollection(self)
        self.Points = _Points()
        self.Directions = _Directions()
        self.Axes = _Axes()
        self.ScCollectors = _Collectors()
        self.ScRuleFactory = _RuleFactory()
        self.UnitCollection = SimpleNamespace(FindObject=lambda name: name)

    def Save(self, a=None, b=None) -> None:
        CALL_LOG.append(("Part.Save", {"path": self.FullPath}))


_SESSION = Session()
_PARTS = _Parts()
_DEX = _Dex()


class _FeaturesNamespace:
    Feature = type(
        "Feature", (_Null,), {"BooleanType": SimpleNamespace(Create="Create", Subtract="Subtract")}
    )
    HolePackage = type("HolePackage", (_Null,), {})
    HolePackageBuilder = _HoleBuilder
    BlockFeatureBuilder = _BlockBuilder
    CylinderBuilder = _CylinderBuilder


Features = _FeaturesNamespace()
GeometricUtilities = SimpleNamespace(
    BooleanOperation=SimpleNamespace(BooleanType=SimpleNamespace(Unite="Unite"))
)
Session = Session
Point3d = Point3d
Vector3d = Vector3d
NXObject = NXObject
Body = Body
Face = Face
Edge = Edge
Section = _Section
ObjectSelector = SimpleNamespace(Scope=SimpleNamespace(SelectedObjects="SelectedObjects"))
SmartObject = SimpleNamespace(UpdateOption=SimpleNamespace(WithinModeling="WithinModeling"))
DisplayPartOption = SimpleNamespace(AllowAdditional="AllowAdditional")
UF = SimpleNamespace(UFSession=SimpleNamespace(GetUFSession=lambda: None))


def reset() -> None:
    CALL_LOG.clear()
    _PARTS.Work = Part()
    _PARTS.Display = _PARTS.Work
