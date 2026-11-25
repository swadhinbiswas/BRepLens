"""OpenCascade (OCP) implementation of the :class:`~breplens.adapters.base.CadAdapter`.

Imports of OCP are confined to this module (and the tessellation helpers), so
the recognition, schema, service and hosted-demo layers stay CAD-agnostic.

Edge convexity
--------------
For each manifold edge shared by faces *A* and *B* we evaluate, at up to three
points along the edge:

* outward face normals ``n1``, ``n2`` (surface normal corrected by face
  orientation),
* the edge tangent ``t`` oriented along face *A*'s wire.

The signed quantity ``(n1 x n2) . t`` is positive for convex edges and negative
for concave edges under OpenCascade's outward-normal convention (calibrated by
tests on a box and a drilled plate). Edges whose adjacent normals differ by
less than ``smooth_angle_deg`` are classified ``smooth`` (tangent blends such as
fillet boundaries).

This is a local, geometry-only criterion; it is deterministic and does not
require a solid classifier. Non-manifold or seam edges are skipped and reported
as warnings.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

from OCP.Bnd import Bnd_Box
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCP.BRepBndLib import BRepBndLib
from OCP.BRepGProp import BRepGProp
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.GeomAbs import (
    GeomAbs_BezierSurface,
    GeomAbs_BSplineSurface,
    GeomAbs_Circle,
    GeomAbs_Cone,
    GeomAbs_Cylinder,
    GeomAbs_Ellipse,
    GeomAbs_Line,
    GeomAbs_OtherCurve,
    GeomAbs_OtherSurface,
    GeomAbs_Plane,
    GeomAbs_Sphere,
    GeomAbs_Torus,
)
from OCP.GeomAPI import GeomAPI_ProjectPointOnSurf
from OCP.GeomLProp import GeomLProp_SLProps
from OCP.gp import gp_Dir, gp_Pnt
from OCP.GProp import GProp_GProps
from OCP.IFSelect import IFSelect_RetDone
from OCP.Interface import Interface_Static
from OCP.STEPControl import STEPControl_Reader
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_REVERSED
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS, TopoDS_Shape
from OCP.TopTools import (
    TopTools_IndexedDataMapOfShapeListOfShape,
    TopTools_IndexedMapOfShape,
)

from breplens.adapters.base import CadAdapter, CadLoadError
from breplens.schema import (
    CurveType,
    EdgeConvexity,
    EdgeRecord,
    FaceRecord,
    MeshData,
    PartGraph,
    PartMetadata,
    SurfaceType,
    Vec3,
)

_SURFACE_MAP: dict[object, SurfaceType] = {
    GeomAbs_Plane: SurfaceType.PLANE,
    GeomAbs_Cylinder: SurfaceType.CYLINDER,
    GeomAbs_Cone: SurfaceType.CONE,
    GeomAbs_Sphere: SurfaceType.SPHERE,
    GeomAbs_Torus: SurfaceType.TORUS,
    GeomAbs_BSplineSurface: SurfaceType.BSPLINE,
    GeomAbs_BezierSurface: SurfaceType.BSPLINE,
    GeomAbs_OtherSurface: SurfaceType.OTHER,
}

_CURVE_MAP: dict[object, CurveType] = {
    GeomAbs_Line: CurveType.LINE,
    GeomAbs_Circle: CurveType.CIRCLE,
    GeomAbs_Ellipse: CurveType.ELLIPSE,
    GeomAbs_OtherCurve: CurveType.OTHER,
}

_SMOOTH_ANGLE_DEG = 12.0
_CONVEXITY_EPS = 1e-7


def _v3(p: gp_Pnt) -> Vec3:
    return Vec3(x=float(p.X()), y=float(p.Y()), z=float(p.Z()))


def _dir3(d: gp_Dir) -> Vec3:
    return Vec3(x=float(d.X()), y=float(d.Y()), z=float(d.Z()))


def _canonical_axis(direction: Vec3) -> Vec3:
    """Give axis directions a deterministic sign for stable JSON/ML features."""
    n = direction.normalized()
    t = n.as_tuple()
    idx = max(range(3), key=lambda i: abs(t[i]))
    if t[idx] < 0:
        return n * -1.0
    return n


def _bbox_of(shape: TopoDS_Shape) -> tuple[Vec3, Vec3]:
    box = Bnd_Box()
    box.SetGap(0.0)
    BRepBndLib.Add_s(shape, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    return (
        Vec3(x=float(xmin), y=float(ymin), z=float(zmin)),
        Vec3(x=float(xmax), y=float(ymax), z=float(zmax)),
    )


def _face_normal_and_area(face: Any) -> tuple[Vec3 | None, float, Vec3]:
    """Return (outward normal if analytic plane, area, centroid)."""
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, props)
    centroid = _v3(props.CentreOfMass())
    adaptor = BRepAdaptor_Surface(face)
    normal: Vec3 | None = None
    if adaptor.GetType() == GeomAbs_Plane:
        n = _dir3(adaptor.Plane().Axis().Direction())
        if face.Orientation() == TopAbs_REVERSED:
            n = n * -1.0
        normal = n.normalized()
    return normal, float(props.Mass()), centroid


def _surface_attributes(face: Any, surface_type: SurfaceType) -> dict[str, object]:
    """Extract axis/radius metadata for analytic surfaces."""
    attrs: dict[str, object] = {}
    adaptor = BRepAdaptor_Surface(face)
    try:
        if surface_type == SurfaceType.CYLINDER:
            cyl = adaptor.Cylinder()
            attrs["axis_origin"] = _v3(cyl.Axis().Location())
            attrs["axis_direction"] = _canonical_axis(_dir3(cyl.Axis().Direction()))
            attrs["radius"] = float(cyl.Radius())
        elif surface_type == SurfaceType.CONE:
            cone = adaptor.Cone()
            attrs["axis_origin"] = _v3(cone.Axis().Location())
            attrs["axis_direction"] = _canonical_axis(_dir3(cone.Axis().Direction()))
            attrs["radius"] = float(cone.RefRadius())
            attrs["semi_angle_deg"] = math.degrees(float(cone.SemiAngle()))
        elif surface_type == SurfaceType.SPHERE:
            sph = adaptor.Sphere()
            attrs["axis_origin"] = _v3(sph.Location())
            attrs["radius"] = float(sph.Radius())
        elif surface_type == SurfaceType.TORUS:
            tor = adaptor.Torus()
            attrs["axis_origin"] = _v3(tor.Axis().Location())
            attrs["axis_direction"] = _canonical_axis(_dir3(tor.Axis().Direction()))
            attrs["radius"] = float(tor.MajorRadius())
            attrs["secondary_radius"] = float(tor.MinorRadius())
    except Exception:  # pragma: no cover - defensive against exotic geometry
        attrs["warnings"] = "failed to read analytic surface attributes"
    return attrs


def _project_uv(surface: object, point_local: gp_Pnt) -> tuple[float, float] | None:
    try:
        proj = GeomAPI_ProjectPointOnSurf(point_local, surface)
        if proj.NbPoints() < 1:
            return None
        u, v = proj.LowerDistanceParameters()
        return float(u), float(v)
    except Exception:  # pragma: no cover
        return None


def _normal_at(face: Any, point_global: Vec3, *, is_plane: bool) -> Vec3 | None:
    """Outward unit normal of *face* at a point near the face."""
    try:
        if is_plane:
            adaptor = BRepAdaptor_Surface(face)
            n = _dir3(adaptor.Plane().Axis().Direction())
            if face.Orientation() == TopAbs_REVERSED:
                n = n * -1.0
            return n.normalized()

        surface = BRep_Tool.Surface_s(face)
        loc: TopLoc_Location = face.Location()
        p_local = gp_Pnt(*point_global.as_tuple())
        if not loc.IsIdentity():
            p_local = p_local.Transformed(loc.Transformation().Inverted())
        uv = _project_uv(surface, p_local)
        if uv is None:
            return None
        props = GeomLProp_SLProps(surface, uv[0], uv[1], 1, 1e-9)
        if not props.IsNormalDefined():
            return None
        n_local = props.Normal()
        n = Vec3(x=float(n_local.X()), y=float(n_local.Y()), z=float(n_local.Z()))
        if not loc.IsIdentity():
            trsf = loc.Transformation()
            n = Vec3(
                x=float(trsf.Value(1, 1)) * n.x
                + float(trsf.Value(1, 2)) * n.y
                + float(trsf.Value(1, 3)) * n.z,
                y=float(trsf.Value(2, 1)) * n.x
                + float(trsf.Value(2, 2)) * n.y
                + float(trsf.Value(2, 3)) * n.z,
                z=float(trsf.Value(3, 1)) * n.x
                + float(trsf.Value(3, 2)) * n.y
                + float(trsf.Value(3, 3)) * n.z,
            )
        if face.Orientation() == TopAbs_REVERSED:
            n = n * -1.0
        return n.normalized()
    except Exception:  # pragma: no cover
        return None


def _cylinder_material_side(face: Any, attrs: dict[str, object]) -> str | None:
    """Return 'outside' for a hole-like cylinder, 'inside' for a boss/fillet.

    A cylinder's outward normal points toward its axis when the material lies
    outside the cylinder (a hole) and away from the axis when the material lies
    inside (a boss or an external round/fillet). This is a geometric fact, not a
    heuristic threshold; the dot product is calibrated on test primitives.
    """
    origin = attrs.get("axis_origin")
    direction = attrs.get("axis_direction")
    radius = attrs.get("radius")
    if (
        not isinstance(origin, Vec3)
        or not isinstance(direction, Vec3)
        or not isinstance(radius, float)
    ):
        return None
    c = Vec3(x=0.0, y=0.0, z=0.0)
    bmin, bmax = _bbox_of(face)
    c = (bmin + bmax) * 0.5
    d = direction.normalized()
    w = c - origin
    perp = w - d * w.dot(d)
    u: Vec3
    if perp.norm() > 1e-3:
        u = perp.normalized()
    else:
        corners = [
            Vec3(
                x=(bmax.x if ix else bmin.x),
                y=(bmax.y if iy else bmin.y),
                z=(bmax.z if iz else bmin.z),
            )
            for ix in (0, 1)
            for iy in (0, 1)
            for iz in (0, 1)
        ]
        best = max(
            corners,
            key=lambda p: ((p - origin) - d * (p - origin).dot(d)).norm(),
        )
        v = (best - origin) - d * (best - origin).dot(d)
        if v.norm() < 1e-9:
            return None
        u = v.normalized()
    q = origin + d * w.dot(d)
    p = q + u * radius
    normal = _normal_at(face, p, is_plane=False)
    if normal is None:
        return None
    return "outside" if normal.dot(u) < -0.5 else "inside"


def _edge_geometry(
    edge: object,
) -> tuple[float, CurveType, Vec3, list[tuple[float, Vec3, Vec3]]]:
    """Return length, curve class, midpoint, and samples (t, point, tangent)."""
    curve = BRepAdaptor_Curve(edge)
    t0, t1 = float(curve.FirstParameter()), float(curve.LastParameter())
    props = GProp_GProps()
    BRepGProp.LinearProperties_s(edge, props)
    length = float(props.Mass())
    try:
        ctype = _CURVE_MAP.get(curve.GetType(), CurveType.OTHER)
    except Exception:  # pragma: no cover
        ctype = CurveType.OTHER
    mid = _v3(curve.Value(0.5 * (t0 + t1)))
    samples: list[tuple[float, Vec3, Vec3]] = []
    for frac in (0.25, 0.5, 0.75):
        t = t0 + frac * (t1 - t0)
        p = curve.Value(t)
        d = curve.DN(t, 1)
        tangent = Vec3(x=float(d.X()), y=float(d.Y()), z=float(d.Z())).normalized()
        samples.append((t, _v3(p), tangent))
    return length, ctype, mid, samples


def _wire_edge_orientations(
    shape: TopoDS_Shape, face_map: TopTools_IndexedMapOfShape, edge_map: TopTools_IndexedMapOfShape
) -> dict[tuple[int, int], bool]:
    """(face_id, edge_id) -> True when the wire traverses the edge reversed."""
    out: dict[tuple[int, int], bool] = {}
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while face_explorer.More():
        face = TopoDS.Face_s(face_explorer.Current())
        fid = face_map.FindIndex(face)
        edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)
        while edge_explorer.More():
            edge = TopoDS.Edge_s(edge_explorer.Current())
            eid = edge_map.FindIndex(edge)
            if eid > 0 and fid > 0:
                out[(fid - 1, eid - 1)] = edge.Orientation() == TopAbs_REVERSED
            edge_explorer.Next()
        face_explorer.Next()
    return out


def _classify_convexity(
    face_a: Any,
    face_b: Any,
    samples: list[tuple[float, Vec3, Vec3]],
    *,
    face_a_plane: bool,
    face_b_plane: bool,
    face_a_reversed_in_wire: bool,
    smooth_angle_deg: float = _SMOOTH_ANGLE_DEG,
) -> tuple[EdgeConvexity, float | None]:
    """Classify convexity from signed normals/tangent; see module docstring."""
    votes: list[int] = []
    angles: list[float] = []
    for _t, p, tangent in samples:
        n1 = _normal_at(face_a, p, is_plane=face_a_plane)
        n2 = _normal_at(face_b, p, is_plane=face_b_plane)
        if n1 is None or n2 is None or tangent.norm() < 1e-12:
            continue
        angle = math.degrees(math.acos(max(-1.0, min(1.0, n1.dot(n2)))))
        angles.append(angle)
        t = tangent * (-1.0 if face_a_reversed_in_wire else 1.0)
        signed = n1.cross(n2).dot(t)
        if abs(signed) > _CONVEXITY_EPS:
            votes.append(1 if signed > 0 else -1)
    mean_angle = sum(angles) / len(angles) if angles else None
    if mean_angle is not None and mean_angle < smooth_angle_deg:
        return EdgeConvexity.SMOOTH, mean_angle
    if not votes:
        return EdgeConvexity.UNKNOWN, mean_angle
    score = sum(votes) / len(votes)
    if score > 0.34:
        return EdgeConvexity.CONVEX, mean_angle
    if score < -0.34:
        return EdgeConvexity.CONCAVE, mean_angle
    return EdgeConvexity.UNKNOWN, mean_angle


class OccAdapter(CadAdapter):
    """OpenCascade adapter built on the pip-installable OCP package."""

    name = "occ"
    supported_suffixes = (".step", ".stp")

    def __init__(self, *, smooth_angle_deg: float = _SMOOTH_ANGLE_DEG) -> None:
        assert 0.0 < smooth_angle_deg < 45.0
        self.smooth_angle_deg = smooth_angle_deg

    def _read_step_face_ids(
        self, path: str | Path, face_map: TopTools_IndexedMapOfShape
    ) -> dict[int, int]:
        """Read integer face names written by CAD systems (MFCAD-style labels).

        Only used by dataset converters; normal recognition does not need it.
        Returns face-index -> STEP name id, empty when names are absent.
        """
        try:
            reader = STEPControl_Reader()
            if reader.ReadFile(str(path)) != IFSelect_RetDone:
                return {}
            reader.TransferRoots()
            shape = reader.OneShape()
            second = TopTools_IndexedMapOfShape()
            TopExp.MapShapes_s(shape, TopAbs_FACE, second)
            if second.Extent() != face_map.Extent():
                return {}
            transfer = reader.WS().TransferReader()
            out: dict[int, int] = {}
            for i in range(1, second.Extent() + 1):
                face = TopoDS.Face_s(second.FindKey(i))
                item = transfer.EntityFromShapeResult(face, 1)
                if item is None:
                    continue
                try:
                    name = item.Name().ToCString()
                    out[i - 1] = int(name)
                except Exception:
                    continue
            return out
        except Exception:  # pragma: no cover - converter-only convenience
            return {}

    # ------------------------------------------------------------------ load
    def _read_shape(self, path: str | Path) -> TopoDS_Shape:
        p = Path(path)
        if not p.exists():
            raise CadLoadError(f"file not found: {p}")
        if p.suffix.lower() not in self.supported_suffixes:
            raise CadLoadError(f"unsupported STEP suffix: {p.suffix}")
        if p.stat().st_size == 0:
            raise CadLoadError(f"empty file: {p}")
        if p.stat().st_size > 500 * 1024 * 1024:
            raise CadLoadError("file exceeds 500 MiB safety limit")
        try:
            Interface_Static.SetCVal_s("xstep.cascade.unit", "MM")
            reader = STEPControl_Reader()
            status = reader.ReadFile(str(p))
            if status != IFSelect_RetDone:
                raise CadLoadError(f"OpenCascade could not read STEP file (status {status})")
            reader.TransferRoots()
            shape = reader.OneShape()
        except CadLoadError:
            raise
        except Exception as exc:  # pragma: no cover - OCC raises varied types
            raise CadLoadError(f"failed to read STEP file: {exc}") from exc
        if shape.IsNull():
            raise CadLoadError("STEP file produced a null shape")
        n_faces = 0
        ex = TopExp_Explorer(shape, TopAbs_FACE)
        while ex.More():
            n_faces += 1
            ex.Next()
        if n_faces == 0:
            raise CadLoadError("STEP file contains no faces/solid bodies")
        return shape

    def load_graph(
        self, path: str | Path, *, name: str | None = None, read_step_face_ids: bool = False
    ) -> PartGraph:
        t0 = time.perf_counter()
        shape = self._read_shape(path)
        face_map = TopTools_IndexedMapOfShape()
        TopExp.MapShapes_s(shape, TopAbs_FACE, face_map)
        edge_map = TopTools_IndexedMapOfShape()
        TopExp.MapShapes_s(shape, TopAbs_EDGE, edge_map)
        edge_orient = _wire_edge_orientations(shape, face_map, edge_map)
        step_ids = self._read_step_face_ids(path, face_map) if read_step_face_ids else {}

        faces: list[FaceRecord] = []
        face_is_plane: dict[int, bool] = {}
        face_topo: dict[int, object] = {}
        for i in range(1, face_map.Extent() + 1):
            face = TopoDS.Face_s(face_map.FindKey(i))
            adaptor = BRepAdaptor_Surface(face)
            stype = _SURFACE_MAP.get(adaptor.GetType(), SurfaceType.OTHER)
            normal, area, centroid = _face_normal_and_area(face)
            bmin, bmax = _bbox_of(face)
            attrs = _surface_attributes(face, stype)
            attrs.pop("warnings", None)
            material_side = None
            if stype == SurfaceType.CYLINDER:
                material_side = _cylinder_material_side(face, attrs)
            rec = FaceRecord(
                id=i - 1,
                surface_type=stype,
                area=max(area, 0.0),
                centroid=centroid,
                bbox_min=bmin,
                bbox_max=bmax,
                normal=normal,
                face_orientation=int(face.Orientation()),
                material_side=material_side,
                step_face_id=step_ids.get(i - 1),
                **attrs,  # type: ignore[arg-type]
            )
            faces.append(rec)
            face_is_plane[i - 1] = stype == SurfaceType.PLANE
            face_topo[i - 1] = face

        ancestors = TopTools_IndexedDataMapOfShapeListOfShape()
        TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, ancestors)
        edges: list[EdgeRecord] = []
        warnings_out: list[str] = []

        for ai in range(1, ancestors.Extent() + 1):
            edge = TopoDS.Edge_s(ancestors.FindKey(ai))
            eid = edge_map.FindIndex(edge) - 1
            if eid < 0:
                continue
            face_list = ancestors.FindFromIndex(ai)
            unique: list[int] = []
            for face_shape in face_list:
                fid = face_map.FindIndex(face_shape) - 1
                if fid >= 0 and fid not in unique:
                    unique.append(fid)
            if len(unique) != 2:
                # Seam edges (one face) and non-manifold junctions are not AAG arcs.
                continue
            a, b = unique
            try:
                length, ctype, midpoint, samples = _edge_geometry(edge)
            except Exception as exc:  # pragma: no cover
                warnings_out.append(f"edge {eid} skipped: {exc}")
                continue
            if length < 1e-9:
                continue
            reversed_in_a = edge_orient.get((a, eid), False)
            # Straight edges have constant adjacent normals along their length;
            # one interior sample is sufficient and avoids expensive surface
            # projections. Curved edges keep three samples for robustness.
            sampled = [samples[1]] if ctype == CurveType.LINE else samples
            convexity, angle = _classify_convexity(
                face_topo[a],
                face_topo[b],
                sampled,
                face_a_plane=face_is_plane[a],
                face_b_plane=face_is_plane[b],
                face_a_reversed_in_wire=reversed_in_a,
                smooth_angle_deg=self.smooth_angle_deg,
            )
            edges.append(
                EdgeRecord(
                    id=eid,
                    face_a=a,
                    face_b=b,
                    convexity=convexity,
                    length=length,
                    curve_type=ctype,
                    midpoint=midpoint,
                    dihedral_angle_deg=angle,
                )
            )

        bmin, bmax = _bbox_of(shape)
        vol_props = GProp_GProps()
        try:
            BRepGProp.VolumeProperties_s(shape, vol_props)
            volume = float(vol_props.Mass())
        except Exception:  # pragma: no cover
            volume = None
        p = Path(path)
        meta = PartMetadata(
            name=name or p.stem,
            source_format="step",
            units="mm",
            source_path=str(p),
            n_faces=len(faces),
            n_edges=len(edges),
            bbox_min=bmin,
            bbox_max=bmax,
            volume=volume,
        )
        graph = PartGraph(metadata=meta, faces=faces, edges=edges, warnings=warnings_out)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        graph.metadata.notes = f"graph_build_ms={elapsed_ms:.1f}"
        return graph

    # ------------------------------------------------------------ tessellate
    def tessellate(self, path: str | Path, *, deflection: float = 0.2) -> MeshData:
        if not (0.001 <= deflection <= 5.0):
            raise CadLoadError("deflection must be between 0.001 and 5.0 mm")
        shape = self._read_shape(path)
        try:
            BRepMesh_IncrementalMesh(shape, deflection, False, 0.3, True)
        except Exception as exc:  # pragma: no cover
            raise CadLoadError(f"tessellation failed: {exc}") from exc
        face_map = TopTools_IndexedMapOfShape()
        TopExp.MapShapes_s(shape, TopAbs_FACE, face_map)
        verts: list[tuple[float, float, float]] = []
        tris: list[tuple[int, int, int]] = []
        tri_faces: list[int] = []
        for i in range(1, face_map.Extent() + 1):
            face = TopoDS.Face_s(face_map.FindKey(i))
            loc = TopLoc_Location()
            tri = BRep_Tool.Triangulation_s(face, loc)
            if tri is None:
                continue
            reversed_face = face.Orientation() == TopAbs_REVERSED
            node_offset = len(verts)
            for n in range(1, tri.NbNodes() + 1):
                p = tri.Node(n)
                if not loc.IsIdentity():
                    p = p.Transformed(loc.Transformation())
                verts.append((float(p.X()), float(p.Y()), float(p.Z())))
            for t in range(1, tri.NbTriangles() + 1):
                a, b, c = tri.Triangle(t).Get()
                if reversed_face:
                    b, c = c, b
                tris.append((node_offset + a - 1, node_offset + b - 1, node_offset + c - 1))
                tri_faces.append(i - 1)
        mesh = MeshData(
            vertices=verts,
            triangles=tris,
            triangle_face_ids=tri_faces,
            deflection=float(deflection),
        )
        mesh.validate_lengths()
        return mesh


__all__ = ["OccAdapter"]
