"""Versioned, validated data model for BRepLens.

This module is CAD-agnostic and dependency-light: it may be imported by the
hosted demo, the API, the recognizers and the GNN without pulling in OpenCascade,
CadQuery or NX Open.

All lengths are millimetres and all angles degrees in the serialized form.
``schema_version`` is bumped on any backwards-incompatible change; the JSON
Schema is exported by ``scripts/export_schema.py``.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "1.0.0"


class Vec3(BaseModel):
    """A 3D vector or point in part coordinates (mm)."""

    model_config = ConfigDict(frozen=True)

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __add__(self, other: Vec3) -> Vec3:
        return Vec3(x=self.x + other.x, y=self.y + other.y, z=self.z + other.z)

    def __sub__(self, other: Vec3) -> Vec3:
        return Vec3(x=self.x - other.x, y=self.y - other.y, z=self.z - other.z)

    def __mul__(self, scalar: float) -> Vec3:
        return Vec3(x=self.x * scalar, y=self.y * scalar, z=self.z * scalar)

    __rmul__ = __mul__

    def dot(self, other: Vec3) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: Vec3) -> Vec3:
        return Vec3(
            x=self.y * other.z - self.z * other.y,
            y=self.z * other.x - self.x * other.z,
            z=self.x * other.y - self.y * other.x,
        )

    def norm(self) -> float:
        return math.sqrt(self.dot(self))

    def normalized(self, eps: float = 1e-12) -> Vec3:
        n = self.norm()
        if n < eps:
            return Vec3()
        return self * (1.0 / n)

    def distance_to(self, other: Vec3) -> float:
        return (self - other).norm()

    def as_tuple(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @classmethod
    def of(cls, values: tuple[float, float, float]) -> Vec3:
        """Construct from a plain ``(x, y, z)`` tuple."""
        return cls(x=float(values[0]), y=float(values[1]), z=float(values[2]))


class SurfaceType(StrEnum):
    """Analytic surface classes tracked by the neutral IR."""

    PLANE = "plane"
    CYLINDER = "cylinder"
    CONE = "cone"
    SPHERE = "sphere"
    TORUS = "torus"
    BSPLINE = "bspline"
    OTHER = "other"


class EdgeConvexity(StrEnum):
    """Local dihedral character of an edge between two faces."""

    CONVEX = "convex"
    CONCAVE = "concave"
    SMOOTH = "smooth"
    UNKNOWN = "unknown"


class CurveType(StrEnum):
    """Analytic curve classes tracked for edges."""

    LINE = "line"
    CIRCLE = "circle"
    ELLIPSE = "ellipse"
    BSPLINE = "bspline"
    OTHER = "other"


class FaceRecord(BaseModel):
    """One B-Rep face, attributed for recognition and learning."""

    model_config = ConfigDict(extra="forbid")

    id: int = Field(ge=0)
    surface_type: SurfaceType
    area: float = Field(ge=0.0)
    centroid: Vec3
    bbox_min: Vec3
    bbox_max: Vec3
    normal: Vec3 | None = None
    axis_origin: Vec3 | None = None
    axis_direction: Vec3 | None = None
    radius: float | None = None
    secondary_radius: float | None = None
    semi_angle_deg: float | None = None
    face_orientation: int = 0
    material_side: str | None = None
    step_face_id: int | None = None
    label: str | None = None
    label_id: int | None = None

    @field_validator("material_side")
    @classmethod
    def _side_vocab(cls, v: str | None) -> str | None:
        if v is not None and v not in {"inside", "outside"}:
            raise ValueError("material_side must be 'inside', 'outside' or None")
        return v

    @field_validator("axis_direction", "normal")
    @classmethod
    def _unit_or_none(cls, v: Vec3 | None) -> Vec3 | None:
        if v is None:
            return None
        return v.normalized()

    @model_validator(mode="after")
    def _bbox_ordered(self) -> FaceRecord:
        for lo, hi in zip(self.bbox_min.as_tuple(), self.bbox_max.as_tuple(), strict=True):
            if lo > hi + 1e-9:
                raise ValueError("bbox_min must not exceed bbox_max")
        return self


class EdgeRecord(BaseModel):
    """One B-Rep edge shared by two faces, with convexity attribution."""

    model_config = ConfigDict(extra="forbid")

    id: int = Field(ge=0)
    face_a: int = Field(ge=0)
    face_b: int = Field(ge=0)
    convexity: EdgeConvexity
    length: float = Field(ge=0.0)
    curve_type: CurveType
    midpoint: Vec3
    dihedral_angle_deg: float | None = None

    @model_validator(mode="after")
    def _distinct_faces(self) -> EdgeRecord:
        if self.face_a == self.face_b:
            raise ValueError("an edge must join two distinct faces")
        return self


class PartMetadata(BaseModel):
    """Provenance and units for a converted part."""

    model_config = ConfigDict(extra="forbid")

    name: str = ""
    source_format: str = "step"
    units: str = "mm"
    source_path: str | None = None
    n_faces: int = 0
    n_edges: int = 0
    bbox_min: Vec3 = Field(default_factory=Vec3)
    bbox_max: Vec3 = Field(default_factory=Vec3)
    volume: float | None = None
    n_features: int = 0
    generator: str | None = None
    family: str | None = None
    notes: str | None = None


class PartGraph(BaseModel):
    """Neutral Attributed Adjacency Graph for one part."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    metadata: PartMetadata = Field(default_factory=PartMetadata)
    faces: list[FaceRecord] = Field(default_factory=list)
    edges: list[EdgeRecord] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def face(self, face_id: int) -> FaceRecord:
        """Return a face by id (ids are contiguous and index-aligned)."""
        if not 0 <= face_id < len(self.faces):
            raise KeyError(f"face id {face_id} not in part with {len(self.faces)} faces")
        return self.faces[face_id]

    def adjacency(self) -> dict[int, list[tuple[int, EdgeRecord]]]:
        """Map face id -> [(neighbour face id, connecting edge), ...]."""
        adj: dict[int, list[tuple[int, EdgeRecord]]] = {f.id: [] for f in self.faces}
        for e in self.edges:
            adj.setdefault(e.face_a, []).append((e.face_b, e))
            adj.setdefault(e.face_b, []).append((e.face_a, e))
        return adj

    def neighbors(self, face_id: int) -> set[int]:
        out: set[int] = set()
        for e in self.edges:
            if e.face_a == face_id:
                out.add(e.face_b)
            elif e.face_b == face_id:
                out.add(e.face_a)
        return out

    def edge_between(self, a: int, b: int) -> EdgeRecord | None:
        for e in self.edges:
            if (e.face_a == a and e.face_b == b) or (e.face_a == b and e.face_b == a):
                return e
        return None


class FeatureType(StrEnum):
    """Machining feature classes recognized by BRepLens."""

    THROUGH_HOLE = "through_hole"
    BLIND_HOLE = "blind_hole"
    COUNTERBORE = "counterbore"
    COUNTERSINK = "countersink"
    POCKET = "pocket"
    SLOT = "slot"
    STEP = "step"
    BOSS = "boss"
    FILLET = "fillet"
    CHAMFER = "chamfer"
    UNKNOWN = "unknown"


MACHINING_FEATURES: tuple[FeatureType, ...] = (
    FeatureType.THROUGH_HOLE,
    FeatureType.BLIND_HOLE,
    FeatureType.COUNTERBORE,
    FeatureType.COUNTERSINK,
    FeatureType.POCKET,
    FeatureType.SLOT,
    FeatureType.STEP,
    FeatureType.BOSS,
    FeatureType.FILLET,
    FeatureType.CHAMFER,
)


class FeatureParams(BaseModel):
    """Measured parameters of a feature. Only fields measured are populated."""

    model_config = ConfigDict(extra="forbid")

    diameter: float | None = None
    inner_diameter: float | None = None
    outer_diameter: float | None = None
    depth: float | None = None
    radius: float | None = None
    length: float | None = None
    width: float | None = None
    height: float | None = None
    angle_deg: float | None = None
    chamfer_width: float | None = None
    wall_count: int | None = None
    is_through: bool | None = None
    extra: dict[str, float] = Field(default_factory=dict)


class Feature(BaseModel):
    """One recognized machining feature with measured parameters."""

    model_config = ConfigDict(extra="forbid")

    id: str
    type: FeatureType
    confidence: float = Field(ge=0.0, le=1.0)
    face_ids: list[int] = Field(default_factory=list)
    params: FeatureParams = Field(default_factory=FeatureParams)
    position: Vec3 | None = None
    axis: Vec3 | None = None
    source: str = "rules"
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _has_geometry(self) -> Feature:
        if not self.face_ids:
            raise ValueError("feature must reference at least one face")
        return self


class PartSummary(BaseModel):
    """Cheap, serializable summary used by lists and the similarity index."""

    model_config = ConfigDict(extra="forbid")

    n_faces: int = 0
    n_edges: int = 0
    n_cylinders: int = 0
    n_planes: int = 0
    n_cones: int = 0
    n_tori: int = 0
    n_concave_edges: int = 0
    n_convex_edges: int = 0
    n_smooth_edges: int = 0
    bbox_size: Vec3 = Field(default_factory=Vec3)
    volume: float | None = None
    surface_area: float | None = None
    n_features: int = 0


class TimingBreakdown(BaseModel):
    """Per-stage wall-clock timings in milliseconds."""

    model_config = ConfigDict(extra="forbid")

    load_ms: float = 0.0
    graph_ms: float = 0.0
    recognition_ms: float = 0.0
    total_ms: float = 0.0


class FeatureSet(BaseModel):
    """Versioned output of one recognition run on one part."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    part: str = ""
    mode: str = "rules"
    created_utc: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    features: list[Feature] = Field(default_factory=list)
    summary: PartSummary = Field(default_factory=PartSummary)
    timings: TimingBreakdown = Field(default_factory=TimingBreakdown)
    warnings: list[str] = Field(default_factory=list)
    model_version: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class MeshData(BaseModel):
    """Tessellation for the viewer: triangle soup plus per-face attribution."""

    model_config = ConfigDict(extra="forbid")

    vertices: list[tuple[float, float, float]]
    triangles: list[tuple[int, int, int]]
    triangle_face_ids: list[int]
    face_labels: dict[int, str] = Field(default_factory=dict)
    deflection: float = 0.0

    def validate_lengths(self) -> None:
        """Raise if the triangle/vertex arrays are inconsistent."""
        if len(self.triangles) != len(self.triangle_face_ids):
            raise ValueError("triangle_face_ids must match triangles")
        n = len(self.vertices)
        for tri in self.triangles:
            if any(i < 0 or i >= n for i in tri):
                raise ValueError("triangle index out of range")
