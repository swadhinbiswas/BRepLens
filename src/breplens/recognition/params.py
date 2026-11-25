"""Geometry measurement helpers shared by the rule recognizer and hybrid mode.

All functions consume only the neutral :class:`~breplens.schema.PartGraph`.
They prefer measured extents from face bounding boxes (which OpenCascade
computes exactly for trimmed faces) and avoid assumptions about orientation.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from breplens.schema import (
    EdgeConvexity,
    FaceRecord,
    PartGraph,
    PartSummary,
    SurfaceType,
    Vec3,
)


@dataclass(frozen=True)
class RuleConfig:
    """Tolerances in millimetres / dimensionless dot products."""

    radius_tol: float = 0.15
    axis_tol: float = 0.35
    parallel_dot: float = 0.995
    cap_size_factor: float = 2.4
    min_area: float = 6.0
    min_depth: float = 0.6
    smooth_edge_min: int = 2


def face_center(face: FaceRecord) -> Vec3:
    return (face.bbox_min + face.bbox_max) * 0.5


def parallel(a: Vec3, b: Vec3, dot_min: float = 0.995) -> bool:
    return abs(a.normalized().dot(b.normalized())) >= dot_min


def line_distance(origin_a: Vec3, dir_a: Vec3, origin_b: Vec3) -> float:
    d = dir_a.normalized()
    w = origin_b - origin_a
    return (w - d * w.dot(d)).norm()


def axis_interval(face: FaceRecord) -> tuple[float, float]:
    """Project the face bbox onto its axis and return (min, max)."""
    if face.axis_direction is None or face.axis_origin is None:
        face_center(face)
        return (0.0, 0.0)
    d = face.axis_direction
    o = face.axis_origin
    ts = [(corner - o).dot(d) for corner in bbox_corners(face)]
    return (min(ts), max(ts))


def bbox_corners(face: FaceRecord) -> list[Vec3]:
    lo, hi = face.bbox_min, face.bbox_max
    return [
        Vec3(x=(hi.x if ix else lo.x), y=(hi.y if iy else lo.y), z=(hi.z if iz else lo.z))
        for ix in (0, 1)
        for iy in (0, 1)
        for iz in (0, 1)
    ]


def project_point(point: Vec3, origin: Vec3, direction: Vec3) -> float:
    return (point - origin).dot(direction)


def point_at(origin: Vec3, direction: Vec3, t: float) -> Vec3:
    return origin + direction * t


def radial_extent(face: FaceRecord) -> float:
    """Max perpendicular distance of the face bbox corners from its axis."""
    if face.axis_origin is None or face.axis_direction is None:
        return 0.0
    o, d = face.axis_origin, face.axis_direction
    best = 0.0
    for c in bbox_corners(face):
        w = c - o
        best = max(best, (w - d * w.dot(d)).norm())
    return best


def adjacency(graph: PartGraph) -> dict[int, list[tuple[int, int]]]:
    """Neighbour map with edge ids for fast traversal."""
    adj: dict[int, list[tuple[int, int]]] = {f.id: [] for f in graph.faces}
    for e in graph.edges:
        adj.setdefault(e.face_a, []).append((e.face_b, e.id))
        adj.setdefault(e.face_b, []).append((e.face_a, e.id))
    return adj


def edge_of(graph: PartGraph, a: int, b: int) -> int | None:
    for e in graph.edges:
        if (e.face_a == a and e.face_b == b) or (e.face_a == b and e.face_b == a):
            return e.id
    return None


def concave_neighbors(graph: PartGraph, face_id: int) -> list[int]:
    out: list[int] = []
    for e in graph.edges:
        if e.convexity != EdgeConvexity.CONCAVE:
            continue
        if e.face_a == face_id:
            out.append(e.face_b)
        elif e.face_b == face_id:
            out.append(e.face_a)
    return out


def smooth_neighbors(graph: PartGraph, face_id: int) -> list[int]:
    out: list[int] = []
    for e in graph.edges:
        if e.convexity != EdgeConvexity.SMOOTH:
            continue
        if e.face_a == face_id:
            out.append(e.face_b)
        elif e.face_b == face_id:
            out.append(e.face_a)
    return out


def neighbor_ids(graph: PartGraph, face_id: int) -> list[int]:
    out: list[int] = []
    for e in graph.edges:
        if e.face_a == face_id:
            out.append(e.face_b)
        elif e.face_b == face_id:
            out.append(e.face_a)
    return out


def grow_concave_region(
    graph: PartGraph,
    seeds: list[int],
    *,
    normal: Vec3,
    t_max: float,
    exclude: set[int] | None = None,
    max_faces: int = 64,
) -> set[int]:
    """Flood a region through concave edges, bounded along *normal*."""
    exclude = exclude or set()
    seen = {s for s in seeds if s not in exclude}
    q = deque(seen)
    while q and len(seen) < max_faces:
        fid = q.popleft()
        for nb in concave_neighbors(graph, fid):
            if nb in seen or nb in exclude:
                continue
            f = graph.faces[nb]
            t = (face_center(f) - graph.faces[seeds[0]].centroid).dot(normal)
            if t < -0.5 or t > t_max + 0.5:
                continue
            seen.add(nb)
            q.append(nb)
    return seen


def opening_level(
    graph: PartGraph,
    floor: FaceRecord,
    walls: set[int],
    *,
    claimed: set[int],
    normal: Vec3,
) -> float:
    """Find the parallel face that forms the opening of a recess.

    Looks at faces adjacent to the walls with a parallel outward normal above
    the floor, and falls back to the part extent along ``normal``.
    """
    floor_t = floor.centroid.dot(normal)
    candidates: list[float] = []
    for wid in walls:
        for nb in neighbor_ids(graph, wid):
            if nb == floor.id or nb in claimed:
                continue
            f = graph.faces[nb]
            if f.surface_type != SurfaceType.PLANE or f.normal is None:
                continue
            if f.normal.dot(normal) < 0.99:
                continue
            t = f.centroid.dot(normal)
            if t > floor_t + 1e-3:
                candidates.append(t)
    if candidates:
        return min(candidates)
    proj = [c.dot(normal) for c in part_corners(graph)]
    return max(proj)


def part_corners(graph: PartGraph) -> list[Vec3]:
    lo, hi = graph.metadata.bbox_min, graph.metadata.bbox_max
    return [
        Vec3(x=(hi.x if ix else lo.x), y=(hi.y if iy else lo.y), z=(hi.z if iz else lo.z))
        for ix in (0, 1)
        for iy in (0, 1)
        for iz in (0, 1)
    ]


def footprint_dims(face: FaceRecord, normal: Vec3) -> tuple[float, float]:
    """Return (long, short) extents of the face bbox perpendicular to *normal*."""
    n = normal.normalized()
    u = Vec3(x=1.0, y=0.0, z=0.0)
    if abs(n.dot(u)) > 0.9:
        u = Vec3(x=0.0, y=1.0, z=0.0)
    u = (u - n * n.dot(u)).normalized()
    v = n.cross(u).normalized()
    us = [c.dot(u) for c in bbox_corners(face)]
    vs = [c.dot(v) for c in bbox_corners(face)]
    dims = sorted((max(us) - min(us), max(vs) - min(vs)), reverse=True)
    return dims[0], dims[1]


def plane_offset(face: FaceRecord) -> float:
    if face.normal is None:
        return 0.0
    return face.centroid.dot(face.normal)


def summarize_part(graph: PartGraph, n_features: int = 0) -> PartSummary:
    size = graph.metadata.bbox_max - graph.metadata.bbox_min
    return PartSummary(
        n_faces=len(graph.faces),
        n_edges=len(graph.edges),
        n_cylinders=sum(f.surface_type == SurfaceType.CYLINDER for f in graph.faces),
        n_planes=sum(f.surface_type == SurfaceType.PLANE for f in graph.faces),
        n_cones=sum(f.surface_type == SurfaceType.CONE for f in graph.faces),
        n_tori=sum(f.surface_type == SurfaceType.TORUS for f in graph.faces),
        n_concave_edges=sum(e.convexity == EdgeConvexity.CONCAVE for e in graph.edges),
        n_convex_edges=sum(e.convexity == EdgeConvexity.CONVEX for e in graph.edges),
        n_smooth_edges=sum(e.convexity == EdgeConvexity.SMOOTH for e in graph.edges),
        bbox_size=size,
        volume=graph.metadata.volume,
        surface_area=sum(f.area for f in graph.faces),
        n_features=n_features,
    )


def dihedral_between(a: FaceRecord, b: FaceRecord) -> float | None:
    if a.normal is None or b.normal is None:
        return None
    return math.degrees(math.acos(max(-1.0, min(1.0, a.normal.dot(b.normal)))))


__all__ = [
    "RuleConfig",
    "adjacency",
    "axis_interval",
    "bbox_corners",
    "concave_neighbors",
    "dihedral_between",
    "edge_of",
    "face_center",
    "footprint_dims",
    "grow_concave_region",
    "line_distance",
    "neighbor_ids",
    "opening_level",
    "parallel",
    "part_corners",
    "plane_offset",
    "point_at",
    "project_point",
    "radial_extent",
    "smooth_neighbors",
    "summarize_part",
]
