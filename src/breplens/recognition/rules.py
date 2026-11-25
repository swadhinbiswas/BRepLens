"""Rule-based machining feature recognition on the neutral AAG.

The recognizer is deterministic and purely geometric. It never reads labels.
Detection order matters because features share faces:

1. coaxial counterbores and countersinks (they claim cylinders/shoulders/cones),
2. remaining hole barrels,
3. bosses (external cylinders with a top cap and a large base),
4. pockets, slots and steps (planar floors with concave walls),
5. fillets (tangent blends) and chamfers (narrow 2-neighbour planar strips).

All parameters are measured from face geometry in millimetres. Confidence
values are fixed, documented heuristics -- not statistical probabilities -- and
are marked as such in the feature output.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from breplens.recognition.params import (
    RuleConfig,
    axis_interval,
    bbox_corners,
    concave_neighbors,
    face_center,
    footprint_dims,
    line_distance,
    neighbor_ids,
    opening_level,
    parallel,
    part_corners,
    radial_extent,
    smooth_neighbors,
    summarize_part,
)
from breplens.schema import (
    EdgeConvexity,
    FaceRecord,
    Feature,
    FeatureParams,
    FeatureSet,
    FeatureType,
    PartGraph,
    SurfaceType,
    TimingBreakdown,
    Vec3,
)


@dataclass
class _Segment:
    """A set of coaxial cylindrical faces of equal radius forming one barrel."""

    faces: list[int] = field(default_factory=list)
    radius: float = 0.0
    axis_origin: Vec3 = field(default_factory=Vec3)
    axis_dir: Vec3 = field(default_factory=lambda: Vec3(x=0.0, y=0.0, z=1.0))
    tmin: float = 0.0
    tmax: float = 0.0

    @property
    def diameter(self) -> float:
        return 2.0 * self.radius

    @property
    def length(self) -> float:
        return max(0.0, self.tmax - self.tmin)


def _merge_segments(graph: PartGraph, cfg: RuleConfig) -> list[_Segment]:
    cylinders = [
        f
        for f in graph.faces
        if f.surface_type == SurfaceType.CYLINDER
        and f.material_side == "outside"
        and not smooth_neighbors(graph, f.id)
    ]
    segs: list[_Segment] = []
    for f in cylinders:
        if f.axis_direction is None or f.axis_origin is None or f.radius is None:
            continue
        tmin, tmax = axis_interval(f)
        placed = False
        for s in segs:
            if (
                abs(s.radius - f.radius) <= cfg.radius_tol
                and parallel(s.axis_dir, f.axis_direction, cfg.parallel_dot)
                and line_distance(s.axis_origin, s.axis_dir, f.axis_origin) <= cfg.axis_tol
                and not (tmax < s.tmin - 0.6 or tmin > s.tmax + 0.6)
            ):
                s.faces.append(f.id)
                s.tmin = min(s.tmin, tmin)
                s.tmax = max(s.tmax, tmax)
                placed = True
                break
        if not placed:
            segs.append(
                _Segment(
                    faces=[f.id],
                    radius=f.radius,
                    axis_origin=f.axis_origin,
                    axis_dir=f.axis_direction,
                    tmin=tmin,
                    tmax=tmax,
                )
            )
    return segs


def _axis_clusters(segs: list[_Segment], cfg: RuleConfig) -> list[list[_Segment]]:
    clusters: list[list[_Segment]] = []
    for s in segs:
        for c in clusters:
            if (
                parallel(c[0].axis_dir, s.axis_dir, cfg.parallel_dot)
                and line_distance(c[0].axis_origin, c[0].axis_dir, s.axis_origin) <= cfg.axis_tol
            ):
                origin = c[0].axis_origin
                offset = (s.axis_origin - origin).dot(c[0].axis_dir)
                s.tmin += offset
                s.tmax += offset
                s.axis_origin = origin
                c.append(s)
                break
        else:
            clusters.append([s])
    return clusters


def _cap_plane(
    graph: PartGraph,
    seg: _Segment,
    claimed: set[int],
    cfg: RuleConfig,
) -> FaceRecord | None:
    diameter = seg.diameter
    for fid in seg.faces:
        for nb in neighbor_ids(graph, fid):
            if nb in claimed:
                continue
            f = graph.faces[nb]
            if f.surface_type != SurfaceType.PLANE or f.normal is None:
                continue
            if not parallel(f.normal, seg.axis_dir, 0.99):
                continue
            diag = (f.bbox_max - f.bbox_min).norm()
            if diag > cfg.cap_size_factor * max(diameter, 1e-6):
                continue
            t = (f.centroid - seg.axis_origin).dot(seg.axis_dir)
            if seg.tmin - 0.5 <= t <= seg.tmax + 0.5:
                return f
    return None


def _point_on_axis(seg: _Segment, t: float) -> Vec3:
    return seg.axis_origin + seg.axis_dir * t


def _make_feature(
    ftype: FeatureType,
    face_ids: list[int],
    params: FeatureParams,
    *,
    position: Vec3 | None,
    axis: Vec3 | None,
    confidence: float,
    notes: list[str] | None = None,
) -> Feature:
    return Feature(
        id="",  # assigned deterministically at the end
        type=ftype,
        confidence=confidence,
        face_ids=sorted(set(face_ids)),
        params=params,
        position=position,
        axis=axis,
        source="rules",
        notes=notes or [],
    )


def recognize_rules(graph: PartGraph, config: RuleConfig | None = None) -> FeatureSet:
    """Recognize features on *graph* with geometric rules."""
    cfg = config or RuleConfig()
    t0 = time.perf_counter()
    features: list[Feature] = []
    claimed: set[int] = set()
    notes: list[str] = []

    segs = _merge_segments(graph, cfg)

    # -------------------------------------------------- counterbore/countersink
    for cluster in _axis_clusters(segs, cfg):
        axis_dir = cluster[0].axis_dir
        cone_faces: list[FaceRecord] = []
        for f in graph.faces:
            if f.surface_type != SurfaceType.CONE or f.id in claimed:
                continue
            if f.axis_direction is None or f.axis_origin is None:
                continue
            if not parallel(f.axis_direction, axis_dir, cfg.parallel_dot):
                continue
            if line_distance(cluster[0].axis_origin, axis_dir, f.axis_origin) > 1.0:
                continue
            if any(f.id in neighbor_ids(graph, sf) for s in cluster for sf in s.faces):
                cone_faces.append(f)

        active = [s for s in cluster if not set(s.faces) <= claimed]
        if cone_faces and active:
            cone = max(cone_faces, key=lambda f: radial_extent(f))
            cyl = min(active, key=lambda s: s.radius)
            cone_tmin, cone_tmax = axis_interval(cone)
            cone_height = max(0.0, cone_tmax - cone_tmin)
            semi = abs(cone.semi_angle_deg or 0.0)
            angle = 2.0 * semi
            r_outer = cyl.radius + cone_height * math.tan(math.radians(semi))
            if r_outer > cyl.radius + 0.2 and 0.0 < angle < 150.0:
                cone_top_local = max(
                    (c - cyl.axis_origin).dot(cyl.axis_dir) for c in bbox_corners(cone)
                )
                t_open = max(cone_top_local, cyl.tmax)
                feats = [cone.id, *cyl.faces]
                claimed.update(feats)
                features.append(
                    _make_feature(
                        FeatureType.COUNTERSINK,
                        feats,
                        FeatureParams(
                            outer_diameter=2.0 * r_outer,
                            inner_diameter=cyl.diameter,
                            angle_deg=angle,
                            depth=cyl.length,
                            is_through=True,
                        ),
                        position=_point_on_axis(cyl, t_open),
                        axis=axis_dir,
                        confidence=0.95,
                        notes=["angle is the full included angle"],
                    )
                )
                continue

        by_radius: dict[float, _Segment] = {}
        for s in active:
            key = round(s.radius, 3)
            by_radius.setdefault(key, s)
        if len(by_radius) >= 2:
            radii = sorted(by_radius)
            outer, inner = by_radius[radii[-1]], by_radius[radii[0]]
            outer_neighbors: set[int] = set()
            for fid in outer.faces:
                outer_neighbors.update(neighbor_ids(graph, fid))
            inner_neighbors: set[int] = set()
            for fid in inner.faces:
                inner_neighbors.update(neighbor_ids(graph, fid))
            shoulder: FaceRecord | None = None
            for cand in sorted(outer_neighbors & inner_neighbors):
                f = graph.faces[cand]
                t_cand = (f.centroid - outer.axis_origin).dot(axis_dir)
                if (
                    f.surface_type == SurfaceType.PLANE
                    and f.normal is not None
                    and parallel(f.normal, axis_dir, 0.99)
                    and cand not in claimed
                    and min(outer.tmin, inner.tmin) - 0.5
                    <= t_cand
                    <= max(outer.tmax, inner.tmax) + 0.5
                ):
                    shoulder = f
                    break
            if shoulder is not None and outer.radius > inner.radius + cfg.radius_tol:
                face_ids = [*outer.faces, *inner.faces, shoulder.id]
                claimed.update(face_ids)
                t_open = max(outer.tmax, inner.tmax)
                features.append(
                    _make_feature(
                        FeatureType.COUNTERBORE,
                        face_ids,
                        FeatureParams(
                            outer_diameter=outer.diameter,
                            inner_diameter=inner.diameter,
                            depth=outer.length,
                            is_through=True,
                        ),
                        position=_point_on_axis(outer, t_open),
                        axis=axis_dir,
                        confidence=0.95,
                    )
                )
                for r in radii[1:-1]:
                    s = by_radius[r]
                    claimed.update(s.faces)
                    notes.append(f"extra coaxial cylinder radius {s.radius:.2f} merged")
                continue

        # ---------------------------------------------------------- plate holes
        for s in active:
            if set(s.faces) <= claimed:
                continue
            cap = _cap_plane(graph, s, claimed, cfg)
            cap_t = (cap.centroid - s.axis_origin).dot(s.axis_dir) if cap else None
            if cap is not None:
                open_t = s.tmax if (cap_t is not None and cap_t < (s.tmin + s.tmax) / 2) else s.tmin
                is_through = False
            else:
                open_t = s.tmax
                is_through = True
            depth = s.length
            if depth < cfg.min_depth:
                continue
            face_ids = [*s.faces] + ([cap.id] if cap else [])
            claimed.update(face_ids)
            features.append(
                _make_feature(
                    FeatureType.THROUGH_HOLE if is_through else FeatureType.BLIND_HOLE,
                    face_ids,
                    FeatureParams(
                        diameter=s.diameter,
                        depth=depth,
                        is_through=is_through,
                    ),
                    position=_point_on_axis(s, open_t),
                    axis=s.axis_dir,
                    confidence=0.95,
                )
            )

    # ----------------------------------------------------------------- bosses
    for f in graph.faces:
        if (
            f.id in claimed
            or f.surface_type != SurfaceType.CYLINDER
            or f.material_side != "inside"
            or f.radius is None
            or f.axis_direction is None
            or f.axis_origin is None
        ):
            continue
        seg = _Segment(
            faces=[f.id],
            radius=f.radius,
            axis_origin=f.axis_origin,
            axis_dir=f.axis_direction,
            tmin=axis_interval(f)[0],
            tmax=axis_interval(f)[1],
        )
        cap = _cap_plane(graph, seg, claimed, cfg)
        if cap is None:
            continue
        # A boss needs a large base plane at the other end (unlike a fillet).
        base: FaceRecord | None = None
        cap_t = (cap.centroid - seg.axis_origin).dot(seg.axis_dir)
        for nb in neighbor_ids(graph, f.id):
            g = graph.faces[nb]
            if (
                g.surface_type == SurfaceType.PLANE
                and g.normal is not None
                and parallel(g.normal, seg.axis_dir, 0.99)
                and (g.bbox_max - g.bbox_min).norm() > 3.0 * seg.diameter
            ):
                gt = (g.centroid - seg.axis_origin).dot(seg.axis_dir)
                if gt < cap_t - 0.5:
                    base = g
                    break
        if base is None:
            continue
        base_t = (base.centroid - seg.axis_origin).dot(seg.axis_dir)
        height = abs(cap_t - base_t)
        if height < cfg.min_depth:
            continue
        claimed.update({f.id, cap.id})
        features.append(
            _make_feature(
                FeatureType.BOSS,
                [f.id, cap.id],
                FeatureParams(diameter=seg.diameter, height=height),
                position=_point_on_axis(seg, min(base_t, cap_t)),
                axis=seg.axis_dir,
                confidence=0.9,
            )
        )

    # --------------------------------------------- pockets, slots and steps
    floors = sorted(
        (
            f
            for f in graph.faces
            if f.surface_type == SurfaceType.PLANE
            and f.normal is not None
            and f.id not in claimed
            and f.area >= cfg.min_area
            and any(nb not in claimed for nb in concave_neighbors(graph, f.id))
        ),
        key=lambda f: f.area,
        reverse=True,
    )
    for floor in floors:
        if floor.id in claimed or floor.normal is None:
            continue
        n = floor.normal.normalized()
        floor_t = floor.centroid.dot(n)
        part_max = max(c.dot(n) for c in part_corners(graph))
        if part_max - floor_t < cfg.min_depth:
            continue
        raw_walls = set(concave_neighbors(graph, floor.id))
        opening = opening_level(
            graph,
            floor,
            raw_walls,
            claimed=claimed,
            normal=n,
        )
        depth = opening - floor_t
        if depth < cfg.min_depth:
            continue
        walls: set[int] = set()
        for fid in raw_walls:
            if fid in claimed:
                continue
            f = graph.faces[fid]
            t_mid = (face_center(f) - floor.centroid).dot(n)
            # A wall must rise above the floor by a meaningful fraction of depth;
            # this rejects vertical walls being mistaken for floors.
            if t_mid < 0.15 * depth:
                continue
            if f.normal is not None and f.normal.dot(n) > 0.95:
                continue
            walls.add(fid)
        if not walls:
            continue
        length, width = footprint_dims(floor, n)
        cyl_walls = [wid for wid in walls if graph.faces[wid].surface_type == SurfaceType.CYLINDER]
        is_slot = (
            bool(cyl_walls)
            and width > 1.0
            and any(
                abs((graph.faces[wid].radius or 0.0) * 2.0 - width) <= max(0.8, 0.2 * width)
                for wid in cyl_walls
            )
        )
        if is_slot:
            ftype = FeatureType.SLOT
        elif len(walls) == 1:
            ftype = FeatureType.STEP
        else:
            ftype = FeatureType.POCKET
        face_ids = [floor.id, *sorted(walls)]
        if any(fid in claimed for fid in face_ids):
            continue
        claimed.update(face_ids)
        position = face_center(floor) + n * depth
        params = FeatureParams(length=length, width=width)
        if ftype == FeatureType.STEP:
            params = FeatureParams(length=length, width=width, height=depth)
        else:
            params.depth = depth
        features.append(
            _make_feature(
                ftype,
                face_ids,
                params,
                position=position,
                axis=n,
                confidence=0.9,
            )
        )

    # ---------------------------------------------------------------- fillets
    for f in graph.faces:
        if f.id in claimed:
            continue
        if f.surface_type not in (SurfaceType.CYLINDER, SurfaceType.TORUS, SurfaceType.SPHERE):
            continue
        if len(set(smooth_neighbors(graph, f.id))) < cfg.smooth_edge_min:
            continue
        radius = (
            f.radius if f.surface_type != SurfaceType.TORUS else (f.secondary_radius or f.radius)
        )
        if radius is None or radius < 0.2:
            continue
        if f.axis_direction is not None and f.axis_origin is not None:
            t0, t1 = axis_interval(f)
            position = _point_on_axis(
                _Segment(
                    faces=[f.id],
                    radius=radius,
                    axis_origin=f.axis_origin,
                    axis_dir=f.axis_direction,
                    tmin=t0,
                    tmax=t1,
                ),
                0.5 * (t0 + t1),
            )
            length = t1 - t0
        else:
            position = f.centroid
            length = (f.bbox_max - f.bbox_min).norm()
        claimed.add(f.id)
        features.append(
            _make_feature(
                FeatureType.FILLET,
                [f.id],
                FeatureParams(radius=radius, length=length),
                position=position,
                axis=f.axis_direction,
                confidence=0.85,
            )
        )

    # --------------------------------------------------------------- chamfers
    total_area = max(1.0, sum(g.area for g in graph.faces))
    for f in graph.faces:
        if f.id in claimed or f.surface_type != SurfaceType.PLANE or f.normal is None:
            continue
        extents = sorted(
            (
                f.bbox_max.x - f.bbox_min.x,
                f.bbox_max.y - f.bbox_min.y,
                f.bbox_max.z - f.bbox_min.z,
            )
        )
        width, _mid_dim, long_dim = extents
        if width <= 0.05 or width > 0.3 * long_dim:
            continue
        if f.area > 0.02 * total_area:
            continue
        if max(abs(f.normal.x), abs(f.normal.y), abs(f.normal.z)) > 0.95:
            continue
        neighbor_ids(graph, f.id)
        edge_map = {
            tuple(sorted((e.face_a, e.face_b))): e
            for e in graph.edges
            if e.face_a == f.id or e.face_b == f.id
        }
        long_edges = [
            e
            for (a, b), e in edge_map.items()
            if e.length >= 0.5 * long_dim and (a == f.id or b == f.id)
        ]
        if len(long_edges) < 2 or any(e.convexity != EdgeConvexity.CONVEX for e in long_edges):
            continue
        angles = []
        for e in long_edges:
            nb = e.face_b if e.face_a == f.id else e.face_a
            g = graph.faces[nb]
            if g.normal is not None:
                angles.append(math.degrees(math.acos(max(-1.0, min(1.0, f.normal.dot(g.normal))))))
        angle = sum(angles) / len(angles) if angles else 45.0
        claimed.add(f.id)
        features.append(
            _make_feature(
                FeatureType.CHAMFER,
                [f.id],
                FeatureParams(chamfer_width=width, angle_deg=angle),
                position=f.centroid,
                axis=None,
                confidence=0.8,
            )
        )

    # ------------------------------------------------------------ deterministic
    order = {
        t: i
        for i, t in enumerate(
            (
                FeatureType.COUNTERBORE,
                FeatureType.COUNTERSINK,
                FeatureType.THROUGH_HOLE,
                FeatureType.BLIND_HOLE,
                FeatureType.BOSS,
                FeatureType.POCKET,
                FeatureType.SLOT,
                FeatureType.STEP,
                FeatureType.FILLET,
                FeatureType.CHAMFER,
            )
        )
    }
    features.sort(key=lambda f: (order.get(f.type, 99), min(f.face_ids)))
    for i, feat in enumerate(features):
        feat.id = f"{feat.type.value}_{i}"
    elapsed = (time.perf_counter() - t0) * 1000.0
    summary = summarize_part(graph, len(features))
    return FeatureSet(
        part=graph.metadata.name,
        mode="rules",
        features=features,
        summary=summary,
        timings=TimingBreakdown(recognition_ms=elapsed, total_ms=elapsed),
        warnings=notes + graph.warnings,
    )


__all__ = ["recognize_rules"]
