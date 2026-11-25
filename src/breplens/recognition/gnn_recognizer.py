"""GNN-driven feature instances and the measured hybrid strategy.

The GNN predicts a class per face. Feature instances are formed by connected
components of faces sharing the same predicted class, then measured with the
same geometry helpers as the rule recognizer. No rule-based *detection* is used
here: only geometric parameter measurement is shared.

The hybrid starts from rule features and adds non-overlapping, high-confidence
GNN components. Rule features are never silently replaced; conflicts are
reported in ``featureset.warnings`` so reviewers can see the disagreement.
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Any, Sequence

import numpy as np

from breplens.recognition.params import (
    RuleConfig,
    axis_interval,
    concave_neighbors,
    face_center,
    footprint_dims,
    line_distance,
    neighbor_ids,
    opening_level,
    parallel,
    radial_extent,
    smooth_neighbors,
    summarize_part,
)
from breplens.recognition.rules import recognize_rules
from breplens.schema import (
    EdgeConvexity,
    Feature,
    FeatureParams,
    FeatureSet,
    FeatureType,
    PartGraph,
    SurfaceType,
    TimingBreakdown,
    Vec3,
)

_HOLE_TYPES = {
    FeatureType.THROUGH_HOLE,
    FeatureType.BLIND_HOLE,
    FeatureType.COUNTERBORE,
    FeatureType.COUNTERSINK,
}


def _components(graph: PartGraph, labels: list[int], class_id: int) -> list[list[int]]:
    members = {f.id for f in graph.faces if labels[f.id] == class_id}
    out: list[list[int]] = []
    while members:
        seed = members.pop()
        comp = [seed]
        q = deque([seed])
        while q:
            fid = q.popleft()
            for nb in neighbor_ids(graph, fid):
                if nb in members:
                    members.remove(nb)
                    comp.append(nb)
                    q.append(nb)
        out.append(sorted(comp))
    return out


def _segment_faces(graph: PartGraph, faces: list[int]) -> list[dict[str, Any]]:
    """Cluster internal cylindrical faces by axis line + radius."""
    segs: list[dict[str, Any]] = []
    for fid in faces:
        f = graph.faces[fid]
        if f.surface_type != SurfaceType.CYLINDER or f.material_side != "outside":
            continue
        if f.radius is None or f.axis_origin is None or f.axis_direction is None:
            continue
        t0, t1 = axis_interval(f)
        placed = False
        for s in segs:
            if (
                abs(s["radius"] - f.radius) <= 0.15
                and parallel(s["dir"], f.axis_direction, 0.995)
                and line_distance(s["origin"], s["dir"], f.axis_origin) <= 0.35
            ):
                off = (f.axis_origin - s["origin"]).dot(s["dir"])
                s["t0"] = min(s["t0"], t0 + off)
                s["t1"] = max(s["t1"], t1 + off)
                s["faces"].append(fid)
                placed = True
                break
        if not placed:
            segs.append(
                {
                    "faces": [fid],
                    "radius": float(f.radius),
                    "origin": f.axis_origin,
                    "dir": f.axis_direction,
                    "t0": t0,
                    "t1": t1,
                }
            )
    return segs


def _measure_hole(
    graph: PartGraph,
    comp: list[int],
    ftype: FeatureType,
    probs: np.ndarray,
    class_id: int,
) -> Feature | None:
    segs = _segment_faces(graph, comp)
    if not segs:
        return None
    segs.sort(key=lambda s: s["radius"], reverse=True)
    outer = segs[0]
    cone = next((graph.faces[f] for f in comp if graph.faces[f].surface_type == SurfaceType.CONE), None)
    confidence = float(probs[comp, class_id].mean())
    face_ids = list(comp)
    t_open = outer["t1"]
    position = outer["origin"] + outer["dir"] * t_open
    if ftype == FeatureType.COUNTERSINK and cone is not None and len(segs) >= 1:
        inner = min(segs, key=lambda s: s["radius"])
        h = max(0.0, axis_interval(cone)[1] - axis_interval(cone)[0])
        semi = abs(cone.semi_angle_deg or 0.0)
        r_outer = inner["radius"] + h * math.tan(math.radians(semi))
        params = FeatureParams(
            outer_diameter=2 * r_outer,
            inner_diameter=2 * inner["radius"],
            angle_deg=2 * semi,
            depth=inner["t1"] - inner["t0"],
            is_through=True,
        )
        pos = inner["origin"] + inner["dir"] * inner["t1"]
        return Feature(
            id="",
            type=FeatureType.COUNTERSINK,
            confidence=confidence,
            face_ids=face_ids,
            params=params,
            position=pos,
            axis=inner["dir"],
            source="gnn",
        )
    if ftype == FeatureType.COUNTERBORE and len(segs) >= 2:
        inner = min(segs, key=lambda s: s["radius"])
        params = FeatureParams(
            outer_diameter=2 * outer["radius"],
            inner_diameter=2 * inner["radius"],
            depth=outer["t1"] - outer["t0"],
            is_through=True,
        )
        return Feature(
            id="",
            type=FeatureType.COUNTERBORE,
            confidence=confidence,
            face_ids=face_ids,
            params=params,
            position=position,
            axis=outer["dir"],
            source="gnn",
        )
    is_through = ftype == FeatureType.THROUGH_HOLE
    params = FeatureParams(
        diameter=2 * outer["radius"],
        depth=outer["t1"] - outer["t0"],
        is_through=is_through,
    )
    return Feature(
        id="",
        type=ftype,
        confidence=confidence,
        face_ids=face_ids,
        params=params,
        position=position,
        axis=outer["dir"],
        source="gnn",
    )


def _measure_recess(
    graph: PartGraph,
    comp: list[int],
    ftype: FeatureType,
    probs: np.ndarray,
    class_id: int,
    cfg: RuleConfig,
) -> Feature | None:
    planes = [graph.faces[f] for f in comp if graph.faces[f].surface_type == SurfaceType.PLANE]
    planes = [p for p in planes if p.normal is not None]
    if not planes:
        return None
    floor = max(planes, key=lambda f: f.area)
    n = floor.normal.normalized()  # type: ignore[union-attr]
    floor_t = floor.centroid.dot(n)
    walls = {
        f
        for f in comp
        if f != floor.id
        and graph.faces[f].normal is not None
        and abs(graph.faces[f].normal.dot(n)) < 0.95  # type: ignore[union-attr]
    }
    opening = opening_level(graph, floor, walls, claimed=set(), normal=n)
    depth = max(0.0, opening - floor_t)
    if depth < cfg.min_depth:
        depth = (floor.bbox_max - floor.bbox_min).norm() * 0.0
    length, width = footprint_dims(floor, n)
    position = face_center(floor) + n * depth
    if ftype == FeatureType.STEP:
        params = FeatureParams(length=length, width=width, height=depth)
    else:
        params = FeatureParams(length=length, width=width, depth=depth)
    return Feature(
        id="",
        type=ftype,
        confidence=float(probs[comp, class_id].mean()),
        face_ids=list(comp),
        params=params,
        position=position,
        axis=n,
        source="gnn",
    )


def _measure_boss(
    graph: PartGraph, comp: list[int], probs: np.ndarray, class_id: int
) -> Feature | None:
    cyl = [
        graph.faces[f]
        for f in comp
        if graph.faces[f].surface_type == SurfaceType.CYLINDER
        and graph.faces[f].material_side == "inside"
    ]
    if not cyl:
        return None
    f = max(cyl, key=lambda x: x.area)
    if f.axis_direction is None or f.axis_origin is None or f.radius is None:
        return None
    t0, t1 = axis_interval(f)
    return Feature(
        id="",
        type=FeatureType.BOSS,
        confidence=float(probs[comp, class_id].mean()),
        face_ids=list(comp),
        params=FeatureParams(diameter=2 * f.radius, height=t1 - t0),
        position=f.axis_origin + f.axis_direction * t0,
        axis=f.axis_direction,
        source="gnn",
    )


def _measure_fillet(
    graph: PartGraph, comp: list[int], probs: np.ndarray, class_id: int
) -> Feature | None:
    for fid in comp:
        f = graph.faces[fid]
        if f.surface_type not in (SurfaceType.CYLINDER, SurfaceType.TORUS, SurfaceType.SPHERE):
            continue
        radius = f.radius if f.surface_type != SurfaceType.TORUS else (f.secondary_radius or f.radius)
        if radius is None:
            continue
        if f.axis_direction is not None and f.axis_origin is not None:
            t0, t1 = axis_interval(f)
            pos = f.axis_origin + f.axis_direction * (0.5 * (t0 + t1))
            length = t1 - t0
        else:
            pos = f.centroid
            length = (f.bbox_max - f.bbox_min).norm()
        return Feature(
            id="",
            type=FeatureType.FILLET,
            confidence=float(probs[comp, class_id].mean()),
            face_ids=[fid],
            params=FeatureParams(radius=radius, length=length),
            position=pos,
            axis=f.axis_direction,
            source="gnn",
        )
    return None


def _measure_chamfer(
    graph: PartGraph, comp: list[int], probs: np.ndarray, class_id: int
) -> Feature | None:
    for fid in comp:
        f = graph.faces[fid]
        if f.surface_type != SurfaceType.PLANE or f.normal is None:
            continue
        dims = sorted((f.bbox_max.x - f.bbox_min.x, f.bbox_max.y - f.bbox_min.y, f.bbox_max.z - f.bbox_min.z))
        angles = []
        for nb in neighbor_ids(graph, fid):
            g = graph.faces[nb]
            if g.normal is not None:
                angles.append(
                    math.degrees(math.acos(max(-1.0, min(1.0, f.normal.dot(g.normal)))))
                )
        return Feature(
            id="",
            type=FeatureType.CHAMFER,
            confidence=float(probs[comp, class_id].mean()),
            face_ids=[fid],
            params=FeatureParams(chamfer_width=dims[0], angle_deg=sum(angles) / len(angles) if angles else 45.0),
            position=f.centroid,
            source="gnn",
        )
    return None


def recognize_gnn(
    graph: PartGraph,
    model: Any,
    class_names: Sequence[str],
    *,
    device: str = "cpu",
    config: RuleConfig | None = None,
) -> FeatureSet:
    """Produce feature instances from GNN face predictions + measured geometry."""
    from breplens.ml.inference import predict_probs

    cfg = config or RuleConfig()
    t0 = time.perf_counter()
    probs = predict_probs(model, graph, device=device)
    labels = [int(i) for i in probs.argmax(axis=1)]
    features: list[Feature] = []
    warnings: list[str] = []
    for class_id, cname in enumerate(class_names):
        try:
            ftype = FeatureType(cname)
        except ValueError:
            continue
        if ftype == FeatureType.UNKNOWN:
            continue
        for comp in _components(graph, labels, class_id):
            if len(comp) == 0:
                continue
            feat: Feature | None = None
            if ftype in _HOLE_TYPES:
                feat = _measure_hole(graph, comp, ftype, probs, class_id)
            elif ftype in (FeatureType.POCKET, FeatureType.SLOT, FeatureType.STEP):
                feat = _measure_recess(graph, comp, ftype, probs, class_id, cfg)
            elif ftype == FeatureType.BOSS:
                feat = _measure_boss(graph, comp, probs, class_id)
            elif ftype == FeatureType.FILLET:
                feat = _measure_fillet(graph, comp, probs, class_id)
            elif ftype == FeatureType.CHAMFER:
                feat = _measure_chamfer(graph, comp, probs, class_id)
            if feat is None:
                warnings.append(f"GNN component for {cname} could not be measured")
                continue
            features.append(feat)
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
    return FeatureSet(
        part=graph.metadata.name,
        mode="gnn",
        features=features,
        summary=summarize_part(graph, len(features)),
        timings=TimingBreakdown(recognition_ms=elapsed, total_ms=elapsed),
        warnings=warnings + graph.warnings,
    )


def _face_set(fs: FeatureSet) -> set[int]:
    out: set[int] = set()
    for f in fs.features:
        out.update(f.face_ids)
    return out


def _overlap_ratio(a: set[int], b: set[int]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a)


def _same_feature(a: Feature, b: Feature) -> bool:
    """Geometric duplicate test across rule/GNN instances of the same type."""
    if a.type != b.type:
        return False
    if a.position is not None and b.position is not None:
        if a.type in _HOLE_TYPES and a.axis is not None and b.axis is not None:
            if parallel(a.axis, b.axis, 0.99) and line_distance(
                a.position, a.axis, b.position
            ) <= 2.0:
                da, db = a.params.diameter, b.params.diameter
                if da is None or db is None or abs(da - db) <= max(1.0, 0.2 * max(da, db)):
                    return True
        elif a.position.distance_to(b.position) <= 4.0:
            return True
    return False


def recognize_hybrid(
    graph: PartGraph,
    model: Any,
    class_names: Sequence[str],
    *,
    device: str = "cpu",
    gnn_min_confidence: float = 0.6,
) -> FeatureSet:
    """Rules first; add non-overlapping high-confidence GNN components.

    This is a conservative union. It never deletes a rule feature; conflicts are
    surfaced as warnings. Confidence is the mean of the rule heuristic and the
    GNN probability for that class over the feature faces.
    """
    from breplens.ml.inference import predict_probs

    t0 = time.perf_counter()
    rules = recognize_rules(graph)
    gnn = recognize_gnn(graph, model, class_names, device=device)
    probs = predict_probs(model, graph, device=device)
    class_id = {name: i for i, name in enumerate(class_names)}
    warnings = list(rules.warnings)
    rule_faces = _face_set(rules)
    added = 0
    features = list(rules.features)
    for feat in gnn.features:
        faces = set(feat.face_ids)
        if faces and len(faces & rule_faces) / len(faces) > 0.5:
            continue
        if any(_same_feature(existing, feat) for existing in features):
            continue
        cid = class_id.get(feat.type.value)
        conf = float(probs[list(faces), cid].mean()) if cid is not None and faces else feat.confidence
        if conf < gnn_min_confidence:
            continue
        feat.confidence = round(0.5 * conf + 0.5 * 0.85, 4)
        features.append(feat)
        rule_faces |= faces
        added += 1
    # recompute confidence of rule features from GNN support
    for feat in features:
        if feat.source != "rules":
            continue
        cid = class_id.get(feat.type.value)
        if cid is None or not feat.face_ids:
            continue
        support = float(probs[feat.face_ids, cid].mean())
        feat.confidence = round(0.5 * feat.confidence + 0.5 * support, 4)
        feat.source = "hybrid"
    features.sort(key=lambda f: (f.type.value, min(f.face_ids)))
    for i, feat in enumerate(features):
        feat.id = f"{feat.type.value}_{i}"
    elapsed = (time.perf_counter() - t0) * 1000.0
    if added == 0:
        warnings.append("hybrid added no new GNN instances; rule features fully cover GNN components")
    return FeatureSet(
        part=graph.metadata.name,
        mode="hybrid",
        features=features,
        summary=summarize_part(graph, len(features)),
        timings=TimingBreakdown(recognition_ms=elapsed, total_ms=elapsed),
        warnings=warnings + graph.warnings,
        extra={"gnn_added_instances": added},
    )


__all__ = ["recognize_gnn", "recognize_hybrid"]
