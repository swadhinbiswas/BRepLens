"""Synthetic machined-part generator with construction-derived ground truth.

CadQuery is imported lazily inside builder functions so the hosted inference path
never pays for the (VTK-heavy) CadQuery import. Each part family is built from
documented analytic parameters; ground-truth face labels are derived by matching
*expected construction geometry* to the exported B-Rep, never from the feature
recognizer. This makes the dataset an independent test bed (and the documented
fallback dataset if MFCAD++ is unavailable).
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from breplens.schema import FeatureType, PartGraph, SurfaceType, Vec3

SYNTHETIC_CLASSES: tuple[str, ...] = (
    "base",
    "through_hole",
    "blind_hole",
    "counterbore",
    "countersink",
    "pocket",
    "slot",
    "step",
    "boss",
    "fillet",
    "chamfer",
)
CLASS_TO_ID: dict[str, int] = {name: i for i, name in enumerate(SYNTHETIC_CLASSES)}

FAMILIES: tuple[str, ...] = (
    "plate_holes",
    "plate_blind",
    "counterbore",
    "countersink",
    "pocket",
    "slot",
    "stepped_block",
    "boss_plate",
    "fillet_block",
    "chamfer_block",
    "bracket_mixed",
)


@dataclass
class GTFeature:
    """Ground-truth feature with construction parameters."""

    gid: str
    type: FeatureType
    params: dict[str, Any]
    position: tuple[float, float, float] | None = None
    axis: tuple[float, float, float] | None = None
    face_ids: list[int] = field(default_factory=list)


@dataclass
class FaceHint:
    """Construction geometry used to attribute one or more faces."""

    gid: str
    label: FeatureType
    surface: str
    radius: float | None = None
    axis_point: tuple[float, float, float] | None = None
    axis_dir: tuple[float, float, float] | None = None
    semi_angle_deg: float | None = None
    plane_normal: tuple[float, float, float] | None = None
    plane_point: tuple[float, float, float] | None = None
    tol: float = 0.35
    grow_concave_from_floor: bool = False
    opening_z: float | None = None


@dataclass
class PartSpec:
    """Everything needed to build, label and describe one synthetic part."""

    name: str
    family: str
    builder: str
    params: dict[str, Any]
    features: list[GTFeature]
    hints: list[FaceHint]
    seed: int = 0


class _ShapeLike(Protocol):
    def val(self) -> Any: ...


def _cq() -> Any:
    import cadquery as cq

    return cq


def _base_box(cq: Any, length: float, width: float, height: float) -> Any:
    return cq.Workplane("XY").box(length, width, height, centered=(True, True, False))


def _plate_holes(params: dict[str, Any]) -> Any:
    cq = _cq()
    l, w, h = params["length"], params["width"], params["height"]
    d = params["hole_d"]
    positions = params["hole_positions"]
    part = _base_box(cq, l, w, h).faces(">Z").workplane()
    part = part.pushPoints([(x, y) for x, y in positions]).hole(d)
    return part


def _plate_blind(params: dict[str, Any]) -> Any:
    cq = _cq()
    l, w, h = params["length"], params["width"], params["height"]
    d, depth = params["hole_d"], params["depth"]
    cx, cy = params["hole_xy"]
    return (
        _base_box(cq, l, w, h).faces(">Z").workplane().center(cx, cy).circle(d / 2).cutBlind(-depth)
    )


def _counterbore(params: dict[str, Any]) -> Any:
    cq = _cq()
    l, w, h = params["length"], params["width"], params["height"]
    d, cd, cdepth = params["hole_d"], params["cbore_d"], params["cbore_depth"]
    cx, cy = params["hole_xy"]
    return _base_box(cq, l, w, h).faces(">Z").workplane().center(cx, cy).cboreHole(d, cd, cdepth)


def _countersink(params: dict[str, Any]) -> Any:
    cq = _cq()
    l, w, h = params["length"], params["width"], params["height"]
    d, cd, angle = params["hole_d"], params["csk_d"], params["csk_angle"]
    cx, cy = params["hole_xy"]
    return _base_box(cq, l, w, h).faces(">Z").workplane().center(cx, cy).cskHole(d, cd, angle)


def _pocket(params: dict[str, Any]) -> Any:
    cq = _cq()
    l, w, h = params["length"], params["width"], params["height"]
    pl, pw, depth = params["pocket_l"], params["pocket_w"], params["depth"]
    cx, cy = params["pocket_xy"]
    return (
        _base_box(cq, l, w, h).faces(">Z").workplane().center(cx, cy).rect(pl, pw).cutBlind(-depth)
    )


def _slot(params: dict[str, Any]) -> Any:
    cq = _cq()
    l, w, h = params["length"], params["width"], params["height"]
    sl, sw, depth = params["slot_l"], params["slot_w"], params["depth"]
    cx, cy = params["slot_xy"]
    return (
        _base_box(cq, l, w, h)
        .faces(">Z")
        .workplane()
        .center(cx, cy)
        .slot2D(sl, sw)
        .cutBlind(-depth)
    )


def _stepped_block(params: dict[str, Any]) -> Any:
    cq = _cq()
    l, w, h = params["length"], params["width"], params["height"]
    step_h = params["step_height"]
    cut = (
        cq.Workplane("XY")
        .box(l / 2, w, step_h, centered=(True, True, False))
        .translate((-l / 4, 0, h - step_h))
    )
    return _base_box(cq, l, w, h).cut(cut)


def _boss_plate(params: dict[str, Any]) -> Any:
    cq = _cq()
    l, w, h = params["length"], params["width"], params["height"]
    d, bh = params["boss_d"], params["boss_h"]
    cx, cy = params["boss_xy"]
    boss = cq.Workplane("XY").workplane(offset=h).center(cx, cy).circle(d / 2).extrude(bh)
    return _base_box(cq, l, w, h).union(boss)


def _fillet_block(params: dict[str, Any]) -> Any:
    cq = _cq()
    l, w, h = params["length"], params["width"], params["height"]
    r = params["fillet_r"]
    return _base_box(cq, l, w, h).edges("|Z").fillet(r)


def _chamfer_block(params: dict[str, Any]) -> Any:
    cq = _cq()
    l, w, h = params["length"], params["width"], params["height"]
    c = params["chamfer_c"]
    return _base_box(cq, l, w, h).edges("|Z").chamfer(c)


def _bracket_mixed(params: dict[str, Any]) -> Any:
    cq = _cq()
    l, w, h = params["length"], params["width"], params["height"]
    pl, pw, depth = params["pocket_l"], params["pocket_w"], params["depth"]
    cx, cy = params["pocket_xy"]
    part = _base_box(cq, l, w, h)
    # Outer chamfers first: chamfering after pocketing would also chamfer the
    # pocket's internal vertical edges and change the intended feature count.
    part = part.edges("|Z").chamfer(params["chamfer_c"])
    by_diameter: dict[float, list[tuple[float, float]]] = {}
    for x, y, d in params["holes"]:
        by_diameter.setdefault(float(d), []).append((float(x), float(y)))
    top = part.faces(">Z").workplane()
    for d, pts in by_diameter.items():
        top = top.pushPoints(pts).hole(d)
    part = top.faces(">Z").workplane().center(cx, cy).rect(pl, pw).cutBlind(-depth)
    return part


_BUILDERS: dict[str, Callable[[dict[str, Any]], Any]] = {
    "plate_holes": _plate_holes,
    "plate_blind": _plate_blind,
    "counterbore": _counterbore,
    "countersink": _countersink,
    "pocket": _pocket,
    "slot": _slot,
    "stepped_block": _stepped_block,
    "boss_plate": _boss_plate,
    "fillet_block": _fillet_block,
    "chamfer_block": _chamfer_block,
    "bracket_mixed": _bracket_mixed,
}


def _hint_cylinder(
    gid: str,
    label: FeatureType,
    radius: float,
    axis_point: tuple[float, float, float],
    axis_dir: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> FaceHint:
    return FaceHint(
        gid=gid,
        label=label,
        surface=SurfaceType.CYLINDER.value,
        radius=radius,
        axis_point=axis_point,
        axis_dir=axis_dir,
    )


def _hint_plane(
    gid: str,
    label: FeatureType,
    normal: tuple[float, float, float],
    point: tuple[float, float, float],
    *,
    grow: bool = False,
    opening_z: float | None = None,
    tol: float = 0.35,
) -> FaceHint:
    return FaceHint(
        gid=gid,
        label=label,
        surface=SurfaceType.PLANE.value,
        plane_normal=normal,
        plane_point=point,
        grow_concave_from_floor=grow,
        opening_z=opening_z,
        tol=tol,
    )


def _default_params(family: str, rng: random.Random) -> dict[str, Any]:
    """Build one parameter set with deterministic jitter."""
    j = lambda base, frac=0.15: round(base * (1.0 + rng.uniform(-frac, frac)), 2)  # noqa: E731
    if family == "plate_holes":
        l, w, h = j(60), j(40), j(10)
        d = j(6, 0.2)
        xs = [-l / 2 + 18, l / 2 - 18]
        ys = [-w / 2 + 13, w / 2 - 13]
        return {
            "length": l,
            "width": w,
            "height": h,
            "hole_d": d,
            "hole_positions": [(x, y) for x in xs for y in ys],
        }
    if family == "plate_blind":
        l, w, h = j(60), j(40), j(14)
        return {
            "length": l,
            "width": w,
            "height": h,
            "hole_d": j(10, 0.2),
            "depth": round(h * 0.55, 2),
            "hole_xy": (0.0, 0.0),
        }
    if family == "counterbore":
        l, w, h = j(60), j(40), j(14)
        return {
            "length": l,
            "width": w,
            "height": h,
            "hole_d": j(6, 0.15),
            "cbore_d": j(13, 0.15),
            "cbore_depth": j(5, 0.2),
            "hole_xy": (0.0, 0.0),
        }
    if family == "countersink":
        l, w, h = j(60), j(40), j(14)
        return {
            "length": l,
            "width": w,
            "height": h,
            "hole_d": j(6, 0.15),
            "csk_d": j(13, 0.15),
            "csk_angle": 90.0,
            "hole_xy": (0.0, 0.0),
        }
    if family == "pocket":
        l, w, h = j(70), j(45), j(16)
        return {
            "length": l,
            "width": w,
            "height": h,
            "pocket_l": j(30, 0.25),
            "pocket_w": j(18, 0.25),
            "depth": j(6, 0.25),
            "pocket_xy": (0.0, 0.0),
        }
    if family == "slot":
        l, w, h = j(70), j(45), j(16)
        return {
            "length": l,
            "width": w,
            "height": h,
            "slot_l": j(38, 0.2),
            "slot_w": j(11, 0.2),
            "depth": j(5, 0.2),
            "slot_xy": (0.0, 0.0),
        }
    if family == "stepped_block":
        l, w, h = j(60), j(40), j(20)
        return {
            "length": l,
            "width": w,
            "height": h,
            "step_height": j(h * 0.45, 0.2),
        }
    if family == "boss_plate":
        l, w, h = j(60), j(40), j(10)
        return {
            "length": l,
            "width": w,
            "height": h,
            "boss_d": j(16, 0.2),
            "boss_h": j(8, 0.2),
            "boss_xy": (0.0, 0.0),
        }
    if family == "fillet_block":
        l, w, h = j(60), j(40), j(20)
        return {"length": l, "width": w, "height": h, "fillet_r": j(6, 0.2)}
    if family == "chamfer_block":
        l, w, h = j(60), j(40), j(20)
        return {"length": l, "width": w, "height": h, "chamfer_c": j(3, 0.2)}
    if family == "bracket_mixed":
        l, w, h = 80.0, 50.0, 15.0
        return {
            "length": l,
            "width": w,
            "height": h,
            "pocket_l": 32.0,
            "pocket_w": 18.0,
            "depth": 5.0,
            "pocket_xy": (-10.0, 0.0),
            "hole_d": 6.0,
            "holes": [(-10.0, 0.0, 8.0), (22.0, 12.0, 6.0), (22.0, -12.0, 6.0)],
            "chamfer_c": 2.0,
        }
    raise ValueError(f"unknown family {family}")


def make_spec(family: str, rng: random.Random, index: int) -> PartSpec:
    """Create one fully-specified part (geometry, ground truth, face hints)."""
    p = _default_params(family, rng)
    hints: list[FaceHint] = []
    feats: list[GTFeature] = []
    name = f"{family}_{index:05d}"

    if family == "plate_holes":
        h, d = p["height"], p["hole_d"]
        for i, (x, y) in enumerate(p["hole_positions"]):
            gid = f"through_hole_{i}"
            feats.append(
                GTFeature(
                    gid,
                    FeatureType.THROUGH_HOLE,
                    {"diameter": d, "depth": h, "is_through": True},
                    position=(x, y, h),
                    axis=(0.0, 0.0, -1.0),
                )
            )
            hints.append(_hint_cylinder(gid, FeatureType.THROUGH_HOLE, d / 2, (x, y, 0.0)))
    elif family == "plate_blind":
        h, d, depth = p["height"], p["hole_d"], p["depth"]
        cx, cy = p["hole_xy"]
        gid = "blind_hole_0"
        feats.append(
            GTFeature(
                gid,
                FeatureType.BLIND_HOLE,
                {"diameter": d, "depth": depth, "is_through": False},
                position=(cx, cy, h),
                axis=(0.0, 0.0, -1.0),
            )
        )
        hints.append(_hint_cylinder(gid, FeatureType.BLIND_HOLE, d / 2, (cx, cy, 0.0)))
        hints.append(
            _hint_plane(
                gid,
                FeatureType.BLIND_HOLE,
                (0.0, 0.0, 1.0),
                (cx, cy, h - depth),
            )
        )
    elif family == "counterbore":
        h = p["height"]
        cx, cy = p["hole_xy"]
        gid = "counterbore_0"
        feats.append(
            GTFeature(
                gid,
                FeatureType.COUNTERBORE,
                {
                    "outer_diameter": p["cbore_d"],
                    "inner_diameter": p["hole_d"],
                    "depth": p["cbore_depth"],
                    "is_through": True,
                },
                position=(cx, cy, h),
                axis=(0.0, 0.0, -1.0),
            )
        )
        hints.append(_hint_cylinder(gid, FeatureType.COUNTERBORE, p["cbore_d"] / 2, (cx, cy, 0.0)))
        hints.append(_hint_cylinder(gid, FeatureType.COUNTERBORE, p["hole_d"] / 2, (cx, cy, 0.0)))
        hints.append(
            _hint_plane(
                gid,
                FeatureType.COUNTERBORE,
                (0.0, 0.0, 1.0),
                (cx, cy, h - p["cbore_depth"]),
            )
        )
    elif family == "countersink":
        h = p["height"]
        cx, cy = p["hole_xy"]
        gid = "countersink_0"
        feats.append(
            GTFeature(
                gid,
                FeatureType.COUNTERSINK,
                {
                    "outer_diameter": p["csk_d"],
                    "inner_diameter": p["hole_d"],
                    "angle_deg": p["csk_angle"],
                    "is_through": True,
                },
                position=(cx, cy, h),
                axis=(0.0, 0.0, -1.0),
            )
        )
        hints.append(
            FaceHint(
                gid=gid,
                label=FeatureType.COUNTERSINK,
                surface=SurfaceType.CONE.value,
                axis_point=(cx, cy, 0.0),
                axis_dir=(0.0, 0.0, 1.0),
                semi_angle_deg=p["csk_angle"] / 2,
                tol=0.8,
            )
        )
        hints.append(_hint_cylinder(gid, FeatureType.COUNTERSINK, p["hole_d"] / 2, (cx, cy, 0.0)))
    elif family in ("pocket", "slot"):
        is_slot = family == "slot"
        h = p["height"]
        cx, cy = p["pocket_xy"] if not is_slot else p["slot_xy"]
        length = p["slot_l"] if is_slot else p["pocket_l"]
        width = p["slot_w"] if is_slot else p["pocket_w"]
        ftype = FeatureType.SLOT if is_slot else FeatureType.POCKET
        gid = f"{ftype.value}_0"
        feats.append(
            GTFeature(
                gid,
                ftype,
                {"length": length, "width": width, "depth": p["depth"]},
                position=(cx, cy, h),
                axis=(0.0, 0.0, 1.0),
            )
        )
        hints.append(
            _hint_plane(
                gid,
                ftype,
                (0.0, 0.0, 1.0),
                (cx, cy, h - p["depth"]),
                grow=True,
                opening_z=h,
            )
        )
    elif family == "stepped_block":
        h = p["height"]
        sh = p["step_height"]
        gid = "step_0"
        feats.append(
            GTFeature(
                gid,
                FeatureType.STEP,
                {"length": p["length"] / 2, "width": p["width"], "height": sh},
                position=(-p["length"] / 4, 0.0, h),
                axis=(0.0, 0.0, 1.0),
            )
        )
        hints.append(
            _hint_plane(
                gid,
                FeatureType.STEP,
                (0.0, 0.0, 1.0),
                (p["length"] / 4, 0.0, h - sh),
                grow=True,
                opening_z=h,
            )
        )
    elif family == "boss_plate":
        h = p["height"]
        cx, cy = p["boss_xy"]
        gid = "boss_0"
        feats.append(
            GTFeature(
                gid,
                FeatureType.BOSS,
                {"diameter": p["boss_d"], "height": p["boss_h"]},
                position=(cx, cy, h),
                axis=(0.0, 0.0, 1.0),
            )
        )
        hints.append(_hint_cylinder(gid, FeatureType.BOSS, p["boss_d"] / 2, (cx, cy, 0.0)))
        hints.append(
            _hint_plane(
                gid,
                FeatureType.BOSS,
                (0.0, 0.0, 1.0),
                (cx, cy, h + p["boss_h"]),
            )
        )
    elif family == "fillet_block":
        r = p["fillet_r"]
        for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
            cx = sx * (p["length"] / 2 - r)
            cy = sy * (p["width"] / 2 - r)
            gid = f"fillet_{i}"
            feats.append(
                GTFeature(
                    gid,
                    FeatureType.FILLET,
                    {"radius": r, "length": p["height"]},
                    position=(cx, cy, p["height"] / 2),
                    axis=(0.0, 0.0, 1.0),
                )
            )
            hints.append(_hint_cylinder(gid, FeatureType.FILLET, r, (cx, cy, 0.0)))
    elif family == "chamfer_block":
        c = p["chamfer_c"]
        for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
            nx, ny = sx / math.sqrt(2), sy / math.sqrt(2)
            x = sx * p["length"] / 2 - sx * c / 2
            y = sy * p["width"] / 2 - sy * c / 2
            gid = f"chamfer_{i}"
            feats.append(
                GTFeature(
                    gid,
                    FeatureType.CHAMFER,
                    {"chamfer_width": c, "angle_deg": 45.0},
                    position=(sx * p["length"] / 2, sy * p["width"] / 2, p["height"] / 2),
                )
            )
            hints.append(
                _hint_plane(
                    gid,
                    FeatureType.CHAMFER,
                    (nx, ny, 0.0),
                    (x, y, p["height"] / 2),
                    tol=0.6,
                )
            )
    elif family == "bracket_mixed":
        h = p["height"]
        cx, cy = p["pocket_xy"]
        gid = "pocket_0"
        feats.append(
            GTFeature(
                gid,
                FeatureType.POCKET,
                {"length": p["pocket_l"], "width": p["pocket_w"], "depth": p["depth"]},
                position=(cx, cy, h),
                axis=(0.0, 0.0, 1.0),
            )
        )
        hints.append(
            _hint_plane(
                gid,
                FeatureType.POCKET,
                (0.0, 0.0, 1.0),
                (cx, cy, h - p["depth"]),
                grow=True,
                opening_z=h,
            )
        )
        for i, (x, y, d) in enumerate(p["holes"]):
            hgid = f"through_hole_{i}"
            inside_pocket = abs(x - cx) <= p["pocket_l"] / 2 and abs(y - cy) <= p["pocket_w"] / 2
            entry_z = h - p["depth"] if inside_pocket else h
            feats.append(
                GTFeature(
                    hgid,
                    FeatureType.THROUGH_HOLE,
                    {"diameter": d, "depth": entry_z, "is_through": True},
                    position=(x, y, entry_z),
                    axis=(0.0, 0.0, -1.0),
                )
            )
            hints.append(_hint_cylinder(hgid, FeatureType.THROUGH_HOLE, d / 2, (x, y, 0.0)))
        c = p["chamfer_c"]
        for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
            nx, ny = sx / math.sqrt(2), sy / math.sqrt(2)
            x = sx * p["length"] / 2 - sx * c / 2
            y = sy * p["width"] / 2 - sy * c / 2
            cgid = f"chamfer_{i}"
            feats.append(
                GTFeature(
                    cgid,
                    FeatureType.CHAMFER,
                    {"chamfer_width": c, "angle_deg": 45.0},
                    position=(sx * p["length"] / 2, sy * p["width"] / 2, h / 2),
                )
            )
            hints.append(
                _hint_plane(cgid, FeatureType.CHAMFER, (nx, ny, 0.0), (x, y, h / 2), tol=0.6)
            )
    else:  # pragma: no cover
        raise ValueError(family)

    return PartSpec(name=name, family=family, builder=family, params=p, features=feats, hints=hints)


def _line_distance(
    point: tuple[float, float, float],
    direction: tuple[float, float, float],
    other_point: tuple[float, float, float],
) -> float:
    p = Vec3.of(point)
    d = Vec3.of(direction).normalized()
    w = Vec3.of(other_point) - p
    return (w - d * w.dot(d)).norm()


def _vec_close(a: tuple[float, float, float], b: tuple[float, float, float], tol: float) -> bool:
    return Vec3.of(a).distance_to(Vec3.of(b)) <= tol


def assign_labels(graph: PartGraph, spec: PartSpec) -> PartGraph:
    """Attach construction-derived face labels and feature face ids."""
    by_gid: dict[str, list[int]] = {f.gid: [] for f in spec.features}
    for face in graph.faces:
        face.label = "base"
        face.label_id = CLASS_TO_ID["base"]

    for hint in spec.hints:
        matched: list[int] = []
        for face in graph.faces:
            ok = False
            if hint.surface == "cylinder" and face.surface_type == SurfaceType.CYLINDER:
                if hint.radius is None or abs((face.radius or 0.0) - hint.radius) <= hint.tol:
                    if hint.axis_point is not None and face.axis_direction is not None:
                        origin = face.axis_origin or Vec3()
                        if _line_distance(
                            hint.axis_point,
                            hint.axis_dir or (0.0, 0.0, 1.0),
                            origin.as_tuple(),
                        ) <= max(hint.tol, 0.5):
                            par = abs(
                                face.axis_direction.dot(Vec3.of(hint.axis_dir or (0.0, 0.0, 1.0)))
                            )
                            ok = par > 0.99
                    else:
                        ok = True
            elif hint.surface == "cone" and face.surface_type == SurfaceType.CONE:
                if hint.axis_point is not None and face.axis_direction is not None:
                    ok = (
                        _line_distance(
                            hint.axis_point,
                            hint.axis_dir or (0.0, 0.0, 1.0),
                            (face.axis_origin or Vec3()).as_tuple(),
                        )
                        <= max(hint.tol, 0.8)
                        and abs(face.axis_direction.dot(Vec3.of(hint.axis_dir or (0.0, 0.0, 1.0))))
                        > 0.99
                    )
                    if ok and hint.semi_angle_deg is not None and face.semi_angle_deg is not None:
                        ok = abs(abs(face.semi_angle_deg) - hint.semi_angle_deg) <= 3.0
            elif hint.surface == "plane" and face.surface_type == SurfaceType.PLANE:
                if hint.plane_normal is not None and face.normal is not None:
                    n = Vec3.of(hint.plane_normal).normalized()
                    ok = face.normal.dot(n) > 0.99
                    if ok and hint.plane_point is not None:
                        p = Vec3.of(hint.plane_point)
                        ok = abs((face.centroid - p).dot(n)) <= hint.tol
            elif hint.surface == "torus" and face.surface_type == SurfaceType.TORUS:
                ok = (
                    hint.radius is None
                    or abs((face.secondary_radius or 0.0) - hint.radius) <= hint.tol
                )
            if ok:
                matched.append(face.id)
                face.label = hint.label.value
                face.label_id = CLASS_TO_ID[hint.label.value]

        if hint.grow_concave_from_floor and matched:
            # Pocket/slot/step walls: flood through concave edges from the floor,
            # bounded by the opening plane so outer faces are not absorbed.
            from collections import deque

            seen = set(matched)
            q = deque(matched)
            zmax = hint.opening_z if hint.opening_z is not None else float("inf")
            while q:
                fid = q.popleft()
                for e in graph.edges:
                    if e.convexity.value != "concave":
                        continue
                    nb = None
                    if e.face_a == fid:
                        nb = e.face_b
                    elif e.face_b == fid:
                        nb = e.face_a
                    if nb is None or nb in seen:
                        continue
                    f = graph.faces[nb]
                    if f.centroid.z > zmax + 0.5:
                        continue
                    if (
                        f.surface_type == SurfaceType.PLANE
                        and f.normal is not None
                        and f.normal.z > 0.99
                        and abs(f.centroid.z - zmax) < 0.5
                    ):
                        continue
                    seen.add(nb)
                    q.append(nb)
                    face = graph.faces[nb]
                    face.label = hint.label.value
                    face.label_id = CLASS_TO_ID[hint.label.value]
                    matched.append(nb)
        by_gid.setdefault(hint.gid, []).extend(matched)

    for feature in spec.features:
        feature.face_ids = sorted(set(by_gid.get(feature.gid, [])))
        if not feature.face_ids:
            graph.warnings.append(f"ground-truth feature {feature.gid} matched no faces")
    return graph


def build_part(spec: PartSpec, out_dir: str | Path, *, adapter: Any = None) -> dict[str, Any]:
    """Build STEP, load neutral graph, label it and persist both artifacts."""
    cq = _cq()
    out = Path(out_dir)
    (out / "step").mkdir(parents=True, exist_ok=True)
    (out / "graphs").mkdir(parents=True, exist_ok=True)
    (out / "truth").mkdir(parents=True, exist_ok=True)
    shape = _BUILDERS[spec.builder](spec.params)
    solid = shape.val()
    if not solid.isValid():
        raise RuntimeError(f"generated invalid solid for {spec.name}")
    step_path = out / "step" / f"{spec.name}.step"
    cq.exporters.export(shape, str(step_path))

    if adapter is None:
        from breplens.adapters.occ import OccAdapter

        adapter = OccAdapter()
    graph = adapter.load_graph(step_path, name=spec.name)
    graph.metadata.generator = "breplens.synthetic"
    graph.metadata.family = spec.family
    graph = assign_labels(graph, spec)
    graph.metadata.n_features = len(spec.features)

    truth = {
        "name": spec.name,
        "family": spec.family,
        "seed": spec.seed,
        "params": spec.params,
        "features": [asdict(f) for f in spec.features],
        "classes": list(SYNTHETIC_CLASSES),
        "step": str(step_path.relative_to(out)),
        "n_faces": len(graph.faces),
        "n_edges": len(graph.edges),
        "warnings": graph.warnings,
    }
    (out / "graphs" / f"{spec.name}.json").write_text(graph.model_dump_json(indent=1))
    (out / "truth" / f"{spec.name}.json").write_text(json.dumps(truth, indent=1, default=str))
    return truth


def generate_samples(out_dir: str | Path) -> list[Path]:
    """Generate the deterministic 10-part sample set shipped with the demo."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(20261124)
    samples: list[Path] = []
    artifacts = out.parent / "data" / "sample_artifacts"
    sample_families = [f for f in FAMILIES if f != "chamfer_block"]
    for i, family in enumerate(sample_families):
        spec = make_spec(family, rng, i)
        spec.name = f"sample_{i:02d}_{family}"
        build_part(spec, artifacts)
        src = artifacts / "step" / f"{spec.name}.step"
        dst = out / src.name
        dst.write_bytes(src.read_bytes())
        samples.append(dst)
    return samples


def generate_dataset(
    out_dir: str | Path, *, per_family: int, seed: int = 20261124
) -> dict[str, Any]:
    """Generate a balanced dataset (all families) and return a manifest dict."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for fam_i, family in enumerate(FAMILIES):
        rng = random.Random(seed + fam_i * 1000)
        for j in range(per_family):
            spec = make_spec(family, rng, j)
            spec.seed = seed + fam_i * 1000 + j
            truth = build_part(spec, out)
            manifest.append(truth)
    manifest.sort(key=lambda d: d["name"])
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    return {
        "families": list(FAMILIES),
        "per_family": per_family,
        "n_parts": len(manifest),
        "seed": seed,
    }


__all__ = [
    "CLASS_TO_ID",
    "FAMILIES",
    "SYNTHETIC_CLASSES",
    "GTFeature",
    "PartSpec",
    "assign_labels",
    "build_part",
    "generate_dataset",
    "generate_samples",
    "make_spec",
]
