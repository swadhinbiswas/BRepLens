"""Shared, dependency-light evaluation for rules, GNN and hybrid recognizers.

Every metric is computed from real predictions and ground truth supplied by the
caller; nothing is hard-coded. The same functions are used by the rules, GNN and
hybrid evaluation scripts so their numbers are directly comparable.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from breplens.recognition.params import line_distance, parallel
from breplens.schema import Feature, FeatureType, PartGraph, Vec3

HOLE_TYPES = {
    FeatureType.THROUGH_HOLE,
    FeatureType.BLIND_HOLE,
    FeatureType.COUNTERBORE,
    FeatureType.COUNTERSINK,
}


@dataclass
class MatchPair:
    gt_index: int
    pred_index: int
    distance: float


def _as_vec(value: Any) -> Vec3 | None:
    if value is None:
        return None
    if isinstance(value, Vec3):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return Vec3(x=float(value[0]), y=float(value[1]), z=float(value[2]))
    if isinstance(value, dict):
        return Vec3(x=float(value["x"]), y=float(value["y"]), z=float(value["z"]))
    return None


def _truth_field(truth: dict[str, Any], pred: Feature, key: str) -> Any:
    if key in truth:
        return truth[key]
    params = truth.get("params", {})
    return params.get(key)


def _compatible(pred: Feature, truth: dict[str, Any], tol: float = 4.0) -> bool:
    if pred.type.value != truth["type"]:
        return False
    pos_p = pred.position
    pos_t = _as_vec(truth.get("position"))
    axis_p = pred.axis
    axis_t = _as_vec(truth.get("axis"))
    if pred.type in HOLE_TYPES and pos_p is not None and pos_t is not None:
        if axis_t is not None and axis_p is not None:
            if not parallel(axis_p, axis_t, 0.99):
                return False
            if line_distance(pos_t, axis_t, pos_p) > 2.0:
                return False
            dt = abs((pos_p - pos_t).dot(axis_t.normalized()))
            if pred.type in (FeatureType.BLIND_HOLE, FeatureType.THROUGH_HOLE):
                # openings should be at the same end; allow 3 mm measurement noise
                if dt > 3.0:
                    return False
            return True
        if pos_p.distance_to(pos_t) > tol:
            return False
    if pos_p is not None and pos_t is not None:
        if pred.type in (FeatureType.FILLET, FeatureType.CHAMFER):
            if pos_p.distance_to(pos_t) > 6.0:
                return False
    return True


def match_features(
    preds: Sequence[Feature], truth_features: Sequence[dict[str, Any]]
) -> list[MatchPair]:
    """Greedy one-to-one matching by type and measured position/axis."""
    candidates: list[tuple[float, int, int]] = []
    for pi, pred in enumerate(preds):
        for gi, truth in enumerate(truth_features):
            if not _compatible(pred, truth):
                continue
            pos_p = pred.position
            pos_t = _as_vec(truth.get("position"))
            if pos_p is not None and pos_t is not None:
                if pred.type in HOLE_TYPES and pred.axis is not None:
                    d = 0.0
                else:
                    d = pos_p.distance_to(pos_t)
            else:
                d = 0.0
            candidates.append((d, pi, gi))
    candidates.sort()
    used_p: set[int] = set()
    used_g: set[int] = set()
    out: list[MatchPair] = []
    for d, pi, gi in candidates:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        out.append(MatchPair(gt_index=gi, pred_index=pi, distance=d))
    return out


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def per_class_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], class_names: Sequence[str]
) -> dict[str, Any]:
    n = len(class_names)
    tp = [0] * n
    fp = [0] * n
    fn = [0] * n
    support = [0] * n
    for t, p in zip(y_true, y_pred, strict=True):
        support[t] += 1
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1
    per: dict[str, Any] = {}
    f1s = []
    for i, name in enumerate(class_names):
        prec = _safe_div(tp[i], tp[i] + fp[i])
        rec = _safe_div(tp[i], tp[i] + fn[i])
        f1 = _safe_div(2 * prec * rec, prec + rec)
        per[name] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": support[i],
            "tp": tp[i],
            "fp": fp[i],
            "fn": fn[i],
        }
        if support[i] > 0:
            f1s.append(f1)
    acc = _safe_div(sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == p), len(y_true))
    confusion = [[0] * n for _ in range(n)]
    for t, p in zip(y_true, y_pred, strict=True):
        confusion[t][p] += 1
    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(float(np.mean(f1s)) if f1s else 0.0, 4),
        "per_class": per,
        "confusion_matrix": confusion,
        "class_names": list(class_names),
        "n_samples": len(y_true),
    }


def face_metrics_from_ids(
    gt_ids: Iterable[int | None],
    pred_ids: Iterable[int],
    class_names: Sequence[str],
) -> dict[str, Any]:
    gt = [int(x) if x is not None else 0 for x in gt_ids]
    pred = [int(x) for x in pred_ids]
    if len(gt) != len(pred):
        raise ValueError("gt and pred face label arrays must have equal length")
    return per_class_metrics(gt, pred, class_names)


def predicted_face_labels(
    graph: PartGraph,
    feature_set: Any,
    class_ids: dict[str, int],
    *,
    fallback: Sequence[int] | None = None,
) -> list[int]:
    """Map feature face membership to per-face class ids.

    Unclaimed faces are labelled ``base`` (0) unless *fallback* provides an
    alternative (used by hybrid mode to fill gaps from the GNN).
    """
    out = list(fallback) if fallback is not None else [class_ids.get("base", 0)] * len(graph.faces)
    for feat in feature_set.features:
        cid = class_ids.get(feat.type.value)
        if cid is None:
            continue
        for fid in feat.face_ids:
            if 0 <= fid < len(out):
                out[fid] = cid
    return out


def parameter_errors(
    matched: Sequence[tuple[Feature, dict[str, Any]]],
) -> dict[str, dict[str, float]]:
    """Absolute parameter errors for matched features, grouped by feature type."""
    errors: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for pred, truth in matched:
        p_params = pred.params.model_dump(exclude_none=True)
        t_params = truth.get("params", {})
        for key, pval in p_params.items():
            if key == "extra" or not isinstance(pval, (int, float)):
                continue
            tval = t_params.get(key)
            if tval is None or not isinstance(tval, (int, float)):
                continue
            if key in ("length", "width"):
                # pocket/step orientation is ambiguous; compare sorted dims below
                continue
            errors[pred.type.value][key].append(abs(float(pval) - float(tval)))
        # sorted footprint dimensions for recesses
        if pred.type.value in {"pocket", "slot", "step"}:
            p_dims = sorted([pred.params.length, pred.params.width])
            t_dims = sorted([t_params.get("length"), t_params.get("width")])
            if all(v is not None for v in [*p_dims, *t_dims]):
                for i, (pv, tv) in enumerate(zip(p_dims, t_dims, strict=True)):
                    errors[pred.type.value][f"footprint_dim_{i}"].append(
                        abs(float(pv) - float(tv))
                    )
    out: dict[str, dict[str, float]] = {}
    for ftype, keys in errors.items():
        out[ftype] = {}
        for key, vals in keys.items():
            arr = np.asarray(vals, dtype=float)
            out[ftype][key] = {
                "mean_abs": round(float(arr.mean()), 4),
                "median_abs": round(float(np.median(arr)), 4),
                "p95_abs": round(float(np.percentile(arr, 95)), 4),
                "max_abs": round(float(arr.max()), 4),
                "n": int(arr.size),
            }
    return out


def evaluate_feature_sets(
    records: Sequence[dict[str, Any]],
    *,
    class_names: Sequence[str],
    class_ids: dict[str, int],
) -> dict[str, Any]:
    """Evaluate records of ``{name, split, graph, truth, pred}``.

    ``pred`` is a FeatureSet. Returns feature-level P/R/F1, per-class metrics,
    face-level metrics and parameter errors, all from real predictions.
    """
    y_true_faces: list[int] = []
    y_pred_faces: list[int] = []
    per_type_tp: Counter[str] = Counter()
    per_type_fp: Counter[str] = Counter()
    per_type_fn: Counter[str] = Counter()
    gt_counts: Counter[str] = Counter()
    pred_counts: Counter[str] = Counter()
    matched_pairs: list[tuple[Feature, dict[str, Any]]] = []
    exact_parts = 0
    total_parts = 0
    infer_ms: list[float] = []
    feature_confusions: Counter[str] = Counter()

    for rec in records:
        graph: PartGraph = rec["graph"]
        truth = rec["truth"]
        fs = rec["pred"]
        total_parts += 1
        if len(fs.features) == len(truth["features"]):
            # count as structurally exact only if every GT is matched one-to-one
            pairs = match_features(fs.features, truth["features"])
            if len(pairs) == len(truth["features"]):
                exact_parts += 1
        pairs = match_features(fs.features, truth["features"])
        matched_gt = {p.gt_index for p in pairs}
        matched_pred = {p.pred_index for p in pairs}
        for p in pairs:
            pred = fs.features[p.pred_index]
            gt = truth["features"][p.gt_index]
            per_type_tp[pred.type.value] += 1
            matched_pairs.append((pred, gt))
        for gi, gt in enumerate(truth["features"]):
            gt_counts[gt["type"]] += 1
            if gi not in matched_gt:
                per_type_fn[gt["type"]] += 1
        for pi, pred in enumerate(fs.features):
            pred_counts[pred.type.value] += 1
            if pi not in matched_pred:
                per_type_fp[pred.type.value] += 1
        # closest predicted type for GT misses (for failure analysis)
        for gi, gt in enumerate(truth["features"]):
            if gi in matched_gt:
                continue
            best: tuple[float, str] | None = None
            gpos = _as_vec(gt.get("position"))
            for pred in fs.features:
                if gpos is None or pred.position is None:
                    continue
                d = pred.position.distance_to(gpos)
                if best is None or d < best[0]:
                    best = (d, pred.type.value)
            if best is not None:
                feature_confusions[f"{gt['type']} -> {best[1]}"] += 1

        gt_ids = [f.label_id if f.label_id is not None else 0 for f in graph.faces]
        pred_ids = predicted_face_labels(graph, fs, class_ids)
        y_true_faces.extend(gt_ids)
        y_pred_faces.extend(pred_ids)
        infer_ms.append(fs.timings.recognition_ms)

    class_f1 = []
    per_class: dict[str, Any] = {}
    for name in class_names:
        tp = per_type_tp[name]
        fp = per_type_fp[name]
        fn = per_type_fn[name]
        prec = _safe_div(tp, tp + fp)
        rec = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * prec * rec, prec + rec)
        per_class[name] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "gt_count": gt_counts[name],
            "pred_count": pred_counts[name],
        }
        if gt_counts[name] > 0 or pred_counts[name] > 0:
            class_f1.append(f1)
    total_tp = sum(per_type_tp.values())
    total_fp = sum(per_type_fp.values())
    total_fn = sum(per_type_fn.values())
    prec = _safe_div(total_tp, total_tp + total_fp)
    rec = _safe_div(total_tp, total_tp + total_fn)
    face_metrics = per_class_metrics(y_true_faces, y_pred_faces, class_names)
    return {
        "n_parts": total_parts,
        "n_gt_features": sum(gt_counts.values()),
        "n_pred_features": sum(pred_counts.values()),
        "feature": {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(_safe_div(2 * prec * rec, prec + rec), 4),
            "macro_f1": round(float(np.mean(class_f1)) if class_f1 else 0.0, 4),
            "exact_part_rate": round(_safe_div(exact_parts, total_parts), 4),
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "per_class": per_class,
            "confusions": dict(feature_confusions.most_common(15)),
        },
        "face": face_metrics,
        "parameter_errors": parameter_errors(matched_pairs),
        "latency_ms": {
            "mean": round(float(np.mean(infer_ms)), 3) if infer_ms else 0.0,
            "median": round(float(np.median(infer_ms)), 3) if infer_ms else 0.0,
            "p95": round(float(np.percentile(infer_ms, 95)), 3) if infer_ms else 0.0,
        },
    }


def dataset_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    faces = [len(r["graph"].faces) for r in records]
    feats = [len(r["truth"]["features"]) for r in records]
    fams = Counter(r.get("family", "") for r in records)
    return {
        "n_parts": len(records),
        "families": dict(sorted(fams.items())),
        "faces_total": int(sum(faces)),
        "faces_mean": round(float(np.mean(faces)), 2) if faces else 0.0,
        "features_total": int(sum(feats)),
        "classes": sorted({f["type"] for r in records for f in r["truth"]["features"]}),
    }


__all__ = [
    "HOLE_TYPES",
    "MatchPair",
    "dataset_summary",
    "evaluate_feature_sets",
    "face_metrics_from_ids",
    "match_features",
    "parameter_errors",
    "per_class_metrics",
    "predicted_face_labels",
]
