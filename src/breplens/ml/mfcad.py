"""MFCAD dataset label mapping and conversion helpers.

The MFCAD dataset (Cao et al., 2020; MIT-licensed repository) stores per-face
labels as a Python pickle (``.face_truth``) indexed by integer face names written
into each STEP file. Label 15 is ``stock`` (non-feature); 0-14 are machining
feature classes.

Reference: https://github.com/hducg/MFCAD
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from breplens.schema import FeatureType

MFCAD_CLASSES: tuple[str, ...] = (
    "rectangular_through_slot",
    "triangular_through_slot",
    "rectangular_passage",
    "triangular_passage",
    "6sides_passage",
    "rectangular_through_step",
    "2sides_through_step",
    "slanted_through_step",
    "rectangular_blind_step",
    "triangular_blind_step",
    "rectangular_blind_slot",
    "rectangular_pocket",
    "triangular_pocket",
    "6sides_pocket",
    "chamfer",
    "stock",
)
MFCAD_CLASS_TO_ID: dict[str, int] = {name: i for i, name in enumerate(MFCAD_CLASSES)}
STOCK_ID = MFCAD_CLASS_TO_ID["stock"]

# Best-effort mapping used only for face-level comparison of the BRepLens rule
# recognizer on MFCAD. Passage classes have no cylindrical equivalent, so they
# map to UNKNOWN and are reported as misses rather than being silently renamed.
MFCAD_TO_BREPLENS: dict[str, FeatureType] = {
    "rectangular_through_slot": FeatureType.SLOT,
    "triangular_through_slot": FeatureType.SLOT,
    "rectangular_passage": FeatureType.UNKNOWN,
    "triangular_passage": FeatureType.UNKNOWN,
    "6sides_passage": FeatureType.UNKNOWN,
    "rectangular_through_step": FeatureType.STEP,
    "2sides_through_step": FeatureType.STEP,
    "slanted_through_step": FeatureType.STEP,
    "rectangular_blind_step": FeatureType.STEP,
    "triangular_blind_step": FeatureType.STEP,
    "rectangular_blind_slot": FeatureType.SLOT,
    "rectangular_pocket": FeatureType.POCKET,
    "triangular_pocket": FeatureType.POCKET,
    "6sides_pocket": FeatureType.POCKET,
    "chamfer": FeatureType.CHAMFER,
    "stock": FeatureType.UNKNOWN,
}


def load_face_truth(step_path: str | Path) -> list[int]:
    """Load the per-face label ids stored next to a STEP file."""
    p = Path(step_path)
    label_path = p.with_suffix(".face_truth")
    obj = pickle.loads(label_path.read_bytes())
    if not isinstance(obj, list) or not all(isinstance(v, int) for v in obj):
        raise ValueError(f"unexpected face truth format in {label_path}")
    return obj


def family_from_labels(labels: list[int]) -> str:
    names = sorted({MFCAD_CLASSES[i] for i in labels if 0 <= i < STOCK_ID})
    return "+".join(names) if names else "stock_only"


def truth_record(step_path: str | Path, labels_by_face: list[int], split: str) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for lid in labels_by_face:
        counts[MFCAD_CLASSES[lid]] = counts.get(MFCAD_CLASSES[lid], 0) + 1
    return {
        "name": Path(step_path).stem,
        "dataset": "MFCAD",
        "split": split,
        "n_faces": len(labels_by_face),
        "labels": labels_by_face,
        "class_names": list(MFCAD_CLASSES),
        "label_counts": counts,
        "family": family_from_labels(labels_by_face),
        "license": "MIT (dataset repository), cite Cao et al. 2020",
        "source": "https://github.com/hducg/MFCAD",
        "step": Path(step_path).name,
    }


__all__ = [
    "MFCAD_CLASSES",
    "MFCAD_CLASS_TO_ID",
    "MFCAD_TO_BREPLENS",
    "STOCK_ID",
    "family_from_labels",
    "load_face_truth",
    "truth_record",
]
