"""CAD-agnostic part signatures and a cosine-similarity index.

The signature combines interpretable geometry descriptors (surface mix, size,
edge convexity, radii, area quantiles, recognized-feature histogram) with a
Weisfeiler-Lehman colour histogram of the attributed graph. It is deterministic,
requires no training, and is cheap enough for the hosted demo.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from breplens.schema import FeatureSet, PartGraph, SurfaceType

WL_DIM = 128
SURFACE_ORDER = tuple(t.value for t in SurfaceType)
FEATURE_ORDER = (
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


def _stable_hash(token: str) -> int:
    return int(hashlib.md5(token.encode()).hexdigest()[:8], 16)


def _wl_histogram(graph: PartGraph, iterations: int = 3, dim: int = WL_DIM) -> np.ndarray:
    if not graph.faces:
        return np.zeros(dim, dtype=np.float32)
    colors: dict[int, str] = {}
    for f in graph.faces:
        n = f.normal
        nq = "n" if n is None else f"{round(n.x, 1)},{round(n.y, 1)},{round(n.z, 1)}"
        colors[f.id] = (
            f"{f.surface_type.value}|{round(math.log1p(f.area), 1)}|"
            f"{round(math.log1p(f.radius or 0.0), 1)}|{nq}"
        )
    out = np.zeros(dim, dtype=np.float32)
    for it in range(iterations + 1):
        counts: dict[int, int] = {}
        for color in colors.values():
            counts[_stable_hash(color) % dim] = counts.get(_stable_hash(color) % dim, 0) + 1
        for b, c in counts.items():
            out[b] += c / max(1, len(graph.faces))
        if it == iterations:
            break
        new_colors: dict[int, str] = {}
        adjacency: dict[int, list[tuple[str, str]]] = {f.id: [] for f in graph.faces}
        for e in graph.edges:
            adjacency[e.face_a].append((e.convexity.value, colors[e.face_b]))
            adjacency[e.face_b].append((e.convexity.value, colors[e.face_a]))
        for fid in colors:
            neigh = "|".join(f"{c}:{col}" for c, col in sorted(adjacency[fid]))
            new_colors[fid] = f"{colors[fid]}#{neigh}"
        colors = new_colors
    norm = np.linalg.norm(out)
    return out / norm if norm > 0 else out


def part_signature(
    graph: PartGraph,
    features: FeatureSet | None = None,
    *,
    wl_dim: int = WL_DIM,
    wl_iterations: int = 2,
) -> np.ndarray:
    """Return an L2-normalized, block-weighted signature vector for a part.

    Each descriptor block is normalized before weighting so the 128-bin WL
    histogram cannot drown out a ten-bin feature histogram. Weights were chosen
    once on the synthetic/MFCAD retrieval tasks and are documented here; no
    per-query tuning is performed.
    """
    n = max(1, len(graph.faces))
    blocks: list[np.ndarray] = []
    weights: list[float] = []

    size = sorted((graph.metadata.bbox_max - graph.metadata.bbox_min).as_tuple(), reverse=True)
    diag = max(1e-6, math.sqrt(sum(s * s for s in size)))
    bbox_vol = max(1e-9, size[0] * size[1] * size[2])
    geom = np.asarray(
        [
            size[0] / diag,
            size[1] / diag,
            size[2] / diag,
            math.log1p(graph.metadata.volume or 0.0) / 20.0,
            math.log1p(sum(f.area for f in graph.faces)) / 15.0,
            min(1.0, n / 100.0),
            min(1.0, len(graph.edges) / 200.0),
            min(1.0, (graph.metadata.volume or 0.0) / bbox_vol),
        ],
        dtype=np.float32,
    )
    blocks.append(geom)
    weights.append(1.5)

    surf = np.asarray(
        [sum(1 for f in graph.faces if f.surface_type.value == st) / n for st in SURFACE_ORDER],
        dtype=np.float32,
    )
    blocks.append(surf)
    weights.append(1.0)

    conv = np.asarray(
        [
            sum(1 for e in graph.edges if e.convexity.value == c) / max(1, len(graph.edges))
            for c in ("convex", "concave", "smooth")
        ],
        dtype=np.float32,
    )
    blocks.append(conv)
    weights.append(1.5)

    feat = np.asarray(
        [
            min(
                1.0,
                (sum(1 for f in features.features if f.type.value == ft) if features else 0) / 8.0,
            )
            for ft in FEATURE_ORDER
        ],
        dtype=np.float32,
    )
    blocks.append(feat)
    weights.append(2.5)

    max_area = max((f.area for f in graph.faces), default=1.0)
    area_hist = np.zeros(8, dtype=np.float32)
    for f in graph.faces:
        b = min(7, max(0, int(math.log2(max(f.area, 1e-6) / max_area) + 8)))
        area_hist[b] += 1
    area_hist /= n
    blocks.append(area_hist)
    weights.append(1.2)

    radii = sorted(f.radius for f in graph.faces if f.radius and f.radius > 1e-6)
    r_hist = np.zeros(8, dtype=np.float32)
    for r in radii:
        idx = min(7, max(0, int(math.log2(max(r, 1e-3)) + 4)))
        r_hist[idx] += 1
    if radii:
        r_hist /= len(radii)
    blocks.append(r_hist)
    weights.append(1.0)

    e_hist = np.zeros(8, dtype=np.float32)
    for e in graph.edges:
        idx = min(7, max(0, int(math.log2(max(e.length, 1e-3)) + 4)))
        e_hist[idx] += 1
    if graph.edges:
        e_hist /= len(graph.edges)
    blocks.append(e_hist)
    weights.append(1.0)

    d_hist = np.zeros(6, dtype=np.float32)
    for e in graph.edges:
        a = e.dihedral_angle_deg or 0.0
        d_hist[min(5, max(0, int(a // 30)))] += 1
    if graph.edges:
        d_hist /= len(graph.edges)
    blocks.append(d_hist)
    weights.append(1.0)

    blocks.append(_wl_histogram(graph, wl_iterations, wl_dim))
    weights.append(0.7)

    parts = []
    for block, weight in zip(blocks, weights, strict=True):
        norm = float(np.linalg.norm(block))
        parts.append((block / norm if norm > 0 else block) * weight)
    arr = np.concatenate(parts).astype(np.float32)
    norm = float(np.linalg.norm(arr))
    return arr / norm if norm > 0 else arr


@dataclass
class LibraryEntry:
    id: str
    name: str
    family: str
    vector: np.ndarray
    metadata: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "family": self.family,
            "vector": [round(float(x), 6) for x in self.vector.tolist()],
            "metadata": self.metadata,
        }


@dataclass
class SearchHit:
    id: str
    name: str
    family: str
    score: float
    metadata: dict


class SignatureIndex:
    """Brute-force cosine index; exact and fast for libraries up to ~100k."""

    def __init__(self, dim: int | None = None) -> None:
        self.entries: list[LibraryEntry] = []
        self._matrix: np.ndarray | None = None
        self.dim = dim

    def add(self, entry: LibraryEntry) -> None:
        self.entries.append(entry)
        self._matrix = None
        self.dim = entry.vector.shape[0]

    def add_graph(
        self,
        graph: PartGraph,
        *,
        name: str | None = None,
        family: str | None = None,
        features: FeatureSet | None = None,
        metadata: dict | None = None,
    ) -> LibraryEntry:
        nm = name or graph.metadata.name
        entry = LibraryEntry(
            id=nm,
            name=nm,
            family=family or (graph.metadata.family or ""),
            vector=part_signature(graph, features),
            metadata={
                "n_faces": len(graph.faces),
                "bbox": [
                    [round(v, 3) for v in graph.metadata.bbox_min.as_tuple()],
                    [round(v, 3) for v in graph.metadata.bbox_max.as_tuple()],
                ],
                **(metadata or {}),
            },
        )
        self.add(entry)
        return entry

    @property
    def matrix(self) -> np.ndarray:
        if self._matrix is None:
            self._matrix = (
                np.vstack([e.vector for e in self.entries]).astype(np.float32)
                if self.entries
                else np.zeros((0, self.dim or 1), dtype=np.float32)
            )
        return self._matrix

    def search(
        self,
        query: np.ndarray,
        k: int = 5,
        *,
        exclude_id: str | None = None,
    ) -> list[SearchHit]:
        if not self.entries:
            return []
        q = np.asarray(query, dtype=np.float32)
        q = q / max(1e-12, float(np.linalg.norm(q)))
        scores = self.matrix @ q
        order = np.argsort(-scores)
        hits: list[SearchHit] = []
        for idx in order:
            entry = self.entries[int(idx)]
            if exclude_id is not None and entry.id == exclude_id:
                continue
            hits.append(
                SearchHit(
                    id=entry.id,
                    name=entry.name,
                    family=entry.family,
                    score=float(scores[int(idx)]),
                    metadata=entry.metadata,
                )
            )
            if len(hits) >= k:
                break
        return hits

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "dim": self.dim,
            "entries": [e.to_json() for e in self.entries],
        }
        out.write_text(json.dumps(payload))
        return out

    @classmethod
    def load(cls, path: str | Path) -> SignatureIndex:
        payload = json.loads(Path(path).read_text())
        index = cls(dim=int(payload["dim"]))
        for e in payload["entries"]:
            index.add(
                LibraryEntry(
                    id=e["id"],
                    name=e["name"],
                    family=e["family"],
                    vector=np.asarray(e["vector"], dtype=np.float32),
                    metadata=e.get("metadata", {}),
                )
            )
        return index


def precision_at_k(
    queries: Iterable[np.ndarray],
    query_families: Sequence[str],
    query_ids: Sequence[str],
    index: SignatureIndex,
    k: int,
    *,
    min_jaccard: float | None = None,
) -> dict[str, float]:
    """Retrieval metrics; optionally use family-set Jaccard as relevance."""
    precisions: list[float] = []
    recalls: list[float] = []
    aps: list[float] = []

    def relevant(a: str, b: str) -> bool:
        if min_jaccard is None:
            return a == b
        sa, sb = set(a.split("+")), set(b.split("+"))
        return len(sa & sb) / max(1, len(sa | sb)) >= min_jaccard

    for q, fam, qid in zip(queries, query_families, query_ids, strict=True):
        hits = index.search(q, k=k, exclude_id=qid)
        if not hits:
            continue
        relevant_total = sum(1 for e in index.entries if relevant(e.family, fam) and e.id != qid)
        good = [1 if relevant(h.family, fam) else 0 for h in hits]
        precisions.append(sum(good) / len(hits))
        recalls.append(sum(good) / relevant_total if relevant_total else 0.0)
        ap = 0.0
        seen = 0
        for i, g in enumerate(good, 1):
            if g:
                seen += 1
                ap += seen / i
        aps.append(ap / max(1, sum(good)))
    suffix = f"@{k}" if min_jaccard is None else f"_jaccard{min_jaccard:g}@{k}"
    return {
        f"precision{suffix}": round(float(np.mean(precisions)), 4) if precisions else 0.0,
        f"recall{suffix}": round(float(np.mean(recalls)), 4) if recalls else 0.0,
        f"map{suffix}": round(float(np.mean(aps)), 4) if aps else 0.0,
        "n_queries": len(precisions),
    }


__all__ = [
    "FEATURE_ORDER",
    "LibraryEntry",
    "SearchHit",
    "SignatureIndex",
    "part_signature",
    "precision_at_k",
]
