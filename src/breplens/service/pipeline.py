"""Recognition pipeline shared by API, worker and tests."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from breplens.adapters.occ import OccAdapter
from breplens.recognition.gnn_recognizer import recognize_gnn, recognize_hybrid
from breplens.recognition.rules import recognize_rules
from breplens.schema import FeatureSet, PartGraph
from breplens.service.config import Settings
from breplens.service.storage import Storage

log = logging.getLogger(__name__)


class ModelRegistry:
    """Lazy, process-wide GNN cache (safe for CPU inference)."""

    def __init__(self) -> None:
        self._model: Any = None
        self._class_names: list[str] | None = None
        self._meta: dict[str, Any] = {}
        self._path: Path | None = None
        self._error: str | None = None

    def get(self, path: Path, device: str = "cpu") -> tuple[Any, list[str], dict[str, Any]] | None:
        if self._path != path:
            self._model = None
            self._path = path
            self._error = None
        if self._model is None:
            try:
                from breplens.ml.inference import load_for_inference

                self._model, self._class_names, self._meta = load_for_inference(path, device=device)
                log.info("loaded GNN model %s", path)
            except Exception as exc:  # pragma: no cover - environment dependent
                self._error = f"{type(exc).__name__}: {exc}"
                log.warning("GNN unavailable (%s); falling back to rules", self._error)
                return None
        assert self._class_names is not None
        return self._model, self._class_names, self._meta

    @property
    def error(self) -> str | None:
        return self._error


def recognize(
    graph: PartGraph,
    *,
    mode: str,
    settings: Settings,
    registry: ModelRegistry,
) -> FeatureSet:
    if mode == "rules":
        return recognize_rules(graph)
    loaded = registry.get(settings.model_path, settings.model_device)
    if loaded is None:
        fs = recognize_rules(graph)
        fs.warnings.append(f"GNN unavailable ({registry.error}); served rules instead")
        return fs
    model, classes, meta = loaded
    if mode == "gnn":
        fs = recognize_gnn(graph, model, classes, device=settings.model_device)
    elif mode == "hybrid":
        fs = recognize_hybrid(graph, model, classes, device=settings.model_device)
    else:
        raise ValueError(f"unsupported mode {mode}")
    fs.model_version = str(meta.get("dataset", "unknown")) + ":" + str(meta.get("trained_utc", ""))
    return fs


def process_job(
    storage: Storage,
    settings: Settings,
    job: dict[str, Any],
    registry: ModelRegistry,
    *,
    adapter: OccAdapter | None = None,
) -> FeatureSet:
    """Run one claimed job, persist its FeatureSet and update job status."""
    job_id = job["id"]
    part = storage.get_part(job["part_id"])
    if part is None:
        raise RuntimeError(f"part {job['part_id']} disappeared")
    adapter = adapter or OccAdapter()
    t0 = time.perf_counter()
    graph = adapter.load_graph(part["path"], name=Path(part["filename"]).stem)
    graph_ms = (time.perf_counter() - t0) * 1000.0
    storage.set_part_faces(part["id"], len(graph.faces))
    fs = recognize(graph, mode=job["mode"], settings=settings, registry=registry)
    total_ms = (time.perf_counter() - t0) * 1000.0
    fs.timings.load_ms = round(graph_ms, 3)
    fs.timings.graph_ms = round(graph_ms, 3)
    fs.timings.total_ms = round(total_ms, 3)
    fs.part = part["filename"]
    out = settings.data_dir / "results" / f"{job_id}.json"
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(fs.model_dump_json(indent=1))
    tmp.replace(out)
    storage.finish_job(
        job_id,
        status="succeeded",
        result_path=str(out),
        timings={
            "load_ms": graph_ms,
            "total_ms": total_ms,
            "recognition_ms": fs.timings.recognition_ms,
        },
    )
    return fs


def load_result(storage: Storage, job: dict[str, Any]) -> FeatureSet:
    if job["status"] != "succeeded" or not job.get("result_path"):
        raise FileNotFoundError("job has no result yet")
    return FeatureSet.model_validate_json(Path(job["result_path"]).read_text())


def similarity_for_graph(
    graph: PartGraph,
    features: FeatureSet,
    index_path: Path,
    *,
    k: int = 5,
) -> list[dict[str, Any]]:
    from breplens.similarity.signature import SignatureIndex, part_signature

    if not index_path.exists():
        return []
    index = SignatureIndex.load(index_path)
    vec = part_signature(graph, features)
    return [
        {
            "id": h.id,
            "name": h.name,
            "family": h.family,
            "score": round(h.score, 4),
            "metadata": h.metadata,
        }
        for h in index.search(vec, k=k, exclude_id=graph.metadata.name)
    ]


def load_similarity_index(path: Path) -> Any:
    from breplens.similarity.signature import SignatureIndex

    return SignatureIndex.load(path) if path.exists() else SignatureIndex()


__all__ = [
    "ModelRegistry",
    "load_result",
    "load_similarity_index",
    "process_job",
    "recognize",
    "similarity_for_graph",
]
