"""NX journal entry points. Run inside NX: File > Execute > NX Open (Python).

STATUS: UNTESTED against a live NX session. See docs/NX_GUIDE.md.

Example (from an NX journal or external NX Python):
    import sys; sys.path.insert(0, r"C:\\path\\to\\BRepLens\\src")
    from breplens.adapters.nx_journal import run_on_open_part
    report = run_on_open_part(features_json=r"C:\\path\\features.json", mode="rules")
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from breplens.adapters.nx import UNTESTED_LIVE, NxOpenAdapter
from breplens.recognition.gnn_recognizer import recognize_hybrid
from breplens.recognition.rules import recognize_rules
from breplens.schema import FeatureSet


def run_on_open_part(
    *,
    features_json: str | Path | None = None,
    mode: str = "rules",
    model_path: str | Path | None = None,
    part_path: str | Path | None = None,
) -> dict[str, Any]:
    """Recognize the open NX part and optionally write features back.

    If *features_json* is supplied, those features are written back instead of
    re-running recognition. Returns a JSON-serializable report.
    """
    adapter = NxOpenAdapter()
    if part_path is None:
        part_path = adapter._work_part().FullPath
    if features_json is not None:
        fs = FeatureSet.model_validate_json(Path(features_json).read_text())
        source = str(features_json)
    else:
        graph = adapter.load_graph(part_path)
        if mode == "rules":
            fs = recognize_rules(graph)
        elif mode == "gnn":
            from breplens.ml.inference import load_for_inference
            from breplens.recognition.gnn_recognizer import recognize_gnn

            if model_path is None:
                raise ValueError("model_path is required for GNN mode")
            model, classes, _ = load_for_inference(model_path, device="cpu")
            fs = recognize_gnn(graph, model, classes, device="cpu")
        elif mode == "hybrid":
            from breplens.ml.inference import load_for_inference

            if model_path is None:
                raise ValueError("model_path is required for hybrid mode")
            model, classes, _ = load_for_inference(model_path, device="cpu")
            fs = recognize_hybrid(graph, model, classes, device="cpu")
        else:
            raise ValueError(f"unsupported mode {mode}")
        source = f"recognized:{mode}"
    report: dict[str, Any] = json.loads(adapter.write_features(part_path, fs))
    report.update({"source": source, "status": UNTESTED_LIVE, "n_features": len(fs.features)})
    return report


def main(argv: list[str] | None = None) -> int:
    """Journal main: argv = [features_json?, mode?, model_path?]."""
    args = list(sys.argv[1:] if argv is None else argv)
    features = args[0] if args else None
    mode = args[1] if len(args) > 1 else "rules"
    model = args[2] if len(args) > 2 else None
    report = run_on_open_part(features_json=features, mode=mode, model_path=model)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
