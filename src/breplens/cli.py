"""Command-line interface for BRepLens.

Examples:
    breplens recognize part.step --mode hybrid
    breplens schema --out docs/schema.v1.json
    breplens version
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from breplens.schema import SCHEMA_VERSION, FeatureSet


def _recognize(args: argparse.Namespace) -> int:
    from breplens.adapters.occ import OccAdapter
    from breplens.recognition.gnn_recognizer import recognize_gnn, recognize_hybrid
    from breplens.recognition.rules import recognize_rules

    adapter = OccAdapter()
    graph = adapter.load_graph(args.part)
    if args.mode == "rules":
        fs = recognize_rules(graph)
    else:
        from breplens.ml.inference import load_for_inference

        model, classes, _ = load_for_inference(args.model, device="cpu")
        fs = (
            recognize_gnn(graph, model, classes, device="cpu")
            if args.mode == "gnn"
            else recognize_hybrid(graph, model, classes, device="cpu")
        )
    if args.out:
        Path(args.out).write_text(fs.model_dump_json(indent=2))
    print(
        json.dumps(
            {
                "part": fs.part,
                "mode": fs.mode,
                "n_features": len(fs.features),
                "types": [f.type.value for f in fs.features],
                "schema_version": fs.schema_version,
                "out": args.out,
            },
            indent=2,
        )
    )
    return 0


def _schema(args: argparse.Namespace) -> int:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "feature_set": FeatureSet.model_json_schema(),
    }
    text = json.dumps(payload, indent=2)
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="breplens", description="BRepLens feature recognition")
    ap.add_argument("--version", action="version", version=f"BRepLens {SCHEMA_VERSION}")
    sub = ap.add_subparsers(dest="command", required=True)

    rec = sub.add_parser("recognize", help="recognize features in a STEP file")
    rec.add_argument("part")
    rec.add_argument("--mode", choices=["rules", "gnn", "hybrid"], default="hybrid")
    rec.add_argument(
        "--model", default=str(Path(__file__).resolve().parents[2] / "models" / "gnn_synthetic.pt")
    )
    rec.add_argument("--out", default=None)
    rec.set_defaults(func=_recognize)

    sch = sub.add_parser("schema", help="print/export the feature JSON schema")
    sch.add_argument("--out", default=None)
    sch.set_defaults(func=_schema)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
