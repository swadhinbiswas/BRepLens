#!/usr/bin/env python
"""Reproducible performance benchmark for graph build and recognition stages."""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import platform
import pstats
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from breplens.adapters.occ import OccAdapter
from breplens.ml.inference import load_for_inference
from breplens.recognition.gnn_recognizer import recognize_gnn, recognize_hybrid
from breplens.recognition.rules import recognize_rules
from breplens.schema import PartGraph
from breplens.similarity.signature import part_signature


def _median_ms(fn, items) -> tuple[float, float, list[float]]:
    times = []
    for item in items:
        t = time.perf_counter()
        fn(item)
        times.append((time.perf_counter() - t) * 1000)
    return statistics.mean(times), statistics.median(times), times


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", type=int, default=80)
    ap.add_argument("--out", default=str(ROOT / "results" / "perf.json"))
    ap.add_argument("--baseline", default=None, help="earlier perf JSON to compare against")
    ap.add_argument(
        "--optimized-stage", action="append", default=None, help="stage to report before/after for"
    )
    args = ap.parse_args()
    data = ROOT / "data" / "synthetic"
    split = json.loads((data / "splits.json").read_text())["assignments"]
    names = [n for n, s in sorted(split.items()) if s == "test"][: args.parts]
    graphs: list[PartGraph] = [
        PartGraph.model_validate_json((data / "graphs" / f"{n}.json").read_text()) for n in names
    ]
    steps = [data / "step" / f"{n}.step" for n in names[: min(40, len(names))]]
    model, classes, _ = load_for_inference(ROOT / "models" / "gnn_synthetic.pt", device="cpu")
    adapter = OccAdapter()

    graph_mean, graph_med, _ = _median_ms(adapter.load_graph, steps)
    rules_mean, rules_med, _ = _median_ms(recognize_rules, graphs)
    gnn_mean, gnn_med, _ = _median_ms(
        lambda g: recognize_gnn(g, model, classes, device="cpu"), graphs
    )
    hyb_mean, hyb_med, _ = _median_ms(
        lambda g: recognize_hybrid(g, model, classes, device="cpu"), graphs
    )
    sig_mean, sig_med, _ = _median_ms(part_signature, graphs)

    def tess(p: Path) -> None:
        adapter.tessellate(p, deflection=0.25)

    tess_mean, tess_med, tess_times = _median_ms(tess, steps[:15])

    def e2e(p: Path) -> None:
        g = adapter.load_graph(p)
        fs = recognize_hybrid(g, model, classes, device="cpu")
        fs.model_dump_json()

    e2e_mean, e2e_med, e2e_times = _median_ms(e2e, steps[:10])

    prof = cProfile.Profile()
    prof.enable()
    for g in graphs[:100]:
        recognize_rules(g)
    prof.disable()
    stream = io.StringIO()
    pstats.Stats(prof, stream=stream).sort_stats("cumulative").print_stats(12)
    profile_top = [line.strip() for line in stream.getvalue().splitlines() if line.strip()][-14:]

    payload = {
        "environment": {
            "python": platform.python_version(),
            "host": platform.node(),
            "cpu": platform.processor(),
            "torch_threads": __import__("torch").get_num_threads(),
        },
        "n_parts": len(names),
        "stages_ms": {
            "graph_from_step": {
                "mean": round(graph_mean, 3),
                "median": round(graph_med, 3),
                "n": len(steps),
            },
            "rules": {
                "mean": round(rules_mean, 3),
                "median": round(rules_med, 3),
                "n": len(graphs),
            },
            "gnn": {"mean": round(gnn_mean, 3), "median": round(gnn_med, 3), "n": len(graphs)},
            "hybrid": {"mean": round(hyb_mean, 3), "median": round(hyb_med, 3), "n": len(graphs)},
            "signature": {
                "mean": round(sig_mean, 3),
                "median": round(sig_med, 3),
                "n": len(graphs),
            },
            "tessellate": {
                "mean": round(tess_mean, 3),
                "median": round(tess_med, 3),
                "n": len(tess_times),
            },
            "end_to_end_step_hybrid_json": {
                "mean": round(e2e_mean, 3),
                "median": round(e2e_med, 3),
                "n": len(e2e_times),
            },
        },
        "profile_rules_top": profile_top,
        "command": " ".join(sys.argv),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    payload["provenance"] = {
        "script": "scripts/benchmark_perf.py",
        "seed": None,
        "command": " ".join(sys.argv),
        "generated_utc": payload["generated_utc"],
    }
    before = Path(args.out)
    baseline = Path(args.baseline) if args.baseline else (before if before.exists() else None)
    if baseline and baseline.exists():
        try:
            prior = json.loads(baseline.read_text())
            payload["previous_run"] = prior.get("stages_ms")
            optimised = set(args.optimized_stage or [])
            deltas = {}
            for stage, metrics in payload["stages_ms"].items():
                if optimised and stage not in optimised:
                    continue
                old = prior.get("stages_ms", {}).get(stage)
                if old:
                    deltas[stage] = {
                        "before_mean": old["mean"],
                        "after_mean": metrics["mean"],
                        "speedup": round(old["mean"] / metrics["mean"], 3)
                        if metrics["mean"]
                        else None,
                    }
            payload["before_after"] = deltas
        except Exception:
            pass
    before.write_text(json.dumps(payload, indent=1))
    print(json.dumps({k: v["mean"] for k, v in payload["stages_ms"].items()}, indent=1))
    if "before_after" in payload:
        print(json.dumps(payload["before_after"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
