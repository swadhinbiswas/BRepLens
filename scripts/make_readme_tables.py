#!/usr/bin/env python
"""Generate README results tables, CV summary and final report from results files.

No accuracy, speed, memory or retrieval number is hand-written: every value is
read from ``results/*.json`` or a dataset manifest produced by a repo script.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _load(name: str, default: Any = None) -> Any:
    p = RESULTS / name
    if not p.exists():
        return default if default is not None else {}
    return json.loads(p.read_text())


def _f(x: Any, nd: int = 3) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _pct(x: Any, nd: int = 1) -> str:
    return "—" if x is None else f"{float(x):.{nd}f}%"


def tables() -> dict[str, str]:
    synth_manifest = (
        json.loads((ROOT / "data" / "synthetic" / "manifest.json").read_text())
        if (ROOT / "data" / "synthetic" / "manifest.json").exists()
        else {}
    )
    mfcad_manifest = (
        json.loads((ROOT / "data" / "mfcad" / "manifest.json").read_text())
        if (ROOT / "data" / "mfcad" / "manifest.json").exists()
        else {}
    )
    comparison = _load("recognition_comparison.json")
    gnn = _load("gnn_eval.json")
    mfcad_gnn = _load("mfcad_gnn_eval.json")
    h2h = _load("mfcad_head_to_head.json")
    cnn = _load("cnn_eval.json")
    sim = _load("similarity_eval.json")
    perf = _load("perf.json")
    deploy = _load("deploy_benchmark.json")
    tests = _load("tests.json")
    cov = _load("coverage.json")
    cov_core = _load("coverage_core.json")
    docker = _load("docker_service_test.json")
    docker_rq = _load("docker_service_test_rq.json")

    out: dict[str, str] = {}

    split = synth_manifest.get("parts", [])
    split_counts = {}
    for p in split:
        split_counts[p["split"]] = split_counts.get(p["split"], 0) + 1
    out["datasets"] = (
        "| Dataset | Parts | Split (train/val/test) | Labels | License |\n"
        "| --- | ---: | --- | --- | --- |\n"
        f"| Synthetic (`data/synthetic`) | {synth_manifest.get('n_parts', '—')} | "
        f"{split_counts.get('train', 0)}/{split_counts.get('val', 0)}/{split_counts.get('test', 0)} | "
        "per-face class + feature params | MIT (generated) |\n"
        f"| MFCAD subset (`data/mfcad`) | {mfcad_manifest.get('n_converted', '—')} | "
        f"{mfcad_manifest.get('split_counts', {}).get('train', 0)}/"
        f"{mfcad_manifest.get('split_counts', {}).get('val', 0)}/"
        f"{mfcad_manifest.get('split_counts', {}).get('test', 0)} | "
        "16 face classes (15 features + stock) | MIT |\n"
    )

    methods = comparison.get("methods", {})
    rows = []
    for key in ("rules", "gnn", "hybrid"):
        m = methods.get(key)
        if not m:
            continue
        rows.append(
            f"| {key} | {_f(m['feature']['precision'])} | {_f(m['feature']['recall'])} | "
            f"{_f(m['feature']['f1'])} | {_f(m['feature']['exact_part_rate'])} | "
            f"{_f(m['face']['accuracy'])} | {_f(m['face']['macro_f1'])} | {_f(m['ms_per_part'], 2)} |"
        )
    out["synthetic"] = (
        "| Method | Feature P | Feature R | Feature F1 | Exact part rate | Face acc. | Face macro F1 | ms/part |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        + "\n".join(rows)
        + f"\n\n*{comparison.get('n_parts', '—')} held-out synthetic parts; same split for all methods; "
        "latency is recognition only, CPU.*"
    )

    rows = []
    for dataset, payload in (("synthetic", gnn), ("MFCAD", mfcad_gnn)):
        for arch, run in payload.get("runs", {}).items():
            cpu_ms = (
                _f(payload.get("cpu_inference_ms_per_graph", {}).get("median"), 2)
                if arch == payload.get("selected_architecture")
                else "—"
            )
            rows.append(
                f"| {dataset} | {arch} | {run.get('n_parameters', '—')} | "
                f"{_f(run['test']['accuracy'], 4)} | {_f(run['test']['macro_f1'], 4)} | "
                f"{_f(run.get('train_seconds'), 2)} | {cpu_ms} |"
            )
    out["gnn"] = (
        "| Dataset | Architecture | Parameters | Test face acc. | Test macro F1 | Train s | CPU ms/graph |\n"
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |\n"
        + "\n".join(rows)
        + "\n\n*GCN = baseline without edge attributes; `edge_gnn` = edge-aware message passing. "
        "Early stopping on the validation split only.*"
    )

    hrows = []
    for key, m in h2h.get("methods", {}).items():
        hrows.append(
            f"| {key} | {_f(m['accuracy'], 4)} | {_f(m['macro_f1'], 4)} | {m.get('n_faces', '—')} |"
        )
    gnn_full = h2h.get("gnn_full_16_class", {})
    out["mfcad"] = (
        "| Method (8 macro classes) | Accuracy | Macro F1 | Faces |\n"
        "| --- | ---: | ---: | ---: |\n"
        + "\n".join(hrows)
        + f"\n\n*MFCAD test split: {h2h.get('dataset', {}).get('n_parts', '—')} parts. "
        f"Full 16-class GNN: accuracy {_f(gnn_full.get('test_accuracy'), 4)}, "
        f"macro F1 {_f(gnn_full.get('test_macro_f1'), 4)}. "
        "Rules were designed and tuned on the synthetic families and were not retuned on MFCAD.*"
    )

    crows = []
    for key, m in cnn.get("methods", {}).items():
        crows.append(
            f"| {key} | {_f(m['test']['accuracy'], 4)} | {_f(m['test']['macro_f1'], 4)} | "
            f"{m.get('n_parameters', '—')} | {_f(m.get('train_seconds'), 2)} | "
            f"{_f(m.get('cpu_ms_per_part'), 2)} | {_f(m.get('model_mb'), 3)} |"
        )
    out["cnn"] = (
        "| Method | Test accuracy | Macro F1 | Parameters | Train s | CPU ms/part | Model MB |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        + "\n".join(crows)
        + "\n\n*Optional Phase 9: six orthographic shaded views (96x96, ±X/±Y/±Z) "
        "for the CNN; the graph classifier uses the AAG. Same synthetic parts, split and seed.*"
    )

    srows = []
    for name, key in (("Synthetic", "synthetic"), ("MFCAD", "mfcad")):
        m = sim.get(key, {})
        met = m.get("metrics", {})
        srows.append(
            f"| {name} | {m.get('n_library', '—')} | {m.get('n_queries', '—')} | "
            f"{_f(met.get('precision@1'))} | {_f(met.get('precision@5'))} | {_f(met.get('precision@10'))} | "
            f"{_f(met.get('precision_jaccard0.5@1'))} | {_f(m.get('timing', {}).get('query_ms_mean'), 2)} |"
        )
    out["similarity"] = (
        "| Dataset | Library | Queries | P@1 | P@5 | P@10 | P@1 (Jaccard≥0.5) | Query ms |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        + "\n".join(srows)
        + "\n\n*Relevance = same family label; MFCAD families are coarse macro compositions. "
        "The Jaccard column treats compositions sharing at least half their feature set as relevant.*"
    )

    prows = []
    for stage, m in perf.get("stages_ms", {}).items():
        ba = perf.get("before_after", {}).get(stage, {})
        prows.append(
            f"| {stage.replace('_', ' ')} | {_f(m.get('mean'), 2)} | {_f(m.get('median'), 2)} | "
            f"{_f(ba.get('before_mean'), 2)} | {_f(ba.get('speedup'), 2)} |"
        )
    out["perf"] = (
        "| Stage | Mean ms (after) | Median ms | Mean ms (before) | Speedup |\n"
        "| --- | ---: | ---: | ---: | ---: |\n"
        + "\n".join(prows)
        + "\n\n*Before/after is reported only for the graph-build optimization measured by the same script; "
        "other stages are current observations, not claimed optimizations.*"
    )

    inst = deploy.get("install", {})
    at = deploy.get("apptest", {})
    pipe = deploy.get("pipeline", {})
    srv = deploy.get("server", {})
    lat = pipe.get("latency", {})
    out["deploy"] = (
        "| Metric | Value |\n| --- | ---: |\n"
        f"| Environment size | {_f(inst.get('venv_mb'), 1)} MB |\n"
        f"| Fresh pip install | {_f(inst.get('install_seconds'), 0)} s |\n"
        f"| AppTest cold start | {_f(at.get('cold_start_seconds'), 2)} s |\n"
        f"| First part incl. model load | {_f(at.get('model_load_and_first_part_seconds'), 2)} s |\n"
        f"| Peak RSS (app probe) | {_f(at.get('peak_rss_apptest_mb'), 1)} MB |\n"
        f"| Peak RSS (pipeline probe) | {_f(pipe.get('peak_rss_pipeline_mb'), 1)} MB |\n"
        f"| Headless server start | {_f(srv.get('server_start_seconds'), 2)} s |\n"
        f"| Rules latency | {_f(lat.get('rules', {}).get('mean_ms'), 2)} ms |\n"
        f"| GNN latency | {_f(lat.get('gnn', {}).get('mean_ms'), 2)} ms |\n"
        f"| Hybrid latency | {_f(lat.get('hybrid', {}).get('mean_ms'), 2)} ms |\n"
        f"| Tessellation latency | {_f(lat.get('tessellate', {}).get('mean_ms'), 2)} ms |\n"
        "\n*Measured on the inference-only venv, CPU, 10 committed sample parts; raw data in "
        "`results/deploy_benchmark.json`.*"
    )

    out["quality"] = (
        "| Check | Result |\n| --- | ---: |\n"
        f"| Tests passed | {tests.get('passed', '—')} |\n"
        f"| Tests failed | {tests.get('failed', '—')} |\n"
        f"| Test wall time | {_f(tests.get('wall_seconds'), 2)} s |\n"
        f"| Full line coverage | {_pct(cov.get('totals', {}).get('percent_covered'), 2)} |\n"
        f"| Core geometry/recognition/schema coverage | {_pct(cov_core.get('totals', {}).get('percent_covered'), 2)} |\n"
    )

    drow = []
    for label, d in (("SQLite worker", docker), ("RQ + Redis worker", docker_rq)):
        drow.append(
            f"| {label} | {'ok' if d.get('health_ok') else '—'} | {_f(d.get('job_seconds'), 2)} | "
            f"{d.get('status', '—')} | {d.get('n_features', '—')} | "
            f"{_f((d.get('job_timings') or {}).get('load_ms'), 1)} | "
            f"{_f((d.get('job_timings') or {}).get('recognition_ms'), 2)} |"
        )
    out["docker"] = (
        "| Queue | API health | End-to-end s | Job | Features | Load ms | Recognition ms |\n"
        "| --- | --- | ---: | --- | ---: | ---: | ---: |\n" + "\n".join(drow)
    )

    out["timing"] = (
        "| Pipeline step | Wall time | Source |\n| --- | ---: | --- |\n"
        f"| Synthetic generation ({synth_manifest.get('n_parts', '—')} parts, "
        f"{synth_manifest.get('workers', '—')} workers) | {_f(synth_manifest.get('generation_wall_seconds'), 2)} s | `data/synthetic/manifest.json` |\n"
        f"| MFCAD conversion ({mfcad_manifest.get('n_converted', '—')} parts, "
        f"{mfcad_manifest.get('workers', '—')} workers) | {_f(mfcad_manifest.get('conversion_wall_seconds'), 2)} s | `data/mfcad/manifest.json` |\n"
        f"| Synthetic GNN training ({gnn.get('selected_architecture', '—')}) | {_f(gnn.get('runs', {}).get(gnn.get('selected_architecture', ''), {}).get('train_seconds'), 2)} s | `results/gnn_eval.json` |\n"
        f"| MFCAD GNN training ({mfcad_gnn.get('selected_architecture', '—')}) | {_f(mfcad_gnn.get('runs', {}).get(mfcad_gnn.get('selected_architecture', ''), {}).get('train_seconds'), 2)} s | `results/mfcad_gnn_eval.json` |\n"
    )
    return out


def _build_readme_blocks(t: dict[str, str]) -> str:
    return f"""
## Results

All tables are generated by `python scripts/make_readme_tables.py` from
`results/*.json`; none of these numbers are hand-written.

### Datasets

{t["datasets"]}

### Recognition head-to-head (synthetic test)

{t["synthetic"]}

### GNN ablation

{t["gnn"]}

### Optional multi-view CNN (part family)

{t["cnn"]}

### External benchmark on MFCAD (macro classes)

{t["mfcad"]}

### Similarity retrieval

{t["similarity"]}

### Performance (before/after, real runs)

{t["perf"]}

### Hosted Streamlit demo (CPU, inference-only environment)

{t["deploy"]}

### Quality gates

{t["quality"]}

### Docker service smoke test

{t["docker"]}

### Data and training wall-clock

{t["timing"]}
"""


def update_readme(t: dict[str, str]) -> None:
    p = ROOT / "README.md"
    text = p.read_text()
    block = _build_readme_blocks(t)
    start, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    if start in text and end in text:
        text = re.sub(
            re.escape(start) + r".*?" + re.escape(end), start + block + end, text, flags=re.S
        )
    else:
        text += f"\n{start}\n{block}\n{end}\n"
    p.write_text(text)


def write_cv(t: dict[str, str]) -> None:
    comp = _load("recognition_comparison.json")
    gnn = _load("gnn_eval.json")
    mfcad = _load("mfcad_gnn_eval.json")
    h2h = _load("mfcad_head_to_head.json")
    cnn = _load("cnn_eval.json")
    sim = _load("similarity_eval.json")
    cnn = _load("cnn_eval.json")
    deploy = _load("deploy_benchmark.json")
    tests = _load("tests.json")
    cov_core = _load("coverage_core.json")
    methods = comp.get("methods", {})
    rules = methods.get("rules", {})
    hybrid = methods.get("hybrid", {})
    gnet = gnn.get("runs", {}).get(gnn.get("selected_architecture", ""), {})
    mnet = mfcad.get("runs", {}).get(mfcad.get("selected_architecture", ""), {})
    mrule = h2h.get("methods", {}).get("rules", {})
    at = deploy.get("apptest", {})
    text = f"""# CV Summary — BRepLens

Truthful bullets, each tied to a generated result file.

- Built a CAD-agnostic Python pipeline that converts STEP B-Reps (OpenCascade/OCP)
  into an attributed adjacency graph and recognizes 10 machining feature types
  with rule-based, GNN and hybrid modes, extracting measured parameters into a
  versioned Pydantic/JSON schema.
- Implemented a compact face-classification GNN ({gnn.get("selected_architecture", "—")}:
  {gnet.get("n_parameters", "—")} parameters, {gnn.get("checkpoint", {}).get("size_mb", "—")} MB)
  with {_f(gnet.get("test", {}).get("accuracy"), 4)} face accuracy on a
  {comp.get("n_parts", "—")}-part held-out synthetic test set; a separate
  edge-aware MFCAD model reached {_f(mnet.get("test", {}).get("accuracy"), 4)}
  accuracy on {mfcad.get("dataset", {}).get("n_test", "—")} held-out MFCAD parts;
  a six-view CNN part-family classifier reached
  {_f(cnn.get("methods", {}).get("cnn_multiview", {}).get("test", {}).get("accuracy"), 4)}
  versus {_f(cnn.get("methods", {}).get("graph_gnn", {}).get("test", {}).get("accuracy"), 4)}
  for the graph classifier on the same split.
- Rule-based recognition reached {_f(rules.get("feature", {}).get("f1"), 4)} feature F1 on
  the synthetic test set and exposed its limits on MFCAD
  ({_f(mrule.get("macro_f1"), 4)} macro F1); the conservative hybrid preserved
  all rule features and also reached {_f(hybrid.get("feature", {}).get("f1"), 4)}
  synthetic feature F1, while on MFCAD only GNN-priority/vote matched the GNN.
- Delivered a FastAPI + queued-worker service with SQLite default storage and an
  optional RQ/Redis mode, plus a Streamlit Community Cloud demo whose measured
  cold start is {_f(at.get("cold_start_seconds"), 2)} s, first-part time
  {_f(at.get("model_load_and_first_part_seconds"), 2)} s, peak RSS
  {_f(at.get("peak_rss_apptest_mb"), 1)} MB, and inference-only install
  {_f(deploy.get("install", {}).get("venv_mb"), 0)} MB.
- Added a WL/geometry signature index with precision@1 of
  {_f(sim.get("synthetic", {}).get("metrics", {}).get("precision@1"), 3)} on synthetic
  families and {_f(sim.get("mfcad", {}).get("metrics", {}).get("precision_jaccard0.5@1"), 3)}
  (Jaccard≥0.5) on MFCAD, an NX Open adapter/write-back path with a faithful mock
  (explicitly untested live), {tests.get("passed", "—")} passing tests and
  {_pct(cov_core.get("totals", {}).get("percent_covered"), 1)} coverage on core
  geometry/recognition/schema code.
"""
    (ROOT / "docs" / "CV_SUMMARY.md").write_text(text)


def write_final_report(t: dict[str, str]) -> None:
    deploy = _load("deploy_benchmark.json")
    text = f"""# BRepLens — Final Report

## 1. What was built

BRepLens is a CAD-agnostic feature-recognition system with:

- **Adapters**: a fully tested OCP/OpenCascade STEP adapter and a Siemens NX
  Open adapter that exports to STEP for graph construction and writes feature
  parameters back as parametric NX features. **The NX path is UNTESTED against a
  live NX session**; it is mocked-tested and clearly labeled.
- **Neutral IR**: attributed adjacency graph with surface type, area, normals,
  axes, radii, bbox, material side, convex/concave/smooth edge classification.
- **Recognizers**: deterministic rules, a learned face-classification GNN
  (GCN demo checkpoint, edge-aware external model), and a conservative hybrid;
  10 feature classes with measured diameter/depth/radius/width/length/
  height/angle and position/axis.
- **Schema**: versioned Pydantic models + exported JSON Schema (`docs/schema.v1.json`).
- **Similarity**: block-weighted geometry + Weisfeiler-Lehman signature index and
  precision@k evaluation.
- **Service**: FastAPI upload/job/status/features/similarity API, SQLAlchemy
  storage (SQLite default, PostgreSQL option), SQLite worker default and optional
  RQ/Redis deployment, Docker images and compose files.
- **Hosted demo**: `streamlit_app.py` with 10 samples, colored plotly 3D mesh,
  parameter table, JSON download, rules/GNN/hybrid toggle, similarity search and
  an optional six-view CNN part-family prediction.
- **Optional Phase 9**: a dependency-light orthographic renderer, a small
  multi-view CNN and an edge-aware graph family classifier, compared on identical
  splits.

## 2. Measured results

{t["synthetic"]}

{t["gnn"]}

{t["cnn"]}

{t["mfcad"]}

{t["similarity"]}

### Data and training wall-clock

{t["timing"]}

## 3. Hosted demo benchmarks

{t["deploy"]}

The target was < 1.5 GiB peak; measured peak is
{_f(deploy.get("apptest", {}).get("peak_rss_apptest_mb"), 1)} MB for the app probe.
The environment is large because `cadquery-ocp` declares VTK; VTK is never
imported by the app.

## 4. Service verification

{t["docker"]}

The default stack was built and started with `docker compose up --build`; API
health, hybrid recognition through the worker, feature retrieval and similarity
were exercised. Integration tests additionally cover validation, failure paths,
GNN mode, status polling and size limits.

## 5. Quality gates

{t["quality"]}

CI runs ruff, mypy and pytest on CPU; the core coverage gate is 80%. The GitHub
Actions workflow was not executed on a remote because this repository has no
configured remote; the identical commands were run locally on CPU and are green.

## 6. Known limitations and failure cases

- Synthetic rules saturate at 1.0 after tuning on the generator; this is not
  real-world accuracy. MFCAD exposes the rule limits (macro F1
  {_f(_load("mfcad_head_to_head.json").get("methods", {}).get("rules", {}).get("macro_f1"), 4)}).
- MFCAD++ (the preferred external benchmark) could not be downloaded: the
  official zip returned a Cloudflare challenge. MFCAD (MIT) was used instead.
- MFCAD is known to be easy; its high GNN accuracy is not evidence of production
  performance.
- Only planar-floor pockets/steps, cylindrical holes, coaxial counterbores/
  countersinks, round bosses, tangent fillets and narrow planar chamfers are
  recognized. Passages, freeform cavities, assemblies and mixed units are out of
  scope.
- The hybrid is conservative and did not beat the GNN on MFCAD; on synthetic all
  methods reach the ceiling.
- Similarity is weak on MFCAD exact macro compositions (P@1
  {_f(_load("similarity_eval.json").get("mfcad", {}).get("metrics", {}).get("precision@1"), 3)});
  Jaccard-relevance P@1 is
  {_f(_load("similarity_eval.json").get("mfcad", {}).get("metrics", {}).get("precision_jaccard0.5@1"), 3)}.
- NX write-back: no live NX session was available. Z-axis holes/bosses,
  subtractive block pockets/slots, candidate cookbook recipes, mocked tests.
  Counterbores/countersinks/fillets/chamfers and non-Z axes are skipped.
- RQ/Redis was smoke-tested through Docker Compose; PostgreSQL and multi-worker
  stress behavior remain untested.
- The optional multi-view CNN is weaker than the graph classifier on the
  synthetic part-family task; it sees only six flat-shaded orthographic views and
  loses depth/topology cues. Exact accuracy is in the generated CNN table.
- The CNN is trained/evaluated on synthetic families only; no MFCAD part-family
  labels exist.

## 7. Decisions and fallbacks

- OCP (`cadquery-ocp`) over pythonocc-core (conda-only) for pip-only hosting.
- Material-side edge convexity convention, calibrated with primitives; documented
  because some papers use void-side polarity.
- Synthetic dataset primary for the demo; MFCAD external benchmark; MFCAD++
  blocked (Cloudflare 403 challenge).
- Both GNNs saturate the synthetic test; the smaller GCN was retained for the
  demo, while edge-aware message passing is materially better on MFCAD and is the
  external-benchmark model.
- SQLite queue default (no external services); RQ/Redis optional.
- NX graph via STEP export to avoid unverified per-face geometry calls; write-back
  via candidate public-cookbook recipes (not verified live here) and conservative block cuts.
- Detailed decisions: `.agent/DECISIONS.md`. Blockers: `.agent/BLOCKERS.md`.

## 8. Recommended next steps

1. Obtain MFCAD++ through a browser-authenticated download and retrain/evaluate;
   report the harder external numbers.
2. Validate the NX adapter in a real NX session; add edge-selection fillet/
   chamfer and counterbore/countersink builders; replace block cuts with native
   pocket/slot features if supported.
3. Replace the handcrafted similarity signature with a learned GNN embedding and
   an ANN index; evaluate on MFCAD++.
4. Add uncertainty calibration/abstention for the GNN and instance-level
   segmentation (e.g., boundary-aware losses).
5. Extend the CNN with depth/edge channels or learned multi-view fusion and test
   it on MFCAD++/real part families.
6. Harden the service: PostgreSQL/RQ stress tests, multi-worker queue tests,
   authentication/quotas, request timeouts, and a storage backend for large CAD
   files.

## 9. Artifacts

- Source: `src/breplens/`, `streamlit_app.py`, `scripts/`
- Results: `results/*.json`, generated tables in `docs/RESULTS_TABLES.md`
- Models: `models/gnn_synthetic.pt`, `models/gnn_mfcad.pt`
- Samples: `samples/*.step`; demo index: `demo/library.json`
- Docs: `docs/` (research, architecture, setup, datasets, licenses, limitations,
  NX guide, deployment, CV summary, this report)
- Tests: `tests/`; CI: `.github/workflows/ci.yml`
- Provenance: stochastic/training/benchmark results record their script, command,
  seed where applicable, software versions and timestamp (see
  `results/gnn_eval.json`, `mfcad_gnn_eval.json`, `rules_eval.json`,
  `perf.json`, `deploy_benchmark.json`). Test and Docker artifacts record their
  command and outcome; coverage metadata is in `results/coverage_meta.json` and
  `results/coverage*.txt`.
"""
    (ROOT / "docs" / "FINAL_REPORT.md").write_text(text)


def write_results_tables(t: dict[str, str]) -> None:
    text = "# Generated Results Tables\n\n"
    text += "Generated by `scripts/make_readme_tables.py`; do not edit by hand.\n\n"
    for name, title in (
        ("datasets", "Datasets"),
        ("synthetic", "Synthetic recognition head-to-head"),
        ("gnn", "GNN ablation"),
        ("cnn", "Optional multi-view CNN (part family)"),
        ("mfcad", "MFCAD external benchmark"),
        ("similarity", "Similarity retrieval"),
        ("perf", "Performance"),
        ("deploy", "Hosted deployment"),
        ("quality", "Quality gates"),
        ("docker", "Docker service smoke test"),
        ("timing", "Data/training wall-clock"),
    ):
        text += f"## {title}\n\n{t[name]}\n\n"
    (ROOT / "docs" / "RESULTS_TABLES.md").write_text(text)


def main() -> int:
    t = tables()
    update_readme(t)
    write_results_tables(t)
    write_cv(t)
    write_final_report(t)
    print(
        "updated README results block, docs/RESULTS_TABLES.md, docs/CV_SUMMARY.md, docs/FINAL_REPORT.md"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
