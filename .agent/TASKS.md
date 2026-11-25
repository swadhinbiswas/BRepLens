# BRepLens Master Tasks

Status: `todo` | `in-progress` | `done` | `blocked`. Owner = agent role.
A phase is done only when (1) tests pass, (2) scripts produced real results files,
(3) reviewer approved, (4) STATE/TASKS updated and committed.

## Phase 0 — Bootstrap (devops)
- [x] 0.1 Environment audit (OS, Python, uv, CUDA, disk) -> docs/SETUP.md + results/env_audit.json
- [x] 0.2 Repo skeleton (src, tests, docs, scripts, docker, results, .agent, samples)
- [x] 0.3 pyproject.toml (Python 3.11+, pinned deps), ruff + mypy config
- [x] 0.4 pre-commit config
- [x] 0.5 GitHub Actions CI (ruff, mypy, pytest, CPU-only)
- [x] 0.6 Agent definitions (.opencode/agents/*.md) + .agent files
- [x] 0.7 Baseline commit + tag scaffold

## Phase 1 — Research (researcher)
- [x] 1.1 B-Rep -> AAG + convexity classification, classical feature recognition limits
- [x] 1.2 Learning approaches: MFCAD/MFCAD++, UV-Net, BRepNet, AAGNet; input/arch/accuracy; what to implement
- [x] 1.3 Datasets: verified links, size, license, labels, format
- [x] 1.4 NX Open Python API (traversal, queries, hole/pocket/slot creation, journal vs external)
- [x] 1.5 pythonocc vs OCP vs CadQuery for pip-only hosting
- [x] 1.6 Graph similarity options
- [x] 1.7 Decisions section; reviewer gate

## Phase 2 — Architecture (architect)
- [x] 2.1 docs/ARCHITECTURE.md with Mermaid + module boundaries
- [x] 2.2 CadAdapter ABC + neutral IR
- [x] 2.3 Versioned feature JSON schema (Pydantic + exported JSON Schema)
- [x] 2.4 Error handling strategy
- [x] 2.5 Reviewer gate (consistent with research)

## Phase 3 — Geometry foundation (geometry-engineer, qa-engineer)
- [x] 3.1 OccAdapter: load STEP, faces/edges, classification, bbox, tessellation
- [x] 3.2 Surface classification (plane/cylinder/cone/torus/sphere/other)
- [x] 3.3 Edge convexity (convex/concave/smooth) calibrated on real primitives
- [x] 3.4 AAG build + NetworkX + PyG/tensor export
- [x] 3.5 Synthetic CadQuery generator with ground truth (features + per-face labels)
- [x] 3.6 AAG visualization script
- [x] 3.7 Tests (unit + integration), reviewer gate

## Phase 4a — Rule-based recognition (recognizer-engineer)
- [x] 4a.1 through/blind hole, counterbore, countersink
- [x] 4a.2 pocket, slot, step, boss
- [x] 4a.3 fillet, chamfer
- [x] 4a.4 Parameter extraction (diameter, depth, radius, axis, position, dims)
- [x] 4a.5 Tolerance/unit handling, intersecting features, negative tests
- [x] 4a.6 docs/LIMITATIONS.md; measured results -> results/rules_eval.json

## Phase 4b — Dataset + GNN (data-engineer, ml-engineer)
- [x] 4b.1 Verify/download MFCAD++ (or fallback synthetic); data card
- [x] 4b.2 STEP -> cached graph conversion (multiprocessing, resumable)
- [x] 4b.3 Baseline GNN trained + sanity-checked
- [x] 4b.4 Stronger model chosen in research
- [x] 4b.5 Per-class P/R/F1, confusion matrix, face accuracy, timings -> results/gnn_eval.json
- [x] 4b.6 Head-to-head vs rules on same test set + failure analysis
- [x] 4b.7 Hybrid strategy + measured results -> results/hybrid_eval.json
- [x] 4b.8 Export small CPU weights (models/)

## Phase 4c — NX Open (nx-engineer)
- [x] 4c.1 NxOpenAdapter (traversal, surface/edge queries)
- [x] 4c.2 Parametric write-back (holes, pockets, slots)
- [x] 4c.3 Faithful NXOpen mock + unit tests
- [x] 4c.4 Journal entry points + docs/NX_GUIDE.md
- [x] 4c.5 "UNTESTED against live NX" labels everywhere

## Phase 5 — Similarity search (similarity-engineer)
- [x] 5.1 Signatures (face histograms, dims, WL kernel)
- [x] 5.2 Index + search API
- [x] 5.3 Retrieval evaluation precision@k -> results/similarity_eval.json

## Phase 6 — Service layer (backend-engineer)
- [x] 6.1 FastAPI endpoints (upload, submit, status, features, similarity)
- [x] 6.2 Job queue (RQ+Redis) + SQLite fallback worker
- [x] 6.3 SQLite default, PostgreSQL option, size limits, structured logging
- [x] 6.4 Dockerfile + docker-compose (api, worker, redis)
- [x] 6.5 Integration tests + measured latencies

## Phase 7 — Performance (perf-engineer)
- [x] 7.1 Profile pipeline (cProfile/py-spy)
- [x] 7.2 Optimize hot paths (vectorization/spatial index/cache)
- [x] 7.3 Before/after benchmark -> results/perf.json

## Phase 8 — Streamlit app + hosted deployment (frontend-engineer, devops)
- [x] 8.1 streamlit_app.py (upload, samples, 3D color, table, JSON, mode toggle, similarity)
- [x] 8.2 requirements.txt inference-only pinned; packages.txt only if needed
- [x] 8.3 Fresh-venv pip-only install verification
- [x] 8.4 Streamlit headless E2E on samples; install size, cold start, latency, peak memory -> results/deploy_benchmark.json
- [x] 8.5 docs/DEPLOY_STREAMLIT.md
- [x] 8.6 Reviewer gate

## Phase 9 — Optional CNN (only if all else passes)
- [ ] 9.1 Multi-view render + small CNN part-family classification, fair comparison
- [x] 9.2 Or documented as future work

## Phase 10 — Finalization (docs-writer, qa-engineer, reviewer)
- [x] 10.1 README (overview, Mermaid, 3-command quickstart, generated results tables, limitations, structure, NX, repro, deploy)
- [x] 10.2 docs/CV_SUMMARY.md (truthful measured bullets)
- [x] 10.3 docs/LIMITATIONS.md, docs/LICENSES.md
- [ ] 10.4 Full audit: tests, coverage, CI, results provenance, secrets, licenses
- [x] 10.5 docs/FINAL_REPORT.md
- [ ] 10.6 Tag v1.0.0


## Notes
- Phase 9 (optional CNN): not implemented; documented as future work in LIMITATIONS/FINAL_REPORT.
- Phase 10.4 full audit: reviewer round 2 pending.
- Phase 10.6 tag: after round 2 approval.
