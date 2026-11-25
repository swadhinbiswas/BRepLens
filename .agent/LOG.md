# BRepLens Log (append-only)

- 2026-11-24 20:16 — Session start. Environment audited: CachyOS, Python 3.14 system (3.11 via uv), RTX 3060 Ti 8GB, 195GB free, 32GB RAM, 20 threads. OpenCode v2.4.1 agent docs verified.
- 2026-11-24 20:20 — Repo skeleton + 14 agent definitions + .agent state files created. Dependency install started in background (PID 211544).
- 2026-11-24 20:55 — Phase 1 research delivered (docs/RESEARCH.md, 751 lines; MFCAD++ URL blocked; NX APIs unverified).
- 2026-11-24 21:05 — Phase 3: OccAdapter + AAG + synthetic generator; convexity calibrated (box/pocket/hole/fillet); 10 sample STEPs.
- 2026-11-24 21:20 — Phase 4a/4b: rules; synthetic 1650 parts; GNN baseline+edge-aware; MFCAD 3000 parts converted (MIT).
- 2026-11-24 21:35 — Evaluations: rules synthetic F1 0.9968, GNN 1.0, hybrid 1.0 (recovers rule misses); MFCAD macro rules 0.218 / GNN 0.9993.
- 2026-11-24 21:40 — Torch 2.11.0+cu128 verified on RTX 3060 Ti; checkpoint 0.47 MB; CPU 0.37 ms/graph.
- 2026-11-24 21:15 — Phase 1-8 complete; Docker SQLite and RQ/Redis stacks smoke-tested; deploy benchmark recorded; reviewer round 1 rejected with concrete defects.
- 2026-11-24 21:30 — Fixed all round-1 blockers (ruff, CLI, bracket GT depth, NX wording, zero-depth stub, is_outer/smooth threshold, generated docs); 80 tests pass, ruff/mypy clean, core coverage 82.2%.
- 2026-11-24 21:35 — Round-2 audit requested.
- 2026-11-24 21:22 — Round-2 review found stale generated docs, a remaining NX wording regression, a false provenance claim, a rules latency timer bug (fillet t0 shadowing), and missing env_audit/hybrid_eval files.
- 2026-11-24 21:30 — Fixed timer bug (rules latency now ~0.17 ms mean), GNN recess now returns None (tested), regenerated final docs, added env_audit.json/hybrid_eval.json/coverage_meta.json, regenerated datasets with full-run timings, synchronized state. 83 tests pass; ruff/mypy clean.
- 2026-11-24 21:35 — Final independent gate APPROVED (.agent/REVIEW_FINAL.md); round-3's only note (overbroad SETUP provenance sentence) fixed. v1.0.0 prepared.
- 2026-11-25 — Phase 9 implemented: six-view software renderer, 29k-param CNN vs 124k-param graph family classifier on identical 1650-part split; CNN 0.8171 acc / 0.8119 F1, graph 1.0; 92 tests; docs/CV generated; v1.1.0 review underway.
- 2026-11-25 — Phase 9 APPROVED (.agent/REVIEW_phase9.md); fixed docstring/pillow pin; regenerated docs, quality gates green (92 tests, 83.04% core); v1.1.0 tagged.
- 2026-11-25 06:33 — Streamlit app redesigned: hero, KPI cards, Overview/Features/Compare/Similar/JSON tabs, feature filters, CSV/JSON downloads, sample landing cards, family CNN caption; all 10 samples exercised headlessly; em/en dash count zero; deploy probe 878.1 MB.
