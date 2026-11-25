# BRepLens State

## Current
- Phase: 9 optional CNN implemented and APPROVED; README, logo and project tags finalized for v1.2.0.
- Last completed: rules 1.0 feature F1 (246 synthetic test parts); GNN demo checkpoint (GCN,
  18,635 params, 1.0 face accuracy synthetic); edge-aware MFCAD model 120,336 params,
  0.9986 face accuracy; hybrid 1.0 on synthetic; MFCAD macro: rules 0.218, GNN 0.9983 F1.
- Similarity: synthetic P@1 1.000; MFCAD exact P@1 0.150, Jaccard≥0.5 P@1 0.746.
- Phase 9 CNN: CNN test acc 0.8171 / macro F1 0.8119 vs graph family 1.0; 29k CNN params;
  6 views 96x96; render cache 1650 parts in 3.02 s.
- Hosted: peak RSS 878.1 MB, cold start 1.815 s, first part 0.148 s, GNN 1.17 ms, hybrid 1.70 ms.
- Service: both SQLite and RQ/Redis Docker stacks smoke-tested end-to-end; 92 tests pass;
  coverage 71.79% full / 83.04% core; ruff + mypy clean.
- Next action: none; v1.2.0 handoff. README rewritten, logo assets added, CI pipeline removed by request, local quality gate retained.

## Environment (audited 2026-11-24)
- OS: CachyOS Linux (Arch-based), kernel rolling
- CPU: Intel i5-13600K, 20 threads; RAM 32 GiB; disk 195 GiB free on /home
- GPU: NVIDIA RTX 3060 Ti 8 GiB, driver 620.73.04, CUDA UMD 13.5
- Python: dev venv 3.11.18; deployment venv 3.12.16 (pip-only)
- Tooling: uv 0.13.9, git, Docker, gcc; conda/mamba absent
- PyTorch 2.11.0+cu128 (dev) / 2.11.0+cpu (deploy); PyG 2.8.1; OCP 7.9.3.1.1

## Background jobs
- (none)

## Results files (source of truth for docs)
- results/rules_eval.json, recognition_comparison.json
- results/gnn_eval.json, mfcad_gnn_eval.json, mfcad_head_to_head.json
- results/similarity_eval.json, perf.json, deploy_benchmark.json
- results/tests.json, coverage.json, coverage_core.json
- results/docker_service_test.json, docker_service_test_rq.json
- data/synthetic/manifest.json, data/mfcad/manifest.json
