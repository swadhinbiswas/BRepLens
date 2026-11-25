# BRepLens State

## Current
- Phase: 4b GNN/evaluation complete; Phase 2 architecture, 4c NX, 5 similarity, 6 service, 7 perf, 8 deploy pending.
- Last completed: synthetic (1650 parts) + MFCAD (3000 parts) conversion; rules/GNN/hybrid evaluated.
  - rules synthetic test: feature F1 0.9968, face acc 0.9984
  - GNN synthetic test: feature F1 1.0, face acc 1.0
  - hybrid synthetic test: feature F1 1.0 (recovers rule misses)
  - MFCAD external macro face: rules 0.403 acc / 0.218 F1; GNN 0.9992 / 0.9993
- Next action: commit; write architecture docs; implement similarity + service + app + NX; tests and reviewer gates.
- Session rule: re-read TASKS.md/STATE.md at session start; keep them current.

## Environment (audited 2026-11-24)
- OS: CachyOS Linux (Arch-based), kernel rolling
- CPU: Intel i5-13600K, 20 threads; RAM 32 GiB; disk 195 GiB free on /home
- GPU: NVIDIA RTX 3060 Ti 8 GiB, driver 620.73.04, CUDA UMD 13.5
- Python: system 3.14.10; project venv uses uv-managed CPython 3.11.18
- Tooling: uv 0.13.9, git, docker, gcc present; conda/mamba absent
- Project venv: .venv (Python 3.11)

## Background jobs
- (none running)

## Results files (source of truth for docs)
- results/rules_eval.json — rules on synthetic test (51 parts); regenerated for 246 later
- results/gnn_eval.json — synthetic GNN training/eval
- results/mfcad_gnn_eval.json — MFCAD external GNN
- results/mfcad_head_to_head.json — rules/GNN/hybrid macro face comparison
- results/recognition_comparison.json — rules/GNN/hybrid on synthetic test (246 parts)
