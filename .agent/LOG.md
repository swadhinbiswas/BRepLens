# BRepLens Log (append-only)

- 2026-11-24 20:16 — Session start. Environment audited: CachyOS, Python 3.14 system (3.11 via uv), RTX 3060 Ti 8GB, 195GB free, 32GB RAM, 20 threads. OpenCode v2.4.1 agent docs verified.
- 2026-11-24 20:20 — Repo skeleton + 14 agent definitions + .agent state files created. Dependency install started in background (PID 211544).
- 2026-11-24 20:55 — Phase 1 research delivered (docs/RESEARCH.md, 751 lines; MFCAD++ URL blocked; NX APIs unverified).
- 2026-11-24 21:05 — Phase 3: OccAdapter + AAG + synthetic generator; convexity calibrated (box/pocket/hole/fillet); 10 sample STEPs.
- 2026-11-24 21:20 — Phase 4a/4b: rules; synthetic 1650 parts; GNN baseline+edge-aware; MFCAD 3000 parts converted (MIT).
- 2026-11-24 21:35 — Evaluations: rules synthetic F1 0.9968, GNN 1.0, hybrid 1.0 (recovers rule misses); MFCAD macro rules 0.218 / GNN 0.9993.
- 2026-11-24 21:40 — Torch 2.11.0+cu128 verified on RTX 3060 Ti; checkpoint 0.47 MB; CPU 0.37 ms/graph.
