# BRepLens Decisions

Format: ID — decision — options considered — reason — date

## D001 — Agent definition format
OpenCode V2 documents Markdown agents in `.opencode/agents/<name>.md` with YAML frontmatter (`description`, `mode`, `model`, `permissions`, `steps`); primary agents launch them through the `subagent` tool (V1 called it `task`). Verified at https://opencode.ai/docs/agents on 2026-11-24 with opencode v2.4.1 installed.
Options: V1 `.opencode/agent/` singular dir + `permission`/`tools` fields; V2 plural dir + `permissions` list. Chosen V2.
Reason: current installed version is v2.4.1 and V1 frontmatter is deprecated.

## D002 — Python version
CPython 3.11.18 (uv-managed) inside `.venv`.
Options: 3.14 system (too new for OCP/CadQuery/torch wheels), 3.12 (viable), 3.11 (LTS-ish, best wheel coverage for cadquery-ocp and PyG).
Reason: verified wheel availability and reproducibility; 3.11 satisfies "3.11+".

## D003 — Geometry kernel
`cadquery-ocp` (OCP) as the STEP/B-Rep engine; CadQuery only for synthetic part generation (dev dependency).
Options: pythonocc-core (conda-only; no conda installed), OCP (pip wheels), CadQuery-on-OCP.
Reason: pip-only compatibility for dev + hosted deployment; verified installation in Phase 0.

## D004 — GNN stack
PyTorch + PyTorch Geometric if the verified install works on this GPU; otherwise pure-PyTorch message passing (documented fallback).
Reason: mission requirement and CPU-hosted inference size; decision finalized after install verification.

## D005 — Queue
SQLite-backed worker as the always-available default, with an RQ+Redis mode for docker-compose. Core service never requires Redis to run tests.
Reason: CI and Streamlit-hosted environment must work without external services; Redis path still delivered.

## D006 — Edge convexity convention
Convex = material-side interior dihedral < 180°, determined by the sign of (n1 x n2)·t with t oriented by face A's wire; smooth when adjacent normals differ by < 12°. Calibrated against a box (all convex), a drilled through hole (rims convex), a blind hole (wall-floor concave), a pocket (floor-wall concave) and tangent fillets (smooth).
Options: void-side polarity (many papers call hole rims concave); material-side polarity.
Reason: material-side is testable from B-Rep alone with a single convention, and is documented in the schema and tests. AAGNet/MFCAD labels are per-face, so no dataset conflict.

## D007 — Primary vs external dataset
Synthetic CadQuery dataset (1650 parts, 11 classes, construction-derived labels) is the primary train/test set for the hosted demo because it covers all 10 BRepLens feature types with parameters. MFCAD (MIT, 15,488 STEP; 3000 converted) is the external face-classification benchmark.
Options: MFCAD++ (1.5 GB, CC BY) was the primary target; direct download was blocked by Cloudflare (HTTP 403 challenge) on 2026-11-24.
Reason: mission fallback requires documenting an unavailable dataset and using a verified alternative; MFCAD is officially available and independently labeled. MFCAD++ remains documented as a future external benchmark.

## D008 — GNN
Edge-aware message passing (`edge_gnn`: MLP on [h_src,h_dst,edge_attr], GRU update, residual+LayerNorm) with a GCN baseline; hidden 64, 3 layers, 15% dropout, class-weighted CE, early stopping on val macro-F1.
Measured (after feature cleanup): synthetic face acc 1.000 for both; MFCAD face acc 0.9986 (edge_gnn) vs 0.9381 (GCN). Both saturate synthetic, so the smaller GCN (18.6k params, 0.08 MB) is the demo checkpoint; edge_gnn (120k params, 0.47 MB) is the MFCAD external model. CPU inference is sub-millisecond per graph for both.
Options: GCN, GAT, GraphSAGE, AAGNet-scale model; BRepNet rejected (license/complexity).
Reason: best accuracy per parameter on this hardware and small enough for free hosting.
