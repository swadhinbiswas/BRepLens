# BRepLens — Final Report

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
  parameter table, JSON download, rules/GNN/hybrid toggle and similarity search.

## 2. Measured results

| Method | Feature P | Feature R | Feature F1 | Exact part rate | Face acc. | Face macro F1 | ms/part |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rules | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.18 |
| gnn | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.63 |
| hybrid | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 2.60 |

*246 held-out synthetic parts; same split for all methods; latency is recognition only, CPU.*

| Dataset | Architecture | Parameters | Test face acc. | Test macro F1 | Train s | CPU ms/graph |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| synthetic | gcn | 18635 | 1.0000 | 1.0000 | 2.35 | 0.54 |
| synthetic | edge_gnn | 120011 | 1.0000 | 1.0000 | 1.20 | — |
| MFCAD | gcn | 18960 | 0.9381 | 0.8909 | 4.90 | — |
| MFCAD | edge_gnn | 120336 | 0.9986 | 0.9982 | 4.50 | 0.43 |

*GCN = baseline without edge attributes; `edge_gnn` = edge-aware message passing. Early stopping on the validation split only.*

| Method (8 macro classes) | Accuracy | Macro F1 | Faces |
| --- | ---: | ---: | ---: |
| rules | 0.4034 | 0.2180 | 10146 |
| gnn | 0.9989 | 0.9983 | 10146 |
| hybrid_rule_priority | 0.7422 | 0.6366 | 10146 |
| hybrid_gnn_priority | 0.9989 | 0.9983 | 10146 |
| hybrid_vote | 0.9989 | 0.9983 | 10146 |

*MFCAD test split: 448 parts. Full 16-class GNN: accuracy 0.9986, macro F1 0.9982. Rules were designed and tuned on the synthetic families and were not retuned on MFCAD.*

| Dataset | Library | Queries | P@1 | P@5 | P@10 | P@1 (Jaccard≥0.5) | Query ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Synthetic | 1404 | 246 | 1.000 | 1.000 | 1.000 | 1.000 | 0.33 |
| MFCAD | 2552 | 448 | 0.150 | 0.133 | 0.118 | 0.746 | 0.85 |

*Relevance = same family label; MFCAD families are coarse macro compositions. The Jaccard column treats compositions sharing at least half their feature set as relevant.*

### Data and training wall-clock

| Pipeline step | Wall time | Source |
| --- | ---: | --- |
| Synthetic generation (1,650 parts, 16 workers) | 5.48 s | `data/synthetic/manifest.json` |
| MFCAD conversion (3,000 parts, 16 workers) | 10.39 s | `data/mfcad/manifest.json` |
| Synthetic GNN training (gcn) | 2.35 s | `results/gnn_eval.json` |
| MFCAD GNN training (edge_gnn) | 4.50 s | `results/mfcad_gnn_eval.json` |


## 3. Hosted demo benchmarks

| Metric | Value |
| --- | ---: |
| Environment size | 2413.2 MB |
| Fresh pip install | 147 s |
| AppTest cold start | 0.53 s |
| First part incl. model load | 1.32 s |
| Peak RSS (app probe) | 854.8 MB |
| Peak RSS (pipeline probe) | 856.6 MB |
| Headless server start | 0.41 s |
| Rules latency | 0.17 ms |
| GNN latency | 1.24 ms |
| Hybrid latency | 1.62 ms |
| Tessellation latency | 7.07 ms |

*Measured on the inference-only venv, CPU, 10 committed sample parts; raw data in `results/deploy_benchmark.json`.*

The target was < 1.5 GiB peak; measured peak is
854.8 MB for the app probe.
The environment is large because `cadquery-ocp` declares VTK; VTK is never
imported by the app.

## 4. Service verification

| Queue | API health | End-to-end s | Job | Features | Load ms | Recognition ms |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| SQLite worker | ok | 1.90 | succeeded | 8 | 31.8 | 4.35 |
| RQ + Redis worker | ok | 2.04 | succeeded | 8 | 20.7 | 5.30 |

The default stack was built and started with `docker compose up --build`; API
health, hybrid recognition through the worker, feature retrieval and similarity
were exercised. Integration tests additionally cover validation, failure paths,
GNN mode, status polling and size limits.

## 5. Quality gates

| Check | Result |
| --- | ---: |
| Tests passed | 80 |
| Tests failed | 0 |
| Test wall time | 6.21 s |
| Full line coverage | 74.01% |
| Core geometry/recognition/schema coverage | 82.20% |


CI runs ruff, mypy and pytest on CPU; the core coverage gate is 80%. The GitHub
Actions workflow was not executed on a remote because this repository has no
configured remote; the identical commands were run locally on CPU and are green.

## 6. Known limitations and failure cases

- Synthetic rules saturate at 1.0 after tuning on the generator; this is not
  real-world accuracy. MFCAD exposes the rule limits (macro F1
  0.2180).
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
  0.150);
  Jaccard-relevance P@1 is
  0.746.
- NX write-back: no live NX session was available. Z-axis holes/bosses,
  subtractive block pockets/slots, candidate cookbook recipes, mocked tests.
  Counterbores/countersinks/fillets/chamfers and non-Z axes are skipped.
- RQ/Redis and PostgreSQL paths are delivered but were not stress-tested here.
- No multi-view CNN (optional Phase 9) was implemented.

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
  via verified public recipes and conservative block cuts.
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
5. Optional Phase 9: multi-view CNN part-family classifier with a fair comparison.
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
- Provenance: every results JSON records its command, seed, versions and
  timestamp; `results/tests.json`, `results/coverage*.json`,
  `results/docker_service_test.json`.
