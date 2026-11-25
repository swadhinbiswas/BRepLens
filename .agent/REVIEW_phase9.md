# Phase 9 Review — optional multi-view CNN

Verdict: **APPROVED**

All exit criteria were verified by running the commands and reading the produced result files.

## Gates (run just now)
- `ruff check src tests scripts streamlit_app.py` → All checks passed (exit 0).
- `mypy src/breplens` → Success: no issues found in 31 source files (exit 0).
- `pytest -o addopts='' -q` → 92 passed, 1 warning (exit 0), matching `results/tests.json` (`passed: 92`, `failed: 0`, generated 2026-11-25T06:19:34Z).
- Coverage totals recomputed from `results/coverage_core.json` = 83.0352% (1371 statements) and `results/coverage.json` = 71.7911%; README lines 276-280 state 92 tests, 71.79% full, 83.04% core. Trace OK.

## 1. Renderer (`src/breplens/ml/views.py`)
- Deterministic: pure numpy + Pillow painter's algorithm, fixed light, fixed view directions; `tests/unit/test_views.py::test_render_is_deterministic` asserts byte equality and passes.
- No VTK/OpenGL/display: only numpy, PIL, schema; no xvfb/OSMesa import. Streamlit AppTest runs it headless.
- Empty meshes: `render_views` returns zero arrays for empty vertices/triangles (views.py:57-58); `test_render_empty_mesh` passes.
- Tests exist for shape/content, determinism, empty mesh, PNG helper.

## 2. CNN vs graph comparison (`scripts/train_family_models.py`, `src/breplens/ml/family.py`)
- Same data/split/seed: both loaders walk the same `data/synthetic/manifest.json`; split counts are train 1156 / val 248 / test 246 for source manifest, render manifest, and graph files. `set_seed(42)` is called before both trainers and each trainer reseeds.
- No label leakage: images contain renders only; normalization statistics are computed from the train split only (family.py:80-84) and stored in checkpoint metadata. Early stopping/best-state selection uses the validation split only; the test split is evaluated once after reloading the best state.
- `results/cnn_eval.json`: CNN accuracy 0.8171, macro F1 0.8119, 29,115 params, 0.1204 MB, 0.534 ms/part; graph 1.0/1.0, 124,107 params. README lines 213-214 match exactly (0.8171/0.8119/29115/6.15/0.53/0.120 and 1.0000/1.0000/124107/1.61/0.56/0.484); `docs/RESULTS_TABLES.md:38` matches. `docs/FINAL_REPORT.md:150-153` honestly reports the CNN as weaker and synthetic-only. CNN losing to the graph model is reported, not hidden.

## 3. Streamlit + deployment
- `streamlit_app.py:342-362` loads `models/family_cnn.pt`, reads `metadata.image_norm.{mean,std}`, and normalizes the six rendered views with those checkpoint values before softmax.
- Both AppTest tests pass; a manual AppTest hybrid run produced no exception and the caption `Part family (multi-view CNN): **pocket** · 35% confidence`.
- `requirements.txt` remains inference-only (CPU torch, torch-geometric, OCP, numpy, networkx, pydantic, streamlit, plotly, pandas, trimesh); no training-only deps.
- `results/deploy_benchmark.json` was regenerated 2026-11-25T06:18:46Z, after the 06:15 app change; cold start 1.825 s, peak RSS 873.2 MB, limits 1500 MB — reflected in `docs/DEPLOY_STREAMLIT.md`/STATE.

## 4. Docs / state
- `python scripts/make_readme_tables.py` leaves the working tree clean (docs stable; only the untracked review file differs).
- No stale "CNN not implemented"/"future work" claims. `docs/LIMITATIONS.md:131-139` correctly states the CNN's lack of depth/edges/topology, synthetic-only training, and that it does not drive recognition. `docs/LIMITATIONS.md:141-143` places depth/edge fusion and learned multi-view aggregation under "Not implemented" (future work), not the CNN itself.
- `.agent/TASKS.md` Phase 9 marked complete; `.agent/STATE.md` records Phase 9 numbers, the 92-test/coverage state, and next action (Phase 9 review, tag v1.1.0).

## Non-blocking notes (not grounds for rejection)
1. `src/breplens/ml/family.py:3-4` module docstring still says "three orthographic shaded views (isometric/top/front)", inconsistent with the implemented six views (±X/±Y/±Z). Code and user docs are correct; fix the docstring for consistency.
2. `src/breplens/ml/views.py` imports Pillow directly, but `requirements.txt` does not list `Pillow` explicitly; it is currently satisfied transitively by `streamlit==1.67.0`. Declaring it directly would harden the inference-only environment.
3. `docs/FINAL_REPORT.md` only cross-references the generated CNN table rather than embedding the numbers; the exact numbers live in README and `docs/RESULTS_TABLES.md` and match `results/cnn_eval.json`.

## Evidence of committed artifacts
- `git ls-files` confirms both `models/family_cnn.pt` and `models/family_gnn.pt` are tracked; the CNN checkpoint metadata includes seed 42, test accuracy, and image normalization statistics.
