# Deploying the Streamlit demo

Target: **Streamlit Community Cloud**, CPU-only, no NX, no Redis, no external
database. The app runs recognition in-process for 10 committed sample parts plus
uploads under 25 MiB / 3,000 faces.

## 1. Prerequisites

- A GitHub repository containing this project.
- The committed artifacts:
  - `streamlit_app.py`
  - `requirements.txt` (inference-only, CPU torch)
  - `packages.txt` (system libraries for OCP)
  - `samples/*.step` (10 generated sample parts)
  - `models/gnn_synthetic.pt` (small committed checkpoint)
  - `demo/library.json` (60-part signature library)
  - `src/breplens/` (the package; the app adds `src` to `sys.path`)
- No secrets or `.env` are needed.

`requirements.txt` pins:
`torch==2.11.0+cpu` from the PyTorch CPU index, `torch-geometric`,
`cadquery-ocp`, `numpy`, `networkx`, `pydantic`, `streamlit`, `plotly`, `pandas`,
`trimesh`. It intentionally excludes CadQuery, FastAPI, SQLAlchemy, pytest and
training tooling.

## 2. Deploy

1. Push the repository to GitHub (no remote is configured in this working copy;
   add one and push).
2. Go to https://share.streamlit.io → **New app**.
3. Select the repository, branch, and main file `streamlit_app.py`.
4. Open **Advanced settings**:
   - Python version: **3.12** (verified) or **3.11**; `pyproject.toml` requires
     `>=3.11,<3.13`.
   - Secrets: none.
5. Deploy. The first build installs CPU wheels; it is large and can take a few minutes; the exact
   environment size and install time are recorded in `results/deploy_benchmark.json`.
6. Open the app and select a sample. The first recognition loads the GNN and
   takes longer; subsequent samples are cached.

## 3. Local verification (same dependency set)

```bash
python3.12 -m venv .deploy-venv
.deploy-venv/bin/pip install -r requirements.txt
.deploy-venv/bin/python scripts/deploy_probe.py
.deploy-venv/bin/streamlit run streamlit_app.py
```

`scripts/deploy_probe.py` is the source of `results/deploy_benchmark.json`:
it measures install size/time, AppTest cold start, per-sample latency, peak RSS,
and a headless server start. The README deployment table is generated from that
file.

## 4. Hosted constraints and behavior

- **Memory**: target < 1.5 GiB peak; measured peak is recorded in
  `results/deploy_benchmark.json`. VTK is installed as an OCP dependency but is
  never imported.
- **CPU**: rules ~sub-millisecond; GNN/hybrid low single-digit milliseconds on
  the sample parts; tessellation dominates per-part time.
- **Sleep**: Community Cloud sleeps idle apps; first request after sleep pays
  model/OCP import and graph caching again.
- **Uploads**: `.step`/`.stp`, ISO-10303-21 magic checked, 25 MiB, 3,000 faces.
  Invalid/empty files produce a clear error; OCP exceptions are caught.
- **NX is not available** in the hosted demo; NX write-back is a separate
  desktop workflow (`docs/NX_GUIDE.md`, UNTESTED live).
- **FastAPI/queue** is not part of the hosted demo; it ships via Docker
  (`docker-compose.yml`).

## 5. Troubleshooting

| Symptom | Cause/fix |
| --- | --- |
| `ImportError: libGL.so.1` | `packages.txt` missing `libgl1` / `libglib2.0-0`; redeploy after adding them. |
| Model load warning, rules fallback | `models/gnn_synthetic.pt` missing or torch/PyG install failed; check the build log. |
| `ModuleNotFoundError: breplens` | `streamlit_app.py` must remain at repo root (it inserts `src` on `sys.path`). |
| App killed for memory | Check `results/deploy_benchmark.json`; reduce samples/limits or update the pinned wheels. |
| Upload rejected | File is >25 MiB, not STEP, or has no faces; export AP214/AP242. |
| Similarity empty | `demo/library.json` missing; rebuild with `python scripts/evaluate_similarity.py`. |

## 6. Rebuilding committed artifacts

```bash
python scripts/generate_synthetic.py --per-family 150 --workers 16
python scripts/evaluate_similarity.py          # rewrites demo/library.json
python scripts/train_gnn.py --dataset synthetic --epochs 35
python scripts/deploy_probe.py                 # rewrites results/deploy_benchmark.json
```

After changing model/data, commit the updated `models/` and `demo/` files and
re-run the deploy probe.
