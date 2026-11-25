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
  - `models/gnn_synthetic.pt` and `models/family_cnn.pt` (small committed checkpoints)
  - `demo/library.json` (60-part signature library)
  - `src/breplens/` (the package; the app adds `src` to `sys.path`)
- No secrets or `.env` are needed.

`requirements.txt` pins:
`torch==2.11.0+cpu` from the PyTorch CPU index, `torch-geometric`,
`cadquery-ocp`, `numpy`, `networkx`, `pydantic`, `streamlit`, `plotly`, `pandas`,
`trimesh`. It intentionally excludes CadQuery, FastAPI, SQLAlchemy, pytest and
training tooling.

## 2. Deploy this repository

The public repository is:

```text
https://github.com/swadhinbiswas/BRepLens
```

1. Open https://share.streamlit.io and sign in with GitHub.
2. Choose **New app**.
3. Select repository `swadhinbiswas/BRepLens`, branch `main`, and main file
   `streamlit_app.py`.
4. Open **Advanced settings**:
   - Python version: **3.12** (verified with CPython 3.12.16) or **3.14**
     (verified with CPython 3.14.10 and wheels only). The requirements file is
     compatible with both.
   - Secrets: none.
5. Choose a subdomain, then click **Deploy**.
6. Watch the build log. The first build installs the CPU wheels listed in
   `requirements.txt` and the system packages in `packages.txt`. It is large and
   takes a few minutes; the exact timings are in `results/deploy_benchmark.json`.
7. Open the app and load a sample. The first recognition loads the models, so it
   is slower than later requests.

If an earlier build hangs while preparing NumPy metadata, it is using the old
`numpy==2.2.6` pin on Python 3.14. Update to `numpy==2.3.5`, stop the old build,
push, and reboot the app. NumPy 2.3.5 has CPython 3.14 wheels; no compiler is
needed.

After the first successful deploy, every push to `main` triggers a rebuild. Use
the app menu to reboot, clear the cache, or view logs. The app has no database,
no secrets, and no persistent upload storage.

### What Streamlit Cloud hosts, and what it does not

Streamlit Community Cloud hosts **only the demo process**. It does not host the
FastAPI service, the worker, Redis, or PostgreSQL. The demo is standalone and
does not call the API.

For the full stack, run Docker Compose on a VM, VPS, or container platform that
supports a long running container:

```bash
git clone https://github.com/swadhinbiswas/BRepLens.git
cd BRepLens
docker compose up -d --build
```

Then put a reverse proxy with HTTPS in front of port 8000, or use the platform's
managed TLS. Keep the Streamlit demo on Community Cloud and the API on the VM if
you want the simplest split.

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

- **Memory**: the local hosted-equivalent probe peaks below 1 GiB; exact values
  are in `results/deploy_benchmark.json`. Community Cloud's limits can change, so
  check the dashboard if the app is killed. VTK is installed as an OCP dependency
  but is never imported by the app.
- **Storage**: uploads are written to temporary files and are not persisted
  across restarts. The sample library and model weights are committed files.
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
