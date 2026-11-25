# BRepLens Blockers

## B001 — MFCAD++ download blocked
- Problem: official MFCAD++ zip `https://pure.qub.ac.uk/files/278385243/MFCAD_dataset.zip` returns an HTML Cloudflare challenge (5,578 bytes, HTTP 200 after challenge; direct request 403). Landing page is reachable.
- Impact: cannot use MFCAD++ as the primary external benchmark.
- Fallback applied: use MFCAD (github.com/hducg/MFCAD, MIT, 15,488 STEP + `.face_truth`) as external face-classification benchmark; synthetic CadQuery dataset remains primary for features/parameters. documented in D007 and docs/DATASETS.md.
- Status: applied, resolved for project purposes; MFCAD++ listed as future work.

## B002 — CadQuery install pulls VTK
- Problem: `cadquery 2.8` installs VTK (~200 MB), too heavy for the hosted demo.
- Fallback: CadQuery is dev-only; the hosted app imports OCP directly and never imports CadQuery. requirements.txt excludes cadquery/vtk. Verified by import tracing in the deploy benchmark.
- Status: applied.

## B003 — OCP list iterators
- Problem: OCP 7.9 has no `TopTools_ListIteratorOfListOfShape`.
- Fix: iterate `TopTools_ListOfShape` directly (it implements `__iter__`), verified locally.
- Status: resolved.


## B004 — Reviewer round 1 defects (resolved)
- CI ruff failure in benchmark script: fixed unused noqa/unpacked vars; ruff clean.
- Missing `breplens.cli` entry point: added `src/breplens/cli.py` with recognize/schema commands and a subprocess test.
- Synthetic bracket GT depth bug (outside holes depth 0, pocket hole 5): fixed to entry-to-exit depth; regenerated data; all parameter MAE now 0 on the synthetic test.
- NX "verified recipe" wording contradicted research: relabeled as candidate public-cookbook recipes, unverified live; docs/code now consistently UNTESTED.
- GNN zero-depth stub: recess measurement now returns None (warning) if depth < min_depth.
- Missing README/report generators and RESULTS_TABLES: added `scripts/make_readme_tables.py`, generated all docs from results.
- `is_outer` constant feature and unused smooth threshold: removed is_outer; smooth threshold now used by the classifier.
- Stale .agent files: synchronized.
- Status: resolved; reviewer round 2 requested.
