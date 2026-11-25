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
