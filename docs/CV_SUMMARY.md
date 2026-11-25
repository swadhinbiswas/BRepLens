# CV Summary — BRepLens

Truthful bullets, each tied to a generated result file.

- Built a CAD-agnostic Python pipeline that converts STEP B-Reps (OpenCascade/OCP)
  into an attributed adjacency graph and recognizes 10 machining feature types
  with rule-based, GNN and hybrid modes, extracting measured parameters into a
  versioned Pydantic/JSON schema.
- Implemented a compact face-classification GNN (gcn:
  18635 parameters, 0.0764 MB)
  with 1.0000 face accuracy on a
  246-part held-out synthetic test set; a separate
  edge-aware MFCAD model reached 0.9986
  accuracy on 448 held-out MFCAD parts;
  a six-view CNN part-family classifier reached
  0.8171
  versus 1.0000
  for the graph classifier on the same split.
- Rule-based recognition reached 1.0000 feature F1 on
  the synthetic test set and exposed its limits on MFCAD
  (0.2180 macro F1); the conservative hybrid preserved
  all rule features and also reached 1.0000
  synthetic feature F1, while on MFCAD only GNN-priority/vote matched the GNN.
- Delivered a FastAPI + queued-worker service with SQLite default storage and an
  optional RQ/Redis mode, plus a Streamlit Community Cloud demo whose measured
  cold start is 1.82 s, first-part time
  0.12 s, peak RSS
  873.2 MB, and inference-only install
  2413 MB.
- Added a WL/geometry signature index with precision@1 of
  1.000 on synthetic
  families and 0.746
  (Jaccard≥0.5) on MFCAD, an NX Open adapter/write-back path with a faithful mock
  (explicitly untested live), 92 passing tests and
  83.0% coverage on core
  geometry/recognition/schema code.
