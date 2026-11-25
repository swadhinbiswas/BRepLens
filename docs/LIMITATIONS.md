# Limitations and Failure Cases

Honest scope of the v1.0.0 implementation. Numbers referenced here live in
`results/*.json` and the generated tables in `docs/RESULTS_TABLES.md`.

## Recognition: rules

- **Synthetic ceiling is not real-world accuracy.** Rules are designed against
  the same 11 synthetic families used for evaluation and reach 1.0 feature F1
  there. On MFCAD (external, macro classes) rules are far weaker; see
  `results/mfcad_head_to_head.json`. Synthetic numbers must not be quoted as
  general machining-feature accuracy.
- Feature vocabulary is fixed. Polygonal **passages**, triangular/6-sided
  pockets, arbitrary slots and non-planar cavities are not recognized. MFCAD's
  passage classes are deliberately mapped to "unknown" and counted as misses.
- Holes:
  - only cylindrical barrels with **material-outside** surfaces;
  - a fillet or chamfer blended into the hole rim introduces smooth edges and
    the cylinder is excluded from hole candidates;
  - blind holes with a **conical drill tip** are not fully parameterized (a flat
    cap is recognized; depth is the cylindrical length);
  - hole axes are reported at the opening end, and the "top" is chosen by the
    canonical axis sign; there is no machining-setup inference;
  - counterbores/countersinks require a coaxial cone or shoulder and a single
    axis; eccentric or multi-step bores beyond two radii are only partially
    merged.
- Pockets/slots/steps require a planar floor. Fully curved floors, drafted walls
  beyond the normal test, and undercut cavities are not handled. A slot is
  identified when a cylindrical wall radius matches half the footprint width;
  filleted pocket corners can therefore be confused with slot ends.
- Boss recognition requires an external cylinder, a small top cap and a large
  base plane. Bosses with freeform or blended tops are skipped.
- Fillets require at least two smooth (tangent) boundary edges. Edge blends that
  end in non-tangent conditions, variable-radius blends and face blends are not
  recognized.
- Chamfers are planar, narrow (≤30% of the longest bbox extent), non-axis-aligned
  strips with convex long edges and ≤2% of total surface area. Round/concave
  chamfers, chamfers on curved edges and very large chamfers are missed. A thin
  non-axis-aligned rib can be a false positive.
- Intersecting features are handled only when the face-level evidence survives:
  the included bracket has a hole opening into a pocket floor. Arbitrary
  feature-feature occlusion/merging is not solved.
- Units: STEP import forces millimetres. Mixed-unit assemblies and unitless
  exports are not detected.
- Assemblies are flattened by the STEP reader; components/instances are not
  modelled separately. The demo caps parts at 3,000 faces.

## Recognition: GNN

- The deployed GNN classifies **faces**, not instances. Feature instances are
  formed by connected components of predicted faces and then measured; a single
  misclassified boundary face can split or merge an instance.
- The demo model is trained on the synthetic 11-class dataset. It inherits the
  synthetic generator's biases (axis-aligned, clean, single solid).
- MFCAD has a **separate 16-class checkpoint** for the external benchmark. It is
  not the demo model and MFCAD contains no holes/fillets.
- Confidence is the mean softmax over a component, not a calibrated probability.
  No temperature scaling or conformal calibration was performed.
- Face accuracy on MFCAD is very high, but MFCAD is an easy, CAD-generated
  benchmark; this does not establish performance on real production parts or
  MFCAD++ (download blocked, see `docs/DATASETS.md`).
- Training used a single seed and a small model (hidden 64, 3 layers). No
  hyperparameter sweep or ensembling is reported.

## Hybrid

- The hybrid is conservative: rules first, then non-overlapping high-confidence
  GNN instances; it never deletes a rule feature. It cannot fix a wrong rule
  feature.
- On the synthetic test set all three methods saturate at 1.0, so the hybrid
  provides no measured gain there. On MFCAD, rule-priority hybrid is worse than
  the GNN; GNN-priority/vote matches the GNN. The hybrid's value is
  interpretable rules plus learned recovery, not a proven accuracy win.
- Thresholds (`gnn_min_confidence=0.6`, overlap 50%) were chosen conservatively
  and not tuned on the test split.

## Similarity

- Signature retrieval is strong on synthetic families (precision@1 1.0) but weak
  on MFCAD when relevance is the exact macro-composition string (precision@1
  ~0.15); with Jaccard ≥ 0.5 family overlap it rises to ~0.75. MFCAD parts are
  visually similar stock blocks and the handcrafted signature has limited
  discriminative power there.
- The index is brute-force cosine; it is exact but not intended for >100k parts
  without an ANN index (FAISS/ScaNN).
- The demo library contains 60 synthetic parts, not user uploads.

## NX Open

- **Every NX path is UNTESTED against a live NX session.** No NX installation is
  available in this environment.
- Graph extraction uses NX's STEP exporter and then the tested OCP adapter, so
  NX face tags/IDs are not preserved in the neutral IR.
- Write-back supports only Z-axis holes, Z-axis pockets/slots and Z-axis bosses
  via the candidate cookbook recipes; counterbores/countersinks and fillets/
  chamfers are skipped with warnings (edge selection unverified).
- Native Pocket/Slot builder names could not be verified from public docs, so
  pockets/slots are created as parametric subtractive block features; slot ends
  are therefore not rounded. This is documented, not silently claimed.
- NX's embedded Python may not have OCP/PyG installed; the journal requires a
  compatible Python environment with `cadquery-ocp` (and `torch` for GNN modes).

## Service and deployment

- SQLite is the tested queue/storage path. PostgreSQL is supported via
  `DATABASE_URL` through SQLAlchemy but was not exercised in this environment.
- Both queue paths were exercised with Docker Compose: the SQLite worker and the
  RQ+Redis worker. PostgreSQL remains untested. Multi-worker stress behavior is
  not covered by a load test.
- One worker claim is used; multi-worker correctness relies on the conditional
  update but was not stress-tested.
- The Streamlit app is a demo, not a hardened multi-tenant service: no auth, no
  per-user quotas, 25 MiB upload / 3,000 face limits, and no request timeout
  beyond the host's process limits.
- Streamlit Community Cloud limits are approximate and can change; measure the
  deployed app again after platform updates.
- `cadquery-ocp` installs VTK as a declared dependency (~2.4 GB environment),
  even though the app never imports VTK. The measured peak RSS is under the
  documented target, but install time is significant.

## Not implemented

- Phase 9 multi-view CNN part-family classification: listed as future work; the
  graph approach already meets the project gates and the optional phase was not
  required after review.
- MFCAD++ training/evaluation (download blocked).
- UV-Net/BRepNet/AAGNet-scale architectures; only a compact edge-aware GNN and a
  GCN baseline are implemented.
- Learned similarity embeddings / ANN index.
- Full NX feature coverage (counterbore/countersink, fillet, chamfer, arbitrary
  axes).
- STEP assemblies, PMI/tolerance semantics, sheet metal, freeform surfaces.
- Machining cost/cycle-time estimation or CAM toolpath generation.
