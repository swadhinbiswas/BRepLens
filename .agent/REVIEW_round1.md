# Independent Review — Round 1 (phases 0-4)

Reviewer verdict: **REJECTED** (as presented at the time of review), with 14
defects. Re-review is required after fixes. The reviewer ran pytest, ruff, mypy,
the rules evaluation and the three-way synthetic evaluation, verified GNN
provenance/y_true arrays, checked 25 MFCAD label mappings against raw data, and
spot-checked research links.

## Blocking defects and resolution

| # | Defect | Resolution |
| --- | --- | --- |
| 1 | Ruff failed in `scripts/benchmark_perf.py` (RUF100/RUF059) | Fixed; `ruff check` clean. |
| 2 | `breplens` entry point pointed at missing `cli.py` | Added `breplens.cli` (recognize/schema) + subprocess test. |
| 3 | Synthetic bracket through-hole ground-truth depth was 0/5 instead of through depth | Fixed to entry-to-exit depth; regenerated data; parameter MAE now 0. |
| 4 | NX docs claimed "verified" recipes while research said unverified | Relabeled as candidate public-cookbook recipes, unverified live; consistent UNTESTED labeling. |
| 5 | GNN recess emitted a feature with depth 0 when measurement failed | Now returns `None` (component warning) below `min_depth`. |
| 6 | Referenced generators/results docs missing | Added `scripts/make_readme_tables.py`; generated README block, RESULTS_TABLES, CV, FINAL_REPORT. |
| 7 | Stale `.agent/TASKS.md`/`STATE.md` | Synchronized with actual phase/result state. |
| 8 | LICENSES overclaimed pyproject exact pins | Corrected to exact pins in requirements files, ranges in pyproject. |
| 9 | Unused `smooth_angle_deg`; constant `is_outer` feature | Smooth threshold now passed into classification; `is_outer` removed and models retrained. |
| 10 | Missing blind-hole convexity test; no CLI test | Added blind-hole convexity, smooth-angle parameter, CLI tests. |
| 11 | Missing deploy/perf provenance fields | Added `provenance` blocks; reran both. |
| 12 | Synthetic wall labels derived through concave flood (same signal as rules) | Documented as a caveat in DATASETS/LIMITATIONS. |
| 13 | MFCAD++ counts/license recheck blocked | Docs now say "reported by landing page"; automated recheck blocked by Cloudflare. |
| 14 | `test_hybrid_superset` name overstated | Now asserts hybrid is a superset of rule features and no duplicate faces. |

Reviewer also confirmed: 77 tests passed, mypy clean, rules eval reproduced,
GNN selection used validation only, seeds present, MFCAD labels matched raw
`.face_truth` in a 25-part sample, schema regenerates, and import-isolation tests
exist.

Remaining unverified at round 1: GNN retraining from scratch, full MFCAD
conversion, live NX, and phases 5-8. These were subsequently addressed for
round 2 (both MFCAD conversion and training were rerun; phases 5-8 were added
and tested).
