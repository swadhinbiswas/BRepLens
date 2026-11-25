# Independent Review — Round 2 (v1.0.0 gate)

**Verdict: REJECTED** (current working-tree snapshot, 2026-11-24 ~21:21 UTC).

The gate commands are green and the round-1 functional/CLI/eval fixes hold, but the
generated documentation is now stale against the owner's concurrent rerun, and one
round-1 NX wording regression plus a false provenance claim remain. These are
concrete, fixable blockers; no product code was edited by this reviewer.

## Evidence (commands and exact results)

| Check | Command | Result |
| --- | --- | --- |
| Lint | `ruff check src tests scripts streamlit_app.py` | `All checks passed!` exit 0 |
| Types | `mypy src/breplens` | `Success: no issues found in 29 source files` exit 0 |
| Tests | `pytest -q` | **80 passed**, 1 warning, exit 0; matches `results/tests.json` (`passed: 80, failed: 0`) and README quality table |
| CLI | `breplens --version` | `BRepLens 1.0.0`, exit 0; `breplens schema` also emits JSON |
| Rules through-hole depth | `jq .parameter_errors.through_hole.depth results/rules_eval.json` | `mean_abs=0.0`, `max_abs=0.0`, `n=167` (all parameter families 0.0) |
| NX labeling | `docs/NX_GUIDE.md:3-6,112`, `src/breplens/adapters/nx.py:3-19`, `nx_journal.py:3` | UNTESTED / not independently verified; no live-run claim in NX guide or code |
| Perf/deploy provenance | `jq .provenance results/perf.json results/deploy_benchmark.json` | both present with script, command, timestamp; perf also environment/host |
| Deploy memory | `jq '.apptest.peak_rss_apptest_mb, .pipeline.peak_rss_pipeline_mb' results/deploy_benchmark.json` | 854.8 MB / 856.6 MB, under 1500 MB |
| Requirements | `cat requirements.txt` | `--extra-index-url .../whl/cpu`, `torch==2.11.0+cpu`; no CUDA/training-only deps |
| Docker artifacts | `jq '{status,health_ok,job_seconds}' results/docker_service_test*.json` | both `succeeded`, `health_ok: true` |
| Demo library/samples | `jq '.entries|length' demo/library.json`; `ls samples/*.step|wc -l` | 60 entries / 10 samples, matching README |

Generated-doc check: at 21:19:30 UTC a clean copy of `README.md`,
`docs/RESULTS_TABLES.md`, `docs/CV_SUMMARY.md`, and `docs/FINAL_REPORT.md` was
**byte-identical** to fresh output from `scripts/make_readme_tables.py` against
the then-current inputs. That is no longer true (see blocker 1).

## Blocking defects

1. **Generated docs are stale against current result files; re-running the
   generator changes published numbers.** The owner process reran evaluations,
   manifests, and Docker smoke tests between 21:20:02 and 21:21:01 UTC after the
   docs were generated at 21:18:38. Running `make_readme_tables.py` now diffs:
   - `README.md:190-192`, `docs/RESULTS_TABLES.md:16-18`, `docs/FINAL_REPORT.md:30-32`:
     recognition latency `0.18/1.63/2.60` → `0.19/1.68/2.69` ms/part.
   - `README.md:289-290` and `docs/FINAL_REPORT.md:76-77`: SQLite Docker end-to-end
     `1.90 s` → `2.10 s` (current `results/docker_service_test.json` has
     `job_seconds: 2.105`).
   - `README.md:282-283`, `docs/RESULTS_TABLES.md:108-109`, `docs/FINAL_REPORT.md:66-67`:
     synthetic generation `5.48 s` → `0.04 s` and MFCAD conversion `10.39 s` → `1.00 s`.
   The last pair is especially unsafe to publish: current manifests
   (`data/synthetic/manifest.json` `generation_wall_seconds: 0.04`,
   `data/mfcad/manifest.json` `conversion_wall_seconds: 1.0`) look like
   incremental/resumed-run timings, not full 1,650-part / 3,000-part wall-clock.
   The README claims these tables trace to `results/*.json`; currently they do not.
   Re-run the full pipelines or keep the original full-run timings, then regenerate
   docs and commit. (`README.md:175-176`, `:282-283`; `docs/RESULTS_TABLES.md:3`.)

2. **NX "verified" wording regression in `docs/FINAL_REPORT.md:158-159`.** It says
   write-back uses “verified public recipes and conservative block cuts”, while
   `docs/NX_GUIDE.md:3-6,112`, `src/breplens/adapters/nx.py:3-19` and the current
   generator text (`scripts/make_readme_tables.py:441-442`, “candidate public-cookbook
   recipes (not verified live here)”) all say UNTESTED/unverified. This is the same
   class as round-1 blocker #4. The generator has already been corrected but the
   report was not regenerated.

3. **False provenance claim in `docs/FINAL_REPORT.md:187-189`.** “every results JSON
   records its command, seed, versions and timestamp” is not true:
   `results/coverage.json` and `results/coverage_core.json` have no `provenance`
   block (`jq 'has("provenance")'` → false); `results/tests.json` has no
   seed/versions; `results/docker_service_test*.json` have no seed/versions.
   Narrow the claim to files that actually carry provenance, or add provenance to
   those result files.

## Non-blocking notes

- Working tree is dirty: 16 tracked files modified (core recognition/service code,
  results, generated docs, generator) and the new review file untracked. v1.0.0
  should be committed/tagged only after the docs are regenerated from final inputs.
- `requirements.txt` is inference-only CPU torch; deployment probe installed it in
  `.deploy-venv` successfully (2,413 MB env, 147 s install). The README quickstart
  (`uv venv` / `uv pip install -r requirements.txt` / `streamlit run streamlit_app.py`)
  is syntactically valid but was not run end-to-end with `uv`; the equivalent pip
  install is proven by the deploy probe.
- `Dockerfile` runs as root (no `USER`); no secrets found in tracked config/source.
  MIT license is present. Security hardening (non-root, pinned image digest) is
  recommended but not a v1.0.0 correctness blocker.
- Synthetic rules saturate at 1.000 on the generator and all parameter errors are
  0.0; this is documented in `docs/LIMITATIONS.md`/FINAL_REPORT as generator
  saturation, not real-world accuracy. MFCAD rule limits are reported honestly.
- CI commands match local green commands; the workflow has not run on a remote
  (documented). No obvious fabricated metrics outside the generated tables were
  found: 60-part library and 10 samples match on disk.
