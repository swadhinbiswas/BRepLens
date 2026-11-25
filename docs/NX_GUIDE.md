# Siemens NX Guide

> **STATUS: UNTESTED AGAINST A LIVE NX SESSION.**
> No NX installation was available. The adapter follows recipes that a public
> cookbook reports as live-tested in NX 2506, but BRepLens has not independently
> verified them or run them in NX. Treat write-back as needing a pilot validation
> before production.

## 1. What the adapter does

`src/breplens/adapters/nx.py` implements `NxOpenAdapter(CadAdapter)`:

- `body_summary()` traverses `session.Parts.Work.Bodies`, `body.GetFaces()`,
  `face.SolidFaceType` and `face.GetEdges()` for diagnostics.
- `load_graph()` exports the work part to STEP with
  `session.DexManager.CreateStepCreator()` and reads it with the tested
  `OccAdapter`. This avoids unverified per-face geometry calls. NX face tags are
  not preserved.
- `write_features()` creates parametric NX features from a BRepLens
  `FeatureSet` JSON:
  - through/blind holes → `Features.CreateHolePackageBuilder` (candidate recipe);
  - pockets/slots → parametric subtractive `BlockFeatureBuilder` (native
    pocket/slot builder names could not be verified);
  - bosses → `Features.CreateCylinderBuilder` + Unite (candidate recipe);
  - counterbore/countersink/fillet/chamfer → skipped with explicit warnings.
- `src/breplens/adapters/nx_journal.py` is the journal entry point that can
  recognize the open part and write the features back.

## 2. Requirements

1. Siemens NX with NX Open Python enabled.
2. A Python environment visible to NX containing:
   - `cadquery-ocp` (used by `OccAdapter`),
   - BRepLens source on `sys.path`,
   - for GNN modes: CPU `torch` and `torch-geometric`,
   - optionally the model at `models/gnn_synthetic.pt`.
3. NX Open classes are imported lazily; importing BRepLens core never imports
   `NXOpen` (enforced by `tests/test_architecture.py`).

## 3. Run inside NX

### Option A: write an existing features JSON

1. Export BRepLens features from the API/demo (`*_features.json`).
2. Open the part in NX.
3. File → Execute → NX Open (Python), select a journal such as:

```python
import sys
sys.path.insert(0, r"C:\path\to\BRepLens\src")
from breplens.adapters.nx_journal import run_on_open_part

report = run_on_open_part(features_json=r"C:\path\to\part_features.json")
print(report)
```

4. Review the returned JSON report: `created`, `skipped`, `warnings`.
5. Inspect the NX part history; each created feature should be editable.

### Option B: recognize + write back in one journal

```python
import sys
sys.path.insert(0, r"C:\path\to\BRepLens\src")
from breplens.adapters.nx_journal import run_on_open_part

# rules only (no torch required inside NX)
report = run_on_open_part(mode="rules")
print(report)

# hybrid (requires torch + torch-geometric in NX's Python)
report = run_on_open_part(
    mode="hybrid",
    model_path=r"C:\path\to\BRepLens\models\gnn_synthetic.pt",
)
```

### Option C: external NX session

If running an external Python process against a configured NX session, call the
same functions; `NxOpenAdapter.available()` reports whether `NXOpen` imports.

## 4. Verification checklist (required before trusting it)

- [ ] `NXOpen` imports in the journal's Python.
- [ ] `body_summary()` returns the expected body/face counts.
- [ ] `load_graph()` succeeds on a simple box and the counts match the STEP.
- [ ] Create one hole at a known point; compare diameter/depth/position in NX.
- [ ] Create one pocket and one slot; confirm the material removal volume.
- [ ] Create one boss; confirm it is united into the target body.
- [ ] Check that skipped features and warnings are surfaced, never silent.
- [ ] Save/REOPEN the part and confirm feature history/parameters survive.
- [ ] Repeat on a non-Z-axis feature (currently expected to be skipped).
- [ ] Record NX version and NX Open build in a validation note.

## 5. API provenance and unverified items

Reported by a public NX 2506 cookbook / community docs (not independently verified here):

- `Session.GetSession()`, `Parts.Work`, `part.Bodies`, `body.GetFaces()`,
  `face.GetEdges()`, `face.SolidFaceType`.
- `DexManager.CreateStepCreator()` with AP242/selection options.
- `Features.CreateHolePackageBuilder`, `HolePackageBuilder.GeneralSimpleHole*`,
  `HolePosition` section + point rule, `BooleanOperation.SetTargetBodies`,
  `CommitFeature`.
- `Features.CreateBlockFeatureBuilder` with `SetOriginAndLengths`,
  `SetBooleanOperationAndTarget`.
- `Features.CreateCylinderBuilder` with axis/diameter/height, Boolean Unite.
- `EdgeBlendBuilder.AddChainset(collector, "radius")` and ChamferBuilder (not
  used by write-back in v1).

**UNVERIFIED in this project** (wrapped defensively, may need adjustment):

- `session.Parts.OpenActiveDisplay(path, option)` for opening `.prt` files.
- `HolePackageBuilder.Holetype.Simple` spelling/enum on the target NX version.
- Whether hole depth `Value` accepts a value larger than the part; through holes
  use a depth slightly larger than the cylinder length.
- Any non-Z-axis placement; hole orientation via the builder is not implemented.
- Native pocket/slot builders (names could not be confirmed), hence block cuts.
- The exact behavior of the STEP exporter's `ExportAs`/selection block across NX
  versions.

`tests/unit/test_nx.py` exercises these paths against `tests/fakes/NXOpen`, a
faithful mock; that validates translation logic, not Siemens' live API.

## 6. Failure handling

- Per-feature failures are caught, recorded in `skipped` and `warnings`, and do
  not abort the whole write-back.
- Geometry outside the supported subset (non-Z axes, counterbores, fillets) is
  skipped with a reason.
- The adapter never claims a feature was created unless `CommitFeature` returned.
