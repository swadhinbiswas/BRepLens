# BRepLens — Research: B-Rep Feature Recognition, Datasets, Kernels, and Similarity

Status: research complete for the Phase 1.1–1.6 content, with explicit verification gaps listed; Phase 1.7 reviewer gate pending.
Research date: 2026-11-24. Author: researcher subagent.
Scope: exactly the six areas requested; every URL claim below was checked by an HTTP
request or by inspecting an installed package. Where verification failed, the text says
**UNVERIFIED** and records what was tried.

Verification legend:

- **[HTTP 200]** — fetched successfully on 2026-11-24 (curl or Python urllib).
- **[HTTP 403/404]** — reached the server, but content was blocked or absent.
- **[LOCAL]** — verified by running the installed package in `.venv` (CPython 3.11.18).
- **[UNVERIFIED]** — could not be confirmed from an official/primary source.

Hardware/runtime context used for feasibility reasoning (from `.agent/STATE.md` and
`nvidia-smi`): Intel i5-13600K (20 threads), 32 GiB RAM, NVIDIA RTX 3060 Ti 8 GiB
(driver 620.73.04), CPython 3.11.18 project venv. `torch`/`torch-geometric` were still
being installed while this document was written; their CUDA availability is therefore
**UNVERIFIED**. `cadquery-ocp 7.9.3.1.1`, `cadquery 2.8.0`, `networkx 3.6.1`, and
`numpy 2.2.6` were importable **[LOCAL]**.

---

## 1. B-Rep to Attributed Adjacency Graph (AAG) and edge convexity

### 1.1 B-Rep entities that matter

A boundary representation describes a solid by its boundary: faces (trimmed surfaces),
edges (trimmed curves), vertices, and the incidence/ordering relations between them.
For feature recognition the useful entities are:

- **Face** — a connected region of a surface (plane, cylinder, cone, sphere, torus,
  B-spline, …), bounded by one or more wires.
- **Edge** — a curve segment shared by one or two faces.
- **Coedge / half-edge** — one oriented use of an edge by a face; the coedge stores
  `next`/`previous` around the face loop and the mating coedge on the adjacent face.
  BRepNet builds its convolution on coedges for exactly this reason
  ([BRepNet README](https://github.com/AutodeskAILab/BRepNet), [HTTP 200]).

### 1.2 Attributed Adjacency Graph (AAG)

The AAG was introduced by Joshi and Chang, “Graph-based heuristics for recognition of
machined features from a 3D solid model”, *Computer-Aided Design* 20(2), 1988,
DOI [10.1016/0010-4485(88)90050-4](https://doi.org/10.1016/0010-4485(88)90050-4)
(DOI metadata verified through the Crossref API **[HTTP 200]**; Semantic Scholar lists
834 citations, abstract elided by the publisher).

The structure is restated precisely in the open-access EAG paper
([“Edge adjacency graph and neural network architecture for machining feature
recognition”, Research Square PDF](https://www.researchsquare.com/article/rs-4908235/latest.pdf),
**[HTTP 200]**, 20 pages):

> “The AAG data structure consists of a graph where the nodes represent faces on a
> B-Rep model and connections between nodes represent the edges on the model.”
> — EAG paper, §2 (PDF p. 2)

So, for a solid `S`:

- `V = {f_1 … f_n}` — one node per B-Rep face.
- `E = {(f_i, f_j)}` — one arc per shared edge (a pair of faces may share more than one
  edge; keep parallel arcs, or aggregate them explicitly).
- Node attributes: surface type, area, normal/axis, radius, convexity-relevant flags,
  bounding-box extents, and optionally UV-grid samples.
- Arc/edge attributes: convex (0), concave (1), smooth/tangent (separate class), plus
  edge length, dihedral angle, and curve type.

**Terminology caveat.** Implementations differ on the 0/1 polarity and on whether
smooth edges are included. The verified source above fixes the graph structure but not
the polarity; this project will document its own convention in the schema and keep it
configurable. That is an implementation decision, not a claim about the 1988 paper’s
exact encoding.

### 1.3 Edge convexity classification

For a manifold edge shared by faces A and B, with outward unit normals `n_A`, `n_B` at
a point on the edge and unit tangent `t` along the edge:

1. Evaluate a point `p` strictly inside the edge (mid-parameter is adequate for
   straight/analytic edges; for curved edges evaluate at several parameters and take a
   majority).
2. Evaluate outward normals `n_A(p)`, `n_B(p)` from the two faces’ surfaces.
3. Evaluate `t(p)` from the edge curve, oriented consistently with the coedge order in
   face A.
4. Compute the signed triple product `s = (n_A × n_B) · t`.
   - `s > +tol` → convex edge.
   - `s < -tol` → concave edge.
   - `|s| <= tol` and `n_A · n_B > 0` → smooth/tangent edge (no sharp dihedral).
5. Reject/flag edges where the two faces are not G1 but the normals are numerically
   unstable (very short edges, seam edges, degenerate slivers).

This is the standard dihedral-sign test; the orientation convention must be validated
on primitives (cube → all convex; cylindrical hole through a block → concave at the
hole/block intersection; fillet → smooth). The project’s Phase 3.3 test plan already
requires exactly that calibration. OCC exposes the required queries; the following
imports were confirmed **[LOCAL]**:

```python
from OCP.TopExp import TopExp_Explorer, TopExp
from OCP.TopAbs import TopAbs_FACE, TopAbs_EDGE
from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCP.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder
```

Use `BRepAdaptor_Surface` for surface type/parameters and `BRepAdaptor_Curve` for the
edge curve; `TopExp.MapShapesAndAncestors_s` (the bound name; a plain
`MapShapesAndAncestors` attribute does **not** exist in OCP 7.9.3.1.1) with a
`TopTools_IndexedDataMapOfShapeListOfShape`, or an explorer plus ancestor maps, for
face–edge incidence. These class names are verified as importable in the installed
OCP 7.9.3.1.1; their signatures were not exhaustively tested.

**Practical pitfalls.**

- Seam edges on cylinders/cones appear twice in the same face; do not create a self-loop
  arc, and do not count the seam as a feature edge.
- Periodicity can make the mid-parameter normal discontinuous; sample multiple `u`
  values and reject inconsistent signs.
- Tolerance must scale with model size; a fixed `1e-7` is wrong for millimetre-scale
  parts from different CAD systems.
- AAG construction is cheap (`O(F + E)`), but *feature matching* on the AAG (subgraph
  isomorphism / rule search) is the expensive and brittle part.

### 1.4 Classical rule-based recognition and its documented limits

The classical pipeline is: build AAG → remove convex-only connections → decompose into
concave-connected subgraphs → match each subgraph against feature templates → extract
parameters. The EAG paper states this directly:

> “These methods separate the multiple machining features in an input model by removing
> AAG nodes with only convex connections, then by removing any remaining convex
> connections to produce a disconnected graph where the constituent subgraphs each
> represent a single machining feature.” — EAG paper, §1

Documented limitations (all from verified primary/secondary text):

1. **Convex features are missed.** The same EAG paper: rule-based AAG methods are
   “incapable of detecting convex machining features such as bosses or pocket islands.”
2. **No learning of new categories.** “They cannot learn on their own to detect
   additional categories of machining features.” — EAG paper.
3. **Intersecting features are hard.** AAGNet’s abstract states that “traditional
   rule-based MFR methods struggle to handle intersecting features due to the complexity
   of representing their variable topological structures”
   ([AAGNet README](https://github.com/whjdark/AAGNet), **[HTTP 200]**; paper DOI
   [10.1016/j.rcim.2023.102661](https://doi.org/10.1016/j.rcim.2023.102661)).
4. **Template/subgraph explosion.** Every feature variant and every interaction needs a
   new rule/template; the number of rules grows combinatorially with feature count and
   intersection patterns. This is the practical consequence of (3), and is why the
   learning literature exists.
5. **Smooth edges and blends** do not fit the binary convex/concave model; fillets and
   chamfers need special handling or a separate class.
6. **Stock/feature ambiguity.** The same concave subgraph can be explained by different
   feature decompositions (feature interaction, merged faces, split faces). The EAG
   paper explicitly calls out merge-face and split-face failures for its own learned
   model, which also affect rule-based decomposition.

The often-cited 2018 review attributed to “Gavilanes et al.” could **not** be located.
Crossref author/title queries and DuckDuckGo returned no matching machining-feature
review by that author. **UNVERIFIED — do not cite it.** Verified substitutes:

- A. Verma, S. Rajotia, “A review of machining feature recognition methodologies”,
  *International Journal of Computer Integrated Manufacturing*, 2010,
  DOI [10.1080/09511921003642121](https://doi.org/10.1080/09511921003642121)
  (Crossref **[HTTP 200]**).
- Y. Shi, Y. Zhang, K. Xia, R. Harik, “A Critical Review of Feature Recognition
  Techniques”, *Computer-Aided Design and Applications*, 2020,
  DOI [10.14733/cadaps.2020.861-899](https://doi.org/10.14733/cadaps.2020.861-899)
  (Crossref **[HTTP 200]**; full PDF host was slow/truncated in this session).
- “Automatic Feature Recognition Techniques for the Integration of CAD and CAM:
  A Review”, 2024, DOI [10.1520/ssms20230016](https://doi.org/10.1520/ssms20230016)
  (Crossref **[HTTP 200]**).

### 1.5 Consequence for BRepLens

Rules are still valuable as a deterministic, explainable baseline and for parameter
extraction (diameter, depth, axis, position) where the geometry is unambiguous. They are
not sufficient for intersecting features, convex features (bosses/islands), or unseen
feature vocabularies. This motivates the hybrid architecture: rules for exact,
well-formed features + a learned gAAG model for the rest, with a documented fallback
when the two disagree.

---

## 2. Learning approaches for machining feature recognition

### 2.1 Comparison table

Reported numbers are copied from the cited source; they are **not** directly comparable
across datasets. “Params” is the source’s count.

| Model | Input representation | Architecture | Reported result (dataset) | Params | Code | License |
|---|---|---|---|---|---|---|
| **UV-Net** (CVPR 2021) | 10×10×7 face UV-grids (xyz, normal, trimming mask) + 10×6 edge UV-grids (xyz, tangent); face-adjacency graph | 2D/1D CNN encoders + graph message passing (DGL) | MFCAD per-face **99.95%**, IoU **99.87%**; MFCAD classification 99.94%; ABC segmentation 88.87%/IoU 50.37% | 1.23M (seg), 1.34M (cls) | [AutodeskAILab/UV-Net](https://github.com/AutodeskAILab/UV-Net) **[HTTP 200]** | MIT (GitHub API) |
| **BRepNet** (CVPR 2021) | Coedges + topological walks defined by a kernel JSON; face/edge/coedge features | Coedge-centred MLP convolution, pooling to faces/edges | Fusion 360 Gallery s2.0.0 per-face **92.52%**, IoU **77.10%**; ECC baseline 90.36%/72.08; ASM heuristic 64.57%/49.53 | 359,100 | [AutodeskAILab/BRepNet](https://github.com/AutodeskAILab/BRepNet) **[HTTP 200]** | CC BY-NC-SA 4.0 (LICENSE file **[HTTP 200]**) |
| **AAGNet** (RCIM 2024) | gAAG: face nodes + edge arcs with topological, geometric, extended attributes; edge features | Graph encoder (NodeConv/EdgeConv, MLP) + multi-task heads: semantic segmentation, instance segmentation, bottom-face segmentation | MFInstSeg instance **99.94%**, F1 **98.84%** (EAGIS Table 3); MFCAD++ **99.57%** per MFCAD++ README (secondary citation via BrepMFR) | 0.44M (MFInstSeg) / 0.38M (MFCAD++ table) | [whjdark/AAGNet](https://github.com/whjdark/AAGNet) **[HTTP 200]** | MIT (GitHub API) |
| **Hierarchical CADNet** (CAD 2022) | STL mesh graph + B-Rep face-adjacency graph; edge-convexity variant | Hierarchical GCN (TensorFlow) | MFCAD++ **97.37%** (Edge) / **96.52%** (Adj) | 9.76M / 6.62M | [QUB GitLab](https://gitlab.com/qub_femg/machine-learning/hierarchical-cadnet) **[HTTP 200]** | not stated in README **[UNVERIFIED]** |
| **EAGIS** (2024, edge-aware baseline) | Edge Adjacency Graph (EAG): edge-centric nodes/connections | Small GNN, instance segmentation | MFInstSeg **99.57%**, F1 **98.04%** | 0.10M | [Research Square PDF](https://www.researchsquare.com/article/rs-4908235/latest.pdf) **[HTTP 200]** | **[UNVERIFIED]** |
| **GraphSAGE baseline** | AAG/graph | SAGEConv stack | MFInstSeg **99.82%**, F1 96.64% | 1.09M | EAGIS Table 3 | — |
| **GAT baseline** | AAG/graph | GATConv stack | MFInstSeg **99.86%**, F1 97.43% | 1.24M | EAGIS Table 3 | — |

Notes on provenance:

- UV-Net and BRepNet numbers were extracted directly from the CVF open-access PDFs
  (`pdftotext`), not from README marketing: UV-Net Table 2; BRepNet Table 1.
- AAGNet’s MFInstSeg numbers are from EAGIS Table 3, an independent paper’s comparison.
  Its MFCAD++ 99.57% is from the MFCAD++ README’s benchmark table, which credits
  BrepMFR [3]; it is a secondary citation.
- Hierarchical CADNet’s MFCAD++ numbers are from the official MFCAD++ README benchmark
  table (same secondary-source caveat).
- The EAGIS PDF is a Research Square preprint (Springer Nature template), not a
  peer-reviewed venue record at the time of fetch; treat its numbers as preprint-grade.
- AAGNet code uses **DGL**, not PyG, and requires `pythonocc-core` 7.5.1 / Python 3.10
  per its README. A PyG port is an implementation task, not a drop-in install.

### 2.2 What each model actually needs

**UV-Net.** Needs STEP → B-Rep → per-face UV sampling on the trimmed region, per-edge
curve sampling, then a face-adjacency graph. The hard parts are trimming-aware UV
sampling and consistent orientation; the network itself is small and simple. Its MFCAD
result is excellent but the dataset is synthetic and highly regular (16 labels, single
stock block with features), so the number overstates real-world difficulty.

**BRepNet.** Needs coedge extraction, kernel definitions, and per-coedge feature
vectors. It is the most faithful to B-Rep topology and reports a strong Fusion 360
segmentation result (92.52% face accuracy). But: the code is
**CC BY-NC-SA 4.0** (non-commercial), the kernel abstraction is more complex to port,
and the Fusion 360 dataset itself is non-commercial research-only. That combination is
a licensing problem for a portfolio project that may be hosted publicly and described as
production-capable. The architecture is still valuable as a design reference.

**AAGNet.** The gAAG is just a property graph: face nodes, edge arcs, node/edge feature
vectors, plus instance/bottom-face labels. It maps cleanly onto PyTorch Geometric
`Data` objects (`x`, `edge_index`, `edge_attr`, `y`, `batch`). The reference code’s
encoders are standard (GCN/SAGE/GIN/GAT/GATv2/DeeperGCN plus custom NodeConv/EdgeConv),
and the multi-task heads are MLPs. This is the most realistic strong model to
reimplement. Its published parameter counts (0.38–0.44M) are tiny by modern standards.

**Graph/edge-aware baseline.** A 2-layer GraphSAGE or GAT on the same gAAG, with edge
attributes concatenated or used in an edge-conditioned convolution, is a fair baseline
and is ~1M parameters. EAGIS demonstrates that an edge-centric graph with ~0.1M
parameters is competitive on instance segmentation; this is a strong argument for
keeping the baseline small and the representation good.

### 2.3 Feasibility on an 8 GB RTX 3060 Ti and CPU inference

- **Parameter memory is negligible.** All models above are ≤10M parameters; weights are
  a few MB to tens of MB. The 8 GB limit is not challenged by the models.
- **Activation memory** scales with nodes/edges per batch. AAG graphs are small (tens to
  low hundreds of faces per part); a batch of 32–64 graphs is far below 8 GB. This is a
  reasoned estimate, not a measured benchmark; Phase 4b must record actual peak memory.
- **The real cost is data preparation.** STEP parsing, face/edge classification, and
  graph conversion dominate wall-clock time and are CPU-bound. The plan already calls
  for multiprocessing, caching, and resumability (Phase 4b.2).
- **CPU inference** for a 0.4M-parameter GNN on a 50-face graph is milliseconds to tens
  of milliseconds per part, plus OCP STEP load time (OCP import + box creation measured
  at 0.339 s and 402.9 MB peak RSS **[LOCAL]**). A small model exported to TorchScript
  or ONNX is appropriate for the hosted demo.

**Recommendation:** implement an **AAGNet-style gAAG encoder in PyTorch Geometric**
(edge-aware message passing + multi-task heads), with a **GraphSAGE/GAT baseline** on the
same graphs. Do **not** make BRepNet the primary implementation (license + engineering
cost), and do not make UV-Net primary (UV sampling + DGL + weaker fit to the AAG-centric
architecture). UV-Net is a reasonable optional comparison if time permits.

---

## 3. Datasets

### 3.1 Verified dataset table

| Dataset | Official URL | Size | License | Labels | Format | Verification |
|---|---|---|---|---|---|---|
| **MFCAD++** | [QUB Pure landing page](https://pure.qub.ac.uk/en/datasets/mfcad-dataset-dataset-for-paper-hierarchical-cadnet-learning-from) | 1.5 GB zip | **CC BY** | 25 IDs (0–24): 24 machining features + Stock | STEP with per-face labels in `ADVANCED_FACES`; train/val/test txt splits | Landing page **[HTTP 200]**; direct zip **[HTTP 403]** to curl |
| **MFCAD** | [github.com/hducg/MFCAD](https://github.com/hducg/MFCAD) | GitHub reports 189,930 KB (~190 MB) | MIT (repository LICENSE) | 16 segmentation labels | STEP + `.face_truth` | Repo/API **[HTTP 200]**; `git ls-tree` verified 15,488 `.step` + 15,488 `.face_truth` |
| **Fusion 360 Gallery — Segmentation** | [repo](https://github.com/AutodeskAILab/Fusion360GalleryDataset) / [s2.0.0 zip](https://fusion-360-gallery-dataset.s3-us-west-2.amazonaws.com/segmentation/s2.0.0/s2.0.0.zip) | s2.0.0: 3,326,261,510 bytes (~3.2 GB); s2.0.1: 3.1 GB | **Non-commercial research only** (custom license) | 8 modeling-operation labels | STEP + per-face/per-triangle/per-point labels | Repo **[HTTP 200]**; S3 HEAD **[HTTP 200]**; license text **[HTTP 200]** |
| **MFInstSeg** (optional) | AAGNet README links to Baidu AI Studio / Google Drive | 60,000+ STEP | **[UNVERIFIED]** | instance + semantic + bottom-face | STEP + gAAG conversion | README **[HTTP 200]**; download hosts not fetched |

### 3.2 MFCAD++ (primary candidate)

- **Paper:** A. R. Colligan, T. T. Robinson, D. C. Nolan, Y. Hua, W. Cao,
  “Hierarchical CADNet: Learning from B-Reps for Machining Feature Recognition”,
  *Computer-Aided Design* 147 (2022), DOI
  [10.1016/j.cad.2022.103226](https://doi.org/10.1016/j.cad.2022.103226)
  (Crossref **[HTTP 200]**; OpenAlex confirms CC-BY hybrid OA).
- **Generation code:** [gitlab.com/qub_femg/machine-learning/mfcad2-dataset](https://gitlab.com/qub_femg/machine-learning/mfcad2-dataset)
  **[HTTP 200]**; README links the dataset landing page; LICENSE is MIT.
- **Verified counts from the landing page:** train = 41,766; val = 8,950; test = 8,949;
  **total = 59,665**. Note: the commonly repeated “59,655” is contradicted by the
  official page; use 59,665 unless a re-count of the zip proves otherwise.
- **Labels (verified `dataset_description.txt`):** 0 Chamfer, 1 Through hole,
  2 Triangular passage, 3 Rectangular passage, 4 6-sides passage, 5 Triangular through
  slot, 6 Rectangular through slot, 7 Circular through slot, 8 Rectangular through step,
  9 2-sides through step, 10 Slanted through step, 11 O-ring, 12 Blind hole,
  13 Triangular pocket, 14 Rectangular pocket, 15 6-sides pocket, 16 Circular end pocket,
  17 Rectangular blind slot, 18 Vertical circular end blind slot, 19 Horizontal circular
  end blind slot, 20 Triangular blind step, 21 Circular blind step, 22 Rectangular blind
  step, 23 Round, 24 Stock.
- **Format detail:** unlike MFCAD’s pickled `.face_truth` lists, MFCAD++ writes class
  labels directly to the STEP `ADVANCED_FACES`; labels can be extracted from the STEP
  file. This is a significant preprocessing advantage.
- **Licence:** CC BY on the QUB dataset record (verified in page metadata: “Licence:
  CC BY”). CC BY permits commercial use with attribution, unlike Fusion 360 Gallery.
- **Download caveat:** `https://pure.qub.ac.uk/files/278385243/MFCAD_dataset.zip`
  returned **HTTP 403** to curl (bot protection). The landing page is reachable; a human
  browser download is likely required. The GitLab README points to the same landing page.
  A CI pipeline must not assume the zip is fetchable unattended.

### 3.3 MFCAD (fallback / smoke-test dataset)

- **Repo:** [github.com/hducg/MFCAD](https://github.com/hducg/MFCAD) **[HTTP 200]**,
  MIT, default branch `master`.
- **Actual size:** `git clone --depth 1 --filter=blob:none --no-checkout` followed by
  `git ls-tree -r --name-only HEAD` produced **30,983 entries**, of which **15,488** are
  `.step` and **15,488** are `.face_truth` **[LOCAL verification]**.
- **Labels:** the UV-Net paper states MFCAD is “15,488 3D shapes … 16 different
  segmentation labels … applied per face” (UV-Net PDF, §4.1). The label mapping is not
  in the minimal README; MFCAD++’s dataset description says MFCAD used pickled
  `.face_truth` lists. Extracting the mapping is a Phase 4b task, not assumed here.
- **Why it is the fallback:** MIT license, small, directly cloneable, and UV-Net reports
  99.95% face accuracy on it. It is easier to obtain than MFCAD++ but has fewer classes
  and simpler single-stock-block geometry.

### 3.4 Fusion 360 Gallery (secondary; licence-constrained)

- **Repo:** [AutodeskAILab/Fusion360GalleryDataset](https://github.com/AutodeskAILab/Fusion360GalleryDataset)
  **[HTTP 200]**, 765 stars, license API reports NOASSERTION (custom license).
- **Segmentation subset:** README currently lists **35,680 parts** (s2.0.1). The BRepNet
  paper (s2.0.0) says **35,858 models**. Both are documented; do not silently pick one.
- **Format:** STEP + per-face/per-triangle/per-point labels; 8 labels: ExtrudeSide,
  ExtrudeEnd, CutSide, CutEnd, Fillet, Chamfer, RevolveSide, RevolveEnd (BRepNet PDF,
  §4).
- **Download:** s2.0.0 zip HEAD returned **HTTP 200**, `Content-Length: 3,326,261,510`
  bytes. s2.0.1 is 3.1 GB and an extended STEP set is 483 MB / 42,912 files.
- **Licence (verified full text):** “You may access, use, reproduce and modify the
  Dataset, in each case, only for **non-commercial research purposes**.” Redistribution
  of the dataset in its entirety is prohibited; portions/modifications are allowed only
  under the same non-commercial terms. This is **not** suitable as the primary dataset
  for a publicly hosted demo that could be read as commercial, and it cannot be
  redistributed in the repo.
- **Why secondary:** the labels are modeling operations, not machining features, and
  the licence is restrictive. Good for a BRepNet comparison, not for the product’s
  primary training set.

### 3.5 Dataset decision inputs

- MFCAD++: best label richness + CC BY + per-face labels in STEP; download friction.
- MFCAD: MIT + trivial clone + 16 labels; weaker realism, older label storage.
- Fusion 360: best real-world geometry but non-commercial and operation labels.
- Synthetic CadQuery: always available, no licence issue, ground truth by construction;
  necessary as a CI/offline fallback and for adversarial tests.

---

## 4. Siemens NX Open Python API

### 4.1 What was actually verified

Official Siemens documentation could **not** be retrieved in a form containing API text.
Evidence:

1. `https://docs.sw.siemens.com/en-US/product/209349590/doc/PL202006250426018413/nxopen_python_ref`
   → **HTTP 200**, but a 16,631-byte JavaScript shell titled “Documentation Center”;
   the only viewer script is `.../doc-center/v1.1.16/not-found-viewer.js`. No NXOpen
   content in the response.
2. Same result for `.../PL20211004053733141/nxopen_apis` and the bare doc IDs.
3. `https://docs.sw.siemens.com/robots.txt` → **HTTP 200** with:
   `User-agent: *` / `Disallow: /` — the docs site explicitly disallows crawling.
4. `https://docs.sw.siemens.com/en-US/product/209349590` → **HTTP 404**.
5. `https://docs.sw.siemens.com/sitemap.xml` → **HTTP 404**.

Therefore **no exact NXOpen class or method name below is verified**. This section
records the intended design and the probe plan; it does not claim API signatures.

### 4.2 Required capabilities (to verify on a licensed NX install)

| Capability | What the adapter must do | API status |
|---|---|---|
| Session acquisition | Get the current NX session from a journal; connect from an external Python process | **UNVERIFIED** |
| Part/body traversal | Enumerate bodies in the work part; enumerate faces/edges per body | **UNVERIFIED** |
| Face queries | Surface type (plane/cylinder/cone/sphere/torus/BSurf), radius, axis, normal, area | **UNVERIFIED** |
| Edge queries | Adjacent faces, curve type, convexity/dihedral, length | **UNVERIFIED** |
| Hole creation | Through/blind, diameter, depth, tip angle, position/direction | **UNVERIFIED** |
| Pocket creation | Rectangular/circular/general, depth, corner radius, taper, floor | **UNVERIFIED** |
| Slot creation | Rectangular/ball-end/T-slot/U-slot, width, depth, length, orientation | **UNVERIFIED** |
| Journaling | Record a session to a replayable Python journal; replay deterministically | **UNVERIFIED** |
| External session | Attach to a running NX or run batch without the UI | **UNVERIFIED** |

Candidate names commonly seen in NX Open material (e.g. `NXOpen.Session`,
`Session.GetSession()`, `PartCollection`, `Body.GetFaces()`, `Face.GetEdges()`,
`Face.SolidFaceType`, `Face.Radius`, `Face.Axis`, `HoleBuilder`, `PocketBuilder`,
`SlotBuilder`, `FeatureCollection`) are listed here **only as search terms for a live
probe**, not as verified API. Every one is **UNVERIFIED** until reproduced against the
installed NX release’s own Python reference (`dir()`/`help()` inside NX or the shipped
`.chm`/HTML docs).

### 4.3 Journaling vs external session

- **Journal-first (recommended for write-back):** record a human modelling a hole/pocket/
  slot in NX, then replay the generated Python journal. This is the safest way to learn
  the exact builder sequence and to keep the operation parametric. It also produces a
  reproducible script for the repo. Status: **UNVERIFIED** until run on NX.
- **External session (recommended for read-only traversal):** a standalone Python process
  attaches to a running NX session or launches a batch part, traverses B-Rep, and writes
  a neutral JSON. This keeps the FastAPI service independent of NX. Status:
  **UNVERIFIED**; exact connection/licensing behaviour must be tested.
- **Mock adapter (mandatory for CI):** implement `NxOpenAdapter` against a small typed
  protocol, with a faithful fake that mimics the entity graph and builder validation.
  Unit tests run without NX; an `UNTESTED_AGAINST_LIVE_NX` marker must appear in code,
  docs, and the UI whenever the real adapter is selected. This is already required by
  Phase 4c.5.
- **Write-back policy:** never attempt direct B-Rep face editing for feature creation;
  use parametric feature builders so the model history stays editable. If a builder is
  unavailable, fail loudly rather than silently producing a non-parametric body.
  Status: design principle, API details **UNVERIFIED**.

### 4.4 Probe script to run inside NX (when available)

Do not guess signatures. Run this in NX’s Python console/journal and save the output to
`.agent/research/nx_api_probe.md`:

```python
# UNVERIFIED probe; run inside NX, not in CI.
import inspect
import NXOpen

sess = NXOpen.Session.GetSession()          # candidate name, verify
print(type(sess))
print([n for n in dir(sess) if "Part" in n or "Undo" in n])

work = sess.Parts.Work                      # candidate
print(type(work))
print([n for n in dir(work) if "Bod" in n or "Feature" in n])

for body in work.Bodies:                    # candidate iteration
    print(type(body), [n for n in dir(body) if "Face" in n or "Edge" in n])
    for face in body.GetFaces():            # candidate
        print(type(face), face.SolidFaceType)
        print([n for n in dir(face) if "Radius" in n or "Axis" in n])
        break
    break
```

The value of this script is the `dir()` output, not the candidate names. It must be
executed on the target NX version before any adapter method is treated as real.

---

## 5. pythonocc-core vs OCP (cadquery-ocp) vs CadQuery for pip-only hosting

### 5.1 Package facts (all checked on 2026-11-24)

| Package | Distribution | CPython 3.11 Linux wheel | Install size | License metadata | API level |
|---|---|---|---|---|---|
| **pythonocc-core** | PyPI page exists (v0.16) but **no release files**; conda-forge has 8.0.1.2 | **No pip wheel** | conda package ~39–40 MB | PyPI: LGPL; conda-forge: **LGPL-3.0-or-later** | Pythonic OCC wrapper |
| **cadquery-ocp (OCP)** | PyPI | **Yes**: `cadquery_ocp-8.0.1.1.0-cp311-cp311-manylinux_2_28_x86_64.whl`, **67.0 MB** | **160 MB** on disk (installed 7.9.3.1.1) | PyPI metadata: **Apache-2.0**; underlying OCCT: **LGPL-2.1 + Open CASCADE exception** | Low-level OCC bindings |
| **CadQuery** | PyPI | Pure Python (`cadquery 2.8.0`, requires ≥3.11) | **1.9 MB** package, plus OCP/deps | **Apache-2.0** | High-level fluent Workplane API |

Verified details:

- **pythonocc-core:** `https://pypi.org/pypi/pythonocc-core/json` **[HTTP 200]** reports
  latest `0.16` and an **empty** release file list. `pip install pythonocc-core` therefore
  cannot install a working package from PyPI. Conda-forge API
  (`https://api.anaconda.org/package/conda-forge/pythonocc-core`) **[HTTP 200]** reports
  latest `8.0.1.2`, license `LGPL-3.0-or-later`, with package builds around 39–40 MB (the listing sampled macOS builds).
  Since conda/mamba are absent in this environment, pythonocc-core is not a pip-only
  option.
- **cadquery-ocp:** PyPI JSON **[HTTP 200]** reports `8.0.1.1.0`, `requires_python
  >=3.11,<3.15`, license `Apache-2.0`, and the cp311 manylinux x86_64 wheel at 67.0 MB.
  The installed wheel is `7.9.3.1.1`; `importlib.metadata` reports the same
  `Apache-2.0` field. The OCCT kernel source license files were fetched from the OCCT
  repo **[HTTP 200]**: LGPL-2.1 text plus the Open CASCADE exception (object code using
  OCCT headers may be distributed under terms of your choice, with notice). The OCP
  wheel metadata being Apache-2.0 does not erase the LGPL obligations of the bundled
  OCCT libraries; legal review is required before distribution. Do not describe the
  combined work as “Apache-2.0 only”.
- **CadQuery:** PyPI JSON **[HTTP 200]** reports `2.8.0`, `requires_python >=3.11`,
  license “Apache Public License 2.0”. It depends on OCP; it is not a separate kernel.

### 5.2 Measured runtime behaviour

**[LOCAL]** in the project venv:

- `import OCP` + `BRepPrimAPI_MakeBox` + `BRepAdaptor_Surface`: **0.339 s**,
  **402.9 MB max RSS** for the process. This is the realistic per-process floor for a
  geometry worker; a FastAPI worker that imports OCP pays it once at startup.
- Installed `OCP` directory: **160 MB**; installed `cadquery` directory: **1.9 MB**;
  full `.venv`: **2.1 GB** (includes Streamlit/VTK, not needed for inference-only).
- OCP traversal API imports confirmed: `TopExp_Explorer`, `TopAbs_FACE`,
  `TopAbs_EDGE`, `BRep_Tool`, `BRepAdaptor_Surface`, `BRepAdaptor_Curve`,
  `GeomAbs_Plane`, `GeomAbs_Cylinder` **[LOCAL]**.

### 5.3 API ergonomics

- **OCP** is the raw OCCT binding: verbose, exceptions instead of Pythonic errors,
  handles and downcasts (`TopoDS.Face_s`), but it exposes everything needed for AAG
  construction (surface types, UV evaluation, edge curves, ancestor maps). Best for the
  geometry core and for performance-sensitive loops.
- **CadQuery** wraps OCP with a fluent API (`Workplane`, `hole`, `slot`, `cboreHole`,
  `box`, `cylinder`, …). It is excellent for synthetic part generation and tests, but
  it is a modelling API, not a feature-recognition API; reverse traversal still drops to
  `Shape.Faces()`/`Edges()` and OCP. Keep CadQuery as a **dev/test dependency** and use
  OCP at runtime to keep the hosted install lean.
- **pythonocc-core** has a more Pythonic API than raw OCP and is well documented, but it
  is conda-first. It is not viable for a pip-only Streamlit/hosted deployment without
  vendoring or a conda base image.

### 5.4 Memory and hosting implications

- One OCP worker process: ~0.4 GB RSS after import (measured). With 2–4 workers, plan
  for 1–2 GB plus model/runtime overhead; 32 GB dev RAM is ample, and a 1 GB hosted
  instance is tight but plausible for a single worker.
- Avoid importing CadQuery/VTK in the serving path; VTK alone is a large fraction of the
  2.1 GB venv. Keep `requirements.txt` inference-only and pin OCP + NumPy + PyTorch CPU
  (or ONNX Runtime) + FastAPI.
- OCP wheels are manylinux_2_28; the deployment image must have glibc ≥ 2.28. Verify the
  host before pinning.

---

## 6. Graph similarity for part retrieval

### 6.1 Options

**A. Weisfeiler–Lehman (WL) subtree kernel.** Iteratively relabel each node with its
own label plus the multiset of neighbour labels, then count label occurrences per
iteration; the kernel is the dot product of these histograms. It is a strong, cheap
baseline for attributed graphs and is defined for discrete labels. Continuous geometric
attributes must be quantised into bins or replaced by a learned embedding. Reference:
Shervashidze et al., “Weisfeiler-Lehman Graph Kernels”, JMLR 12 (2011),
[https://www.jmlr.org/papers/v12/shervashidze11a.html](https://www.jmlr.org/papers/v12/shervashidze11a.html)
**[HTTP 200]**. Implementation options: NetworkX provides
`weisfeiler_lehman_graph_hash` and `weisfeiler_lehman_subgraph_hashes` (signatures
verified **[LOCAL]**); GraKeL provides `WeisfeilerLehman` and other kernels
(PyPI `GraKeL 0.1.11`, BSD, cp311 manylinux wheels 2.1 MB **[HTTP 200]**).

**B. Graph edit distance (GED).** Minimum cost of node/edge insertions, deletions, and
substitutions. Exact GED is NP-hard; NetworkX `graph_edit_distance` exists with a
`timeout` parameter (signature verified **[LOCAL]**), but exact GED is only practical for
small graphs or with an upper bound. Use it for validation on small samples, not as the
production index.

**C. Learned embeddings.** Take the GNN’s penultimate face/graph embedding (mean or
attention pooling over face nodes), L2-normalise, and use cosine similarity. This is
cheap at query time and captures geometry/topology that handcrafted histograms miss. It
requires a trained model and a fixed embedding version; store the model hash with the
index. BRepNet’s README explicitly advertises using its embeddings for similarity search
(“use the BRepNet embeddings to search for models containing similar features”,
**[HTTP 200]**); UV-Net demonstrates self-supervised shape embeddings (linear SVM 79.40%,
K-means 58.17% on SolidLetters, UV-Net Table 3). These are evidence that the approach is
established, not a claim that our embeddings will match those numbers.

**D. Vector index.** For ≲100k parts, exact cosine search with a NumPy matrix is simple,
deterministic, and fast enough (60k × 256 float32 ≈ 61 MB). FAISS is the scalable option;
`faiss-cpu 1.16.0` exists on PyPI, the GitHub repo is MIT **[HTTP 200]**, but the cp311
Linux wheel was not confirmed in the PyPI file list during this session — verify before
pinning. Alternatives: HNSWlib, ScaNN, or a managed vector store; all add a dependency
and approximate recall. Do not add one until exact search is measured as too slow.

### 6.2 Evaluation practice (precision@k)

For a query part `q` with a ground-truth set of relevant parts `R(q)` (same family,
same feature multiset, or same manufacturing class):

```
precision@k(q) = |top_k(q) ∩ R(q)| / k
```

Report the mean over queries, plus `recall@k`, mAP, and NDCG if the relevance is graded.
Use **leave-one-out** within a family and **class-held-out** queries (a family never seen
in training) to avoid trivially retrieving near-duplicates. Report `k = 1, 5, 10`, the
embedding/index version, and the exact ground-truth definition. A retrieval metric is
only meaningful if `R(q)` is defined independently of the model (e.g., by the synthetic
generator’s feature list, not by the model’s own cluster).

### 6.3 Recommended similarity stack

1. **Deterministic signature** (always available): face-type histogram, area/volume
   ratios, bounding-box aspect ratios, feature-count vector, WL hash of the discretised
   AAG. Store as a fixed-length vector + hash.
2. **Learned embedding** (when the GNN is trained): pooled penultimate layer, L2-normed.
3. **Index:** exact cosine with NumPy for the first release; FAISS only after measuring
   latency/recall at the real corpus size.
4. **Evaluation:** precision@k / recall@k on a held-out query set, written to
   `results/similarity_eval.json` with the ground-truth definition and index version.

---

## Appendix A — Verification log (2026-11-24)

| URL / resource | Method | Result |
|---|---|---|
| `https://api.crossref.org/works/10.1016/0010-4485(88)90050-4` | GET | 200; title/journal/year verified |
| `https://api.crossref.org/works/10.1016/j.cad.2022.103226` | GET | 200; title/journal/year verified |
| `https://api.crossref.org/works/10.1016/j.rcim.2023.102661` | GET | 200; title/journal/year verified |
| `https://api.crossref.org/works/10.1080/09511921003642121` | GET | 200; Verma & Rajotia 2010 verified |
| `https://api.crossref.org/works/10.14733/cadaps.2020.861-899` | GET | 200; Shi et al. 2020 verified |
| `https://arxiv.org/abs/2006.10211` | GET | 200; UV-Net abstract verified |
| `https://arxiv.org/abs/2104.00706` | GET | 200; BRepNet abstract verified |
| `https://openaccess.thecvf.com/.../UV-Net...pdf` | GET | 200; Table 2 parsed |
| `https://openaccess.thecvf.com/.../BRepNet...pdf` | GET | 200; Table 1 parsed |
| `https://github.com/AutodeskAILab/UV-Net` | API GET | 200; MIT, 214 stars |
| `https://github.com/AutodeskAILab/BRepNet` | API GET | 200; CC BY-NC-SA 4.0, 274 stars |
| `https://github.com/whjdark/AAGNet` | API GET | 200; MIT, 186 stars |
| `https://github.com/hducg/MFCAD` | API GET + `git ls-tree` | 200; 15,488 STEP + 15,488 `.face_truth` |
| `https://github.com/AutodeskAILab/Fusion360GalleryDataset` | API GET | 200; custom license, 765 stars |
| `https://fusion-360-gallery-dataset.s3-us-west-2.amazonaws.com/segmentation/s2.0.0/s2.0.0.zip` | HEAD | 200; 3,326,261,510 bytes |
| `https://raw.githubusercontent.com/AutodeskAILab/Fusion360GalleryDataset/master/LICENSE.md` | GET | 200; non-commercial research-only text verified |
| `https://raw.githubusercontent.com/AutodeskAILab/BRepNet/master/LICENSE` | GET | 200; CC BY-NC-SA 4.0 verified |
| `https://gitlab.com/qub_femg/machine-learning/mfcad2-dataset` | GET/API | 200; README, MIT, labels, benchmark table |
| `https://pure.qub.ac.uk/en/datasets/mfcad-dataset-dataset-for-paper-hierarchical-cadnet-learning-from` | GET | 200; 1.5 GB, CC BY, 59,665 samples |
| `https://pure.qub.ac.uk/files/278385243/MFCAD_dataset.zip` | GET/HEAD | **403** (bot protection) |
| `https://gitlab.com/qub_femg/machine-learning/hierarchical-cadnet` | GET | 200; README verified |
| `https://www.researchsquare.com/article/rs-4908235/latest.pdf` | GET | 200; EAGIS preprint, tables parsed |
| `https://pypi.org/pypi/pythonocc-core/json` | GET | 200; v0.16, **no files** |
| `https://api.anaconda.org/package/conda-forge/pythonocc-core` | GET | 200; 8.0.1.2, LGPL-3.0-or-later |
| `https://pypi.org/pypi/cadquery-ocp/json` | GET | 200; 8.0.1.1.0, cp311 manylinux 67 MB, Apache-2.0 metadata |
| `https://pypi.org/pypi/cadquery/json` | GET | 200; 2.8.0, Apache-2.0, ≥3.11 |
| `https://raw.githubusercontent.com/Open-Cascade-SAS/OCCT/master/LICENSE_LGPL_21.txt` | GET | 200; LGPL-2.1 |
| `https://raw.githubusercontent.com/Open-Cascade-SAS/OCCT/master/OCCT_LGPL_EXCEPTION.txt` | GET | 200; OCCT exception |
| `https://docs.sw.siemens.com/robots.txt` | GET | 200; `Disallow: /` |
| `https://docs.sw.siemens.com/en-US/product/209349590/doc/PL202006250426018413/nxopen_python_ref` | GET | 200 but JS shell / not-found viewer; **no API content** |
| `https://docs.sw.siemens.com/en-US/product/209349590` | GET | **404** |
| `https://www.jmlr.org/papers/v12/shervashidze11a.html` | GET | 200; WL paper verified |
| `https://pypi.org/pypi/GraKeL/json` | GET | 200; 0.1.11, BSD, cp311 wheel 2.1 MB |
| `https://api.github.com/repos/facebookresearch/faiss` | GET | 200; MIT |

## Appendix B — Explicit UNVERIFIED list

1. **“Gavilanes et al. 2018 review”** — no matching record found; do not cite.
2. **All exact NXOpen class/method names and signatures** — official docs inaccessible
   (JS shell + robots disallow); probe on live NX required.
3. **NX journaling vs external-session behaviour** — no official text fetched.
4. **Torch/PyG CUDA availability and versions** — install was still running.
5. **MFInstSeg download URL, size, and license** — only the AAGNet README statement.
6. **Hierarchical CADNet code license** — README does not state one.
7. **EAGIS peer-review status and license** — Research Square preprint; no venue/licence
   metadata verified.
8. **MFCAD label ID → feature-name mapping** — UV-Net gives 16 labels; the mapping file
   was not extracted in this session.
9. **Exact MFCAD dataset license for the STEP files** — repository LICENSE is MIT; a
   separate dataset license was not found.
10. **FAISS cp311 Linux wheel availability** — not confirmed in the PyPI file list.
11. **OCP/OCCT legal conclusion** — licenses verified, but distribution compliance is a
    legal question, not an engineering one.
12. **Peak memory/training time for the GNN** — not measured; only OCP import/box RSS
    was measured (402.9 MB).

---

## 7. Decisions

### D101 — Geometry kernel: `cadquery-ocp` (OCP); CadQuery dev-only

**Options considered:** pythonocc-core; cadquery-ocp; CadQuery; commercial kernels
(Parasolid/ACIS) via NX.

**Decision:** use **OCP** (`cadquery-ocp`) for STEP loading, B-Rep traversal, surface
queries, tessellation, and AAG construction. Use **CadQuery** only for synthetic part
generation and tests (dev dependency).

**Reasons:** pythonocc-core has no PyPI release files (v0.16 stub) and is conda-forge
only; conda is absent. OCP has a verified cp311 manylinux wheel (67 MB), imports in
0.34 s and ~403 MB RSS, and exposes every OCCT query needed. CadQuery adds a pleasant
generation API but is not needed at inference time, keeping the hosted install smaller.

**Risks:** OCP wheel metadata says Apache-2.0 while the bundled OCCT is LGPL-2.1 with
the Open CASCADE exception; legal review required before distribution. manylinux_2_28
requires glibc ≥ 2.28. Pin an exact OCP version; the API is not stability-guaranteed
across major OCCT upgrades.

### D102 — GNN: AAGNet-style gAAG encoder in PyTorch Geometric, with a GraphSAGE/GAT baseline

**Options considered:** UV-Net (UV-grid CNN + GNN); BRepNet (coedge topological walks);
AAGNet (gAAG multi-task); Hierarchical CADNet (mesh + B-Rep hierarchy); plain GraphSAGE/
GAT on AAG.

**Decision:** implement the **gAAG representation** and an **edge-aware PyG model** with
multi-task heads (semantic + instance + bottom-face where labels allow), plus a
**2-layer GraphSAGE or GAT baseline** on the same graphs. Keep the model ≤1M parameters.

**Reasons:** AAGNet is the best fit for the available data and hardware: 0.38–0.44M
parameters, standard graph ops, clean PyG mapping, MIT code, and reported 99.94%/98.84%
on MFInstSeg. UV-Net requires trimming-aware UV sampling and DGL; BRepNet requires
coedge kernels and is CC BY-NC-SA (non-commercial), which conflicts with a publicly
hosted portfolio demo. Hierarchical CADNet needs mesh generation and TensorFlow.

**Risks:** AAGNet’s reference code is DGL + pythonocc-core 7.5.1 / Python 3.10; the PyG
port must be validated numerically. Reported MFCAD++ numbers are secondary citations.
The 8 GB GPU is ample for the model; the bottleneck is STEP→graph conversion. Record
actual peak memory and per-part latency in Phase 4b/7.

### D103 — Dataset: MFCAD++ primary, MFCAD fallback, synthetic CadQuery fallback

**Options considered:** MFCAD++; MFCAD; Fusion 360 Gallery; MFInstSeg; synthetic only.

**Decision:** **MFCAD++** as the primary training/evaluation set; **MFCAD** as the
smoke-test/fallback set; **synthetic CadQuery** parts as the always-available fallback
and adversarial test set. Use **Fusion 360 Gallery** only for an optional BRepNet
comparison, never as the primary dataset.

**Reasons:** MFCAD++ is CC BY, has 59,665 STEP models, 24 feature classes + stock, and
per-face labels embedded in the STEP file. MFCAD is MIT and trivially cloneable but has
only 16 labels. Fusion 360 is non-commercial research-only and labels modeling
operations, not machining features. MFInstSeg is large but its download/licence could
not be verified.

**Risks:** the MFCAD++ zip returns 403 to automated download; a human must fetch it and
the repo must document the exact file/hash. Licence attribution (CC BY) must appear in
the data card and UI. Verify the 59,665 count after extraction; do not hard-code the
commonly cited 59,655.

### D104 — Similarity: WL signature + learned embedding; exact cosine; precision@k

**Options considered:** WL subtree kernel; GED; learned GNN embedding; FAISS/HNSW;
graph database.

**Decision:** ship a deterministic **WL + handcrafted signature** baseline and a
**learned GNN embedding** when the model is trained. Use **exact cosine search with
NumPy** for the first release; add FAISS only if measured latency at the real corpus
size justifies it. Evaluate with **precision@k / recall@k** on held-out queries and
write the results to `results/similarity_eval.json`.

**Reasons:** WL is cheap, deterministic, and explainable; the learned embedding captures
geometry the histogram misses. Exact search over ~60k vectors is small and avoids an
approximate-recall variable. GED is too expensive for production and is kept as a
small-sample validation tool.

**Risks:** discretisation choices change WL results; document bin edges. Learned
embeddings drift with model versions; store the model hash with the index. Ground truth
for “similar” must be defined independently of the model.

### D105 — NX: adapter + mock first; journal-first write-back; all API names UNVERIFIED

**Options considered:** direct NX Open integration now; mock-only; journal replay;
external session.

**Decision:** implement `NxOpenAdapter` behind a typed protocol, with a faithful mock
and `UNTESTED_AGAINST_LIVE_NX` labels everywhere. Use **journal-first** recording for
hole/pocket/slot write-back and **external session** for read-only traversal when a live
NX is available. Do not claim any NXOpen class/method as verified until the probe in
§4.4 is run on the target NX release.

**Reasons:** official Siemens docs are behind a JS shell and `robots.txt: Disallow: /`;
no API text could be fetched. Guessing signatures would violate the project’s
no-fabrication rule. A mock-first adapter keeps CI green and makes the live-NX gap
explicit.

**Risks:** live NX behaviour (licensing, session attachment, builder validation) may
differ from the mock; the first live run must be treated as a discovery exercise, not a
test pass. Parametric write-back may be impossible for some feature variants; fail
loudly and fall back to read-only analysis.
