"""BRepLens Streamlit demo (CPU-only, hosted-demo friendly).

Design constraints:
* no CadQuery/VTK import (OCP only) to stay within ~1.5 GiB peak RSS,
* models and sample graphs cached with ``st.cache_resource`` / ``st.cache_data``,
* all recognition runs in-process; uploads are validated before OCP sees them.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from breplens.adapters.base import CadLoadError  # noqa: E402
from breplens.adapters.occ import OccAdapter  # noqa: E402
from breplens.recognition.gnn_recognizer import recognize_gnn, recognize_hybrid  # noqa: E402
from breplens.recognition.rules import recognize_rules  # noqa: E402
from breplens.schema import FeatureSet, PartGraph  # noqa: E402
from breplens.similarity.signature import SignatureIndex, part_signature  # noqa: E402

MAX_UPLOAD_MB = 25
MAX_FACES = 3000
SAMPLES_DIR = ROOT / "samples"
LIBRARY_PATH = ROOT / "demo" / "library.json"
MODEL_PATH = ROOT / "models" / "gnn_synthetic.pt"

TYPE_COLORS = {
    "through_hole": "#1f77b4",
    "blind_hole": "#17becf",
    "counterbore": "#9467bd",
    "countersink": "#8c564b",
    "pocket": "#2ca02c",
    "slot": "#98df8a",
    "step": "#ff7f0e",
    "boss": "#ffbb78",
    "fillet": "#d62728",
    "chamfer": "#e377c2",
    "base": "#bdbdbd",
}
TYPE_LABELS = {
    "through_hole": "Through hole",
    "blind_hole": "Blind hole",
    "counterbore": "Counterbore",
    "countersink": "Countersink",
    "pocket": "Pocket",
    "slot": "Slot",
    "step": "Step",
    "boss": "Boss",
    "fillet": "Fillet",
    "chamfer": "Chamfer",
    "base": "Base / unrecognized",
}


@st.cache_resource(show_spinner=False)
def get_adapter() -> OccAdapter:
    return OccAdapter()


@st.cache_resource(show_spinner="Loading GNN weights…")
def get_model():
    try:
        from breplens.ml.inference import load_for_inference

        return load_for_inference(MODEL_PATH, device="cpu")
    except Exception as exc:  # pragma: no cover - deployment diagnosis
        return None, None, {"error": f"{type(exc).__name__}: {exc}"}


@st.cache_resource(show_spinner=False)
def get_library() -> SignatureIndex:
    return SignatureIndex.load(LIBRARY_PATH) if LIBRARY_PATH.exists() else SignatureIndex()


@st.cache_data(show_spinner=False, max_entries=16)
def load_graph_cached(data: bytes, name: str) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".step", delete=False) as fh:
        fh.write(data)
        tmp = Path(fh.name)
    try:
        graph = get_adapter().load_graph(tmp, name=name)
    finally:
        tmp.unlink(missing_ok=True)
    return json.loads(graph.model_dump_json())


@st.cache_data(show_spinner=False, max_entries=16)
def tessellate_cached(data: bytes, deflection: float) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".step", delete=False) as fh:
        fh.write(data)
        tmp = Path(fh.name)
    try:
        mesh = get_adapter().tessellate(tmp, deflection=deflection)
    finally:
        tmp.unlink(missing_ok=True)
    return json.loads(mesh.model_dump_json())


@st.cache_data(show_spinner="Recognizing features…", max_entries=24)
def recognize_cached(graph_json: str, mode: str) -> dict:
    graph = PartGraph.model_validate_json(graph_json)
    if mode == "rules":
        fs = recognize_rules(graph)
    else:
        model, classes, meta = get_model()
        if model is None:
            fs = recognize_rules(graph)
            fs.warnings.append(
                f"GNN unavailable in this deployment; used rules ({meta.get('error')})"
            )
        elif mode == "gnn":
            fs = recognize_gnn(graph, model, classes, device="cpu")
        else:
            fs = recognize_hybrid(graph, model, classes, device="cpu")
    return json.loads(fs.model_dump_json())


def _mesh_figure(mesh: dict, fs: FeatureSet, deflection: float) -> go.Figure:
    vertices = np.asarray(mesh["vertices"], dtype=float)
    triangles = np.asarray(mesh["triangles"], dtype=int)
    tri_faces = np.asarray(mesh["triangle_face_ids"], dtype=int)
    face_type: dict[int, str] = {}
    for feat in fs.features:
        for fid in feat.face_ids:
            face_type.setdefault(fid, feat.type.value)
    labels = np.array([face_type.get(int(fid), "base") for fid in tri_faces], dtype=object)
    fig = go.Figure()
    for key in [*TYPE_COLORS.keys()]:
        mask = labels == key
        if not mask.any():
            continue
        tri = triangles[mask]
        fig.add_trace(
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=tri[:, 0],
                j=tri[:, 1],
                k=tri[:, 2],
                color=TYPE_COLORS[key],
                opacity=1.0 if key != "base" else 0.55,
                name=TYPE_LABELS[key],
                legendgroup=key,
                showlegend=True,
                flatshading=True,
                lighting=dict(ambient=0.55, diffuse=0.8, specular=0.25, roughness=0.6),
            )
        )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=560,
        scene=dict(
            aspectmode="data",
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
            bgcolor="#0e1117",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=0.0, x=0.0),
        paper_bgcolor="#0e1117",
        font_color="#fafafa",
    )
    return fig


def _feature_rows(fs: FeatureSet) -> list[dict]:
    rows = []
    for f in fs.features:
        p = f.params
        rows.append(
            {
                "id": f.id,
                "type": TYPE_LABELS.get(f.type.value, f.type.value),
                "confidence": round(f.confidence, 2),
                "diameter": p.diameter,
                "depth": p.depth,
                "radius": p.radius,
                "length": p.length,
                "width": p.width,
                "height": p.height,
                "angle°": p.angle_deg,
                "faces": len(f.face_ids),
                "position (mm)": (
                    f"({f.position.x:.2f}, {f.position.y:.2f}, {f.position.z:.2f})"
                    if f.position
                    else "—"
                ),
            }
        )
    return rows


def _similarity(query_graph: PartGraph, fs: FeatureSet, k: int) -> list[dict]:
    index = get_library()
    if not index.entries:
        return []
    vec = part_signature(query_graph, fs)
    return [
        {
            "part": h.name,
            "family": h.family,
            "score": round(h.score, 3),
            "faces": h.metadata.get("n_faces"),
        }
        for h in index.search(vec, k=k)
    ]


def main() -> None:
    st.set_page_config(page_title="BRepLens", page_icon="🔍", layout="wide")
    st.title("BRepLens — machining feature recognition")
    st.caption(
        "STEP → Attributed Adjacency Graph → rules / GNN / hybrid → measured features. "
        "Everything runs on CPU in this demo."
    )

    with st.sidebar:
        st.header("Input")
        sample_names = sorted(p.name for p in SAMPLES_DIR.glob("*.step"))
        choice = st.selectbox(
            "Built-in sample",
            ["Upload a STEP file", *sample_names],
            help="Samples are generated by BRepLens (committed, redistributable).",
        )
        upload = st.file_uploader("…or upload STEP (.step/.stp, ≤25 MiB)", type=["step", "stp"])
        mode = st.radio(
            "Recognizer",
            ["hybrid", "rules", "gnn"],
            index=0,
            help="rules = deterministic geometry; gnn = learned face classification; hybrid = rules + GNN recovery.",
        )
        k = st.slider("Similar parts", 1, 10, 5)
        deflection = st.slider("Mesh detail (mm)", 0.05, 1.0, 0.25, 0.05)
        st.divider()
        st.caption("GNN weights: models/gnn_synthetic.pt (committed, CPU-trained export).")

    data: bytes | None = None
    name = ""
    if upload is not None:
        if upload.size > MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"File is larger than the {MAX_UPLOAD_MB} MiB demo limit.")
        else:
            candidate = upload.getvalue()
            if not candidate.lstrip().startswith(b"ISO-10303-21"):
                st.error(
                    "This is not an ISO-10303-21 STEP file. Export as STEP AP214/AP242 and retry."
                )
            else:
                data = candidate
                name = upload.name
    elif choice != "Upload a STEP file":
        path = SAMPLES_DIR / choice
        data = path.read_bytes()
        name = path.name

    if data is None:
        st.info("Choose one of the 10 built-in samples or upload a STEP file to begin.")
        st.markdown(
            """
**What you can do**
- Inspect feature-colored 3D geometry and measured parameters
- Download the validated feature JSON (`schema_version` 1.0.0)
- Compare rule-based, GNN and hybrid recognition
- Retrieve similar parts from a 60-part signature library
"""
        )
        return

    try:
        graph_dict = load_graph_cached(data, name)
        graph = PartGraph.model_validate(graph_dict)
    except CadLoadError as exc:
        st.error(f"Could not read this STEP file: {exc}")
        return
    except Exception as exc:
        st.error(f"Unexpected error while reading the file: {type(exc).__name__}: {exc}")
        return

    if len(graph.faces) > MAX_FACES:
        st.error(f"Part has {len(graph.faces)} faces; the demo limit is {MAX_FACES}.")
        return

    try:
        fs = FeatureSet.model_validate(recognize_cached(graph.model_dump_json(), mode))
    except Exception as exc:
        st.error(f"Recognition failed: {type(exc).__name__}: {exc}")
        return

    col_left, col_right = st.columns([1.2, 1])

    with col_right:
        st.subheader("Measured features")
        if fs.features:
            st.dataframe(_feature_rows(fs), width="stretch", hide_index=True)
        else:
            st.warning("No machining features were recognized in this part.")
        st.caption(
            f"{len(graph.faces)} faces · {len(graph.edges)} edges · "
            f"recognition {fs.timings.recognition_ms:.1f} ms · mode **{fs.mode}**"
        )
        st.download_button(
            "⬇ Download feature JSON",
            data=fs.model_dump_json(indent=2),
            file_name=f"{Path(name).stem}_features.json",
            mime="application/json",
        )
        if fs.warnings:
            with st.expander("Warnings", expanded=False):
                for w in fs.warnings:
                    st.write("•", w)
        if fs.model_version:
            st.caption(f"Model: {fs.model_version}")

    with col_left:
        st.subheader("3D view")
        try:
            mesh = tessellate_cached(data, deflection)
            st.plotly_chart(_mesh_figure(mesh, fs, deflection), width="stretch")
        except Exception as exc:
            st.warning(f"3D tessellation unavailable: {type(exc).__name__}: {exc}")

    st.subheader("Similar parts")
    try:
        hits = _similarity(graph, fs, k)
        if hits:
            st.dataframe(hits, width="stretch", hide_index=True)
        else:
            st.info("Similarity library is not available in this deployment.")
    except Exception as exc:
        st.warning(f"Similarity search failed: {type(exc).__name__}: {exc}")

    with st.expander("Feature JSON preview"):
        st.json(fs.model_dump(mode="json"))


if __name__ == "__main__":
    main()
