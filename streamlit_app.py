"""BRepLens Streamlit demo (CPU-only, hosted-demo friendly).

Design constraints:
* no CadQuery/VTK import (OCP only) to stay within the hosted memory budget,
* models and sample graphs cached with st.cache_resource / st.cache_data,
* all recognition runs in-process; uploads are validated before OCP sees them,
* all user-facing copy uses ASCII punctuation: no em dashes.
"""

from __future__ import annotations

import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from breplens.adapters.base import CadLoadError  # noqa: E402
from breplens.ml.family import FAMILY_NAMES, load_cnn  # noqa: E402
from breplens.ml.views import render_views  # noqa: E402
from breplens.recognition.gnn_recognizer import recognize_gnn, recognize_hybrid  # noqa: E402
from breplens.recognition.rules import recognize_rules  # noqa: E402
from breplens.schema import FeatureSet, MeshData, PartGraph  # noqa: E402
from breplens.similarity.signature import SignatureIndex, part_signature  # noqa: E402

try:
    from breplens.adapters.occ import OccAdapter

    CAD_IMPORT_ERROR: str | None = None
except Exception as exc:
    OccAdapter = None  # type: ignore[assignment,misc]
    CAD_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

MAX_UPLOAD_MB = 25
MAX_FACES = 3000
SAMPLES_DIR = ROOT / "samples"
LIBRARY_PATH = ROOT / "demo" / "library.json"
MODEL_PATH = ROOT / "models" / "gnn_synthetic.pt"

TYPE_COLORS = {
    "through_hole": "#4ea1ff",
    "blind_hole": "#22d3ee",
    "counterbore": "#a78bfa",
    "countersink": "#fb923c",
    "pocket": "#4ade80",
    "slot": "#a3e635",
    "step": "#f59e0b",
    "boss": "#fbbf24",
    "fillet": "#fb7185",
    "chamfer": "#f472b6",
    "base": "#64748b",
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
MODE_INFO = {
    "hybrid": ("Hybrid", "Rules first, then high-confidence GNN recovery. Best default."),
    "rules": ("Rules", "Deterministic geometry. Fast and interpretable."),
    "gnn": ("GNN", "Learned face classification plus measured geometry."),
}
SAMPLE_LABELS = {
    "sample_00_plate_holes.step": "Plate with 4 through holes",
    "sample_01_plate_blind.step": "Plate with blind hole",
    "sample_02_counterbore.step": "Counterbore",
    "sample_03_countersink.step": "Countersink",
    "sample_04_pocket.step": "Rectangular pocket",
    "sample_05_slot.step": "Rounded slot",
    "sample_06_stepped_block.step": "Stepped block",
    "sample_07_boss_plate.step": "Round boss",
    "sample_08_fillet_block.step": "Filleted block",
    "sample_09_bracket_mixed.step": "Mixed bracket",
}


def _safe(text: object) -> str:
    """Keep user-facing copy free of em/en dashes even for backend messages."""
    return str(text).replace("\u2014", "-").replace("\u2013", "-")


@st.cache_resource(show_spinner=False)
def get_adapter() -> OccAdapter:
    if OccAdapter is None:
        raise RuntimeError(f"CAD kernel unavailable: {CAD_IMPORT_ERROR}")
    return OccAdapter()


@st.cache_resource(show_spinner="Loading GNN weights...")
def get_model():
    try:
        from breplens.ml.inference import load_for_inference

        return load_for_inference(MODEL_PATH, device="cpu")
    except Exception as exc:  # pragma: no cover - deployment diagnosis
        return None, None, {"error": f"{type(exc).__name__}: {exc}"}


@st.cache_resource(show_spinner=False)
def get_family_model():
    path = ROOT / "models" / "family_cnn.pt"
    if not path.exists():
        return None, None
    try:
        return load_cnn(path, device="cpu")
    except Exception:
        return None, None


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


@st.cache_data(show_spinner="Recognizing features...", max_entries=24)
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


@st.cache_data(show_spinner=False, max_entries=8)
def compare_modes_cached(graph_json: str) -> dict[str, dict]:
    return {mode: recognize_cached(graph_json, mode) for mode in ("rules", "gnn", "hybrid")}


@st.cache_data(show_spinner=False, max_entries=24)
def family_prediction_cached(mesh_json: str) -> tuple[str | None, float | None]:
    model, payload = get_family_model()
    if model is None or payload is None:
        return None, None
    import torch

    mesh = MeshData.model_validate_json(mesh_json)
    meta = payload.get("metadata", {})
    norm = meta.get("image_norm", {})
    mean = torch.tensor(norm.get("mean", [0.5] * 6), dtype=torch.float32).view(1, -1, 1, 1)
    std = torch.tensor(norm.get("std", [0.5] * 6), dtype=torch.float32).view(1, -1, 1, 1)
    views = render_views(mesh)
    x = (torch.from_numpy(views).float().unsqueeze(0) / 255.0 - mean) / std
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=-1)[0]
    idx = int(probs.argmax())
    return FAMILY_NAMES[idx], float(probs[idx])


def _styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1420px;}
        [data-testid="stMetric"] {
            background: rgba(127,127,127,.055);
            border: 1px solid rgba(127,127,127,.22);
            border-radius: 10px;
            padding: 10px 12px;
        }
        [data-testid="stMetricLabel"] {font-size: .76rem; letter-spacing: .03em;}
        [data-testid="stMetricValue"] {font-size: 1.55rem;}
        .stTabs [data-baseweb="tab-list"] {gap: .35rem; border-bottom: 1px solid rgba(148,163,184,.18);}
        .stTabs [data-baseweb="tab"] {border-radius: 9px 9px 0 0; padding: 8px 14px;}
        [data-testid="stSidebar"] {border-right: 1px solid rgba(148,163,184,.18);}
        .small-note {color:#94a3b8; font-size:.78rem;}
        .feature-chip {
            display:inline-block; padding:2px 9px; border-radius:999px; font-size:.72rem;
            border:1px solid rgba(148,163,184,.35); margin:2px 4px 2px 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _header() -> None:
    st.title("BRepLens")
    st.caption("Machining feature recognition and measured parameter extraction for STEP parts.")
    st.caption(
        "STEP AP214 / AP242   |   schema 1.0.0   |   CPU inference   |   rules / GNN / hybrid"
    )
    st.divider()


def _mesh_figure(mesh: dict, fs: FeatureSet, *, base_opacity: float = 0.55) -> go.Figure:
    vertices = np.asarray(mesh["vertices"], dtype=float)
    triangles = np.asarray(mesh["triangles"], dtype=int)
    tri_faces = np.asarray(mesh["triangle_face_ids"], dtype=int)
    face_type: dict[int, str] = {}
    for feat in fs.features:
        for fid in feat.face_ids:
            face_type.setdefault(fid, feat.type.value)
    labels = np.array([face_type.get(int(fid), "base") for fid in tri_faces], dtype=object)
    fig = go.Figure()
    for key in TYPE_COLORS:
        mask = labels == key
        if not mask.any():
            continue
        if key == "base" and base_opacity <= 0.01:
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
                opacity=(base_opacity if key == "base" else 1.0),
                name=TYPE_LABELS[key],
                legendgroup=key,
                showlegend=True,
                flatshading=True,
                lighting=dict(ambient=0.6, diffuse=0.85, specular=0.25, roughness=0.55),
            )
        )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=590,
        scene=dict(
            aspectmode="data",
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
            bgcolor="#0b1220",
            xaxis=dict(gridcolor="#1e293b"),
            yaxis=dict(gridcolor="#1e293b"),
            zaxis=dict(gridcolor="#1e293b"),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=0.0, x=0.0, font=dict(color="#e2e8f0")),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e2e8f0",
    )
    return fig


def _similarity_figure(hits: list[dict]) -> go.Figure:
    ordered = list(reversed(hits))
    fig = go.Figure(
        go.Bar(
            x=[h["score"] for h in ordered],
            y=[h["part"] for h in ordered],
            orientation="h",
            marker_color="#38bdf8",
            text=[f"{h['score']:.3f}" for h in ordered],
            textposition="outside",
            hovertemplate="%{y}<br>similarity %{x:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=max(220, 42 * len(ordered)),
        margin=dict(l=10, r=30, t=10, b=10),
        xaxis=dict(range=[0, 1], title="Cosine similarity", gridcolor="#1e293b"),
        yaxis_title="",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#cbd5e1",
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
                "confidence": round(f.confidence, 3),
                "diameter": p.diameter,
                "depth": p.depth,
                "radius": p.radius,
                "length": p.length,
                "width": p.width,
                "height": p.height,
                "angle (deg)": p.angle_deg,
                "faces": len(f.face_ids),
                "position (mm)": (
                    f"({f.position.x:.2f}, {f.position.y:.2f}, {f.position.z:.2f})"
                    if f.position
                    else "-"
                ),
            }
        )
    return rows


def _feature_table(fs: FeatureSet) -> pd.DataFrame:
    rows = _feature_rows(fs)
    if not rows:
        return pd.DataFrame(columns=["id", "type", "confidence", "faces", "position (mm)"])
    df = pd.DataFrame(rows)
    keep = ["id", "type", "confidence"]
    for col in ("diameter", "depth", "radius", "length", "width", "height", "angle (deg)"):
        if col in df and df[col].notna().any():
            keep.append(col)
    keep.extend(["faces", "position (mm)"])
    return df[keep]


def _feature_type_counts(fs: FeatureSet) -> Counter[str]:
    return Counter(f.type.value for f in fs.features)


def _similarity(query_graph: PartGraph, fs: FeatureSet, k: int) -> tuple[list[dict], int]:
    index = get_library()
    if not index.entries:
        return [], 0
    vec = part_signature(query_graph, fs)
    hits = [
        {
            "part": h.name,
            "family": h.family,
            "score": round(h.score, 4),
            "faces": h.metadata.get("n_faces"),
        }
        for h in index.search(vec, k=k)
    ]
    return hits, len(index.entries)


def _select_sample(name: str) -> None:
    st.session_state["sample_choice"] = name


def _landing() -> None:
    st.subheader("Choose a sample part")
    st.caption("Pick a generated part below, or upload a STEP file from the sidebar.")
    names = sorted(SAMPLE_LABELS)
    for row_start in (0, 5):
        cols = st.columns(5)
        for col, name in zip(cols, names[row_start : row_start + 5], strict=False):
            with col:
                st.button(
                    SAMPLE_LABELS[name],
                    key=f"open_{name}",
                    on_click=_select_sample,
                    args=(name,),
                    width="stretch",
                )
    st.divider()
    st.markdown("**Workflow**")
    st.caption(
        "1. Load a part.  2. Pick an engine.  3. Inspect and compare features.  4. Export JSON or CSV."
    )


def _sidebar() -> tuple[str, object, str, int, float, float, bool]:
    with st.sidebar:
        st.markdown("### Part input")
        sample_names = sorted(SAMPLE_LABELS)
        choice = st.selectbox(
            "Built-in sample",
            ["Upload a STEP file", *sample_names],
            key="sample_choice",
            help="Samples are generated by BRepLens and committed under samples/.",
        )
        upload = st.file_uploader(
            "Upload STEP",
            type=["step", "stp"],
            help=f"ISO-10303-21 STEP file, max {MAX_UPLOAD_MB} MiB and {MAX_FACES} faces.",
        )
        st.markdown("### Recognition")
        mode = st.radio(
            "Engine",
            ["hybrid", "rules", "gnn"],
            key="mode",
            help="Hybrid is the default. Rules is fastest. GNN is the learned face classifier.",
        )
        st.caption(MODE_INFO[mode][1])
        st.markdown("### Viewer")
        deflection = st.slider("Mesh detail (mm)", 0.05, 1.0, 0.25, 0.05)
        base_opacity = st.slider("Base surface opacity", 0.0, 1.0, 0.55, 0.05)
        show_base = st.toggle("Show base surfaces", value=True)
        st.markdown("### Retrieval")
        k = st.slider("Similar parts", 1, 10, 5)
        st.divider()
        st.markdown(
            '<div class="small-note">Schema v1.0.0<br>GNN checkpoint: models/gnn_synthetic.pt'
            "<br>Family CNN: models/family_cnn.pt<br>NX write-back is not available in this demo.</div>",
            unsafe_allow_html=True,
        )
    return choice, upload, mode, k, deflection, base_opacity, show_base


def _resolve_input(choice: str, upload: object) -> tuple[bytes | None, str]:
    if upload is not None:
        if upload.size > MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"File is larger than the {MAX_UPLOAD_MB} MiB demo limit.")
            return None, ""
        candidate = upload.getvalue()
        if not candidate.lstrip().startswith(b"ISO-10303-21"):
            st.error(
                "This is not an ISO-10303-21 STEP file. Export as STEP AP214 or AP242 and retry."
            )
            return None, ""
        return candidate, upload.name
    if choice != "Upload a STEP file":
        path = SAMPLES_DIR / choice
        if not path.exists():
            st.error("The selected sample is missing from this deployment.")
            return None, ""
        return path.read_bytes(), path.name
    return None, ""


def _render_overview(
    graph: PartGraph,
    fs: FeatureSet,
    mesh: dict | None,
    family: tuple[str | None, float | None],
    *,
    base_opacity: float,
    show_base: bool,
) -> None:
    left, right = st.columns([1.35, 1], gap="large")
    with left:
        st.subheader("3D view")
        if mesh is None:
            st.warning("Tessellation is unavailable for this part.")
        else:
            opacity = base_opacity if show_base else 0.0
            st.plotly_chart(
                _mesh_figure(mesh, fs, base_opacity=opacity),
                width="stretch",
                config={"displaylogo": False, "responsive": True},
            )
    with right:
        st.subheader("At a glance")
        fam_label, fam_conf = family
        if fam_label:
            st.markdown("**Part family**")
            st.progress(min(1.0, max(0.0, fam_conf or 0.0)), text=f"{fam_label} ({fam_conf:.0%})")
        bbox = graph.metadata.bbox_max - graph.metadata.bbox_min
        c1, c2 = st.columns(2)
        c1.metric("Bounding box", f"{bbox.x:.1f} x {bbox.y:.1f} x {bbox.z:.1f} mm")
        c2.metric("Volume", f"{(graph.metadata.volume or 0) / 1000:.1f} cm3")
        st.markdown("**Feature mix**")
        counts = _feature_type_counts(fs)
        if counts:
            chips = " ".join(
                f'<span class="feature-chip" style="border-color:{TYPE_COLORS[k]};color:{TYPE_COLORS[k]}">'
                f"{TYPE_LABELS[k]} {v}</span>"
                for k, v in sorted(counts.items())
            )
            st.markdown(chips, unsafe_allow_html=True)
        else:
            st.caption("No machining features were recognized.")
        st.markdown("**Quick feature table**")
        table = _feature_table(fs)
        if len(table):
            st.dataframe(table.head(6), width="stretch", hide_index=True)
        if fs.warnings:
            with st.expander(f"Warnings ({len(fs.warnings)})", expanded=False):
                for warning in fs.warnings:
                    st.write(f"- {_safe(warning)}")


def _render_features(fs: FeatureSet, name: str) -> None:
    st.subheader("Measured feature set")
    if not fs.features:
        st.warning("No machining features were recognized in this part.")
        return
    all_types = sorted({f.type.value for f in fs.features})
    c1, c2 = st.columns([2, 1])
    with c1:
        chosen = st.multiselect(
            "Feature types",
            options=all_types,
            default=all_types,
            format_func=lambda key: TYPE_LABELS.get(key, key),
        )
    with c2:
        min_conf = st.slider("Minimum confidence", 0.0, 1.0, 0.0, 0.05)
    filtered = [f for f in fs.features if f.type.value in chosen and f.confidence >= min_conf]
    if not filtered:
        st.info("No features match the current filter.")
        return
    filtered_set = fs.model_copy(update={"features": filtered})
    table = _feature_table(filtered_set)
    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        column_config={
            "confidence": st.column_config.ProgressColumn(
                "confidence", min_value=0.0, max_value=1.0, format="%.2f"
            ),
            "diameter": st.column_config.NumberColumn("diameter (mm)", format="%.2f"),
            "depth": st.column_config.NumberColumn("depth (mm)", format="%.2f"),
            "radius": st.column_config.NumberColumn("radius (mm)", format="%.2f"),
            "length": st.column_config.NumberColumn("length (mm)", format="%.2f"),
            "width": st.column_config.NumberColumn("width (mm)", format="%.2f"),
            "height": st.column_config.NumberColumn("height (mm)", format="%.2f"),
            "angle (deg)": st.column_config.NumberColumn("angle (deg)", format="%.1f"),
        },
    )
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        st.download_button(
            "Download JSON",
            data=fs.model_dump_json(indent=2),
            file_name=f"{Path(name).stem}_features.json",
            mime="application/json",
            width="stretch",
        )
    with c2:
        st.download_button(
            "Download CSV",
            data=table.to_csv(index=False).encode("utf-8"),
            file_name=f"{Path(name).stem}_features.csv",
            mime="text/csv",
            width="stretch",
        )
    with c3:
        st.caption(
            f"{len(filtered)} of {len(fs.features)} features shown. "
            f"Mode {fs.mode}. Schema {fs.schema_version}."
        )


def _render_compare(graph: PartGraph) -> None:
    st.subheader("Compare engines")
    st.caption(
        "All three engines run on the same graph. Counts and timings are measured in this session."
    )
    modes = compare_modes_cached(graph.model_dump_json())
    cols = st.columns(3)
    for col, key in zip(cols, ("rules", "gnn", "hybrid"), strict=True):
        fs = FeatureSet.model_validate(modes[key])
        with col:
            st.metric(
                MODE_INFO[key][0],
                f"{len(fs.features)} features",
                help=MODE_INFO[key][1],
            )
            st.caption(f"{fs.timings.recognition_ms:.2f} ms")
    all_types = sorted(
        {f.type.value for mode in modes.values() for f in FeatureSet.model_validate(mode).features}
    )
    rows = []
    for key in all_types:
        rows.append(
            {
                "Feature type": TYPE_LABELS.get(key, key),
                "rules": sum(
                    1
                    for f in FeatureSet.model_validate(modes["rules"]).features
                    if f.type.value == key
                ),
                "gnn": sum(
                    1
                    for f in FeatureSet.model_validate(modes["gnn"]).features
                    if f.type.value == key
                ),
                "hybrid": sum(
                    1
                    for f in FeatureSet.model_validate(modes["hybrid"]).features
                    if f.type.value == key
                ),
            }
        )
    if rows:
        frame = pd.DataFrame(rows)
        fig = go.Figure()
        for mode, color in (("rules", "#38bdf8"), ("gnn", "#a78bfa"), ("hybrid", "#4ade80")):
            fig.add_bar(name=mode, x=frame["Feature type"], y=frame[mode], marker_color=color)
        fig.update_layout(
            barmode="group",
            height=330,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis_title="Feature count",
            xaxis_title="",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#cbd5e1",
            legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
        st.dataframe(frame, width="stretch", hide_index=True)
    else:
        st.info("All engines returned zero features on this part.")


def _render_similar(graph: PartGraph, fs: FeatureSet, k: int) -> None:
    st.subheader("Similar parts")
    st.caption("Nearest neighbors by block-weighted geometry and Weisfeiler-Lehman signature.")
    try:
        hits, library_size = _similarity(graph, fs, k)
    except Exception as exc:
        st.warning(f"Similarity search failed: {_safe(exc)}")
        return
    if not hits:
        st.info("Similarity library is not available in this deployment.")
        return
    st.caption(f"Library size: {library_size} parts. Showing top {len(hits)}.")
    left, right = st.columns([1, 1.1], gap="large")
    with left:
        st.plotly_chart(_similarity_figure(hits), width="stretch", config={"displaylogo": False})
    with right:
        st.dataframe(
            hits,
            width="stretch",
            hide_index=True,
            column_config={
                "score": st.column_config.ProgressColumn(
                    "similarity", min_value=0.0, max_value=1.0, format="%.3f"
                ),
            },
        )


def _render_json(fs: FeatureSet, name: str) -> None:
    st.subheader("Feature JSON")
    st.caption(
        "Versioned, validated output. This is the same payload served by the FastAPI service."
    )
    st.download_button(
        "Download feature JSON",
        data=fs.model_dump_json(indent=2),
        file_name=f"{Path(name).stem}_features.json",
        mime="application/json",
    )
    st.json(fs.model_dump(mode="json"), expanded=False)


def main() -> None:
    st.set_page_config(
        page_title="BRepLens",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _styles()
    _header()
    if CAD_IMPORT_ERROR:
        st.error("The CAD kernel could not load in this environment.")
        st.code(CAD_IMPORT_ERROR)
        st.info(
            "Redeploy this app with Python 3.12. The project supports Python 3.11 "
            "and 3.12. Streamlit cannot change Python in an existing app, so delete "
            "the app and create it again with Python 3.12 selected in Advanced settings."
        )
        return
    choice, upload, mode, k, deflection, base_opacity, show_base = _sidebar()
    data, name = _resolve_input(choice, upload)
    if data is None:
        _landing()
        return

    try:
        graph_dict = load_graph_cached(data, name)
        graph = PartGraph.model_validate(graph_dict)
    except CadLoadError as exc:
        st.error(f"Could not read this STEP file: {_safe(exc)}")
        _landing()
        return
    except Exception as exc:
        st.error(f"Unexpected error while reading the file: {_safe(exc)}")
        _landing()
        return

    if len(graph.faces) > MAX_FACES:
        st.error(f"Part has {len(graph.faces)} faces. The demo limit is {MAX_FACES}.")
        _landing()
        return

    try:
        fs = FeatureSet.model_validate(recognize_cached(graph.model_dump_json(), mode))
    except Exception as exc:
        st.error(f"Recognition failed: {_safe(exc)}")
        return

    mesh: dict | None = None
    family: tuple[str | None, float | None] = (None, None)
    try:
        mesh = tessellate_cached(data, deflection)
        family = family_prediction_cached(json.dumps(mesh))
    except Exception as exc:
        st.warning(f"Tessellation or family inference failed: {_safe(exc)}")

    bbox = graph.metadata.bbox_max - graph.metadata.bbox_min
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Features", len(fs.features), help="Recognized machining features in the selected mode."
    )
    m2.metric("Faces / edges", f"{len(graph.faces)} / {len(graph.edges)}")
    m3.metric(
        "Recognition",
        f"{fs.timings.recognition_ms:.2f} ms",
        help="CPU recognition time, excluding tessellation.",
    )
    m4.metric(
        "Part family",
        family[0] or "n/a",
        delta=f"{family[1]:.0%} confidence" if family[0] and family[1] else None,
        help="Optional six-view CNN prediction on synthetic families.",
    )
    st.caption(
        f"{name} | bbox {bbox.x:.1f} x {bbox.y:.1f} x {bbox.z:.1f} mm | "
        f"mode {fs.mode} | schema {fs.schema_version}"
    )

    tabs = st.tabs(["Overview", "Features", "Compare engines", "Similar parts", "JSON"])
    with tabs[0]:
        _render_overview(
            graph,
            fs,
            mesh,
            family,
            base_opacity=base_opacity,
            show_base=show_base,
        )
    with tabs[1]:
        _render_features(fs, name)
    with tabs[2]:
        _render_compare(graph)
    with tabs[3]:
        _render_similar(graph, fs, k)
    with tabs[4]:
        _render_json(fs, name)


if __name__ == "__main__":
    main()
