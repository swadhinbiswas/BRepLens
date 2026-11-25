"""Graph exports for the neutral Attributed Adjacency Graph.

The neutral IR (:class:`breplens.schema.PartGraph`) is converted on demand to
NetworkX (visualization) or PyTorch Geometric / tensors (learning). Feature
vectors are deterministic and documented in ``FEATURE_NAMES`` so training and
CPU inference cannot drift.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from breplens.schema import EdgeConvexity, PartGraph, SurfaceType, Vec3

if TYPE_CHECKING:  # pragma: no cover
    import networkx as nx

SURFACE_TYPES: tuple[str, ...] = tuple(t.value for t in SurfaceType)
CONVEXITIES: tuple[str, ...] = tuple(c.value for c in EdgeConvexity)

# Node features: one-hot surface (7) + log area + normal (3) + axis (3) +
# log radius + log bbox diagonal + centroid offset (3) + is_outer.
FEATURE_NAMES: tuple[str, ...] = (
    *(f"surface_{t}" for t in SURFACE_TYPES),
    "log_area",
    "normal_x",
    "normal_y",
    "normal_z",
    "axis_x",
    "axis_y",
    "axis_z",
    "log_radius",
    "log_bbox_diag",
    "centroid_dx",
    "centroid_dy",
    "centroid_dz",
    "is_outer",
)

# Edge features: one-hot convexity (4) + log length + normalized dihedral (2).
EDGE_FEATURE_NAMES: tuple[str, ...] = (
    *(f"convexity_{c}" for c in CONVEXITIES),
    "log_length",
    "sin_dihedral",
    "cos_dihedral",
)


def _safe_log(x: float, scale: float = 100.0) -> float:
    return float(math.log1p(max(0.0, x)) / math.log1p(scale))


def _bbox_diag(graph: PartGraph) -> float:
    d = graph.metadata.bbox_max - graph.metadata.bbox_min
    return max(d.norm(), 1e-9)


def node_features(graph: PartGraph) -> np.ndarray:
    """Return an ``(n_faces, len(FEATURE_NAMES))`` float32 matrix."""
    diag = _bbox_diag(graph)
    center = (graph.metadata.bbox_max + graph.metadata.bbox_min) * 0.5
    rows: list[list[float]] = []
    for f in graph.faces:
        row = [0.0] * len(FEATURE_NAMES)
        row[SURFACE_TYPES.index(f.surface_type.value)] = 1.0
        base = len(SURFACE_TYPES)
        row[base] = _safe_log(f.area, 1e5)
        n = f.normal or Vec3()
        row[base + 1 : base + 4] = [n.x, n.y, n.z]
        ax = f.axis_direction or Vec3()
        row[base + 4 : base + 7] = [ax.x, ax.y, ax.z]
        row[base + 7] = _safe_log(f.radius or 0.0, 100.0)
        d = (f.bbox_max - f.bbox_min).norm()
        row[base + 8] = _safe_log(d, diag if diag > 0 else 1.0)
        row[base + 9 : base + 12] = [
            (f.centroid.x - center.x) / diag,
            (f.centroid.y - center.y) / diag,
            (f.centroid.z - center.z) / diag,
        ]
        row[base + 12] = 1.0 if f.is_outer else 0.0
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def edge_features(graph: PartGraph) -> np.ndarray:
    """Return an ``(n_edges, len(EDGE_FEATURE_NAMES))`` float32 matrix."""
    rows: list[list[float]] = []
    for e in graph.edges:
        row = [0.0] * len(EDGE_FEATURE_NAMES)
        row[CONVEXITIES.index(e.convexity.value)] = 1.0
        base = len(CONVEXITIES)
        row[base] = _safe_log(e.length, 1000.0)
        ang = math.radians(e.dihedral_angle_deg or 90.0)
        row[base + 1] = math.sin(ang)
        row[base + 2] = math.cos(ang)
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def edge_index(graph: PartGraph) -> np.ndarray:
    """Bidirectional COO edge index, shape ``(2, 2*n_edges)``."""
    n = len(graph.edges)
    idx = np.zeros((2, 2 * n), dtype=np.int64)
    for i, e in enumerate(graph.edges):
        idx[0, i] = e.face_a
        idx[1, i] = e.face_b
        idx[0, n + i] = e.face_b
        idx[1, n + i] = e.face_a
    return idx


def edge_features_bidirectional(graph: PartGraph) -> np.ndarray:
    """Edge feature matrix duplicated to match :func:`edge_index`."""
    feats = edge_features(graph)
    return np.concatenate([feats, feats], axis=0)


def to_networkx(graph: PartGraph) -> nx.Graph:
    """Build a NetworkX graph with the same attribution."""
    import networkx as nx

    g = nx.Graph()
    for f in graph.faces:
        g.add_node(
            f.id,
            surface_type=f.surface_type.value,
            area=f.area,
            radius=f.radius,
            centroid=f.centroid.as_tuple(),
            label=f.label,
        )
    for e in graph.edges:
        g.add_edge(
            e.face_a,
            e.face_b,
            convexity=e.convexity.value,
            length=e.length,
            curve_type=e.curve_type.value,
        )
    return g


def to_pyg_data(graph: PartGraph, *, num_classes: int | None = None) -> Any:
    """Build a ``torch_geometric.data.Data`` object for face classification.

    Optional face labels are read from ``FaceRecord.label_id``; graphs missing
    labels produce ``y = -1``.
    """
    import torch
    from torch_geometric.data import Data

    x = torch.from_numpy(node_features(graph))
    ei = torch.from_numpy(edge_index(graph))
    ea = torch.from_numpy(edge_features_bidirectional(graph))
    y = torch.tensor(
        [f.label_id if f.label_id is not None else -1 for f in graph.faces],
        dtype=torch.long,
    )
    data = Data(x=x, edge_index=ei, edge_attr=ea, y=y, num_nodes=len(graph.faces))
    data.part_name = graph.metadata.name
    if num_classes is not None:
        data.num_classes = num_classes
    return data


def graph_feature_vector(graph: PartGraph) -> np.ndarray:
    """Global mean/max pooled feature vector, useful for simple baselines."""
    x = node_features(graph)
    if x.size == 0:
        return np.zeros(2 * len(FEATURE_NAMES), dtype=np.float32)
    return np.concatenate([x.mean(axis=0), x.max(axis=0)]).astype(np.float32)


__all__ = [
    "CONVEXITIES",
    "EDGE_FEATURE_NAMES",
    "FEATURE_NAMES",
    "SURFACE_TYPES",
    "edge_features",
    "edge_features_bidirectional",
    "edge_index",
    "graph_feature_vector",
    "node_features",
    "to_networkx",
    "to_pyg_data",
]
