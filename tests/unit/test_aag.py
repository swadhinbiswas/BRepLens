from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from breplens.geometry.aag import (
    EDGE_FEATURE_NAMES,
    FEATURE_NAMES,
    edge_features,
    edge_features_bidirectional,
    edge_index,
    graph_feature_vector,
    node_features,
    to_networkx,
    to_pyg_data,
)


def test_feature_shapes(occ_adapter, samples_dir: Path) -> None:
    g = occ_adapter.load_graph(samples_dir / "sample_04_pocket.step")
    x = node_features(g)
    e = edge_features(g)
    idx = edge_index(g)
    assert x.shape == (len(g.faces), len(FEATURE_NAMES))
    assert e.shape == (len(g.edges), len(EDGE_FEATURE_NAMES))
    assert idx.shape == (2, 2 * len(g.edges))
    assert np.isfinite(x).all()
    assert np.isfinite(e).all()
    assert np.allclose(np.linalg.norm(x, axis=1) > 0, True)


def test_edge_index_is_bidirectional(occ_adapter, samples_dir: Path) -> None:
    g = occ_adapter.load_graph(samples_dir / "sample_04_pocket.step")
    idx = edge_index(g)
    n = len(g.edges)
    assert (idx[:, :n] == idx[::-1, n:]).all()
    ea = edge_features_bidirectional(g)
    assert ea.shape[0] == 2 * n


def test_networkx_export(occ_adapter, samples_dir: Path) -> None:
    g = occ_adapter.load_graph(samples_dir / "sample_04_pocket.step")
    nxg = to_networkx(g)
    assert nxg.number_of_nodes() == len(g.faces)
    assert nxg.number_of_edges() == len(g.edges)
    assert nxg.nodes[0]["surface_type"] in {
        "plane",
        "cylinder",
        "cone",
        "torus",
        "sphere",
        "bspline",
        "other",
    }


def test_pyg_export(occ_adapter, samples_dir: Path) -> None:
    pytest.importorskip("torch_geometric")
    g = occ_adapter.load_graph(samples_dir / "sample_04_pocket.step")
    data = to_pyg_data(g, num_classes=11)
    assert data.x.shape[0] == len(g.faces)
    assert data.edge_index.shape[1] == 2 * len(g.edges)
    assert data.edge_attr.shape[0] == 2 * len(g.edges)


def test_global_vector(occ_adapter, samples_dir: Path) -> None:
    g = occ_adapter.load_graph(samples_dir / "sample_04_pocket.step")
    v = graph_feature_vector(g)
    assert v.shape == (2 * len(FEATURE_NAMES),)
    assert np.isfinite(v).all()
