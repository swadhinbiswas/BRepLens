from __future__ import annotations

import ast
from pathlib import Path


def _imports(path: Path) -> list[tuple[str, int, bool]]:
    tree = ast.parse(path.read_text())
    out: list[tuple[str, int, bool]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out.append((a.name, node.lineno, False))
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.append((node.module, node.lineno, True))
    return out


def test_no_nxopen_in_core(root: Path) -> None:
    allowed = {
        root / "src" / "breplens" / "adapters" / "nx.py",
        root / "src" / "breplens" / "adapters" / "nx_journal.py",
    }
    offenders = []
    for p in (root / "src" / "breplens").rglob("*.py"):
        if p in allowed:
            continue
        for name, line, _ in _imports(p):
            if name == "NXOpen" or name.startswith("NXOpen."):
                offenders.append(f"{p.relative_to(root)}:{line} imports {name}")
    assert offenders == []


def test_no_cadquery_outside_generator(root: Path) -> None:
    allowed = {root / "src" / "breplens" / "geometry" / "synthetic.py"}
    offenders = []
    for p in (root / "src" / "breplens").rglob("*.py"):
        if p in allowed:
            continue
        for name, line, _ in _imports(p):
            if name == "cadquery" or name.startswith("cadquery."):
                offenders.append(f"{p.relative_to(root)}:{line} imports {name}")
    assert offenders == []


def test_no_direct_ocp_outside_adapters(root: Path) -> None:
    offenders = []
    for p in (root / "src" / "breplens").rglob("*.py"):
        if "adapters" in p.parts:
            continue
        for name, line, _ in _imports(p):
            if name == "OCP" or name.startswith("OCP."):
                offenders.append(f"{p.relative_to(root)}:{line} imports {name}")
    assert offenders == []


def test_torch_only_ml_and_lazy_exporter(root: Path) -> None:
    allowed = {root / "src" / "breplens" / "geometry" / "aag.py"}
    offenders = []
    for p in (root / "src" / "breplens" / "ml").glob("*.py"):
        allowed.add(p)
    for p in (root / "src" / "breplens").rglob("*.py"):
        if p in allowed:
            continue
        for name, line, _ in _imports(p):
            if name == "torch" or name.startswith("torch.") or name.startswith("torch_geometric"):
                offenders.append(f"{p.relative_to(root)}:{line} imports {name}")
    assert offenders == []


def test_streamlit_app_does_not_import_vtk_or_cadquery(root: Path) -> None:
    text = (root / "streamlit_app.py").read_text()
    for banned in ("import cadquery", "from cadquery", "import vtk", "from vtk"):
        assert banned not in text


def test_schema_version_pinned(root: Path) -> None:
    from breplens.schema import SCHEMA_VERSION

    schema = root / "docs" / "schema.v1.json"
    if schema.exists():
        import json

        payload = json.loads(schema.read_text())
        assert payload["schema_version"] == SCHEMA_VERSION
