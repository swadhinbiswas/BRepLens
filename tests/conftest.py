from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SAMPLES = ROOT / "samples"


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def samples_dir() -> Path:
    return SAMPLES


@pytest.fixture(scope="session")
def occ_adapter():
    from breplens.adapters.occ import OccAdapter

    return OccAdapter()


@pytest.fixture
def fake_nx(monkeypatch):
    """Load the NXOpen mock for adapter tests only."""
    fake_path = str(ROOT / "tests" / "fakes")
    if fake_path not in sys.path:
        sys.path.insert(0, fake_path)
    for name in list(sys.modules):
        if name == "NXOpen" or name.startswith("NXOpen."):
            del sys.modules[name]
    import NXOpen

    NXOpen.reset()
    yield NXOpen
    for name in list(sys.modules):
        if name == "NXOpen" or name.startswith("NXOpen."):
            del sys.modules[name]
    if fake_path in sys.path:
        sys.path.remove(fake_path)
