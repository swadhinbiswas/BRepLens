from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.slow
def test_streamlit_rules_mode(root: Path) -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(root / "streamlit_app.py"), default_timeout=180).run()
    assert not at.exception
    at.radio[0].set_value("rules").run()
    at.selectbox[0].select("sample_09_bracket_mixed.step").run()
    assert not at.exception
    assert any(h.value == "3D view" for h in at.subheader)
    assert len(at.dataframe) >= 1
    assert at.selectbox[0].options


@pytest.mark.slow
def test_streamlit_hybrid_mode(root: Path) -> None:
    pytest.importorskip("torch")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(root / "streamlit_app.py"), default_timeout=300).run()
    at.radio[0].set_value("hybrid").run()
    at.selectbox[0].select("sample_04_pocket.step").run()
    assert not at.exception
    assert any("pocket" in str(df.value).lower() for df in at.dataframe)
