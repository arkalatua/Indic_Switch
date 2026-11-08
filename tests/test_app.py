"""Streamlit app behaviour: runs cleanly, validates input, hides internals, rate-limits."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")


def _app():
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception
    return at


def _process_btn(at):
    return next(b for b in at.button if b.label == "Process text")


def test_text_mode_runs_and_escapes_markup():
    at = _app()
    at.text_area[0].set_value("main nhi jaunga <script>alert(1)</script> **bold**")
    _process_btn(at).click().run()
    assert not at.exception and not at.error
    res = at.session_state["result"]
    assert res["raw_transcript"].startswith("main nhi")
    # Markdown rendered for transcripts is escaped (no live HTML/markdown)
    rendered = [m.value for m in at.info]
    assert any("\\*\\*bold\\*\\*" in v and "\\<script\\>" in v for v in rendered)
    # Token chips are HTML-escaped
    # Token chips are HTML-escaped ("<" and ">" are separate tokens, each escaped)
    chips = "".join(m.value for m in at.markdown)
    assert "<script" not in chips and "</script" not in chips
    assert "&lt;" in chips and "&gt;" in chips


def test_empty_text_rejected():
    at = _app()
    at.text_area[0].set_value("   ")
    _process_btn(at).click().run()
    assert at.error and "enter some text" in at.error[0].value
    assert "result" not in at.session_state


def test_text_length_capped():
    at = _app()
    assert at.text_area[0].max_chars == 5000


def test_cooldown_blocks_rapid_repeat():
    at = _app()
    _process_btn(at).click().run()
    assert "result" in at.session_state
    _process_btn(at).click().run()
    assert at.warning and "Please wait" in at.warning[0].value


def test_internal_errors_are_generic(monkeypatch):
    import src.pipeline as pipeline

    def boom(self, text, timings=None):
        raise RuntimeError("secret internal path C:/x/y")

    monkeypatch.setattr(pipeline.IndicSwitch, "process_text", boom)
    at = _app()
    _process_btn(at).click().run()
    assert at.error
    msg = at.error[0].value
    assert "could not be processed" in msg and "secret" not in msg and "Traceback" not in msg
