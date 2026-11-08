"""Web API: behaviour, input validation and security properties."""
import io
import json
import os

import numpy as np
import pytest

os.environ.setdefault("INDICSWITCH_WARM_ASR", "0")  # tests never load a Whisper model

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

import server.app as srv  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_limits():
    srv._hits.clear()
    yield
    srv._hits.clear()


@pytest.fixture
def client():
    with TestClient(srv.app) as c:
        yield c


def wav_bytes(seconds=1.0, sr=16000):
    import soundfile as sf

    buf = io.BytesIO()
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    sf.write(buf, (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class FakeASR:
    def transcribe_array(self, wav, sr):
        from src.asr import ASRResult

        return ASRResult(text="main nhi jaunga haan haan haan haan haan", audio_seconds=len(wav) / sr,
                         elapsed_seconds=0.01, config="fake")


class FakeStreamingASR:
    class cfg:
        @staticmethod
        def tag():
            return "fake"

    def iter_segments(self, wav, sr):
        yield {"start": 0.0, "end": 0.4, "text": "main nhi"}
        yield {"start": 0.4, "end": 1.0, "text": "jaunga haan haan haan haan haan"}


def parse_sse(text):
    return [json.loads(line[6:]) for block in text.split("\n\n") for line in block.split("\n") if line.startswith("data: ")]


def test_stream_sends_segments_then_result(client, monkeypatch):
    monkeypatch.setattr(srv, "get_asr", lambda *a: FakeStreamingASR())
    r = client.post("/api/audio/stream", files={"file": ("x.wav", wav_bytes(1.0), "audio/wav")},
                    data={"settings": json.dumps({"lid": "rule"})})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    assert "content-encoding" not in r.headers  # never gzip-buffered
    events = parse_sse(r.text)
    assert [e["type"] for e in events] == ["start", "segment", "segment", "result"]
    result = events[-1]["result"]
    assert result["raw_transcript"] == "main nhi jaunga haan haan haan haan haan"
    assert result["normalized_transcript"].startswith("main nahi") or result["stats"]["edits"] >= 4
    assert len(result["segments"]) == 2
    assert srv._jobs.acquire(blocking=False)  # the job slot was released
    srv._jobs.release()


def test_stream_error_is_generic_and_releases_slot(client, monkeypatch):
    class Broken(FakeStreamingASR):
        def iter_segments(self, wav, sr):
            raise RuntimeError("secret C:/internal")
            yield  # pragma: no cover

    monkeypatch.setattr(srv, "get_asr", lambda *a: Broken())
    r = client.post("/api/audio/stream", files={"file": ("x.wav", wav_bytes(), "audio/wav")}, data={"settings": "{}"})
    events = parse_sse(r.text)
    assert events[-1]["type"] == "error" and "secret" not in r.text
    for _ in range(srv.MAX_CONCURRENT_JOBS):
        assert srv._jobs.acquire(blocking=False)
    for _ in range(srv.MAX_CONCURRENT_JOBS):
        srv._jobs.release()


def test_stream_validates_like_audio(client):
    r = client.post("/api/audio/stream", files={"file": ("x.wav", b"not audio" * 20, "audio/wav")}, data={"settings": "{}"})
    assert r.status_code == 400
    r = client.post("/api/audio/stream", files={"file": ("x.wav", wav_bytes(), "audio/wav")},
                    data={"settings": json.dumps({"model": "large"})})
    assert r.status_code == 422


def test_static_compression_and_caching(client):
    r = client.get("/assets/kit/system.css", headers={"Accept-Encoding": "gzip"})
    assert r.headers.get("content-encoding") == "gzip" and "max-age=604800" in r.headers["cache-control"]
    r = client.get("/", headers={"Accept-Encoding": "gzip"})
    assert r.headers["cache-control"] == "no-cache" and "etag" in r.headers
    assert client.get("/", headers={"If-None-Match": r.headers["etag"]}).status_code == 304


def test_config_lists_fixed_ids_only(client):
    r = client.get("/api/config")
    assert r.status_code == 200
    cfg = r.json()
    assert {m["id"] for m in cfg["lid_models"]} <= {"xlmr", "ngram", "rule"}
    assert "spec" not in json.dumps(cfg) and "models/" not in json.dumps(cfg)  # no server paths leak


def test_text_roundtrip(client):
    r = client.post("/api/text", json={"text": "main nhi jaunga, bohot kaam hai", "lid": "rule"})
    assert r.status_code == 200
    body = r.json()
    assert body["tokens"][0] == "main" and "stats" in body and "offsets" not in body
    assert r.headers["cache-control"] == "no-store"


@pytest.mark.parametrize("payload", [
    {"text": "", "lid": "rule"},
    {"text": "   ", "lid": "rule"},
    {"text": "x" * 5001, "lid": "rule"},
    {"text": "ok", "lid": "../../etc/passwd"},
    {"text": "ok", "lid": "rule", "target_script": "evil"},
    {"text": "ok", "lid": "rule", "min_confidence": 5},
    {"text": "ok", "lid": "rule", "protected_words": ["a"] * 51},
])
def test_text_validation_rejects_and_never_echoes(client, payload):
    r = client.post("/api/text", json=payload)
    assert r.status_code == 422
    assert "etc/passwd" not in r.text and "x" * 50 not in r.text


def test_security_headers_on_pages_and_api(client):
    for path in ("/", "/api/config"):
        h = client.get(path).headers
        assert "default-src 'self'" in h["content-security-policy"] and "unsafe-inline" not in h["content-security-policy"]
        assert h["x-content-type-options"] == "nosniff"
        assert h["x-frame-options"] == "DENY"
        assert h["referrer-policy"] == "no-referrer"
        assert "microphone=(self)" in h["permissions-policy"]


def test_docs_disabled(client):
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404


def test_static_files_do_not_expose_project(client):
    for path in ("/../server/app.py", "/.taste-skill/build.json", "/assets/", "/%2e%2e/requirements.txt"):
        r = client.get(path)
        assert r.status_code == 404, path
    assert client.get("/assets/fonts/mona-sans-latin.woff2").headers["content-type"] == "font/woff2"


def test_oversized_body_rejected(client):
    r = client.post("/api/text", content=b"{" + b" " * (srv.MAX_JSON_BYTES + 10) + b"}",
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 413


def test_audio_rejects_non_wav(client):
    r = client.post("/api/audio", files={"file": ("x.wav", b"<?php echo 1; ?>" * 10, "audio/wav")},
                    data={"settings": "{}"})
    assert r.status_code == 400 and "could not read" in r.json()["error"].lower()


def test_audio_rejects_bad_settings(client):
    r = client.post("/api/audio", files={"file": ("x.wav", wav_bytes(), "audio/wav")},
                    data={"settings": json.dumps({"model": "openai/whisper-large"})})
    assert r.status_code == 422


def test_audio_too_long(client, monkeypatch):
    monkeypatch.setattr(srv, "MAX_AUDIO_SECONDS", 0.5)
    r = client.post("/api/audio", files={"file": ("x.wav", wav_bytes(1.0), "audio/wav")}, data={"settings": "{}"})
    assert r.status_code == 400 and "longer" in r.json()["error"]


def test_audio_roundtrip_with_fake_asr(client, monkeypatch):
    monkeypatch.setattr(srv, "get_asr", lambda *a: FakeASR())
    r = client.post("/api/audio", files={"file": ("x.wav", wav_bytes(1.0, sr=44100), "audio/wav")},
                    data={"settings": json.dumps({"lid": "rule", "model": "base", "language": "hindi"})})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["normalized_transcript"].startswith("main")
    assert body["stats"]["edits"] >= 4  # the loop was collapsed
    assert abs(body["audio_seconds"] - 1.0) < 0.01  # resampled from 44.1 kHz


def test_rate_limit(client, monkeypatch):
    monkeypatch.setitem(srv.RATE_LIMITS, "text", (3, 60.0))
    codes = [client.post("/api/text", json={"text": "ok", "lid": "rule"}).status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]
    r = client.post("/api/text", json={"text": "ok", "lid": "rule"})
    assert "retry-after" in r.headers


def test_busy_returns_503(client, monkeypatch):
    import threading

    monkeypatch.setattr(srv, "_jobs", threading.BoundedSemaphore(1))
    srv._jobs.acquire()
    try:
        assert client.post("/api/text", json={"text": "ok", "lid": "rule"}).status_code == 503
    finally:
        srv._jobs.release()


def test_internal_error_is_generic(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("secret path C:/internal/thing")

    monkeypatch.setattr(srv.IndicSwitch, "process_text", boom)
    with TestClient(srv.app, raise_server_exceptions=False) as c:
        r = c.post("/api/text", json={"text": "ok", "lid": "rule"})
    assert r.status_code == 500 and "secret" not in r.text and "Traceback" not in r.text


def test_score(client):
    r = client.post("/api/score", json={"reference": "main kal jaunga", "raw": "main jaunga", "normalized": "main kal jaunga"})
    assert r.status_code == 200
    assert r.json()["normalized"]["wer"] == 0.0 and r.json()["raw"]["wer"] > 0
