"""IndicSwitch web app: a JSON API over the pipeline plus the static frontend in ``web/``.

    uvicorn server.app:app --host 127.0.0.1 --port 8000

Security design (see SECURITY.md):
* The browser decodes any audio and uploads 16 kHz mono WAV only; the server
  parses nothing but WAV (checked by header and by libsndfile).
* Clients choose from fixed ids (models, languages, scripts); no path or model
  name from a request ever reaches the filesystem or the model hub.
* Request bodies are capped while streaming; per-IP rate limits; a global cap
  on concurrent heavy jobs; strict security headers (CSP without inline code).
* Errors return generic messages; details go to the server log only. The
  interactive API docs are disabled.
* Nothing is stored: audio lives in memory for the length of one request.
"""
from __future__ import annotations

import io
import json
import logging
import mimetypes
import os
import threading
import time
from collections import OrderedDict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError, field_validator

from src.correction import CorrectionConfig
from src.evaluation import boundaries, corpus_wer, switch_density
from src.language_identification import BaseLID, load_lid
from src.pipeline import IndicSwitch

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
# Correct types for static files regardless of the OS registry (nosniff is set on every response).
for _ext, _type in ((".woff2", "font/woff2"), (".js", "text/javascript"), (".mjs", "text/javascript"),
                    (".css", "text/css"), (".svg", "image/svg+xml")):
    mimetypes.add_type(_type, _ext)
log = logging.getLogger("indicswitch.server")

# ---- limits -----------------------------------------------------------------
MAX_UPLOAD_BYTES = 25 * 1024 * 1024        # 16 kHz 16-bit mono: ~13 min
MAX_JSON_BYTES = 64 * 1024
MAX_AUDIO_SECONDS = 600
MIN_AUDIO_SECONDS = 0.1
MAX_TEXT_CHARS = 5000
MAX_PROTECTED_WORDS = 50
MAX_CONCURRENT_JOBS = int(os.environ.get("INDICSWITCH_MAX_JOBS", "2"))
RATE_LIMITS = {"text": (30, 60.0), "audio": (6, 60.0), "score": (60, 60.0)}  # (requests, per seconds)
TRUST_PROXY = os.environ.get("INDICSWITCH_TRUST_PROXY") == "1"   # honour X-Forwarded-For only behind your proxy
FORCE_HTTPS = os.environ.get("INDICSWITCH_HTTPS") == "1"         # send HSTS when served over HTTPS
GENERIC_ERROR = "This input could not be processed. Please try a different file or text."

WHISPER_MODELS = {"base": "openai/whisper-base", "small": "openai/whisper-small", "tiny": "openai/whisper-tiny"}
LANGUAGES = {"auto": None, "hindi": "hindi", "english": "english"}
DEFAULT_MODEL = os.environ.get("INDICSWITCH_DEFAULT_MODEL", "small")
# Repetition penalty for faster-whisper decoding (>1 curbs Whisper's repetition loops). Chosen on 5 CoSHE
# clips and checked on 15 others that were not used for the choice; see README, ASR results.
REPETITION_PENALTY = float(os.environ.get("INDICSWITCH_REPETITION_PENALTY", "1.1"))
WARM_ASR = os.environ.get("INDICSWITCH_WARM_ASR", "1") == "1"


def lid_registry() -> dict[str, dict]:
    """Language-ID models available on this server, by fixed id."""
    reg = {}
    if (ROOT / "models" / "lid_model" / "config.json").exists():
        reg["xlmr"] = {"label": "XLM-R (fine-tuned)", "spec": str(ROOT / "models" / "lid_model")}
    if (ROOT / "models" / "ngram_lid.pkl.sha256").exists():
        reg["ngram"] = {"label": "Character n-gram", "spec": str(ROOT / "models" / "ngram_lid.pkl")}
    reg["rule"] = {"label": "Script rule", "spec": "rule"}
    return reg


# ---- model caches -------------------------------------------------------------
_lock = threading.Lock()
_lids: dict[str, BaseLID] = {}
_asrs: OrderedDict = OrderedDict()   # at most 2 Whisper models in memory, keyed by size
_load_lock = threading.Lock()        # one model load at a time (no double memory spike)
_jobs = threading.BoundedSemaphore(MAX_CONCURRENT_JOBS)


def asr_engine() -> str:
    from src.asr import faster_whisper_available

    return "faster-whisper" if faster_whisper_available() else "transformers"


def get_lid(lid_id: str) -> BaseLID:
    reg = lid_registry()
    if lid_id not in reg:
        raise HTTPException(400, "Unknown language model.")
    with _lock:
        if lid_id not in _lids:
            _lids[lid_id] = load_lid(reg[lid_id]["spec"])
        return _lids[lid_id]


def get_asr(model: str, language: str, block_urdu: bool):
    """A recogniser for these settings. faster-whisper: one loaded model per size, shared by every
    language/script setting (those are decoding options). Transformers: one pipeline per setting."""
    from src.asr import DecodeConfig, FasterWhisperASR, WhisperASR, load_faster_model

    faster = asr_engine() == "faster-whisper"
    cfg = DecodeConfig(model_name=WHISPER_MODELS[model], language=LANGUAGES[language], use_vad=faster,
                       suppress_scripts=("arabic",) if block_urdu else (),
                       repetition_penalty=REPETITION_PENALTY if faster else 1.0)
    key = model if faster else (model, language, block_urdu)
    with _lock:
        cached = _asrs.get(key)
        if cached is not None:
            _asrs.move_to_end(key)
    if cached is None:
        with _load_lock:
            with _lock:
                cached = _asrs.get(key)
            if cached is None:
                cached = load_faster_model(cfg.model_name) if faster else WhisperASR(cfg)
                with _lock:
                    _asrs[key] = cached
                    while len(_asrs) > 2:
                        _asrs.popitem(last=False)
    return FasterWhisperASR(cfg, shared_model=cached) if faster else cached


# ---- rate limiting --------------------------------------------------------------
_hits: dict[tuple[str, str], deque] = {}
_hits_lock = threading.Lock()


def client_ip(request: Request) -> str:
    if TRUST_PROXY:
        fwd = request.headers.get("x-forwarded-for", "")
        if fwd:
            return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def rate_limit(request: Request, bucket: str) -> None:
    limit, window = RATE_LIMITS[bucket]
    key = (client_ip(request), bucket)
    now = time.monotonic()
    with _hits_lock:
        q = _hits.setdefault(key, deque())
        while q and now - q[0] > window:
            q.popleft()
        if len(q) >= limit:
            retry = int(window - (now - q[0])) + 1
            raise HTTPException(429, "Too many requests. Please wait a moment.", headers={"Retry-After": str(retry)})
        q.append(now)
        if len(_hits) > 10_000:  # bound memory: drop idle clients
            for k in [k for k, v in _hits.items() if not v or now - v[-1] > window]:
                del _hits[k]


class job_slot:
    """Context manager: take one of the concurrent heavy-job slots or answer 503."""

    def __enter__(self):
        if not _jobs.acquire(blocking=False):
            raise HTTPException(503, "The server is busy. Please try again in a moment.")

    def __exit__(self, *exc):
        _jobs.release()


# ---- request models -------------------------------------------------------------
class CorrectionSettings(BaseModel):
    lid: str = "ngram"
    min_confidence: float = Field(0.80, ge=0.5, le=0.99)
    canonicalize: bool = True
    collapse_loops: bool = True
    target_script: Literal["keep", "roman", "native", "mixed"] = "keep"
    protected_words: list[str] = Field(default_factory=list, max_length=MAX_PROTECTED_WORDS)

    @field_validator("lid")
    @classmethod
    def _known_lid(cls, v):
        if v not in lid_registry():
            raise ValueError("unknown language model")
        return v

    @field_validator("protected_words")
    @classmethod
    def _short_words(cls, v):
        words = [w.strip().lower() for w in v if w.strip()]
        if any(len(w) > 40 for w in words):
            raise ValueError("protected words must be 40 characters or fewer")
        return words

    def correction_config(self) -> CorrectionConfig:
        return CorrectionConfig(min_confidence=self.min_confidence, canonicalize=self.canonicalize,
                                collapse_loops=self.collapse_loops, target_script=self.target_script,
                                protected_words=set(self.protected_words))


class TextRequest(CorrectionSettings):
    text: str = Field(min_length=1, max_length=MAX_TEXT_CHARS)

    @field_validator("text")
    @classmethod
    def _not_blank(cls, v):
        if not v.strip():
            raise ValueError("text is empty")
        return v


class AudioSettings(CorrectionSettings):
    model: Literal["base", "small", "tiny"] = "small"
    language: Literal["auto", "hindi", "english"] = "hindi"
    block_urdu: bool = True


class ScoreRequest(BaseModel):
    reference: str = Field(min_length=1, max_length=MAX_TEXT_CHARS * 4)
    raw: str = Field(max_length=200_000)
    normalized: str = Field(max_length=200_000)


# ---- helpers --------------------------------------------------------------------
def summarize(result: dict) -> dict:
    """Add the counts the UI shows; drop internals it does not need."""
    tags = result["language_tags"]
    lang = [t for t in tags if t != "OTH"]
    result["stats"] = {
        "tokens": len(tags),
        "hi": tags.count("HI"),
        "en": tags.count("EN"),
        "oth": tags.count("OTH"),
        "switches": len(boundaries(tags)),
        "switch_density": round(switch_density(tags), 3) if lang else 0.0,
        "edits": len(result["edits"]),
    }
    result.pop("offsets", None)
    return result


def decode_wav(data: bytes) -> tuple[np.ndarray, int]:
    """Parse a WAV upload into mono float32 at 16 kHz; raise ValueError if it is not a valid WAV."""
    import soundfile as sf

    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a WAV file")
    info = sf.info(io.BytesIO(data))
    if info.format != "WAV" or info.channels < 1 or info.channels > 8 or not (8_000 <= info.samplerate <= 96_000):
        raise ValueError("unsupported WAV")
    if info.frames / info.samplerate > MAX_AUDIO_SECONDS:
        raise ValueError("too long")
    wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != 16_000:
        import librosa

        wav = librosa.resample(wav, orig_sr=sr, target_sr=16_000)
        sr = 16_000
    return np.ascontiguousarray(wav, dtype=np.float32), sr


# ---- app ------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_: FastAPI):
    # Warm the default language model and Whisper in the background so the first visitor does not wait.
    def warm():
        try:
            get_lid(next(iter(lid_registry())))
            if WARM_ASR:
                get_asr(DEFAULT_MODEL, "hindi", True)
        except Exception:
            log.exception("model warm-up failed")
    threading.Thread(target=warm, daemon=True).start()
    yield


app = FastAPI(title="IndicSwitch", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
# gzip text responses (HTML, CSS, JS, JSON); fonts are already compressed and the event stream is excluded.
app.add_middleware(GZipMiddleware, minimum_size=1024)

CSP = ("default-src 'self'; script-src 'self'; style-src 'self'; font-src 'self'; img-src 'self' data: blob:; "
       "media-src 'self' blob:; connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; "
       "frame-ancestors 'none'")


@app.middleware("http")
async def security_and_size(request: Request, call_next):
    # Reject oversized bodies early (declared length); streaming bodies are capped per route.
    limit = MAX_UPLOAD_BYTES + 64 * 1024 if request.url.path.startswith("/api/audio") else MAX_JSON_BYTES
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            too_big = int(declared) > limit
        except ValueError:
            return JSONResponse({"error": "Bad request."}, status_code=400)
        if too_big:
            return JSONResponse({"error": "The upload is too large."}, status_code=413)
    elif request.method in ("POST", "PUT", "PATCH"):
        return JSONResponse({"error": "A Content-Length header is required."}, status_code=411)
    response = await call_next(request)
    h = response.headers
    h["Content-Security-Policy"] = CSP
    h["X-Content-Type-Options"] = "nosniff"
    h["Referrer-Policy"] = "no-referrer"
    h["X-Frame-Options"] = "DENY"
    h["Cross-Origin-Opener-Policy"] = "same-origin"
    h["Permissions-Policy"] = "microphone=(self), camera=(), geolocation=(), payment=()"
    if FORCE_HTTPS:
        h["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    path = request.url.path
    if path.startswith("/api/"):
        h["Cache-Control"] = "no-store"
    elif path.startswith(("/assets/fonts/", "/assets/kit/")):
        h["Cache-Control"] = "public, max-age=604800"   # vendored files that do not change between releases
    else:
        h["Cache-Control"] = "no-cache"                 # revalidate with the ETag: a 304 when unchanged
    return response


@app.exception_handler(HTTPException)
async def http_error(_: Request, exc: HTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code, headers=exc.headers)


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, exc: RequestValidationError):
    # Name the field and the problem; never echo the submitted value back.
    problems = []
    for e in exc.errors()[:5]:
        field = ".".join(str(p) for p in e.get("loc", [])[1:]) or "request"
        problems.append(f"{field}: {e.get('msg', 'invalid')}")
    return JSONResponse({"error": "Some settings are not valid.", "fields": problems}, status_code=422)


@app.exception_handler(Exception)
async def unhandled(_: Request, exc: Exception):
    log.exception("unhandled error", exc_info=exc)
    return JSONResponse({"error": GENERIC_ERROR}, status_code=500)


@app.get("/api/config")
def config():
    return {
        "lid_models": [{"id": k, "label": v["label"]} for k, v in lid_registry().items()],
        "default_lid": next(iter(lid_registry())),
        "whisper_models": list(WHISPER_MODELS),
        "default_model": DEFAULT_MODEL,
        "engine": asr_engine(),
        "languages": list(LANGUAGES),
        "limits": {"max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024), "max_audio_seconds": MAX_AUDIO_SECONDS,
                   "max_text_chars": MAX_TEXT_CHARS},
    }


@app.post("/api/text")
def process_text(req: TextRequest, request: Request):
    rate_limit(request, "text")
    with job_slot():
        pipe = IndicSwitch(lid=get_lid(req.lid), correction=req.correction_config())
        return summarize(pipe.process_text(req.text))


class SettingsError(Exception):
    def __init__(self, fields: list[str]):
        self.fields = fields


@app.exception_handler(SettingsError)
async def settings_error(_: Request, exc: SettingsError):
    return JSONResponse({"error": "Some settings are not valid.", "fields": exc.fields}, status_code=422)


def read_audio_request(file: UploadFile, settings: str) -> tuple[AudioSettings, np.ndarray, int]:
    """Validate the settings and the WAV upload shared by both audio routes."""
    if len(settings) > MAX_JSON_BYTES:
        raise HTTPException(413, "Settings are too large.")
    try:
        s = AudioSettings.model_validate_json(settings)
    except ValidationError as exc:
        raise SettingsError([f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:5]]) from None
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "The upload is too large.")
    try:
        wav, sr = decode_wav(data)
    except ValueError as exc:
        msg = ("The recording is longer than 10 minutes." if str(exc) == "too long"
               else "Could not read this audio. Please try another file.")
        raise HTTPException(400, msg) from None
    except Exception:
        log.exception("wav decode failure")
        raise HTTPException(400, "Could not read this audio. Please try another file.") from None
    finally:
        del data
    if len(wav) / sr < MIN_AUDIO_SECONDS:
        raise HTTPException(400, "The recording is empty.")
    return s, wav, sr


@app.post("/api/audio")
def process_audio(request: Request, file: Annotated[UploadFile, File()], settings: Annotated[str, Form()] = "{}"):
    rate_limit(request, "audio")
    s, wav, sr = read_audio_request(file, settings)
    with job_slot():
        pipe = IndicSwitch(asr=get_asr(s.model, s.language, s.block_urdu), lid=get_lid(s.lid),
                           correction=s.correction_config())
        return summarize(pipe.run_array(wav, sr))


def sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@app.post("/api/audio/stream")
def process_audio_stream(request: Request, file: Annotated[UploadFile, File()],
                         settings: Annotated[str, Form()] = "{}"):
    """Same as /api/audio, as Server-Sent Events: ``start``, one ``segment`` per decoded piece of
    speech (so text appears while the rest is still being transcribed), then ``result`` or ``error``."""
    from src.asr import ASRResult

    rate_limit(request, "audio")
    s, wav, sr = read_audio_request(file, settings)
    if not _jobs.acquire(blocking=False):
        raise HTTPException(503, "The server is busy. Please try again in a moment.")
    try:
        pipe = IndicSwitch(asr=get_asr(s.model, s.language, s.block_urdu), lid=get_lid(s.lid),
                           correction=s.correction_config())
    except BaseException:
        _jobs.release()
        raise

    def events():
        try:
            t0 = time.perf_counter()
            seconds = len(wav) / sr
            yield sse({"type": "start", "audio_seconds": round(seconds, 2)})
            asr = pipe.asr
            if hasattr(asr, "iter_segments"):
                segments = []
                for seg in asr.iter_segments(wav, sr):
                    segments.append(seg)
                    yield sse({"type": "segment", **seg})
                res = ASRResult(text=" ".join(x["text"] for x in segments), segments=segments, audio_seconds=seconds,
                                elapsed_seconds=time.perf_counter() - t0, config="faster-whisper," + asr.cfg.tag())
            else:
                res = asr.transcribe_array(wav, sr)
            yield sse({"type": "result", "result": summarize(pipe.finish(res, t0))})
        except Exception:
            log.exception("stream failure")
            yield sse({"type": "error", "error": GENERIC_ERROR})
        finally:
            _jobs.release()   # also runs when the client disconnects and the stream is closed

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


@app.post("/api/score")
def score(req: ScoreRequest, request: Request):
    rate_limit(request, "score")
    out = {}
    for name, hyp in (("raw", req.raw), ("normalized", req.normalized)):
        exact, agn = corpus_wer([req.reference], [hyp]), corpus_wer([req.reference], [hyp], romanize=True)
        out[name] = {"wer": exact["wer"], "cer": exact["cer"], "wer_script_agnostic": agn["wer"],
                     "cer_script_agnostic": agn["cer"]}
    return out


# Static frontend last, so /api routes win.
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
