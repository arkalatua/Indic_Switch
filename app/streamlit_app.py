"""IndicSwitch demo.

    streamlit run app/streamlit_app.py

Upload audio (or paste an ASR transcript for text-only mode), see the raw
Whisper transcript, per-token language tags with confidence, the
conservatively normalized transcript, the edit log, and — given a reference —
WER/CER for raw vs normalized.

Security notes:
* No accounts or stored data: every result lives only in the visitor's own
  session (``st.session_state``); uploaded audio is written to a private temp
  file and deleted immediately after processing.
* All inputs are validated (type, size, duration, length); every widget offers
  only fixed choices or bounded values.
* Abuse protection: per-session cooldown plus a global cap on concurrent
  heavy jobs so one visitor cannot exhaust the server.
* Errors show a generic message; details go to the server log only
  (see also .streamlit/config.toml: showErrorDetails = "none").
* Serve behind HTTPS (Streamlit Community Cloud / Hugging Face Spaces do
  this automatically; otherwise use a TLS-terminating reverse proxy).
"""
import html
import logging
import os
import re
import sys
import tempfile
import threading
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.correction import CorrectionConfig  # noqa: E402
from src.evaluation import corpus_wer  # noqa: E402
from src.pipeline import IndicSwitch  # noqa: E402

log = logging.getLogger("indicswitch.app")

# ---- limits ---------------------------------------------------------------
MAX_UPLOAD_MB = 25                 # also enforced by server.maxUploadSize
MAX_AUDIO_SECONDS = 600            # 10 minutes
MAX_TEXT_CHARS = 5000
MAX_PROTECTED_CHARS = 300
MAX_PROTECTED_WORDS = 50
COOLDOWN_SECONDS = 5               # per session, between heavy requests
MAX_CONCURRENT_JOBS = 2            # across all sessions
ALLOWED_EXT = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
WHISPER_MODELS = ["openai/whisper-base", "openai/whisper-tiny", "openai/whisper-small"]
GENERIC_ERROR = "Sorry, this input could not be processed. Please try a different file or text."

st.set_page_config(page_title="IndicSwitch", page_icon="🗣️", layout="wide")

TAG_COLORS = {"HI": "#e8590c", "EN": "#1c7ed6", "OTH": "#868e96"}


@st.cache_resource
def job_slots() -> threading.BoundedSemaphore:
    """Shared across sessions: caps concurrent pipeline runs."""
    return threading.BoundedSemaphore(MAX_CONCURRENT_JOBS)


def run_guarded(fn):
    """Run ``fn`` with cooldown, concurrency cap and generic error handling.
    Returns the result, or None if refused/failed (a message is shown)."""
    now = time.monotonic()
    last = st.session_state.get("last_run", 0.0)
    if now - last < COOLDOWN_SECONDS:
        st.warning(f"Please wait {COOLDOWN_SECONDS - (now - last):.0f}s before sending another request.")
        return None
    slots = job_slots()
    if not slots.acquire(blocking=False):
        st.warning("The server is busy right now. Please try again in a moment.")
        return None
    st.session_state["last_run"] = now
    try:
        return fn()
    except Exception:  # never show internals to visitors
        log.exception("pipeline failure")
        st.error(GENERIC_ERROR)
        return None
    finally:
        slots.release()


_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~$])")


def md_escape(text: str) -> str:
    """Escape Markdown/LaTeX so transcripts render as plain text."""
    return _MD_SPECIAL.sub(r"\\\1", text)


def available_lids():
    opts = {}
    if (ROOT / "models" / "lid_model" / "config.json").exists():
        opts["XLM-R (fine-tuned)"] = str(ROOT / "models" / "lid_model")
    if (ROOT / "models" / "ngram_lid.pkl.sha256").exists():
        opts["Char n-gram + LR"] = str(ROOT / "models" / "ngram_lid.pkl")
    opts["Script rule (no training)"] = "rule"
    return opts


@st.cache_resource(show_spinner="Loading Whisper…")
def get_asr(model_name, language, beams, use_vad):
    from src.asr import DecodeConfig, WhisperASR

    if model_name not in WHISPER_MODELS:  # defence in depth; the widget only offers these
        raise ValueError("unsupported model")
    return WhisperASR(DecodeConfig(model_name=model_name, language=language, num_beams=beams, use_vad=use_vad))


@st.cache_resource(show_spinner="Loading LID model…")
def get_lid(spec):
    from src.language_identification import load_lid

    return load_lid(spec)


def process_upload(up, pipe: IndicSwitch):
    """Validate and transcribe an uploaded file; the temp copy is always deleted."""
    from src.preprocessing import audio_duration, load_audio

    ext = Path(up.name).suffix.lower()
    if ext not in ALLOWED_EXT:
        st.error("Unsupported file type.")
        return None
    if up.size > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"File is larger than {MAX_UPLOAD_MB} MB.")
        return None
    fd, tmp_path = tempfile.mkstemp(suffix=ext)  # private (0600) file, fixed whitelisted suffix
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(up.getvalue())
        try:
            wav, sr = load_audio(tmp_path)
        except Exception:
            log.exception("audio decode failure")
            st.error("Could not read this audio file.")
            return None
        if audio_duration(wav, sr) > MAX_AUDIO_SECONDS:
            st.error(f"Audio is longer than {MAX_AUDIO_SECONDS // 60} minutes.")
            return None
        if audio_duration(wav, sr) < 0.1:
            st.error("Audio is empty.")
            return None
        return pipe.run_array(wav, sr)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            log.warning("could not delete temp file %s", tmp_path)


def render_tags(tokens, tags, confs, threshold):
    spans = []
    for w, t, c in zip(tokens, tags, confs, strict=True):
        t = t if t in TAG_COLORS else "OTH"
        color = TAG_COLORS[t]
        dim = "opacity:0.55;" if c < threshold else ""
        spans.append(
            f"<span title='{t} · {float(c):.2f}' style='display:inline-block;margin:2px;padding:2px 6px;"
            f"border-radius:6px;border:1.5px solid {color};{dim}'>{html.escape(w)}"
            f"<sub style='color:{color};font-weight:600;margin-left:3px'>{t}</sub></span>")
    return "<div style='line-height:2.1'>" + "".join(spans) + "</div>"


with st.sidebar:
    st.header("Settings")
    lid_opts = available_lids()
    lid_label = st.selectbox("Language ID model", list(lid_opts))
    st.subheader("ASR (Whisper)")
    model_name = st.selectbox("Model", WHISPER_MODELS)
    lang_choice = st.selectbox("Language prompt", ["auto", "hindi", "english"],
                               help="Forcing a language can suppress code-switching — compare, don't assume.")
    beams = st.slider("Beams", 1, 5, 1)
    use_vad = st.checkbox("Silero VAD segmentation", False)
    st.subheader("Correction")
    min_conf = st.slider("Confidence threshold", 0.5, 0.99, 0.80, 0.01)
    canonicalize = st.checkbox("Canonicalize Romanized-Hindi spellings", True)
    collapse = st.checkbox("Collapse hallucination loops", True)
    script = st.selectbox("Render script", ["keep", "roman", "native", "mixed"],
                          help="mixed = Hindi in Devanagari, English in Latin (CoSHE convention). "
                               "native/mixed need `ai4bharat-transliteration` for Roman→Devanagari.")
    protected = st.text_input("Protected words (comma-separated)", "", max_chars=MAX_PROTECTED_CHARS)

st.title("IndicSwitch")
st.caption("Code-switch-aware post-processing for Hinglish ASR · raw Whisper output is always preserved · "
           "nothing you upload is stored")

protected_words = {w.strip().lower() for w in protected.split(",") if w.strip()}
if len(protected_words) > MAX_PROTECTED_WORDS:
    st.sidebar.warning(f"Only the first {MAX_PROTECTED_WORDS} protected words are used.")
    protected_words = set(sorted(protected_words)[:MAX_PROTECTED_WORDS])
cfg = CorrectionConfig(min_confidence=min_conf, canonicalize=canonicalize, collapse_loops=collapse,
                       target_script=script, protected_words=protected_words)

tab_audio, tab_text = st.tabs(["🎙️ Audio", "📝 Transcript only"])
with tab_audio:
    up = st.file_uploader(f"Upload audio (max {MAX_UPLOAD_MB} MB, {MAX_AUDIO_SECONDS // 60} min)",
                          type=[e.lstrip(".") for e in sorted(ALLOWED_EXT)])
    if up is not None:
        st.audio(up)
        if st.button("Transcribe", type="primary"):
            def _run_audio():
                asr = get_asr(model_name, None if lang_choice == "auto" else lang_choice, beams, use_vad)
                pipe = IndicSwitch(asr=asr, lid=get_lid(lid_opts[lid_label]), correction=cfg)
                with st.spinner("Running pipeline…"):
                    return process_upload(up, pipe)

            res = run_guarded(_run_audio)
            if res is not None:
                st.session_state["result"] = res
with tab_text:
    txt = st.text_area("Paste an ASR transcript", "main kal office nhi ja raha, bohot kaam hai yaar. meeting kab hai?",
                       max_chars=MAX_TEXT_CHARS)
    if st.button("Process text"):
        if not txt.strip():
            st.error("Please enter some text.")
        else:
            res = run_guarded(lambda: IndicSwitch(lid=get_lid(lid_opts[lid_label]), correction=cfg).process_text(txt))
            if res is not None:
                st.session_state["result"] = res

result = st.session_state.get("result")
if result:
    c1, c2 = st.columns(2)
    c1.markdown("**Raw transcript**")
    c1.info(md_escape(result["raw_transcript"]) or "∅")
    c2.markdown("**Normalized transcript**")
    c2.success(md_escape(result["normalized_transcript"]) or "∅")

    st.markdown("**Token language tags** (faded = below confidence threshold, left unchanged)")
    st.markdown(render_tags(result["tokens"], result["language_tags"], result["confidences"], min_conf),
                unsafe_allow_html=True)  # every token is html-escaped in render_tags

    m = st.columns(4)
    tags = result["language_tags"]
    m[0].metric("Tokens", len(tags))
    m[1].metric("HI / EN", f"{tags.count('HI')} / {tags.count('EN')}")
    m[2].metric("Edits", len(result["edits"]))
    if result.get("rtf"):
        m[3].metric("Real-time factor", f"{result['rtf']:.2f}")

    if result["edits"]:
        with st.expander(f"Edit log ({len(result['edits'])})", expanded=True):
            st.dataframe([{k: e[k] for k in ("index", "before", "after", "rule", "tag", "confidence")}
                          for e in result["edits"]])
    with st.expander("Timings / segments"):
        st.json({"timings": result.get("timings"), "segments": result.get("segments", []),
                 "asr_config": result.get("asr_config")})

    st.divider()
    st.subheader("Evaluate against a reference")
    ref = st.text_input("Reference transcript", max_chars=MAX_TEXT_CHARS)
    if ref.strip():
        rows = []
        for name, hyp in (("raw", result["raw_transcript"]), ("normalized", result["normalized_transcript"])):
            s, a = corpus_wer([ref], [hyp]), corpus_wer([ref], [hyp], romanize=True)
            rows.append({"system": name, "WER": round(s["wer"], 4), "CER": round(s["cer"], 4),
                         "WER (script-agnostic)": round(a["wer"], 4), "CER (script-agnostic)": round(a["cer"], 4)})
        st.table(rows)
