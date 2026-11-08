"""Whisper ASR wrappers (inference only).

Two interchangeable backends with the same interface (``transcribe_array``,
``transcribe``; ``iter_segments`` on the faster one):

* :class:`FasterWhisperASR` – faster-whisper (CTranslate2, int8 on CPU). No
  PyTorch needed; its Silero VAD runs on onnxruntime. Measured on CoSHE clips:
  model load 3.4 s vs ~60 s, Whisper Base RTF 0.155 vs 1.2-1.4. The default.
* :class:`WhisperASR` – Hugging Face Transformers pipeline, kept for
  experiments and as a fallback when faster-whisper is not installed.

Decoding settings (language, beams, prompt) are explicit experimental
variables: forcing ``language="hindi"`` can suppress English words or render
them in Devanagari, so the plan compares prompted vs unprompted decoding
rather than assuming either is better.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .preprocessing import TARGET_SR, audio_duration, load_audio


@dataclass
class DecodeConfig:
    model_name: str = "openai/whisper-base"
    language: str | None = None  # None = auto-detect; e.g. "hindi", "english"
    task: str = "transcribe"
    num_beams: int = 1
    prompt: str | None = None  # optional initial prompt (e.g. Hinglish vocabulary)
    use_vad: bool = False
    vad_max_chunk_s: float = 25.0
    vad_pad_s: float = 0.25
    chunk_length_s: float = 30.0
    batch_size: int = 8
    fp16: bool = True
    # Script-constrained decoding: block tokens that start characters of these
    # scripts. Whisper often renders Hindi speech in Urdu (Perso-Arabic) script.
    suppress_scripts: tuple[str, ...] = ()
    # Temperature fallback (Whisper paper): re-decode when the output is too
    # repetitive (compression ratio) or low-confidence (avg log-prob).
    fallback: bool = False
    # faster-whisper only: penalise tokens already generated (>1.0 reduces repetition loops).
    repetition_penalty: float = 1.0
    compute_type: str = "int8"
    # faster-whisper with use_vad: decode in VAD chunks of at most this many seconds (0 = whole 30 s windows).
    chunk_s: float = 15.0

    def tag(self) -> str:
        parts = [self.model_name.split("/")[-1], f"lang={self.language or 'auto'}", f"beams={self.num_beams}"]
        if self.repetition_penalty != 1.0:
            parts.append(f"rp={self.repetition_penalty:g}")
        if self.suppress_scripts:
            parts.append("no-" + "+".join(self.suppress_scripts))
        if self.fallback:
            parts.append("fallback")
        if self.prompt:
            parts.append("prompt")
        if self.use_vad:
            parts.append("vad")
        return ",".join(parts)


# UTF-8 lead bytes that begin characters of each script (all 2-byte ranges).
_SCRIPT_LEAD_BYTES = {
    "arabic": {0xD8, 0xD9, 0xDA, 0xDB, 0xDD},  # U+0600–06FF, U+0740–077F
}


def script_token_ids(tokenizer, scripts) -> list[int]:
    """Vocabulary ids whose byte sequence contains a lead byte of ``scripts``.

    Whisper's byte-level BPE splits rare characters into partial-byte tokens,
    so decoding tokens to text misses many of them; we inspect raw bytes via
    the tokenizer's byte decoder instead. Blocking every token containing a
    lead byte makes it impossible to *start* a character of that script.
    """
    lead = set().union(*(_SCRIPT_LEAD_BYTES[s] for s in scripts)) if scripts else set()
    if not lead:
        return []
    byte_decoder = getattr(tokenizer, "byte_decoder", None)
    if byte_decoder is None:  # fast tokenizer: rebuild the GPT-2 byte map
        bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
        cs, n = bs[:], 0
        for b in range(256):
            if b not in bs:
                bs.append(b); cs.append(256 + n); n += 1
        byte_decoder = {chr(c): b for b, c in zip(bs, cs, strict=True)}
    ids = []
    for tok, idx in tokenizer.get_vocab().items():
        try:
            raw = bytes(byte_decoder[c] for c in tok)
        except KeyError:  # special tokens
            continue
        if any(b in lead for b in raw):
            ids.append(idx)
    return sorted(ids)


@dataclass
class ASRResult:
    text: str
    segments: list[dict] = field(default_factory=list)  # {"start","end","text"} seconds
    audio_seconds: float = 0.0
    elapsed_seconds: float = 0.0
    config: str = ""

    @property
    def rtf(self) -> float:
        return self.elapsed_seconds / self.audio_seconds if self.audio_seconds else float("nan")


class WhisperASR:
    def __init__(self, config: DecodeConfig | None = None, device: str | None = None):
        import torch
        from transformers import pipeline

        self.cfg = config or DecodeConfig()
        cuda = torch.cuda.is_available()
        self.device = device or ("cuda:0" if cuda else "cpu")
        dtype = torch.float16 if (cuda and self.cfg.fp16 and self.device != "cpu") else torch.float32
        self.pipe = pipeline("automatic-speech-recognition", model=self.cfg.model_name,
                             device=self.device, torch_dtype=dtype)
        self._vad = None

    # -- helpers ---------------------------------------------------------
    def _script_suppress_ids(self) -> list[int]:
        return script_token_ids(self.pipe.tokenizer, self.cfg.suppress_scripts)

    def _generate_kwargs(self) -> dict:
        kw = {"task": self.cfg.task, "num_beams": self.cfg.num_beams}
        if self.cfg.language:
            kw["language"] = self.cfg.language
        if self.cfg.suppress_scripts:
            if not hasattr(self, "_suppress_cache"):
                base = list(self.pipe.model.generation_config.suppress_tokens or [])
                self._suppress_cache = sorted(set(base) | set(self._script_suppress_ids()))
            kw["suppress_tokens"] = self._suppress_cache
        if self.cfg.fallback:
            kw.update(temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), compression_ratio_threshold=1.35,
                      logprob_threshold=-1.0, do_sample=False)
        if self.cfg.prompt:
            ids =self.pipe.tokenizer.get_prompt_ids(self.cfg.prompt, return_tensors="pt")
            kw["prompt_ids"] = ids.to(self.pipe.device)
        return kw

    def _speech_chunks(self, wav, sr):
        """Silero VAD → merged chunks ≤ vad_max_chunk_s with padding. Returns [(start_s, end_s)]."""
        import torch
        from silero_vad import get_speech_timestamps, load_silero_vad

        if self._vad is None:
            self._vad = load_silero_vad()
        ts = get_speech_timestamps(torch.from_numpy(wav), self._vad, sampling_rate=sr, return_seconds=True)
        dur = audio_duration(wav, sr)
        chunks: list[list[float]] = []
        for t in ts:
            s, e = max(0.0, t["start"] - self.cfg.vad_pad_s), min(dur, t["end"] + self.cfg.vad_pad_s)
            if chunks and s - chunks[-1][1] < 0.3 and e - chunks[-1][0] <= self.cfg.vad_max_chunk_s:
                chunks[-1][1] = e
            else:
                chunks.append([s, e])
        return [tuple(c) for c in chunks]

    # -- main ------------------------------------------------------------
    def transcribe_array(self, wav, sr: int = TARGET_SR) -> ASRResult:
        t0 = time.perf_counter()
        dur = audio_duration(wav, sr)
        gen = self._generate_kwargs()
        segments = []
        if self.cfg.use_vad:
            chunks = self._speech_chunks(wav, sr)
            inputs = [{"raw": wav[int(s * sr):int(e * sr)], "sampling_rate": sr} for s, e in chunks]
            outs = self.pipe(inputs, generate_kwargs=gen, batch_size=self.cfg.batch_size) if inputs else []
            for (s, e), o in zip(chunks, outs, strict=True):
                segments.append({"start": round(s, 2), "end": round(e, 2), "text": o["text"].strip()})
            text = " ".join(seg["text"] for seg in segments if seg["text"])
        else:
            out = self.pipe({"raw": wav, "sampling_rate": sr}, generate_kwargs=gen, return_timestamps=True,
                            chunk_length_s=self.cfg.chunk_length_s if dur > 30 else None,
                            batch_size=self.cfg.batch_size)
            text = out["text"].strip()
            for ch in out.get("chunks", []) or []:
                s, e = ch.get("timestamp", (None, None))
                segments.append({"start": s, "end": e, "text": ch["text"].strip()})
        # U+FFFD is a decoding artefact (generation stopped mid-way through a multi-byte character), not speech.
        text = " ".join(text.replace("�", "").split())
        for seg in segments:
            seg["text"] = " ".join(seg["text"].replace("�", "").split())
        return ASRResult(text=text, segments=segments, audio_seconds=dur,
                         elapsed_seconds=time.perf_counter() - t0, config=self.cfg.tag())

    def transcribe(self, audio_path) -> ASRResult:
        wav, sr = load_audio(audio_path)
        return self.transcribe_array(wav, sr)


# ---------------------------------------------------------------------------
# faster-whisper backend
# ---------------------------------------------------------------------------

_LANG_CODES = {"hindi": "hi", "english": "en", "tamil": "ta"}
_SUPPRESS_CACHE: dict[tuple, list[int]] = {}


def speech_chunks(wav, max_s: float, sr: int = TARGET_SR) -> list[tuple[int, int]]:
    """Silero VAD speech regions, merged into chunks of at most ``max_s`` seconds (split at pauses).

    Decoding chunk by chunk lets text appear after the first chunk instead of after a whole 30 s
    window. Measured on 15 held-out CoSHE clips (Whisper Small, int8): 15 s chunks gave first text
    after 6.4 s on average vs 9.4 s, with WER 0.713 vs 0.728; 10 s chunks were faster but less accurate.
    """
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    regions = get_speech_timestamps(wav, VadOptions(min_silence_duration_ms=300, speech_pad_ms=200,
                                                    max_speech_duration_s=max_s), sampling_rate=sr)
    chunks: list[list[int]] = []
    for r in regions:
        if chunks and (r["end"] - chunks[-1][0]) / sr <= max_s and (r["start"] - chunks[-1][1]) / sr < 1.0:
            chunks[-1][1] = r["end"]
        else:
            chunks.append([r["start"], r["end"]])
    return [(a, b) for a, b in chunks]


def load_faster_model(model_name: str, compute_type: str = "int8", device: str | None = None):
    """Load a faster-whisper model: int8 on CPU (all cores), float16 on CUDA."""
    import os

    from faster_whisper import WhisperModel

    try:
        import ctranslate2

        cuda = ctranslate2.get_cuda_device_count() > 0
    except Exception:
        cuda = False
    device = device or ("cuda" if cuda else "cpu")
    name = model_name.split("/")[-1].removeprefix("whisper-")  # openai/whisper-small -> small
    return WhisperModel(name, device=device, compute_type="float16" if device == "cuda" else compute_type,
                        cpu_threads=(os.cpu_count() or 4) if device == "cpu" else 0)


def _clean(text: str) -> str:
    # U+FFFD is a decoding artefact (generation stopped mid-way through a multi-byte character), not speech.
    return " ".join(text.replace("\ufffd", "").split())


class FasterWhisperASR:
    """faster-whisper (CTranslate2) backend: int8 on CPU, float16 on CUDA. No PyTorch import.

    Built-in safeguards used: Silero VAD (``use_vad``, on by default here), no conditioning on
    previously decoded text (stops one hallucinated loop from feeding the next window), the
    repetition penalty, Urdu-script suppression and optional temperature fallback.
    """

    def __init__(self, config: DecodeConfig | None = None, device: str | None = None, shared_model=None):
        """``shared_model``: an already-loaded ``faster_whisper.WhisperModel`` to reuse. Language,
        script blocking and penalties are decoding options, so one model serves every setting."""
        self.cfg = config or DecodeConfig(use_vad=True)
        if shared_model is not None:
            self.model = shared_model
        else:
            self.model = load_faster_model(self.cfg.model_name, self.cfg.compute_type, device)
        self._suppress = [-1]  # -1 keeps Whisper's default suppressed symbols
        if self.cfg.suppress_scripts:
            key = (id(self.model), tuple(self.cfg.suppress_scripts))
            if key not in _SUPPRESS_CACHE:
                _SUPPRESS_CACHE[key] = script_token_ids(self.model.hf_tokenizer, self.cfg.suppress_scripts)
            self._suppress += _SUPPRESS_CACHE[key]

    def _options(self) -> dict:
        lang = self.cfg.language
        opts = dict(language=_LANG_CODES.get(lang, lang) if lang else None, task=self.cfg.task,
                    beam_size=self.cfg.num_beams, vad_filter=self.cfg.use_vad, condition_on_previous_text=False,
                    suppress_tokens=self._suppress, repetition_penalty=self.cfg.repetition_penalty,
                    initial_prompt=self.cfg.prompt or None)
        if self.cfg.fallback:
            opts.update(temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), compression_ratio_threshold=2.4, log_prob_threshold=-1.0)
        else:
            opts.update(temperature=0.0)
        return opts

    def iter_segments(self, wav, sr: int = TARGET_SR):
        """Yield ``{"start", "end", "text"}`` segments as they are decoded (lazy: decoding happens while iterating)."""
        if sr != TARGET_SR:
            import librosa

            wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
        if not (self.cfg.use_vad and self.cfg.chunk_s > 0):
            pieces = [(0, len(wav))]
            opts = self._options()
        else:
            pieces = speech_chunks(wav, self.cfg.chunk_s)
            opts = {**self._options(), "vad_filter": False}  # VAD already applied by speech_chunks
        for a, b in pieces:   # no speech found: nothing is decoded, so silence cannot be hallucinated into text
            offset = a / TARGET_SR
            segments, _info = self.model.transcribe(wav[a:b], **opts)
            for s in segments:
                text = _clean(s.text)
                if text:
                    yield {"start": round(offset + s.start, 2), "end": round(offset + s.end, 2), "text": text}

    def transcribe_array(self, wav, sr: int = TARGET_SR) -> ASRResult:
        t0 = time.perf_counter()
        segments = list(self.iter_segments(wav, sr))
        return ASRResult(text=" ".join(s["text"] for s in segments), segments=segments,
                         audio_seconds=audio_duration(wav, sr), elapsed_seconds=time.perf_counter() - t0,
                         config="faster-whisper," + self.cfg.tag())

    def transcribe(self, audio_path) -> ASRResult:
        wav, sr = load_audio(audio_path)
        return self.transcribe_array(wav, sr)


def faster_whisper_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return False
    return True


def make_asr(config: DecodeConfig | None = None, backend: str = "auto"):
    """``backend``: ``auto`` (faster-whisper when installed), ``faster`` or ``transformers``."""
    if backend == "faster" or (backend == "auto" and faster_whisper_available()):
        return FasterWhisperASR(config)
    return WhisperASR(config)
