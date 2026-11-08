"""Whisper ASR wrapper (inference only) with optional Silero VAD segmentation.

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

    def tag(self) -> str:
        parts = [self.model_name.split("/")[-1], f"lang={self.language or 'auto'}", f"beams={self.num_beams}"]
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
        return ASRResult(text=text, segments=segments, audio_seconds=dur,
                         elapsed_seconds=time.perf_counter() - t0, config=self.cfg.tag())

    def transcribe(self, audio_path) -> ASRResult:
        wav, sr = load_audio(audio_path)
        return self.transcribe_array(wav, sr)
