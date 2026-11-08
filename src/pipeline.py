"""End-to-end IndicSwitch pipeline: audio → Whisper → tokens → LID → correction.

Each stage is independently usable; :meth:`IndicSwitch.process_text` runs the
text half only (useful for evaluating LID + correction on existing ASR output
without re-running Whisper).
"""

from __future__ import annotations

import time
from typing import Any

from .correction import CodeSwitchCorrector, CorrectionConfig
from .language_identification import BaseLID, load_lid
from .preprocessing import normalize_text, tokenize_with_offsets


class IndicSwitch:
    def __init__(self, asr=None, lid: BaseLID | str | None = "rule", correction: CorrectionConfig | None = None,
                 decode_config=None):
        self._asr = asr
        self._decode_config = decode_config
        self.lid = lid if isinstance(lid, BaseLID) else load_lid(lid)
        self.corrector = CodeSwitchCorrector(correction)

    @property
    def asr(self):
        if self._asr is None:  # lazy: text-only use never loads Whisper
            from .asr import make_asr  # faster-whisper when installed, else Transformers

            self._asr = make_asr(self._decode_config)
        return self._asr

    def process_text(self, raw_transcript: str, timings: dict | None = None) -> dict[str, Any]:
        timings = {} if timings is None else timings
        t = time.perf_counter()
        text = normalize_text(raw_transcript)
        toks = tokenize_with_offsets(text)
        words = [x.text for x in toks]
        timings["tokenize_s"] = time.perf_counter() - t

        t = time.perf_counter()
        tags, confs = self.lid.predict(words) if words else ([], [])
        timings["lid_s"] = time.perf_counter() - t

        t = time.perf_counter()
        corr = self.corrector.correct(text, tags, confs, toks)
        timings["correction_s"] = time.perf_counter() - t

        return {
            "raw_transcript": raw_transcript,
            "tokens": words,
            "offsets": [(x.start, x.end) for x in toks],
            "language_tags": tags,
            "confidences": [round(c, 4) for c in confs],
            "normalized_transcript": corr.text,
            "edits": [e.__dict__ for e in corr.edits],
            "lid_model": self.lid.name,
            "timings": timings,
        }

    def __call__(self, audio_path) -> dict[str, Any]:
        from .preprocessing import load_audio

        wav, sr = load_audio(audio_path)
        return self.run_array(wav, sr)

    def run_array(self, wav, sr: int) -> dict[str, Any]:
        """Full pipeline on an already-loaded mono waveform."""
        t0 = time.perf_counter()
        return self.finish(self.asr.transcribe_array(wav, sr), t0)

    def finish(self, asr_res, t0: float) -> dict[str, Any]:
        """Tag and correct a finished ASR result; ``t0`` is when the audio work started.
        Shared by the one-shot and the streaming paths."""
        timings: dict = {"asr_s": asr_res.elapsed_seconds}
        out = self.process_text(asr_res.text, timings)
        timings["total_s"] = time.perf_counter() - t0
        out.update({
            "segments": asr_res.segments,
            "audio_seconds": asr_res.audio_seconds,
            "rtf": timings["total_s"] / asr_res.audio_seconds if asr_res.audio_seconds else None,
            "asr_config": asr_res.config,
        })
        return out


_default: IndicSwitch | None = None


def indic_switch(audio_path, lid: str | None = None, correction: CorrectionConfig | None = None) -> dict[str, Any]:
    """Single-call interface from the plan. Reuses a cached pipeline when
    called with default arguments."""
    global _default
    if lid is None and correction is None:
        if _default is None:
            from pathlib import Path

            default_lid = Path(__file__).resolve().parent.parent / "models" / "lid_model"
            spec = str(default_lid) if (default_lid / "config.json").exists() else (
                str(default_lid.parent / "ngram_lid.pkl") if (default_lid.parent / "ngram_lid.pkl").exists() else "rule")
            _default = IndicSwitch(lid=spec)
        return _default(audio_path)
    return IndicSwitch(lid=lid or "rule", correction=correction)(audio_path)
