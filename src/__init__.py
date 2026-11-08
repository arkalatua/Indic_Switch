"""IndicSwitch: code-switch-aware post-processing for pretrained ASR on Indian languages.

Modules are kept import-light: heavy dependencies (torch, transformers, librosa)
are imported lazily inside the functions that need them, so the text-side
components (preprocessing, LID baseline, correction, evaluation) run on a plain
CPU environment.
"""

__version__ = "0.1.0"
