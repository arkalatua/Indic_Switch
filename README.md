# IndicSwitch

**Code-switch-aware speech recognition post-processing for Hinglish.** IndicSwitch takes a pretrained Whisper model's output and adds:
* a token-level language tagger (HI / EN / OTH)
* a cautious, confidence-gated correction step that is aware of the writing script (Devanagari or Roman)

It keeps what the speaker said rather than translating or rewriting it, and the raw Whisper transcript is always kept alongside the corrected one.

> Scope: an ASR *enhancement* system built on Google Colab Free, not a foundation model trained from scratch.

## Architecture

```
audio ─► load (mono, 16 kHz) ─► [Silero VAD] ─► Whisper Base ─► raw transcript ──────────────────────┐
                                                                    │                                │
                                          offset-preserving tokenizer                                │
                                                                    │                                │
                        LID: XLM-R  |  char n-gram + LR  |  script rule  →  tags + confidences       │
                                                                    │                                │
            conservative corrector (loop collapse · spelling canonicalization · script rendering)    │
                                                                    │                                │
                                    normalized transcript + edit log ◄───────── raw kept ◄───────────┘
```

| Stage | Module | Notes |
|---|---|---|
| Audio ingestion | `src/preprocessing.py` | soundfile/librosa, mono 16 kHz |
| Segmentation | `src/asr.py` | optional Silero VAD, chunks of 25 s or less with padding |
| ASR | `src/asr.py` | `openai/whisper-base`, inference only. Language, beams and prompt are experimental variables |
| Tokenization | `src/preprocessing.py` | words + punctuation with character offsets |
| Language ID | `src/language_identification.py` | fine-tuned `FacebookAI/xlm-roberta-base`; char n-gram LR baseline; script rule floor |
| Post-processing | `src/correction.py` | confidence-gated, token-local, logged edits; protects names, acronyms, OTH and user-listed words |
| Transliteration | `src/transliteration.py` | rule-based Devanagari → Roman (with schwa deletion); optional IndicXlit for Roman → Devanagari |
| Evaluation | `src/evaluation.py` | WER, CER (exact and script-agnostic), LID macro-F1, boundary F1, switch density, RTF, peak memory |
| Pipeline | `src/pipeline.py` | `indic_switch(audio_path)` returns raw text, tokens, tags, confidences, normalized text, edits and timings |

### Correction rules (each one can be switched off for ablations)
1. **Loop collapse**: Whisper sometimes repeats a phrase over and over. Any 1–4-word phrase repeated 4 or more times in a row is cut to one copy. Natural repetition such as "haan haan" is kept.
2. **Canonicalization**: Romanized Hindi variant spellings (`nhi`, `nahin` → `nahi`; `bohot` → `bahut`) are unified. This applies **only** to tokens tagged HI with confidence at or above the threshold (default 0.80). Variants that are also English words (`he`, `me`, `or`, `the`, `men`) and Hindi words with their own meaning (`nai`, `samaj`) are left out of the lexicon.
3. **Script rendering**:
   * `roman`: English loanwords that Whisper wrote in Devanagari (`मीटिंग`) become English (`meeting`), unless the tagger is confident the word is Hindi. Other Devanagari is romanized.
   * `native`: confident HI tokens written in Roman script go to Devanagari via IndicXlit.
   * `mixed`: each token is written in its own language's script, i.e. Hindi in Devanagari and English in Latin. This is the CoSHE-500 reference convention. It combines the English step of `roman` with the Hindi step of `native`.

## Quick start

```bash
pip install -r requirements.txt            # on Colab, torch is preinstalled
python -m pytest -q                        # 72 tests, CPU only

# Day 2: language ID data (COMI-LINGUA, CC-BY-4.0)
python scripts/prepare_comi_lingua.py

# Day 3: language ID models
python scripts/train_ngram_lid.py                                   # CPU, minutes
python scripts/train_xlmr_lid.py --batch-size 4 --grad-accum 4      # GPU (Colab T4)
python scripts/evaluate_lid.py --model models/lid_model --split test

# Day 2: ASR evaluation manifest (check the dataset card and column names first)
python scripts/build_manifest.py hf --dataset Trelis/CoSHE-500 --split eval --text-col transcription --name coshe500 --all-test

# Days 1/6: ASR runs (E1/E2), then the experiment matrix and ablations (E3–E6)
python scripts/run_asr.py --manifest data/manifests/coshe500.csv --split test
python scripts/run_asr.py --manifest data/manifests/coshe500.csv --split test --language hindi --beams 3
python scripts/run_experiments.py experiments/asr/*.jsonl --lid models/lid_model --lid models/ngram_lid.pkl --lid rule

# Day 7: the web app
uvicorn server.app:app --host 127.0.0.1 --port 8000     # then open http://127.0.0.1:8000
```

## Web app

`server/app.py` (FastAPI) serves a JSON API and the frontend in `web/`, which is plain HTML, CSS and JS with no build step. You can upload or record audio, or paste text. The app shows:
* the corrected transcript, with each fix underlined
* every word tagged Hindi, English or other, dashed when the tagger was too unsure to fix it
* what Whisper first wrote, with removed and changed words marked
* the fix log with reasons, word and switch counts, the real-time factor, and scoring against a reference
* a JSON download of the full result

The browser converts any audio format to 16 kHz mono WAV before upload, so the server only parses WAV and needs no FFmpeg. Nothing is stored on the server.

| Endpoint | Purpose |
|---|---|
| `GET /api/config` | Available language models (fixed ids) and limits |
| `POST /api/text` | Tag and correct text: `{"text": "...", "lid": "ngram", "target_script": "keep", ...}` |
| `POST /api/audio` | Multipart: `file` (WAV) plus `settings` (JSON: `model` base/small/tiny, `language`, `block_urdu`, correction settings) |
| `POST /api/audio/stream` | Same input as `/api/audio`, answered as Server-Sent Events: `start`, one `segment` per decoded chunk, then `result` (what the page uses, so text appears while the rest is transcribed) |
| `POST /api/score` | WER and CER of the raw and corrected text against a reference |

Speech recognition uses **faster-whisper** (CTranslate2, int8 on CPU) with Whisper Small by default. It is loaded when the server starts, with no PyTorch import (644 MB resident). If faster-whisper is not installed, the server falls back to the Transformers pipeline.

The design follows the taste-skill system, dealt rather than picked: palette forest-pale-lime, Mona Sans + Geist Mono type, sharp radius, arrow buttons. Its CSS and motion files are vendored unmodified in `web/assets/kit/` under their MIT licence. Fonts are self-hosted under the SIL OFL.

Settings via environment variables:
* `INDICSWITCH_HTTPS=1`: send HSTS when served over HTTPS.
* `INDICSWITCH_TRUST_PROXY=1`: rate-limit by `X-Forwarded-For`, only behind your own proxy.
* `INDICSWITCH_MAX_JOBS`: the number of concurrent heavy jobs (default 2).

Python API:
```python
from src.pipeline import IndicSwitch
from src.correction import CorrectionConfig

p = IndicSwitch(lid="models/ngram_lid.pkl", correction=CorrectionConfig(min_confidence=0.8))
p.process_text("main kal office nhi ja raha, bohot kaam hai")   # text-only (no Whisper)
p("clip.wav")                                                   # full pipeline
```

## Colab notebooks

| Notebook | Day | Purpose |
|---|---|---|
| `01_whisper_baseline.ipynb` | 1 | audio basics, Whisper Base with auto/hindi/english decoding, baseline table |
| `02_dataset_preparation.ipynb` | 2 | inspect COMI-LINGUA / CoSHE-500, language ID splits, ASR manifest |
| `03_lid_training.ipynb` | 3 | n-gram baseline, label-alignment check, XLM-R fine-tuning, error analysis |
| `04_correction_pipeline.ipynb` | 4–5 | corrector examples, IndicXlit, `indic_switch()`, stress tests (silence, noise, long audio, VAD) |
| `05_evaluation.ipynb` | 6 | experiment matrix, switch-density breakdown, improved and degraded examples |

Each notebook's first cell clones this repo on Colab and installs the requirements.

## Experiment matrix

| ID | System | Question |
|---|---|---|
| E1 | Whisper Base, default decoding | baseline |
| E2 | Whisper Base, alternate beam/language/prompt | does decoding change code-mixed output? |
| E3 | Whisper + XLM-R language ID | how accurate is token language ID? |
| E4 | Whisper + XLM-R language ID + correction | does post-processing help WER/CER or usability? |
| E5 | Whisper + n-gram language ID + correction | can a cheaper tagger do as well? |
| E6 | Script conversion (`roman` / `native`) | effect of script on readability and metrics |

Fair-comparison rules:
* Every system runs on the same clips.
* Scoring normalization is fixed.
* Rules are tuned on validation only, never on test.
* Raw and post-processed scores are reported separately.
* Regressions (`degraded` column) are always reported.

## Results

*Final table to be filled in from `experiments/results.csv` and `experiments/lid_*_test.json` after the full Colab runs. No numbers are claimed before they are measured.*

### Measured so far

**Language ID on the COMI-LINGUA official test split** (5,000 sentences, 121,537 tokens, CPU):

| Model | Macro-F1 | Accuracy | F1 HI / EN / OTH | Boundary F1 | Throughput |
|---|---|---|---|---|---|
| Script rule (no training) | 0.836 | 0.839 | n/a | 0.723 | 200k tok/s |
| Char n-gram + LR | **0.969** | **0.971** | 0.979 / 0.942 / 0.986 | **0.908** | 18k tok/s |
| XLM-R base | not yet trained (needs GPU) | | | | |

The most frequent errors are the genuinely ambiguous words `me` (English "me" vs. Hindi "में") and `to` (English "to" vs. Hindi "तो").

**ASR smoke tests.** These are *not* final results: only the first 20 or 5 CoSHE-500 clips, run on CPU with nothing tuned.

| Clips | Decoding | Raw WER | + correction | Notes |
|---|---|---|---|---|
| 20 | Base, auto language | 1.104 | 0.998 | often outputs **Urdu script** |
| 20 | Base, `hindi`, 3 beams | 1.516 | 0.986 | hallucination loops; sometimes translates to English |
| 20 | Base, `hindi`, Urdu tokens blocked | 1.277 | n/a | drifts into Thai/Tamil script instead |
| 20 | Base, `hindi`, Urdu blocked, fallback | 1.036 | n/a | RTF 6.7 on CPU |
| 5 | Base, `hindi`, 3 beams | 1.337 | 0.952 | |
| 5 | **Small**, `hindi`, Urdu blocked | 1.353 | 0.959 | correct script and real words, but loops truncate content (CER 0.966 → 0.904) |

Takeaways:
1. Whisper Base is too weak on conversational Hinglish. Use Whisper Small for *inference* on the Colab T4; only fine-tuning Small is out of scope.
2. Loop collapse is the correction rule with the most measurable effect. It made no utterance worse in these runs.

**Faster engine (faster-whisper, CTranslate2 int8, 8-core CPU).** Same clips, same scoring. Settings: `hindi`, Urdu tokens blocked, Silero VAD, no conditioning on previous text, greedy decoding.

| Clips | Model | Repetition penalty | RTF | Raw WER | Raw CER |
|---|---|---|---|---|---|
| 5 | Base | 1.0 | **0.155** (old engine: 1.2–1.4) | 0.995 | 0.948 |
| 5 | Small | 1.0 | 0.516 | 0.975 | 0.840 |
| 5 | Small | 1.1 | 0.881 | 0.826 | 0.692 |
| 15 held-out | Small | 1.0 | 0.728 | 0.859 | 0.722 |
| 15 held-out | **Small** | **1.1** | 0.771 | **0.728** | **0.563** |

* The repetition penalty was chosen on clips 1–5, then checked on clips 6–20, which were not used for the choice. On those it cut WER by 15% and CER by 22% (relative), so it is the web app's default (`INDICSWITCH_REPETITION_PENALTY`).
* Model load takes 3.4 s, against about 60 s for the Transformers pipeline. The web server holds Whisper Small in 644 MB with no PyTorch import.
* With the loops gone, the text correction rules barely change WER or CER on this audio (held-out CER 0.563 raw vs 0.567 corrected). The decoding settings, not the correction rules, are where the accuracy came from.
* **Chunked decoding** (the web app's default): the audio is cut at pauses into chunks of up to 15 s and decoded chunk by chunk, so text streams to the page sooner. On the 15 held-out clips the first text arrived after 6.4 s on average, against 9.4 s with whole 30 s windows, with WER 0.713 (vs 0.728) and CER 0.547 (vs 0.563) at the same speed. 10 s chunks were faster still but less accurate (WER 0.763). Silence produces no text at all.
* These clips are still part of CoSHE-500's evaluation split, and the held-out clips informed two choices (penalty, chunk size), so treat these as development numbers until the full 500-clip run.

| System | WER | CER | WER (script-agnostic) | LID macro-F1 | Boundary F1 | RTF | Hardware |
|---|---|---|---|---|---|---|---|
| | | | | | | | |

## Limitations
* Whisper is frozen, so post-processing cannot fix acoustic errors. It can only make spelling and script consistent and remove obvious hallucination loops. If WER does not improve, the contribution is the language ID accuracy and the analysis of why.
* The lexicons are small seed lists. Grow them from validation-split error analysis.
* Romanized Hindi spelling varies a lot, so reference transcript conventions strongly affect WER. Script-agnostic WER is reported for this reason.
* Tanglish (Tamil–English) is a stretch goal. The tokenizer and script detection already handle Tamil script, but there is no Tamil lexicon or language ID model yet.

## Repository layout
```
notebooks/      Colab notebooks 01–05 (thin wrappers over src/ and scripts/)
src/            preprocessing, asr, language_identification, correction, transliteration, evaluation, pipeline, data_sources
scripts/        prepare_comi_lingua, train_ngram_lid, train_xlmr_lid, evaluate_lid, build_manifest, run_asr, run_experiments
server/         app.py: FastAPI backend for the web app
web/            the web app frontend: index.html, assets/ (app.css, app.js, fonts, vendored kit)
data/           lexicons/, manifests/, processed/ (see data/README.md)
models/         lid_model/ (not in Git), ngram_lid.pkl (not in Git)
experiments/    asr/, results.csv, ablations.json, lid_*.json
tests/          pytest suite
```

## References
Whisper (arXiv 2212.04356) · IndicSUPERB (arXiv 2208.11761) · HingBERT (WILDRE 2022) · COMI-LINGUA (Findings of EMNLP 2025) · IITG-HingCoS · IndicXlit

## Security
See [SECURITY.md](SECURITY.md) for the security review, how to deploy the demo safely over HTTPS, and how to re-run the checks.
