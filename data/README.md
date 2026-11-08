# Data

Large files (audio, processed JSONL, checkpoints) are **not** committed. Manifests and stats are small and are committed.

| Path | Contents | In Git |
|---|---|---|
| `lexicons/hinglish_variants.json` | Seed Romanized-Hindi spelling variants → canonical form | yes |
| `lexicons/en_devanagari.json` | Seed English loanwords as written in Devanagari → English | yes |
| `processed/lid_{train,validation,test}.jsonl` | COMI-LINGUA LID examples `{"tokens": [...], "tags": [...]}` | no (regenerate) |
| `processed/lid_stats.json` | Split sizes, tag distribution, rows skipped, duplicates removed | yes |
| `manifests/*.csv` | `id, audio_path, transcript, source, licence, speaker_id, duration, split, extra` | yes |
| `audio/<name>/` | 16 kHz mono WAV decoded by `scripts/build_manifest.py` | no |
| `samples/` | Your own demo clips | no |

## Sources and licences (check each dataset card before use)

| Dataset | Use | Licence / access | Notes |
|---|---|---|---|
| [COMI-LINGUA](https://huggingface.co/datasets/LingoIITGN/COMI-LINGUA) (`LID`) | LID train/val/test | CC-BY-4.0 | Text only. Uses human annotator columns (majority vote), never `Predicted tags`. Labels `hi`/`en`/`ot` → `HI`/`EN`/`OTH`. |
| [CoSHE-500](https://huggingface.co/datasets/Trelis/CoSHE-500) | Held-out Hinglish ASR evaluation (500 clips, split `eval`) | CC-BY-NC-4.0 (non-commercial, attribute) | Columns `audio`, `transcription`. References: Hindi in Devanagari, English in Latin. No speaker ids, so used as test only. |
| [IITG-HingCoS](https://www.iitg.ac.in/eee/emstlab/HingCoS_Database/HingCoS.html) | Hinglish ASR eval | Access request required | Request early. Use `build_manifest.py folder` once you have it. |
| Own recordings | Real-world check | Consent forms | Speakers disjoint across splits; transcripts manually verified. |

## Splitting rules
* LID: COMI-LINGUA's official `test` split is kept intact. Validation is carved from `train`, grouping near-duplicate sentences. Train sentences that duplicate a test sentence are dropped.
* ASR: `src.preprocessing.assign_splits` unions rows that share a speaker **or** a near-duplicate transcript into one group, then splits by group. The result is speaker-independent and duplicate-safe as far as the metadata allows.
* No audio–text pairs are synthesized.

## Reference transcript convention
WER/CER use `src.evaluation.scoring_normalize`: NFC, lowercase, punctuation stripped, whitespace collapsed. The *script-agnostic* variant also romanizes Devanagari, so a correct Hindi word written in a different script is not counted as an error. Both are always reported.
