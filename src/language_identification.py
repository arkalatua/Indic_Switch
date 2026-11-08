"""Token-level language identification (HI / EN / OTH).

Three interchangeable backends share the interface
``predict(words) -> (tags, confidences)``:

* :class:`ScriptRuleLID` – script heuristic, no training (sanity floor).
* :class:`NgramLID`      – character n-gram + context features, Logistic Regression.
* :class:`XLMRLID`       – fine-tuned ``FacebookAI/xlm-roberta-base`` token classifier.

Label alignment convention for XLM-R (documented per the plan): each word's
label is assigned to its *first* subword; continuation subwords and special
tokens get ``-100`` and are ignored by the loss. At inference, a word's
prediction and confidence are the softmax over its first subword's logits.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import pickle
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence

from .preprocessing import word_script

LABELS = ["HI", "EN", "OTH"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

_PUNCT_OR_NUM = re.compile(r"^[\W\d_]+$", re.UNICODE)


def is_trivially_oth(word: str) -> bool:
    """Punctuation, digits and symbols are OTH by definition."""
    return bool(_PUNCT_OR_NUM.match(word))


# ---------------------------------------------------------------------------
# Data I/O: one JSON object per line: {"tokens": [...], "tags": [...]}
# ---------------------------------------------------------------------------

def read_lid_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


class BaseLID:
    name = "base"

    def predict(self, words: Sequence[str]) -> tuple[list[str], list[float]]:
        raise NotImplementedError

    def predict_batch(self, sentences: Sequence[Sequence[str]]) -> list[tuple[list[str], list[float]]]:
        return [self.predict(s) for s in sentences]


# ---------------------------------------------------------------------------
# Rule baseline
# ---------------------------------------------------------------------------

class ScriptRuleLID(BaseLID):
    """Devanagari → HI, Latin → EN, punctuation/digits → OTH.

    Deliberately naive: it cannot recognise Romanized Hindi, which is exactly
    the gap a learned model must close. Useful as a floor in the results table.
    """

    name = "script_rule"

    def predict(self, words):
        tags, confs = [], []
        for w in words:
            s = word_script(w)
            if is_trivially_oth(w):
                tags.append("OTH"); confs.append(1.0)
            elif s == "devanagari":
                tags.append("HI"); confs.append(0.9)
            elif s == "latin":
                tags.append("EN"); confs.append(0.5)
            else:
                tags.append("OTH"); confs.append(0.5)
        return tags, confs


# ---------------------------------------------------------------------------
# Character n-gram + Logistic Regression
# ---------------------------------------------------------------------------

def _word_feature_text(w: str) -> str:
    return f"<{w.lower()}>"


def _space_split(s: str) -> list[str]:  # module-level so the vectorizer pickles
    return s.split(" ")


def _context_feature_text(words: Sequence[str], i: int) -> str:
    prev_w = words[i - 1].lower() if i > 0 else "<s>"
    next_w = words[i + 1].lower() if i + 1 < len(words) else "</s>"
    return f"P={prev_w} N={next_w} S={word_script(words[i])}"


class NgramLID(BaseLID):
    """Character 1–5-gram TF-IDF of the word, plus previous/next word and
    script identity as context features, fed to a multinomial Logistic
    Regression."""

    name = "char_ngram_lr"

    def __init__(self, C: float = 4.0, ngram_range=(1, 5), max_features: int = 200_000, rule_oth: bool = True):
        self.C = C
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.rule_oth = rule_oth
        self._char_vec = self._ctx_vec = self._clf = None

    def _features(self, sentences: Sequence[Sequence[str]], fit: bool = False):
        from scipy.sparse import hstack

        words = [_word_feature_text(w) for s in sentences for w in s]
        ctx = [_context_feature_text(s, i) for s in sentences for i in range(len(s))]
        if fit:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._char_vec = TfidfVectorizer(analyzer="char", ngram_range=self.ngram_range,
                                             max_features=self.max_features, sublinear_tf=True, lowercase=False)
            self._ctx_vec = TfidfVectorizer(analyzer=_space_split, lowercase=False)
            return hstack([self._char_vec.fit_transform(words), 0.5 * self._ctx_vec.fit_transform(ctx)]).tocsr()
        return hstack([self._char_vec.transform(words), 0.5 * self._ctx_vec.transform(ctx)]).tocsr()

    def fit(self, rows: Sequence[dict]) -> "NgramLID":
        from sklearn.linear_model import LogisticRegression

        sents = [r["tokens"] for r in rows]
        y = [t for r in rows for t in r["tags"]]
        X = self._features(sents, fit=True)
        self._clf = LogisticRegression(C=self.C, max_iter=3000)
        self._clf.fit(X, y)
        return self

    def predict_batch(self, sentences):
        sentences = [list(s) for s in sentences]
        flat_n = sum(len(s) for s in sentences)
        if flat_n == 0:
            return [([], []) for _ in sentences]
        probs = self._clf.predict_proba(self._features(sentences))
        classes = list(self._clf.classes_)
        out, k = [], 0
        for s in sentences:
            tags, confs = [], []
            for w in s:
                p = probs[k]; k += 1
                if self.rule_oth and is_trivially_oth(w):
                    tags.append("OTH"); confs.append(1.0)
                    continue
                j = int(p.argmax())
                tags.append(str(classes[j])); confs.append(float(p[j]))
            out.append((tags, confs))
        return out

    def predict(self, words):
        return self.predict_batch([words])[0]

    def save(self, path: str | Path) -> None:
        """Pickle the model and write a SHA-256 sidecar (``<path>.sha256``)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = pickle.dumps(self)
        path.write_bytes(data)
        Path(f"{path}.sha256").write_text(hashlib.sha256(data).hexdigest(), encoding="ascii")

    @staticmethod
    def load(path: str | Path) -> "NgramLID":
        """Load a model saved by :meth:`save`.

        Unpickling can execute arbitrary code, so only load files you produced
        yourself. The SHA-256 sidecar check refuses files that were modified
        or replaced after saving, or that were never saved by this code.
        """
        path = Path(path)
        sidecar = Path(f"{path}.sha256")
        if not sidecar.exists():
            raise ValueError(f"Refusing to load {path}: missing checksum file {sidecar.name}")
        data = path.read_bytes()
        if not hmac.compare_digest(hashlib.sha256(data).hexdigest(), sidecar.read_text(encoding="ascii").strip()):
            raise ValueError(f"Refusing to load {path}: checksum mismatch (file modified or replaced)")
        model = pickle.loads(data)  # noqa: S301 - integrity-checked file we wrote ourselves
        if not isinstance(model, NgramLID):
            raise TypeError(f"{path} does not contain an NgramLID model")
        return model


# ---------------------------------------------------------------------------
# XLM-R token classifier
# ---------------------------------------------------------------------------

@dataclass
class XLMRTrainConfig:
    model_name: str = "FacebookAI/xlm-roberta-base"
    learning_rate: float = 2e-5
    batch_size: int = 4
    grad_accum: int = 4
    epochs: int = 3
    max_length: int = 128
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    fp16: bool = True  # turned off automatically without CUDA
    early_stopping_patience: int = 2
    seed: int = 13
    eval_steps: int | None = None  # None → evaluate each epoch


def align_labels(word_ids: Sequence[int | None], word_labels: Sequence[int]) -> list[int]:
    """First-subword labelling; continuation subwords and specials → -100."""
    out, prev = [], None
    for wid in word_ids:
        if wid is None or wid == prev:
            out.append(-100)
        else:
            out.append(word_labels[wid])
        prev = wid
    return out


def train_xlmr_lid(train_rows: Sequence[dict], val_rows: Sequence[dict], out_dir: str | Path,
                   cfg: XLMRTrainConfig | None = None) -> dict:
    """Fine-tune XLM-R for token LID. Saves the best checkpoint (by
    validation macro-F1) and tokenizer to ``out_dir``; returns final metrics."""
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification,
                              EarlyStoppingCallback, Trainer, TrainingArguments, set_seed)

    cfg = cfg or XLMRTrainConfig()
    set_seed(cfg.seed)
    tok = AutoTokenizer.from_pretrained(cfg.model_name)

    def encode(batch):
        enc = tok(batch["tokens"], is_split_into_words=True, truncation=True, max_length=cfg.max_length)
        enc["labels"] = [align_labels(enc.word_ids(i), [LABEL2ID[t] for t in tags])
                         for i, tags in enumerate(batch["tags"])]
        return enc

    def to_ds(rows):
        ds = Dataset.from_list([{"tokens": r["tokens"], "tags": r["tags"]} for r in rows])
        return ds.map(encode, batched=True, remove_columns=["tokens", "tags"])

    train_ds, val_ds = to_ds(train_rows), to_ds(val_rows)

    def compute_metrics(p):
        from sklearn.metrics import f1_score, accuracy_score

        logits, labels = p
        preds = logits.argmax(-1)
        mask = labels != -100
        y, yh = labels[mask], preds[mask]
        return {"macro_f1": f1_score(y, yh, average="macro", labels=list(range(len(LABELS))), zero_division=0),
                "accuracy": accuracy_score(y, yh)}

    model = AutoModelForTokenClassification.from_pretrained(
        cfg.model_name, num_labels=len(LABELS), label2id=LABEL2ID, id2label=ID2LABEL)
    use_cuda = torch.cuda.is_available()
    strategy = "steps" if cfg.eval_steps else "epoch"
    args = TrainingArguments(
        output_dir=str(Path(out_dir) / "_checkpoints"),
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size * 4,
        gradient_accumulation_steps=cfg.grad_accum,
        num_train_epochs=cfg.epochs,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        fp16=cfg.fp16 and use_cuda,
        eval_strategy=strategy,
        save_strategy=strategy,
        eval_steps=cfg.eval_steps,
        save_steps=cfg.eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        report_to=[],
        seed=cfg.seed,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                      data_collator=DataCollatorForTokenClassification(tok),
                      compute_metrics=compute_metrics,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.early_stopping_patience)])
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    trainer.train()
    metrics = trainer.evaluate()
    metrics["peak_gpu_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3) if use_cuda else None
    metrics["gpu"] = torch.cuda.get_device_name(0) if use_cuda else "cpu"
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    with open(Path(out_dir) / "train_config.json", "w", encoding="utf-8") as f:
        json.dump({"config": asdict(cfg), "val_metrics": metrics,
                   "alignment": "first-subword label; continuation/special = -100"}, f, indent=2)
    return metrics


class XLMRLID(BaseLID):
    name = "xlmr"

    def __init__(self, model_dir: str | Path, device: str | None = None, max_words: int = 60,
                 max_length: int = 128, rule_oth: bool = True):
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForTokenClassification.from_pretrained(str(model_dir)).to(self.device).eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.max_words = max_words
        self.max_length = max_length
        self.rule_oth = rule_oth

    def _predict_window(self, windows: list[list[str]]):
        import torch

        enc = self.tok(windows, is_split_into_words=True, truncation=True, max_length=self.max_length,
                       padding=True, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**{k: v.to(self.device) for k, v in enc.items()}).logits
        probs = logits.softmax(-1).cpu().numpy()
        results = []
        for b, words in enumerate(windows):
            first = {}
            for pos, wid in enumerate(enc.word_ids(b)):
                if wid is not None and wid not in first:
                    first[wid] = pos
            tags, confs = [], []
            for i, w in enumerate(words):
                if self.rule_oth and is_trivially_oth(w):
                    tags.append("OTH"); confs.append(1.0); continue
                if i not in first:  # truncated away — should be rare given max_words
                    tags.append("OTH"); confs.append(0.0); continue
                p = probs[b, first[i]]
                j = int(p.argmax())
                tags.append(self.id2label[j]); confs.append(float(p[j]))
            results.append((tags, confs))
        return results

    def predict_batch(self, sentences, batch_size: int = 16):
        # Split long sentences into windows of ``max_words`` words.
        windows, owners = [], []
        for si, s in enumerate(sentences):
            s = list(s)
            for k in range(0, max(len(s), 1), self.max_words):
                windows.append(s[k:k + self.max_words]); owners.append(si)
        merged = [([], []) for _ in sentences]
        for k in range(0, len(windows), batch_size):
            chunk = windows[k:k + batch_size]
            for (tags, confs), si in zip(self._predict_window(chunk), owners[k:k + batch_size], strict=True):
                merged[si][0].extend(tags); merged[si][1].extend(confs)
        return merged

    def predict(self, words):
        return self.predict_batch([words])[0]


def load_lid(spec: str | Path | None) -> BaseLID:
    """``"rule"`` → ScriptRuleLID; ``*.pkl`` → NgramLID; directory → XLMRLID."""
    if spec is None or str(spec) == "rule":
        return ScriptRuleLID()
    p = Path(spec)
    if p.suffix == ".pkl":
        return NgramLID.load(p)
    if p.is_dir() and (p / "config.json").exists():
        return XLMRLID(p)
    raise FileNotFoundError(f"No LID model found at {spec!r}")
