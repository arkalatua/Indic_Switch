from src.asr import DecodeConfig, script_token_ids


def _gpt2_byte_encoder():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {b: chr(c) for b, c in zip(bs, cs, strict=True)}


class FakeTok:
    """Byte-level BPE vocab: one token per string below (no byte_decoder attr → exercises the rebuild path)."""

    def __init__(self, pieces):
        enc = _gpt2_byte_encoder()
        self.vocab = {"".join(enc[b] for b in p.encode("utf-8")) if isinstance(p, str) else
                      "".join(enc[b] for b in p): i for i, p in enumerate(pieces)}
        self.vocab["<|endoftext|>"] = len(pieces)

    def get_vocab(self):
        return self.vocab


def test_script_token_ids_blocks_arabic_only():
    pieces = ["hello", " नहीं", "کہ", b"\xd9", b"\x87", " office", "ہے"]  # partial bytes included
    ids = script_token_ids(FakeTok(pieces), ("arabic",))
    assert ids == [2, 3, 6]  # full Arabic tokens + the lone Arabic lead byte; not the lone continuation byte


def test_speech_chunks_silence_and_limits():
    import numpy as np

    pytest = __import__("pytest")
    pytest.importorskip("faster_whisper")
    from src.asr import speech_chunks

    assert speech_chunks(np.zeros(16000 * 5, dtype=np.float32), 15) == []   # silence: nothing to decode
    noise = (np.random.default_rng(0).standard_normal(16000 * 40) * 0.3).astype(np.float32)
    for a, b in speech_chunks(noise, 15):                                   # whatever VAD finds, chunks stay <= 15 s
        assert 0 <= a < b <= len(noise) and (b - a) / 16000 <= 15.5


def test_decode_config_tag():
    tag = DecodeConfig(language="hindi", suppress_scripts=("arabic",), fallback=True).tag()
    assert tag == "whisper-base,lang=hindi,beams=1,no-arabic,fallback"
