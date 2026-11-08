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


def test_decode_config_tag():
    tag = DecodeConfig(language="hindi", suppress_scripts=("arabic",), fallback=True).tag()
    assert tag == "whisper-base,lang=hindi,beams=1,no-arabic,fallback"
