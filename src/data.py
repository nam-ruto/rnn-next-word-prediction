# src/data.py
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import re
from typing import Iterable


SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


def tokenize(text: str) -> list[str]:
    """
    Simple word tokenizer that keeps punctuation as separate tokens.
    Example: "Hello, world!" -> ["hello", ",", "world", "!"]
    """
    return re.findall(r"\w+|[^\w\s]", text.lower())


@dataclass
class Vocab:
    stoi: dict[str, int]
    itos: dict[int, str]
    pad_id: int
    unk_id: int
    bos_id: int
    eos_id: int

    @property
    def size(self) -> int:
        return len(self.stoi)

    def encode(self, tokens: list[str], add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        for tok in tokens:
            ids.append(self.stoi.get(tok, self.unk_id))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> list[str]:
        special_ids = {self.pad_id, self.unk_id, self.bos_id, self.eos_id}
        out = []
        for i in ids:
            if skip_special and i in special_ids:
                continue
            out.append(self.itos.get(i, "<unk>"))
        return out


def build_vocab(tokens: Iterable[str], min_freq: int = 1) -> Vocab:
    """
    Build vocab with special tokens and min frequency cutoff.
    Rare tokens go to <unk>.
    """
    counts = Counter(tokens)

    # Special tokens first, fixed ids
    stoi: dict[str, int] = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
    next_id = len(stoi)

    # Add tokens by frequency (then alpha for stability)
    items = [(tok, c) for tok, c in counts.items() if c >= min_freq and tok not in stoi]
    items.sort(key=lambda x: (-x[1], x[0]))

    for tok, _ in items:
        stoi[tok] = next_id
        next_id += 1

    itos = {i: t for t, i in stoi.items()}

    return Vocab(
        stoi=stoi,
        itos=itos,
        pad_id=stoi["<pad>"],
        unk_id=stoi["<unk>"],
        bos_id=stoi["<bos>"],
        eos_id=stoi["<eos>"],
    )


def load_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def prepare_corpus(
    text: str,
    min_freq: int = 1,
    add_bos_eos: bool = False,
) -> tuple[list[int], Vocab]:
    """
    Tokenize text -> build vocab -> encode into ids.
    If add_bos_eos=True, inserts <bos> and <eos> around each line (sentence-ish).
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    tokenized_lines = [tokenize(ln) for ln in lines]

    all_tokens = [tok for line in tokenized_lines for tok in line]
    vocab = build_vocab(all_tokens, min_freq=min_freq)

    ids: list[int] = []
    if add_bos_eos:
        for line_tokens in tokenized_lines:
            ids.extend(vocab.encode(line_tokens, add_bos=True, add_eos=True))
    else:
        ids = vocab.encode([tok for line in tokenized_lines for tok in line])

    return ids, vocab


def iter_next_word_batches(
    data_ids: list[int],
    seq_len: int,
    stride: int | None = None,
):
    """
    Yields (inputs, targets) where:
      inputs  = data[t : t+seq_len]
      targets = data[t+1 : t+seq_len+1]

    stride defaults to seq_len (non-overlapping chunks).
    """
    if stride is None:
        stride = seq_len
    n = len(data_ids)

    t = 0
    while t + seq_len + 1 <= n:
        inputs = data_ids[t : t + seq_len]
        targets = data_ids[t + 1 : t + seq_len + 1]
        yield inputs, targets
        t += stride
