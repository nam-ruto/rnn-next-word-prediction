# src/sample.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.model import RNNConfig, WordRNNLM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample from a trained word-level RNN LM (NumPy-only).")

    p.add_argument("--run", type=str, required=True, help="Run directory (e.g., runs/rnn_lm).")
    p.add_argument("--ckpt", type=str, default="checkpoint.npz", help="Checkpoint filename inside run dir.")
    p.add_argument("--start", type=str, default=None, help="Start word token (must be in vocab).")
    p.add_argument("--length", type=int, default=30, help="Number of tokens to generate.")
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--seed", type=int, default=123)

    return p.parse_args()


def load_vocab(vocab_path: Path) -> tuple[dict[str, int], dict[int, str]]:
    """
    Expects vocab.txt lines: "<id>\\t<lemma>"
    Written by train.py.
    """
    stoi: dict[str, int] = {}
    itos: dict[int, str] = {}

    for line in vocab_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        idx_str, tok = line.split("\t", 1)
        idx = int(idx_str)
        itos[idx] = tok
        stoi[tok] = idx

    return stoi, itos


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run)
    ckpt_path = run_dir / args.ckpt
    vocab_path = run_dir / "vocab.txt"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not vocab_path.exists():
        raise FileNotFoundError(f"Vocab not found: {vocab_path}")

    stoi, itos = load_vocab(vocab_path)

    # Load weights
    ckpt = np.load(ckpt_path)
    V = ckpt["by"].shape[0]
    E = ckpt["Wemb"].shape[0]
    H = ckpt["Whh"].shape[0]

    cfg = RNNConfig(vocab_size=V, emb_size=E, hidden_size=H, seed=args.seed)
    model = WordRNNLM(cfg)

    # Assign parameters
    model.Wemb = ckpt["Wemb"]
    model.Wxh = ckpt["Wxh"]
    model.Whh = ckpt["Whh"]
    model.bh = ckpt["bh"]
    model.Why = ckpt["Why"]
    model.by = ckpt["by"]

    rng = np.random.default_rng(args.seed)

    # Choose start token
    if args.start is None:
        start_id = int(rng.integers(0, V))
    else:
        if args.start not in stoi:
            raise ValueError(
                f"Start token '{args.start}' not in vocab. "
                f"Try one of: {', '.join(list(stoi.keys())[:30])} ..."
            )
        start_id = stoi[args.start]

    text = model.generate(
        start_id=start_id,
        id_to_word=itos,
        length=args.length,
        temperature=args.temperature,
        top_k=args.top_k if args.top_k > 0 else None,
        seed=args.seed,
    )

    print(text)


if __name__ == "__main__":
    main()
