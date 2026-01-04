# src/train.py
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.data import load_text_file, prepare_corpus, iter_next_word_batches
from src.model import RNNConfig, WordRNNLM


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a word-level RNN language model (NumPy-only).")

    # Data
    p.add_argument("--data", type=str, required=True, help="Path to a text file corpus.")
    p.add_argument("--min-freq", type=int, default=1, help="Min token frequency to keep in vocab.")
    p.add_argument("--add-bos-eos", action="store_true", help="Wrap each non-empty line with <bos>/<eos>.")

    # Model
    p.add_argument("--emb-size", type=int, default=64)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grad-clip", type=float, default=5.0)

    # Training
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--stride", type=int, default=16, help="Step size between chunks (<=seq-len gives overlap).")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--print-every", type=int, default=200)
    p.add_argument("--sample-len", type=int, default=25)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=30)

    # Output
    p.add_argument("--outdir", type=str, default="runs/rnn_lm", help="Directory to save logs/artifacts.")

    return p.parse_args()


def compute_perplexity(loss: float, num_tokens: int) -> float:
    # Perplexity = exp(avg negative log-likelihood)
    if num_tokens <= 0:
        return float("inf")
    return float(math.exp(loss / num_tokens))


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    text = load_text_file(args.data)
    data_ids, vocab = prepare_corpus(
        text=text,
        min_freq=args.min_freq,
        add_bos_eos=args.add_bos_eos,
    )

    cfg = RNNConfig(
        vocab_size=vocab.size,
        emb_size=args.emb_size,
        hidden_size=args.hidden_size,
        seed=args.seed,
        grad_clip=args.grad_clip,
    )
    model = WordRNNLM(cfg)

    # Prepare iterable batches (we'll cycle them)
    batches = list(iter_next_word_batches(data_ids, seq_len=args.seq_len, stride=args.stride))
    if not batches:
        raise ValueError("Corpus too small for the chosen seq_len. Reduce --seq-len or provide more text.")

    # Training state
    hprev = model.init_hidden()
    running_loss = 0.0
    running_tokens = 0

    # Log basics
    (outdir / "meta.txt").write_text(
        "\n".join(
            [
                f"data={args.data}",
                f"vocab_size={vocab.size}",
                f"min_freq={args.min_freq}",
                f"add_bos_eos={args.add_bos_eos}",
                f"emb_size={args.emb_size}",
                f"hidden_size={args.hidden_size}",
                f"seq_len={args.seq_len}",
                f"stride={args.stride}",
                f"lr={args.lr}",
                f"steps={args.steps}",
                f"seed={args.seed}",
            ]
        ),
        encoding="utf-8",
    )

    # Save vocab for later inspection
    (outdir / "vocab.txt").write_text(
        "\n".join([f"{i}\t{vocab.itos[i]}" for i in range(vocab.size)]),
        encoding="utf-8",
    )

    print(f"Loaded {len(data_ids)} tokens | vocab={vocab.size} | batches={len(batches)}")
    print(f"Output dir: {outdir}")

    progress = tqdm(range(1, args.steps + 1))
    for step in progress:
        inputs, targets = batches[(step - 1) % len(batches)]

        loss, hprev, cache = model.forward(inputs, targets, hprev)
        grads = model.backward(cache)
        model.step(grads, lr=args.lr)

        running_loss += loss
        running_tokens += len(targets)

        if step % args.print_every == 0:
            ppl = compute_perplexity(running_loss, running_tokens)

            # quick sample
            start_id = inputs[0]
            sample_text = model.generate(
                start_id=start_id,
                id_to_word=vocab.itos,
                length=args.sample_len,
                temperature=args.temperature,
                top_k=args.top_k if args.top_k > 0 else None,
                seed=args.seed + step,
            )

            msg = f"step={step} | avg_loss={running_loss/running_tokens:.4f} | ppl={ppl:.2f}"
            progress.set_description(msg)

            print("\n" + msg)
            print("sample:", sample_text)
            print("-" * 80)

            # reset running stats
            running_loss = 0.0
            running_tokens = 0

    # Save a minimal checkpoint (NumPy arrays)
    ckpt_path = outdir / "checkpoint.npz"
    np.savez(
        ckpt_path,
        Wemb=model.Wemb,
        Wxh=model.Wxh,
        Whh=model.Whh,
        bh=model.bh,
        Why=model.Why,
        by=model.by,
    )
    print(f"Saved checkpoint to: {ckpt_path}")


if __name__ == "__main__":
    main()
