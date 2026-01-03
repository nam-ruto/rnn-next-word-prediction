# src/model.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    """
    logits: (V, 1)
    returns: (V, 1) probabilities
    """
    z = logits - np.max(logits)
    exp = np.exp(z)
    return exp / (np.sum(exp) + 1e-12)


@dataclass
class RNNConfig:
    vocab_size: int
    emb_size: int = 64
    hidden_size: int = 128
    seed: int = 42
    grad_clip: float = 5.0


class WordRNNLM:
    """
    Word-level RNN language model:
      x_t (word id) -> embedding e_t -> hidden h_t -> vocab logits -> softmax
    Trained with next-word prediction and BPTT.
    """

    def __init__(self, cfg: RNNConfig):
        self.cfg = cfg
        V, E, H = cfg.vocab_size, cfg.emb_size, cfg.hidden_size
        rng = np.random.default_rng(cfg.seed)

        # Parameters
        # Embedding matrix maps one-hot word to embedding: e = Wemb @ onehot
        self.Wemb = rng.normal(0, 0.01, (E, V))

        # RNN parameters
        self.Wxh = rng.normal(0, 0.01, (H, E))
        self.Whh = rng.normal(0, 0.01, (H, H))
        self.bh = np.zeros((H, 1))

        # Output projection
        self.Why = rng.normal(0, 0.01, (V, H))
        self.by = np.zeros((V, 1))

        # Optimizer state (Adagrad)
        self.mWemb = np.zeros_like(self.Wemb)
        self.mWxh = np.zeros_like(self.Wxh)
        self.mWhh = np.zeros_like(self.Whh)
        self.mbh = np.zeros_like(self.bh)
        self.mWhy = np.zeros_like(self.Why)
        self.mby = np.zeros_like(self.by)

    # ---------- Core methods ----------
    def init_hidden(self) -> np.ndarray:
        """Return zero initial hidden state: (H, 1)."""
        return np.zeros((self.cfg.hidden_size, 1))

    def forward(
        self,
        inputs: list[int],
        targets: list[int],
        hprev: np.ndarray,
    ) -> tuple[float, np.ndarray, dict]:
        """
        inputs/targets: length T sequences of word ids
        hprev: (H,1) hidden state from previous chunk
        returns: (loss, h_last, cache)
        """
        assert len(inputs) == len(targets), "inputs and targets must match length"
        T = len(inputs)
        V = self.cfg.vocab_size

        # Cache for backprop
        xs: dict[int, np.ndarray] = {}
        es: dict[int, np.ndarray] = {}
        hs: dict[int, np.ndarray] = {-1: hprev}
        ps: dict[int, np.ndarray] = {}

        loss = 0.0
        for t in range(T):
            # One-hot (V,1)
            x = np.zeros((V, 1))
            x[inputs[t]] = 1.0
            xs[t] = x

            # Embedding (E,1)
            e = self.Wemb @ x
            es[t] = e

            # Hidden (H,1)
            h = np.tanh(self.Wxh @ e + self.Whh @ hs[t - 1] + self.bh)
            hs[t] = h

            # Vocab distribution
            logits = self.Why @ h + self.by
            p = softmax(logits)
            ps[t] = p

            loss += -np.log(p[targets[t], 0] + 1e-12)

        cache = {"xs": xs, "es": es, "hs": hs, "ps": ps, "inputs": inputs, "targets": targets}
        return loss, hs[T - 1], cache

    def backward(self, cache: dict) -> dict[str, np.ndarray]:
        """
        Backprop Through Time (BPTT) over one sequence chunk.
        Returns gradients dictionary.
        """
        xs: dict[int, np.ndarray] = cache["xs"]
        es: dict[int, np.ndarray] = cache["es"]
        hs: dict[int, np.ndarray] = cache["hs"]
        ps: dict[int, np.ndarray] = cache["ps"]
        inputs: list[int] = cache["inputs"]
        targets: list[int] = cache["targets"]

        V, H = self.cfg.vocab_size, self.cfg.hidden_size
        T = len(inputs)

        dWemb = np.zeros_like(self.Wemb)
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dbh = np.zeros_like(self.bh)
        dWhy = np.zeros_like(self.Why)
        dby = np.zeros_like(self.by)

        dhnext = np.zeros((H, 1))

        for t in reversed(range(T)):
            # Gradient at output
            dy = ps[t].copy()
            dy[targets[t]] -= 1.0  # dL/dlogits

            dWhy += dy @ hs[t].T
            dby += dy

            # Backprop into hidden
            dh = self.Why.T @ dy + dhnext
            dtanh = (1.0 - hs[t] * hs[t]) * dh

            dbh += dtanh
            dWxh += dtanh @ es[t].T
            dWhh += dtanh @ hs[t - 1].T

            # Backprop into embedding
            de = self.Wxh.T @ dtanh          # (E,1)
            dWemb += de @ xs[t].T            # (E,V)

            dhnext = self.Whh.T @ dtanh

        # Gradient clipping
        clip = self.cfg.grad_clip
        for g in (dWemb, dWxh, dWhh, dbh, dWhy, dby):
            np.clip(g, -clip, clip, out=g)

        return {
            "Wemb": dWemb,
            "Wxh": dWxh,
            "Whh": dWhh,
            "bh": dbh,
            "Why": dWhy,
            "by": dby,
        }

    def step(self, grads: dict[str, np.ndarray], lr: float = 0.1, eps: float = 1e-8) -> None:
        """
        Adagrad update.
        """
        for (param_name, mem_name) in [
            ("Wemb", "mWemb"),
            ("Wxh", "mWxh"),
            ("Whh", "mWhh"),
            ("bh", "mbh"),
            ("Why", "mWhy"),
            ("by", "mby"),
        ]:
            param = getattr(self, param_name)
            mem = getattr(self, mem_name)
            g = grads[param_name]

            mem += g * g
            param -= lr * g / (np.sqrt(mem) + eps)

    # ---------- Inference / sampling ----------
    def predict_next(self, word_id: int, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        One-step forward for inference.
        Returns (probabilities (V,), new_hidden (H,1))
        """
        V = self.cfg.vocab_size
        x = np.zeros((V, 1))
        x[word_id] = 1.0

        e = self.Wemb @ x
        h = np.tanh(self.Wxh @ e + self.Whh @ h + self.bh)
        logits = self.Why @ h + self.by
        p = softmax(logits).reshape(-1)
        return p, h

    def generate(
        self,
        start_id: int,
        id_to_word: dict[int, str],
        length: int = 20,
        temperature: float = 1.0,
        top_k: int | None = None,
        seed: int | None = None,
    ) -> str:
        """
        Generate a word sequence.
        - temperature: >1.0 more random, <1.0 more conservative
        - top_k: if set, sample only from top-k words each step
        """
        rng = np.random.default_rng(seed)
        h = self.init_hidden()
        word_id = start_id
        out_words = [id_to_word[word_id]]

        for _ in range(length):
            p, h = self.predict_next(word_id, h)

            # Temperature
            if temperature != 1.0:
                logits = np.log(p + 1e-12) / temperature
                p = np.exp(logits - np.max(logits))
                p = p / (np.sum(p) + 1e-12)

            # Top-k filtering
            if top_k is not None and 1 <= top_k < len(p):
                idx = np.argpartition(p, -top_k)[-top_k:]
                p_top = p[idx]
                p_top = p_top / (np.sum(p_top) + 1e-12)
                word_id = int(rng.choice(idx, p=p_top))
            else:
                word_id = int(rng.choice(len(p), p=p))

            out_words.append(id_to_word[word_id])

        return " ".join(out_words)
