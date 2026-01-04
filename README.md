# RNN Word-level Next Word Prediction

A simple word-level Recurrent Neural Network (RNN) implementation for next-word prediction, built from scratch using only **NumPy**.

## Features

- **Word-level Language Modeling**: Predicts the next word in a sequence.
- **From Scratch**: RNN forward pass, Backpropagation Through Time (BPTT), and Adagrad optimizer implemented using only NumPy.
- **Data Preprocessing**: Custom tokenizer, vocabulary building with frequency cutoffs, and support for special tokens (`<pad>`, `<unk>`, `<bos>`, `<eos>`).
- **Flexible Inference**: Generate text with adjustable temperature and top-k sampling.
- **Modern Python Tooling**: Managed with [uv](https://github.com/astral-sh/uv).

## Installation

This project uses `uv` for dependency management.

1.  **Install uv** (if you haven't):
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

2.  **Sync dependencies**:
    ```bash
    uv sync
    ```

## Usage

### Training

To train the model on a text corpus (e.g., the provided `tiny_corpus.txt`):

```bash
uv run python src/train.py \
    --data experiments/tiny_corpus.txt \
    --steps 1000 \
    --lr 0.1 \
    --seq-len 8 \
    --outdir runs/my_run
```

**Key Arguments:**
- `--data`: Path to the text file.
- `--steps`: Number of training steps.
- `--seq-len`: Length of the sequence for each BPTT step.
- `--add-bos-eos`: Optional flag to wrap lines with `<bos>` and `<eos>` tokens.

### Text Generation (Sampling)

After training, you can generate text using the saved checkpoint:

```bash
uv run python src/sample.py \
    --run runs/my_run \
    --start "i" \
    --length 20 \
    --temperature 0.8 \
    --top-k 10
```

**Key Arguments:**
- `--run`: The directory where the model and vocab were saved.
- `--start`: The word to start the generation with.
- `--length`: Number of tokens to generate.
- `--temperature`: Controls randomness (higher = more random).
- `--top-k`: Limits sampling to the top K most likely tokens.

## Project Structure

```text
.
├── src/
│   ├── data.py    # Tokenization and Vocab utilities
│   ├── model.py   # RNN implementation and WordRNNLM class
│   ├── train.py   # Training script
│   └── sample.py  # Inference script
├── experiments/   # Example datasets (e.g., tiny_corpus.txt)
├── main.py        # Quick testing script
└── pyproject.toml # Project configuration and dependencies
```

## Implementation Details

- **Embedding Layer**: Maps word IDs to dense vectors.
- **RNN Layer**: Standard RNN with `tanh` activation.
- **Output Layer**: Softmax over the vocabulary.
- **Optimizer**: Adagrad to handle sparse updates and adapt learning rates.
- **Gradient Clipping**: Prevents exploding gradients during training.
