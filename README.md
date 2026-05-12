# DA6401 Assignment 3 — Transformer for Neural Machine Translation

**W&B Report:** [DA6401 Assignment 3 Report](https://wandb.ai/ch22b007-indian-institute-of-technology-madras/da6401-a3-experiments/reports/DA6401-Assignment-3-Report--VmlldzoxNjg0NDY3MA)

---

## Overview

Implements the Transformer architecture ("Attention Is All You Need") from scratch in PyTorch for German→English translation on the Multi30k dataset. Achieves **BLEU ≥ 35** using beam search decoding.

## Project Structure

```
da6401_assignment_3/
├── model.py          # Transformer architecture (encoder, decoder, multi-head attention)
├── train.py          # Training loop, beam search decode, BLEU evaluation, greedy decode
├── dataset.py        # Multi30k dataset loading, spaCy tokenization, vocab building
├── lr_scheduler.py   # Noam learning rate schedule
├── experiments.py    # W&B experiments 2.1–2.5
├── requirements.txt
└── ex.md             # Detailed implementation notes and debugging journal
```

## Model Architecture

| Hyperparameter | Value |
|---|---|
| `d_model` | 256 |
| `num_heads` | 8 |
| `num_layers` (enc+dec) | 3 each |
| `d_ff` | 512 |
| `dropout` | 0.1 |
| `max_len` | 256 |

- **Positional encoding:** Sinusoidal (default) or learned (`pe_type='learned'`)
- **Attention scaling:** Optional `1/sqrt(d_k)` (ablation in exp 2.2)
- **Decoding:** Beam search (beam_size=4, length_penalty α=0.6)
- **Loss:** Cross-entropy with optional label smoothing (ε=0.1)

## Quick Start

```bash
pip install -r requirements.txt
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm

# Train
python train.py

# Translate a sentence
python -c "from model import infer; print(infer('Ein Hund läuft im Park.'))"

# Run W&B experiments
python experiments.py 2.1   # Noam vs fixed LR
python experiments.py 2.2   # Scaled vs unscaled attention
python experiments.py 2.3   # Attention rollout & head specialization
python experiments.py 2.4   # Sinusoidal vs learned PE
python experiments.py 2.5   # Label smoothing ablation
```

## W&B Experiments

| Exp | Topic | Key finding |
|---|---|---|
| 2.1 | Noam LR vs fixed LR | Noam warms up smoothly; fixed LR diverges early |
| 2.2 | Scaled vs unscaled attention | Removing `1/sqrt(d_k)` causes logit explosion and entropy collapse |
| 2.3 | Attention rollout & head specialization | Heads differ in distance, entropy, and attended positions |
| 2.4 | Sinusoidal vs learned PE | Learned PE matches sinusoidal on short sequences |
| 2.5 | Label smoothing | ε=0.1 reduces overconfidence and improves validation BLEU |

## Key Files

**[model.py](model.py)** — `Transformer`, `EncoderLayer`, `DecoderLayer`, `MultiHeadAttention` (stores `last_attn_weights` and `last_raw_scores` after each forward pass), `LearnedPositionalEncoding`. Auto-downloads checkpoint from Google Drive on first run.

**[train.py](train.py)** — `train_model()`, `evaluate_bleu()`, `beam_search_decode()`, `greedy_decode()`, `_corpus_bleu()` (pure numpy BLEU with brevity penalty).

**[experiments.py](experiments.py)** — Five self-contained experiment runners. Each accepts the base model config, trains or evaluates, and logs all metrics/plots to W&B.
