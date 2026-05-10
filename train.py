"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional

import wandb
from tqdm import tqdm

from model import Transformer, make_src_mask, make_tgt_mask


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need".

    Smoothed target distribution:
        y_smooth[correct] = 1 - eps
        y_smooth[other]   = eps / (vocab_size - 1)
        y_smooth[pad]     = 0  (excluded from loss)

    Args:
        vocab_size (int)  : Number of output classes.
        pad_idx    (int)  : Index of <pad> token — receives 0 probability.
        smoothing  (float): Smoothing factor ε (default 0.1).
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : shape [batch * tgt_len, vocab_size]  (raw model output)
            target : shape [batch * tgt_len]              (gold token indices)

        Returns:
            Scalar loss value (mean over non-pad tokens).
        """
        log_probs = F.log_softmax(logits, dim=-1)   # [N, vocab_size]

        # Build smoothed target distribution
        smooth_targets = torch.full_like(log_probs, self.smoothing / (self.vocab_size - 1))
        smooth_targets.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        smooth_targets[:, self.pad_idx] = 0.0   # pad gets zero probability mass

        # KL divergence: sum over vocab, mean over non-pad positions
        non_pad_mask = (target != self.pad_idx)
        loss = -(smooth_targets * log_probs).sum(dim=-1)   # [N]
        loss = loss[non_pad_mask].mean()
        return loss


# ══════════════════════════════════════════════════════════════════════
#   TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """
    Run one epoch of training or evaluation.

    Args:
        data_iter  : DataLoader yielding (src, tgt) batches of token indices.
        model      : Transformer instance.
        loss_fn    : LabelSmoothingLoss (or any nn.Module loss).
        optimizer  : Optimizer (None during eval).
        scheduler  : NoamScheduler instance (None during eval).
        epoch_num  : Current epoch index (for logging).
        is_train   : If True, perform backward pass and scheduler step.
        device     : 'cpu' or 'cuda'.

    Returns:
        avg_loss : Average loss over the epoch (float).
    """
    model.train() if is_train else model.eval()

    total_loss  = 0.0
    total_tokens = 0
    pad_idx = model.pad_idx

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in tqdm(data_iter, desc=f"{'Train' if is_train else 'Val'} epoch {epoch_num}"):
            src, tgt = batch
            src = src.to(device)   # [batch, src_len]
            tgt = tgt.to(device)   # [batch, tgt_len]

            # Teacher-forcing: decoder input is tgt[:-1], target is tgt[1:]
            tgt_input  = tgt[:, :-1]
            tgt_target = tgt[:, 1:]

            src_mask = make_src_mask(src, pad_idx)
            tgt_mask = make_tgt_mask(tgt_input, pad_idx)

            logits = model(src, tgt_input, src_mask, tgt_mask)
            # [batch, tgt_len-1, vocab] → [batch*(tgt_len-1), vocab]
            logits_flat  = logits.contiguous().view(-1, logits.size(-1))
            targets_flat = tgt_target.contiguous().view(-1)

            loss = loss_fn(logits_flat, targets_flat)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            n_tokens = (tgt_target != pad_idx).sum().item()
            total_loss   += loss.item() * n_tokens
            total_tokens += n_tokens

    avg_loss = total_loss / max(total_tokens, 1)

    log_key = "train_loss" if is_train else "val_loss"
    wandb.log({log_key: avg_loss, "epoch": epoch_num})

    return avg_loss


# ══════════════════════════════════════════════════════════════════════
#   GREEDY DECODING
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    Args:
        model        : Trained Transformer.
        src          : Source token indices, shape [1, src_len].
        src_mask     : shape [1, 1, 1, src_len].
        max_len      : Maximum number of tokens to generate.
        start_symbol : Vocabulary index of <sos>.
        end_symbol   : Vocabulary index of <eos>.
        device       : 'cpu' or 'cuda'.

    Returns:
        ys : Generated token indices, shape [1, out_len].
             Includes start_symbol; stops at (and includes) end_symbol
             or when max_len is reached.
    """
    model.eval()
    pad_idx = model.pad_idx

    with torch.no_grad():
        memory = model.encode(src, src_mask)   # [1, src_len, d_model]
        ys = torch.full((1, 1), start_symbol, dtype=torch.long, device=device)

        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys, pad_idx)
            logits   = model.decode(memory, src_mask, ys, tgt_mask)  # [1, cur_len, vocab]
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [1, 1]
            ys = torch.cat([ys, next_tok], dim=1)
            if next_tok.item() == end_symbol:
                break

    return ys


# ══════════════════════════════════════════════════════════════════════
#   BLEU EVALUATION
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.

    Args:
        model           : Trained Transformer (in eval mode).
        test_dataloader : DataLoader over the test split.
                          Each batch yields (src, tgt) token-index tensors.
        tgt_vocab       : Vocabulary object with idx_to_token mapping.
                          Must support  tgt_vocab.itos[idx]  or
                          tgt_vocab.lookup_token(idx).
        device          : 'cpu' or 'cuda'.
        max_len         : Max decode length per sentence.

    Returns:
        bleu_score : Corpus-level BLEU (float, range 0–100).
    """
    model.eval()
    pad_idx = model.pad_idx
    sos_idx = tgt_vocab.stoi.get('<sos>', 2)
    eos_idx = tgt_vocab.stoi.get('<eos>', 3)

    predictions = []
    references  = []

    with torch.no_grad():
        for src, tgt in tqdm(test_dataloader, desc="BLEU eval"):
            src = src.to(device)
            tgt = tgt.to(device)

            for i in range(src.size(0)):
                src_i    = src[i].unsqueeze(0)                    # [1, src_len]
                src_mask = make_src_mask(src_i, pad_idx)

                out = greedy_decode(model, src_i, src_mask, max_len,
                                    start_symbol=sos_idx, end_symbol=eos_idx, device=device)
                out_ids = out[0].tolist()

                pred_words = []
                for t in out_ids:
                    if t == eos_idx:
                        break
                    if t not in (sos_idx, pad_idx):
                        pred_words.append(tgt_vocab.lookup_token(t))

                ref_ids = tgt[i].tolist()
                ref_words = []
                for t in ref_ids:
                    if t == eos_idx:
                        break
                    if t not in (sos_idx, pad_idx):
                        ref_words.append(tgt_vocab.lookup_token(t))

                predictions.append(" ".join(pred_words))
                references.append([" ".join(ref_words)])

    # Corpus-level BLEU via sacrebleu (returns 0–100)
    try:
        import sacrebleu as sb
        bleu = sb.corpus_bleu(predictions, list(zip(*references)))
        return bleu.score
    except ImportError:
        pass

    # Fallback: evaluate library
    try:
        import evaluate
        metric = evaluate.load("sacrebleu")
        for pred, refs in zip(predictions, references):
            metric.add(prediction=pred, references=refs)
        return metric.compute()["score"]
    except Exception:
        pass

    # Last-resort: NLTK sentence BLEU averaged to corpus BLEU
    from nltk.translate.bleu_score import corpus_bleu as nltk_corpus_bleu, SmoothingFunction
    tokenized_refs  = [[r[0].split() for r in refs] for refs in references]
    tokenized_preds = [p.split() for p in predictions]
    score = nltk_corpus_bleu(
        tokenized_refs, tokenized_preds,
        smoothing_function=SmoothingFunction().method1,
    )
    return score * 100.0


# ══════════════════════════════════════════════════════════════════════
# ❺  CHECKPOINT UTILITIES  (autograder loads your model from disk)
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimiser + scheduler state to disk.

    Saves a dict with keys:
        'epoch', 'model_state_dict', 'optimizer_state_dict',
        'scheduler_state_dict', 'model_config'
    """
    src_embed = model.src_embed
    tgt_embed = model.tgt_embed

    model_config = {
        'src_vocab_size': src_embed.num_embeddings,
        'tgt_vocab_size': tgt_embed.num_embeddings,
        'd_model':   model.d_model,
        'N':         len(model.encoder.layers),
        'num_heads': model.encoder.layers[0].self_attn.num_heads,
        'd_ff':      model.encoder.layers[0].ffn.linear1.out_features,
        'dropout':   model.encoder.layers[0].dropout.p,
        'pad_idx':   model.pad_idx,
    }

    torch.save({
        'epoch':                epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'model_config':         model_config,
        # Save vocab objects so Transformer() can restore infer() capability
        'src_vocab':            getattr(model, 'src_vocab', None),
        'tgt_vocab':            getattr(model, 'tgt_vocab', None),
    }, path)


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) state from disk.

    Returns:
        epoch : The epoch at which the checkpoint was saved (int).
    """
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint.get('epoch', 0)


# ══════════════════════════════════════════════════════════════════════
#   EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment() -> None:
    """
    Set up and run the full training experiment with W&B logging.
    """
    from torch.utils.data import DataLoader
    from torch.nn.utils.rnn import pad_sequence
    from dataset import Multi30kDataset
    from lr_scheduler import NoamScheduler

    # ── Hyperparameters ──────────────────────────────────────────────
    config = dict(
        d_model      = 256,
        N            = 3,
        num_heads    = 8,
        d_ff         = 512,
        dropout      = 0.1,
        warmup_steps = 4000,
        num_epochs   = 20,
        batch_size   = 128,
        max_len      = 100,
        label_smooth = 0.1,
        device       = "cuda" if torch.cuda.is_available() else "cpu",
    )

    wandb.init(project="da6401-a3", config=config)
    cfg = wandb.config
    device = cfg.device

    # ── Dataset & vocabularies ───────────────────────────────────────
    train_ds = Multi30kDataset(split='train')
    src_vocab, tgt_vocab = train_ds.build_vocab(min_freq=2)
    train_ds.process_data()

    val_ds = Multi30kDataset(split='validation', src_vocab=src_vocab, tgt_vocab=tgt_vocab)
    val_ds.process_data()

    test_ds = Multi30kDataset(split='test', src_vocab=src_vocab, tgt_vocab=tgt_vocab)
    test_ds.process_data()

    pad_idx = Multi30kDataset.PAD_IDX

    def collate_fn(batch):
        src_seqs, tgt_seqs = zip(*batch)
        src_pad = pad_sequence([torch.tensor(s) for s in src_seqs],
                               batch_first=True, padding_value=pad_idx)
        tgt_pad = pad_sequence([torch.tensor(t) for t in tgt_seqs],
                               batch_first=True, padding_value=pad_idx)
        return src_pad, tgt_pad

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    # ── Model ─────────────────────────────────────────────────────────
    model = Transformer(
        src_vocab_size = len(src_vocab),
        tgt_vocab_size = len(tgt_vocab),
        d_model   = cfg.d_model,
        N         = cfg.N,
        num_heads = cfg.num_heads,
        d_ff      = cfg.d_ff,
        dropout   = cfg.dropout,
        pad_idx   = pad_idx,
    ).to(device)

    model.src_vocab = src_vocab
    model.tgt_vocab = tgt_vocab

    # ── Optimizer, Scheduler, Loss ────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = NoamScheduler(optimizer, d_model=cfg.d_model, warmup_steps=cfg.warmup_steps)
    loss_fn   = LabelSmoothingLoss(len(tgt_vocab), pad_idx, smoothing=cfg.label_smooth)

    best_val_loss = float('inf')
    best_ckpt = "best_checkpoint.pt"

    # ── Training loop ─────────────────────────────────────────────────
    for epoch in range(cfg.num_epochs):
        train_loss = run_epoch(train_loader, model, loss_fn, optimizer, scheduler,
                               epoch_num=epoch, is_train=True,  device=device)
        val_loss   = run_epoch(val_loader,   model, loss_fn, None,      None,
                               epoch_num=epoch, is_train=False, device=device)

        print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        save_checkpoint(model, optimizer, scheduler, epoch, path=f"checkpoint_epoch{epoch}.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, path=best_ckpt)

    # ── Final BLEU on test set ─────────────────────────────────────────
    load_checkpoint(best_ckpt, model)
    bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device, max_len=cfg.max_len)
    wandb.log({'test_bleu': bleu})
    print(f"Test BLEU: {bleu:.2f}")
    wandb.finish()


if __name__ == "__main__":
    run_training_experiment()
