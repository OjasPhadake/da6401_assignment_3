"""
experiments.py — W&B Experiment Runner for DA6401 Assignment 3

Usage:
    python experiments.py 2.1   # Noam Scheduler vs Fixed LR
    python experiments.py 2.2   # Scaling factor 1/sqrt(d_k) ablation
    python experiments.py 2.3   # Attention head visualisation (uses best_checkpoint.pt)
    python experiments.py 2.4   # Sinusoidal vs Learned positional encoding
    python experiments.py 2.5   # Label smoothing eps=0.1 vs eps=0.0
    python experiments.py all   # Run all experiments sequentially
"""

import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import wandb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from dataset import Multi30kDataset
from model import Transformer, make_src_mask, make_tgt_mask
from train import LabelSmoothingLoss, beam_search_decode, evaluate_bleu
from lr_scheduler import NoamScheduler

# ── Constants ─────────────────────────────────────────────────────────

WANDB_PROJECT = "da6401-a3-experiments"
PAD_IDX       = Multi30kDataset.PAD_IDX
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

BASE_CFG = dict(
    d_model=256, N=3, num_heads=8, d_ff=512,
    dropout=0.1, warmup_steps=4000, batch_size=128, label_smooth=0.1,
)

# ── Shared helpers ─────────────────────────────────────────────────────

def load_data(batch_size: int = 128):
    """Return (train_loader, val_loader, test_loader, src_vocab, tgt_vocab)."""
    print("Loading Multi30k …")
    train_ds = Multi30kDataset(split='train')
    src_vocab, tgt_vocab = train_ds.build_vocab(min_freq=2)
    train_ds.process_data()

    val_ds  = Multi30kDataset(split='validation', src_vocab=src_vocab, tgt_vocab=tgt_vocab)
    val_ds.process_data()
    test_ds = Multi30kDataset(split='test',       src_vocab=src_vocab, tgt_vocab=tgt_vocab)
    test_ds.process_data()

    def collate(batch):
        srcs, tgts = zip(*batch)
        s = pad_sequence([torch.tensor(x) for x in srcs], batch_first=True, padding_value=PAD_IDX)
        t = pad_sequence([torch.tensor(x) for x in tgts], batch_first=True, padding_value=PAD_IDX)
        return s, t

    kw = dict(collate_fn=collate, num_workers=0)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **kw),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **kw),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **kw),
        src_vocab, tgt_vocab,
    )


def make_model(src_vocab, tgt_vocab, cfg, **kwargs) -> Transformer:
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=cfg['d_model'], N=cfg['N'],
        num_heads=cfg['num_heads'], d_ff=cfg['d_ff'],
        dropout=cfg['dropout'], pad_idx=PAD_IDX,
        **kwargs,
    ).to(DEVICE)
    model.src_vocab = src_vocab
    model.tgt_vocab = tgt_vocab
    return model


def train_one_epoch(
    model, loader, loss_fn, optimizer, scheduler,
    grad_watch=None,        # list of (name, param) — log grad norm each step
    log_confidence=False,   # log softmax(correct token) to wandb each step
    global_step=0,
    grad_log_limit=None,    # stop logging grad norms after this many steps
):
    """Train for one epoch. Returns (avg_loss, global_step)."""
    model.train()
    total_loss = total_tokens = 0
    pad_idx = model.pad_idx

    for src, tgt in tqdm(loader, leave=False, desc="train"):
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        tgt_in, tgt_tgt = tgt[:, :-1], tgt[:, 1:]

        logits = model(src, tgt_in, make_src_mask(src, pad_idx), make_tgt_mask(tgt_in, pad_idx))
        flat_logits  = logits.contiguous().view(-1, logits.size(-1))
        flat_targets = tgt_tgt.contiguous().view(-1)

        loss = loss_fn(flat_logits, flat_targets)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # — gradient norm logging (first grad_log_limit steps) —
        if grad_watch and (grad_log_limit is None or global_step < grad_log_limit):
            log_dict = {"step": global_step}
            for name, param in grad_watch:
                if param.grad is not None:
                    log_dict[f"grad_norm/{name}"] = param.grad.norm().item()
            wandb.log(log_dict)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        n_tok = (tgt_tgt != pad_idx).sum().item()
        total_loss   += loss.item() * n_tok
        total_tokens += n_tok

        # — prediction confidence logging —
        if log_confidence:
            with torch.no_grad():
                probs = F.softmax(flat_logits.detach(), dim=-1)
                non_pad = flat_targets != pad_idx
                correct_p = probs.gather(1, flat_targets.unsqueeze(1)).squeeze(1)
                wandb.log({"train_confidence": correct_p[non_pad].mean().item(),
                           "step": global_step})

        global_step += 1

    return total_loss / max(total_tokens, 1), global_step


@torch.no_grad()
def eval_one_epoch(model, loader, loss_fn):
    model.eval()
    total_loss = total_tokens = 0
    pad_idx = model.pad_idx
    for src, tgt in tqdm(loader, leave=False, desc="val"):
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        tgt_in, tgt_tgt = tgt[:, :-1], tgt[:, 1:]
        logits = model(src, tgt_in, make_src_mask(src, pad_idx), make_tgt_mask(tgt_in, pad_idx))
        loss   = loss_fn(logits.contiguous().view(-1, logits.size(-1)),
                         tgt_tgt.contiguous().view(-1))
        n_tok         = (tgt_tgt != pad_idx).sum().item()
        total_loss   += loss.item() * n_tok
        total_tokens += n_tok
    return total_loss / max(total_tokens, 1)


def run_training(run_name, group, cfg, num_epochs, src_vocab, tgt_vocab,
                 train_loader, val_loader, val_loader_bleu=None,
                 model_kwargs=None, fixed_lr=None,
                 grad_watch=None, grad_log_limit=None,
                 log_confidence=False, label_smooth=None,
                 extra_config=None):
    """
    Generic training loop for one experimental condition.
    Returns the trained model.
    """
    model_kwargs  = model_kwargs  or {}
    label_smooth  = label_smooth  if label_smooth is not None else cfg['label_smooth']
    extra_config  = extra_config  or {}

    wandb.init(
        project=WANDB_PROJECT,
        name=run_name,
        group=group,
        config={**cfg, "num_epochs": num_epochs, "label_smooth": label_smooth, **extra_config},
    )

    model     = make_model(src_vocab, tgt_vocab, cfg, **model_kwargs)
    lr        = 1.0 if fixed_lr is None else fixed_lr
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = (NoamScheduler(optimizer, d_model=cfg['d_model'], warmup_steps=cfg['warmup_steps'])
                 if fixed_lr is None else None)
    loss_fn   = LabelSmoothingLoss(len(tgt_vocab), PAD_IDX, smoothing=label_smooth)

    global_step = 0
    for epoch in range(num_epochs):
        train_loss, global_step = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scheduler,
            grad_watch=grad_watch, log_confidence=log_confidence,
            global_step=global_step, grad_log_limit=grad_log_limit,
        )
        val_loss = eval_one_epoch(model, val_loader, loss_fn)
        print(f"  [{run_name}] ep {epoch:02d} | train={train_loss:.4f} | val={val_loss:.4f} "
              f"| lr={optimizer.param_groups[0]['lr']:.2e}")
        wandb.log({"train_loss": train_loss, "val_loss": val_loss,
                   "lr": optimizer.param_groups[0]['lr'], "epoch": epoch})

    # Optional BLEU on validation set at end
    if val_loader_bleu is not None:
        bleu = evaluate_bleu(model, val_loader_bleu, tgt_vocab, device=DEVICE, max_len=80)
        print(f"  [{run_name}] val BLEU = {bleu:.2f}")
        wandb.log({"val_bleu": bleu})

    wandb.finish()
    return model


# ══════════════════════════════════════════════════════════════════════
#  2.1  Noam Scheduler vs Fixed Learning Rate
# ══════════════════════════════════════════════════════════════════════

def run_exp_2_1():
    print("\n" + "="*60)
    print("Experiment 2.1 — Noam Scheduler vs Fixed LR")
    print("="*60)
    NUM_EPOCHS = 15
    train_loader, val_loader, _, src_vocab, tgt_vocab = load_data(BASE_CFG['batch_size'])

    # Run 1: Noam scheduler (lr=1.0 base, warmup 4000 steps)
    run_training(
        run_name="noam_scheduler", group="2.1-lr-schedule",
        cfg=BASE_CFG, num_epochs=NUM_EPOCHS,
        src_vocab=src_vocab, tgt_vocab=tgt_vocab,
        train_loader=train_loader, val_loader=val_loader,
        extra_config={"scheduler": "noam"},
    )

    # Run 2: Fixed LR = 1e-4, no warmup
    run_training(
        run_name="fixed_lr_1e-4", group="2.1-lr-schedule",
        cfg=BASE_CFG, num_epochs=NUM_EPOCHS,
        src_vocab=src_vocab, tgt_vocab=tgt_vocab,
        train_loader=train_loader, val_loader=val_loader,
        fixed_lr=1e-4,
        extra_config={"scheduler": "fixed_1e-4"},
    )


# ══════════════════════════════════════════════════════════════════════
#  2.2  Scaling Factor 1/sqrt(d_k) Ablation — five panels
# ══════════════════════════════════════════════════════════════════════

def _run_2_2_single(
    run_name: str,
    group: str,
    use_scale: bool,
    cfg: dict,
    src_vocab,
    tgt_vocab,
    train_loader,
    val_loader,
    num_epochs: int = 10,
    grad_log_limit: int = 1000,
    warmup_steps: int = None,
    fixed_lr: float = None,
):
    """
    Train one variant and log all five panels:
      P1 grad_norm/enc{i}_{Wq|Wk}   — W_q / W_k gradient norms
      P2 attn_logit/{mean|std|max}   — raw QKᵀ statistics
      P3 attn_entropy                — Shannon entropy of attention distribution
      (P4/P5 distinguished by cfg and warmup_steps at the call-site)
    Per-step metrics are logged for the first `grad_log_limit` steps only.
    """
    warmup_steps = warmup_steps if warmup_steps is not None else cfg['warmup_steps']

    wandb.init(
        project=WANDB_PROJECT, name=run_name, group=group,
        config={**cfg, "use_scale": use_scale, "num_epochs": num_epochs,
                "warmup_steps": warmup_steps, "fixed_lr": fixed_lr,
                "d_k": cfg['d_model'] // cfg['num_heads']},
    )

    model     = make_model(src_vocab, tgt_vocab, cfg, use_scale=use_scale)
    lr        = fixed_lr if fixed_lr is not None else 1.0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = (None if fixed_lr is not None else
                 NoamScheduler(optimizer, d_model=cfg['d_model'], warmup_steps=warmup_steps))
    loss_fn   = LabelSmoothingLoss(len(tgt_vocab), PAD_IDX, smoothing=cfg['label_smooth'])

    pad_idx       = model.pad_idx
    last_enc_attn = model.encoder.layers[-1].self_attn   # reference for attention metrics

    # All W_q / W_k params across encoder layers (Panel 1)
    grad_params = []
    for i, layer in enumerate(model.encoder.layers):
        grad_params.append((f"enc{i}_Wq", layer.self_attn.W_q.weight))
        grad_params.append((f"enc{i}_Wk", layer.self_attn.W_k.weight))

    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = total_tokens = 0

        for src, tgt in tqdm(train_loader, leave=False, desc=run_name):
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            tgt_in, tgt_tgt = tgt[:, :-1], tgt[:, 1:]

            logits = model(src, tgt_in,
                           make_src_mask(src, pad_idx),
                           make_tgt_mask(tgt_in, pad_idx))
            flat_logits  = logits.contiguous().view(-1, logits.size(-1))
            flat_targets = tgt_tgt.contiguous().view(-1)
            loss = loss_fn(flat_logits, flat_targets)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # ── Per-step diagnostics (first grad_log_limit steps) ──────
            if global_step < grad_log_limit:
                log = {"step": global_step}

                # Panel 1 — Gradient norms of W_q / W_k
                for name, param in grad_params:
                    if param.grad is not None:
                        log[f"grad_norm/{name}"] = param.grad.norm().item()

                # Panel 2 — Raw QKᵀ logit magnitude (before any scaling)
                raw = last_enc_attn.last_raw_scores   # [B, H, Lq, Lk]
                if raw is not None:
                    raw_f = raw.float()
                    log["attn_logit/mean"]    = raw_f.mean().item()
                    log["attn_logit/std"]     = raw_f.std().item()
                    log["attn_logit/max_abs"] = raw_f.abs().max().item()

                # Panel 3 — Shannon entropy of attention weights
                w = last_enc_attn.last_attn_weights   # [B, H, Lq, Lk]
                if w is not None:
                    w_f = w.float().clamp(min=1e-9)
                    entropy = -(w_f * w_f.log()).sum(-1).mean().item()
                    log["attn_entropy"] = entropy

                wandb.log(log)

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            n_tok         = (tgt_tgt != pad_idx).sum().item()
            total_loss   += loss.item() * n_tok
            total_tokens += n_tok
            global_step  += 1

        val_loss  = eval_one_epoch(model, val_loader, loss_fn)
        avg_loss  = total_loss / max(total_tokens, 1)
        print(f"  [{run_name}] ep {epoch:02d} | train={avg_loss:.4f} | val={val_loss:.4f}")
        wandb.log({"train_loss": avg_loss, "val_loss": val_loss, "epoch": epoch})

    wandb.finish()


def run_exp_2_2():
    print("\n" + "="*60)
    print("Experiment 2.2 — Scaling Factor 1/sqrt(d_k)  (5 panels)")
    print("="*60)

    train_loader, val_loader, _, src_vocab, tgt_vocab = load_data(BASE_CFG['batch_size'])

    # ── Panels 1-3: base architecture, with vs without scale ──────────
    # Panels 1-3 are logged by both runs; W&B overlays them automatically.
    for use_scale, name in [(True, "with_scale"), (False, "no_scale")]:
        _run_2_2_single(name, "2.2-main", use_scale, BASE_CFG,
                        src_vocab, tgt_vocab, train_loader, val_loader,
                        num_epochs=10, grad_log_limit=1000)

    # ── Panel 4: Large d_k stress test (no_scale only) ────────────────
    # Keep d_model=256 but reduce num_heads → d_k grows: 32→64→128.
    # Without 1/√d_k the variance of QKᵀ scales with d_k; instability
    # worsens dramatically as d_k increases.
    for num_heads, label in [(4, "no_scale_dk64"), (2, "no_scale_dk128")]:
        cfg_stress = {**BASE_CFG, "num_heads": num_heads}
        _run_2_2_single(label, "2.2-stress-dk", False, cfg_stress,
                        src_vocab, tgt_vocab, train_loader, val_loader,
                        num_epochs=5, grad_log_limit=1000)

    # ── Panel 5: LR sensitivity (no_scale, short vs normal warmup) ────
    # Short warmup ramps up LR faster → LR peak is hit in first 500 steps
    # instead of 4000, magnifying gradient instability in the unscaled model.
    for warmup, label in [(500, "no_scale_warmup500"), (4000, "no_scale_warmup4000")]:
        _run_2_2_single(label, "2.2-lr-sensitivity", False, BASE_CFG,
                        src_vocab, tgt_vocab, train_loader, val_loader,
                        num_epochs=5, grad_log_limit=1000, warmup_steps=warmup)


# ══════════════════════════════════════════════════════════════════════
#  2.3  Attention Head Visualisation
# ══════════════════════════════════════════════════════════════════════

def _plot_attention_heads(attn: torch.Tensor, tokens: list, title: str):
    """
    Create a matplotlib figure with one subplot per attention head.

    attn  : [1, num_heads, src_len, src_len] (CPU tensor)
    tokens: list of source token strings (length == src_len)
    Returns a matplotlib Figure.
    """
    attn_np = attn[0].cpu().numpy()   # [heads, src_len, src_len]
    num_heads, L, _ = attn_np.shape
    ncols = 4
    nrows = math.ceil(num_heads / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
    axes = axes.flatten()

    for h in range(num_heads):
        ax = axes[h]
        im = ax.imshow(attn_np[h], aspect='auto', cmap='Blues', vmin=0.0, vmax=attn_np[h].max())
        ax.set_title(f"Head {h}", fontsize=9)
        ax.set_xticks(range(L))
        ax.set_yticks(range(L))
        ax.set_xticklabels(tokens, fontsize=6, rotation=45, ha='right')
        ax.set_yticklabels(tokens, fontsize=6)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for h in range(num_heads, len(axes)):
        axes[h].set_visible(False)

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    return fig


def _attention_rollout(model: Transformer, src: torch.Tensor) -> np.ndarray:
    """
    Compute Attention Rollout across all encoder layers.
    Returns a [src_len, src_len] numpy array.
    Reference: Abnar & Zuidema, "Quantifying Attention Flow", 2020.
    """
    src_mask = make_src_mask(src, model.pad_idx)
    model.eval()
    with torch.no_grad():
        model.encode(src, src_mask)

    rollout = None
    for layer in model.encoder.layers:
        A = layer.self_attn.last_attn_weights[0]   # [heads, L, L]
        A_mean = A.mean(dim=0).cpu().numpy()        # [L, L]  — average over heads
        I = np.eye(A_mean.shape[0])
        # Add residual, normalise rows
        A_hat = 0.5 * A_mean + 0.5 * I
        A_hat /= A_hat.sum(axis=-1, keepdims=True)
        rollout = A_hat if rollout is None else rollout @ A_hat

    return rollout   # [L, L]


def run_exp_2_3():
    print("\n" + "="*60)
    print("Experiment 2.3 — Attention Head Visualisation")
    print("="*60)

    # Load the trained model (no-arg construction → downloads checkpoint)
    print("Loading best_checkpoint.pt …")
    model = Transformer().to(DEVICE)
    model.eval()
    model._ensure_vocab()

    import spacy
    spacy_de = spacy.blank("de")

    # A few diverse test sentences
    sentences = [
        "ein mann sitzt auf einem stuhl .",
        "zwei kinder spielen auf dem spielplatz .",
        "eine frau läuft durch den park und lächelt .",
    ]

    wandb.init(project=WANDB_PROJECT, name="attention_viz", group="2.3-attention")

    for sent_idx, sentence in enumerate(sentences):
        tokens = [tok.text.lower() for tok in spacy_de.tokenizer(sentence)]
        sos, eos = model.src_vocab.stoi.get('<sos>', 2), model.src_vocab.stoi.get('<eos>', 3)
        ids = [sos] + model.src_vocab.lookup_indices(tokens) + [eos]
        display_tokens = ['<sos>'] + tokens + ['<eos>']

        src = torch.tensor(ids, dtype=torch.long, device=DEVICE).unsqueeze(0)  # [1, L]

        # — Per-head heatmap (last encoder layer) —
        attn = model.get_encoder_attention(src)   # [1, heads, L, L]
        fig_heads = _plot_attention_heads(
            attn, display_tokens,
            title=f"Encoder Last-Layer Attention: \"{sentence}\""
        )
        wandb.log({f"attn_heads/sentence_{sent_idx}": wandb.Image(fig_heads)})
        plt.close(fig_heads)

        # — Attention Rollout —
        rollout = _attention_rollout(model, src)   # [L, L]
        fig_roll, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(rollout, aspect='auto', cmap='Purples')
        ax.set_xticks(range(len(display_tokens)))
        ax.set_yticks(range(len(display_tokens)))
        ax.set_xticklabels(display_tokens, fontsize=7, rotation=45, ha='right')
        ax.set_yticklabels(display_tokens, fontsize=7)
        plt.colorbar(im, ax=ax)
        ax.set_title(f"Attention Rollout: \"{sentence}\"", fontsize=9)
        plt.tight_layout()
        wandb.log({f"attn_rollout/sentence_{sent_idx}": wandb.Image(fig_roll)})
        plt.close(fig_roll)

        print(f"  Logged attention maps for: {sentence}")

    wandb.finish()


# ══════════════════════════════════════════════════════════════════════
#  2.4  Sinusoidal vs Learned Positional Encoding
# ══════════════════════════════════════════════════════════════════════

def run_exp_2_4():
    print("\n" + "="*60)
    print("Experiment 2.4 — Sinusoidal vs Learned Positional Encoding")
    print("="*60)
    NUM_EPOCHS = 15
    train_loader, val_loader, _, src_vocab, tgt_vocab = load_data(BASE_CFG['batch_size'])

    for pe_type, run_name in [("sinusoidal", "sinusoidal_pe"), ("learned", "learned_pe")]:
        run_training(
            run_name=run_name, group="2.4-positional-encoding",
            cfg=BASE_CFG, num_epochs=NUM_EPOCHS,
            src_vocab=src_vocab, tgt_vocab=tgt_vocab,
            train_loader=train_loader, val_loader=val_loader,
            val_loader_bleu=val_loader,
            model_kwargs={"pe_type": pe_type},
            extra_config={"pe_type": pe_type},
        )


# ══════════════════════════════════════════════════════════════════════
#  2.5  Label Smoothing Ablation (eps=0.1 vs eps=0.0)
# ══════════════════════════════════════════════════════════════════════

def run_exp_2_5():
    print("\n" + "="*60)
    print("Experiment 2.5 — Label Smoothing Ablation")
    print("="*60)
    NUM_EPOCHS = 15
    train_loader, val_loader, _, src_vocab, tgt_vocab = load_data(BASE_CFG['batch_size'])

    for eps, run_name in [(0.1, "label_smooth_0.1"), (0.0, "label_smooth_0.0")]:
        wandb.init(
            project=WANDB_PROJECT, name=run_name, group="2.5-label-smoothing",
            config={**BASE_CFG, "label_smooth": eps, "num_epochs": NUM_EPOCHS},
        )

        model     = make_model(src_vocab, tgt_vocab, BASE_CFG)
        optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
        scheduler = NoamScheduler(optimizer, d_model=BASE_CFG['d_model'],
                                  warmup_steps=BASE_CFG['warmup_steps'])
        loss_fn   = LabelSmoothingLoss(len(tgt_vocab), PAD_IDX, smoothing=eps)

        global_step = 0
        for epoch in range(NUM_EPOCHS):
            train_loss, global_step = train_one_epoch(
                model, train_loader, loss_fn, optimizer, scheduler,
                log_confidence=True, global_step=global_step,
            )
            val_loss = eval_one_epoch(model, val_loader, loss_fn)
            # Also log val confidence
            val_conf = _val_confidence(model, val_loader, loss_fn)
            print(f"  [{run_name}] ep {epoch:02d} | train={train_loss:.4f} | val={val_loss:.4f} "
                  f"| val_conf={val_conf:.4f}")
            wandb.log({"train_loss": train_loss, "val_loss": val_loss,
                       "val_confidence": val_conf, "epoch": epoch})

        wandb.finish()


@torch.no_grad()
def _val_confidence(model, loader, loss_fn):
    """Mean softmax probability of the correct token over the validation set."""
    model.eval()
    pad_idx = model.pad_idx
    total_conf, total_tokens = 0.0, 0

    for src, tgt in loader:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        tgt_in, tgt_tgt = tgt[:, :-1], tgt[:, 1:]
        logits = model(src, tgt_in, make_src_mask(src, pad_idx), make_tgt_mask(tgt_in, pad_idx))
        flat_logits  = logits.contiguous().view(-1, logits.size(-1))
        flat_targets = tgt_tgt.contiguous().view(-1)
        probs        = F.softmax(flat_logits, dim=-1)
        non_pad      = flat_targets != pad_idx
        correct_p    = probs.gather(1, flat_targets.unsqueeze(1)).squeeze(1)
        total_conf   += correct_p[non_pad].sum().item()
        total_tokens += non_pad.sum().item()

    return total_conf / max(total_tokens, 1)


# ══════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════

EXPERIMENTS = {
    "2.1": run_exp_2_1,
    "2.2": run_exp_2_2,
    "2.3": run_exp_2_3,
    "2.4": run_exp_2_4,
    "2.5": run_exp_2_5,
}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    key = sys.argv[1].strip()
    if key == "all":
        for fn in EXPERIMENTS.values():
            fn()
    elif key in EXPERIMENTS:
        EXPERIMENTS[key]()
    else:
        print(f"Unknown experiment '{key}'. Choose from: {list(EXPERIMENTS)} or 'all'.")
        sys.exit(1)
