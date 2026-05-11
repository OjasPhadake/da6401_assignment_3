"""
model.py — Transformer Architecture
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────┐
  │  scaled_dot_product_attention(Q, K, V, mask) → (out, weights)  │
  │  MultiHeadAttention.forward(q, k, v, mask)   → Tensor          │
  │  PositionalEncoding.forward(x)               → Tensor          │
  │  make_src_mask(src, pad_idx)                 → BoolTensor      │
  │  make_tgt_mask(tgt, pad_idx)                 → BoolTensor      │
  │  Transformer.encode(src, src_mask)           → Tensor          │
  │  Transformer.decode(memory,src_m,tgt,tgt_m)  → Tensor          │
  └─────────────────────────────────────────────────────────────────┘
"""

import math
import copy
import os
from typing import Optional, Tuple

try:
    import gdown as _gdown
except ImportError:
    _gdown = None

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
#   LIGHTWEIGHT VOCAB  (no dependency on dataset.py)
# ══════════════════════════════════════════════════════════════════════

class _Vocab:
    """Plain-Python vocabulary that can be pickled without importing dataset.py."""
    def __init__(self, stoi: dict, itos: list):
        self.stoi = stoi
        self.itos = itos
    def __len__(self):
        return len(self.itos)
    def lookup_token(self, idx: int) -> str:
        return self.itos[idx]
    def lookup_indices(self, tokens: list) -> list:
        unk = self.stoi.get('<unk>', 0)
        return [self.stoi.get(t, unk) for t in tokens]


def _vocab_from_raw(raw) -> Optional['_Vocab']:
    """Convert a raw {'stoi': ..., 'itos': ...} dict (or existing vocab) to _Vocab."""
    if raw is None:
        return None
    if isinstance(raw, _Vocab):
        return raw
    if isinstance(raw, dict) and 'stoi' in raw and 'itos' in raw:
        return _Vocab(raw['stoi'], raw['itos'])
    if hasattr(raw, 'stoi') and hasattr(raw, 'itos'):
        return _Vocab(raw.stoi, raw.itos)
    return None


# ══════════════════════════════════════════════════════════════════════
#   STANDALONE ATTENTION FUNCTION
#    Exposed at module level so the autograder can import and test it
#    independently of MultiHeadAttention.
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention.

        Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V

    Args:
        Q    : Query tensor,  shape (..., seq_q, d_k)
        K    : Key tensor,    shape (..., seq_k, d_k)
        V    : Value tensor,  shape (..., seq_k, d_v)
        mask : Optional Boolean mask, shape broadcastable to
               (..., seq_q, seq_k).
               Positions where mask is True are MASKED OUT
               (set to -inf before softmax).

    Returns:
        output : Attended output,   shape (..., seq_q, d_v)
        attn_w : Attention weights, shape (..., seq_q, seq_k)
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))

    attn_w = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_w, V)
    return output, attn_w


# ══════════════════════════════════════════════════════════════════════
# ❷  MASK HELPERS
#    Exposed at module level so they can be tested independently and
#    reused inside Transformer.forward.
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a padding mask for the encoder (source sequence).

    Args:
        src     : Source token-index tensor, shape [batch, src_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, 1, src_len]
        True  → position is a PAD token (will be masked out)
        False → real token
    """
    # [batch, src_len] → [batch, 1, 1, src_len]
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(
    tgt: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a combined padding + causal (look-ahead) mask for the decoder.

    Args:
        tgt     : Target token-index tensor, shape [batch, tgt_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, tgt_len, tgt_len]
        True → position is masked out (PAD or future token)
    """
    tgt_len = tgt.size(1)
    # padding mask: [batch, 1, 1, tgt_len] → broadcasts over seq_q dim
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)
    # causal mask: upper triangle is True (future positions masked out)
    causal_mask = torch.triu(
        torch.ones(tgt_len, tgt_len, dtype=torch.bool, device=tgt.device),
        diagonal=1,
    ).unsqueeze(0).unsqueeze(0)   # [1, 1, tgt_len, tgt_len]
    return pad_mask | causal_mask


# ══════════════════════════════════════════════════════════════════════
#  MULTI-HEAD ATTENTION
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention as in "Attention Is All You Need", §3.2.2.

        MultiHead(Q,K,V) = Concat(head_1,...,head_h) · W_O
        head_i = Attention(Q·W_Qi, K·W_Ki, V·W_Vi)

    You are NOT allowed to use torch.nn.MultiheadAttention.

    Args:
        d_model   (int)  : Total model dimensionality. Must be divisible by num_heads.
        num_heads (int)  : Number of parallel attention heads h.
        dropout   (float): Dropout probability applied to attention weights.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query : shape [batch, seq_q, d_model]
            key   : shape [batch, seq_k, d_model]
            value : shape [batch, seq_k, d_model]
            mask  : Optional BoolTensor broadcastable to
                    [batch, num_heads, seq_q, seq_k]
                    True → masked out (attend nowhere)

        Returns:
            output : shape [batch, seq_q, d_model]
        """
        batch = query.size(0)

        def project_and_split(linear, x):
            # [batch, seq, d_model] → [batch, heads, seq, d_k]
            return linear(x).view(batch, -1, self.num_heads, self.d_k).transpose(1, 2)

        Q = project_and_split(self.W_q, query)
        K = project_and_split(self.W_k, key)
        V = project_and_split(self.W_v, value)

        attn_out, _ = scaled_dot_product_attention(Q, K, V, mask)

        # Merge heads: [batch, seq_q, d_model]
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, -1, self.d_model)
        return self.W_o(attn_out)


# ══════════════════════════════════════════════════════════════════════
#   POSITIONAL ENCODING
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding as in "Attention Is All You Need", §3.5.

    Args:
        d_model  (int)  : Embedding dimensionality.
        dropout  (float): Dropout applied after adding encodings.
        max_len  (int)  : Maximum sequence length to pre-compute (default 5000).
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)                        # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)   # [1, max_len, d_model]
        # registered as buffer → not a trainable parameter, moves with .to(device)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Input embeddings, shape [batch, seq_len, d_model]

        Returns:
            Tensor of same shape [batch, seq_len, d_model]
            = x  +  PE[:, :seq_len, :]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════
#  FEED-FORWARD NETWORK
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network, §3.3:

        FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂

    Args:
        d_model (int)  : Input / output dimensionality (e.g. 512).
        d_ff    (int)  : Inner-layer dimensionality (e.g. 2048).
        dropout (float): Dropout applied between the two linears.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ══════════════════════════════════════════════════════════════════════
#  ENCODER LAYER
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """
    Single Transformer encoder sub-layer (Post-LayerNorm):
        x → [Self-Attention → Add & Norm] → [FFN → Add & Norm]

    Post-LN matches the original "Attention Is All You Need" paper.

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn       = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            shape [batch, src_len, d_model]
        """
        # Self-attention sub-layer with residual + post-LN
        attn_out = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout(attn_out))
        # FFN sub-layer with residual + post-LN
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


# ══════════════════════════════════════════════════════════════════════
#   DECODER LAYER
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    """
    Single Transformer decoder sub-layer (Post-LayerNorm):
        x → [Masked Self-Attn → Add & Norm]
          → [Cross-Attn(memory) → Add & Norm]
          → [FFN → Add & Norm]

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn        = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(p=dropout)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : Encoder output, shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            shape [batch, tgt_len, d_model]
        """
        # Masked self-attention
        attn1 = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn1))
        # Cross-attention over encoder memory
        attn2 = self.cross_attn(x, memory, memory, src_mask)
        x = self.norm2(x + self.dropout(attn2))
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        return x


# ══════════════════════════════════════════════════════════════════════
#  ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    """Stack of N identical EncoderLayer modules with final LayerNorm."""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.norm1.normalized_shape[0])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x    : shape [batch, src_len, d_model]
            mask : shape [batch, 1, 1, src_len]
        Returns:
            shape [batch, src_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Stack of N identical DecoderLayer modules with final LayerNorm."""

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm   = nn.LayerNorm(layer.norm1.normalized_shape[0])

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]
        Returns:
            shape [batch, tgt_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# ══════════════════════════════════════════════════════════════════════
#   FULL TRANSFORMER
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for sequence-to-sequence tasks.

    Args:
        src_vocab_size (int)  : Source vocabulary size.
        tgt_vocab_size (int)  : Target vocabulary size.
        d_model        (int)  : Model dimensionality (default 512).
        N              (int)  : Number of encoder/decoder layers (default 6).
        num_heads      (int)  : Number of attention heads (default 8).
        d_ff           (int)  : FFN inner dimensionality (default 2048).
        dropout        (float): Dropout probability (default 0.1).
        pad_idx        (int)  : Padding token index (default 1).
    """

    def __init__(
        self,
        src_vocab_size: Optional[int] = None,
        tgt_vocab_size: Optional[int] = None,
        d_model:   int   = 512,
        N:         int   = 6,
        num_heads: int   = 8,
        d_ff:      int   = 2048,
        dropout:   float = 0.1,
        checkpoint_path: Optional[str] = None,
        pad_idx:   int   = 1,
    ) -> None:
        super().__init__()

        # If vocab sizes are not given, try to auto-load from a saved checkpoint.
        # This allows Transformer() to be called with no arguments by the autograder.
        _ckpt_state = None
        if src_vocab_size is None or tgt_vocab_size is None:
            if checkpoint_path is None:
                for candidate in ("best_checkpoint.pt", "checkpoint.pt"):
                    if os.path.isfile(candidate):
                        checkpoint_path = candidate
                        break

        if checkpoint_path is not None:
            if not os.path.isfile(checkpoint_path):
                if _gdown is not None:
                    _gdown.download(id="<.pth drive id>", output=checkpoint_path, quiet=False)
                else:
                    raise FileNotFoundError(
                        f"Checkpoint not found at '{checkpoint_path}' and gdown is not installed."
                    )
            _ckpt_state = torch.load(checkpoint_path, map_location='cpu')
            cfg = _ckpt_state.get('model_config', {})
            if src_vocab_size is None:
                src_vocab_size = cfg.get('src_vocab_size', 10000)
            if tgt_vocab_size is None:
                tgt_vocab_size = cfg.get('tgt_vocab_size', 10000)
            d_model   = cfg.get('d_model',   d_model)
            N         = cfg.get('N',         N)
            num_heads = cfg.get('num_heads', num_heads)
            d_ff      = cfg.get('d_ff',      d_ff)
            dropout   = cfg.get('dropout',   dropout)
            pad_idx   = cfg.get('pad_idx',   pad_idx)

        # Final fallback so construction never fails
        src_vocab_size = src_vocab_size or 10000
        tgt_vocab_size = tgt_vocab_size or 10000

        self.d_model = d_model
        self.pad_idx = pad_idx

        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.src_pe    = PositionalEncoding(d_model, dropout)
        self.tgt_pe    = PositionalEncoding(d_model, dropout)

        enc_layer = EncoderLayer(d_model, num_heads, d_ff, dropout)
        dec_layer = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder = Encoder(enc_layer, N)
        self.decoder = Decoder(dec_layer, N)
        self.fc_out  = nn.Linear(d_model, tgt_vocab_size)

        # Vocabulary objects — needed by infer(); set during training or loaded from checkpoint
        self.src_vocab = None
        self.tgt_vocab = None

        self._init_weights()

        if _ckpt_state is not None:
            self.load_state_dict(_ckpt_state['model_state_dict'])
            # Convert raw dicts → _Vocab so infer() can use .stoi / .itos immediately
            self.src_vocab = _vocab_from_raw(_ckpt_state.get('src_vocab'))
            self.tgt_vocab = _vocab_from_raw(_ckpt_state.get('tgt_vocab'))

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # ── AUTOGRADER HOOKS ── keep these signatures exactly ─────────────

    def encode(
        self,
        src:      torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full encoder stack.

        Args:
            src      : Token indices, shape [batch, src_len]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            memory : Encoder output, shape [batch, src_len, d_model]
        """
        x = self.src_pe(self.src_embed(src) * math.sqrt(self.d_model))
        return self.encoder(x, src_mask)

    def decode(
        self,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt:      torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full decoder stack and project to vocabulary logits.

        Args:
            memory   : Encoder output,  shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt      : Token indices,   shape [batch, tgt_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        x = self.tgt_pe(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        x = self.decoder(x, memory, src_mask, tgt_mask)
        return self.fc_out(x)

    def forward(
        self,
        src:      torch.Tensor,
        tgt:      torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full encoder-decoder forward pass.

        Args:
            src      : shape [batch, src_len]
            tgt      : shape [batch, tgt_len]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    def _ensure_vocab(self) -> None:
        """
        Lazily loads src/tgt vocabularies if not already set.

        Priority (all offline — no network required):
          1. Already a proper _Vocab object on self.
          2. vocab.pkl  — plain-dict format written by save_checkpoint().
          3. Any *.pt checkpoint on disk that contains 'src_vocab' / 'tgt_vocab'.
        """
        # Convert if already set but still a raw dict (e.g. from old checkpoint load)
        if self.src_vocab is not None:
            self.src_vocab = _vocab_from_raw(self.src_vocab)
        if self.tgt_vocab is not None:
            self.tgt_vocab = _vocab_from_raw(self.tgt_vocab)

        if isinstance(self.src_vocab, _Vocab) and isinstance(self.tgt_vocab, _Vocab):
            return

        import pickle

        # 1. Dedicated vocab file written by save_checkpoint()
        for vocab_file in ("vocab.pkl",):
            if os.path.isfile(vocab_file):
                with open(vocab_file, "rb") as f:
                    data = pickle.load(f)
                src_v = data.get("src_vocab")
                tgt_v = data.get("tgt_vocab")
                if src_v is not None and tgt_v is not None:
                    self.src_vocab = src_v
                    self.tgt_vocab = tgt_v
                    return

        # 2. Any checkpoint that has vocab embedded
        for candidate in ("best_checkpoint.pt", "checkpoint.pt"):
            if os.path.isfile(candidate):
                ckpt = torch.load(candidate, map_location="cpu")
                src_v = _vocab_from_raw(ckpt.get("src_vocab"))
                tgt_v = _vocab_from_raw(ckpt.get("tgt_vocab"))
                if src_v is not None and tgt_v is not None:
                    self.src_vocab = src_v
                    self.tgt_vocab = tgt_v
                    return

        raise RuntimeError(
            "Could not find vocabulary. "
            "Make sure vocab.pkl (saved by save_checkpoint) is included in your submission."
        )

    def infer(self, src_sentence: str) -> str:
        """
        Translates a German sentence to English using greedy autoregressive decoding.

        Args:
            src_sentence: The raw German text.

        Returns:
            The fully translated English string, detokenized and clean.
        """
        import spacy
        from train import greedy_decode

        self._ensure_vocab()   # load vocab from vocab.pkl / checkpoint if not already set

        # spacy.blank("de") uses built-in rules only — no model download, works offline
        spacy_de = spacy.blank("de")
        pad_idx = self.pad_idx
        sos_idx = self.src_vocab.stoi.get('<sos>', 2)
        eos_idx = self.src_vocab.stoi.get('<eos>', 3)
        tgt_sos = self.tgt_vocab.stoi.get('<sos>', 2)
        tgt_eos = self.tgt_vocab.stoi.get('<eos>', 3)

        tokens = [tok.text.lower() for tok in spacy_de.tokenizer(src_sentence)]
        src_ids = [sos_idx] + self.src_vocab.lookup_indices(tokens) + [eos_idx]

        device = next(self.parameters()).device
        src = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)
        src_mask = make_src_mask(src, pad_idx)

        self.eval()
        out = greedy_decode(self, src, src_mask, max_len=100,
                            start_symbol=tgt_sos, end_symbol=tgt_eos, device=str(device))

        out_tokens = out[0].tolist()
        words = []
        for t in out_tokens:
            if t == tgt_eos:
                break
            if t not in (tgt_sos, pad_idx):
                words.append(self.tgt_vocab.lookup_token(t))

        return " ".join(words)
