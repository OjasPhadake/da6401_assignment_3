"""
dataset.py — Multi30k Dataset Loading and Tokenization
DA6401 Assignment 3: "Attention Is All You Need"
"""

from collections import Counter
from datasets import load_dataset
import spacy
from torch.utils.data import Dataset


# ══════════════════════════════════════════════════════════════════════
#   VOCABULARY
# ══════════════════════════════════════════════════════════════════════

class Vocab:
    """Simple vocabulary with special tokens at fixed indices."""

    UNK = '<unk>'   # index 0
    PAD = '<pad>'   # index 1
    SOS = '<sos>'   # index 2
    EOS = '<eos>'   # index 3

    def __init__(self, stoi: dict, itos: list):
        self.stoi = stoi   # token → index
        self.itos = itos   # index → token

    def __len__(self):
        return len(self.itos)

    def lookup_token(self, idx: int) -> str:
        return self.itos[idx]

    def lookup_indices(self, tokens: list) -> list:
        unk_idx = self.stoi[self.UNK]
        return [self.stoi.get(t, unk_idx) for t in tokens]


def _build_vocab(counter: Counter, min_freq: int) -> Vocab:
    specials = [Vocab.UNK, Vocab.PAD, Vocab.SOS, Vocab.EOS]
    tokens   = sorted(t for t, c in counter.items() if c >= min_freq)
    itos     = specials + tokens
    stoi     = {t: i for i, t in enumerate(itos)}
    return Vocab(stoi, itos)


# ══════════════════════════════════════════════════════════════════════
#   DATASET
# ══════════════════════════════════════════════════════════════════════

class Multi30kDataset(Dataset):
    """
    Wraps the bentrevett/multi30k HuggingFace dataset.

    Usage (training split — builds vocab):
        ds = Multi30kDataset(split='train')
        src_vocab, tgt_vocab = ds.build_vocab(min_freq=2)
        ds.process_data()

    Usage (val / test splits — reuse existing vocab):
        ds = Multi30kDataset(split='validation', src_vocab=src_vocab, tgt_vocab=tgt_vocab)
        ds.process_data()
    """

    PAD_IDX = 1
    SOS_IDX = 2
    EOS_IDX = 3
    UNK_IDX = 0

    def __init__(self, split: str = 'train', src_vocab=None, tgt_vocab=None):
        self.split = split
        # HuggingFace dataset — fields: 'de', 'en'
        self.data = load_dataset("bentrevett/multi30k", split=split)

        # Spacy tokenizers
        self.spacy_de = spacy.load("de_core_news_sm")
        self.spacy_en = spacy.load("en_core_web_sm")

        self.src_vocab  = src_vocab
        self.tgt_vocab  = tgt_vocab
        self.processed  = None   # list of (src_ids, tgt_ids) after process_data()

    # ── Tokenizers ────────────────────────────────────────────────────

    def tokenize_de(self, text: str) -> list:
        return [tok.text.lower() for tok in self.spacy_de.tokenizer(text)]

    def tokenize_en(self, text: str) -> list:
        return [tok.text.lower() for tok in self.spacy_en.tokenizer(text)]

    # ── Vocabulary building (training split only) ─────────────────────

    def build_vocab(self, min_freq: int = 2):
        """
        Builds src (de) and tgt (en) vocabularies from the current split.

        Returns:
            (src_vocab, tgt_vocab) : Vocab objects with stoi / itos / lookup_token.
        """
        de_counter: Counter = Counter()
        en_counter: Counter = Counter()

        for item in self.data:
            de_counter.update(self.tokenize_de(item['de']))
            en_counter.update(self.tokenize_en(item['en']))

        self.src_vocab = _build_vocab(de_counter, min_freq)
        self.tgt_vocab = _build_vocab(en_counter, min_freq)
        return self.src_vocab, self.tgt_vocab

    # ── Tokenize & index ─────────────────────────────────────────────

    def process_data(self):
        """
        Tokenize every sentence pair and convert to integer index lists.
        Wraps each sequence with <sos> and <eos>.

        Populates self.processed as a list of (src_ids, tgt_ids).
        """
        assert self.src_vocab is not None and self.tgt_vocab is not None, \
            "Call build_vocab() (or pass vocab objects) before process_data()"

        sos, eos = self.SOS_IDX, self.EOS_IDX
        processed = []
        for item in self.data:
            src_ids = [sos] + self.src_vocab.lookup_indices(self.tokenize_de(item['de'])) + [eos]
            tgt_ids = [sos] + self.tgt_vocab.lookup_indices(self.tokenize_en(item['en'])) + [eos]
            processed.append((src_ids, tgt_ids))

        self.processed = processed
        return processed

    # ── PyTorch Dataset interface ─────────────────────────────────────

    def __len__(self):
        return len(self.processed) if self.processed is not None else len(self.data)

    def __getitem__(self, idx):
        return self.processed[idx]
