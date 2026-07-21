from __future__ import annotations

import re
from typing import Any

import numpy as np
import torch


class URLPreprocessor:
    """Prepare URL strings for character-level CNN training and inference."""

    def __init__(self, vocab: dict[str, int] | None = None, max_length: int = 128) -> None:
        self.vocab = vocab or self._build_default_vocab()
        self.max_length = max_length

    def encode_url(self, url: str) -> list[int]:
        if url is None:
            return [0]
        normalized = self._normalize(url)
        return [self.vocab.get(char, self.vocab["<UNK>"]) for char in normalized]

    def pad_sequence(self, indices: list[int], length: int | None = None) -> list[int]:
        target_length = self.max_length if length is None else length
        if len(indices) >= target_length:
            return indices[:target_length]
        return indices + [self.vocab["<PAD>"]] * (target_length - len(indices))

    def encode_batch(self, urls: list[str], device: str | None = None) -> torch.Tensor:
        encoded = [self.pad_sequence(self.encode_url(url)) for url in urls]
        tensor = torch.tensor(encoded, dtype=torch.long)
        return tensor.to(device) if device else tensor

    def _normalize(self, url: str) -> str:
        url = str(url).strip().lower()
        url = re.sub(r"\s+", "", url)
        return url

    def _build_default_vocab(self) -> dict[str, int]:
        alphabet = list("abcdefghijklmnopqrstuvwxyz0123456789:/._-?=&%#@")
        chars = ["<PAD>", "<UNK>"] + alphabet
        return {char: idx for idx, char in enumerate(chars)}
