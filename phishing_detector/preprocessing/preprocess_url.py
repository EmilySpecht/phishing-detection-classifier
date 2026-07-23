from __future__ import annotations

import re
import torch

from urllib.parse import urlparse

# Utilizar o normalizador de URL caso utilizar, por exemplo, o dataset majestic_million, 
# pois ele contém URLs simples, apenas com dominio e subdominio, sem informações como http e www,
# e este normalizador vai padronizar as URLs dos datasets utilizados para não haver viés no momento
# do treinamento
def normalize_url(url: str) -> str:
    url = url.strip().lower()

    # adiciona protocolo se não existir
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    parsed = urlparse(url)

    scheme = parsed.scheme
    netloc = parsed.netloc

    # remove www.
    # if netloc.startswith("www."):
    #     netloc = netloc[4:]

    path = parsed.path
    query = parsed.query

    normalized = f"{scheme}://{netloc}{path}"

    if query:
        normalized += f"?{query}"

    return normalized


class URLPreprocessor:
    """Prepara strings de URL para um modelo de rede neural convolucional em nível de caractere (Char-CNN)."""

    def __init__(self, vocab: dict[str, int] | None = None, max_length: int = 128) -> None:
        self.vocab = vocab or self._build_default_vocab()
        self.max_length = max_length

    def encode_url(self, url: str) -> list[int]:
        """
        Exemplo de saída:
        [
            9,    # h
            21,   # t
            21,   # t
            17,   # p
        ]
        """
        if url is None:
            return [0]
        normalized = self._normalize(url)
        
        encoded = []

        for char in normalized:
            encoded.append(
                self.vocab.get(char, self.vocab["<UNK>"])
            )

        return encoded
        # return [self.vocab.get(char, self.vocab["<UNK>"]) for char in normalized]

    def pad_sequence(self, indices: list[int], length: int | None = None) -> list[int]:
        """Faz o padding e o truncamento da sequência de índices para que todas as URLs tenham o mesmo tamanho"""
        target_length = self.max_length if length is None else length
        if len(indices) >= target_length:
            return indices[:target_length]
        return indices + [self.vocab["<PAD>"]] * (target_length - len(indices))

    def encode_batch(self, urls: list[str], device: str | None = None) -> torch.Tensor:
        """prepara um batch de URLs para ser enviado à Char-CNN"""
        encoded = [self.pad_sequence(self.encode_url(url)) for url in urls]
        tensor = torch.tensor(encoded, dtype=torch.long)
        return tensor.to(device) if device else tensor

    def _normalize(self, url: str) -> str:
        """transforma tudo em caixa baixa e remove todos os espaços em branco"""
        url = str(url).strip().lower()
        url = re.sub(r"\s+", "", url)
        return url

    def _build_default_vocab(self) -> dict[str, int]:
        """
        Exemplo de saída:
        {
            "<PAD>": 0,
            "<UNK>": 1,
            "a": 2,
            "b": 3,
            "c": 4,
            ...
        }
        """   
        alphabet = list("abcdefghijklmnopqrstuvwxyz0123456789:/._-?=&%#@") # Alfabeto minúsculo, dígitos e símbolos comuns em URLs (:/._-?=&%#@).
        chars = ["<PAD>", "<UNK>"] + alphabet
        return {char: idx for idx, char in enumerate(chars)}

    