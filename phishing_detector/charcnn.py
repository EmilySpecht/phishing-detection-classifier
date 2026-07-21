from __future__ import annotations

import os
from typing import Any

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from phishing_detector.preprocessing.preprocess_url import URLPreprocessor


class URLDataset(Dataset):
    def __init__(self, urls: list[str], labels: list[int], preprocessor: URLPreprocessor) -> None:
        self.urls = urls
        self.labels = labels
        self.preprocessor = preprocessor

    def __len__(self) -> int:
        return len(self.urls)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.preprocessor.encode_batch([self.urls[idx]])[0]
        label = torch.tensor([self.labels[idx]], dtype=torch.float32)
        return encoded, label


class CharCNN(nn.Module):
    """Character-level CNN for URL classification."""

    def __init__(self, vocab_size: int, embedding_dim: int = 32, num_filters: int = 64, max_length: int = 128) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = nn.Conv1d(embedding_dim, num_filters, kernel_size=5)
        self.conv2 = nn.Conv1d(embedding_dim, num_filters, kernel_size=3)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(num_filters * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x).permute(0, 2, 1)
        conv1 = torch.relu(self.conv1(embedded))
        conv2 = torch.relu(self.conv2(embedded))
        pooled1 = torch.max(conv1, dim=2).values
        pooled2 = torch.max(conv2, dim=2).values
        flat = torch.cat([pooled1, pooled2], dim=1)
        flat = self.dropout(flat)
        return torch.sigmoid(self.fc(flat))


class URLCharCNNModel:
    """Training and inference wrapper for the URL branch."""

    def __init__(self, preprocessor: URLPreprocessor, model: CharCNN | None = None, device: str | None = None) -> None:
        self.preprocessor = preprocessor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model or CharCNN(vocab_size=len(preprocessor.vocab))
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.BCELoss()

    def fit(self, urls: list[str], labels: list[int], epochs: int = 8, batch_size: int = 16) -> None:
        dataset = URLDataset(urls, labels, self.preprocessor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        self.model.train()

        for _ in range(epochs):
            for inputs, target in loader:
                inputs = inputs.to(self.device)
                target = target.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs.squeeze(1), target.squeeze(1))
                loss.backward()
                self.optimizer.step()

    def predict_proba(self, urls: list[str]) -> list[float]:
        self.model.eval()
        with torch.no_grad():
            inputs = self.preprocessor.encode_batch(urls, device=self.device)
            outputs = self.model(inputs)
            return outputs.squeeze(1).cpu().tolist()

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.model.state_dict(), "preprocessor": self.preprocessor}, path)

    @classmethod
    def load(cls, path: str, preprocessor: URLPreprocessor | None = None) -> "URLCharCNNModel":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        preprocessor = preprocessor or checkpoint["preprocessor"]
        model = cls(preprocessor=preprocessor)
        model.model.load_state_dict(checkpoint["state_dict"])
        return model
