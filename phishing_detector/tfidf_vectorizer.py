from __future__ import annotations

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


class EmailTfidfModel:
    """Wraps TF-IDF vectorization for the email branch."""

    def __init__(self, max_features: int = 2000, ngram_range: tuple[int, int] = (1, 2)) -> None:
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
        )

    def fit(self, texts: list[str]) -> None:
        self.vectorizer.fit(texts)

    def transform(self, texts: list[str]) -> object:
        return self.vectorizer.transform(texts)

    def save(self, path: str) -> None:
        joblib.dump(self.vectorizer, path)

    @classmethod
    def load(cls, path: str) -> "EmailTfidfModel":
        model = cls.__new__(cls)
        model.vectorizer = joblib.load(path)
        return model
