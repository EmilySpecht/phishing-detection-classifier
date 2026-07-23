from __future__ import annotations

import re
from typing import Any

import pandas as pd



class EmailPreprocessor:
    """Limpa e padroniza os campos textuais de um e-mail antes de serem usados no treinamento ou na inferência"""

    def __init__(self) -> None:
        self._html_tag_pattern = re.compile(r"<[^>]+>")
        self._non_alphanumeric_pattern = re.compile(r"[^a-z0-9\s]")

    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply text cleaning to the email dataset."""
        required_columns = {"sender", "subject", "body", "label"}
        missing = required_columns.difference(df.columns)
        if missing:
            raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

        cleaned = df.loc[:, ["sender", "subject", "body", "label"]].copy()
        cleaned = cleaned.dropna().reset_index(drop=True)
        cleaned["sender"] = cleaned["sender"].fillna("").astype(str)
        cleaned["subject"] = cleaned["subject"].fillna("").astype(str)
        cleaned["body"] = cleaned["body"].fillna("").astype(str)
        cleaned["label"] = cleaned["label"].astype(int)

        cleaned["sender"] = cleaned["sender"].apply(self._clean_text)
        cleaned["subject"] = cleaned["subject"].apply(self._clean_text)
        cleaned["body"] = cleaned["body"].apply(self._clean_text)
        cleaned["combined_text"] = cleaned.apply(
            lambda row: self._build_combined_text(row["sender"], row["subject"], row["body"]),
            axis=1,
        )
        return cleaned

    def preprocess_record(self, sender: str, subject: str, body: str) -> str:
        """Transform a single email into the text representation used by the model."""
        sender_clean = self._clean_text(sender)
        subject_clean = self._clean_text(subject)
        body_clean = self._clean_text(body)
        return self._build_combined_text(sender_clean, subject_clean, body_clean)

    def _build_combined_text(self, sender: str, subject: str, body: str) -> str:
        return f"[Sender] {sender} [Subject] {subject} [Body] {body}".strip()

    def _clean_text(self, text: str) -> str:
        if text is None:
            return ""
        text = str(text)
        text = self._html_tag_pattern.sub(" ", text)
        text = re.sub(r"\s+", " ", text)
        text = text.lower()
        text = self._non_alphanumeric_pattern.sub(" ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
