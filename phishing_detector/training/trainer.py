from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from phishing_detector.configs.config import ProjectConfig
from phishing_detector.dataset_loader import DatasetLoader
from phishing_detector.email_model import EmailBranchModel
from phishing_detector.fusion.fusion_model import FusionModel
from phishing_detector.charcnn import URLCharCNNModel
from phishing_detector.preprocessing.preprocess_email import EmailPreprocessor
from phishing_detector.preprocessing.preprocess_url import URLPreprocessor
from phishing_detector.tfidf_vectorizer import EmailTfidfModel
from phishing_detector.evaluation.metrics import evaluate_model
from phishing_detector.save_load import save_artifact


class PhishingTrainer:
    """Coordinate training for the email branch, URL branch and fusion model."""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.loader = DatasetLoader(config)
        self.email_preprocessor = EmailPreprocessor()
        self.url_preprocessor = URLPreprocessor()

    def load_datasets(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        email_df = self.loader.load_email_dataset()
        url_df = self.loader.load_url_dataset()
        return email_df, url_df

    def train_all(self, email_df: pd.DataFrame, url_df: pd.DataFrame) -> dict[str, Any]:
        self.config.models_dir.mkdir(parents=True, exist_ok=True)

        email_processed = self.email_preprocessor.preprocess_dataframe(email_df)
        email_vectorizer = EmailTfidfModel()
        email_vectorizer.fit(email_processed["combined_text"].tolist())

        X_train_email = email_vectorizer.transform(email_processed["combined_text"].tolist())
        y_train_email = email_processed["label"].to_numpy()
        email_model = EmailBranchModel()
        email_model.fit(X_train_email, y_train_email)

        url_processed = url_df.copy()
        url_processed = url_processed.dropna().reset_index(drop=True)
        url_processed["Label"] = url_processed["Label"].astype(int)
        url_model = URLCharCNNModel(preprocessor=self.url_preprocessor)
        url_model.fit(url_processed["URL"].tolist(), url_processed["Label"].tolist(), epochs=6, batch_size=4)

        email_scores = np.array(email_model.predict_proba(X_train_email), dtype=float)
        url_scores = np.array(url_model.predict_proba(url_processed["URL"].tolist()), dtype=float)

        fusion_features = np.column_stack([email_scores, url_scores])
        fusion_model = FusionModel()
        fusion_model.fit(fusion_features, url_processed["Label"].to_numpy())

        save_artifact(email_vectorizer, self.config.models_dir / "email_tfidf.joblib")
        save_artifact(email_model, self.config.models_dir / "email_xgb.joblib")
        url_model.save(str(self.config.models_dir / "url_charcnn.pt"))
        fusion_model.save(str(self.config.models_dir / "fusion_xgb.joblib"))

        email_metrics = evaluate_model(
            y_true=email_processed["label"].to_numpy(),
            y_pred=(np.array(email_scores) >= 0.5).astype(int),
        )
        url_metrics = evaluate_model(
            y_true=url_processed["Label"].to_numpy(),
            y_pred=(np.array(url_scores) >= 0.5).astype(int),
        )
        fusion_metrics = evaluate_model(
            y_true=url_processed["Label"].to_numpy(),
            y_pred=(np.array(fusion_model.predict_proba(fusion_features)) >= 0.5).astype(int),
        )

        return {
            "email_metrics": email_metrics,
            "url_metrics": url_metrics,
            "fusion_metrics": fusion_metrics,
        }
