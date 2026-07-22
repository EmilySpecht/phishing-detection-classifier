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
from phishing_detector.preprocessing.preprocess_url import normalize_url
from sklearn.model_selection import train_test_split

class PhishingTrainer:
    """Coordinate training for the email branch, URL branch and fusion model."""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.loader = DatasetLoader(config)
        self.email_preprocessor = EmailPreprocessor()
        self.url_preprocessor = URLPreprocessor()

    def load_datasets(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        email_df = self.loader.load_email_dataset()
        phish_df = self.loader.load_phishtank_dataset()
        phiusiil_df = self.loader.load_phiusiil_dataset()
        # majestic_df = self.loader.load_majestic_dataset()
        # malicious_df = self.loader.load_malicious_phish_dataset()
        return email_df, phish_df, phiusiil_df
      
    def train_email(self, email_df: pd.DataFrame):
        print(f"E-mails: {len(email_df)}")
        
        email_processed = self.email_preprocessor.preprocess_dataframe(email_df)

        X = email_processed["combined_text"]
        y = email_processed["label"]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            stratify=y,
            random_state=42,
        )
        
        email_vectorizer = EmailTfidfModel()
        email_vectorizer.fit(X_train.tolist())
        
        X_train_vec = email_vectorizer.transform(X_train.tolist())
        X_test_vec = email_vectorizer.transform(X_test.tolist())
        
        email_model = EmailBranchModel()
        email_model.fit(X_train_vec, y_train.to_numpy())

        email_scores = np.array(
            email_model.predict_proba(X_test_vec),
            dtype=float,
        )

        save_artifact(email_vectorizer, self.config.models_dir / "email_tfidf.joblib")
        save_artifact(email_model, self.config.models_dir / "email_xgb.joblib")

        email_metrics = evaluate_model(
            y_true=y_test.to_numpy(),
            y_pred=(email_scores >= 0.5).astype(int),
        )

        return email_metrics
    
    def filter_malicious_path_legitimate_urls(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        legitimate_df = (
            df[df["type"].str.lower() == "benign"]
            .rename(columns={"url": "URL"})
            .copy()
        )

        return legitimate_df
    
    def filter_phiusiil_legitimate_urls(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        legitimate_df = df[df["label"] == 0].copy()

        return legitimate_df

    def train_url(self, phish_df: pd.DataFrame, phiusiil_df: pd.DataFrame):
        legitimate_df = self.filter_phiusiil_legitimate_urls(phiusiil_df)

        legitimate_df = legitimate_df.sample(
            n=len(phish_df),
            random_state=42
        )
        
        print(f"URLs phishing: {len(phish_df)}")
        print(f"URLs legítimas: {len(legitimate_df)}")

        legitimate_df["Label"] = 0
        phish_df["Label"] = 1

        phish_df = phish_df[["URL", "Label"]]
        legitimate_df = legitimate_df[["URL", "Label"]]

        url_df = pd.concat(
            [phish_df, legitimate_df],
            ignore_index=True,
        )
        

        url_df = url_df.sample(
            frac=1,
            random_state=42,
        ).reset_index(drop=True)

        url_df = url_df.dropna(subset=["URL"]).reset_index(drop=True)

        # url_df["URL"] = url_df["URL"].apply(normalize_url)
        url_df["Label"] = url_df["Label"].astype(int)
        
        print(url_df.head(20))
        print("Benign:",
            url_df[url_df["Label"] == 0]["URL"].str.startswith("https://").mean())

        print("Phishing:",
            url_df[url_df["Label"] == 1]["URL"].str.startswith("https://").mean())

        X = url_df["URL"]
        y = url_df["Label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            stratify=y,
            random_state=42,
        )

        url_model = URLCharCNNModel(
            preprocessor=self.url_preprocessor
        )
        
        url_model.fit(
            X_train.tolist(),
            y_train.tolist(),
            epochs=20,
            batch_size=32,
        )

        url_scores = np.array(
            url_model.predict_proba(
                X_test.tolist()
            ),
            dtype=float,
        )

        url_model.save(
            str(self.config.models_dir / "url_charcnn.pt")
        )

        url_metrics = evaluate_model(
            y_true=y_test.to_numpy(),
            y_pred=(url_scores >= 0.5).astype(int),
        )

        return url_metrics
    

    def train_all(self, email_df: pd.DataFrame, phish_df: pd.DataFrame, phiusiil_df: pd.DataFrame) -> dict[str, Any]:
        self.config.models_dir.mkdir(parents=True, exist_ok=True)

        email_metrics = self.train_email(email_df)
        url_metrics = self.train_url(phish_df, phiusiil_df)

        return {
            "email_metrics": email_metrics,
            "url_metrics": url_metrics,
        }

