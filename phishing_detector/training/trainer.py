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
from sklearn.model_selection import StratifiedKFold, train_test_split

class PhishingTrainer:
    """Coordinate training for the email branch, URL branch and fusion model."""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config
        self.loader = DatasetLoader(config)
        self.email_preprocessor = EmailPreprocessor()
        self.url_preprocessor = URLPreprocessor()

    def load_datasets(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        email_df = self.loader.load_email_dataset()
        nazario_df = self.loader.load_nazario_dataset()
        phish_df = self.loader.load_phishtank_dataset()
        phiusiil_df = self.loader.load_phiusiil_dataset()
        # majestic_df = self.loader.load_majestic_dataset()
        # malicious_df = self.loader.load_malicious_phish_dataset()
        return email_df, phish_df, phiusiil_df, nazario_df
      
    def _mean_metrics(self, metrics_list: list[dict[str, float]]) -> dict[str, float]:
        if not metrics_list:
            return {}

        return {
            metric: float(np.mean([m[metric] for m in metrics_list]))
            for metric in metrics_list[0]
        }

    def train_email(self, email_df: pd.DataFrame, nazario_df: pd.DataFrame, n_splits: int = 5):
        email_df = pd.concat(
            [email_df, nazario_df],
            ignore_index=True,
        )

        print(f"CEAS: {len(email_df)}")
        print(f"Nazario: {len(nazario_df)}")
        
        email_processed = self.email_preprocessor.preprocess_dataframe(email_df)

        X = email_processed["combined_text"].tolist()
        y = email_processed["label"].to_numpy(dtype=int)

        if n_splits > 1 and len(y) >= n_splits:
            fold_metrics: list[dict[str, float]] = []
            cv = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=self.config.seed,
            )

            for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
                print(f"Email fold {fold}/{n_splits}")

                X_train = [X[i] for i in train_idx]
                X_validate = [X[i] for i in val_idx]
                y_train = y[train_idx]
                y_validate = y[val_idx]

                email_vectorizer = EmailTfidfModel()
                email_vectorizer.fit(X_train)

                X_train_vec = email_vectorizer.transform(X_train)
                X_validate_vec = email_vectorizer.transform(X_validate)

                email_model = EmailBranchModel()
                email_model.fit(X_train_vec, y_train)

                email_scores = np.array(
                    email_model.predict_proba(X_validate_vec),
                    dtype=float,
                )

                fold_results = evaluate_model(
                    y_true=y_validate,
                    y_pred=(email_scores >= 0.5).astype(int),
                    y_prob=email_scores,
                )
                print(
                    f"  accuracy={fold_results['accuracy']:.4f} "
                    f"precision={fold_results['precision']:.4f} "
                    f"recall={fold_results['recall']:.4f} "
                    f"roc_auc={fold_results['roc_auc']:.4f}"
                )

                fold_metrics.append(fold_results)

            email_metrics = self._mean_metrics(fold_metrics)
            print(f"Email cross-validation mean accuracy: {email_metrics['accuracy']:.4f}")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                stratify=y,
                random_state=self.config.seed,
            )
            
            email_vectorizer = EmailTfidfModel()
            email_vectorizer.fit(X_train)
            
            X_train_vec = email_vectorizer.transform(X_train)
            X_test_vec = email_vectorizer.transform(X_test)
            
            email_model = EmailBranchModel()
            email_model.fit(X_train_vec, y_train)

            email_scores = np.array(
                email_model.predict_proba(X_test_vec),
                dtype=float,
            )

            email_metrics = evaluate_model(
                y_true=y_test,
                y_pred=(email_scores >= 0.5).astype(int),
                y_prob=email_scores,
            )

        final_vectorizer = EmailTfidfModel()
        final_vectorizer.fit(X)
        X_vec = final_vectorizer.transform(X)
        final_model = EmailBranchModel()
        final_model.fit(X_vec, y)

        save_artifact(final_vectorizer, self.config.models_dir / "email_tfidf.joblib")
        save_artifact(final_model, self.config.models_dir / "email_xgb.joblib")

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

    def train_url(
        self,
        phish_df: pd.DataFrame,
        phiusiil_df: pd.DataFrame,
        n_splits: int = 10,
        epochs: int = 10,
        batch_size: int = 32,
    ):
        legitimate_df = self.filter_phiusiil_legitimate_urls(phiusiil_df)

        legitimate_df = legitimate_df.sample(
            n=len(phish_df),
            random_state=self.config.seed,
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
            random_state=self.config.seed,
        ).reset_index(drop=True)

        url_df = url_df.dropna(subset=["URL"]).reset_index(drop=True)

        url_df["URL"] = url_df["URL"].apply(normalize_url)
        url_df["Label"] = url_df["Label"].astype(int)

        X = url_df["URL"].tolist()
        y = url_df["Label"].to_numpy(dtype=int)

        if n_splits > 1 and len(y) >= n_splits:
            fold_metrics: list[dict[str, float]] = []
            cv = StratifiedKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=self.config.seed,
            )

            for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
                print(f"URL fold {fold}/{n_splits}")

                X_train = [X[i] for i in train_idx]
                X_validate = [X[i] for i in val_idx]
                y_train = y[train_idx].tolist()
                y_validate = y[val_idx]

                url_model = URLCharCNNModel(preprocessor=self.url_preprocessor)
                url_model.fit(
                    X_train,
                    y_train,
                    epochs=epochs,
                    batch_size=batch_size,
                )

                url_scores = np.array(
                    url_model.predict_proba(X_validate),
                    dtype=float,
                )

                fold_results = evaluate_model(
                    y_true=y_validate,
                    y_pred=(url_scores >= 0.5).astype(int),
                    y_prob=url_scores,
                )
                print(
                    f"  accuracy={fold_results['accuracy']:.4f} "
                    f"precision={fold_results['precision']:.4f} "
                    f"recall={fold_results['recall']:.4f} "
                    f"roc_auc={fold_results['roc_auc']:.4f}"
                )

                fold_metrics.append(fold_results)

            url_metrics = self._mean_metrics(fold_metrics)
            print(f"URL cross-validation mean accuracy: {url_metrics['accuracy']:.4f}")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                stratify=y,
                random_state=self.config.seed,
            )

            url_model = URLCharCNNModel(
                preprocessor=self.url_preprocessor,
            )
            
            url_model.fit(
                X_train,
                y_train.tolist(),
                epochs=epochs,
                batch_size=batch_size,
            )

            url_scores = np.array(
                url_model.predict_proba(X_test),
                dtype=float,
            )

            url_metrics = evaluate_model(
                y_true=y_test,
                y_pred=(url_scores >= 0.5).astype(int),
                y_prob=url_scores,
            )

        final_url_model = URLCharCNNModel(preprocessor=self.url_preprocessor)
        final_url_model.fit(
            X,
            y.tolist(),
            epochs=epochs,
            batch_size=batch_size,
        )

        final_url_model.save(
            str(self.config.models_dir / "url_charcnn.pt")
        )

        return url_metrics
    

    def train_all(
        self,
        email_df: pd.DataFrame,
        phish_df: pd.DataFrame,
        phiusiil_df: pd.DataFrame,
        nazario_df: pd.DataFrame,
        n_splits: int = 10,
        epochs: int = 10,
        batch_size: int = 32,
    ) -> dict[str, Any]:
        self.config.models_dir.mkdir(parents=True, exist_ok=True)

        email_metrics = self.train_email(email_df, nazario_df)
        url_metrics = self.train_url(
            phish_df,
            phiusiil_df,
            n_splits=n_splits,
            epochs=epochs,
            batch_size=batch_size,
        )

        return {
            "email_metrics": email_metrics,
            "url_metrics": url_metrics,
        }

