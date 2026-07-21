from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from phishing_detector.configs.config import ProjectConfig


class DatasetLoader:
    """Load email and URL datasets, falling back to synthetic data if files are absent."""

    def __init__(self, config: ProjectConfig) -> None:
        self.config = config

    def load_email_dataset(self) -> pd.DataFrame:
        dataset_path = self.config.datasets_dir / "ceas_08.csv"
        print(f"Dataset path generated {dataset_path}")
        
        if dataset_path.exists():
            print(f"Loading email dataset from {dataset_path}")
            return pd.read_csv(dataset_path)
        print(f"Creating synthetic email dataset")
        return self._create_synthetic_email_dataset()

    def load_url_dataset(self) -> pd.DataFrame:
        dataset_path = self.config.datasets_dir / "phishtank_urls.csv"
        if dataset_path.exists():
            return pd.read_csv(dataset_path)
        return self._create_synthetic_url_dataset()

    def _create_synthetic_email_dataset(self) -> pd.DataFrame:
        records = [
            {"sender": "alice@example.com", "subject": "Invoice", "body": "Please review the attached invoice", "label": 0},
            {"sender": "support@bank.com", "subject": "Security alert", "body": "Verify your account now", "label": 1},
            {"sender": "bob@example.net", "subject": "Meeting", "body": "We will meet tomorrow", "label": 0},
            {"sender": "paypal-alert@secure-login.com", "subject": "Urgent", "body": "Click here to confirm your password", "label": 1},
            {"sender": "carol@example.org", "subject": "Project update", "body": "Thanks for the update", "label": 0},
            {"sender": "amazon-support@verify-account.net", "subject": "Action required", "body": "Update your billing information", "label": 1},
        ]
        return pd.DataFrame(records)

    def _create_synthetic_url_dataset(self) -> pd.DataFrame:
        records = [
            {"url": "https://example.com/login", "label": 0},
            {"url": "https://secure-payment.example.com/checkout", "label": 0},
            {"url": "https://login.verify-account-now.com", "label": 1},
            {"url": "http://paypal-secure-update.net/confirm", "label": 1},
            {"url": "https://docs.example.org", "label": 0},
            {"url": "https://amazon-account-verification.info", "label": 1},
        ]
        return pd.DataFrame(records)
