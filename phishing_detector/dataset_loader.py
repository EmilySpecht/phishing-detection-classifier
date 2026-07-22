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
        
        if dataset_path.exists():
            print(f"Loading email dataset from {dataset_path}", end="\n\n")
            return pd.read_csv(dataset_path)
          
        print(f"Creating synthetic email dataset", end="\n\n")
        return self._create_synthetic_email_dataset()

    def load_phishtank_dataset(self) -> pd.DataFrame:
        dataset_path = self.config.datasets_dir / "phishtank_urls.csv"
        
        if dataset_path.exists():
            print(f"Loading phishing url dataset from {dataset_path}", end="\n\n")
            return pd.read_csv(dataset_path)
          
        print(f"Creating synthetic phishing url dataset", end="\n\n")
        return self._create_synthetic_phish_dataset()
      
    def load_majestic_dataset(self) -> pd.DataFrame:
        dataset_path = self.config.datasets_dir / "majestic_million.csv"
        
        if dataset_path.exists():
            print(f"Loading legitim url dataset from {dataset_path}", end="\n\n")
            return pd.read_csv(dataset_path)
          
        print(f"Creating synthetic legitimate url dataset", end="\n\n")
        return self._create_synthetic_legitim_dataset()

    def load_malicious_phish_dataset(self) -> pd.DataFrame:
        dataset_path = self.config.datasets_dir / "malicious_phish.csv"
        
        if dataset_path.exists():
            print(f"Loading malicious and legitimate url dataset from {dataset_path}", end="\n\n")
            return pd.read_csv(dataset_path)
          
        print(f"Creating synthetic malicious and legitimate url dataset", end="\n\n")
        return self._create_synthetic_legitim_dataset()
    
    
    def load_phiusiil_dataset(self) -> pd.DataFrame:
        dataset_path = self.config.datasets_dir / "phiusiil.csv"
        
        if dataset_path.exists():
            print(f"Loading malicious and legitimate url dataset from {dataset_path}", end="\n\n")
            return pd.read_csv(dataset_path)
            
        print(f"Creating synthetic malicious and legitimate url dataset", end="\n\n")
        return self._create_synthetic_legitim_dataset()
    
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

    def _create_synthetic_phish_dataset(self) -> pd.DataFrame:
        records = [
            {"URL": "https://example.com/login", "Label": 1},
            {"URL": "https://secure-payment.example.com/checkout", "Label": 1},
            {"URL": "https://login.verify-account-now.com", "Label": 1},
            {"URL": "http://paypal-secure-update.net/confirm", "Label": 1},
            {"URL": "https://docs.example.org", "Label": 1},
            {"URL": "https://amazon-account-verification.info", "Label": 1},
        ]
        return pd.DataFrame(records)

    def _create_synthetic_legitim_dataset(self) -> pd.DataFrame:
        records = [
            {"URL": "google.com", "Label": 0},
            {"URL": "ufrgs.br", "Label": 0},
            {"URL": "microsoft.com", "Label": 0},
            {"URL": "amazon.com", "Label": 0},
            {"URL": "facebook.com", "Label": 0},
            {"URL": "instagram.com", "Label": 0},
        ]
        return pd.DataFrame(records)
