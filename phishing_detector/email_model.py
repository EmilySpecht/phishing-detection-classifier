from __future__ import annotations

import joblib
from xgboost import XGBClassifier


class EmailBranchModel:
    """XGBoost classifier for the email-text branch."""

    def __init__(self) -> None:
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=1,
            random_state=42,
        )

    def fit(self, X, y) -> None:
        self.model.fit(X, y)

    def predict_proba(self, X) -> list[float]:
        probabilities = self.model.predict_proba(X)
        if isinstance(probabilities, list):
            return [float(prob[1]) if isinstance(prob, (list, tuple)) else float(prob) for prob in probabilities]
        if hasattr(probabilities, "ndim") and probabilities.ndim == 1:
            return [float(probabilities[1])]
        return probabilities[:, 1].astype(float).tolist()

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str) -> "EmailBranchModel":
        model = cls.__new__(cls)
        model.model = joblib.load(path)
        return model
