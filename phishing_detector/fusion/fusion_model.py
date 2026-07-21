from __future__ import annotations

import joblib
from xgboost import XGBClassifier


class FusionModel:
    """Late-fusion classifier that combines email and URL branch probabilities."""

    def __init__(self) -> None:
        self.model = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=1,
            random_state=42,
        )

    def fit(self, X, y) -> None:
        self.model.fit(X, y)

    def predict_proba(self, X) -> list[float]:
        return self.model.predict_proba(X)[:, 1].tolist()

    def save(self, path: str) -> None:
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str) -> "FusionModel":
        model = cls.__new__(cls)
        model.model = joblib.load(path)
        return model
