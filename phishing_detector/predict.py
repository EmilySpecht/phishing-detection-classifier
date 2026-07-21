from __future__ import annotations

from pathlib import Path
from typing import Any

from phishing_detector.charcnn import URLCharCNNModel
from phishing_detector.configs.config import get_config
from phishing_detector.email_model import EmailBranchModel
from phishing_detector.preprocessing.preprocess_email import EmailPreprocessor
from phishing_detector.preprocessing.preprocess_url import URLPreprocessor
from phishing_detector.tfidf_vectorizer import EmailTfidfModel
from phishing_detector.preprocessing.preprocess_url import normalize_url

def predict_email(
    sender: str,
    subject: str,
    body: str,
    url: str,
    model_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run inference for a single email using the trained branches and fusion model."""
    config = get_config(str(Path(__file__).resolve().parents[1]))
    model_dir = Path(model_dir) if model_dir is not None else config.models_dir

    email_preprocessor = EmailPreprocessor()
    url_preprocessor = URLPreprocessor()

    email_text = email_preprocessor.preprocess_record(sender, subject, body)
    tfidf_model = EmailTfidfModel.load(str(model_dir / "email_tfidf.joblib"))
    email_model = EmailBranchModel.load(str(model_dir / "email_xgb.joblib"))
    url_model = URLCharCNNModel.load(str(model_dir / "url_charcnn.pt"), preprocessor=url_preprocessor)

    email_features = tfidf_model.transform([email_text])
    email_score = float(email_model.predict_proba(email_features)[0])
    
    url = normalize_url(url)
    url_score = float(url_model.predict_proba([url])[0])
    
    
    # Por enquanto temos uma média simples, mas é interessante 
    # analisar qual das analises tem mais peso na decisão final, e ponderar de acordo com isso.
    # exemplo: fusion_prob = (email_score * 0.7 + url_score * 0.3) / (0.7 + 0.3)
    fusion_prob = (email_score + url_score) / 2

    return {
        "classe": "phishing" if fusion_prob >= 0.7 else "legitimo",
        "probabilidade": round(fusion_prob, 4),
        "score_email": round(email_score, 4),
        "score_url": round(url_score, 4),
    }
