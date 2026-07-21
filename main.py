from pathlib import Path

from phishing_detector.configs.config import get_config
from phishing_detector.training.trainer import PhishingTrainer
from phishing_detector.evaluation.metrics import evaluate_model
from phishing_detector.predict import predict_email


def main() -> None:
    config = get_config()
    trainer = PhishingTrainer(config)

    email_df, url_df = trainer.load_datasets()
    training_summary = trainer.train_all(email_df, url_df)

    print("Training completed successfully.")
    print(f"Email branch accuracy: {training_summary['email_metrics']['accuracy']:.4f}")
    print(f"URL branch accuracy: {training_summary['url_metrics']['accuracy']:.4f}")
    print(f"Fusion accuracy: {training_summary['fusion_metrics']['accuracy']:.4f}")

    sample = predict_email(
        sender="alice@example.com",
        subject="Urgent invoice",
        body="Please review the attached invoice before the deadline.",
        url="https://secure-payment.example.com/login",
        model_dir=config.models_dir,
    )
    print("Sample prediction:")
    print(sample)


if __name__ == "__main__":
    main()
