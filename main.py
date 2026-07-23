from pathlib import Path

from numpy.random import sample

from phishing_detector.configs.config import get_config
from phishing_detector.training.trainer import PhishingTrainer
from phishing_detector.evaluation.metrics import evaluate_model
from phishing_detector.predict import predict_email


def main() -> None:
    config = get_config()
    trainer = PhishingTrainer(config)

    email_df, phish_df, phiusiil_df, nazario_df = trainer.load_datasets()
    training_summary = trainer.train_all(email_df, phish_df, phiusiil_df, nazario_df)

    print("Training completed successfully.")

    print("Email branch metrics:")
    for metric, value in training_summary["email_metrics"].items():
        print(f"  {metric}: {value:.4f}")

    print("URL branch metrics:")
    for metric, value in training_summary["url_metrics"].items():
        print(f"  {metric}: {value:.4f}")

    for sample_data in samples:
        result = predict_email(
            sender=sample_data["sender"],
            subject=sample_data["subject"],
            body=sample_data["body"],
            url=sample_data["url"],
            model_dir=config.models_dir,
        )

        print(f"=== {sample_data['name']} ===")
        print(f"Remetente: {sample_data['sender']}")
        print(f"URL: {sample_data['url']}")
        print(result)
        print()
        
        
        
samples = [
    {
        "name": "Exemplo 1",
        "sender": "alice@example.com",
        "subject": "Urgent invoice",
        "body": "Please review the attached invoice before the deadline.",
        "url": "https://www.secure-payment.example.com/login",
    },
    {
        "name": "Exemplo 2",
        "sender": "restaurant.variatto@gmail.com",
        "subject": "Congratulations, you've been selected!",
        "body": "Provide some details to receive the prize: CPF, ID, address, and phone number..",
        "url": "https://www.variatto-restaurant.com/form-prize",
    },
    {
        "name": "Exemplo 3",
        "sender": "nfe@homecenter.com.br",
        "subject": "NFe 7593 sent by issuer ESTILO HOME CENTER LTDA",
        "body": "Hello, please find the DANFE and XML attached.",
        "url": "https://www.homecenter.com.br/my-account/my-orders",
    },
    {
        "name": "Exemplo 4",
        "sender": "contact@google.com",
        "subject": "Welcome to Google",
        "body": "Thank you for creating your account.",
        "url": "https://www.google.com/maps",
    },
]

if __name__ == "__main__":
    main()
