from pathlib import Path

from phishing_detector.configs.config import get_config
from phishing_detector.training.trainer import PhishingTrainer
from phishing_detector.evaluation.metrics import evaluate_model
from phishing_detector.predict import predict_email


def main() -> None:
    config = get_config()
    trainer = PhishingTrainer(config)

    email_df, phish_df, malicious_df = trainer.load_datasets()
    training_summary = trainer.train_all(email_df, phish_df, malicious_df)

    print("Training completed successfully.")
    print(f"Email branch accuracy: {training_summary['email_metrics']['accuracy']:.4f}")
    print(f"URL branch accuracy: {training_summary['url_metrics']['accuracy']:.4f}")

    # sample = predict_email(
    #     sender="alice@example.com",
    #     subject="Urgent invoice",
    #     body="Please review the attached invoice before the deadline.",
    #     url="https://secure-payment.example.com/login",
    #     model_dir=config.models_dir,
    # )
    
    # sample = predict_email(
    #     sender="restaurante.variatto@gmail.com",
    #     subject="Parabéns, você foi sorteado!",
    #     body="Envie alguns dados para receber o prêmio. CPF, RG, endereço e telefone.",
    #     url="https://variatto-restaurantes.com/premio",
    #     model_dir=config.models_dir,
    # )
    
    # sample = predict_email(
    #     sender="nfe@estilohomecenter.com.br",
    #     subject="NFe 7593 enviada pelo emitente ESTILO HOME CENTER LTDA",
    #     body="Olá, Segue DANFE e XML em anexo.",
    #     url="https://www.estilohomecenter.com.br/minha-conta/meus-pedidos",
    #     model_dir=config.models_dir,
    # )
    
    sample = predict_email(
        sender="contato@google.com",
        subject="Bem-vindo ao Google",
        body="Obrigado por criar sua conta.",
        url="",
        model_dir=config.models_dir,
    )
        
    print("Sample prediction:")
    print(sample)


if __name__ == "__main__":
    main()
