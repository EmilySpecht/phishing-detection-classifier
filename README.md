# Phishing Detection Classifier

This project implements a modular phishing detection pipeline with three trained components:

1. An email-text branch based on TF-IDF + XGBoost.
2. A URL branch based on a character-level CNN in PyTorch.
3. A late-fusion XGBoost model combining both probabilities.

## Project structure

- phishing_detector/: package with modular training and inference code.
- data/: runtime data storage.
- datasets/: expected dataset location for CEAS_08 and PhishTank-style CSV files.
- models/: trained model artifacts.
- notebooks/: optional experiments.

## Quick start

source .venv/bin/activate

Install dependencies:

```bash
pip install -r requirements.txt
```

Train and evaluate the full pipeline:

```bash
python main.py
```

Run a single prediction:

```bash
python -c "from phishing_detector.predict import predict_email; print(predict_email('alice@example.com','Invoice','Please review the invoice','https://example.com/login'))"
```

## Notes

If the expected dataset files are not present, the loader will generate a small synthetic dataset so the project remains runnable.
