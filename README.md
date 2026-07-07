# BERT Fine-Tuning for Sentiment Analysis

Fine-tunes `bert-base-uncased` on the SST-2 dataset for binary sentiment classification (positive / negative). Supports both **full fine-tuning** and **LoRA (PEFT)** parameter-efficient fine-tuning.

## Results

| Mode | Accuracy | F1 | Precision | Recall |
|------|----------|----|-----------|--------|
| Full Fine-Tune | ~93% | ~93% | ~93% | ~93% |
| LoRA (PEFT) | ~91% | ~91% | ~91% | ~91% |

## Project Structure

```
Sentiment-Classification-using-BERT/
├── .github/
│   └── workflows/
│       └── ci.yml     # GitHub Actions CI workflow
├── data/
│   └── dummy.csv      # Dummy dataset for offline CPU runs
├── train.py           # Fine-tuning script (full + LoRA)
├── evaluate.py        # Detailed evaluation + plots
├── app.py             # FastAPI inference server
├── requirements.txt
├── Dockerfile
├── README.md
├── .gitignore
├── .dockerignore
└── outputs/bert_sentiment/
    ├── metrics.json   # Evaluation metrics (committed to git)
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── pr_curve.png
    └── confidence_dist.png
    # Note: best_model/ and checkpoint-* directories are ignored by git due to size

```

## Setup

```bash
pip install -r requirements.txt
```

## Training

```bash
# Full fine-tuning
python train.py --mode full

# LoRA / PEFT fine-tuning (parameter-efficient)
python train.py --mode lora
```

## Evaluation

```bash
python evaluate.py
```
Generates: confusion matrix, ROC curve, PR curve, confidence distribution, sample predictions.

## Inference API

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was absolutely fantastic!"}'
```

### Response

```json
{
  "text": "This movie was absolutely fantastic!",
  "label": "positive",
  "confidence": 0.9823,
  "probabilities": {"negative": 0.0177, "positive": 0.9823},
  "inference_time_ms": 14.3
}
```

## Docker

```bash
docker build -t bert-sentiment .
docker run -p 8000:8000 bert-sentiment
```

## Key Concepts Demonstrated

- **Transformer fine-tuning** on downstream classification task
- **LoRA (PEFT)** — only ~0.5% of parameters trained vs full fine-tuning
- **HuggingFace Trainer API** with early stopping
- **Production deployment** via FastAPI + Docker
- **Evaluation** — F1, precision, recall, ROC AUC, PR curve

## Author
    Pavithra Nagineni