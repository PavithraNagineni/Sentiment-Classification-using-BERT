# BERT Fine-Tuning for Sentiment Analysis

Fine-tunes `bert-base-uncased` on the SST-2 dataset for binary sentiment classification (positive / negative). Supports both **full fine-tuning** and **LoRA (PEFT)** parameter-efficient fine-tuning.

---

## 🚀 Live Demo

### 🌐 Frontend (Vercel)
https://sentiment-classification-using-bert.vercel.app/

### ⚡ Backend API (Render)
https://sentiment-classification-using-bert-71tb.onrender.com

---

## Results

| Mode | Accuracy | F1 | Precision | Recall |
|------|----------|----|-----------|--------|
| Full Fine-Tune | ~93% | ~93% | ~93% | ~93% |
| LoRA (PEFT) | ~91% | ~91% | ~91% | ~91% |

---

## Project Structure

```text
Sentiment-Classification-using-BERT/
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/
│   └── dummy.csv
├── train.py
├── evaluate.py
├── app.py
├── requirements.txt
├── Dockerfile
├── README.md
├── .gitignore
├── .dockerignore
└── outputs/bert_sentiment/
    ├── metrics.json
    ├── confusion_matrix.png
    ├── roc_curve.png
    ├── pr_curve.png
    └── confidence_dist.png
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Training

### Full Fine-Tuning

```bash
python train.py --mode full
```

### LoRA (PEFT)

```bash
python train.py --mode lora
```

---

## Evaluation

```bash
python evaluate.py
```

Generates:

- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- Confidence Distribution
- Sample Predictions

---

## Inference API

Run locally:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Local API

```
http://localhost:8000/predict
```

### Deployed API

```
https://sentiment-classification-using-bert-71tb.onrender.com/predict
```

### Example Request

```bash
curl -X POST https://sentiment-classification-using-bert-71tb.onrender.com/predict \
-H "Content-Type: application/json" \
-d '{"text":"This movie was absolutely fantastic!"}'
```

### Response

```json
{
  "text": "This movie was absolutely fantastic!",
  "label": "positive",
  "confidence": 0.9823,
  "probabilities": {
    "negative": 0.0177,
    "positive": 0.9823
  },
  "inference_time_ms": 14.3
}
```

---

## Docker

```bash
docker build -t bert-sentiment .
docker run -p 8000:8000 bert-sentiment
```

---

## Key Concepts Demonstrated

- Transformer fine-tuning using **BERT**
- Parameter-efficient fine-tuning with **LoRA (PEFT)**
- Hugging Face **Trainer API**
- Early Stopping
- Production-ready **FastAPI REST API**
- Docker Containerization
- Deployment on **Render**
- Frontend Deployment on **Vercel**
- Evaluation using Accuracy, Precision, Recall, F1-Score, ROC-AUC, and PR Curve

---

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- PEFT (LoRA)
- FastAPI
- Docker
- Vercel
- Render
- Scikit-learn
- Matplotlib

---

## Output Demo
 <img width="1872" height="952" alt="Screenshot 2026-07-11 100525" src="https://github.com/user-attachments/assets/bb6a8c66-d096-460e-8219-13b396a9f2d7" />

---


## Author

**Pavithra Nagineni**

GitHub: https://github.com/PavithraNagineni
