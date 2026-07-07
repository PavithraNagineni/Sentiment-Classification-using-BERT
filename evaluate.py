"""
evaluate.py — Detailed Evaluation & Visualizations
====================================================
Generates confusion matrix, ROC curve, PR curve,
per-class metrics, and sample predictions.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve,
)
import json, os, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

MODEL_PATH = os.path.abspath("./outputs/bert_sentiment/best_model")
OUTPUT_DIR = os.path.abspath("./outputs/bert_sentiment")
LABELS     = ["negative", "positive"]
MAX_LENGTH = 128
BATCH_SIZE = 32
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_predictions(dataset, tokenizer, model):
    all_preds, all_probs, all_labels = [], [], []
    for i in range(0, len(dataset), BATCH_SIZE):
        batch = dataset[i: i + BATCH_SIZE]
        inputs = tokenizer(
            batch["sentence"], return_tensors="pt",
            truncation=True, max_length=MAX_LENGTH, padding=True,
        ).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            probs   = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
        all_preds.extend(np.argmax(probs, axis=-1).tolist())
        all_probs.extend(probs[:, 1].tolist())   # prob of positive class
        all_labels.extend(batch["label"])
    return np.array(all_preds), np.array(all_probs), np.array(all_labels)


def plot_roc(labels, probs, save_path):
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc     = auc(fpr, tpr)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — BERT Sentiment")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    return roc_auc


def plot_pr_curve(labels, probs, save_path):
    precision, recall, _ = precision_recall_curve(labels, probs)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, color="steelblue", lw=2, label=f"PR (AUC = {pr_auc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve — BERT Sentiment")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confidence_dist(preds, probs, labels, save_path):
    correct   = probs[preds == labels]
    incorrect = probs[preds != labels]
    plt.figure(figsize=(8, 5))
    plt.hist(correct,   bins=30, alpha=0.7, color="green",  label="Correct")
    plt.hist(incorrect, bins=30, alpha=0.7, color="red",    label="Incorrect")
    plt.xlabel("Confidence (P(positive))")
    plt.ylabel("Count")
    plt.title("Confidence Distribution — Correct vs Incorrect")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    if not os.path.isdir(MODEL_PATH):
        raise FileNotFoundError(
            f"No saved model found at {MODEL_PATH}. "
            "Run training first: python train.py --mode full"
        )

    print("Loading model and dataset...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)
    model.eval()
    if os.path.exists("data/dummy.csv"):
        dataset = load_dataset("csv", data_files={"validation": "data/dummy.csv"})["validation"]
    else:
        dataset = load_dataset("nyu-mll/glue", "sst2")["validation"]


    print("Running inference...")
    preds, probs, labels = get_predictions(dataset, tokenizer, model)

    # Classification report
    report = classification_report(labels, preds, target_names=LABELS)
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(report)

    # Plots
    roc_auc = plot_roc(labels, probs, f"{OUTPUT_DIR}/roc_curve.png")
    plot_pr_curve(labels, probs, f"{OUTPUT_DIR}/pr_curve.png")
    plot_confidence_dist(preds, probs, labels, f"{OUTPUT_DIR}/confidence_dist.png")

    # Confusion matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABELS, yticklabels=LABELS)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=150)
    plt.close()

    # Sample predictions
    print("\nSample Predictions:")
    samples = [
        "This film was absolutely brilliant and touching.",
        "Terrible waste of time. Worst movie I've seen.",
        "The acting was okay but the story was boring.",
        "Outstanding performance by the entire cast!",
        "I wouldn't recommend this to anyone.",
    ]
    inputs = tokenizer(samples, return_tensors="pt",
                       truncation=True, max_length=MAX_LENGTH, padding=True).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        sample_probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
    for text, prob in zip(samples, sample_probs):
        label = LABELS[np.argmax(prob)]
        conf  = np.max(prob)
        print(f"  [{label.upper():8s} {conf:.2%}] {text}")

    print(f"\nAll plots saved to {OUTPUT_DIR}/")
    print(f"ROC AUC: {roc_auc:.4f}")


if __name__ == "__main__":
    main()
