"""
BERT Fine-Tuning for Sentiment Analysis
========================================
Fine-tunes bert-base-uncased on SST-2 / IMDb using HuggingFace Transformers.
Supports LoRA (PEFT) fine-tuning for parameter-efficient training.

Usage:
    python train.py --mode full        # full fine-tuning
    python train.py --mode lora        # LoRA / PEFT fine-tuning
    python train.py --mode eval        # evaluate saved model
"""

import os
import argparse
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_NAME   = "bert-base-uncased"
OUTPUT_DIR   = "./outputs/bert_sentiment"
DATASET_NAME = "nyu-mll/glue"        # HuggingFace SST-2 (binary: positive / negative)
MAX_LENGTH   = 64
BATCH_SIZE   = 4
NUM_EPOCHS   = 1
LR           = 2e-5
SEED         = 42
LABELS       = ["negative", "positive"]


# ── Tokenization ─────────────────────────────────────────────────────────────
def tokenize(examples, tokenizer):
    return tokenizer(
        examples["sentence"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,   # dynamic padding via DataCollatorWithPadding
    )


# ── Metrics ──────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy":  accuracy_score(labels, preds),
        "f1":        f1_score(labels, preds, average="weighted"),
        "precision": precision_score(labels, preds, average="weighted"),
        "recall":    recall_score(labels, preds, average="weighted"),
    }


# ── LoRA setup ───────────────────────────────────────────────────────────────
def apply_lora(model):
    """Apply LoRA adapters via peft library."""
    try:
        from peft import get_peft_model, LoraConfig, TaskType
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,
            lora_alpha=16,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model
    except ImportError:
        logger.warning("peft not installed — falling back to full fine-tuning.")
        return model


# ── Plotting ─────────────────────────────────────────────────────────────────
def plot_confusion_matrix(labels, preds, save_path):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=LABELS, yticklabels=LABELS,
    )
    plt.title("Confusion Matrix — BERT Sentiment")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Confusion matrix saved → {save_path}")


def plot_training_loss(log_history, save_path):
    train_loss = [(e["epoch"], e["loss"]) for e in log_history if "loss" in e]
    eval_loss  = [(e["epoch"], e["eval_loss"]) for e in log_history if "eval_loss" in e]
    if not train_loss:
        return
    epochs_t, losses_t = zip(*train_loss)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_t, losses_t, label="Train Loss", marker="o")
    if eval_loss:
        epochs_e, losses_e = zip(*eval_loss)
        plt.plot(epochs_e, losses_e, label="Val Loss", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    logger.info(f"Loss curve saved → {save_path}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main(mode: str = "full"):
    torch.manual_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model_path = os.path.join(OUTPUT_DIR, "best_model")
    if mode == "eval" and not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"No saved model found at {model_path}. "
            "Run training first: python train.py --mode full"
        )

    logger.info(f"Loading dataset: {DATASET_NAME}")

    dataset = load_dataset("nyu-mll/glue", "sst2")

    if mode == "eval":
        logger.info(f"Loading saved model from {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    else:
        logger.info(f"Loading tokenizer: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        logger.info(f"Loading model: {MODEL_NAME}")
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=2,
            id2label={0: "negative", 1: "positive"},
            label2id={"negative": 0, "positive": 1},
        )
        if mode == "lora":
            logger.info("Applying LoRA adapters (PEFT)...")
            model = apply_lora(model)

    tokenized = dataset.map(
        lambda x: tokenize(x, tokenizer),
        batched=True,
        remove_columns=["sentence", "idx"],
    )
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        learning_rate=LR,
        weight_decay=0.01,
        warmup_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        seed=SEED,
        fp16=torch.cuda.is_available(),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    if mode in ("full", "lora"):
        logger.info("Starting training...")
        trainer.train()
        trainer.save_model(f"{OUTPUT_DIR}/best_model")
        tokenizer.save_pretrained(f"{OUTPUT_DIR}/best_model")

        # Plot loss curve
        plot_training_loss(
            trainer.state.log_history,
            f"{OUTPUT_DIR}/loss_curve.png",
        )

    # Evaluate
    logger.info("Evaluating on validation set...")
    preds_output = trainer.predict(tokenized["validation"])
    preds  = np.argmax(preds_output.predictions, axis=-1)
    labels = preds_output.label_ids

    report = classification_report(labels, preds, target_names=LABELS)
    logger.info(f"\nClassification Report:\n{report}")

    metrics = {
        "accuracy":  float(accuracy_score(labels, preds)),
        "f1":        float(f1_score(labels, preds, average="weighted")),
        "precision": float(precision_score(labels, preds, average="weighted")),
        "recall":    float(recall_score(labels, preds, average="weighted")),
        "mode":      mode,
    }
    with open(f"{OUTPUT_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics: {metrics}")

    plot_confusion_matrix(labels, preds, f"{OUTPUT_DIR}/confusion_matrix.png")
    logger.info("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full", "lora", "eval"], default="full")
    args = parser.parse_args()
    main(args.mode)
