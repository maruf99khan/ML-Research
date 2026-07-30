"""
Full evaluation on the test split: overall metrics, confusion matrix, and a
per-generator breakdown of the LLM-fake class (Gemini vs GPT-4o-mini vs Llama
vs Claude Haiku) -- this is the Phase 5 addition testing whether the
detectability asymmetry reported in "When Machines Lie Differently" (AI-fake
easier to catch than human-fake) replicates here, and whether it varies by
which model generated the fake.

Usage:
    python src/evaluate.py --checkpoint outputs/checkpoints/cmaf_full_best.pt
"""
import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from configs import config as cfg
from src.dataset import MultiBanFakeDetectDataset
from src.model import MultiBanFakeDetectModel


def plot_confusion_matrix(cm, labels, out_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def evaluate_full(checkpoint_path: str, output_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_model_name = cfg.FALLBACK_TEXT_MODEL_NAME if cfg.USE_FALLBACK else cfg.TEXT_MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)

    model = MultiBanFakeDetectModel().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_ds = MultiBanFakeDetectDataset(cfg.COMBINED_MANIFEST, "test", tokenizer)
    test_loader = DataLoader(test_ds, batch_size=cfg.EVAL_BATCH_SIZE, shuffle=False,
                              num_workers=cfg.NUM_WORKERS)

    all_preds, all_labels, all_ids = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["pixel_values"].to(device),
            )
            preds = logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(batch["label"].tolist())
            all_ids.extend(batch["sample_id"])

    labels_str = [cfg.ID2LABEL[i] for i in range(cfg.NUM_CLASSES)]
    report = classification_report(all_labels, all_preds, target_names=labels_str,
                                    output_dict=True, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(cfg.NUM_CLASSES)))

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "test_classification_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    plot_confusion_matrix(cm, labels_str, os.path.join(output_dir, "confusion_matrix.png"))

    print("Overall macro-F1:", report["macro avg"]["f1-score"])
    print(pd.DataFrame(report).T)

    # ---- Per-generator breakdown for the LLM-fake class ----
    manifest = pd.read_csv(cfg.COMBINED_MANIFEST)
    manifest = manifest.set_index("sample_id")
    pred_df = pd.DataFrame({"sample_id": all_ids, "pred": all_preds, "true": all_labels})
    pred_df["generator"] = pred_df["sample_id"].map(
        lambda sid: manifest.loc[sid, "generator"] if sid in manifest.index and "generator" in manifest.columns else None
    )
    llm_fake_id = cfg.LABEL2ID["llm_fake"]
    llm_subset = pred_df[pred_df["true"] == llm_fake_id]

    if llm_subset["generator"].notna().any():
        per_gen = llm_subset.groupby("generator").apply(
            lambda g: (g["pred"] == llm_fake_id).mean()
        ).rename("recall_llm_fake_detected")
        per_gen_path = os.path.join(output_dir, "per_generator_recall.csv")
        per_gen.to_csv(per_gen_path)
        print("\nPer-generator LLM-fake detection recall:")
        print(per_gen)
        print(f"Saved -> {per_gen_path}")
    else:
        print("\nNo 'generator' column found on manifest for LLM-fake rows; "
              "skipping per-generator breakdown.")

    # ---- Human-fake vs LLM-fake detectability comparison ----
    human_fake_id = cfg.LABEL2ID["human_fake"]
    human_recall = report[cfg.ID2LABEL[human_fake_id]]["recall"]
    llm_recall = report[cfg.ID2LABEL[llm_fake_id]]["recall"]
    print(f"\nHuman-Fake recall: {human_recall:.3f} | LLM-Fake recall: {llm_recall:.3f}")
    if llm_recall > human_recall:
        print("Consistent with the English-language finding that LLM-generated fake "
              "content is easier to detect than human-written fake content.")
    else:
        print("Does NOT replicate the English-language asymmetry -- worth discussing "
              "in the paper's error analysis / limitations.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=cfg.METRICS_DIR)
    args = parser.parse_args()
    evaluate_full(args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
