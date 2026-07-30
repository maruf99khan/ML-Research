"""
Training script for MultiBanFakeDetect.

CONFIRMED SETTINGS (Kaggle T4, 14.56GB VRAM):
  BATCH_SIZE=4, GRAD_ACCUM_STEPS=8, FREEZE_TEXT_LAYERS=12, FREEZE_IMAGE=True
  gradient_checkpointing=True
  Binary baseline result: macro_F1=0.8792 (epoch 3)

Modes:
  full       — text + image (main model)
  text_only  — image zeroed out (ablation)
  image_only — text zeroed out (ablation)

FIXES vs v1:
- Last grad-accum batch now always flushed after epoch loop
- Per-class F1 printed every epoch (critical for 3-class monitoring)
- classification_report saved to JSON every epoch for analysis later

Usage:
    from src.train import train
    train(run_name="cmaf_ternary", mode="full", num_epochs=10)
"""
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (f1_score, confusion_matrix,
                              classification_report)
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from configs import config as cfg
from src.dataset import MultiBanFakeDetectDataset, compute_class_weights
from src.model import MultiBanFakeDetectModel


def set_seed(seed: int = cfg.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(
    run_name:    str = "cmaf_full",
    mode:        str = "full",
    num_epochs:  int = cfg.NUM_EPOCHS,
    resume_from: str = None,
):
    assert mode in ("full", "text_only", "image_only"), \
        f"mode must be full/text_only/image_only, got: {mode}"

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Mode: {mode} | Run: {run_name}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.TEXT_MODEL_NAME)

    train_ds = MultiBanFakeDetectDataset(cfg.COMBINED_MANIFEST, "train", tokenizer)
    val_ds   = MultiBanFakeDetectDataset(cfg.COMBINED_MANIFEST, "val",   tokenizer)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
        num_workers=cfg.NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.EVAL_BATCH_SIZE, shuffle=False,
        num_workers=cfg.NUM_WORKERS, pin_memory=True,
    )

    model = MultiBanFakeDetectModel(
        freeze_text_layers=cfg.FREEZE_TEXT_LAYERS,
        freeze_image=cfg.FREEZE_IMAGE,
    ).to(device)
    model.text_encoder.gradient_checkpointing_enable()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,}")

    if resume_from:
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Resumed from {resume_from} "
              f"(epoch {ckpt['epoch']}, F1={ckpt['val_macro_f1']:.4f})")

    class_weights = compute_class_weights(cfg.COMBINED_MANIFEST).to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)
    print(f"Class weights: {class_weights.tolist()}")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY,
    )
    total_steps = (len(train_loader) // cfg.GRAD_ACCUM_STEPS) * num_epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(cfg.WARMUP_RATIO * total_steps),
        num_training_steps=total_steps,
    )

    best_f1    = -1.0
    no_improve = 0
    history    = []

    for epoch in range(num_epochs):
        # ── TRAIN ──────────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for step, batch in enumerate(pbar):
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            imgs = batch["pixel_values"].to(device)
            labs = batch["label"].to(device)

            if mode == "text_only":
                imgs = torch.zeros_like(imgs)
            elif mode == "image_only":
                ids  = torch.zeros_like(ids)
                mask = torch.ones_like(mask)

            logits = model(ids, mask, imgs)
            loss   = criterion(logits, labs) / cfg.GRAD_ACCUM_STEPS
            loss.backward()

            if (step + 1) % cfg.GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * cfg.GRAD_ACCUM_STEPS
            pbar.set_postfix(loss=f"{epoch_loss/(step+1):.4f}")

        # FIX: flush any remaining accumulated gradients at end of epoch
        if (step + 1) % cfg.GRAD_ACCUM_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # ── VALIDATE ───────────────────────────────────────────────────────
        model.eval()
        all_preds, all_labels = [], []
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                ids  = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                imgs = batch["pixel_values"].to(device)
                labs = batch["label"].to(device)

                if mode == "text_only":
                    imgs = torch.zeros_like(imgs)
                elif mode == "image_only":
                    ids  = torch.zeros_like(ids)
                    mask = torch.ones_like(mask)

                logits    = model(ids, mask, imgs)
                val_loss += criterion(logits, labs).item() * labs.size(0)
                all_preds.extend(logits.argmax(-1).cpu().tolist())
                all_labels.extend(labs.cpu().tolist())

        val_loss /= len(all_labels)
        macro_f1  = f1_score(all_labels, all_preds, average="macro")
        cm        = confusion_matrix(all_labels, all_preds,
                                     labels=list(range(cfg.NUM_CLASSES)))

        # FIX: print per-class F1 every epoch
        label_names = [cfg.ID2LABEL[i] for i in range(cfg.NUM_CLASSES)]
        report      = classification_report(
            all_labels, all_preds,
            target_names=label_names,
            output_dict=True,
            zero_division=0,
        )
        print(f"\nEpoch {epoch+1}: val_loss={val_loss:.4f} | macro_f1={macro_f1:.4f}")
        for cls in label_names:
            if cls in report:
                print(f"  {cls}: P={report[cls]['precision']:.3f} "
                      f"R={report[cls]['recall']:.3f} "
                      f"F1={report[cls]['f1-score']:.3f} "
                      f"n={report[cls]['support']}")
        print("Confusion matrix:\n", cm)

        history.append({
            "epoch":      epoch + 1,
            "train_loss": epoch_loss / len(train_loader),
            "val_loss":   val_loss,
            "macro_f1":   macro_f1,
            "per_class":  {cls: report[cls] for cls in label_names if cls in report},
        })

        if macro_f1 > best_f1:
            best_f1    = macro_f1
            no_improve = 0
            ckpt_path  = os.path.join(cfg.CHECKPOINT_DIR, f"{run_name}_best.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch":            epoch + 1,
                "val_macro_f1":     macro_f1,
                "mode":             mode,
                "per_class_report": report,
            }, ckpt_path)
            print(f"  ✓ New best saved → {ckpt_path}")
        else:
            no_improve += 1
            if no_improve >= cfg.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch+1} "
                      f"(no improvement for {cfg.EARLY_STOPPING_PATIENCE} epochs)")
                break

    log_path = os.path.join(cfg.LOG_DIR, f"{run_name}_history.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"\nBest val macro-F1: {best_f1:.4f}")
    print(f"History saved → {log_path}")
    return best_f1
