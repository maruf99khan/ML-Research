"""
Training script — FINAL

CONFIRMED on Kaggle T4x2:
  Binary baseline: batch=4, freeze_text=12, freeze_image=True → macro-F1=0.8792
  Ternary target:  batch=8, freeze_text=6,  freeze_image=False (verify with memory test first)

BUGS FIXED:
  - Last grad-accum batch now always flushed
  - Per-class F1 printed every epoch
  - OOM fallback: tries requested config, falls back to safe config

Usage:
    from src.train import train, memory_test
    memory_test()   # run first to verify batch=8 fits
    train("cmaf_ternary", mode="full")
    train("text_only",    mode="text_only")
    train("image_only",   mode="image_only")
"""
import gc, json, os, random, sys
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from configs import config as cfg
from src.dataset import MultiBanFakeDetectDataset, compute_class_weights
from src.model import MultiBanFakeDetectModel


def set_seed(seed=cfg.SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def memory_test():
    """Run before training to verify batch=8 fits on T4x2."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | GPUs: {torch.cuda.device_count()}")
    print(f"Free VRAM: {torch.cuda.mem_get_info()[0]/1e9:.2f} GB")

    model = MultiBanFakeDetectModel(
        freeze_text_layers=cfg.FREEZE_TEXT_LAYERS,
        freeze_image=cfg.FREEZE_IMAGE,
    ).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    bs   = cfg.BATCH_SIZE
    ids  = torch.zeros(bs, cfg.MAX_TEXT_LEN, dtype=torch.long).to(device)
    mask = torch.ones(bs, cfg.MAX_TEXT_LEN, dtype=torch.long).to(device)
    imgs = torch.zeros(bs, 3, 224, 224).to(device)

    try:
        base = model.module if hasattr(model, 'module') else model
        with torch.no_grad():
            out = base(ids, mask, imgs)
        print(f"✅ batch={bs} fits — output shape {out.shape}")
        print(f"Free VRAM after test: {torch.cuda.mem_get_info()[0]/1e9:.2f} GB")
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print(f"❌ OOM at batch={bs} — will use fallback batch={cfg.FALLBACK_BATCH_SIZE}")
        else:
            raise
    finally:
        del model; gc.collect(); torch.cuda.empty_cache()


def train(
    run_name:    str = "cmaf_ternary",
    mode:        str = "full",
    num_epochs:  int = cfg.NUM_EPOCHS,
    resume_from: str = None,
) -> float:
    assert mode in ("full","text_only","image_only")
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Training: {run_name} | mode={mode} | device={device}")
    print(f"{'='*60}")

    ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, f"{run_name}_best.pt")
    if os.path.exists(ckpt_path) and resume_from is None:
        print(f"Checkpoint exists — skipping. Delete {ckpt_path} to retrain.")
        return None

    tokenizer  = AutoTokenizer.from_pretrained(cfg.TEXT_MODEL_NAME)
    train_ds   = MultiBanFakeDetectDataset(cfg.COMBINED_MANIFEST, "train", tokenizer)
    val_ds     = MultiBanFakeDetectDataset(cfg.COMBINED_MANIFEST, "val",   tokenizer)

    # Try requested settings, fall back on OOM
    model = None
    for ft, fi, bs in [
        (cfg.FREEZE_TEXT_LAYERS, cfg.FREEZE_IMAGE,       cfg.BATCH_SIZE),
        (cfg.FALLBACK_FREEZE_TEXT_LAYERS, cfg.FALLBACK_FREEZE_IMAGE, cfg.FALLBACK_BATCH_SIZE),
    ]:
        try:
            gc.collect(); torch.cuda.empty_cache()
            _m = MultiBanFakeDetectModel(freeze_text_layers=ft, freeze_image=fi).to(device)
            if torch.cuda.device_count() > 1:
                _m = nn.DataParallel(_m)
                print(f"Using {torch.cuda.device_count()} GPUs")
            base = _m.module if hasattr(_m,'module') else _m
            try:
                base.text_encoder.gradient_checkpointing_enable()
            except Exception as e:
                print(f"Warning: gradient_checkpointing failed: {e}")
            # Quick OOM test
            with torch.no_grad():
                dummy = torch.zeros(bs, cfg.MAX_TEXT_LEN, dtype=torch.long).to(device)
                base(dummy, torch.ones_like(dummy), torch.zeros(bs,3,224,224).to(device))
            model = _m; BATCH = bs
            if ft != cfg.FREEZE_TEXT_LAYERS:
                print(f"⚠️  Using fallback: freeze_text={ft}, freeze_image={fi}, batch={bs}")
            else:
                print(f"Config: freeze_text={ft}, freeze_image={fi}, batch={bs}")
            break
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                gc.collect(); torch.cuda.empty_cache(); continue
            raise

    if model is None:
        raise RuntimeError("Cannot load model even with fallback settings")

    if resume_from:
        ckpt = torch.load(resume_from, map_location=device)
        (model.module if hasattr(model,'module') else model).load_state_dict(ckpt['model_state_dict'])
        print(f"Resumed from {resume_from}")

    train_load = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                             num_workers=cfg.NUM_WORKERS, pin_memory=True)
    val_load   = DataLoader(val_ds,   batch_size=cfg.EVAL_BATCH_SIZE, shuffle=False,
                             num_workers=cfg.NUM_WORKERS, pin_memory=True)
    base_model = model.module if hasattr(model,'module') else model

    class_weights = compute_class_weights(cfg.COMBINED_MANIFEST).to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)
    print(f"Class weights: {class_weights.tolist()}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,}")

    optimizer   = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY,
    )
    total_steps = (len(train_load) // cfg.GRAD_ACCUM_STEPS) * num_epochs
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(cfg.WARMUP_RATIO * total_steps),
        num_training_steps=total_steps,
    )

    best_f1 = -1.0; no_improve = 0; history = []
    label_names = [cfg.ID2LABEL[i] for i in range(cfg.NUM_CLASSES)]

    for epoch in range(num_epochs):
        # TRAIN
        model.train(); epoch_loss = 0.0; optimizer.zero_grad(); last_flushed = -1
        pbar = tqdm(train_load, desc=f"Epoch {epoch+1}/{num_epochs}")

        for step, batch in enumerate(pbar):
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            imgs = batch["pixel_values"].to(device)
            labs = batch["label"].to(device)

            if mode == "text_only":    imgs = torch.zeros_like(imgs)
            elif mode == "image_only": ids = torch.zeros_like(ids); mask = torch.ones_like(mask)

            try:
                loss = criterion(model(ids, mask, imgs), labs) / cfg.GRAD_ACCUM_STEPS
                loss.backward()
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    print(f"\n⚠️  OOM at step {step} — skipping batch")
                    optimizer.zero_grad(); gc.collect(); torch.cuda.empty_cache(); continue
                raise

            if (step + 1) % cfg.GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.MAX_GRAD_NORM)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
                last_flushed = step

            epoch_loss += loss.item() * cfg.GRAD_ACCUM_STEPS
            pbar.set_postfix(loss=f"{epoch_loss/(step+1):.4f}")

        # Flush remaining if last batch wasn't flushed
        if last_flushed != step:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.MAX_GRAD_NORM)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()

        # VALIDATE
        model.eval(); all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_load:
                ids  = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                imgs = batch["pixel_values"].to(device)
                labs = batch["label"].to(device)
                if mode == "text_only":    imgs = torch.zeros_like(imgs)
                elif mode == "image_only": ids = torch.zeros_like(ids); mask = torch.ones_like(mask)
                logits = model(ids, mask, imgs)
                all_preds.extend(logits.argmax(-1).cpu().tolist())
                all_labels.extend(labs.cpu().tolist())

        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        report   = classification_report(all_labels, all_preds,
                                          target_names=label_names,
                                          output_dict=True, zero_division=0)
        cm = confusion_matrix(all_labels, all_preds, labels=list(range(cfg.NUM_CLASSES)))

        avg_loss = epoch_loss / len(train_load)
        print(f"\nEpoch {epoch+1}: macro_f1={macro_f1:.4f} | train_loss={avg_loss:.4f}")
        for cls in label_names:
            if cls in report:
                r = report[cls]
                print(f"  {cls}: P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1-score']:.3f} n={int(r['support'])}")
        print(f"  Confusion matrix:\n{cm}")

        history.append({"epoch": epoch+1, "macro_f1": macro_f1, "train_loss": avg_loss})

        if macro_f1 > best_f1:
            best_f1 = macro_f1; no_improve = 0
            torch.save({
                "model_state_dict":  base_model.state_dict(),
                "epoch":             epoch + 1,
                "val_macro_f1":      macro_f1,
                "mode":              mode,
                "per_class_report":  report,
                "confusion_matrix":  cm.tolist(),
            }, ckpt_path)
            print(f"  ✅ New best saved → {ckpt_path}")
        else:
            no_improve += 1
            if no_improve >= cfg.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    with open(os.path.join(cfg.LOG_DIR, f"{run_name}_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest val macro-F1: {best_f1:.4f}")
    print(f"\n⚠️  CLICK SAVE VERSION IN KAGGLE NOW")
    del model; gc.collect(); torch.cuda.empty_cache()
    return best_f1
