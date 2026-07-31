"""
Evaluation script — FINAL

Runs full test-set evaluation for all 3 checkpoints:
  cmaf_ternary (full model)
  text_only    (ablation)
  image_only   (ablation)

Outputs per checkpoint:
  - macro-F1, per-class P/R/F1
  - confusion matrix (saved as JSON + printed)
  - per-generator LLM-fake recall
  - human-fake vs LLM-fake detectability comparison

Usage:
    from src.evaluate import evaluate_all
    evaluate_all()
"""
import gc, json, os, sys
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from configs import config as cfg
from src.dataset import MultiBanFakeDetectDataset
from src.model import MultiBanFakeDetectModel


def evaluate_checkpoint(run_name: str, mode: str = "full") -> dict | None:
    ckpt_path = os.path.join(cfg.CHECKPOINT_DIR, f"{run_name}_best.pt")
    if not os.path.exists(ckpt_path):
        print(f"  ⚠️  No checkpoint for {run_name} — skipping")
        return None

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg.TEXT_MODEL_NAME)
    test_ds   = MultiBanFakeDetectDataset(cfg.COMBINED_MANIFEST, "test", tokenizer)
    test_load = DataLoader(test_ds, batch_size=cfg.EVAL_BATCH_SIZE,
                            shuffle=False, num_workers=cfg.NUM_WORKERS)

    model = MultiBanFakeDetectModel().to(device)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_preds, all_labels, all_ids = [], [], []
    with torch.no_grad():
        for batch in test_load:
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
            all_preds.extend(logits.argmax(-1).cpu().tolist())
            all_labels.extend(labs.cpu().tolist())
            all_ids.extend(batch["sample_id"])

    label_names = [cfg.ID2LABEL[i] for i in range(cfg.NUM_CLASSES)]
    report = classification_report(all_labels, all_preds,
                                    target_names=label_names,
                                    output_dict=True, zero_division=0)
    cm     = confusion_matrix(all_labels, all_preds,
                               labels=list(range(cfg.NUM_CLASSES)))

    print(f"\n{'='*60}")
    print(f"{run_name} — macro-F1: {report['macro avg']['f1-score']:.4f}")
    print(f"{'='*60}")
    for cls in label_names:
        if cls in report:
            r = report[cls]
            print(f"  {cls}: P={r['precision']:.3f} R={r['recall']:.3f} "
                  f"F1={r['f1-score']:.3f} n={int(r['support'])}")
    print(f"\n  Confusion matrix:\n{cm}")

    # Per-generator LLM-fake recall
    mdf     = pd.read_csv(cfg.COMBINED_MANIFEST).set_index("sample_id")
    pred_df = pd.DataFrame({"sample_id": all_ids, "pred": all_preds, "true": all_labels})
    pred_df["generator"] = pred_df["sample_id"].map(
        lambda sid: mdf.loc[sid, "generator"]
        if sid in mdf.index and "generator" in mdf.columns
        and pd.notna(mdf.loc[sid, "generator"]) else None
    )
    llm_id  = cfg.LABEL2ID["llm_fake"]
    hf_id   = cfg.LABEL2ID["human_fake"]
    llm_sub = pred_df[(pred_df["true"] == llm_id) & pred_df["generator"].notna()]

    if len(llm_sub) > 0:
        per_gen = llm_sub.groupby("generator").apply(
            lambda g: (g["pred"] == llm_id).mean()
        ).rename("recall_llm_fake")
        print(f"\n  Per-generator LLM-fake recall:")
        print(per_gen)
        per_gen.to_csv(os.path.join(cfg.METRICS_DIR, f"{run_name}_per_generator.csv"))

    # Human-fake vs LLM-fake asymmetry
    hf_recall  = report.get(cfg.ID2LABEL[hf_id],  {}).get("recall", 0)
    llm_recall = report.get(cfg.ID2LABEL[llm_id], {}).get("recall", 0)
    print(f"\n  Human-Fake recall : {hf_recall:.3f}")
    print(f"  LLM-Fake recall   : {llm_recall:.3f}")
    if llm_recall > hf_recall:
        print("  → LLM-Fake easier to detect (consistent with English findings)")
    else:
        print("  → Human-Fake easier to detect (does NOT replicate English asymmetry)")

    # Save results
    out = {
        "run_name":         run_name,
        "report":           report,
        "confusion_matrix": cm.tolist(),
        "hf_recall":        hf_recall,
        "llm_recall":       llm_recall,
    }
    with open(os.path.join(cfg.METRICS_DIR, f"{run_name}_test_results.json"), "w") as f:
        json.dump(out, f, indent=2)

    del model; gc.collect(); torch.cuda.empty_cache()
    return out


def evaluate_all() -> dict:
    results = {}
    for run_name, mode in [
        ("cmaf_ternary", "full"),
        ("text_only",    "text_only"),
        ("image_only",   "image_only"),
    ]:
        r = evaluate_checkpoint(run_name, mode)
        if r:
            results[run_name] = r

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Macro-F1':<12} {'Real F1':<12} {'HFake F1':<12} {'LLM F1'}")
    for run, r in results.items():
        rep = r["report"]
        print(f"  {run:<18} "
              f"{rep['macro avg']['f1-score']:.4f}      "
              f"{rep.get('real',{}).get('f1-score',0):.4f}      "
              f"{rep.get('human_fake',{}).get('f1-score',0):.4f}      "
              f"{rep.get('llm_fake',{}).get('f1-score',0):.4f}")

    print(f"\n⚠️  CLICK SAVE VERSION IN KAGGLE NOW")
    return results
