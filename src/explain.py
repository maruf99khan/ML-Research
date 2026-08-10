"""
Integrated Gradients explainability — FINAL v4

BUGS FIXED vs v3:
1. Model forced to CPU explicitly — GPU has no room for n_steps=50 IG intermediate
   gradients alongside the full model. Do NOT set CUDA_VISIBLE_DEVICES (breaks GPU
   for other cells) — instead load model directly to cpu device here.
2. DataParallel key stripping — checkpoint was saved from T4x2 (nn.DataParallel),
   so all keys have 'module.' prefix. Must strip before load_state_dict or it crashes.
3. Manifest existence guard — clear error if build_manifest() was not called first.

Produces per prediction:
  - Top-10 attributed text tokens (L2 norm over embedding dim, normalized)
  - Patch-level image attribution (mean abs per spatial position, saved as .npy)

Outputs:
  - ig_examples.csv                  : qualitative examples (10 per class)
  - human_plausibility_template.csv  : 40 samples for manual annotation
  - <sample_id>_patch_attr.npy       : per-sample image patch attributions

Usage:
    from src.explain import run_ig
    run_ig(checkpoint_path="/kaggle/input/datasets/maruf99khan/multibanfakedetect-checkpoints/cmaf_ternary_best.pt")
"""
import gc, os, random, sys
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from configs import config as cfg
from src.dataset import MultiBanFakeDetectDataset
from src.model import MultiBanFakeDetectModel, ForwardWrapper


def _strip_dataparallel(state_dict: dict) -> dict:
    """Remove 'module.' prefix added by nn.DataParallel when saving on multi-GPU."""
    return {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}


def run_ig(checkpoint_path: str = None, split: str = "test") -> None:
    if checkpoint_path is None:
        checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, "cmaf_ternary_best.pt")

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        print("   Add maruf99khan/multibanfakedetect-checkpoints as a notebook input dataset.")
        return

    if not os.path.exists(cfg.COMBINED_MANIFEST):
        print("Manifest not found — run build_manifest() first in session setup.")
        return

    try:
        from captum.attr import IntegratedGradients
    except ImportError:
        print("captum not installed — run: pip install captum -q")
        return

    # FORCE CPU — model fills ~14GB VRAM leaving no room for 50 IG gradient steps.
    # Do NOT set CUDA_VISIBLE_DEVICES — that hides GPU from the whole process.
    device = torch.device("cpu")
    print("Running IG on CPU (required — GPU has no room for IG gradient accumulation)")
    print("Expected runtime: ~30-40 min for 30 examples (10 per class)")

    tokenizer = AutoTokenizer.from_pretrained(cfg.TEXT_MODEL_NAME)
    dataset   = MultiBanFakeDetectDataset(cfg.COMBINED_MANIFEST, split, tokenizer)

    # Load model to CPU, strip DataParallel 'module.' prefix from keys
    model = MultiBanFakeDetectModel()
    ckpt  = torch.load(checkpoint_path, map_location="cpu")
    state = _strip_dataparallel(ckpt["model_state_dict"])
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    wrapper = ForwardWrapper(model).to(device)
    wrapper.eval()

    os.makedirs(cfg.EXPLAIN_DIR, exist_ok=True)
    ig_rows   = []
    per_class = {0: 0, 1: 0, 2: 0}
    target_n  = cfg.IG_NUM_QUALITATIVE_EXAMPLES_PER_CLASS  # 10

    indices = list(range(len(dataset)))
    random.shuffle(indices)

    for idx in indices:
        if all(v >= target_n for v in per_class.values()):
            break
        ex    = dataset[idx]
        label = int(ex["label"])
        if per_class[label] >= target_n:
            continue

        with torch.no_grad():
            logits = model(
                ex["input_ids"].unsqueeze(0).to(device),
                ex["attention_mask"].unsqueeze(0).to(device),
                ex["pixel_values"].unsqueeze(0).to(device),
            )
            pred = int(logits.argmax(-1).item())

        try:
            embeds = model.text_encoder.get_input_embeddings()(
                ex["input_ids"].unsqueeze(0).to(device)
            ).detach().requires_grad_(True)
            pixels = ex["pixel_values"].unsqueeze(0).to(device).clone().requires_grad_(True)
            attn   = ex["attention_mask"].unsqueeze(0).to(device)

            ig = IntegratedGradients(wrapper)
            attrs, delta = ig.attribute(
                inputs=(embeds, pixels),
                baselines=(torch.zeros_like(embeds), torch.zeros_like(pixels)),
                additional_forward_args=(attn,),
                target=pred,
                n_steps=cfg.IG_N_STEPS,
                return_convergence_delta=True,
            )
            text_attr, img_attr = attrs

            token_scores = text_attr.norm(dim=-1).squeeze(0).detach().cpu().numpy()
            token_scores = token_scores / (np.abs(token_scores).max() + 1e-8)
            tokens       = tokenizer.convert_ids_to_tokens(ex["input_ids"].tolist())
            top_idx      = np.argsort(-np.abs(token_scores))[:10]
            top_tokens   = [
                (tokens[i], round(float(token_scores[i]), 3))
                for i in top_idx
                if tokens[i] not in tokenizer.all_special_tokens
            ]

            patch_scores = img_attr.abs().mean(dim=1).squeeze(0).detach().cpu().numpy()
            npy_path     = os.path.join(cfg.EXPLAIN_DIR, f"{ex['sample_id']}_patch_attr.npy")
            np.save(npy_path, patch_scores)

            ig_rows.append({
                "sample_id":       ex["sample_id"],
                "true_label":      cfg.ID2LABEL[label],
                "predicted_label": cfg.ID2LABEL[pred],
                "correct":         label == pred,
                "top_tokens":      str(top_tokens),
                "ig_delta":        float(delta.abs().item()),
                "patch_attr_path": npy_path,
                "text_preview":    ex["text"][:100],
            })
            per_class[label] += 1
            print(f"  [{sum(per_class.values())}/{target_n*3}] "
                  f"{cfg.ID2LABEL[label]} -> {cfg.ID2LABEL[pred]} "
                  f"({'correct' if label==pred else 'wrong'}) | delta={delta.abs().item():.4f}")

        except Exception as e:
            print(f"  IG failed for idx {idx}: {e}")
            continue

    ig_df   = pd.DataFrame(ig_rows)
    ig_path = os.path.join(cfg.EXPLAIN_DIR, "ig_examples.csv")
    ig_df.to_csv(ig_path, index=False, encoding="utf-8")
    print(f"\nIG examples ({len(ig_df)}) -> {ig_path}")

    n_check  = min(cfg.IG_HUMAN_CHECK_SAMPLE_SIZE, len(ig_df))
    plaus    = ig_df.sample(n_check, random_state=cfg.SEED).copy()
    plaus["annotator_1_plausible_1to5"] = ""
    plaus["annotator_2_plausible_1to5"] = ""
    plaus["notes"]                       = ""
    plaus_path = os.path.join(cfg.EXPLAIN_DIR, "human_plausibility_template.csv")
    plaus.to_csv(plaus_path, index=False, encoding="utf-8")
    print(f"Plausibility template ({n_check} samples) -> {plaus_path}")
    print("\nAnnotation guide:")
    print("  1 = attribution makes no sense")
    print("  3 = partially sensible")
    print("  5 = clearly highlights suspicious / fabricated content")
    print("\nHave 2 annotators fill independently, then compute weighted kappa.")

    del model, wrapper
    gc.collect()
