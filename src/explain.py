"""
Integrated Gradients explainability — FINAL

Produces per prediction:
  - Top-10 attributed text tokens (L2 norm over embedding dim, normalized)
  - Patch-level image attribution (mean abs per spatial position, saved as .npy)

Outputs:
  - ig_examples.csv        : qualitative examples (10 per class)
  - human_plausibility_template.csv : 40 samples for manual annotation

Usage:
    from src.explain import run_ig
    run_ig()
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


def run_ig(checkpoint_path: str = None, split: str = "test") -> None:
    if checkpoint_path is None:
        checkpoint_path = os.path.join(cfg.CHECKPOINT_DIR, "cmaf_ternary_best.pt")

    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return

    try:
        from captum.attr import IntegratedGradients
    except ImportError:
        print("❌ captum not installed — run: pip install captum --break-system-packages")
        return

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg.TEXT_MODEL_NAME)
    dataset   = MultiBanFakeDetectDataset(cfg.COMBINED_MANIFEST, split, tokenizer)

    model = MultiBanFakeDetectModel().to(device)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    wrapper = ForwardWrapper(model).to(device)
    wrapper.eval()

    os.makedirs(cfg.EXPLAIN_DIR, exist_ok=True)
    ig_rows   = []
    per_class = {0: 0, 1: 0, 2: 0}
    target_n  = cfg.IG_NUM_QUALITATIVE_EXAMPLES_PER_CLASS

    indices = list(range(len(dataset)))
    random.shuffle(indices)

    for idx in indices:
        if all(v >= target_n for v in per_class.values()):
            break
        ex    = dataset[idx]
        label = int(ex["label"])
        if per_class[label] >= target_n:
            continue

        # Get prediction
        with torch.no_grad():
            logits = model(
                ex["input_ids"].unsqueeze(0).to(device),
                ex["attention_mask"].unsqueeze(0).to(device),
                ex["pixel_values"].unsqueeze(0).to(device),
            )
            pred = int(logits.argmax(-1).item())

        # Integrated Gradients
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

            # Token scores
            token_scores = text_attr.norm(dim=-1).squeeze(0).detach().cpu().numpy()
            token_scores = token_scores / (np.abs(token_scores).max() + 1e-8)
            tokens       = tokenizer.convert_ids_to_tokens(ex["input_ids"].tolist())
            top_idx      = np.argsort(-np.abs(token_scores))[:10]
            top_tokens   = [
                (tokens[i], round(float(token_scores[i]), 3))
                for i in top_idx
                if tokens[i] not in tokenizer.all_special_tokens
            ]

            # Image patch scores
            patch_scores = img_attr.abs().mean(dim=1).squeeze(0).detach().cpu().numpy()
            npy_path     = os.path.join(cfg.EXPLAIN_DIR, f"{ex['sample_id']}_patch_attr.npy")
            np.save(npy_path, patch_scores)

            ig_rows.append({
                "sample_id":        ex["sample_id"],
                "true_label":       cfg.ID2LABEL[label],
                "predicted_label":  cfg.ID2LABEL[pred],
                "correct":          label == pred,
                "top_tokens":       str(top_tokens),
                "ig_delta":         float(delta.abs().item()),
                "patch_attr_path":  npy_path,
                "text_preview":     ex["text"][:100],
            })
            per_class[label] += 1
            print(f"  [{sum(per_class.values())}/{target_n*3}] "
                  f"{cfg.ID2LABEL[label]} → {cfg.ID2LABEL[pred]} "
                  f"({'✓' if label==pred else '✗'}) | delta={delta.abs().item():.4f}")

        except Exception as e:
            print(f"  IG failed for idx {idx}: {e}")
            continue

    # Save qualitative examples
    ig_df    = pd.DataFrame(ig_rows)
    ig_path  = os.path.join(cfg.EXPLAIN_DIR, "ig_examples.csv")
    ig_df.to_csv(ig_path, index=False, encoding="utf-8")
    print(f"\n✅ IG examples ({len(ig_df)}) → {ig_path}")

    # Human plausibility check template
    n_check  = min(cfg.IG_HUMAN_CHECK_SAMPLE_SIZE, len(ig_df))
    plaus    = ig_df.sample(n_check, random_state=cfg.SEED).copy()
    plaus["annotator_1_plausible_1to5"] = ""
    plaus["annotator_2_plausible_1to5"] = ""
    plaus["notes"]                       = ""
    plaus_path = os.path.join(cfg.EXPLAIN_DIR, "human_plausibility_template.csv")
    plaus.to_csv(plaus_path, index=False, encoding="utf-8")
    print(f"✅ Plausibility template ({n_check} samples) → {plaus_path}")
    print("\nAnnotation guide:")
    print("  1 = attribution makes no sense (random words highlighted)")
    print("  3 = partially sensible")
    print("  5 = clearly highlights emotionally charged / suspicious / fabricated content")
    print("\nHave 2 annotators fill independently, then compute weighted kappa.")
    print(f"\n⚠️  CLICK SAVE VERSION IN KAGGLE NOW")

    del model, wrapper; gc.collect(); torch.cuda.empty_cache()
