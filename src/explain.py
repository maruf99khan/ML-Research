"""
Integrated Gradients explainability for MultiBanFakeDetect.

Produces, per prediction:
  - token-level attribution scores over the Bangla input text
  - patch-level attribution scores over the image (as a heatmap overlay)

Also exports a CSV of qualitative examples (IG_NUM_QUALITATIVE_EXAMPLES_PER_CLASS
per class) and a separate CSV sampling IG_HUMAN_CHECK_SAMPLE_SIZE random
predictions for the manual plausibility check added to Phase 4 of the plan
(two annotators rate whether the highlighted words/regions make intuitive sense).

Usage:
    python src/explain.py --checkpoint outputs/checkpoints/cmaf_full_best.pt --split test
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients
from transformers import AutoTokenizer

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from configs import config as cfg
from src.dataset import MultiBanFakeDetectDataset
from src.model import MultiBanFakeDetectModel, ForwardWrapper


def get_text_embeddings(model: MultiBanFakeDetectModel, input_ids: torch.Tensor) -> torch.Tensor:
    embedding_layer = model.text_encoder.get_input_embeddings()
    return embedding_layer(input_ids)


def attribute_single_example(wrapper: ForwardWrapper, model: MultiBanFakeDetectModel,
                              input_ids, attention_mask, pixel_values, target_class: int,
                              device):
    input_ids = input_ids.unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)
    pixel_values = pixel_values.unsqueeze(0).to(device)

    inputs_embeds = get_text_embeddings(model, input_ids).detach()
    inputs_embeds.requires_grad_()
    pixel_values = pixel_values.clone().requires_grad_()

    baseline_embeds = torch.zeros_like(inputs_embeds)
    baseline_pixels = torch.zeros_like(pixel_values)

    # NOTE: inputs=(inputs_embeds, pixel_values) + additional_forward_args=(attention_mask,)
    # is called by Captum as wrapper(inputs_embeds, pixel_values, attention_mask) -- the
    # ForwardWrapper.forward signature matches this order exactly (see model.py comment).
    ig = IntegratedGradients(wrapper)
    attributions, delta = ig.attribute(
        inputs=(inputs_embeds, pixel_values),
        baselines=(baseline_embeds, baseline_pixels),
        additional_forward_args=(attention_mask,),
        target=target_class,
        n_steps=cfg.IG_N_STEPS,
        internal_batch_size=cfg.IG_INTERNAL_BATCH_SIZE,
        return_convergence_delta=True,
    )
    text_attr, image_attr = attributions

    # Token-level score: L2 norm over embedding dim, then normalize to [-1, 1] range for display
    token_scores = text_attr.norm(dim=-1).squeeze(0).detach().cpu().numpy()
    token_scores = token_scores / (np.abs(token_scores).max() + 1e-8)

    # Patch-level score: mean absolute attribution per pixel, downsampled to a coarse grid
    patch_scores = image_attr.abs().mean(dim=1).squeeze(0).detach().cpu().numpy()  # (H, W)

    return token_scores, patch_scores, float(delta.abs().item())


def run_explanations(checkpoint_path: str, split: str, output_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_model_name = cfg.FALLBACK_TEXT_MODEL_NAME if cfg.USE_FALLBACK else cfg.TEXT_MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)

    model = MultiBanFakeDetectModel().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    wrapper = ForwardWrapper(model).to(device)
    wrapper.eval()

    dataset = MultiBanFakeDetectDataset(cfg.COMBINED_MANIFEST, split, tokenizer)

    os.makedirs(output_dir, exist_ok=True)
    qualitative_rows = []
    per_class_counts = {c: 0 for c in cfg.LABEL2ID.values()}

    indices = list(range(len(dataset)))
    np.random.shuffle(indices)

    for idx in indices:
        if all(v >= cfg.IG_NUM_QUALITATIVE_EXAMPLES_PER_CLASS for v in per_class_counts.values()):
            break

        example = dataset[idx]
        label = int(example["label"])
        if per_class_counts[label] >= cfg.IG_NUM_QUALITATIVE_EXAMPLES_PER_CLASS:
            continue

        with torch.no_grad():
            logits = model(
                example["input_ids"].unsqueeze(0).to(device),
                example["attention_mask"].unsqueeze(0).to(device),
                example["pixel_values"].unsqueeze(0).to(device),
            )
            pred_class = int(logits.argmax(dim=-1).item())

        token_scores, patch_scores, delta = attribute_single_example(
            wrapper, model, example["input_ids"], example["attention_mask"],
            example["pixel_values"], target_class=pred_class, device=device,
        )

        tokens = tokenizer.convert_ids_to_tokens(example["input_ids"].tolist())
        top_k = 10
        top_idx = np.argsort(-np.abs(token_scores))[:top_k]
        top_tokens = [(tokens[i], round(float(token_scores[i]), 3)) for i in top_idx
                      if tokens[i] not in tokenizer.all_special_tokens]

        npy_path = os.path.join(output_dir, f"{example['sample_id']}_patch_attr.npy")
        np.save(npy_path, patch_scores)

        qualitative_rows.append({
            "sample_id": example["sample_id"],
            "true_label": cfg.ID2LABEL[label],
            "predicted_label": cfg.ID2LABEL[pred_class],
            "correct": label == pred_class,
            "top_attributed_tokens": str(top_tokens),
            "ig_convergence_delta": delta,
            "patch_attribution_path": npy_path,
        })
        per_class_counts[label] += 1

    qual_df = pd.DataFrame(qualitative_rows)
    qual_path = os.path.join(output_dir, "qualitative_examples.csv")
    qual_df.to_csv(qual_path, index=False, encoding="utf-8")
    print(f"Qualitative examples ({len(qual_df)}) -> {qual_path}")

    # Separate random sample for the human-plausibility check (Phase 4 addition)
    human_check_df = qual_df.sample(
        n=min(cfg.IG_HUMAN_CHECK_SAMPLE_SIZE, len(qual_df)), random_state=cfg.SEED
    ).copy()
    human_check_df["annotator_1_plausible_1to5"] = ""
    human_check_df["annotator_2_plausible_1to5"] = ""
    human_check_df["notes"] = ""
    human_check_path = os.path.join(output_dir, "human_plausibility_check_template.csv")
    human_check_df.to_csv(human_check_path, index=False, encoding="utf-8")
    print(f"Human plausibility-check template -> {human_check_path}")
    print("Have two annotators independently fill the *_plausible_1to5 columns "
          "(1 = attribution makes no sense, 5 = clearly highlights sensational/false claims), "
          "then compute agreement (e.g. weighted kappa) before citing this in the paper.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output-dir", default=cfg.EXPLAIN_DIR)
    args = parser.parse_args()
    run_explanations(args.checkpoint, args.split, args.output_dir)


if __name__ == "__main__":
    main()
