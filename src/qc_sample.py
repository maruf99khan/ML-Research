"""
Draws a manual QC sample from the generated LLM-fake pool (Phase 2) and computes
inter-annotator agreement once two annotators have filled in their ratings.

Step 1 (before annotation):
    python src/qc_sample.py draw --n 200

Step 2 (after both annotators fill in the CSV):
    python src/qc_sample.py score --path outputs/qc/qc_sample_annotated.csv
"""
import argparse
import glob
import os
import sys

import pandas as pd
from sklearn.metrics import cohen_kappa_score

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from configs import config as cfg

QC_DIR = os.path.join(cfg.BASE_DIR, "outputs", "qc")


def draw_sample(n: int, seed: int = cfg.SEED):
    csv_paths = glob.glob(os.path.join(cfg.LLM_FAKE_DIR, "*_samples.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No generated samples found in {cfg.LLM_FAKE_DIR}")

    df = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)
    sample = df.sample(n=min(n, len(df)), random_state=seed).copy()

    # Stratify visibility: keep generator/strategy visible to the researchers doing QC,
    # but shuffle row order so QC isn't done "by batch" (avoids drift within a session).
    sample = sample.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    sample["annotator_1_fluent_1to5"] = ""
    sample["annotator_1_realistic_misinfo_1to5"] = ""
    sample["annotator_1_obviously_fake_yn"] = ""
    sample["annotator_2_fluent_1to5"] = ""
    sample["annotator_2_realistic_misinfo_1to5"] = ""
    sample["annotator_2_obviously_fake_yn"] = ""
    sample["notes"] = ""

    os.makedirs(QC_DIR, exist_ok=True)
    out_path = os.path.join(QC_DIR, "qc_sample_annotated.csv")
    sample.to_csv(out_path, index=False, encoding="utf-8")

    print(f"Drew {len(sample)} samples for manual QC -> {out_path}")
    print("Rating guide:")
    print("  fluent_1to5             : 1 = broken Bangla, 5 = perfectly fluent")
    print("  realistic_misinfo_1to5  : 1 = nonsensical, 5 = genuinely resembles real misinformation")
    print("  obviously_fake_yn       : y if a human reader would instantly know it's fabricated "
          "(bad for the paper -- we want fakes that are NOT obviously fake), n otherwise")
    print("\nHave two independent annotators fill in the annotator_1_* / annotator_2_* columns, "
          "then run:  python src/qc_sample.py score --path <this file>")


def score_agreement(path: str):
    df = pd.read_csv(path)
    required = ["annotator_1_fluent_1to5", "annotator_2_fluent_1to5",
                "annotator_1_realistic_misinfo_1to5", "annotator_2_realistic_misinfo_1to5",
                "annotator_1_obviously_fake_yn", "annotator_2_obviously_fake_yn"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    df = df.dropna(subset=required)
    if len(df) == 0:
        raise ValueError("No fully-annotated rows found -- have both annotators finished?")

    kappa_fluent = cohen_kappa_score(df["annotator_1_fluent_1to5"], df["annotator_2_fluent_1to5"],
                                      weights="quadratic")
    kappa_realistic = cohen_kappa_score(df["annotator_1_realistic_misinfo_1to5"],
                                         df["annotator_2_realistic_misinfo_1to5"], weights="quadratic")
    kappa_obvious = cohen_kappa_score(df["annotator_1_obviously_fake_yn"],
                                       df["annotator_2_obviously_fake_yn"])

    print(f"N annotated       : {len(df)}")
    print(f"Fluency kappa      (quadratic-weighted): {kappa_fluent:.3f}")
    print(f"Realism kappa      (quadratic-weighted): {kappa_realistic:.3f}")
    print(f"Obviously-fake kappa: {kappa_obvious:.3f}")
    print("\nInterpretation guide (Landis & Koch): "
          "<0.20 slight, 0.21-0.40 fair, 0.41-0.60 moderate, 0.61-0.80 substantial, >0.80 almost perfect.")
    print("Report these kappas in the paper's data-quality subsection.")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    draw_p = sub.add_parser("draw")
    draw_p.add_argument("--n", type=int, default=cfg.QC_SAMPLE_SIZE)

    score_p = sub.add_parser("score")
    score_p.add_argument("--path", required=True)

    args = parser.parse_args()
    if args.cmd == "draw":
        draw_sample(args.n)
    elif args.cmd == "score":
        score_agreement(args.path)


if __name__ == "__main__":
    main()
