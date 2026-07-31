"""
QC sampling and inter-annotator agreement — FINAL

Step 1 — Draw 200-sample QC set:
    from src.qc_sample import draw_sample
    draw_sample()

Step 2 — After 2 annotators fill in the CSV:
    from src.qc_sample import score_agreement
    score_agreement()

Annotation guide:
  fluent_1to5            : 1=broken Bangla, 5=perfectly fluent natural Bangla
  realistic_misinfo_1to5 : 1=obviously nonsense, 5=genuinely resembles real misinformation
  obviously_fake_yn      : y=human reader instantly knows it's fake (BAD), n=not obvious (GOOD)
"""
import glob, os, sys
import pandas as pd
from sklearn.metrics import cohen_kappa_score

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from configs import config as cfg


def draw_sample(n: int = cfg.QC_SAMPLE_SIZE, seed: int = cfg.SEED) -> None:
    csvs = glob.glob(os.path.join(cfg.LLM_FAKE_DIR, "*_samples.csv"))
    if not csvs:
        raise FileNotFoundError(f"No generated samples in {cfg.LLM_FAKE_DIR}")

    df     = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)
    sample = df.sample(n=min(n, len(df)), random_state=seed).copy()
    sample = sample.sample(frac=1.0, random_state=seed).reset_index(drop=True)  # shuffle order

    sample["annotator_1_fluent_1to5"]            = ""
    sample["annotator_1_realistic_misinfo_1to5"] = ""
    sample["annotator_1_obviously_fake_yn"]      = ""
    sample["annotator_2_fluent_1to5"]            = ""
    sample["annotator_2_realistic_misinfo_1to5"] = ""
    sample["annotator_2_obviously_fake_yn"]      = ""
    sample["notes"]                               = ""

    os.makedirs(cfg.QC_DIR, exist_ok=True)
    out = os.path.join(cfg.QC_DIR, "qc_sample_annotated.csv")
    sample.to_csv(out, index=False, encoding="utf-8")

    print(f"QC sample ({len(sample)} rows) → {out}")
    print(f"\nAnnotation guide:")
    print(f"  fluent_1to5            : 1=broken Bangla → 5=perfectly fluent")
    print(f"  realistic_misinfo_1to5 : 1=nonsense → 5=genuinely realistic fake news")
    print(f"  obviously_fake_yn      : y=instantly recognizable as fake (BAD)")
    print(f"\nHave 2 annotators fill columns independently.")
    print(f"Then run: score_agreement()")
    print(f"\n⚠️  CLICK SAVE VERSION IN KAGGLE NOW")


def score_agreement(path: str = None) -> None:
    if path is None:
        path = os.path.join(cfg.QC_DIR, "qc_sample_annotated.csv")
    df = pd.read_csv(path)

    required = [
        "annotator_1_fluent_1to5", "annotator_2_fluent_1to5",
        "annotator_1_realistic_misinfo_1to5", "annotator_2_realistic_misinfo_1to5",
        "annotator_1_obviously_fake_yn", "annotator_2_obviously_fake_yn",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.dropna(subset=required)
    if len(df) == 0:
        raise ValueError("No fully-annotated rows found")

    k_fluent    = cohen_kappa_score(df["annotator_1_fluent_1to5"],
                                     df["annotator_2_fluent_1to5"], weights="quadratic")
    k_realistic = cohen_kappa_score(df["annotator_1_realistic_misinfo_1to5"],
                                     df["annotator_2_realistic_misinfo_1to5"], weights="quadratic")
    k_obvious   = cohen_kappa_score(df["annotator_1_obviously_fake_yn"],
                                     df["annotator_2_obviously_fake_yn"])

    # Acceptance stats
    reject_rate = (df["annotator_1_obviously_fake_yn"] == "y").mean()

    print(f"\nQC Results ({len(df)} annotated samples)")
    print(f"{'='*50}")
    print(f"  Fluency kappa      (quadratic-weighted): {k_fluent:.3f}")
    print(f"  Realism kappa      (quadratic-weighted): {k_realistic:.3f}")
    print(f"  Obviously-fake kappa                  : {k_obvious:.3f}")
    print(f"\n  Obviously-fake rejection rate: {reject_rate*100:.1f}%")
    print(f"\nInterpretation (Landis & Koch):")
    print(f"  <0.20 slight | 0.21-0.40 fair | 0.41-0.60 moderate")
    print(f"  0.61-0.80 substantial | >0.80 almost perfect")
    print(f"\nReport these kappas in the paper's Data Quality section.")
    print(f"\n⚠️  CLICK SAVE VERSION IN KAGGLE NOW")
