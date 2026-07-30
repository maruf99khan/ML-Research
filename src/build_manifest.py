"""
Builds / rebuilds the combined 3-class manifest.

Run this ONCE after generating LLM-fake samples to merge everything together.
Also run it initially with just the base dataset (binary) to get started.

Usage (in Kaggle notebook):
    from src.build_manifest import build_manifest
    build_manifest()
"""
import glob
import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from configs import config as cfg

BASE = cfg.DATASET_DIR


def prepare_base_dataset() -> pd.DataFrame:
    """Load the original MultiBanFakeDetect binary dataset."""
    def load_split(csv_name, split_name, img_folder):
        df = pd.read_csv(f"{BASE}/{csv_name}")
        df["sample_id"]  = df["image_id"]
        df["text"]       = df["headline"].fillna("") + " " + df["description"].fillna("")
        df["text"]       = df["text"].str.strip()
        df["label"]      = df["label"].map({0: "real", 1: "human_fake"})
        df["label_id"]   = df["label"].map(cfg.LABEL2ID)
        df["image_path"] = df["image_id"].apply(lambda x: f"{BASE}/{img_folder}/{x}.png")
        df["split"]      = split_name
        df["generator"]  = None
        df["strategy"]   = None
        return df[["sample_id", "text", "image_path", "label", "label_id", "split", "generator", "strategy"]]

    train = load_split("Train.csv",      "train", "Train")
    val   = load_split("Validation.csv", "val",   "Validation")
    test  = load_split("Test.csv",       "test",  "Test")
    return pd.concat([train, val, test], ignore_index=True)


def load_llm_fake() -> pd.DataFrame:
    """Load all generated LLM-fake CSVs if they exist."""
    csvs = glob.glob(os.path.join(cfg.LLM_FAKE_DIR, "*_samples.csv"))
    if not csvs:
        print("No LLM-fake CSVs found — building binary manifest only.")
        return pd.DataFrame()

    dfs = [pd.read_csv(p) for p in csvs]
    df  = pd.concat(dfs, ignore_index=True)
    df["label"]    = "llm_fake"
    df["label_id"] = cfg.LABEL2ID["llm_fake"]

    # Stratified 80/10/10 split per generator
    parts = []
    for gen, gdf in df.groupby("generator"):
        train, temp = train_test_split(gdf, test_size=0.2, random_state=cfg.SEED)
        val, test   = train_test_split(temp, test_size=0.5, random_state=cfg.SEED)
        train = train.copy(); train["split"] = "train"
        val   = val.copy();   val["split"]   = "val"
        test  = test.copy();  test["split"]  = "test"
        parts.extend([train, val, test])
    return pd.concat(parts, ignore_index=True)


def build_manifest():
    base_df = prepare_base_dataset()
    llm_df  = load_llm_fake()

    if len(llm_df) > 0:
        needed_cols = ["sample_id", "text", "image_path", "label", "label_id", "split", "generator", "strategy"]
        combined = pd.concat([base_df, llm_df[needed_cols]], ignore_index=True)
    else:
        combined = base_df

    combined = combined.dropna(subset=["text", "image_path"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(cfg.COMBINED_MANIFEST), exist_ok=True)
    combined.to_csv(cfg.COMBINED_MANIFEST, index=False, encoding="utf-8")

    print("Saved:", cfg.COMBINED_MANIFEST)
    print("\nClass distribution:")
    print(combined.groupby(["split", "label"]).size().unstack(fill_value=0))
    print("\nTotal samples:", len(combined))
    return combined


if __name__ == "__main__":
    build_manifest()
