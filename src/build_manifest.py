"""
Build 3-class manifest from base dataset + generated LLM-fake CSVs.
Run after every generation session and after QC.

Usage:
    from src.build_manifest import build_manifest
    build_manifest()
"""
import glob, os, sys
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from configs import config as cfg

BASE = cfg.DATASET_DIR


def prepare_base() -> pd.DataFrame:
    def load(csv_name, split_name, img_folder):
        df = pd.read_csv(f"{BASE}/{csv_name}")
        df["sample_id"]  = df["image_id"]
        df["text"]       = (df["headline"].fillna("") + " " + df["description"].fillna("")).str.strip()
        df["label"]      = df["label"].map({0: "real", 1: "human_fake"})
        df["label_id"]   = df["label"].map(cfg.LABEL2ID)
        df["image_path"] = df["image_id"].apply(lambda x: f"{BASE}/{img_folder}/{x}.png")
        df["split"]      = split_name
        df["generator"]  = None
        df["strategy"]   = None
        return df[["sample_id","text","image_path","label","label_id","split","generator","strategy"]]
    return pd.concat([
        load("Train.csv",      "train", "Train"),
        load("Validation.csv", "val",   "Validation"),
        load("Test.csv",       "test",  "Test"),
    ], ignore_index=True)


def load_llm_fake() -> pd.DataFrame:
    csvs = glob.glob(os.path.join(cfg.LLM_FAKE_DIR, "*_samples.csv"))
    if not csvs:
        print("No LLM-fake CSVs found — building binary manifest only.")
        return pd.DataFrame()
    dfs = []
    for p in csvs:
        try:
            dfs.append(pd.read_csv(p))
        except Exception as e:
            print(f"  Warning: cannot read {p}: {e}")
    if not dfs:
        return pd.DataFrame()
    df           = pd.concat(dfs, ignore_index=True)
    df["label"]    = "llm_fake"
    df["label_id"] = cfg.LABEL2ID["llm_fake"]
    # Stratified 80/10/10 per generator so every generator appears in every split
    parts = []
    for gen, gdf in df.groupby("generator"):
        if len(gdf) < 10:
            gdf = gdf.copy(); gdf["split"] = "train"; parts.append(gdf); continue
        train, temp = train_test_split(gdf, test_size=0.2, random_state=cfg.SEED)
        val, test   = train_test_split(temp, test_size=0.5, random_state=cfg.SEED)
        train = train.copy(); train["split"] = "train"
        val   = val.copy();   val["split"]   = "val"
        test  = test.copy();  test["split"]  = "test"
        parts.extend([train, val, test])
    return pd.concat(parts, ignore_index=True)


def build_manifest() -> pd.DataFrame:
    base = prepare_base()
    llm  = load_llm_fake()

    if len(llm) > 0:
        cols = ["sample_id","text","image_path","label","label_id","split","generator","strategy"]
        combined = pd.concat([base, llm[cols]], ignore_index=True)
    else:
        combined = base

    combined = combined.dropna(subset=["text","image_path"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(cfg.COMBINED_MANIFEST), exist_ok=True)
    combined.to_csv(cfg.COMBINED_MANIFEST, index=False, encoding="utf-8")

    print(f"Manifest saved: {cfg.COMBINED_MANIFEST}")
    print(f"Total rows: {len(combined)}")
    print("\nClass distribution:")
    print(combined.groupby(["split","label"]).size().unstack(fill_value=0))
    return combined


if __name__ == "__main__":
    build_manifest()
