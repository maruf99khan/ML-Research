"""
PyTorch Dataset for MultiBanFakeDetect.
Confirmed working with real Kaggle data — absolute image paths in manifest.
"""
import os, sys
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from configs import config as cfg

IMAGE_MEAN = [0.5, 0.5, 0.5]
IMAGE_STD  = [0.5, 0.5, 0.5]


def build_image_transform(is_train: bool):
    if is_train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.2),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
    ])


class MultiBanFakeDetectDataset(Dataset):
    def __init__(self, manifest_path: str, split: str, tokenizer,
                 max_len: int = cfg.MAX_TEXT_LEN):
        df = pd.read_csv(manifest_path)
        self.df        = df[df["split"] == split].reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.transform = build_image_transform(is_train=(split == "train"))
        if len(self.df) == 0:
            raise ValueError(f"No rows for split='{split}' in {manifest_path}")

    def __len__(self):
        return len(self.df)

    def _load_image(self, path: str) -> torch.Tensor:
        try:
            return self.transform(Image.open(path).convert("RGB"))
        except Exception:
            return self.transform(Image.new("RGB", (224, 224), (128, 128, 128)))

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc = self.tokenizer(
            str(row["text"]),
            padding="max_length", truncation=True,
            max_length=self.max_len, return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "pixel_values":   self._load_image(str(row["image_path"])),
            "label":          torch.tensor(int(row["label_id"]), dtype=torch.long),
            "sample_id":      str(row["sample_id"]),
            "text":           str(row["text"]),
        }


def compute_class_weights(manifest_path: str, split: str = "train") -> torch.Tensor:
    """Inverse-frequency weights — only for classes present in this split."""
    df     = pd.read_csv(manifest_path)
    df     = df[df["split"] == split]
    counts = df["label_id"].value_counts().sort_index()
    weights = torch.ones(cfg.NUM_CLASSES, dtype=torch.float)
    present = counts.sum()
    for lid, cnt in counts.items():
        weights[int(lid)] = present / (len(counts) * cnt)
    return weights
