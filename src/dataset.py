"""
PyTorch Dataset for MultiBanFakeDetect.

CONFIRMED WORKING on Kaggle with:
- manifest columns: sample_id, text, image_path, label, label_id, split
- image_path is ABSOLUTE (full path already in manifest from prepare_split())
- BanglaBERT-Large tokenizer
- Batch size 4, MAX_TEXT_LEN 256
"""
import os
import sys

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
            transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
        ])
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_MEAN, std=IMAGE_STD),
    ])


class MultiBanFakeDetectDataset(Dataset):
    def __init__(self, manifest_path: str, split: str, tokenizer,
                 max_len: int = cfg.MAX_TEXT_LEN):
        df = pd.read_csv(manifest_path)
        self.df = df[df["split"] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No rows for split='{split}' in {manifest_path}")
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.transform = build_image_transform(is_train=(split == "train"))

    def __len__(self):
        return len(self.df)

    def _load_image(self, image_path: str) -> torch.Tensor:
        # image_path is already absolute in our manifest
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        return self.transform(img)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        text = str(row["text"])
        enc  = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "pixel_values":   self._load_image(row["image_path"]),
            "label":          torch.tensor(int(row["label_id"]), dtype=torch.long),
            "sample_id":      row["sample_id"],
            "text":           text,
        }


def compute_class_weights(manifest_path: str, split: str = "train") -> torch.Tensor:
    """
    Inverse-frequency weights for class-weighted CE loss.
    Only uses classes that actually appear in the split — avoids the
    exploding-weight bug we hit when llm_fake had 0 samples.
    """
    df     = pd.read_csv(manifest_path)
    df     = df[df["split"] == split]
    counts = df["label_id"].value_counts().sort_index()
    # Only weight classes present in this split
    weights = torch.ones(cfg.NUM_CLASSES, dtype=torch.float)
    for label_id, count in counts.items():
        weights[int(label_id)] = 1.0 / count
    # Normalize so mean weight = 1
    weights = weights / weights[counts.index].mean()
    return weights
