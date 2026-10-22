"""Dataset loading, splitting and label handling."""

import os
import re

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

# Extensions used for the train/test split. The dataset also contains .webp
# images that were never used during training; they serve as an extra
# held-out set (see `load_holdout`).
SPLIT_EXTS = (".png", ".jpg", ".jpeg")
HOLDOUT_EXTS = (".webp",)


def _list_images(class_dir, exts):
    return sorted(
        os.path.join(class_dir, f)
        for f in os.listdir(class_dir)
        if f.lower().endswith(exts)
    )


def _class_dirs(root_dir):
    return sorted(
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d))
    )


def split_dataset(root_dir, test_size=0.2, random_seed=42):
    """Stratified per-landmark train/test split of (image_path, label) pairs.

    The label is the raw folder name (e.g. "Eiffel_Tower_in_Paris_France"),
    which is exactly the text the model was trained to generate.
    """
    train_data, test_data = [], []
    for label in _class_dirs(root_dir):
        images = _list_images(os.path.join(root_dir, label), SPLIT_EXTS)
        train_imgs, test_imgs = train_test_split(
            images, test_size=test_size, random_state=random_seed
        )
        train_data.extend((img, label) for img in train_imgs)
        test_data.extend((img, label) for img in test_imgs)
    return train_data, test_data


def load_holdout(root_dir):
    """All .webp images: excluded from training, so guaranteed unseen."""
    data = []
    for label in _class_dirs(root_dir):
        images = _list_images(os.path.join(root_dir, label), HOLDOUT_EXTS)
        data.extend((img, label) for img in images)
    return data


def all_labels(root_dir):
    return _class_dirs(root_dir)


def normalize_text(text):
    """Canonical form used to compare captions with labels.

    Training targets were raw folder names, so the tokenizer splits on
    underscores and the model generates e.g. "himeji _ castle _ in _ himeji".
    Lower-casing and collapsing every non-alphanumeric run to a single space
    maps both that output and the folder name to "himeji castle in himeji".
    """
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def pretty_label(label):
    """Human-readable label for figures, e.g. "Eiffel Tower in Paris France"."""
    return re.sub(r"\s+", " ", label.replace("_", " ")).strip()


class LandmarkDataset(Dataset):
    def __init__(self, data, processor):
        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        inputs = self.processor(
            images=image,
            text=label,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "label_str": label,
        }


def collate_fn(batch):
    """Stack tensor fields; keep `label_str` as a list of strings."""
    collated = {
        k: torch.stack([item[k] for item in batch])
        for k in batch[0]
        if k != "label_str"
    }
    collated["label_str"] = [item["label_str"] for item in batch]
    return collated
