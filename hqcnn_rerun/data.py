"""
data.py - data loading for the controlled rerun.

Expected layout (the split used in the paper):

    DATA/train/<negative>/*.jpg|png|...     DATA/train/<positive>/*
    DATA/val/<negative>/*                   DATA/val/<positive>/*
    DATA/test/<negative>/*                  DATA/test/<positive>/*

Folder names default to negative = "normal", positive = "malignant" (case-insensitive);
any other class folders (for example "benign") are ignored and reported.

Every image is decoded once, resized to 224 x 224 (bilinear, antialiased - the same as
torchvision's Resize((224, 224)) on a PIL image) and cached as uint8, so every run and
every model variant sees identical pixels. Augmentation is applied on the fly to the
cached tensors:  horizontal flip (p = 0.5), rotation (+-20 deg), brightness/contrast
jitter (0.2), translation (10%) - the augmentation described in the paper.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _find_class_dir(split_dir: Path, name: str) -> Path | None:
    for d in split_dir.iterdir():
        if d.is_dir() and d.name.lower() == name.lower():
            return d
    return None


def scan(data_root: str | Path, negative: str = "normal", positive: str = "malignant",
         splits=("train", "val", "test")) -> dict:
    """Return {split: [(path, label), ...]} with label 1 = positive, 0 = negative."""
    root = Path(data_root)
    out, ignored = {}, {}
    for split in splits:
        sdir = root / split
        if not sdir.is_dir():
            raise FileNotFoundError(f"missing split folder: {sdir}")
        items = []
        for name, label in ((negative, 0), (positive, 1)):
            cdir = _find_class_dir(sdir, name)
            if cdir is None:
                raise FileNotFoundError(f"missing class folder '{name}' in {sdir}")
            files = sorted(p for p in cdir.rglob("*") if p.suffix.lower() in IMG_EXT)
            items += [(str(p), label) for p in files]
        others = [d.name for d in sdir.iterdir()
                  if d.is_dir() and d.name.lower() not in (negative.lower(), positive.lower())]
        if others:
            ignored[split] = others
        out[split] = items
    if ignored:
        print(f"[data] ignoring class folders: {ignored}")
    return out


def load_image(path: str, size: int = 224) -> torch.Tensor:
    with Image.open(path) as im:
        im = im.convert("RGB").resize((size, size), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.uint8).copy()
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()          # (3, H, W) uint8


def build_cache(data_root, negative="normal", positive="malignant", size=224,
                cache_dir: str | Path = "cache") -> dict:
    """Decode every image once; returns {split: dict(images uint8 (N,3,H,W), labels (N,), paths)}."""
    listing = scan(data_root, negative, positive)
    key_src = json.dumps({s: [(Path(p).name, Path(p).stat().st_size, y) for p, y in v]
                          for s, v in listing.items()}, sort_keys=True) + f"|{size}|{negative}|{positive}"
    key = hashlib.sha1(key_src.encode()).hexdigest()[:12]
    cache_dir = Path(cache_dir)
    cache_file = cache_dir / f"dmrir_{size}_{key}.pt"
    if cache_file.exists():
        return torch.load(cache_file, weights_only=False)
    cache_dir.mkdir(parents=True, exist_ok=True)
    data = {}
    for split, items in listing.items():
        imgs = torch.stack([load_image(p, size) for p, _ in items]) if items else torch.empty(0, 3, size, size, dtype=torch.uint8)
        data[split] = dict(images=imgs, labels=torch.tensor([y for _, y in items], dtype=torch.long),
                           paths=[p for p, _ in items])
        n_pos = int(data[split]["labels"].sum())
        print(f"[data] {split:5s}: {len(items):5d} images ({n_pos} {positive}, {len(items) - n_pos} {negative})")
    torch.save(data, cache_file)
    return data


class Augment:
    """Augmentation from the paper, applied to a float image in [0, 1]."""

    def __init__(self):
        from torchvision.transforms import v2
        self.t = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomRotation(degrees=20),
            v2.ColorJitter(brightness=0.2, contrast=0.2),
            v2.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ])

    def __call__(self, x):
        return self.t(x)


class CachedImages(Dataset):
    def __init__(self, split_data: dict, train: bool):
        self.images, self.labels = split_data["images"], split_data["labels"]
        self.aug = Augment() if train else None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        x = self.images[i].float() / 255.0
        if self.aug is not None:
            x = self.aug(x).clamp(0.0, 1.0)
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        return x, self.labels[i], i


def make_synthetic_dataset(root: str | Path, per_class=(40, 12, 12), size=96, seed=0):
    """Tiny synthetic stand-in used only by the smoke test (never for results)."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    for split, n in zip(("train", "val", "test"), per_class):
        for cls, hot in (("normal", False), ("malignant", True)):
            d = root / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                img = rng.normal(110, 25, (size, size))
                if hot:
                    yy, xx = np.mgrid[:size, :size]
                    cy, cx = rng.integers(size // 4, 3 * size // 4, 2)
                    img += 70 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (size / 10) ** 2))
                arr = np.clip(img, 0, 255).astype(np.uint8)
                Image.fromarray(np.stack([arr] * 3, -1)).save(d / f"{cls}_{split}_{i:03d}.png")
    return root
