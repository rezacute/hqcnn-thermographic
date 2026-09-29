#!/usr/bin/env python3
"""
leakage_audit.py - does the test set contain near-copies of training images?

DMR-IR stores many thermograms per patient (a dynamic series plus static views). If the
train/val/test split was made per image rather than per patient, the same patient appears
on both sides and test accuracy is inflated. This script checks the split you trained on.

    python leakage_audit.py --data DATA [--id-regex "(\\d+)"]

  * exact duplicates (identical file bytes) across splits
  * perceptual near-duplicates: 256-bit difference hash; for every val/test image, the
    Hamming distance to its nearest training image, compared with the typical distance
    between two unrelated images
  * optional patient-ID overlap, if an ID can be read from file names with --id-regex
Writes leakage_report.json and leakage_pairs.csv (closest training image per test image),
so the closest pairs can be opened and checked by eye.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

from data import scan


def dhash(path: str, size: int = 16) -> np.ndarray:
    with Image.open(path) as im:
        g = np.asarray(im.convert("L").resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    return (g[:, 1:] > g[:, :-1]).reshape(-1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--negative", default="normal")
    ap.add_argument("--positive", default="malignant")
    ap.add_argument("--id-regex", default=None, help="regex whose first group is the patient ID in the file name")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    listing = scan(a.data, a.negative, a.positive)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    sha, hashes, labels = {}, {}, {}
    for split, items in listing.items():
        sha[split] = [hashlib.sha1(Path(p).read_bytes()).hexdigest() for p, _ in items]
        hashes[split] = np.stack([dhash(p) for p, _ in items]) if items else np.zeros((0, 256), bool)
        labels[split] = np.array([y for _, y in items])

    report = dict(sizes={s: len(v) for s, v in listing.items()})
    train_sha = set(sha["train"])
    report["exact_duplicates_in_train"] = {s: int(sum(h in train_sha for h in sha[s])) for s in ("val", "test")}

    H_train = hashes["train"].astype(np.uint8)
    rng = np.random.default_rng(0)
    # typical distance between unrelated images: random train pairs
    i, j = rng.integers(0, len(H_train), 2000), rng.integers(0, len(H_train), 2000)
    keep = i != j
    background = (H_train[i[keep]] != H_train[j[keep]]).sum(1)
    report["unrelated_pair_distance"] = dict(p1=float(np.percentile(background, 1)),
                                             p5=float(np.percentile(background, 5)),
                                             median=float(np.median(background)))
    thr = float(np.percentile(background, 1))
    rows = []
    for split in ("val", "test"):
        H = hashes[split].astype(np.uint8)
        dist = np.stack([(H_train != h).sum(1) for h in H]) if len(H) else np.zeros((0, len(H_train)))
        nn_idx, nn_d = dist.argmin(1), dist.min(1)
        same = labels["train"][nn_idx] == labels[split]
        report[f"{split}_nearest_train_distance"] = dict(
            median=float(np.median(nn_d)), p10=float(np.percentile(nn_d, 10)),
            n_below_unrelated_p1=int((nn_d < thr).sum()), n_identical_hash=int((nn_d == 0).sum()),
            n_within_8_bits=int((nn_d <= 8).sum()), n_within_16_bits=int((nn_d <= 16).sum()),
            frac_nearest_same_label=float(same.mean()))
        if split == "test":
            for k in np.argsort(nn_d):
                rows.append([listing["test"][k][0], listing["train"][nn_idx[k]][0], int(nn_d[k]),
                             int(labels["test"][k]), int(labels["train"][nn_idx[k]])])

    if a.id_regex:
        rx = re.compile(a.id_regex)
        ids = {s: {m.group(1) for p, _ in v if (m := rx.search(Path(p).name))} for s, v in listing.items()}
        report["patient_id_overlap"] = {f"{x}&{y}": len(ids[x] & ids[y])
                                        for x, y in (("train", "val"), ("train", "test"), ("val", "test"))}
        report["patient_ids_per_split"] = {s: len(v) for s, v in ids.items()}
    report["example_file_names"] = {s: [Path(p).name for p, _ in v[:5]] for s, v in listing.items()}

    (out / "leakage_report.json").write_text(json.dumps(report, indent=2))
    with open(out / "leakage_pairs.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["test_image", "closest_train_image", "hamming_distance_of_256", "test_label", "train_label"])
        w.writerows(rows)
    t = report["test_nearest_train_distance"]
    print(json.dumps(report, indent=2))
    print(f"\n{t['n_below_unrelated_p1']} of {report['sizes']['test']} test images are closer to a training image "
          f"than 99% of random training pairs (distance < {thr:.0f}/256); {t['n_within_16_bits']} are within 16/256 bits and "
          f"{t['n_identical_hash']} have an identical hash. Consecutive frames of one patient's dynamic series typically "
          f"fall within a few bits.")
    print("Open the top rows of leakage_pairs.csv to see whether these are the same patient.")


if __name__ == "__main__":
    main()
