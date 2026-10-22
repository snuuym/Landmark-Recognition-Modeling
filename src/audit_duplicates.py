"""Audit the dataset for duplicate images and their effect on evaluation.

1. Exact duplicates: identical files (SHA-256), within and across labels.
2. Train/test leakage: test images whose exact copy lands in the train split.
3. Holdout near-duplicates: .webp images that look like a .jpg/.png image
   (grayscale-thumbnail correlation or perceptual-hash distance). Candidates
   are written out for visual inspection; confirmed ones are read from
   results/holdout_near_duplicates.json.
4. Sensitivity: holdout accuracy with confirmed / all candidates removed.

Example:
    python src/audit_duplicates.py --data-root data/landmarks
"""

import argparse
import hashlib
import itertools
import json
import os
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from data import HOLDOUT_EXTS, SPLIT_EXTS, split_dataset

CORR_THRESHOLD = 0.80
DHASH_THRESHOLD = 10


def image_features(path):
    """SHA-256 of the file, 64-bit difference hash, and a 32x32 thumbnail vector."""
    with open(path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    gray = Image.open(path).convert("L")
    small = np.asarray(gray.resize((9, 8), Image.LANCZOS), dtype=np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    dhash = int("".join("1" if b else "0" for b in bits), 2)
    thumb = np.asarray(gray.resize((32, 32), Image.LANCZOS), dtype=np.float32).ravel()
    thumb -= thumb.mean()
    norm = np.linalg.norm(thumb)
    return sha, dhash, thumb / norm if norm > 0 else thumb


def scan(root):
    rows, thumbs = [], []
    labels = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    files = [
        (os.path.join(label, f), label)
        for label in labels
        for f in sorted(os.listdir(os.path.join(root, label)))
        if f.lower().endswith(SPLIT_EXTS + HOLDOUT_EXTS)
    ]
    for rel, label in tqdm(files, desc="Hashing images"):
        sha, dhash, thumb = image_features(os.path.join(root, rel))
        rows.append({"image": rel, "label": label, "sha256": sha, "dhash": dhash})
        thumbs.append(thumb)
    return pd.DataFrame(rows), np.stack(thumbs)


def exact_duplicates(df):
    groups = [g for _, g in df.groupby("sha256") if len(g) > 1]
    pair_counts = Counter()
    for g in groups:
        for a, b in itertools.combinations(sorted(set(g.label)), 2):
            pair_counts[(a, b)] += 1
    return {
        "duplicate_groups": len(groups),
        "files_in_groups": int(sum(len(g) for g in groups)),
        "groups_spanning_labels": int(sum(g.label.nunique() > 1 for g in groups)),
        "label_pairs_sharing_files": [
            {"labels": list(pair), "shared_groups": n} for pair, n in pair_counts.most_common()
        ],
    }


def split_leakage(df, root):
    train, test = split_dataset(root)
    train_set = {os.path.relpath(p, root) for p, _ in train}
    test_set = {os.path.relpath(p, root) for p, _ in test}
    train_df = df[df.image.isin(train_set)]
    test_df = df[df.image.isin(test_set)]
    leaked = test_df[test_df.sha256.isin(set(train_df.sha256))]
    train_labels = train_df.groupby("sha256").label.agg(set)
    same_label = sum(row.label in train_labels[row.sha256] for row in leaked.itertuples())
    return {
        "n_train": len(train_df),
        "n_test": len(test_df),
        "test_with_exact_copy_in_train": len(leaked),
        "of_which_same_label": int(same_label),
        "of_which_other_label": int(len(leaked) - same_label),
    }


def holdout_candidates(df, thumbs):
    is_holdout = df.image.str.lower().str.endswith(HOLDOUT_EXTS).to_numpy()
    hold, rest = df[is_holdout].reset_index(drop=True), df[~is_holdout].reset_index(drop=True)
    corr = thumbs[is_holdout] @ thumbs[~is_holdout].T
    rest_hashes = rest.dhash.to_numpy(dtype=np.uint64)
    rows = []
    for i, h in enumerate(hold.itertuples()):
        j = int(corr[i].argmax())
        dist = np.array([bin(int(x)).count("1") for x in np.bitwise_xor(rest_hashes, np.uint64(h.dhash))])
        k = int(dist.argmin())
        if corr[i, j] >= CORR_THRESHOLD or dist[k] <= DHASH_THRESHOLD:
            rows.append({
                "holdout": h.image,
                "corr_match": rest.image[j], "corr": round(float(corr[i, j]), 4),
                "dhash_match": rest.image[k], "dhash_dist": int(dist[k]),
            })
    return pd.DataFrame(rows), int(is_holdout.sum())


def sensitivity(results_dir, confirmed, candidates):
    out = {}
    for tag in ("zeroshot", "finetuned"):
        path = os.path.join(results_dir, f"{tag}_holdout_predictions.csv")
        if not os.path.exists(path):
            continue
        preds = pd.read_csv(path)
        out[tag] = {}
        for name, excluded in [("all", set()), ("without_confirmed", confirmed),
                               ("without_all_candidates", candidates)]:
            kept = preds[~preds.image.isin(excluded)]
            out[tag][name] = {
                "n_images": len(kept),
                "top1": round(100 * kept.top1_correct.mean(), 2),
                "top5": round(100 * kept.top5_correct.mean(), 2),
            }
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True)
    p.add_argument("--results-dir", default="results")
    args = p.parse_args()

    df, thumbs = scan(args.data_root)
    candidates, n_holdout = holdout_candidates(df, thumbs)
    candidates.to_csv(os.path.join(args.results_dir, "holdout_duplicate_candidates.csv"), index=False)

    with open(os.path.join(args.results_dir, "holdout_near_duplicates.json")) as f:
        confirmed = set(json.load(f)["confirmed"])

    report = {
        "n_images": len(df),
        "exact_duplicates": exact_duplicates(df),
        "train_test_leakage_current_split": split_leakage(df, args.data_root),
        "holdout": {
            "n_images": n_holdout,
            "n_labels_covered": int(df[df.image.str.lower().str.endswith(HOLDOUT_EXTS)].label.nunique()),
            "near_duplicate_candidates": len(candidates),
            "confirmed_near_duplicates": len(confirmed),
            "candidate_rule": f"thumbnail corr >= {CORR_THRESHOLD} or dHash distance <= {DHASH_THRESHOLD}",
        },
        "holdout_sensitivity": sensitivity(args.results_dir, confirmed, set(candidates.holdout) | confirmed),
    }
    with open(os.path.join(args.results_dir, "duplicate_audit.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k != "exact_duplicates"}, indent=2))


if __name__ == "__main__":
    main()
