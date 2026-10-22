"""Evaluate fine-tuned or zero-shot BLIP on landmark recognition.

Each generated caption is embedded with a sentence encoder and matched to the
closest of all landmark names (cosine similarity), giving Top-1 / Top-5
accuracy. Exact match (normalized caption == normalized label) is reported as
a stricter metric.

Examples:
    python src/evaluate.py --data-root data/landmarks --checkpoint checkpoints --split holdout --tag finetuned
    python src/evaluate.py --data-root data/landmarks --split holdout --tag zeroshot
"""

import argparse
import csv
import json
import os

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from data import all_labels, load_holdout, normalize_text, split_dataset
from model import get_device, load_model

EMBEDDER = "all-MiniLM-L6-v2"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True)
    p.add_argument("--checkpoint", default=None, help="LoRA checkpoint dir; omit for zero-shot BLIP")
    p.add_argument("--split", choices=["test", "holdout"], default="test",
                   help="test: 20%% split of jpg/png images; holdout: unseen .webp images")
    p.add_argument("--tag", default=None, help="Name used for the output files")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=None,
                   help="Generation length; default uses the model's generation config")
    p.add_argument("--device", default=None, help="cuda / mps / cpu; auto-detected by default")
    return p.parse_args()


@torch.no_grad()
def generate_captions(model, processor, data, device, batch_size, max_new_tokens=None):
    gen_kwargs = {} if max_new_tokens is None else {"max_new_tokens": max_new_tokens}
    captions = []
    for i in tqdm(range(0, len(data), batch_size), desc="Generating captions"):
        images = [Image.open(path).convert("RGB") for path, _ in data[i:i + batch_size]]
        inputs = processor(images=images, return_tensors="pt").to(device)
        output_ids = model.generate(**inputs, **gen_kwargs)
        captions.extend(processor.batch_decode(output_ids, skip_special_tokens=True))
    return captions


def rank_labels(captions, labels, embedder):
    """Indices of `labels` sorted by similarity to each caption, shape (N, L)."""
    label_emb = embedder.encode([normalize_text(l) for l in labels], normalize_embeddings=True)
    cap_emb = embedder.encode([normalize_text(c) for c in captions], normalize_embeddings=True)
    sims = cap_emb @ label_emb.T
    return np.argsort(-sims, axis=1)


def main():
    args = parse_args()
    device = args.device or get_device()
    tag = args.tag or ("finetuned" if args.checkpoint else "zeroshot")

    if args.split == "test":
        _, data = split_dataset(args.data_root)
    else:
        data = load_holdout(args.data_root)
    labels = all_labels(args.data_root)
    print(f"{len(data)} images | {len(labels)} candidate landmarks | device={device}")

    model, processor = load_model(args.checkpoint, device)
    captions = generate_captions(model, processor, data, device, args.batch_size, args.max_new_tokens)
    ranked = rank_labels(captions, labels, SentenceTransformer(EMBEDDER, device=device))

    rows, top1, top5, exact = [], 0, 0, 0
    for (path, true_label), caption, order in zip(data, captions, ranked):
        top5_labels = [labels[j] for j in order[:5]]
        hit1 = top5_labels[0] == true_label
        hit5 = true_label in top5_labels
        is_exact = normalize_text(caption) == normalize_text(true_label)
        top1 += hit1
        top5 += hit5
        exact += is_exact
        rows.append({
            "image": os.path.relpath(path, args.data_root),
            "true_label": true_label,
            "caption": caption,
            "pred_label": top5_labels[0],
            "top5": "|".join(top5_labels),
            "top1_correct": int(hit1),
            "top5_correct": int(hit5),
            "exact_match": int(is_exact),
        })

    n = len(rows)
    metrics = {
        "model": tag,
        "split": args.split,
        "n_images": n,
        "n_labels": len(labels),
        "top1": round(100 * top1 / n, 2),
        "top5": round(100 * top5 / n, 2),
        "exact_match": round(100 * exact / n, 2),
    }
    print(json.dumps(metrics, indent=2))

    os.makedirs(args.output_dir, exist_ok=True)
    stem = os.path.join(args.output_dir, f"{tag}_{args.split}")
    with open(f"{stem}_predictions.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(f"{stem}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
