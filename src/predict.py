"""Recognize the landmark in one or more images.

Example:
    python src/predict.py path/to/photo.jpg --checkpoint checkpoints
"""

import argparse
import json
import os

from sentence_transformers import SentenceTransformer

from data import pretty_label
from evaluate import EMBEDDER, generate_captions, rank_labels
from model import get_device, load_model


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("images", nargs="+")
    p.add_argument("--checkpoint", default="checkpoints")
    p.add_argument("--labels", default=None,
                   help="JSON list of candidate landmark names (default: <checkpoint>/labels.json)")
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--device", default=None, help="cuda / mps / cpu; auto-detected by default")
    args = p.parse_args()

    with open(args.labels or os.path.join(args.checkpoint, "labels.json")) as f:
        labels = json.load(f)

    device = args.device or get_device()
    model, processor = load_model(args.checkpoint, device)
    data = [(path, None) for path in args.images]
    captions = generate_captions(model, processor, data, device, batch_size=len(data))
    ranked = rank_labels(captions, labels, SentenceTransformer(EMBEDDER, device=device))

    for path, caption, order in zip(args.images, captions, ranked):
        print(f"\n{path}\n  caption: {pretty_label(caption)}")
        for rank, j in enumerate(order[:args.top_k], 1):
            print(f"  {rank}. {pretty_label(labels[j])}")


if __name__ == "__main__":
    main()
