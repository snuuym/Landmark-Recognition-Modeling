"""Fine-tune BLIP for landmark recognition with LoRA + captioning/ITC loss.

Example:
    python src/train.py --data-root data/landmarks --output-dir checkpoints
"""

import argparse
import csv
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from data import LandmarkDataset, collate_fn, split_dataset
from model import ProjectionHeads, build_lora_model, get_device, image_text_embeddings, itc_loss


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True, help="Folder with one sub-folder per landmark")
    p.add_argument("--output-dir", default="checkpoints")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--itc-weight", type=float, default=0.3, help="Weight of the ITC loss term")
    p.add_argument("--proj-dim", type=int, default=256)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--max-length", type=int, default=32, help="Token length of the target text")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = get_device()

    train_data, test_data = split_dataset(args.data_root, random_seed=args.seed)
    print(f"Train samples: {len(train_data)} | Test samples: {len(test_data)}")

    model, processor = build_lora_model(lora_r=args.lora_r, lora_alpha=args.lora_alpha)
    model.print_trainable_parameters()
    model.to(device)

    hidden = model.base_model.model.config
    heads = ProjectionHeads(
        vision_dim=hidden.vision_config.hidden_size,
        text_dim=hidden.text_config.hidden_size,
        proj_dim=args.proj_dim,
    ).to(device)

    train_loader = DataLoader(
        LandmarkDataset(train_data, processor, max_length=args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device == "cuda",
        collate_fn=collate_fn,
    )

    params = [p for p in model.parameters() if p.requires_grad] + list(heads.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
    )

    os.makedirs(args.output_dir, exist_ok=True)
    history = []
    model.train()
    heads.train()

    for epoch in range(1, args.epochs + 1):
        cap_total, itc_total = 0.0, 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            input_ids = batch["input_ids"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Padding tokens are ignored by the captioning loss.
            labels = input_ids.clone()
            labels[labels == processor.tokenizer.pad_token_id] = -100

            cap_loss = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                labels=labels,
            ).loss

            image_embeds, text_embeds = image_text_embeddings(
                model, pixel_values, input_ids, attention_mask
            )
            itc = itc_loss(
                heads.img_proj(image_embeds), heads.text_proj(text_embeds), batch["label_str"]
            )

            loss = cap_loss + args.itc_weight * itc
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            cap_total += cap_loss.item()
            itc_total += itc.item()

        n = len(train_loader)
        history.append({"epoch": epoch, "cap_loss": cap_total / n, "itc_loss": itc_total / n})
        print(f"Epoch {epoch} | Cap Loss: {cap_total / n:.4f} | ITC Loss: {itc_total / n:.4f}")

    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    torch.save(
        {"img_proj": heads.img_proj.state_dict(), "text_proj": heads.text_proj.state_dict()},
        os.path.join(args.output_dir, "projection_layers.pt"),
    )
    with open(os.path.join(args.output_dir, "train_log.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "cap_loss", "itc_loss"])
        writer.writeheader()
        writer.writerows(history)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
