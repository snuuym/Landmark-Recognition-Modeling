"""BLIP + LoRA model construction, ITC loss and checkpoint loading."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoProcessor, BlipForConditionalGeneration

BASE_MODEL = "Salesforce/blip-image-captioning-base"


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_lora_model(lora_r=16, lora_alpha=32, lora_dropout=0.05):
    """Base BLIP captioner with a frozen ViT and LoRA on the text decoder.

    BLIP's vision attention uses a fused `qkv` projection, so targeting
    `query`/`value` places LoRA adapters only in the BERT-style text decoder
    (self- and cross-attention). The vision encoder is frozen explicitly so
    the pre-trained visual features are preserved.
    """
    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    model = BlipForConditionalGeneration.from_pretrained(BASE_MODEL)

    for name, param in model.named_parameters():
        if "vision_model" in name:
            param.requires_grad = False

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["query", "value"],
        lora_dropout=lora_dropout,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    return model, processor


class ProjectionHeads(nn.Module):
    """Linear maps from vision / text hidden size into a shared ITC space."""

    def __init__(self, vision_dim=768, text_dim=768, proj_dim=256):
        super().__init__()
        self.img_proj = nn.Linear(vision_dim, proj_dim)
        self.text_proj = nn.Linear(text_dim, proj_dim)


def image_text_embeddings(model, pixel_values, input_ids, attention_mask):
    """Mean-pooled vision and text-decoder hidden states, shape (B, 768) each."""
    blip = model.base_model.model
    vision_out = blip.vision_model(pixel_values=pixel_values)
    text_out = blip.text_decoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    )
    image_embeds = vision_out.last_hidden_state.mean(dim=1)
    text_embeds = text_out.hidden_states[-1].mean(dim=1)
    return image_embeds, text_embeds


def itc_loss(image_embeds, text_embeds, temperature=0.07):
    """Symmetric InfoNCE loss; matching image/text pairs lie on the diagonal."""
    image_embeds = F.normalize(image_embeds, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)
    logits = image_embeds @ text_embeds.T / temperature
    targets = torch.arange(logits.size(0), device=logits.device)
    loss_i2t = F.cross_entropy(logits, targets)
    loss_t2i = F.cross_entropy(logits.T, targets)
    return (loss_i2t + loss_t2i) / 2


def load_model(checkpoint=None, device=None):
    """Load the fine-tuned model from `checkpoint`, or zero-shot BLIP if None."""
    device = device or get_device()
    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    model = BlipForConditionalGeneration.from_pretrained(BASE_MODEL)
    if checkpoint is not None:
        model = PeftModel.from_pretrained(model, checkpoint)
    model.to(device).eval()
    return model, processor
