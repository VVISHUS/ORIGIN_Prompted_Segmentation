import torch

from .base import PromptSegModel
from .registry import register


@register("clipseg")
class CLIPSeg(PromptSegModel):
    """CLIPSeg (Lüddecke & Ecker, CVPR 2022): CLIP ViT-B/16 image + text encoders and a light
    FiLM-conditioned transformer decoder that outputs 352×352 mask logits."""

    backbone_prefixes = ("net.clip.",)

    def __init__(self, pretrained="CIDAS/clipseg-rd64-refined", freeze_vision=True, freeze_text=True,
                 unfreeze_last_n_vision_layers=0):
        super().__init__()
        from transformers import AutoProcessor, CLIPSegForImageSegmentation

        self.net = CLIPSegForImageSegmentation.from_pretrained(pretrained)
        processor = AutoProcessor.from_pretrained(pretrained)
        self.tokenizer = processor.tokenizer
        ip = processor.image_processor
        size = ip.size          # int, dict (transformers 4.x) or SizeDict (5.x)
        if not isinstance(size, int):
            size = dict(size) if isinstance(size, dict) else vars(size)
            size = size.get("height") or size.get("shortest_edge")
        self.input_size = int(size)
        self.image_mean, self.image_std = tuple(ip.image_mean), tuple(ip.image_std)

        clip = self.net.clip
        clip.requires_grad_(False)
        if not freeze_vision:
            clip.vision_model.requires_grad_(True)
        elif unfreeze_last_n_vision_layers:
            # The decoder reads hidden_states[i] for i in config.extract_layers (default 3, 6, 9);
            # hidden_states[i] is the output of encoder layer i-1, so layers after the deepest one never
            # reach the loss. "Last n" therefore means the n layers just below that point.
            top = max(self.net.config.extract_layers)
            for layer in clip.vision_model.encoder.layers[max(0, top - unfreeze_last_n_vision_layers):top]:
                layer.requires_grad_(True)
        if not freeze_text:
            clip.text_model.requires_grad_(True)
            clip.text_projection.requires_grad_(True)
        self.text_frozen = freeze_text
        self._text_cache = {}

    def _embed(self, prompts, device):
        tok = self.tokenizer(list(prompts), padding=True, return_tensors="pt").to(device)
        return self.net.get_conditional_embeddings(
            batch_size=len(prompts), input_ids=tok["input_ids"], attention_mask=tok["attention_mask"])

    def encode_text(self, prompts):
        device = self.net.decoder.film_mul.weight.device
        if not self.text_frozen:
            return self._embed(prompts, device)
        # frozen text encoder: each distinct prompt is embedded once and cached
        missing = sorted(set(prompts) - self._text_cache.keys())
        if missing:
            with torch.no_grad():
                for p, e in zip(missing, self._embed(missing, device)):
                    self._text_cache[p] = e.float()
        return torch.stack([self._text_cache[p] for p in prompts]).to(device)

    def forward(self, pixel_values, prompts):
        cond = self.encode_text(prompts)
        logits = self.net(pixel_values=pixel_values, conditional_embeddings=cond).logits
        if logits.dim() == 2:          # HF squeezes the batch dim when B == 1
            logits = logits[None]
        return logits[:, None]
