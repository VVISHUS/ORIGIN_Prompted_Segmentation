from torch import nn


class PromptSegModel(nn.Module):
    """Interface every model implements.

    forward(pixel_values [B,3,S,S], prompts list[str]) -> logits [B,1,h,w]
        pixel_values are resized to `input_size` and normalised with `image_mean` / `image_std`.
        h, w may differ from S; callers upsample logits to the target size.
    Parameters whose names start with one of `backbone_prefixes` get `optim.backbone_lr`.
    """

    input_size = 352
    image_mean = (0.485, 0.456, 0.406)
    image_std = (0.229, 0.224, 0.225)
    backbone_prefixes = ()

    def param_groups(self, lr, backbone_lr, weight_decay):
        head, backbone = [], []
        for name, p in self.named_parameters():
            if p.requires_grad:
                (backbone if name.startswith(self.backbone_prefixes) else head).append(p)
        groups = [dict(params=head, lr=lr, weight_decay=weight_decay),
                  dict(params=backbone, lr=backbone_lr, weight_decay=weight_decay)]
        return [g for g in groups if g["params"]]

    def trainable_state_dict(self):
        """Only the weights training changes; frozen weights come from the pretrained checkpoint."""
        names = {n for n, p in self.named_parameters() if p.requires_grad}
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if k in names}
