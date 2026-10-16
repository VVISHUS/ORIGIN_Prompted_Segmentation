"""Loss registry + weighted sum. Config:

    loss:
      terms:
        - {name: bce, weight: 1.0, pos_weight: 2.0}
        - {name: dice, weight: 1.0}

All losses take logits [B,1,H,W] and soft targets in [0,1] of the same shape.
"""
import torch
import torch.nn.functional as F
from torch import nn

LOSSES = {}


def register(name):
    def deco(cls):
        LOSSES[name] = cls
        return cls
    return deco


@register("bce")
class BCE(nn.Module):
    """Per-pixel cross-entropy. pos_weight > 1 up-weights foreground pixels (cracks cover ~3%)."""

    def __init__(self, pos_weight=1.0):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))

    def forward(self, logits, target):
        return F.binary_cross_entropy_with_logits(logits, target, pos_weight=self.pos_weight)


@register("dice")
class Dice(nn.Module):
    """1 − soft Dice per image, averaged. Measures overlap, so it ignores the large background."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):
        p = logits.sigmoid().flatten(1)
        t = target.flatten(1)
        inter = (p * t).sum(1)
        return (1 - (2 * inter + self.smooth) / (p.sum(1) + t.sum(1) + self.smooth)).mean()


@register("focal")
class Focal(nn.Module):
    """BCE that down-weights easy pixels (Lin et al., 2017)."""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p = logits.sigmoid()
        p_t = p * target + (1 - p) * (1 - target)
        a_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        return (a_t * (1 - p_t) ** self.gamma * bce).mean()


@register("tversky")
class Tversky(nn.Module):
    """Dice generalisation; beta > alpha penalises missed pixels (FN) more than extra ones (FP),
    which helps thin structures that a model tends to under-segment."""

    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super().__init__()
        self.alpha, self.beta, self.smooth = alpha, beta, smooth

    def forward(self, logits, target):
        p = logits.sigmoid().flatten(1)
        t = target.flatten(1)
        tp = (p * t).sum(1)
        fp = (p * (1 - t)).sum(1)
        fn = ((1 - p) * t).sum(1)
        return (1 - (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)).mean()


class CompositeLoss(nn.Module):
    def __init__(self, terms):
        super().__init__()
        self.names = [t["name"] for t in terms]
        self.weights = [float(t.get("weight", 1.0)) for t in terms]
        self.terms = nn.ModuleList()
        for t in terms:
            kwargs = {k: v for k, v in t.items() if k not in ("name", "weight")}
            if t["name"] not in LOSSES:
                raise KeyError(f"unknown loss {t['name']!r}; registered: {sorted(LOSSES)}")
            self.terms.append(LOSSES[t["name"]](**kwargs))

    def forward(self, logits, target):
        parts = {n: fn(logits, target) for n, fn in zip(self.names, self.terms)}
        total = sum(w * parts[n] for n, w in zip(self.names, self.weights))
        return total, {n: v.item() for n, v in parts.items()}


def build_loss(cfg_loss):
    return CompositeLoss([dict(t) for t in cfg_loss.terms])
