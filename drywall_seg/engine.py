"""Training loop and optimiser / scheduler factories."""
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

from .config import save_config
from .data import PromptBank, make_loader
from .evaluation import apply_thresholds, choose_thresholds, evaluate, summarize
from .losses import build_loss
from .models import build_model
from .utils import count_params, get_device, model_size_mb, save_json, seed_everything


def build_optimizer(model, cfg):
    groups = model.param_groups(cfg.lr, cfg.get("backbone_lr", cfg.lr), cfg.get("weight_decay", 0.0))
    if cfg.name == "adamw":
        return torch.optim.AdamW(groups, betas=tuple(cfg.get("betas", (0.9, 0.999))))
    if cfg.name == "sgd":
        return torch.optim.SGD(groups, momentum=cfg.get("momentum", 0.9), nesterov=True)
    raise ValueError(f"unknown optimizer {cfg.name!r} (adamw | sgd)")


def build_scheduler(optimizer, cfg, steps_per_epoch, epochs):
    """Linear warmup, then cosine decay to min_lr_ratio·lr (or constant)."""
    if cfg.name not in ("cosine", "constant"):
        raise ValueError(f"unknown scheduler {cfg.name!r} (cosine | constant)")
    total = steps_per_epoch * epochs
    warm = int(cfg.get("warmup_epochs", 0) * steps_per_epoch)
    floor = cfg.get("min_lr_ratio", 0.0)

    def factor(step):
        if step < warm:
            return (step + 1) / warm
        if cfg.name == "constant":
            return 1.0
        progress = min(1.0, (step - warm) / max(1, total - warm))
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, factor)


def _fmt_val(summary):
    per = " / ".join(f"{c} {v['iou']:.3f}" for c, v in summary["per_class"].items())
    return f"val mIoU {summary['mIoU']:.3f} ({per})"


def train(cfg):
    seed_everything(int(cfg.seed), cfg.get("deterministic", False))
    device = get_device(cfg.get("device", "auto"))
    run_dir = Path(cfg.output_dir) / f"{cfg.experiment}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "config.yaml")

    df = pd.read_csv(cfg.data.index_csv)
    bank = PromptBank.from_config(cfg.prompts)
    model = build_model(cfg.model).to(device)
    train_loader = make_loader(cfg, df, "train", model, bank, train=True)
    val_loader = make_loader(cfg, df, "val", model, bank, train=False)

    criterion = build_loss(cfg.loss).to(device)
    optimizer = build_optimizer(model, cfg.optim)
    epochs = int(cfg.train.epochs)
    steps = len(train_loader)
    if cfg.train.get("max_steps_per_epoch"):
        steps = min(steps, int(cfg.train.max_steps_per_epoch))
    scheduler = build_scheduler(optimizer, cfg.scheduler, steps, epochs)
    use_amp = bool(cfg.train.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    trainable = [p for p in model.parameters() if p.requires_grad]
    thresholds = list(cfg.eval.thresholds)

    total_p, train_p = count_params(model)
    print(f"run: {run_dir}\ndevice: {device} | amp: {use_amp} | params: {total_p / 1e6:.1f}M total, "
          f"{train_p / 1e6:.2f}M trainable | train batches/epoch: {steps} | val pairs: {len(val_loader.dataset)}")

    best, best_epoch, bad = -1.0, 0, 0
    t_start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        t0, sums, n = time.time(), defaultdict(float), 0
        bar = tqdm(train_loader, total=steps, desc=f"epoch {epoch}/{epochs}", leave=False)
        for step, batch in enumerate(bar):
            if step >= steps:
                break
            x = batch["pixel_values"].to(device, non_blocking=True)
            y = batch["target"].to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x, batch["prompt"])
            logits = logits.float()
            if logits.shape[-2:] != y.shape[-2:]:
                logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
            loss, parts = criterion(logits, y)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if cfg.train.get("grad_clip"):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, float(cfg.train.grad_clip))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            sums["loss"] += loss.item()
            for k, v in parts.items():
                sums[k] += v
            n += 1
            bar.set_postfix(loss=f"{sums['loss'] / n:.4f}")

        log = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"], "train_time_s": time.time() - t0,
               **{f"train/{k}": v / max(n, 1) for k, v in sums.items()}}
        line = f"epoch {epoch:>3} | loss {log['train/loss']:.4f}"

        if epoch % int(cfg.eval.get("every", 1)) == 0 or epoch == epochs:
            val = evaluate(model, val_loader, device, thresholds, use_amp)
            thr = choose_thresholds(val, thresholds)
            summary = summarize(apply_thresholds(val, thr))
            log["val/mIoU"], log["val/mDice"] = summary["mIoU"], summary["mDice"]
            for c, v in summary["per_class"].items():
                log.update({f"val/{c}_iou": v["iou"], f"val/{c}_dice": v["dice"],
                            f"val/{c}_leak": v["neg_leak"], f"val/{c}_thr": v["threshold"]})
            line += f" | {_fmt_val(summary)}"
            if summary["mIoU"] > best:
                best, best_epoch, bad = summary["mIoU"], epoch, 0
                torch.save({"model": model.trainable_state_dict(), "thresholds": thr,
                            "epoch": epoch, "val": summary}, run_dir / "best.pt")
                save_json(thr, run_dir / "thresholds.json")
                line += "  ← best"
            else:
                bad += 1

        print(f"{line} | {time.time() - t0:.0f}s")
        with open(run_dir / "history.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(log, default=float) + "\n")
        patience = cfg.train.get("early_stop_patience")
        if patience and bad >= int(patience):
            print(f"early stop: no val improvement for {patience} evaluations")
            break

    info = dict(train_time_s=time.time() - t_start, epochs_run=epoch, best_epoch=best_epoch, best_val_mIoU=best,
                params_total=total_p, params_trainable=train_p, model_size_mb_fp32=model_size_mb(model),
                checkpoint_mb=(run_dir / "best.pt").stat().st_size / 2**20, device=str(device),
                gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None)
    save_json(info, run_dir / "train_info.json")
    print(f"done in {info['train_time_s'] / 60:.1f} min | best val mIoU {best:.3f} @ epoch {best_epoch} | {run_dir}")
    return run_dir
