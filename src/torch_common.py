"""PyTorch utilities shared by Method B (scratch CNN) and Method C (pretrained).

Using one training loop, one augmentation function and one evaluation routine
for both neural methods keeps the comparison fair: the only difference between
B and C is the network (and its pretrained initialisation).
"""
from __future__ import annotations

import copy
import math
import os
import subprocess
import sys
import time


def _cuda_works() -> bool:
    """Probe CUDA in a child process. Some laptops report CUDA as available but
    fail on first use (e.g. a driver older than the CUDA runtime in the wheel).
    If the probe fails we hide the GPU *before* torch is imported so PyTorch
    cleanly falls back to the CPU."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "-1":
        return False
    code = "import torch,sys; x=torch.ones(8,device='cuda'); sys.exit(0 if (x*2).sum().item()==16 else 1)"
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=120)
        return r.returncode == 0
    except Exception:
        return False


if "torch" not in sys.modules and not _cuda_works():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import (NUM_CLASSES, SEED, classification_metrics, labels_of, load_array_cache,
                    set_seed)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# =============================================================================
# Data
# =============================================================================
def load_uint8(items, size: int, tag: str) -> torch.Tensor:
    """N x 3 x size x size uint8 tensor on the CPU (cached as .npy on disk)."""
    arr = load_array_cache(items, size, tag)
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()


def to_float(x_uint8: torch.Tensor, mean, std) -> torch.Tensor:
    x = x_uint8.float().div_(255.0)
    m = torch.tensor(mean, device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=x.device).view(1, 3, 1, 1)
    return (x - m) / s


# =============================================================================
# GPU augmentation (written by the group; every sample gets its own random draw)
# =============================================================================
def gpu_augment(x: torch.Tensor, out_size: int, strength: float = 1.0) -> torch.Tensor:
    """x: float tensor in [0, 1], N x 3 x H x W on DEVICE.

    One random affine transform per image combines
      * random zoom crop (area 50-100 % of the image, aspect 3/4-4/3)
      * random rotation (+-25 degrees; trash photos have no canonical orientation)
      * random horizontal and vertical flips
    followed by per-image colour jitter (brightness, contrast, saturation, small
    hue-free tint) and random erasing (cut-out).
    """
    n = x.shape[0]
    dev = x.device
    # --- geometry ---------------------------------------------------------
    area = torch.empty(n, device=dev).uniform_(1 - 0.5 * strength, 1.0)
    logr = torch.empty(n, device=dev).uniform_(math.log(3 / 4), math.log(4 / 3))
    ratio = torch.exp(logr)
    sx = torch.sqrt(area * ratio).clamp(max=1.0)     # fraction of width kept
    sy = torch.sqrt(area / ratio).clamp(max=1.0)
    tx = (torch.rand(n, device=dev) * 2 - 1) * (1 - sx)   # centre offset, normalised coords
    ty = (torch.rand(n, device=dev) * 2 - 1) * (1 - sy)
    ang = (torch.rand(n, device=dev) * 2 - 1) * math.radians(25) * strength
    fx = torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
    fy = torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
    cos, sin = torch.cos(ang), torch.sin(ang)
    theta = torch.zeros(n, 2, 3, device=dev)
    theta[:, 0, 0] = cos * sx * fx
    theta[:, 0, 1] = -sin * sy * fy
    theta[:, 1, 0] = sin * sx * fx
    theta[:, 1, 1] = cos * sy * fy
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, (n, 3, out_size, out_size), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    # --- photometric --------------------------------------------------------
    j = 0.3 * strength
    b = torch.empty(n, 1, 1, 1, device=dev).uniform_(1 - j, 1 + j)
    c = torch.empty(n, 1, 1, 1, device=dev).uniform_(1 - j, 1 + j)
    s = torch.empty(n, 1, 1, 1, device=dev).uniform_(1 - j, 1 + j)
    x = x * b
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * c + mean
    gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
    x = (x - gray) * s + gray
    tint = torch.empty(n, 3, 1, 1, device=dev).uniform_(1 - 0.05 * strength, 1 + 0.05 * strength)
    x = (x * tint).clamp_(0, 1)
    # --- random erasing (p = 0.25, 2-15 % of the area) -----------------------
    erase = torch.rand(n, device=dev) < 0.25 * strength
    if erase.any():
        H = W = out_size
        ea = torch.empty(n, device=dev).uniform_(0.02, 0.15)
        eh = (torch.sqrt(ea) * H).long().clamp(min=1)
        ew = (torch.sqrt(ea) * W).long().clamp(min=1)
        y0 = (torch.rand(n, device=dev) * (H - eh)).long()
        x0 = (torch.rand(n, device=dev) * (W - ew)).long()
        yy = torch.arange(H, device=dev).view(1, H, 1)
        xx = torch.arange(W, device=dev).view(1, 1, W)
        m = ((yy >= y0.view(-1, 1, 1)) & (yy < (y0 + eh).view(-1, 1, 1)) &
             (xx >= x0.view(-1, 1, 1)) & (xx < (x0 + ew).view(-1, 1, 1)) & erase.view(-1, 1, 1))
        x = torch.where(m.unsqueeze(1), torch.rand_like(x), x)
    return x


# =============================================================================
# Training loop
# =============================================================================
def class_weights(y: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y, minlength=NUM_CLASSES).astype(np.float64)
    w = len(y) / (NUM_CLASSES * np.maximum(counts, 1))
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def predict_logits(model, x_uint8: torch.Tensor, size: int, mean, std, bs: int = 64):
    """Deterministic evaluation: resize to the model's input size, normalise."""
    model.eval()
    out = []
    for i in range(0, len(x_uint8), bs):
        xb = x_uint8[i:i + bs].to(DEVICE)
        if xb.shape[-1] != size:
            xb = F.interpolate(xb.float(), size=(size, size), mode="bilinear",
                               align_corners=False, antialias=True).round().clamp(0, 255)
        out.append(model(to_float(xb, mean, std)).float().cpu())
    return torch.cat(out)


def train_model(model, train_items, val_items, *, size, cache_size=None, mean, std,
                epochs=100, batch_size=32, lr=2e-3, weight_decay=0.05,
                label_smoothing=0.1, warmup_epochs=3, patience=25, aug_strength=1.0,
                param_groups=None, freeze_epochs=0, head_params=None, head_lr=None,
                tag="model", seed=SEED, verbose=True, log_every=5, eval_every=1):
    """Generic supervised training with AdamW + warm-up/cosine LR, class-weighted
    label-smoothed cross-entropy, GPU augmentation and early stopping on the
    validation macro-F1 (the best checkpoint is restored at the end).

    If freeze_epochs > 0, only `head_params` are trained for the first
    freeze_epochs epochs (used by the pretrained model: head first, then all).
    """
    set_seed(seed)
    cache_size = cache_size or size
    # parameters trained in the "full" phase (default: all of them)
    trainable = list(param_groups) if param_groups is not None else list(model.parameters())
    head_params = list(head_params) if head_params is not None else None
    xtr = load_uint8(train_items, cache_size, f"train_{tag}")
    xva = load_uint8(val_items, cache_size, f"val_{tag}")
    ytr = torch.tensor(labels_of(train_items))
    yva = labels_of(val_items)
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights(labels_of(train_items)).to(DEVICE),
                                    label_smoothing=label_smoothing)
    steps_per_epoch = max(1, math.ceil(len(xtr) / batch_size))
    hist = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
            "val_macro_f1": [], "lr": [], "time_s": []}
    best = {"f1": -1, "state": None, "epoch": -1}
    t_start = time.perf_counter()

    def make_opt(params, base_lr, total_epochs, warm):
        opt = torch.optim.AdamW(params, lr=base_lr, weight_decay=weight_decay)
        total = total_epochs * steps_per_epoch
        warm_steps = warm * steps_per_epoch

        def f(step):
            if step < warm_steps:
                return (step + 1) / warm_steps
            p = (step - warm_steps) / max(1, total - warm_steps)
            return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
        return opt, torch.optim.lr_scheduler.LambdaLR(opt, f)

    phase = "head" if freeze_epochs > 0 else "full"
    if phase == "head":
        for p in model.parameters():
            p.requires_grad_(False)
        for p in head_params:
            p.requires_grad_(True)
        opt, sched = make_opt(head_params, head_lr or lr, freeze_epochs, 0)
    else:
        opt, sched = make_opt(trainable, lr, epochs, warmup_epochs)
    bad = 0
    for ep in range(1, epochs + freeze_epochs + 1):
        if phase == "head" and ep == freeze_epochs + 1:
            phase = "full"
            for p in trainable:
                p.requires_grad_(True)
            opt, sched = make_opt(trainable, lr, epochs, warmup_epochs)
            bad = 0
        model.train()
        perm = torch.randperm(len(xtr))
        tot_loss, tot_correct, seen = 0.0, 0, 0
        for i in range(0, len(xtr), batch_size):
            idx = perm[i:i + batch_size]
            if len(idx) < 2:
                continue   # BatchNorm needs >1 sample
            xb = xtr[idx].to(DEVICE, non_blocking=True).float().div_(255.0)
            xb = gpu_augment(xb, size, aug_strength)
            m_ = torch.tensor(mean, device=DEVICE).view(1, 3, 1, 1)
            s_ = torch.tensor(std, device=DEVICE).view(1, 3, 1, 1)
            xb = (xb - m_) / s_
            yb = ytr[idx].to(DEVICE)
            logits = model(xb)
            loss = criterion(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot_loss += loss.item() * len(idx)
            tot_correct += (logits.argmax(1) == yb).sum().item()
            seen += len(idx)
        # ---- validation (every `eval_every` epochs, always on the last one) ----
        last_epoch = ep == epochs + freeze_epochs
        if ep % eval_every != 0 and not last_epoch:
            continue
        vlogits = predict_logits(model, xva, size, mean, std)
        vloss = F.cross_entropy(vlogits, torch.tensor(yva)).item()
        vpred = vlogits.argmax(1).numpy()
        vm = classification_metrics(yva, vpred)
        hist["epoch"].append(ep)
        hist["train_loss"].append(tot_loss / max(seen, 1))
        hist["train_acc"].append(tot_correct / max(seen, 1) * 100)
        hist["val_loss"].append(vloss)
        hist["val_acc"].append(vm["accuracy"])
        hist["val_macro_f1"].append(vm["macro_f1"])
        hist["lr"].append(opt.param_groups[0]["lr"])
        hist["time_s"].append(time.perf_counter() - t_start)
        improved = vm["macro_f1"] > best["f1"] + 1e-4
        if improved:
            best = {"f1": vm["macro_f1"], "epoch": ep,
                    "state": copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})}
            bad = 0
        elif phase == "full":
            bad += eval_every
        if verbose and (ep % log_every == 0 or ep == 1 or improved):
            print(f"  [{tag}] ep {ep:3d} ({phase}) train_loss {hist['train_loss'][-1]:.3f} "
                  f"train_acc {hist['train_acc'][-1]:5.1f} | val_loss {vloss:.3f} "
                  f"val_acc {vm['accuracy']:5.1f} val_F1 {vm['macro_f1']:.3f}"
                  f"{'  *' if improved else ''}  ({hist['time_s'][-1]:.0f}s)")
        if phase == "full" and bad >= patience:
            if verbose:
                print(f"  [{tag}] early stop at epoch {ep} (best epoch {best['epoch']})")
            break
    train_min = (time.perf_counter() - t_start) / 60
    model.load_state_dict(best["state"])
    return model, hist, {"best_epoch": best["epoch"], "best_val_macro_f1": best["f1"],
                         "train_time_min": train_min, "epochs_run": hist["epoch"][-1]}


def evaluate(model, items, size, mean, std, tag, cache_size=None):
    x = load_uint8(items, cache_size or size, tag)
    logits = predict_logits(model, x, size, mean, std)
    pred = logits.argmax(1).numpy()
    return classification_metrics(labels_of(items), pred), pred, torch.softmax(logits, 1).numpy()


def count_params(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def runtime_ms(model, items, size, mean, std, device="cpu", n=200, warmup=10,
               cache_size=None, tag="test"):
    """Per-image latency, batch size 1, includes normalisation; excludes JPEG decode."""
    import copy as _c
    from common import time_per_image
    dev = torch.device(device)
    m = _c.deepcopy(model).to(dev).eval()
    x = load_uint8(items, cache_size or size, tag)
    if x.shape[-1] != size:
        x = F.interpolate(x.float(), size=(size, size), mode="bilinear", align_corners=False,
                          antialias=True).round().clamp(0, 255).to(torch.uint8)
    mean_t = torch.tensor(mean, device=dev).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=dev).view(1, 3, 1, 1)

    @torch.no_grad()
    def one(img):
        xb = img.unsqueeze(0).to(dev).float().div(255.0)
        out = m((xb - mean_t) / std_t)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        return out
    return time_per_image(one, list(x[:n + warmup]), warmup=warmup, n=n)
