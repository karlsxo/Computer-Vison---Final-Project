"""Method C (pretrained, modern): MobileNetV3-Large (Howard et al., ICCV 2019),
ImageNet weights from torchvision, fine-tuned on the same TrashNet split.

* Weights: torchvision MobileNet_V3_Large_Weights.IMAGENET1K_V2.
* Output format: the 1000-class ImageNet layer is replaced by a 6-class layer.
* Correct preprocessing: 224 x 224 input, ImageNet mean/std normalisation (the
  statistics these weights were trained with).
* Training plan: (1) train only the classifier head for a few epochs with the
  backbone frozen, so the randomly initialised head does not destroy the
  pretrained features; (2) unfreeze and fine-tune the network at a small learning
  rate. The stem and first three inverted-residual blocks (features[0:4]) stay
  frozen - they hold generic edge/colour filters and, on high-resolution maps,
  dominate the compute (CPU training: 215 vs 320 s/epoch). Their BatchNorm
  layers are kept in eval mode so their ImageNet statistics are not overwritten.
* Same augmentation function, loss, optimiser and early stopping as Method B.

Run:  python src/pretrained_mobilenet.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch_common as tc   # noqa: E402
import torch                # noqa: E402
import torch.nn as nn       # noqa: E402
import torchvision          # noqa: E402

from common import MODELS_DIR, NUM_CLASSES, RESULTS_DIR, load_split, save_json, set_seed  # noqa: E402

IMG_SIZE = 224
MEAN, STD = tc.IMAGENET_MEAN, tc.IMAGENET_STD
FROZEN_BLOCKS = 4
VARIANT = "large"

HPARAMS = dict(freeze_epochs=3, head_lr=1e-3, epochs=12, batch_size=32, lr=5e-4,
               weight_decay=0.01, label_smoothing=0.05, warmup_epochs=1, patience=6,
               aug_strength=0.6)   # same augmentation strength / smoothing as Method B


class MobileNetTrash(nn.Module):
    def __init__(self, variant=VARIANT, pretrained=True, frozen_blocks=FROZEN_BLOCKS):
        super().__init__()
        if variant == "large":
            w = torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
            net = torchvision.models.mobilenet_v3_large(weights=w)
        else:
            w = torchvision.models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            net = torchvision.models.mobilenet_v3_small(weights=w)
        in_f = net.classifier[3].in_features
        net.classifier[3] = nn.Linear(in_f, NUM_CLASSES)        # new 6-class output layer
        self.net = net
        self.frozen_blocks = frozen_blocks
        for p in self.net.features[:frozen_blocks].parameters():
            p.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.net.features[:self.frozen_blocks].eval()   # keep frozen BN statistics fixed
        return self

    def head_parameters(self):
        return list(self.net.classifier.parameters())

    def trainable_parameters(self):
        return [p for p in self.net.features[self.frozen_blocks:].parameters()] + self.head_parameters()

    def forward(self, x):
        return self.net(x)


def build_model(pretrained=True, variant=VARIANT):
    return MobileNetTrash(variant=variant, pretrained=pretrained)


def train(train_items, val_items, tag="mnv3", variant=VARIANT, **overrides):
    hp = dict(HPARAMS, **overrides)
    model = build_model(variant=variant)
    return tc.train_model(model, train_items, val_items, size=IMG_SIZE, mean=MEAN, std=STD,
                          param_groups=model.trainable_parameters(),
                          head_params=model.head_parameters(), tag=tag, **hp)


def load_trained(path=MODELS_DIR / "mobilenetv3_large.pt", variant=VARIANT):
    m = build_model(pretrained=False, variant=variant)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.eval()


@torch.no_grad()
def predict_proba(model, img_uint8_hwc):
    x = torch.from_numpy(img_uint8_hwc).permute(2, 0, 1).unsqueeze(0)
    logits = tc.predict_logits(model, x, IMG_SIZE, MEAN, STD)
    return torch.softmax(logits, 1)[0].numpy()


def main():
    set_seed()
    print(f"=== Method C: MobileNetV3-{VARIANT} pretrained + fine-tuned (device: {tc.DEVICE}) ===")
    split = load_split()
    model, hist, info = train(split["train"], split["val"], tag="mnv3")
    torch.save(model.state_dict(), MODELS_DIR / f"mobilenetv3_{VARIANT}.pt")
    test_met, pred, proba = tc.evaluate(model, split["test"], IMG_SIZE, MEAN, STD, "test_224")
    rt_cpu = tc.runtime_ms(model, split["test"], IMG_SIZE, MEAN, STD, "cpu", tag="test_224")
    rt_gpu = (tc.runtime_ms(model, split["test"], IMG_SIZE, MEAN, STD, "cuda", tag="test_224")
              if torch.cuda.is_available() else None)
    n_total = tc.count_params(model)
    n_train = sum(p.numel() for p in model.trainable_parameters()) / 1e6
    print(f"TEST accuracy={test_met['accuracy']:.2f}%  macro-F1={test_met['macro_f1']:.4f}  "
          f"runtime CPU={rt_cpu:.2f} ms  params={n_total:.3f} M (trainable {n_train:.3f} M)  "
          f"train={info['train_time_min']:.1f} min (device {tc.DEVICE})")
    save_json({"method": f"MobileNetV3-{VARIANT.capitalize()} (ImageNet-pretrained, fine-tuned)",
               "slot": "C. Pretrained", "hparams": HPARAMS, "frozen_blocks": FROZEN_BLOCKS,
               "weights": "torchvision IMAGENET1K_V2", "train_info": info, "history": hist,
               "test": test_met, "test_pred": pred.tolist(), "runtime_ms_cpu": rt_cpu,
               "runtime_ms_gpu": rt_gpu, "params_M": n_total, "trainable_params_M": n_train,
               "train_time_min": info["train_time_min"], "train_device": str(tc.DEVICE)},
              RESULTS_DIR / "method_C.json")


if __name__ == "__main__":
    main()
