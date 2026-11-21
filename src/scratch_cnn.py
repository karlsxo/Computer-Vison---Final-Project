"""Method B (trained from scratch): our own 4-block CNN in PyTorch.

Architecture (designed by the group, random initialisation, no pretrained weights)

    input 3 x 128 x 128
    Block 1: Conv3x3(3->32)   - BN - ReLU                               - MaxPool2  -> 32 x 64 x 64
    Block 2: Conv3x3(32->64)  - BN - ReLU                               - MaxPool2  -> 64 x 32 x 32
    Block 3: Conv3x3(64->128) - BN - ReLU - Conv3x3(128->128) - BN - ReLU - MaxPool2 -> 128 x 16 x 16
    Block 4: Conv3x3(128->256)- BN - ReLU - Conv3x3(256->256) - BN - ReLU - MaxPool2 -> 256 x 8 x 8
    Global average pooling -> Dropout(0.4) -> Linear(256 -> 6)

Blocks 3-4 use two convolutions because that is where the receptive field grows
large enough to see whole objects; doubling the early blocks would roughly
double the compute for little gain (measured: 209 vs 124 s/epoch on CPU).

Overfitting controls: GPU data augmentation (zoom-crop, rotation, flips, colour
jitter, random erasing), dropout, AdamW weight decay, label smoothing, and early
stopping on validation macro-F1 with the best checkpoint restored.

Run:  python src/scratch_cnn.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch_common as tc   # noqa: E402  (probes CUDA before torch is imported)
import torch                # noqa: E402
import torch.nn as nn       # noqa: E402

from common import MODELS_DIR, RESULTS_DIR, load_split, save_json, set_seed  # noqa: E402

IMG_SIZE = 128
MEAN = (0.5, 0.5, 0.5)      # from-scratch model: simple symmetric normalisation
STD = (0.25, 0.25, 0.25)

# Attempt 1 (aug_strength=1.0, wd 0.05, smoothing 0.1, lr 2e-3) UNDER-fitted: after 15 epochs
# train acc 47 % <= val acc 53 % (log: results/log_method_B_attempt1_strong_aug.txt).
# Attempt 2 (final): milder augmentation and regularisation, slightly higher LR.
HPARAMS = dict(epochs=40, batch_size=32, lr=3e-3, weight_decay=0.02, label_smoothing=0.05,
               warmup_epochs=2, patience=12, aug_strength=0.6)


def conv_bn_relu(cin, cout):
    return [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]


class TrashCNN(nn.Module):
    def __init__(self, num_classes=6, widths=(32, 64, 128, 256), double=(False, False, True, True),
                 dropout=0.4):
        super().__init__()
        layers, cin = [], 3
        for w, d in zip(widths, double):
            block = conv_bn_relu(cin, w)
            if d:
                block += conv_bn_relu(w, w)
            block.append(nn.MaxPool2d(2))
            layers.append(nn.Sequential(*block))
            cin = w
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(cin, num_classes)
        for m in self.modules():   # He initialisation for ReLU networks
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01); nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        return self.fc(self.dropout(torch.flatten(self.pool(x), 1)))


def build_model():
    return TrashCNN()


def train(train_items, val_items, tag="cnn", **overrides):
    hp = dict(HPARAMS, **overrides)
    model = build_model()
    return tc.train_model(model, train_items, val_items, size=IMG_SIZE, mean=MEAN, std=STD,
                          tag=tag, **hp)


def load_trained(path=MODELS_DIR / "scratch_cnn.pt"):
    m = build_model()
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.eval()


@torch.no_grad()
def predict_proba(model, img_uint8_hwc):
    """img: HxWx3 uint8 (any size) -> class probabilities."""
    x = torch.from_numpy(img_uint8_hwc).permute(2, 0, 1).unsqueeze(0)
    logits = tc.predict_logits(model, x, IMG_SIZE, MEAN, STD)
    return torch.softmax(logits, 1)[0].numpy()


def main():
    set_seed()
    print(f"=== Method B: 4-block CNN from scratch (device: {tc.DEVICE}) ===")
    split = load_split()
    model = build_model()
    print(model)
    print(f"Parameters: {tc.count_params(model):.3f} M")
    model, hist, info = train(split["train"], split["val"], tag="cnn")
    torch.save(model.state_dict(), MODELS_DIR / "scratch_cnn.pt")
    test_met, pred, proba = tc.evaluate(model, split["test"], IMG_SIZE, MEAN, STD, "test_128")
    rt_cpu = tc.runtime_ms(model, split["test"], IMG_SIZE, MEAN, STD, "cpu", tag="test_128")
    rt_gpu = (tc.runtime_ms(model, split["test"], IMG_SIZE, MEAN, STD, "cuda", tag="test_128")
              if torch.cuda.is_available() else None)
    print(f"TEST accuracy={test_met['accuracy']:.2f}%  macro-F1={test_met['macro_f1']:.4f}  "
          f"runtime CPU={rt_cpu:.2f} ms  params={tc.count_params(model):.3f} M  "
          f"train={info['train_time_min']:.1f} min (device {tc.DEVICE})")
    save_json({"method": "4-block CNN (scratch)", "slot": "B. From scratch",
               "hparams": HPARAMS, "architecture": str(model), "train_info": info,
               "history": hist, "test": test_met, "test_pred": pred.tolist(),
               "runtime_ms_cpu": rt_cpu, "runtime_ms_gpu": rt_gpu,
               "params_M": tc.count_params(model), "train_time_min": info["train_time_min"],
               "train_device": str(tc.DEVICE)}, RESULTS_DIR / "method_B.json")


if __name__ == "__main__":
    main()
