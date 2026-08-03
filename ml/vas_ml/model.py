"""The classifier: a small depthwise-separable CNN over the (40, 49) log-mel patch.

Deliberately built from five ops only — conv2d (dense and depthwise), batchnorm,
ReLU, global average pool, and one linear layer. Batchnorm folds into the
preceding convolution at export, so the deployed graph is just conv/ReLU/pool/
linear. That is a small enough surface to reimplement exactly in JS (for the
browser) and in C (for the ESP32) without pulling in an interpreter, which is what
makes the same weights runnable in all three places.

~35k parameters. The constraint is not really parameter count on an ESP32-S3, it
is peak activation memory, which is why the first convolution strides immediately.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import CLASSES, DSP_CFG


class ConvBNReLU(nn.Sequential):
    def __init__(self, cin: int, cout: int, k: int = 3, stride: int = 1, groups: int = 1):
        super().__init__(
            nn.Conv2d(cin, cout, k, stride=stride, padding=k // 2, groups=groups, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


class SeparableConv(nn.Sequential):
    """3x3 depthwise then 1x1 pointwise — ~8x cheaper than a dense 3x3 at equal width."""

    def __init__(self, cin: int, cout: int, stride: int = 1):
        super().__init__(
            ConvBNReLU(cin, cin, k=3, stride=stride, groups=cin),
            ConvBNReLU(cin, cout, k=1),
        )


class SoundCNN(nn.Module):
    def __init__(self, n_classes: int = len(CLASSES), width: int = 1.0, dropout: float = 0.2):
        super().__init__()
        w = lambda c: max(8, int(c * width))  # noqa: E731
        self.features = nn.Sequential(
            # (1, 40, 49). The two stride-2 stages come before the network gets
            # wide, which caps peak activation memory at 128 x 10 x 13 = 16640
            # floats. Widening at full 20x25 resolution instead would double the
            # firmware's arena to 128 kB per buffer and not fit beside WiFi on an
            # S3 — see vas_nn_required_arena() in firmware/esp32/vas_nn.c.
            ConvBNReLU(1, w(32), k=3, stride=2),        # (32, 20, 25)
            SeparableConv(w(32), w(64), stride=2),      # (64, 10, 13)
            SeparableConv(w(64), w(64), stride=1),      # (64, 10, 13)
            SeparableConv(w(64), w(128), stride=1),     # (128, 10, 13)
            SeparableConv(w(128), w(128), stride=2),    # (128, 5, 7)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(w(128), n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, n_bins, n_frames) normalised features -> (B, n_classes) logits."""
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.fc(self.drop(x))


def build(**kw) -> SoundCNN:
    return SoundCNN(**kw)


def summarise(model: nn.Module) -> str:
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    with torch.no_grad():
        x = torch.zeros(1, 1, *DSP_CFG.shape)
        y = model(x)
    return (f"SoundCNN  in {tuple(x.shape)[1:]}  out {tuple(y.shape)[1:]}  "
            f"params {total:,} ({train:,} trainable)")


if __name__ == "__main__":
    m = build()
    print(summarise(m))
