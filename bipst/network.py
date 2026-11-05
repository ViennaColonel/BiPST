"""Student architecture, loss, metrics and student-only inference."""

from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Sequence, Iterable
import copy, csv, gc, json, math, random, time
import cv2, numpy as np, torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class ConvBlock(nn.Sequential):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class UpBlock(nn.Module):

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.block = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat([x, skip], dim=1))


class MobileNetV3UNet(nn.Module):
    """Paper architecture: MobileNetV3-Small skips at indices 0, 1, 3, 8, 12; bilinear decoder. Parameter names retain compatibility with released checkpoints."""

    def __init__(self, pretrained_encoder: bool = False):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained_encoder else None
        backbone = mobilenet_v3_small(weights=weights)
        self.encoder = backbone.features
        self.skip_indices = {0, 1, 3, 8, 12}
        self.up4 = UpBlock(576, 48, 128)
        self.up3 = UpBlock(128, 24, 64)
        self.up2 = UpBlock(64, 16, 32)
        self.up1 = UpBlock(32, 16, 24)
        self.head = nn.Sequential(ConvBlock(24, 16), nn.Conv2d(16, 1, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_size = x.shape[-2:]
        skips: dict[int, torch.Tensor] = {}
        for index, layer in enumerate(self.encoder):
            x = layer(x)
            if index in self.skip_indices:
                skips[index] = x
        x = self.up4(skips[12], skips[8])
        x = self.up3(x, skips[3])
        x = self.up2(x, skips[1])
        x = self.up1(x, skips[0])
        x = self.head(x)
        return F.interpolate(x, size=original_size, mode="bilinear", align_corners=False)


def build_student_model(pretrained_encoder: bool = False) -> nn.Module:
    return MobileNetV3UNet(pretrained_encoder=pretrained_encoder)


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-06) -> torch.Tensor:
    """Soft foreground Dice averaged across batch views; epsilon=1e-6."""
    probabilities = torch.sigmoid(logits)
    dims = tuple(range(1, probabilities.ndim))
    intersection = torch.sum(probabilities * targets, dim=dims)
    denominator = torch.sum(probabilities + targets, dim=dims)
    return (1.0 - (2.0 * intersection + eps) / (denominator + eps)).mean()


def _soft_boundary(values: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    dilation = F.max_pool2d(values, kernel_size, stride=1, padding=kernel_size // 2)
    erosion = -F.max_pool2d(-values, kernel_size, stride=1, padding=kernel_size // 2)
    return (dilation - erosion).clamp(0.0, 1.0)


def boundary_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    prediction_boundary = _soft_boundary(torch.sigmoid(logits))
    target_boundary = _soft_boundary(targets)
    dims = tuple(range(1, logits.ndim))
    intersection = torch.sum(prediction_boundary * target_boundary, dim=dims)
    denominator = torch.sum(prediction_boundary + target_boundary, dim=dims)
    return (1.0 - (2.0 * intersection + 1e-06) / (denominator + 1e-06)).mean()


class CombinedSegmentationLoss(nn.Module):
    """The public method uses equally weighted BCE-with-logits and soft Dice; boundary_weight remains zero."""

    def __init__(
        self, bce_weight: float = 1.0, dice_weight: float = 1.0, boundary_weight: float = 0.0
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, dict]:
        bce = self.bce(logits, targets)
        dice = dice_loss(logits, targets)
        boundary = (
            boundary_loss(logits, targets) if self.boundary_weight > 0 else logits.new_tensor(0.0)
        )
        total = self.bce_weight * bce + self.dice_weight * dice + self.boundary_weight * boundary
        return (
            total,
            {
                "bce": float(bce.detach()),
                "dice_loss": float(dice.detach()),
                "boundary_loss": float(boundary.detach()),
            },
        )


@dataclass
class BinaryMetrics:
    dice: float
    iou: float
    precision: float
    recall: float

    def as_dict(self) -> dict[str, float]:
        return {
            "dice": self.dice,
            "iou": self.iou,
            "precision": self.precision,
            "recall": self.recall,
        }


def binary_metrics(prediction: np.ndarray, target: np.ndarray, eps: float = 1e-07) -> BinaryMetrics:
    """Full-image binary overlap metrics. Returns fractions, not percentages."""
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    tp = float(np.logical_and(prediction, target).sum())
    fp = float(np.logical_and(prediction, ~target).sum())
    fn = float(np.logical_and(~prediction, target).sum())
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    return BinaryMetrics(dice, iou, precision, recall)


def model_parameter_count(model: nn.Module) -> int:
    return sum((parameter.numel() for parameter in model.parameters()))


def tensor_image(bgr):
    """Convert an OpenCV BGR image to normalized ImageNet RGB, channel-first."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
        [0.229, 0.224, 0.225], np.float32
    )
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))


@torch.inference_mode()
def predict_native(model, image, tta=False, mixed_precision=True):
    """Reflect-pad 32 pixels on all sides, extend to multiples of 32, infer, then crop to native dimensions. CUDA uses mixed precision by default; CPU uses FP32."""
    model.eval()
    h, w = image.shape[:2]
    margin = 32
    padded = cv2.copyMakeBorder(
        image, margin, margin + -h % 32, margin, margin + -w % 32, cv2.BORDER_REFLECT_101
    )
    device = next(model.parameters()).device
    x = tensor_image(padded)[None].to(device)
    with torch.amp.autocast("cuda", enabled=device.type == "cuda" and mixed_precision):
        out = torch.sigmoid(model(x)).float()
        if tta:
            out = (out + torch.sigmoid(model(x.flip(-1))).float().flip(-1)) / 2
    return out[0, 0, margin : margin + h, margin : margin + w].cpu().numpy()
