from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Sequence, Iterable
import copy, csv, gc, json, math, random, time
import cv2, numpy as np, torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def read_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图像: {path}")
    return image


def read_mask(path: str | Path) -> np.ndarray:
    path = Path(path)
    mask = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"无法读取掩膜: {path}")
    return mask


def write_image(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".png"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise ValueError(f"无法编码图像: {path}")
    encoded.tofile(path)
