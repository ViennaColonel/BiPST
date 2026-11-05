"""Validated settings; relative paths resolve against the configured base_dir."""

from copy import deepcopy
from dataclasses import dataclass, asdict
from pathlib import Path
import json


@dataclass
class TrainingConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 2
    first_epochs: int = 100
    first_patience: int = 10
    later_epochs: int = 20
    auxiliary_epochs: int = 40
    auxiliary_patience: int = 8
    views_per_image: int = 2
    auxiliary_view_floor: int = 56
    crop_size: int = 384
    crop_scale: tuple = (0.8, 1.2)
    brightness_scale: tuple = (0.8, 1.2)
    brightness_shift: tuple = (-10.0, 10.0)
    flip_probability: float = 0.5
    bce_weight: float = 1.0
    dice_weight: float = 1.0
    threshold: float = 0.5
    num_workers: int = 0
    torch_threads: int = 4
    opencv_threads: int = 2
    mixed_precision: bool = True

    def validate(self):
        for key in (
            "batch_size",
            "first_epochs",
            "first_patience",
            "later_epochs",
            "auxiliary_epochs",
            "auxiliary_patience",
            "views_per_image",
            "auxiliary_view_floor",
            "crop_size",
            "torch_threads",
            "opencv_threads",
        ):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError(f"training.{key} must be a positive integer")
        if self.crop_size < 32 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid crop size or optimizer settings")
        if (
            self.num_workers < 0
            or not 0 <= self.flip_probability <= 1
            or not 0 < self.threshold < 1
        ):
            raise ValueError("Invalid worker count or probability")
        for key in ("crop_scale", "brightness_scale", "brightness_shift"):
            values = getattr(self, key)
            if len(values) != 2 or values[0] > values[1]:
                raise ValueError(f"Invalid training.{key}")
        if self.crop_scale[0] <= 0 or self.brightness_scale[0] < 0:
            raise ValueError("Invalid scale")
        if self.bce_weight < 0 or self.dice_weight < 0 or self.bce_weight + self.dice_weight == 0:
            raise ValueError("At least one nonnegative loss weight must be positive")
        return self


def override(config, assignments):
    """Apply KEY=JSON overrides; reject misspelled keys instead of ignoring them."""
    result = deepcopy(config)
    for assignment in assignments or []:
        key, sep, value = assignment.partition("=")
        if not sep:
            raise ValueError("Use --set training.learning_rate=0.0005")
        cursor = result
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                raise ValueError("Unknown setting: " + key)
            cursor = cursor[part]
        if parts[-1] not in cursor:
            raise ValueError("Unknown setting: " + key)
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = value
        cursor[parts[-1]] = parsed
    return result


def read_config(path, assignments=()):
    path = Path(path).resolve()
    spec = override(json.loads(path.read_text("utf-8-sig")), assignments)
    cfg = TrainingConfig(**spec["training"]).validate()
    root = (path.parent / spec["base_dir"]).resolve()
    paths = {key: (root / value).resolve() for key, value in spec["paths"].items()}
    splits = json.loads(paths["splits"].read_text("utf-8-sig"))
    train, val, test = [splits["splits"][name] for name in ("train", "val", "test")]
    ids = train + val + test
    if not train or not val or len(ids) != len(set(ids)):
        raise ValueError("Train/validation must be nonempty and partitions disjoint")
    folded = [k for fold in splits["folds"] for k in fold]
    if len(set(folded)) != len(folded) or set(folded) != set(train):
        raise ValueError("Each training image must occur in exactly one auxiliary group")
    if len(splits["folds"]) < 2 or any(not f for f in splits["folds"]):
        raise ValueError("At least two nonempty auxiliary groups are required")
    return spec, cfg, paths, splits


def default_training():
    return asdict(TrainingConfig())
