"""Online random crops; shared geometry and image-only brightness augmentation.

The targets supplied here are SAM masks fixed for the current learning stage.
No pre-generated crop archive or manual training references are required.
"""

from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from .image_io import read_image, read_mask
from .network import tensor_image


class PairedCropDataset(Dataset):
    def __init__(self, image_paths, mask_dir, config, repeats=2):
        self.paths = list(map(Path, image_paths))
        self.config, self.repeats, self.cache = config, repeats, []
        for path in self.paths:
            image = read_image(path)
            mask = read_mask(Path(mask_dir) / (path.stem + ".png"))
            if image.shape[:2] != mask.shape:
                raise ValueError(f"Image/mask dimensions differ: {path.name}")
            self.cache.append((image, mask))
        if not self.paths or repeats < 1:
            raise ValueError("Training requires nonempty images and positive repeats")

    def __len__(self):
        return len(self.paths) * self.repeats

    def __getitem__(self, index):
        image, mask = self.cache[index % len(self.paths)]
        h, w = mask.shape
        c = self.config
        # Preserve the random draw order of the recorded experiments.
        side = min(h, w, int(c.crop_size * np.random.uniform(*c.crop_scale)))
        y, x = np.random.randint(h - side + 1), np.random.randint(w - side + 1)
        image = cv2.resize(
            image[y : y + side, x : x + side],
            (c.crop_size, c.crop_size),
            interpolation=cv2.INTER_LINEAR,
        )
        mask = cv2.resize(
            mask[y : y + side, x : x + side],
            (c.crop_size, c.crop_size),
            interpolation=cv2.INTER_NEAREST,
        )
        rotation = np.random.randint(4)
        image, mask = np.rot90(image, rotation), np.rot90(mask, rotation)
        if np.random.random() < c.flip_probability:
            image, mask = image[:, ::-1], mask[:, ::-1]
        image = np.clip(
            image.astype(np.float32) * np.random.uniform(*c.brightness_scale)
            + np.random.uniform(*c.brightness_shift),
            0,
            255,
        ).astype("uint8")
        return (
            tensor_image(np.ascontiguousarray(image)),
            torch.from_numpy(np.ascontiguousarray(mask > 127, dtype=np.float32))[None],
            self.paths[index % len(self.paths)].stem,
        )
