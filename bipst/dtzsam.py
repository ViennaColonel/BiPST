"""Original automatic positive-only DT-ZSAM initialization."""

from math import ceil
import cv2
import numpy as np
from sklearn.cluster import DBSCAN


def positive_points(image_bgr):
    channels = [
        cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)[:, :, 0],
    ]
    points = []
    for channel in channels:
        filtered = cv2.bilateralFilter(cv2.medianBlur(channel, 5), 9, 75, 75)
        mask = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        )
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        )
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        clean = np.zeros_like(mask)
        for j in range(1, count):
            if 50 <= int(stats[j, cv2.CC_STAT_AREA]) <= 1_000_000:
                clean[labels == j] = 255
        density = cv2.blur((clean == 0).astype(np.float32), (25, 25))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (density >= 0.70).astype("uint8"), 8
        )
        candidates = []
        for j in range(1, count):
            area = int(stats[j, cv2.CC_STAT_AREA])
            if area < 5:
                continue
            y, x = np.unravel_index(np.argmax(np.where(labels == j, density, -1.0)), density.shape)
            candidates.append((int(x), int(y), area))
        candidates.sort(key=lambda t: t[2], reverse=True)
        kept = candidates[: ceil(0.70 * len(candidates))]
        if not kept and density.max() > 0:
            y, x = np.unravel_index(np.argmax(density), density.shape)
            kept = [(int(x), int(y), 1)]
        points.extend([[x, y] for x, y, _ in kept])
    if not points:
        return np.empty((0, 2), dtype=np.int32)
    array = np.asarray(points, dtype=np.float32)
    labels = DBSCAN(eps=30, min_samples=1).fit(array).labels_
    return np.array(
        [[int(round(float(z))) for z in array[labels == j].mean(0)] for j in np.unique(labels)],
        dtype=np.int32,
    )


def clean_mask(mask):
    mask = np.asarray(mask, dtype=bool)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype("uint8"), 8)
    result = np.zeros_like(mask)
    for j in range(1, n):
        if stats[j, cv2.CC_STAT_AREA] >= 1000:
            result[labels == j] = True
    return result if result.any() else mask


def direct_positive(predictor, image_bgr):
    points = positive_points(image_bgr)
    if not len(points):
        raise ValueError("No automatic positive prompt")
    predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    masks, scores, _ = predictor.predict(
        point_coords=points,
        point_labels=np.ones(len(points), dtype=np.int32),
        multimask_output=True,
    )
    return clean_mask(masks[int(np.argmax(scores))])
