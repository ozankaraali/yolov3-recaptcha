"""Decide which tiles of a reCAPTCHA image challenge contain the requested object.

The model is Ultralytics YOLO26 (2026), trained on the 80 COCO classes. Ultralytics downloads the
weights to the working directory on first use.
"""

from __future__ import annotations

import re

import cv2
import numpy as np
from ultralytics import YOLO

MODEL = "yolo26s.pt"
CONFIDENCE = 0.25
# 4x4 challenges are one photo cut into squares. A square counts when a detection box covers at
# least this fraction of it.
MIN_SQUARE_COVERAGE = 0.1

# reCAPTCHA prompt -> COCO class. Other prompts reCAPTCHA uses (crosswalks, stairs, chimneys,
# bridges, palm trees, mountains or hills, tractors) are not COCO classes; the solver skips them.
TARGETS = {
    "bicycle": "bicycle",
    "bicycles": "bicycle",
    "boat": "boat",
    "boats": "boat",
    "bus": "bus",
    "buses": "bus",
    "car": "car",
    "cars": "car",
    "taxi": "car",
    "taxis": "car",
    "fire hydrant": "fire hydrant",
    "fire hydrants": "fire hydrant",
    "motorcycle": "motorcycle",
    "motorcycles": "motorcycle",
    "traffic light": "traffic light",
    "traffic lights": "traffic light",
}

Rect = tuple[int, int, int, int]  # x0, y0, x1, y1 in canvas pixels
Box = tuple[float, float, float, float, float]  # x0, y0, x1, y1, confidence in canvas pixels


def coco_class(prompt: str) -> str | None:
    """'a fire hydrant' -> 'fire hydrant', 'Buses' -> 'bus'; None if no COCO class matches."""
    words = re.sub(r"[^a-z ]", " ", prompt.lower()).split()
    if words and words[0] in ("a", "an"):
        words = words[1:]
    return TARGETS.get(" ".join(words))


def coverage(rect: Rect, box: Box) -> float:
    """Fraction of `rect` covered by `box`."""
    x0, y0, x1, y1 = rect
    w = min(x1, box[2]) - max(x0, box[0])
    h = min(y1, box[3]) - max(y0, box[1])
    if w <= 0 or h <= 0:
        return 0.0
    return (w * h) / ((x1 - x0) * (y1 - y0))


class TileFinder:
    def __init__(self, weights: str = MODEL, confidence: float = CONFIDENCE):
        self.model = YOLO(weights)
        self.confidence = confidence

    def _detect(self, images: list[np.ndarray], class_name: str) -> list[list[Box]]:
        class_id = next(k for k, v in self.model.names.items() if v == class_name)
        results = self.model.predict(images, conf=self.confidence, classes=[class_id], verbose=False)
        return [
            [(*map(float, xyxy), float(conf)) for xyxy, conf in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist())]
            for r in results
        ]

    def pick(
        self,
        canvas: np.ndarray,
        rects: list[Rect],
        class_name: str,
        separate_images: bool,
        only: list[int] | None = None,
    ) -> tuple[list[int], list[Box]]:
        """Return the indices of the tiles to click and the detections, in canvas coordinates.

        separate_images: True when every tile is its own photo (3x3 challenges and replacement
        tiles), False when the grid is one photo cut into squares (4x4 challenges).
        only: with separate_images, look at these tiles only, such as freshly replaced ones.
        """
        if not separate_images:
            boxes = self._detect([canvas], class_name)[0]
            picked = [i for i, r in enumerate(rects) if any(coverage(r, b) >= MIN_SQUARE_COVERAGE for b in boxes)]
            return picked, boxes

        indices = list(range(len(rects))) if only is None else only
        crops = [canvas[y0:y1, x0:x1] for x0, y0, x1, y1 in (rects[i] for i in indices)]
        picked, boxes = [], []
        for i, tile_boxes in zip(indices, self._detect(crops, class_name)):
            if tile_boxes:
                picked.append(i)
            x0, y0 = rects[i][:2]
            boxes += [(bx0 + x0, by0 + y0, bx1 + x0, by1 + y0, c) for bx0, by0, bx1, by1, c in tile_boxes]
        return picked, boxes


def draw(canvas: np.ndarray, rects: list[Rect], picked: list[int], boxes: list[Box], label: str) -> np.ndarray:
    """Mark detections in red and the tiles to click in green on a copy of `canvas` (BGR)."""
    out = canvas.copy()
    thick = max(2, round(out.shape[1] / 250))
    for i in picked:
        x0, y0, x1, y1 = rects[i]
        overlay = out.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 200, 0), -1)
        out = cv2.addWeighted(overlay, 0.25, out, 0.75, 0)
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 200, 0), thick * 2)
    for x0, y0, x1, y1, conf in boxes:
        p0, p1 = (int(x0), int(y0)), (int(x1), int(y1))
        cv2.rectangle(out, p0, p1, (0, 0, 255), thick)
        text = f"{label} {conf:.2f}"
        scale = out.shape[1] / 1000
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick // 2 or 1)
        cv2.rectangle(out, (p0[0], p0[1] - th - 6), (p0[0] + tw + 4, p0[1]), (0, 0, 255), -1)
        cv2.putText(out, text, (p0[0] + 2, p0[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick // 2 or 1, cv2.LINE_AA)
    return out


def _runs(content: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    """[start, end) runs where `content` is True and at least `min_len` long."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], content.astype(np.int8), [0]))))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2]) if b - a >= min_len]


def find_grid(screenshot: np.ndarray) -> list[Rect]:
    """Find the tiles in a screenshot of the whole challenge popup, row by row.

    Tiles are the blocks below the blue prompt header, separated by white gutters.
    """
    h, w = screenshot.shape[:2]
    b, r = screenshot[..., 0].astype(int), screenshot[..., 2].astype(int)
    blue_rows = ((b > 180) & (b - r > 80)).mean(axis=1)
    header = np.flatnonzero(blue_rows[: h // 2] > 0.5)
    top = int(header.max()) + 1 if header.size else 0

    white = (screenshot > 235).all(axis=2)
    rows = [(a + top, z + top) for a, z in _runs(white[top:].mean(axis=1) < 0.9, h // 10)]
    if not rows:
        return []
    cols = _runs(white[rows[0][0] : rows[-1][1]].mean(axis=0) < 0.9, w // 10)
    return [(x0, y0, x1, y1) for y0, y1 in rows for x0, x1 in cols]
