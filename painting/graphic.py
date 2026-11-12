"""Graphic modes: colour stippling and image-guided growing TRACE.

TRACE is implemented in trace_growth.py. GRAIN is independent and retained
for backwards compatibility with projects saved in earlier releases.
"""
from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from .engine import Painting, Stroke, resize_for_work
from .trace_growth import build_growing_trace

GRAPHIC_STYLES = {
    'trace': 'TRACE — растущая линия',
    'grain': 'GRAIN — цветные точки',
}
PAPER = (249, 247, 241)


def _sampling_weights(target: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY).astype(np.float32)
    blur = cv2.GaussianBlur(gray, (0, 0), 1.15)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.clip(np.hypot(gx, gy) / 105., 0, 1)
    chroma = np.mean(np.abs(target.astype(np.float32) - np.asarray(PAPER, np.float32)), axis=2) / 255.
    tone = np.clip((250.-gray) / 255., 0, 1)
    return np.maximum(.025, .28 + 1.5 * chroma + .42 * edge + .30 * tone)


def build_graphic_painting(
    image_rgb: np.ndarray, style: str, marks: int = 16000,
    working_size: int = 700, seed: int = 42,
    mark_size: float = 1.0, contrast: float = 1.0,
    progress: Callable[[int, int], None] | None = None,
) -> Painting:
    if style not in GRAPHIC_STYLES:
        raise ValueError(f'Неизвестная графическая техника: {style}')
    if not 100 <= marks <= 100_000 or not 128 <= working_size <= 2000:
        raise ValueError('Некорректное количество штрихов или разрешение анализа')
    if not .4 <= mark_size <= 2.5 or not .5 <= contrast <= 2.0:
        raise ValueError('Некорректные настройки графической техники')
    target = resize_for_work(image_rgb, working_size)
    if style == 'trace':
        return build_growing_trace(target, int(marks), int(seed), mark_size, contrast, progress)

    h, w = target.shape[:2]
    rng = np.random.default_rng(seed)
    weights = _sampling_weights(target)
    probs = (weights / weights.sum()).ravel()
    chosen = rng.choice(h * w, size=int(marks), replace=int(marks) > h * w, p=probs)
    ys, xs = divmod(chosen, w)
    smoothed = cv2.GaussianBlur(target, (0, 0), .7)
    ops: list[Stroke] = []
    for i, (yy, xx) in enumerate(zip(ys, xs)):
        px = float(np.clip(xx + rng.uniform(-.4, .4), 0, w - 1))
        py = float(np.clip(yy + rng.uniform(-.4, .4), 0, h - 1))
        rgb = smoothed[int(yy), int(xx)].astype(np.float32)
        rgb = np.clip(np.asarray(PAPER) + (rgb - np.asarray(PAPER)) * contrast, 0, 255)
        radius = float(np.clip(min(w, h) / 460.0 * mark_size * rng.lognormal(0, .30), .38, 4.0))
        ops.append(Stroke('circle', [int(round(px)), int(round(py)), max(1, int(round(radius)))],
                          tuple(int(v) for v in rgb), .91))
        if progress and (i % max(1, int(marks) // 100) == 0 or i == int(marks) - 1):
            progress(i + 1, int(marks))
    return Painting(PAPER, ops, target, w, h, 'grain')
