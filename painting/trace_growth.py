"""Image-driven growing-pen TRACE.

A pen starts at a reproducible random seed and follows one joined, gradually
turning path. New directions are chosen from the image's local contour field
and from the remaining unpainted colour error. Only when a tip gets stuck is
the pen lifted and restarted at a new seed. Operations within each trail have
exactly matching endpoints, so the video shows a line growing, not isolated
strokes appearing around the image.

No object detection or semantic rules; works on arbitrary RGB inputs.
"""
from __future__ import annotations

import math
from typing import Callable

import cv2
import numpy as np

from .engine import Painting, Stroke

PAPER = (249, 247, 241)


def _field(target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Smoothed colour, tangent direction, confidence and ink priority."""
    rgb = cv2.GaussianBlur(target, (0, 0), 0.68).astype(np.float32)
    gray = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (0, 0), 1.2)
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    tensor_sigma = max(2.2, min(target.shape[:2]) / 200)
    aa = cv2.GaussianBlur(dx * dx, (0, 0), tensor_sigma)
    bb = cv2.GaussianBlur(dx * dy, (0, 0), tensor_sigma)
    cc = cv2.GaussianBlur(dy * dy, (0, 0), tensor_sigma)
    theta = 0.5 * np.arctan2(2 * bb, aa - cc) + math.pi / 2
    confidence = np.hypot(aa - cc, 2 * bb) / (aa + cc + 5.0)
    edge = np.clip(np.hypot(dx, dy) / 135.0, 0, 1)
    bg = np.asarray(PAPER, np.float32)
    ink = np.clip(np.linalg.norm(rgb - bg, axis=2) / 380.0, 0.0, 1.0)
    priority = 0.10 + 0.55 * edge + 0.90 * ink
    return rgb, theta, confidence, priority


def _round_pt(x: float, y: float) -> list[float]:
    return [round(float(x), 2), round(float(y), 2)]


def build_growing_trace(
    target: np.ndarray,
    count: int,
    seed: int,
    mark_size: float,
    contrast: float,
    progress: Callable[[int, int], None] | None = None,
) -> Painting:
    """Follow image-guided connected curves, painting while the tip advances.

    `count` = number of visible line segments, not the number of unrelated
    patches. `mark_size` controls width and modestly influences step length.
    The paper remains visible between curves. The same seed reproduces output.
    """
    h, w = target.shape[:2]
    rng = np.random.default_rng(int(seed))
    pixels, theta, conf, salience = _field(target)
    paper = np.asarray(PAPER, np.float32)
    canvas = np.empty((h, w, 3), dtype=np.float32)
    canvas[:] = paper
    # Coverage measures how much ink already exists in the vicinity, not
    # exactly which source pixel has been visited.
    coverage = np.zeros((h, w), np.float32)
    ink = np.linalg.norm(pixels - paper, axis=2) / 380.0
    minside = min(h, w)
    step = float(np.clip(minside / 110.0 * (mark_size ** .26), 2.25, 7.0))
    width_base = float(np.clip(minside / 370.0 * mark_size, .65, 2.4))
    alpha = 0.96
    # A small update footprint prevents overpainting the same spot repeatedly.
    r = max(3, int(math.ceil(width_base * 2.5)))
    axis = np.arange(-r, r + 1, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis)
    stamp = np.exp(-(xx * xx + yy * yy) / (2 * max(1., width_base * 1.7) ** 2)).astype(np.float32) * .65

    def refresh(x: int, y: int):
        x0, x1 = max(0, x - r), min(w, x + r + 1)
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        coverage[y0:y1, x0:x1] += stamp[y0 - y + r:y1 - y + r, x0 - x + r:x1 - x + r]

    def error_at(x: int, y: int) -> float:
        residual = (pixels[y, x] - canvas[y, x]) / 255.0
        return float(np.sqrt(np.dot(residual, residual) / 3))

    def get_seed(neighbor: tuple[float, float] | None = None) -> tuple[float, float]:
        # Start with a random high-information pixel. Subsequent starts prefer
        # unfinished regions near the last tip, then explore the full canvas.
        diff = np.sqrt(np.mean(((pixels - canvas) / 255.0) ** 2, axis=2))
        need = (.02 + diff ** 1.3) * (0.3 + .7 * salience) / (1 + coverage * .8)
        if neighbor is not None and rng.random() < .78:
            cx, cy = neighbor
            rad = min(h, w) * rng.uniform(.13, .30)
            x0 = max(0, int(cx - rad)); x1 = min(w, int(cx + rad + 1))
            y0 = max(0, int(cy - rad)); y1 = min(h, int(cy + rad + 1))
            local = need[y0:y1, x0:x1]
            if local.size and float(local.max()) > .012:
                # Randomized top-quantile choice avoids deterministic scanning.
                threshold = float(np.quantile(local, .91))
                ys, xs = np.where(local >= threshold)
                k = int(rng.integers(0, len(xs)))
                return float(x0 + xs[k]), float(y0 + ys[k])
        threshold = float(np.quantile(need, .84))
        ys, xs = np.where(need >= threshold)
        k = int(rng.integers(len(xs)))
        return float(xs[k]), float(ys[k])

    def orient_flow(x: int, y: int, angle: float) -> tuple[float, float]:
        a = float(theta[y, x])
        if math.cos(a - angle) < 0:
            a += math.pi
        return a, float(conf[y, x])

    def draw_segment(vertices: list[list[float]], col: tuple[int, int, int], width: float):
        """Update optimizer's canvas using the SAME squiggle as the renderer."""
        pts = np.rint(np.asarray(vertices, np.float32)).astype(np.int32)
        pad = max(3, int(math.ceil(width)) + 3)
        x0 = max(0, int(pts[:, 0].min()) - pad)
        y0 = max(0, int(pts[:, 1].min()) - pad)
        x1 = min(w, int(pts[:, 0].max()) + pad + 1)
        y1 = min(h, int(pts[:, 1].max()) + pad + 1)
        if x1 <= x0 or y1 <= y0:
            return
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        pts -= np.array([x0, y0], np.int32)
        cv2.polylines(mask, [pts], False, 255, max(1, int(round(width))), cv2.LINE_AA)
        k = (mask.astype(np.float32) / 255.0 * alpha)[..., None]
        current = canvas[y0:y1, x0:x1]
        current[:] = current * (1 - k) + np.asarray(col, np.float32) * k

    strokes: list[Stroke] = []
    x, y = get_seed()
    angle = float(rng.uniform(-math.pi, math.pi))
    last_tip: tuple[float, float] | None = None
    stuck = 0
    trail_age = 0
    max_age = int(np.clip(count / 12, 220, 1600))
    for i in range(count):
        cx, cy = int(round(x)), int(round(y))
        cx = max(0, min(w - 1, cx)); cy = max(0, min(h - 1, cy))
        flow_angle, coherence = orient_flow(cx, cy, angle)
        # In coherent areas, possible directions gather around the isophotes;
        # elsewhere the tip is free to meander smoothly.
        offsets = (-1.03, -.73, -.44, -.23, 0., .23, .44, .73, 1.03)
        variants = []
        for offset in offsets:
            candidate_angle = angle + offset * (0.75 if coherence > .45 else 1.0)
            for base_angle in ([candidate_angle, .80 * flow_angle + .20 * angle] if coherence > .35 and offset == 0 else [candidate_angle]):
                bx = x + math.cos(base_angle) * step
                by = y + math.sin(base_angle) * step
                if not (1.5 <= bx <= w - 2.5 and 1.5 <= by <= h - 2.5):
                    continue
                mx, my = (x + bx) * .5, (y + by) * .5
                xi = int(round(bx)); yi = int(round(by))
                xmid, ymid = int(round(mx)), int(round(my))
                if abs(int(pixels[yi, xi, 0]) - int(pixels[ymid, xmid, 0])) > 120:
                    color_consistency = -.18
                else:
                    color_consistency = 0
                error = (.57 * error_at(xi, yi) + .43 * error_at(xmid, ymid))
                target_density = .46 + 1.45 * float(ink[yi, xi])
                density_gap = (target_density - coverage[yi, xi]) / (target_density + .2)
                lx = int(np.clip(round(bx + math.cos(base_angle)*step*1.8), 0, w-1))
                ly = int(np.clip(round(by + math.sin(base_angle)*step*1.8), 0, h-1))
                ahead = .46 + 1.45 * float(ink[ly,lx])
                ahead_gap = (ahead - coverage[ly,lx]) / (ahead + .2)
                useful = .65*float(np.clip(density_gap,-1.0,1.0)) + .35*float(np.clip(ahead_gap,-1.0,1.0))
                if coherence > .12:
                    conform = max(0., math.cos(base_angle - flow_angle)) * coherence
                else:
                    conform = 0
                turn = abs((base_angle - angle + math.pi) % math.tau - math.pi)
                score = (1.25 * error + .69 * useful + .10 * ink[yi, xi] +
                         .20 * conform - .16 * turn + color_consistency +
                         float(rng.normal(0, .018)))
                variants.append((score, base_angle, bx, by, xmid, ymid))
        variants.sort(key=lambda t: t[0], reverse=True)
        # Move to the closest unfinished region when local improvement becomes
        # poor. Only endpoints within a trail are visible as connected strokes.
        if not variants or (stuck >= 15 and trail_age > 60) or (trail_age > max_age and variants[0][0] < .63):
            last_tip = (x, y)
            x, y = get_seed(last_tip)
            angle = float(theta[int(round(y)), int(round(x))]) + float(rng.choice([0., math.pi]))
            trail_age = 0
            stuck = 0
            # Recompute this operation from the new location, so no pen-up jump
            # is drawn. This has a negligible effect on requested stroke count.
            cx, cy = int(round(x)), int(round(y))
            bx = float(np.clip(x + math.cos(angle) * step, 1.5, w - 2.5))
            by = float(np.clip(y + math.sin(angle) * step, 1.5, h - 2.5))
            xmid, ymid = (x + bx) * .5, (y + by) * .5
            best_angle = angle
        else:
            score, best_angle, bx, by, xmid, ymid = variants[0]
            if score < .48:
                stuck += 1
            else:
                stuck = max(0, stuck - 2)
        if math.hypot(bx - x, by - y) < .05:
            # Avoid zero-length operations near corners and boundaries.
            bx = float(np.clip(x + step * .5, .5, w - .5))
            by = float(np.clip(y + step * .5, .5, h - .5))
        xmid_i = max(0, min(w - 1, int(round(xmid))))
        ymid_i = max(0, min(h - 1, int(round(ymid))))
        source = pixels[ymid_i, xmid_i]
        color = np.clip(paper + (source - paper) * float(contrast), 0, 255)
        col = tuple(int(round(float(v))) for v in color)
        darkness = float(ink[ymid_i, xmid_i])
        width = float(np.clip(width_base * (.73 + .45 * darkness), .55, 3.0))
        # A fine sideways undulation (squiggle) along a growing centreline.
        # The endpoints stay shared EXACTLY, making the animation a growing
        # pen trace rather than separate scribbles. A shorter rhythm is
        # deliberate: broad smooth contours look like concentric circles.
        dx = bx - x; dy = by - y
        lng = max(1e-6, math.hypot(dx, dy))
        normal_x, normal_y = -dy / lng, dx / lng
        wiggle = min(1.8, step * .24) * float(rng.uniform(.65, 1.20))
        side = float(rng.choice((-1, 1)))
        vertices = [_round_pt(x, y)]
        for t in (.2, .4, .6, .8):
            wave = math.sin(t * math.tau * 2.0) * wiggle * side
            vertices.append(_round_pt(x + dx * t + normal_x * wave,
                                      y + dy * t + normal_y * wave))
        vertices.append(_round_pt(bx, by))
        strokes.append(Stroke('polyline', [vertices, round(width, 3)], col, alpha))
        draw_segment(vertices, col, width)
        refresh(int(round(xmid)), int(round(ymid)))
        x, y = float(bx), float(by)
        angle = float(math.atan2(dy, dx))
        trail_age += 1
        if progress and (i % max(1, count // 100) == 0 or i == count - 1):
            progress(i + 1, count)
    return Painting(PAPER, strokes, target, w, h, 'trace')
