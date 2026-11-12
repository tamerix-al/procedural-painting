"""Progressive reconstruction of pictures with optimized vector-like strokes.

The optimizer works at a configurable analysis resolution, but records every
stroke as geometry. The video renderer redraws geometry at native video size
rather than enlarging a low-resolution bitmap.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import math

import cv2
import numpy as np

BUILTIN_STYLES = {
    "triangles": "Треугольники",
    "circles": "Точки / круги",
    "rectangles": "Прямоугольники",
    "brush": "Мазки кисти (старый)",
    "oil": "Масляные мазки / Oil Strokes",
    "dry_brush": "Сухая кисть / Dry Brush",
}
STYLES = {
    **BUILTIN_STYLES,
    "ellipses": "Эллипсы",
    "rotated_rectangles": "Повернутые прямоугольники",
    "rotated_ellipses": "Повернутые эллипсы",
    "bezier": "Кривые Безье / штрихи",
    "mix": "Смешанные фигуры",
    "grain": "GRAIN — цветные точки",
    "trace": "TRACE — растущая линия",
    "trace_continuous": "TRACE — архивный рисунок 0.8 (только просмотр)",
}


@dataclass(eq=True)
class Stroke:
    kind: str
    geometry: list
    color: tuple[int, int, int]
    alpha: float


@dataclass
class Painting:
    base_color: tuple[int, int, int]
    operations: list[Stroke]
    target_rgb: np.ndarray
    width: int
    height: int
    style: str


def resize_for_work(image_rgb: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    if width < 1 or height < 1:
        raise ValueError("Пустое изображение")
    scale = min(1.0, max_side / max(width, height))
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image_rgb, new_size, interpolation=cv2.INTER_AREA)


def random_shape(
    style: str, cx: float, cy: float, radius: float,
    rng: np.random.Generator, angle: float | None = None,
    brush_ratio: float | None = None,
) -> tuple[str, list]:
    if style == "triangles":
        phase = rng.uniform(0, math.tau)
        angles = phase + np.array([0, 2.1, 4.2]) + rng.uniform(-0.46, 0.46, 3)
        radii = radius * rng.uniform(0.55, 1.5, 3)
        points = np.stack((cx + np.cos(angles) * radii,
                           cy + np.sin(angles) * radii), axis=1)
        return "poly", np.rint(points).astype(int).tolist()
    if style == "rectangles":
        width = max(1.25, radius * rng.uniform(0.65, 2.0))
        height = max(1.25, radius * rng.uniform(0.35, 1.5))
        box = cv2.boxPoints(((float(cx), float(cy)), (width, height),
                             float(rng.uniform(-85, 85))))
        return "poly", np.rint(box).astype(int).tolist()
    if style == "circles":
        return "circle", [round(cx), round(cy),
                          max(1, round(radius * rng.uniform(0.55, 1.3)))]
    if style == "brush":
        if brush_ratio is not None:
            # Compact paint dabs: width and centreline length are tied together.
            # The old brush generated lengths up to 3.3 * radius but widths as
            # little as 0.16 * radius (occasionally >15:1 aspect ratio).
            if angle is None:
                angle = float(rng.uniform(0, math.tau))
            angle += float(rng.normal(0, 0.12))
            thick = max(0.8, float(radius) * float(rng.uniform(.78, 1.18)))
            length = max(0.8, thick * float(brush_ratio) * float(rng.uniform(.74, 1.14)))
            dx, dy = math.cos(angle), math.sin(angle)
            bend = length * float(rng.uniform(-.035, .035))
            pts = [
                [round(cx - dx * length/2, 2), round(cy - dy * length/2, 2)],
                [round(cx - dy*bend, 2), round(cy + dx*bend, 2)],
                [round(cx + dx * length/2, 2), round(cy + dy * length/2, 2)],
            ]
            return "brush", [pts, round(thick, 2)]
        if angle is None:
            angle = float(rng.uniform(0, math.tau))
        angle += float(rng.normal(0, 0.20))
        length = max(1.5, radius * rng.uniform(1.2, 3.3))
        thick = max(1, round(radius * rng.uniform(0.16, 0.45)))
        dx, dy = math.cos(angle), math.sin(angle)
        curve = length * float(rng.uniform(-0.09, 0.09))
        pts = [
            [round(cx - dx * length / 2), round(cy - dy * length / 2)],
            [round(cx - dy * curve), round(cy + dx * curve)],
            [round(cx + dx * length / 2), round(cy + dy * length / 2)],
        ]
        return "brush", [pts, thick]
    raise ValueError(f"Неизвестный стиль: {style}")


def scale_stroke(stroke: Stroke, sx: float, sy: float) -> Stroke:
    """Scale stored geometry for crisp rasterization at output dimensions."""
    if stroke.kind == "poly":
        geometry = [[round(x * sx), round(y * sy)] for x, y in stroke.geometry]
    elif stroke.kind == "circle":
        x, y, r = stroke.geometry
        geometry = [round(x * sx), round(y * sy),
                    max(1, round(r * (sx + sy) / 2))]
    elif stroke.kind == "oil":
        from .oil_brush import validate_oil_geometry
        validate_oil_geometry(stroke.geometry)
        pts,thick,dry,impasto,brush_seed=stroke.geometry
        geometry=[[[round(x*sx,3),round(y*sy,3)] for x,y in pts],
                  max(.3,round(float(thick)*(sx+sy)/2,3)),dry,impasto,brush_seed]
    elif stroke.kind in ("brush", "polyline"):
        points, thick = stroke.geometry
        geometry = [[[round(x * sx), round(y * sy)] for x, y in points],
                    max(0.5, thick * (sx + sy) / 2)]
    elif stroke.kind == "ellipse":
        cx,cy,rx,ry,angle=stroke.geometry
        geometry = [cx*sx, cy*sy, max(0.1,rx*sx), max(0.1,ry*sy), angle]
    elif stroke.kind == "line":  # backwards compatibility with MVP 0.1
        a, b, c, d, thick = stroke.geometry
        geometry = [round(a * sx), round(b * sy), round(c * sx),
                    round(d * sy), max(1, round(thick * (sx + sy) / 2))]
    else:
        raise ValueError(f"Неизвестная фигура: {stroke.kind}")
    return Stroke(stroke.kind, geometry, stroke.color, stroke.alpha)


def mask_for_shape(kind: str, geometry: list, width: int, height: int):
    """Return a tight ROI with an antialiased coverage mask in [0,1]."""
    pad = 3
    if kind == "poly":
        pts = np.asarray(geometry, np.int32)
        if len(pts) < 3:
            return None
        x0, y0 = np.min(pts, axis=0) - pad
        x1, y1 = np.max(pts, axis=0) + pad + 1
    elif kind == "circle":
        cx, cy, r = geometry
        x0, y0, x1, y1 = cx-r-pad, cy-r-pad, cx+r+pad+1, cy+r+pad+1
    elif kind == "oil":
        from .oil_brush import validate_oil_geometry
        validate_oil_geometry(geometry)
        points,thick,*_=geometry
        pts=np.rint(points).astype(np.int32)
        x0, y0 = np.min(pts, axis=0) - thick - pad
        x1, y1 = np.max(pts, axis=0) + thick + pad + 1
    elif kind in ("brush", "polyline"):
        points, thick = geometry
        pts = np.rint(points).astype(np.int32)
        x0, y0 = np.min(pts, axis=0) - thick - pad
        x1, y1 = np.max(pts, axis=0) + thick + pad + 1
    elif kind == "ellipse":
        cx,cy,rx,ry,angle=geometry
        radius=math.hypot(rx,ry)
        x0,y0,x1,y1=cx-radius-pad,cy-radius-pad,cx+radius+pad+1,cy+radius+pad+1
    elif kind == "line":
        ax, ay, bx, by, thick = geometry
        x0, y0, x1, y1 = min(ax,bx)-thick-pad, min(ay,by)-thick-pad, max(ax,bx)+thick+pad+1, max(ay,by)+thick+pad+1
    else:
        raise ValueError(f"Неизвестная фигура: {kind}")
    x0, y0, x1, y1 = max(0,int(x0)), max(0,int(y0)), min(width,int(x1)), min(height,int(y1))
    if x1 <= x0 or y1 <= y0:
        return None
    mask = np.zeros((y1-y0, x1-x0), np.uint8)
    if kind == "poly":
        cv2.fillPoly(mask, [pts - (x0,y0)], 255, lineType=cv2.LINE_AA)
    elif kind == "circle":
        cx,cy,r = geometry
        cv2.circle(mask,(cx-x0,cy-y0),r,255,-1,lineType=cv2.LINE_AA)
    elif kind == "oil":
        points,thick,*_=geometry
        p=np.rint(points).astype(np.int32)-(x0,y0)
        cv2.polylines(mask,[p],False,255,max(1,round(thick)),cv2.LINE_AA)
    elif kind in ("brush", "polyline"):
        points, thick = geometry
        p = np.rint(points).astype(np.int32) - (x0,y0)
        if len(p)>=2:
            cv2.polylines(mask,[p],False,255,max(1,round(thick)),cv2.LINE_AA)
            if kind == "brush":
                cv2.circle(mask,tuple(p[len(p)//2]),max(1,round(thick/2)),255,-1,cv2.LINE_AA)
    elif kind == "ellipse":
        cx,cy,rx,ry,angle=geometry
        cv2.ellipse(mask,(round(cx-x0),round(cy-y0)),
                    (max(1,round(rx)),max(1,round(ry))),
                    angle,0,360,255,-1,cv2.LINE_AA)
    else:
        ax,ay,bx,by,thick=geometry
        cv2.line(mask,(ax-x0,ay-y0),(bx-x0,by-y0),255,thick,cv2.LINE_AA)
    if not np.any(mask):
        return None
    return x0,y0,x1,y1, mask.astype(np.float32) / 255.0


def apply_stroke(canvas: np.ndarray, stroke: Stroke) -> None:
    """Composite one stroke on an RGB float32 canvas in place."""
    if stroke.kind == "oil":
        from .oil_brush import paint_oil
        paint_oil(canvas,stroke.geometry,stroke.color,stroke.alpha)
        return
    h,w = canvas.shape[:2]
    item = mask_for_shape(stroke.kind,stroke.geometry,w,h)
    if item is None:
        return
    x0,y0,x1,y1,cover=item
    alpha=(cover*stroke.alpha)[...,None]
    roi=canvas[y0:y1,x0:x1]
    roi *= (1-alpha)
    roi += np.asarray(stroke.color,np.float32)*alpha


def _target_pyramid(target: np.ndarray) -> list[np.ndarray]:
    # Coarse-to-fine: early large shapes fit broad areas, then sharpen detail.
    k = max(1.0, max(target.shape[:2]) / 500)
    return [cv2.GaussianBlur(target,(0,0),sigmaX=s*k) if s else target
            for s in (5.0,2.5,1.0,0.0)]


def build_painting(
    image_rgb: np.ndarray, style: str = "triangles",
    strokes: int = 1800, candidates: int = 42, working_size: int = 520,
    seed: int = 42, progress: Callable[[int,int],None] | None = None,
    detail: float = 1.0, opacity: float = 1.0,
    oil_dryness: float | None = None, oil_impasto: float | None = None,
    oil_length_ratio: float = 1.6, oil_size: float = 0.85,
    oil_follow_flow: bool = True,
    brush_profile: str = "legacy",
) -> Painting:
    """Find shapes that reduce squared reconstruction error, coarse-to-fine.

    `strokes`, `candidates`, and `working_size` each increase quality/cost.
    `detail` alters the late-stage shape size (1 = default; 2 = finer).
    """
    if style not in BUILTIN_STYLES:
        raise ValueError(f"Неизвестный встроенный стиль: {style}")
    if style in ("oil", "dry_brush"):
        from .oil_plan import build_oil_painting
        return build_oil_painting(image_rgb,style,strokes,candidates,working_size,
                                  seed,progress,detail,opacity,
                                  oil_dryness,oil_impasto,
                                  oil_length_ratio,oil_size,oil_follow_flow)
    if not (0.8 <= oil_length_ratio <= 5.0 and .4 <= oil_size <= 2.0):
        raise ValueError("Параметры длины и размера мазков вне допустимого диапазона")
    if strokes < 1 or candidates < 1 or working_size < 32:
        raise ValueError("Некорректные настройки качества")
    if not (0.1 <= opacity <= 1):
        raise ValueError("Прозрачность должна быть в диапазоне 0.1–1")
    target=resize_for_work(image_rgb,working_size).astype(np.float32)
    h,w=target.shape[:2]
    pyramid=_target_pyramid(target)
    rng=np.random.default_rng(seed)
    base=tuple(int(round(v)) for v in target.mean((0,1)))
    canvas=np.empty_like(target);canvas[:]=base
    operations=[]
    gray=cv2.cvtColor(target.astype(np.uint8),cv2.COLOR_RGB2GRAY)
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=5)
    gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=5)
    short=min(h,w)
    # The structure tensor smooths local edge direction, avoiding wildly
    # alternating brush angles around individual noisy pixels.
    flow_theta=None
    flow_confidence=None
    if style=="brush" and brush_profile=="compact" and oil_follow_flow:
        smoothed=cv2.GaussianBlur(gray, (0,0), max(1.,short/210.)).astype(np.float32)
        fx=cv2.Sobel(smoothed,cv2.CV_32F,1,0,ksize=3)
        fy=cv2.Sobel(smoothed,cv2.CV_32F,0,1,ksize=3)
        sig=max(1.5,short/105.)
        j11=cv2.GaussianBlur(fx*fx,(0,0),sig)
        j12=cv2.GaussianBlur(fx*fy,(0,0),sig)
        j22=cv2.GaussianBlur(fy*fy,(0,0),sig)
        flow_theta=.5*np.arctan2(2*j12,j11-j22)+math.pi/2
        flow_confidence=np.hypot(j11-j22,2*j12)/(j11+j22+1e-3)
    cached_error=None
    prior_stage=-1
    for i in range(strokes):
        fraction=i/max(1,strokes-1)
        stage=min(3,int(fraction*4))
        goal=pyramid[stage]
        if (i%6==0) or stage!=prior_stage or cached_error is None:
            cached_error=np.mean((canvas-goal)**2,axis=2)
        prior_stage=stage
        # Tiny shapes at the end carry texture; earlier shapes cover larger regions.
        if style=="brush" and brush_profile=="compact":
            # Compact from the very beginning, 1-3x width instead of 4-15x.
            # Preserve coarse-to-fine variation while avoiding sweeping bars.
            radius=short*(0.065*(1-fraction)**1.85 + 0.0045/max(.4,detail))*oil_size
        else:
            radius=short*(0.29*(1-fraction)**2.15 + 0.0038/max(0.4,detail))
        radius=max(0.85, radius)
        best_gain=0.0
        best=None
        for _ in range(candidates):
            # Recompute error only every few iterations; most work stays in small ROIs.
            num_samples=32
            ys=rng.integers(0,h,num_samples)
            xs=rng.integers(0,w,num_samples)
            index=int(np.argmax(cached_error[ys,xs]+rng.uniform(0,65,num_samples)))
            cx=xs[index]+rng.normal(0,max(1,radius*0.25))
            cy=ys[index]+rng.normal(0,max(1,radius*0.25))
            angle=None
            if style=="brush":
                xi=int(np.clip(round(cx),0,w-1));yi=int(np.clip(round(cy),0,h-1))
                if brush_profile=="compact":
                    if flow_theta is not None and float(flow_confidence[yi,xi])>.15:
                        angle=float(flow_theta[yi,xi])
                elif abs(gx[yi,xi])+abs(gy[yi,xi]) > 14:
                    angle=float(math.atan2(gy[yi,xi],gx[yi,xi])+math.pi/2)
            kind,geom=random_shape(style,cx,cy,
                                   radius*(rng.uniform(.75,1.25) if brush_profile=="compact" else rng.uniform(.55,1.55)),
                                   rng,angle,brush_ratio=oil_length_ratio if brush_profile=="compact" and style=="brush" else None)
            item=mask_for_shape(kind,geom,w,h)
            if item is None:
                continue
            x0,y0,x1,y1,coverage=item
            region=canvas[y0:y1,x0:x1]
            objective=goal[y0:y1,x0:x1]
            alpha_base=rng.uniform(.27,.72) if style=="triangles" else rng.uniform(.67,.98)
            alpha=float(min(1,alpha_base*opacity))
            a=(coverage*alpha)[...,None]
            power=float(np.sum(a*a))
            if power < .02:
                continue
            color_f=np.clip(np.sum((objective-region*(1-a))*a,axis=(0,1))/power,0,255)
            after=region*(1-a)+color_f*a
            delta=float(np.sum((region-objective)**2)-np.sum((after-objective)**2))
            if delta>best_gain:
                best_gain=delta
                best=Stroke(kind,geom,tuple(int(round(v)) for v in color_f),alpha)
        if best is not None:
            apply_stroke(canvas,best)
            operations.append(best)
        if progress and (i%10==0 or i+1==strokes):
            progress(i+1,strokes)
    return Painting(base,operations,np.clip(target,0,255).astype(np.uint8),w,h,style)
