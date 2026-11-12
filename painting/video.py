"""High-resolution MP4 renderer with editable timing and three original reveals."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import imageio.v2 as imageio
import numpy as np

from .engine import Painting, apply_stroke, scale_stroke
from .layout import LayoutOptions, FrameRenderer

RESOLUTIONS = {
    "540 × 960 (предпросмотр)": (540,960),
    "720 × 1280": (720,1280),
    "1080 × 1920 (Full HD)": (1080,1920),
}

REVEAL_MODES = {
    'soft_wipe': 'Soft Wipe — мягкая граница слева направо',
    'split_reveal': 'Split Reveal — оригинал слева, рисунок справа',
    'crossfade': 'Crossfade — плавное растворение',
}

STROKE_TIMINGS = {
    'uniform': 'Равномерно',
    'moderate': 'Умеренное ускорение в конце',
}


def stroke_progress(time_fraction: float, mode: str = 'moderate') -> float:
    """Map normalized time to normalized number of finished strokes.

    Moderate makes the last ~20% of strokes fit into ~10% of painting time.
    It does not alter or recalculate any saved strokes.
    """
    t = float(np.clip(time_fraction, 0.0, 1.0))
    if mode == 'uniform':
        return t
    if mode == 'moderate':
        return float(np.clip(t - 1.5 * t**4 * (1.0 - t), 0.0, 1.0))
    raise ValueError(f'Неизвестный режим скорости рисования: {mode}')


def transition_frame(painted: np.ndarray, original: np.ndarray,
                     mode: str, progress: float) -> np.ndarray:
    """Return only the artwork pixels; captions and surrounding frame do not change.

    Both images must be uint8 RGB, H×W×3, and of the same shape.
    Endpoints are exact to keep the output stable before and after transition.
    """
    if mode not in REVEAL_MODES:
        raise ValueError(f'Неизвестный переход: {mode}')
    if painted.shape != original.shape or painted.ndim != 3 or painted.shape[2] != 3:
        raise ValueError('Изображения для перехода должны иметь одинаковые размеры RGB')
    t = float(np.clip(progress, 0.0, 1.0))
    if t <= 0.0:
        return painted.copy()
    if t >= 1.0:
        return original.copy()
    eased = t*t*(3 - 2*t)  # smoother beginning/end without stopping mid-transition
    h,w = painted.shape[:2]

    if mode == 'crossfade':
        return cv2.addWeighted(painted, 1.0-eased, original, eased, 0.0)

    if mode == 'soft_wipe':
        feather = max(4, round(w*0.10))
        edge = -feather/2 + (w+feather)*eased
        x = np.arange(w, dtype=np.float32) + 0.5
        k = np.clip((edge - x)/feather + 0.5, 0, 1)
        k = k*k*(3-2*k)  # soft S-curve across feather zone
        k = k.reshape(1,w,1)
        # Float math deliberately keeps fine feather gradation, not a sharp seam.
        return np.clip(painted.astype(np.float32)*(1-k) +
                       original.astype(np.float32)*k,0,255).astype(np.uint8)

    # Split Reveal: the original occupies the left side, with a visible divider.
    # At the end it vanishes and only the original remains.
    split = max(0,min(w,round(w*eased)))
    result = painted.copy()
    result[:,:split] = original[:,:split]
    if 1 <= split < w:
        width=max(1,round(w*.0025))
        # Shadow + light separator deliberately restricted to artwork pixels.
        cv2.line(result,(split,0),(split,h-1),(39,38,36),width+2,cv2.LINE_AA)
        cv2.line(result,(split,0),(split,h-1),(245,242,234),width,cv2.LINE_AA)
    return result


def export_video(
    painting: Painting, original_rgb: np.ndarray, output_path: str | Path,
    seconds: float = 20, fps: int = 30,
    output_size: tuple[int,int] = (1080,1920),
    show_original: bool = True,
    progress: Callable[[int,int],None] | None = None,
    layout: LayoutOptions | None = None,
    crf: int = 17,
    original_hold_seconds: float = 1.0,
    reveal_mode: str = 'soft_wipe',
    reveal_seconds: float = 1.0,
    pre_reveal_hold_seconds: float = 0.5,
    stroke_timing: str = 'moderate',
) -> Path:
    """Draw vector geometry at final video resolution, then animate the reveal.

    All controls here affect video only. Strokes from the calculated project are
    never changed, so ZIP projects can be re-exported freely.
    """
    if seconds <= 0 or fps < 1:
        raise ValueError('Длительность и FPS должны быть положительными')
    if not (12 <= crf <= 30):
        raise ValueError('CRF должен быть в пределах 12–30')
    if min(original_hold_seconds,pre_reveal_hold_seconds,reveal_seconds) < 0:
        raise ValueError('Длительности перехода и паузы не могут быть отрицательными')
    if reveal_mode not in REVEAL_MODES:
        raise ValueError(f'Неизвестный переход: {reveal_mode}')
    if stroke_timing not in STROKE_TIMINGS:
        raise ValueError(f'Неизвестный режим скорости: {stroke_timing}')
    path = Path(output_path)
    path.parent.mkdir(parents=True,exist_ok=True)
    options = layout or LayoutOptions()
    renderer = FrameRenderer((original_rgb.shape[1],original_rgb.shape[0]),
                             output_size,options,painting.style)
    _,_,aw,ah=renderer.art_rect
    sx,sy=aw/painting.width,ah/painting.height
    strokes=[scale_stroke(s,sx,sy) for s in painting.operations]
    if painting.style in ('oil','dry_brush'):
        from .oil_brush import linen_tone
        canvas=linen_tone(painting.base_color,(ah,aw),.75)
    else:
        canvas=np.empty((ah,aw,3),np.float32)
        canvas[:]=painting.base_color
    original_full=cv2.resize(original_rgb,(aw,ah),interpolation=cv2.INTER_LANCZOS4)
    total=len(strokes)
    paint_frames=max(1,round(seconds*fps))
    rendered=0
    pause_frames=max(0,round(fps*pre_reveal_hold_seconds)) if show_original else 0
    reveal_frames=max(2,round(fps*reveal_seconds)) if show_original else 0
    hold_frames=max(0,round(fps*original_hold_seconds)) if show_original else 0
    total_frames=paint_frames+pause_frames+reveal_frames+hold_frames

    writer=imageio.get_writer(str(path),format='FFMPEG',fps=fps,codec='libx264',
                              macro_block_size=None,quality=None,
                              ffmpeg_params=['-crf',str(crf),'-preset','medium'])
    completed=0
    def add_frame(art: np.ndarray, count: int, is_original: bool=False):
        nonlocal completed
        writer.append_data(renderer.compose(art,count,total,is_original=is_original))
        completed+=1
        if progress and (completed%max(1,fps//2)==0 or completed==total_frames):
            progress(completed,total_frames)

    try:
        for i in range(paint_frames):
            position=total*stroke_progress((i+1)/paint_frames,stroke_timing)
            if painting.style in ('oil','dry_brush'):
                wanted=min(total,int(position))
                while rendered<wanted:
                    apply_stroke(canvas,strokes[rendered]);rendered+=1
                if rendered<total and position-rendered>1e-4:
                    from .oil_brush import paint_oil
                    temporary=canvas.copy()
                    candidate=strokes[rendered]
                    paint_oil(temporary,candidate.geometry,candidate.color,
                              candidate.alpha,progress=position-rendered)
                    art=np.clip(temporary,0,255).astype(np.uint8)
                else:
                    art=np.clip(canvas,0,255).astype(np.uint8)
            else:
                wanted=min(total,round(position))
                while rendered<wanted:
                    apply_stroke(canvas,strokes[rendered]);rendered+=1
                art=np.clip(canvas,0,255).astype(np.uint8)
            add_frame(art,rendered)

        if show_original:
            final=np.clip(canvas,0,255).astype(np.uint8)
            for _ in range(pause_frames):
                add_frame(final,total)
            for i in range(reveal_frames):
                fraction=(i+1)/reveal_frames
                art=transition_frame(final,original_full,reveal_mode,fraction)
                add_frame(art,total,is_original=(i==reveal_frames-1))
            for _ in range(hold_frames):
                add_frame(original_full,total,is_original=True)
        if progress:
            progress(total_frames,total_frames)
    finally:
        writer.close()
    return path
