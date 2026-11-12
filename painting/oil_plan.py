"""Automatically place oil strokes over arbitrary user images.

The position/colour optimiser is the project's existing coarse-to-fine Python
brush reconstruction. This module then converts its geometric paths into
bristle strokes, deterministically, so saved plans can be replayed at any size.
"""
from __future__ import annotations

from typing import Callable
import numpy as np


def build_oil_painting(image_rgb: np.ndarray, style: str, strokes: int,
                       candidates: int, working_size: int, seed: int,
                       progress: Callable[[int,int],None] | None,
                       detail: float, opacity: float,
                       dryness: float | None = None,
                       impasto: float | None = None,
                       length_ratio: float = 1.6, size_factor: float = .85,
                       follow_flow: bool = True):
    """Optimise brush placement, then produce reusable oil-bristle operations."""
    from .engine import Painting, Stroke, build_painting

    if style not in ('oil','dry_brush'):
        raise ValueError('Неизвестный тип кисти')
    dry_default=0.24 if style=='oil' else 0.79
    imp_default=0.53 if style=='oil' else 0.16
    dry=float(dryness) if dryness is not None else dry_default
    imp=float(impasto) if impasto is not None else imp_default
    if not .8<=length_ratio<=5.0 or not .4<=size_factor<=2.0:
        raise ValueError('Некорректный размер или длина мазка')
    if not 0<=dry<=1 or not 0<=imp<=1:
        raise ValueError('Сухость и рельеф должны быть в диапазоне от 0 до 1')

    # Compact short-stroke optimization: match short coloured dabs in the
    # reference rather than passing legacy elongated strokes into the oil shader.
    # Placement uses the image error and smoothed local edge directions.
    sketch=build_painting(image_rgb,style='brush',strokes=strokes,
                           candidates=candidates,working_size=working_size,
                           seed=seed,progress=progress,detail=detail,opacity=opacity,
                           brush_profile='compact',oil_length_ratio=length_ratio,
                           oil_size=size_factor,oil_follow_flow=follow_flow)
    rng=np.random.default_rng(seed^0x0A11)
    ops=[]
    total=max(1,len(sketch.operations))
    for i,src in enumerate(sketch.operations):
        if src.kind!='brush':
            continue
        pts,width=src.geometry
        f=i/max(1,total-1)
        # More solid underpainting, more relieved detail late in the sequence.
        dyn_dry=float(np.clip(dry*(0.7+0.4*f)+rng.uniform(-.07,.07),0,1))
        dyn_imp=float(np.clip(imp*(.6+.5*f)+rng.uniform(-.07,.07),0,1))
        # Width already comes from the compact-dab optimiser and is matched
        # to the length. Do not inflate it significantly during oil conversion.
        scaled_width=max(.6,float(width)*(1.00 if style=='oil' else 1.04))
        geom=[[[float(x),float(y)] for x,y in pts],scaled_width,
              round(dyn_dry,4),round(dyn_imp,4),int(rng.integers(0,2**31))]
        ops.append(Stroke('oil',geom,src.color,src.alpha))
    return Painting(sketch.base_color,ops,sketch.target_rgb,sketch.width,sketch.height,style)
