"""CPU oil-paint brush: a Python adaptation of the stroke construction used in
Alex Greenshpun's anidoodle paintedOilKit.ts (Apache-2.0).

Adapted approach: resampled curved centreline; independent, loaded bristles;
paint running out; lit/dark raised rims; configurable dry-brush and impasto.
This module is not an upstream anidoodle package and does not copy its full
TypeScript implementation. See THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations

import math
import cv2
import numpy as np


def validate_oil_geometry(geometry: list) -> None:
    if not isinstance(geometry, (list, tuple)) or len(geometry) != 5:
        raise ValueError('Oil stroke должен содержать [points, width, dry, impasto, seed]')
    pts, width, dry, impasto, seed = geometry
    if not isinstance(pts, (list, tuple)) or len(pts) < 2 or len(pts) > 64:
        raise ValueError('Некорректная траектория масляного мазка')
    if any(not isinstance(p,(list,tuple)) or len(p)!=2 or any(not math.isfinite(float(a)) or abs(float(a))>1e7 for a in p) for p in pts):
        raise ValueError('Некорректные координаты масляного мазка')
    if not (0.3 <= float(width) <= 3000 and 0 <= float(dry) <= 1 and 0 <= float(impasto) <= 1):
        raise ValueError('Некорректные параметры масляной кисти')
    if not 0 <= int(seed) <= 2**32 - 1:
        raise ValueError('Некорректный seed кисти')


def _path_samples(pts: list, step: float = 2.2) -> np.ndarray:
    """Uniformly sample a quadratic Bezier (for three control points)."""
    p = np.asarray(pts, dtype=np.float32)
    if len(p) == 3:
        length = float(np.linalg.norm(p[0]-p[1])+np.linalg.norm(p[1]-p[2]))
        n = int(np.clip(math.ceil(length/step), 6, 320))
        t = np.linspace(0, 1, n, dtype=np.float32)[:,None]
        return (1-t)**2*p[0] + 2*t*(1-t)*p[1] + t*t*p[2]
    segs = np.linalg.norm(np.diff(p,axis=0),axis=1)
    n = int(np.clip(math.ceil(float(segs.sum())/step), 6, 320))
    s = np.concatenate([[0.], np.cumsum(segs)])
    if s[-1] < 1e-5:
        return np.repeat(p[:1],n,axis=0)
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t,s,p[:,0]),np.interp(t,s,p[:,1])],axis=1).astype(np.float32)


def _mix(canvas: np.ndarray, mask: np.ndarray, color: np.ndarray, alpha: float) -> None:
    """Blend only nonzero-mask pixels, avoiding full-image intermediate arrays."""
    ys,xs = np.nonzero(mask)
    if not len(ys):
        return
    a = (mask[ys,xs].astype(np.float32)/255 * alpha)[:,None]
    canvas[ys,xs] = canvas[ys,xs] * (1-a) + color * a


def paint_oil(canvas: np.ndarray, geometry: list, color: tuple[int,int,int],
              alpha: float=1.0, progress: float=1.0) -> None:
    """Paint a deterministic, multi-bristle curved stroke onto an RGB float32 canvas.

    progress 0..1 draws a partial brush stroke, suitable for animation. Geometry
    and seed are saved in painting.json so renders at any resolution are stable.
    """
    if progress <= 0:
        return
    pts,width,dry,impasto,seed = geometry
    width = float(width); dry = float(dry); impasto=float(impasto)
    h,w=canvas.shape[:2]
    samples=_path_samples(pts,step=max(1.8,width/15))
    last=max(2,min(len(samples),math.ceil(len(samples)*min(1.,progress))))
    samples=samples[:last]
    tangent=np.gradient(samples,axis=0)
    denom=np.maximum(np.linalg.norm(tangent,axis=1),1e-6)
    normal=np.stack([-tangent[:,1]/denom,tangent[:,0]/denom],axis=1)
    bound_pad=width*1.25+5
    x0=max(0,int(math.floor(np.min(samples[:,0])-bound_pad)))
    y0=max(0,int(math.floor(np.min(samples[:,1])-bound_pad)))
    x1=min(w,int(math.ceil(np.max(samples[:,0])+bound_pad))+1)
    y1=min(h,int(math.ceil(np.max(samples[:,1])+bound_pad))+1)
    if x1<=x0 or y1<=y0:
        return
    roi=canvas[y0:y1,x0:x1]
    rh,rw=roi.shape[:2]
    xy=samples.copy();xy[:,0]-=x0;xy[:,1]-=y0
    rng=np.random.default_rng(int(seed))
    base=np.asarray(color,dtype=np.float32)
    t=np.linspace(0,1,last,dtype=np.float32)
    profile=np.where(t<.08,.9+1.4*t,np.where(t>.8,1-1.4*(t-.8),1.))
    profile=np.clip(profile*(1+.04*np.sin(t*9+int(seed)%113)),.55,1.15)

    # Thin base coat gives wet paint its connected body. Dry brush skips most
    # of this coat, leaving visible gaps between separately loaded bristles.
    if dry < .85:
        base_mask=np.zeros((rh,rw),np.uint8)
        p=xy.round().astype(np.int32)
        cv2.polylines(base_mask,[p],False,255,max(1,round(width*.80)),cv2.LINE_AA)
        # The thin underpainting is deliberately subdued; bristles supply detail.
        coat_strength=.40 * (1-dry)**1.4
        _mix(roi,base_mask,base,alpha*coat_strength)

    # Each hair is a separate line, with a small independent start/end and
    # colour variation. The drawing is deterministic from the saved seed.
    bristles=int(np.clip(round(width/2.5),6,24))
    if width <= 2:
        bristles=4
    light=np.asarray([255,244,222],dtype=np.float32)
    dark=np.asarray([29,16,12],dtype=np.float32)
    for b in range(bristles):
        u=-1+2*(b+.5)/bristles + rng.uniform(-.35,.35)/bristles
        tone=rng.uniform(-.17,.17)
        load=rng.uniform(.90,1.16)
        start=rng.uniform(0,.05)*(0.4+0.6*abs(u))
        end=1-rng.uniform(0,.08)
        # Dryness increases discontinuity; vary it per hair and along the path.
        noise=rng.random(last)
        remain=(load-(.20+dry*.73)*t*(1+.45*abs(u)))
        threshold=(.06+dry*.56)*noise + .07
        keep=(t>=start)&(t<=end)&(remain>threshold)
        if dry >= .35:
            keep &= (rng.random(last) > (dry**1.65)*(.12+.46*t))
        thickness=max(1,round(width/bristles*1.85))
        displaced=xy + normal * (u*width*profile/2)[:,None]
        if tone>0:
            c=base*(1-tone)+light*tone
        else:
            c=base*(1+tone)+dark*(-tone)
        c=np.clip(c,0,255)
        mask=np.zeros((rh,rw),np.uint8)
        # Each continuous run of paint becomes one short, curved bristle trail.
        begin=None
        for j,ok in enumerate(np.r_[keep,False]):
            if ok and begin is None:
                begin=j
            elif not ok and begin is not None:
                section=displaced[begin:j]
                if len(section)>1:
                    cv2.polylines(mask,[section.round().astype(np.int32)],False,255,thickness,cv2.LINE_AA)
                elif len(section)==1:
                    cv2.circle(mask,tuple(section[0].round().astype(int)),max(1,thickness//2),255,-1,cv2.LINE_AA)
                begin=None
        _mix(roi,mask,c,alpha*(.78+.20*load))

    # Thin highlights and shadowed rims give a restrained raised-paint effect.
    if impasto > .03 and len(xy)>2 and width>=2:
        for side in (-1,1):
            edge=xy+normal*(side*width*profile*.46)[:,None]
            edge_mask=np.zeros((rh,rw),np.uint8)
            cv2.polylines(edge_mask,[edge.round().astype(np.int32)],False,255,
                          max(1,round(min(2.5,width*.08))),cv2.LINE_AA)
            # Light from upper left, as in anidoodle's paintedOilKit.
            lit_side=float(np.mean((normal[:,0]*(-.9)+normal[:,1]*(-.44))*side)) > 0
            tint=light if lit_side else dark
            amount=(.17+.36*impasto) if lit_side else (.14+.27*impasto)
            _mix(roi,edge_mask,base*(1-amount)+tint*amount,alpha*(.3+.35*impasto))


def linen_tone(base: tuple[int,int,int], shape: tuple[int,int], strength: float=1) -> np.ndarray:
    """Small deterministic linen weave at output size, without altering geometry."""
    h,w=shape
    yy,xx=np.indices((h,w),dtype=np.int32)
    cross=((xx%5==0).astype(np.float32)+(yy%5==0).astype(np.float32))
    checker=((xx//5+yy//5)%2).astype(np.float32)
    texture=(cross*1.4-checker*.7-1.0)*strength
    return np.clip(np.asarray(base,np.float32)[None,None,:]+texture[:,:,None],0,255).astype(np.float32)
