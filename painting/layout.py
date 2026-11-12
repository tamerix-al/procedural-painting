"""Editable video composition with Unicode labels and native-resolution art."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .engine import STYLES


@dataclass
class LayoutOptions:
    channel: str = "MY ART CHANNEL"
    technique: str = ""  # Empty = label taken from selected algorithm.
    painting_title: str = "Название картины"
    artist: str = "Имя художника"
    year: str = ""
    background: str = "#F8F7F4"
    foreground: str = "#292827"
    muted: str = "#7B7874"
    accent: str = "#33312E"
    margin: int = 66          # All pixel settings are relative to 1080px width.
    header_y: int = 75
    image_width: int = 100    # Percent of space between left and right margins.
    image_offset_y: int = 0   # Move complete image+caption group up/down.
    caption_gap: int = 50
    header_font_size: int = 28
    title_font_size: int = 52
    meta_font_size: int = 30
    font_family: str = "sans"  # sans, serif
    image_border: bool = True
    show_counter: bool = True
    show_progress: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def _color(value: str) -> tuple[int,int,int]:
    value=value.strip().lstrip('#')
    if len(value)!=6:
        raise ValueError("Цвет должен иметь формат #RRGGBB")
    return tuple(int(value[i:i+2],16) for i in (0,2,4))


@lru_cache(maxsize=32)
def _font(family: str, size: int, bold: bool=False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    size=max(8,size)
    windows=Path('C:/Windows/Fonts')
    if sys.platform.startswith('win'):
        options = ([windows/'georgiab.ttf',windows/'georgia.ttf'] if bold else [windows/'georgia.ttf']) if family=='serif' else ([windows/'arialbd.ttf',windows/'segoeuib.ttf'] if bold else [windows/'arial.ttf',windows/'segoeui.ttf'])
    else:
        options = ([Path('/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf')] if bold else [Path('/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf')]) if family=='serif' else ([Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf')] if bold else [Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')])
    options += [Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),windows/'arial.ttf']
    for path in options:
        if path.exists():
            return ImageFont.truetype(str(path),size)
    return ImageFont.load_default(size=size)


def _size(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int,int]:
    if not text:
        return 0,0
    l,t,r,b=draw.textbbox((0,0),text,font=font)
    return r-l,b-t


def _fit_one_line(draw, text, family, target_size, width, bold=False):
    size=target_size
    while size>9 and _size(draw,text,_font(family,size,bold))[0]>width:
        size-=1
    return _font(family,size,bold)


def _wrap_title(draw,text,font,max_width) -> list[str]:
    words=text.split()
    if not words:
        return []
    lines=[]; line=''
    for word in words:
        trial=(line+' '+word).strip()
        if line and _size(draw,trial,font)[0]>max_width:
            lines.append(line)
            line=word
        else:
            line=trial
    if line:
        lines.append(line)
    return lines


class FrameRenderer:
    """Cache all text and decorations; replace only painting pixels per frame."""
    def __init__(self, art_size: tuple[int,int], size: tuple[int,int],
                 options: LayoutOptions, style: str):
        self.width,self.height=size
        self.options=options
        if self.width%2 or self.height%2:
            raise ValueError("Разрешение должно иметь чётные стороны")
        w,h=size
        scale=w/1080.0
        px=lambda n:max(1,round(n*scale))
        margin=px(options.margin)
        foreground=_color(options.foreground)
        muted=_color(options.muted)
        bg=_color(options.background)
        accent=_color(options.accent)
        base=Image.new('RGB',(w,h),bg)
        draw=ImageDraw.Draw(base)
        header_size=px(options.header_font_size)
        left=(options.channel or '').strip()
        right=(options.technique or STYLES.get(style,style)).strip().upper()
        # Fit both labels into independent halves without collisions.
        half=max(40,(w-2*margin)//2-px(12))
        lf=_fit_one_line(draw,left,'sans',header_size,half,True)
        rf=_fit_one_line(draw,right,'sans',header_size,half,False)
        header_y=px(options.header_y)
        draw.text((margin,header_y),left,fill=foreground,font=lf,anchor='lt')
        draw.text((w-margin,header_y),right,fill=foreground,font=rf,anchor='rt')

        title=(options.painting_title or '').strip()
        artist=(options.artist or '').strip()
        year=(options.year or '').strip()
        avail_w=w-2*margin
        title_size=px(options.title_font_size)
        title_font=_font(options.font_family,title_size,True)
        title_lines=_wrap_title(draw,title,title_font,avail_w)
        while (len(title_lines)>2 or any(_size(draw,l,title_font)[0]>avail_w for l in title_lines)) and title_size>px(19):
            title_size-=1
            title_font=_font(options.font_family,title_size,True)
            title_lines=_wrap_title(draw,title,title_font,avail_w)
        # Still extremely long unbroken strings: truncate safely at measured width.
        title_lines=title_lines[:2]
        title_lines=[self._clip(draw,l,title_font,avail_w) for l in title_lines]
        title_line_h=max(px(30),int(title_size*1.35))
        meta_font=_font('sans',px(options.meta_font_size),False)
        meta_h=int(px(options.meta_font_size)*1.4)
        caption_h=len(title_lines)*title_line_h
        if artist: caption_h+=px(15)+meta_h
        if year: caption_h+=px(5)+meta_h
        if title_lines or artist or year: caption_h+=px(options.caption_gap)

        art_w0,art_h0=art_size
        max_art_w=int(avail_w*max(.30,min(1.0,options.image_width/100)))
        safe_top=header_y+header_size+px(52)
        safe_bottom=px(100) if options.show_progress else px(50)
        max_art_h=max(100,h-safe_top-caption_h-safe_bottom)
        fit=min(max_art_w/art_w0,max_art_h/art_h0)
        art_w=max(1,round(art_w0*fit));art_h=max(1,round(art_h0*fit))
        group_h=art_h+caption_h
        free_top=safe_top
        free_bottom=h-safe_bottom
        group_y=int(free_top+(free_bottom-free_top-group_h)/2)+round(options.image_offset_y*scale)
        group_y=max(safe_top,min(group_y,free_bottom-group_h))
        art_x=(w-art_w)//2
        art_y=group_y
        self.art_rect=(art_x,art_y,art_w,art_h)
        if options.image_border:
            draw.rectangle((art_x-1,art_y-1,art_x+art_w,art_y+art_h),outline=muted,width=max(1,px(2)))
        baseline=art_y+art_h+px(options.caption_gap)
        for line in title_lines:
            draw.text((w//2,baseline),line,font=title_font,fill=foreground,anchor='mt')
            baseline+=title_line_h
        if artist:
            baseline+=px(11) if title_lines else 0
            draw.text((w//2,baseline),self._clip(draw,artist,meta_font,avail_w),font=meta_font,fill=muted,anchor='mt')
            baseline+=meta_h
        if year:
            baseline+=px(5)
            draw.text((w//2,baseline),self._clip(draw,year,meta_font,avail_w),font=meta_font,fill=muted,anchor='mt')
        self.progress_left=margin
        self.progress_right=w-margin
        self.progress_y=h-px(58)
        self.progress_thickness=max(2,px(3))
        self.counter_font=_font('sans',px(20))
        if options.show_progress:
            draw.line((self.progress_left,self.progress_y,self.progress_right,self.progress_y),fill=muted,width=self.progress_thickness)
        self.counter_y=h-px(93)
        self.base=np.asarray(base).copy()
        self.accent_bgr=accent[::-1]
        self.accent=accent
        self.text_color=foreground
        self.scale=scale

    @staticmethod
    def _clip(draw,text,font,max_width):
        if _size(draw,text,font)[0]<=max_width:
            return text
        while text and _size(draw,text+'…',font)[0]>max_width:
            text=text[:-1]
        return text+'…'

    def compose(self, art_rgb: np.ndarray, count:int=0, total:int=1,
                is_original:bool=False) -> np.ndarray:
        """Insert native-resolution artwork and draw optional counters/progress."""
        x,y,w,h=self.art_rect
        if art_rgb.shape[:2] != (h,w):
            art_rgb=cv2.resize(art_rgb,(w,h),interpolation=cv2.INTER_LANCZOS4)
        frame=self.base.copy()
        frame[y:y+h,x:x+w] = art_rgb
        pct=1 if is_original else count/max(1,total)
        if self.options.show_progress:
            endpoint=self.progress_left+int((self.progress_right-self.progress_left)*pct)
            cv2.line(frame,(self.progress_left,self.progress_y),(endpoint,self.progress_y),self.accent,
                     self.progress_thickness,cv2.LINE_AA)
        if self.options.show_counter:
            label='ORIGINAL' if is_original else f'{count} / {total}'
            # Pillow supports Cyrillic and kerning; counter remains crisp.
            canvas=Image.fromarray(frame)
            d=ImageDraw.Draw(canvas)
            d.text((self.progress_right,self.counter_y),label,font=self.counter_font,
                   fill=self.text_color,anchor='rt')
            frame=np.asarray(canvas).copy()
        return frame


def preview_layout(image_rgb:np.ndarray, style:str, options:LayoutOptions,
                   output_size:tuple[int,int]=(540,960)) -> np.ndarray:
    original_h,original_w=image_rgb.shape[:2]
    renderer=FrameRenderer((original_w,original_h),output_size,options,style)
    return renderer.compose(image_rgb,count=0,total=0,is_original=True)
