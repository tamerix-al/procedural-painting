"""Save optimized painting operations separately from MP4 rendering."""
from __future__ import annotations

from dataclasses import asdict
from io import BytesIO
import json
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
from PIL import Image

from .engine import Painting, Stroke
from .layout import LayoutOptions

MODEL_VERSION=1
MAX_OPS=100_000


def painting_to_dict(painting: Painting) -> dict:
    return {
        'version': MODEL_VERSION,
        'base_color': list(painting.base_color),
        'width': painting.width,
        'height': painting.height,
        'style': painting.style,
        'operations': [asdict(o) for o in painting.operations],
    }


def painting_from_dict(data: dict) -> Painting:
    if data.get('version')!=MODEL_VERSION:
        raise ValueError('Несовместимая версия проекта')
    width=int(data['width']);height=int(data['height'])
    if not 1<=width<=10000 or not 1<=height<=10000:
        raise ValueError('Некорректное разрешение модели')
    raw=data['operations']
    if not isinstance(raw,list) or not 1<=len(raw)<=MAX_OPS:
        raise ValueError('Слишком много или мало фигур')
    base=tuple(int(x) for x in data['base_color'])
    if len(base)!=3 or any(x<0 or x>255 for x in base):
        raise ValueError('Некорректный цвет фона')
    operations=[]
    for op in raw:
        col=tuple(int(x) for x in op['color'])
        kind=str(op['kind'])
        alpha=float(op['alpha'])
        if kind not in ('poly','ellipse','circle','brush','oil','line','polyline') or len(col)!=3 or not (0<=alpha<=1):
            raise ValueError('Некорректная операция рисования')
        if kind == 'oil':
            from .oil_brush import validate_oil_geometry
            validate_oil_geometry(op['geometry'])
        operations.append(Stroke(kind,op['geometry'],col,alpha))
    placeholder=np.zeros((1,1,3),np.uint8)
    return Painting(base,operations,placeholder,width,height,str(data['style']))


def export_project(original_rgb:np.ndarray,painting:Painting,layout:LayoutOptions,
                   svg_bytes:bytes | None = None) -> bytes:
    image=BytesIO()
    Image.fromarray(original_rgb.astype(np.uint8),'RGB').save(image,format='PNG')
    out=BytesIO()
    with ZipFile(out,'w',ZIP_DEFLATED,compresslevel=6) as archive:
        archive.writestr('source.png',image.getvalue())
        archive.writestr('painting.json',json.dumps(painting_to_dict(painting),ensure_ascii=False,separators=(',',':')))
        archive.writestr('layout.json',json.dumps(layout.to_dict(),ensure_ascii=False,indent=2))
        if svg_bytes:
            archive.writestr('primitive_original.svg',svg_bytes)
    return out.getvalue()


def import_project(archive_bytes:bytes) -> tuple[np.ndarray,Painting,dict,bytes|None]:
    if len(archive_bytes)>150_000_000:
        raise ValueError('Слишком большой архив проекта')
    with ZipFile(BytesIO(archive_bytes)) as archive:
        names=set(archive.namelist())
        if not {'source.png','painting.json'}.issubset(names):
            raise ValueError('Это не проект Procedural Painting (нет source.png или painting.json)')
        if any(archive.getinfo(name).file_size>120_000_000 for name in ('source.png','painting.json')):
            raise ValueError('Слишком большие данные проекта')
        with Image.open(BytesIO(archive.read('source.png'))) as image:
            image.load()
            original=np.asarray(image.convert('RGB')).copy()
        model=painting_from_dict(json.loads(archive.read('painting.json')))
        opts=json.loads(archive.read('layout.json')) if 'layout.json' in names else {}
        svg=archive.read('primitive_original.svg') if 'primitive_original.svg' in names else None
        return original,model,opts,svg
