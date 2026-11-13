"""Bridge to Michael Fogleman's Primitive CLI (Go, MIT).

The executable is NOT bundled. Primitive performs the optimization; this
module converts its SVG output to our portable sequence of painting operations.
See https://github.com/fogleman/primitive and THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import tempfile
from typing import Callable
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from .engine import Painting, Stroke

# These values correspond to flags in primitive/main.go.
PRIMITIVE_MODES = {
    'triangles': ('Треугольники', 1),
    'rectangles': ('Прямоугольники', 2),
    'ellipses': ('Эллипсы', 3),
    'circles': ('Круги', 4),
    'rotated_rectangles': ('Повернутые прямоугольники', 5),
    'bezier': ('Кривые Безье / штрихи', 6),
    'rotated_ellipses': ('Повернутые эллипсы', 7),
    'mix': ('Смешанные фигуры', 0),
}

NUMBER = re.compile(r'[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?')
TRANSFORM = re.compile(r'([a-zA-Z]+)\s*\(([^)]*)\)')
PROGRESS = re.compile(r'^\s*(\d+):\s+t=', re.MULTILINE)


def locate_primitive(explicit: str | None = None) -> Path | None:
    candidates = []
    if explicit:
        candidates.append(Path(explicit.strip().strip('"')).expanduser())
    if os.environ.get('PAINTING_PRIMITIVE_BIN'):
        candidates.append(Path(os.environ['PAINTING_PRIMITIVE_BIN']).expanduser())
    base = Path(__file__).resolve().parent.parent
    candidates += [base/'tools'/'primitive.exe', base/'tools'/'primitive']
    found = shutil.which('primitive')
    if found:
        candidates.append(Path(found))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None



_INSTALL_LOCK = threading.Lock()

# Primitive does not publish a go.mod with pinned dependencies.  Building it
# with `go install github.com/fogleman/primitive@latest` pulls the newest
# golang.org/x/image, which currently requires a newer Go than the version
# provided by Streamlit Cloud's Debian environment.  Build it in a temporary
# Go module with known-compatible dependencies instead.
PRIMITIVE_BUILD_GO_MOD = """module procedural-painting/primitive-bootstrap

go 1.23.0

require (
    github.com/fogleman/primitive v0.0.0-20200504002142-0373c216458b
    github.com/fogleman/gg v1.3.0
    github.com/golang/freetype v0.0.0-20170609003504-e2365dfdc4a0
    github.com/nfnt/resize v0.0.0-20180221191011-83c6a9932646
    golang.org/x/image v0.27.0
    golang.org/x/text v0.25.0
)
"""


def install_primitive_if_missing(timeout: int = 300) -> Path:
    """Build Primitive on Linux/macOS using pinned Go dependencies.

    Streamlit Cloud can reuse the binary for as long as its instance exists;
    a fresh server image may repeat the build. On Windows the original .bat
    installer is still used.
    """
    existing = locate_primitive()
    if existing is not None:
        return existing
    if os.name == 'nt':
        raise RuntimeError('Для Windows запусти setup_primitive_windows.bat.')
    go = shutil.which('go')
    if go is None:
        raise RuntimeError('Go не найден. Для Streamlit Cloud добавь golang-go в packages.txt.')

    with _INSTALL_LOCK:
        existing = locate_primitive()
        if existing is not None:
            return existing
        root = Path(__file__).resolve().parent.parent
        out_dir = root / 'tools'
        out_dir.mkdir(parents=True, exist_ok=True)
        output = out_dir / 'primitive'
        env = os.environ.copy()
        env['GOTOOLCHAIN'] = 'local'
        try:
            # A private module makes go build honor the explicitly pinned
            # versions. The @latest form of go install ignores the local go.mod.
            with tempfile.TemporaryDirectory(prefix='painting_primitive_go_') as tmp:
                build_dir = Path(tmp)
                (build_dir / 'go.mod').write_text(PRIMITIVE_BUILD_GO_MOD, encoding='utf-8')
                temp_binary = build_dir / 'primitive'
                command = [go, 'build', '-mod=mod', '-o', str(temp_binary),
                           'github.com/fogleman/primitive']
                result = subprocess.run(command, capture_output=True, text=True,
                                        env=env, cwd=build_dir, timeout=timeout,
                                        check=False)
                if result.returncode != 0:
                    message = (result.stderr or result.stdout).strip()[-1800:]
                    raise RuntimeError(f'Ошибка установки Primitive (Go): {message}')
                if not temp_binary.is_file():
                    raise RuntimeError('Go завершил сборку, но исполняемый файл Primitive не найден.')
                temp_binary.chmod(temp_binary.stat().st_mode | 0o111)
                # /tmp and the app directory may be on different mounted
                # filesystems (e.g. Streamlit Community Cloud). os.replace()
                # across these filesystems fails with EXDEV. Copy the completed
                # binary into a temporary file beside its final destination,
                # then atomically replace it within that same filesystem.
                staging_path = None
                try:
                    fd, staging_name = tempfile.mkstemp(
                        prefix='.primitive-install-', suffix='.tmp', dir=out_dir,
                    )
                    os.close(fd)
                    staging_path = Path(staging_name)
                    shutil.copy2(temp_binary, staging_path)
                    staging_path.chmod(staging_path.stat().st_mode | 0o111)
                    os.replace(staging_path, output)
                finally:
                    if staging_path is not None:
                        staging_path.unlink(missing_ok=True)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError('Сборка Primitive превысила допустимое время. Проверь логи приложения.') from exc
        except OSError as exc:
            raise RuntimeError(f'Не удалось запустить Go: {exc}') from exc
        return output


def _num(value: str | None, fallback: float = 0.0) -> float:
    return float(value) if value is not None else fallback


def _numbers(value: str) -> list[float]:
    return [float(v) for v in NUMBER.findall(value)]


def _apply(matrix: np.ndarray, x: float, y: float) -> list[float]:
    p = matrix @ np.array([x, y, 1.0], np.float64)
    return [float(p[0]), float(p[1])]


def _matrix(value: str | None) -> np.ndarray:
    """Parse SVG transforms. Order follows SVG specification (right-to-left)."""
    total = np.eye(3, dtype=np.float64)
    for name, args in TRANSFORM.findall(value or ''):
        v = _numbers(args)
        m = np.eye(3, dtype=np.float64)
        name = name.lower()
        if name == 'translate':
            m[0,2] = v[0]
            m[1,2] = v[1] if len(v)>1 else 0.0
        elif name == 'scale':
            m[0,0] = v[0]
            m[1,1] = v[1] if len(v)>1 else v[0]
        elif name == 'rotate':
            t = math.radians(v[0])
            co,si = math.cos(t),math.sin(t)
            m[:2,:2] = [[co,-si],[si,co]]
            if len(v)==3:
                m[0,2] = v[1]-co*v[1]+si*v[2]
                m[1,2] = v[2]-si*v[1]-co*v[2]
        elif name=='matrix' and len(v)==6:
            a,b,c,d,e,f = v
            m = np.array([[a,c,e],[b,d,f],[0,0,1]], dtype=float)
        else:
            raise ValueError(f'Неподдерживаемое SVG-преобразование: {name}')
        total = total @ m
    return total


def _color(s: str | None) -> tuple[int,int,int]:
    if s is None:
        raise ValueError('В фигуре не задан цвет')
    if s.startswith('#'):
        val = s[1:]
        if len(val)==3:
            val=''.join(ch*2 for ch in val)
        if len(val)!=6:
            raise ValueError(f'Неизвестный SVG-цвет: {s}')
        return tuple(int(val[i:i+2],16) for i in (0,2,4))
    raise ValueError(f'В SVG Primitive ожидается цвет в формате #RRGGBB: {s}')


def _ellipse(center, rx, ry, matrix) -> tuple[list, float]:
    # The transforms emitted by Primitive are rotations, translations and scales.
    c = np.asarray(_apply(matrix,*center))
    axis_x = np.asarray(_apply(matrix,center[0]+rx,center[1]))-c
    axis_y = np.asarray(_apply(matrix,center[0],center[1]+ry))-c
    rr_x, rr_y = float(np.linalg.norm(axis_x)), float(np.linalg.norm(axis_y))
    angle = math.degrees(math.atan2(axis_x[1],axis_x[0]))
    return [round(c[0],4),round(c[1],4),round(rr_x,4),round(rr_y,4),round(angle,4)], abs(rr_x*rr_y)


def _stroke_for_element(el: ET.Element, matrix: np.ndarray, inherited: dict) -> Stroke | None:
    name=el.tag.rsplit('}',1)[-1]
    if name not in ('polygon','rect','circle','ellipse','path'):
        return None
    attrs=dict(inherited)
    attrs.update(el.attrib)
    is_path=name=='path'
    rgb=_color(attrs.get('stroke') if is_path else attrs.get('fill'))
    alpha=float(attrs.get('stroke-opacity' if is_path else 'fill-opacity','1'))
    alpha *= float(attrs.get('opacity','1'))
    alpha=max(0.0,min(1.0,alpha))
    if name=='polygon':
        nums=_numbers(attrs.get('points',''))
        if len(nums)<6 or len(nums)%2:
            raise ValueError('Некорректный polygon в SVG')
        return Stroke('poly',[_apply(matrix,nums[i],nums[i+1]) for i in range(0,len(nums),2)],rgb,alpha)
    if name=='rect':
        x=_num(attrs.get('x')); y=_num(attrs.get('y'))
        w=_num(attrs.get('width'));h=_num(attrs.get('height'))
        if w<=0 or h<=0:
            return None
        pts=[(x,y),(x+w,y),(x+w,y+h),(x,y+h)]
        return Stroke('poly',[_apply(matrix,*p) for p in pts],rgb,alpha)
    if name in ('circle','ellipse'):
        cx=_num(attrs.get('cx')); cy=_num(attrs.get('cy'))
        rx=_num(attrs.get('r')) if name=='circle' else _num(attrs.get('rx'))
        ry=rx if name=='circle' else _num(attrs.get('ry'))
        if rx<=0 or ry<=0:
            return None
        geom,_=_ellipse([cx,cy],rx,ry,matrix)
        # Even circles are kept as ellipses because the video layout can scale x/y.
        return Stroke('ellipse',geom,rgb,alpha)
    # Primitive mode 6 produces a quadratic path: M x y Q cx cy, x y.
    path=attrs.get('d','')
    if not re.match(r'^\s*M\s',path,re.IGNORECASE) or not re.search(r'\bQ\s',path,re.IGNORECASE):
        raise ValueError('Поддерживаются только квадратичные кривые Primitive (M ... Q ...)')
    nums=_numbers(path)
    if len(nums)!=6:
        raise ValueError('Некорректные координаты кривой Безье')
    p0=np.array(nums[0:2]);p1=np.array(nums[2:4]);p2=np.array(nums[4:6])
    pts=[]
    for t in np.linspace(0,1,41):
        p=(1-t)**2*p0+2*(1-t)*t*p1+t*t*p2
        pts.append(_apply(matrix,*p))
    width=_num(attrs.get('stroke-width'),1)
    size_scale=(np.linalg.norm(matrix[:2,0])+np.linalg.norm(matrix[:2,1]))/2
    return Stroke('polyline',[pts,max(0.5,width*size_scale)],rgb,alpha)


def parse_primitive_svg(svg_content: str | bytes, style: str = 'triangles', target_rgb: np.ndarray | None = None) -> Painting:
    """Parse the exact SVG shapes emitted by fogleman/primitive.

    The first ungrouped <rect> is a solid background. Subsequent shapes are
    accumulated in document order. SVG is parsed without running scripts.
    """
    raw=svg_content.encode('utf-8') if isinstance(svg_content,str) else svg_content
    if len(raw)>40_000_000:
        raise ValueError('Слишком большой SVG')
    root=ET.fromstring(raw)
    if root.tag.rsplit('}',1)[-1]!='svg':
        raise ValueError('Это не SVG')
    w=round(_num(root.attrib.get('width')))
    h=round(_num(root.attrib.get('height')))
    if not 1<=w<=10000 or not 1<=h<=10000:
        raise ValueError('В SVG нет корректного размера холста')
    base=(245,245,245)
    strokes=[]
    for node in root:
        tag=node.tag.rsplit('}',1)[-1]
        if tag=='rect' and not strokes:  # Root's flat background.
            try:base=_color(node.attrib.get('fill'))
            except ValueError:pass
            continue
        if tag=='g':
            def walk(el, parent_matrix, inherited):
                tag2=el.tag.rsplit('}',1)[-1]
                matrix=parent_matrix @ _matrix(el.attrib.get('transform'))
                if tag2=='g':
                    inherited={**inherited,**{k:v for k,v in el.attrib.items() if k in ('fill','fill-opacity','stroke','stroke-opacity','opacity')}}
                else:
                    stroke=_stroke_for_element(el,matrix,inherited)
                    if stroke is not None:
                        strokes.append(stroke)
                for child in el:
                    walk(child,matrix,inherited)
            walk(node,np.eye(3),{})
        else:
            stroke=_stroke_for_element(node,np.eye(3),{})
            if stroke is not None:strokes.append(stroke)
    if not strokes:
        raise ValueError('Primitive не вернул ни одной фигуры')
    if target_rgb is None:
        target_rgb=np.zeros((min(64,h),min(64,w),3),np.uint8)
    return Painting(base,strokes,target_rgb,w,h,style)


def run_primitive(
    original_rgb: np.ndarray, style: str, count: int, analysis_size: int = 360,
    alpha: int = 128, workers: int = 0, executable: str | None = None,
    progress: Callable[[int,int],None] | None = None,
) -> tuple[Painting,bytes]:
    """Run Primitive CLI and return (portable Painting, original SVG bytes).

    Note: Primitive's own optimization is not seedable from this CLI.
    """
    binary=locate_primitive(executable)
    if binary is None:
        raise FileNotFoundError('Primitive не установлен: на Windows запусти setup_primitive_windows.bat, в Streamlit Cloud проверь packages.txt и логи установки Go.')
    if style not in PRIMITIVE_MODES:
        raise ValueError(f'Неподдерживаемая техника Primitive: {style}')
    if count<1 or analysis_size<32:
        raise ValueError('Некорректные параметры Primitive')
    if not 0<=alpha<=255:
        raise ValueError('Альфа должна быть 0–255')
    with tempfile.TemporaryDirectory(prefix='painting_primitive_') as tmp:
        target=Path(tmp)/'target.png'
        result=Path(tmp)/'result.svg'
        Image.fromarray(original_rgb.astype(np.uint8),'RGB').save(target)
        cmd=[str(binary),'-i',str(target),'-o',str(result),'-n',str(count),
             '-m',str(PRIMITIVE_MODES[style][1]),'-r',str(analysis_size),
             '-s','1024','-a',str(alpha),'-j',str(max(0,workers)),'-v']
        try:
            startupinfo=None
            if os.name=='nt':
                startupinfo=subprocess.STARTUPINFO()
                startupinfo.dwFlags|=subprocess.STARTF_USESHOWWINDOW
            process=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                     text=True,encoding='utf-8',errors='replace',startupinfo=startupinfo)
            logs=[]
            assert process.stdout is not None
            for line in process.stdout:
                logs.append(line.strip())
                match=PROGRESS.match(line)
                if match and progress:
                    progress(min(int(match.group(1)),count),count)
                if len(logs)>50:
                    logs.pop(0)
            code=process.wait()
            process.stdout.close()
        except OSError as exc:
            raise RuntimeError(f'Не удалось запустить Primitive: {exc}') from exc
        if code!=0 or not result.is_file():
            raise RuntimeError('Primitive завершился с ошибкой. Последние сообщения: '+' | '.join(logs[-8:]))
        raw=result.read_bytes()
    return parse_primitive_svg(raw,style=style),raw
