"""Procedural Painting 0.10 — growing-pen TRACE."""
from __future__ import annotations

from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import streamlit as st

from painting.engine import BUILTIN_STYLES, STYLES, build_painting
from painting.graphic import GRAPHIC_STYLES, build_graphic_painting
from painting.layout import LayoutOptions, preview_layout
from painting.primitive_backend import PRIMITIVE_MODES, locate_primitive, run_primitive
from painting.project_io import export_project, import_project
from painting.video import RESOLUTIONS, REVEAL_MODES, STROKE_TIMINGS, export_video

st.set_page_config(page_title='Procedural Painting 0.10',page_icon='🎨',layout='wide')
st.title('Procedural Painting')
st.caption('Версия 0.10: TRACE — растущая линия с выбором направления по изображению. Старые TRACE-режимы удалены из меню.')

# Read imported project only when its bytes change; otherwise preserve edits to UI.
with st.expander('Открыть ранее рассчитанный проект (.zip)',expanded=False):
    project_file=st.file_uploader('Проект Procedural Painting',type=['zip'],key='project_upload')
    if project_file is not None:
        project_bytes=project_file.getvalue()
        digest=sha256(project_bytes).hexdigest()
        if digest!=st.session_state.get('imported_project_id'):
            try:
                img_array,paint,settings,svg=import_project(project_bytes)
                st.session_state.imported_image=img_array
                st.session_state.cached_painting=paint
                st.session_state.cached_image_hash=sha256(img_array.tobytes()).hexdigest()
                st.session_state.cached_svg=svg
                st.session_state.model_origin='Сохранённый проект ZIP'
                st.session_state.imported_project_id=digest
                for name,value in settings.items():
                    if name in LayoutOptions.__dataclass_fields__:
                        st.session_state[f'layout_{name}']=value
                st.success('Проект открыт. Фигуры уже рассчитаны; можно сразу экспортировать MP4.')
            except Exception as exc:
                st.session_state.pop('imported_image',None)
                st.error(f'Не удалось открыть проект: {exc}')

with st.sidebar:
    st.header('Алгоритм рисования')
    engine=st.radio('Движок',['Primitive (Go)','Painterly (масляная кисть)','Graphic (точки и линии)','Встроенный Python'],horizontal=False)
    is_primitive=engine.startswith('Primitive')
    is_painterly=engine.startswith('Painterly')
    is_graphic=engine.startswith('Graphic')
    choices=(PRIMITIVE_MODES if is_primitive else
             GRAPHIC_STYLES if is_graphic else
             {k:BUILTIN_STYLES[k] for k in ('oil','dry_brush')} if is_painterly else
             {k:v for k,v in BUILTIN_STYLES.items() if k not in ('oil','dry_brush')})
    style=st.selectbox('Техника рисования',list(choices),format_func=lambda k:choices[k][0] if is_primitive else choices[k])
    binary_path=''
    available=True
    if is_primitive:
        auto_bin=locate_primitive()
        if auto_bin:
            st.success(f'Primitive найден: {auto_bin.name}')
        else:
            st.warning('Primitive пока не установлен. Запусти setup_primitive_windows.bat в папке проекта.')
        binary_path=st.text_input('Путь к primitive.exe (необязательно)',value='',placeholder='D:\\...\\primitive.exe')
        available=locate_primitive(binary_path) is not None
        profile=st.selectbox('Качество поиска',('Проверка','Стандарт','Детально','Очень детально'),index=1)
        presets={'Проверка':(70,256),'Стандарт':(180,360),'Детально':(350,512),'Очень детально':(650,640)}
        default_count,default_size=presets[profile]
        count=st.slider('Количество фигур',20,2500,default_count,10)
        analysis_size=st.select_slider('Разрешение анализа',options=[128,192,256,360,512,640,800],value=default_size)
        alpha=st.slider('Прозрачность фигуры (0 = автоматическая)',0,255,128,8)
        workers=st.slider('Параллельные потоки (0 = все)',0,32,0,1)
        st.caption('Primitive самостоятельно оптимизирует форму и цвет. Его итоговый SVG сохраняется вместе с проектом.')
    elif is_graphic:
        profile=st.selectbox('Качество графики',('Проверка','Хорошее','Высокое','Максимальное'),index=1)
        graphic_presets={
            'grain': {'Проверка':(1600,360),'Хорошее':(10000,520),'Высокое':(24000,700),'Максимальное':(48000,850)},
            'trace': {'Проверка':(1500,360),'Хорошее':(8000,520),'Высокое':(18000,700),'Максимальное':(35000,850)},
        }
        default_count,default_size=graphic_presets[style][profile]
        count=st.slider('Количество сегментов линии' if style=='trace' else 'Количество точек',500,60000,default_count,500)
        working_size=st.select_slider('Разрешение анализа',[260,360,520,700,850,1000,1200],value=default_size)
        graphic_mark_size=st.slider('Толщина линии' if style=='trace' else 'Размер точек',0.4,2.5,1.0,0.1)
        graphic_contrast=st.slider('Контраст рисунка',0.5,2.0,1.0,0.1)
        seed=st.number_input('Seed',0,999999,42)
        st.caption('TRACE строит растущие связные траектории. При необходимости перо переносится в новый участок. Старые ZIP содержат прежний рисунок; для нового TRACE пересчитай картину.')
    else:
        profile=st.selectbox('Качество поиска',('Черновик','Хорошее','Высокое','Максимальное'),index=1)
        presets={'Черновик':(500,16,260),'Хорошее':(1500,32,520),'Высокое':(3200,60,700),'Максимальное':(6500,96,1000)}
        if is_painterly:
            presets={'Черновик':(500,16,260),'Хорошее':(1800,32,520),'Высокое':(3600,60,700),'Максимальное':(6500,80,1000)}
        default_count,candidates,working_size=presets[profile]
        count=st.slider('Количество фигур / мазков',200,12000,default_count,100)
        candidates=st.slider('Попыток на каждый мазок',8,160,candidates,4)
        working_size=st.select_slider('Разрешение анализа',[180,260,360,520,700,850,1000,1200],value=working_size)
        detail=st.slider('Мелкие детали',0.5,2.0,1.2,0.1)
        opacity=st.slider('Непрозрачность фигур',0.2,1.0,1.0,0.05)
        seed=st.number_input('Seed',0,999999,42)
        if is_painterly:
            default_dry=.24 if style=='oil' else .79
            default_imp=.53 if style=='oil' else .16
            oil_dryness=st.slider('Сухость кисти (больше = просветы)',0.0,1.0,default_dry,0.05)
            oil_impasto=st.slider('Рельеф масляной краски',0.0,1.0,default_imp,0.05)
            st.markdown('**Форма и расположение мазков**')
            oil_length_ratio=st.slider('Длина мазка относительно ширины',0.8,5.0,1.6,0.1,
                                         help='1.2–1.8 = короткие мазки как в референсе. 3–5 = длинные полосы.')
            oil_size=st.slider('Общая крупность мазков',0.4,2.0,0.85,0.05,
                               help='Меньшее значение даёт более мелкие мазки с самого начала.')
            oil_follow_flow=st.checkbox('Ориентировать по линиям картины',True,
                                        help='По сглаженным направлениям деталей и контуров.')
            st.caption('Изменение длины или крупности требует снова рассчитать рисунок. Старые ZIP открываются без пересчёта.')
            st.caption('Фактурные короткие мазки подбираются по изображению. Параметры длины и размера применяются при следующем расчёте.')
        else:
            oil_dryness=None
            oil_impasto=None
            oil_length_ratio=1.6
            oil_size=.85
            oil_follow_flow=True
            st.caption('Прежний встроенный оптимизатор для сравнения с Primitive.')
    st.divider()
    st.header('Видео')
    seconds=st.slider('Длительность рисования (сек)',5,90,24)
    fps=st.selectbox('FPS',[24,30,60],index=1)
    resolution_label=st.selectbox('Разрешение',list(RESOLUTIONS),index=2)
    crf=st.slider('Сжатие H.264 (меньше = качественнее)',14,25,17)
    st.markdown('**Анимация и финал**')
    stroke_timing=st.selectbox('Скорость появления мазков',list(STROKE_TIMINGS),index=1,
                               format_func=lambda key:STROKE_TIMINGS[key],
                               help='Умеренное ускорение делает последние мелкие мазки быстрее. Сам рисунок не пересчитывается.')
    show_original=st.checkbox('Показать оригинал в конце',True)
    if show_original:
        reveal_mode=st.selectbox('Переход к оригиналу',list(REVEAL_MODES),index=0,
                                 format_func=lambda key:REVEAL_MODES[key])
        reveal_seconds=st.slider('Длительность перехода (сек)',0.3,3.0,1.0,0.1)
        pre_reveal_hold_seconds=st.slider('Пауза перед переходом (сек)',0.0,2.0,0.5,0.1)
        original_hold_seconds=st.slider('Показ оригинала в конце (сек)',0.0,3.0,1.0,0.1)
    else:
        reveal_mode='soft_wipe'
        reveal_seconds=1.0
        pre_reveal_hold_seconds=0.0
        original_hold_seconds=0.0
    st.caption('Скорость и переход применяются при экспорте MP4, без повторного расчёта мазков.')

upload=st.file_uploader('Загрузить картину (JPG, PNG)',type=['png','jpg','jpeg'],key='original_upload')
sample_path=Path(__file__).parent/'examples'/'sample_landscape.png'
image=None
try:
    if upload is not None:
        image=ImageOps.exif_transpose(Image.open(BytesIO(upload.getvalue()))).convert('RGB')
    elif project_file is not None and 'imported_image' in st.session_state:
        image=Image.fromarray(st.session_state.imported_image)
    else:
        sample=st.checkbox('Использовать тестовую картину',value=True)
        if sample:
            image=Image.open(sample_path).convert('RGB')
    if image is not None:
        image.thumbnail((4000,4000))
except (UnidentifiedImageError,OSError,ValueError) as exc:
    st.error(f'Не удалось открыть изображение: {exc}')

original=np.asarray(image).copy() if image is not None else None
image_hash=sha256(original.tobytes()).hexdigest() if original is not None else None
model=st.session_state.get('cached_painting') if image_hash==st.session_state.get('cached_image_hash') else None

st.subheader('Оформление видео')
t1,t2,t3=st.tabs(['Подписи','Рамка и шрифты','Предпросмотр'])
with t1:
    a,b=st.columns(2)
    with a:
        channel=st.text_input('Сверху слева — название канала',value='MY ART CHANNEL',key='layout_channel')
        painting_title=st.text_input('Под картиной — название картины',value='Название картины',key='layout_painting_title')
        artist=st.text_input('Имя художника',value='Имя художника',key='layout_artist')
    with b:
        technique=st.text_input('Сверху справа — техника (пусто = автоматически)',value='',key='layout_technique')
        year=st.text_input('Год картины',value='',key='layout_year')
        font_family=st.selectbox('Шрифт названия картины',['sans','serif'],format_func=lambda k:'Без засечек' if k=='sans' else 'С засечками',key='layout_font_family')
with t2:
    c1,c2=st.columns(2)
    with c1:
        background=st.color_picker('Цвет фона','#F8F7F4',key='layout_background')
        foreground=st.color_picker('Основной текст','#292827',key='layout_foreground')
        muted=st.color_picker('Второстепенный текст / рамка','#7B7874',key='layout_muted')
        accent=st.color_picker('Цвет прогресса','#33312E',key='layout_accent')
        margin=st.slider('Боковые отступы',30,160,66,2,key='layout_margin')
        header_y=st.slider('Отступ верхней строки',30,200,75,5,key='layout_header_y')
        image_width=st.slider('Ширина картины (%)',60,100,100,5,key='layout_image_width')
        image_offset_y=st.slider('Сдвиг картины вверх / вниз',-250,250,0,10,key='layout_image_offset_y')
    with c2:
        caption_gap=st.slider('Промежуток под картиной',10,130,50,5,key='layout_caption_gap')
        header_font_size=st.slider('Шрифт верхней строки',18,55,28,key='layout_header_font_size')
        title_font_size=st.slider('Размер названия картины',26,85,52,key='layout_title_font_size')
        meta_font_size=st.slider('Размер автора и года',18,60,30,key='layout_meta_font_size')
        image_border=st.checkbox('Тонкая рамка картины',True,key='layout_image_border')
        show_counter=st.checkbox('Счётчик фигур',True,key='layout_show_counter')
        show_progress=st.checkbox('Полоса прогресса',True,key='layout_show_progress')

layout=LayoutOptions(channel=channel,technique=technique,painting_title=painting_title,
                     artist=artist,year=year,background=background,foreground=foreground,
                     muted=muted,accent=accent,margin=margin,header_y=header_y,
                     image_width=image_width,image_offset_y=image_offset_y,
                     caption_gap=caption_gap,header_font_size=header_font_size,
                     title_font_size=title_font_size,meta_font_size=meta_font_size,
                     font_family=font_family,image_border=image_border,
                     show_counter=show_counter,show_progress=show_progress)
with t3:
    if original is not None:
        p1,p2=st.columns(2)
        with p1:
            st.image(original,caption='Исходная картина',use_container_width=True)
        with p2:
            preview=preview_layout(original,model.style if model else style,layout,(540,960))
            st.image(preview,caption='Предпросмотр оформления',use_container_width=True)
    else:
        st.info('Загрузите картину для предпросмотра.')

st.download_button('Скачать настройки рамки (JSON)',json.dumps(layout.to_dict(),ensure_ascii=False,indent=2),
                   file_name='painting-layout.json',mime='application/json')

st.divider()
st.subheader('Создание видео')
st.caption('1. Рассчитай фигуры один раз. 2. Меняй подписи, рамку, FPS, скорость мазков и переход, затем экспортируй MP4 без пересчёта.')
c1,c2=st.columns(2)
with c1:
    calculate=st.button('1. Рассчитать новый рисунок',type='primary',disabled=(image is None or not available),use_container_width=True)
with c2:
    # Button appears immediately after calculating thanks to Streamlit reruns.
    export=st.button('2. Экспортировать MP4',disabled=model is None,use_container_width=True)

if calculate:
    st.session_state.pop('result_video',None)
    bar=st.progress(0,text='Запускаю оптимизацию...')
    try:
        def progress(done,total):
            bar.progress(min(1,done/max(1,total)),text=f'Оптимизация: {done}/{total}')
        if is_primitive:
            paint,raw_svg=run_primitive(original,style,int(count),int(analysis_size),int(alpha),int(workers),
                                        executable=binary_path or None,progress=progress)
        elif is_graphic:
            paint=build_graphic_painting(original,style=style,marks=int(count),
                                        working_size=int(working_size),seed=int(seed),
                                        mark_size=float(graphic_mark_size),contrast=float(graphic_contrast),
                                        progress=progress)
            raw_svg=None
        else:
            paint=build_painting(original,style=style,strokes=count,candidates=candidates,
                                 working_size=working_size,seed=int(seed),detail=detail,
                                 opacity=opacity,progress=progress,
                                 oil_dryness=oil_dryness,oil_impasto=oil_impasto,
                                 oil_length_ratio=oil_length_ratio,oil_size=oil_size,
                                 oil_follow_flow=oil_follow_flow)
            raw_svg=None
        st.session_state.cached_painting=paint
        st.session_state.cached_image_hash=image_hash
        st.session_state.cached_svg=raw_svg
        st.session_state.model_origin=engine
        st.success(f'Рисунок рассчитан: {len(paint.operations)} фигур. Теперь нажми «Экспортировать MP4».')
        st.rerun()
    except Exception as exc:
        st.error(f'Ошибка оптимизации: {exc}')

# Read cache again after sidebar widgets, to support upload and successive exports.
model=st.session_state.get('cached_painting') if image_hash==st.session_state.get('cached_image_hash') else None
if model is not None:
    st.info(f'Готовый рисунок: {len(model.operations)} операций, {STYLES.get(model.style,model.style)}. Источник: {st.session_state.get("model_origin","кэш")}. Изменение рамки не пересчитывает фигуры.')
    if st.session_state.get('cached_svg'):
        st.download_button('Скачать исходный SVG Primitive',st.session_state.cached_svg,
                           file_name='primitive-result.svg',mime='image/svg+xml')
    project_archive=export_project(original,model,layout,st.session_state.get('cached_svg'))
    st.download_button('Скачать проект (рисунок + оригинал + настройки) ZIP',project_archive,
                       file_name='painting-project.zip',mime='application/zip')

if export and model is not None:
    st.session_state.pop('result_video',None)
    bar=st.progress(0,text='Собираю видео...')
    try:
        with tempfile.TemporaryDirectory(prefix='painting_video_') as tmp:
            video_path=Path(tmp)/'video.mp4'
            def video_progress(done,total):
                bar.progress(min(1,done/max(1,total)),text=f'Кадры: {done}/{total}')
            export_video(model,original,video_path,seconds=seconds,fps=fps,
                         output_size=RESOLUTIONS[resolution_label],layout=layout,
                         crf=crf,show_original=show_original,progress=video_progress,
                         stroke_timing=stroke_timing,reveal_mode=reveal_mode,
                         reveal_seconds=reveal_seconds,
                         pre_reveal_hold_seconds=pre_reveal_hold_seconds,
                         original_hold_seconds=original_hold_seconds)
            st.session_state.result_video=video_path.read_bytes()
            st.session_state.result_name=f'painting_{model.style}.mp4'
        st.success('MP4 готов. Чтобы изменить оформление, поменяй настройки и экспортируй заново.')
    except Exception as exc:
        st.error(f'Ошибка создания MP4: {exc}')
        st.exception(exc)
if st.session_state.get('result_video'):
    st.video(st.session_state.result_video)
    st.download_button('Скачать готовое MP4',st.session_state.result_video,
                       file_name=st.session_state.get('result_name','painting.mp4'),
                       mime='video/mp4',type='primary')

st.caption('Primitive устанавливается отдельно. Oil Strokes использует Python-адаптацию механики anidoodle (Apache-2.0). GRAIN и TRACE — самостоятельные алгоритмы; подробнее в THIRD_PARTY_NOTICES.md.')
