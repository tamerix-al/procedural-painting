# Procedural Painting

Локальное приложение на Python и Streamlit: превращает загруженную картину в анимацию её алгоритмического рисования. Оформление ролика и экспорт в MP4 настраиваются независимо от расчёта изображения.

## Возможности

- **Painterly:** Oil Strokes и Dry Brush — мазки с фактурой щетины.
- **Primitive (Go):** треугольники, круги, эллипсы, прямоугольники, кривые и смешанные фигуры. Внешний оптимизатор устанавливается отдельно.
- **Graphic:** GRAIN (цветные точки) и экспериментальный TRACE (растущая линия). TRACE пока заметно отличается от референса и находится в разработке.
- **Встроенный Python:** базовые геометрические техники и экспериментальная кисть.
- **Видео:** вертикальный MP4, выбор FPS, разрешения и сжатия; подписи, рамка, сведения о картине, ускорение последних мазков; переходы Soft Wipe, Split Reveal и Crossfade.
- **Повторный экспорт:** рассчитанные фигуры и настройки сохраняются в ZIP-проект. Можно менять оформление и продолжительность без повторного расчёта.

## Запуск на Windows в VS Code

Нужен установленный Python (рекомендуется 3.11 или 3.12). Открой папку проекта в VS Code, открой терминал PowerShell и выполни:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Или запусти `run_windows.bat`: он создаст `.venv`, установит зависимости и откроет приложение.

Для Linux/macOS:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

Обычно интерфейс доступен на `http://localhost:8501`.

### Режим Primitive

Только для Primitive нужен Go: https://go.dev/dl/. Установи Go, перезапусти терминал и запусти `setup_primitive_windows.bat` (Windows) или `setup_primitive_mac_linux.sh` (macOS/Linux). Скрипт устанавливает оригинальный [fogleman/primitive](https://github.com/fogleman/primitive) в `tools/`. Исполняемый файл не включён в репозиторий.

Для остальных режимов Go и Node.js не требуются.

## Использование

1. Загрузи JPG или PNG либо включи тестовую картину.
2. Выбери движок, технику и количество операций.
3. Рассчитай рисунок один раз.
4. Настрой подписи, размеры, FPS, длительность и переход к оригиналу.
5. Экспортируй MP4. Сохрани ZIP-проект, если хочешь вернуться к тому же расчёту позднее.

Для TRACE после изменения алгоритма необходимо рассчитывать картину заново. Старые ZIP-проекты остаются пригодными для экспорта ранее сохранённых операций.

## Структура

```text
app.py                         интерфейс Streamlit
painting/engine.py             базовые типы и операции
painting/primitive_backend.py  связь с Primitive
painting/oil_plan.py           подбор мазков
painting/oil_brush.py          фактура масляной кисти
painting/graphic.py            GRAIN и запуск TRACE
painting/trace_growth.py      экспериментальный TRACE
painting/video.py             MP4 и переходы
painting/layout.py            подписи и рамка
painting/project_io.py        сохранение ZIP-проектов
examples/                       небольшая тестовая картина
tests/                          автоматические проверки
```

## Проверка

```powershell
python -m unittest discover -s tests -v
```

## Исходный код и сторонние проекты

Собственная лицензия проекта пока не выбрана. Права и условия использования сторонних компонентов приведены в [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), а текст Apache-2.0 находится в `THIRD_PARTY_LICENSES/`.

- [Primitive](https://github.com/fogleman/primitive) — внешний Go-оптимизатор, MIT.
- [anidoodle](https://github.com/alexgreensh/anidoodle) — адаптированные подходы к рендерингу масляной кисти, Apache-2.0.

Сгенерированные видеоролики, личные картины, бинарный `primitive.exe` и виртуальное окружение не должны попадать в Git-репозиторий.
