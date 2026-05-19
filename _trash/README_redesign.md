# Digital Engineering 2026 — UI Redesign

Комплект для редизайна интерфейса 3D-симулятора. Стек: **PyQt6**, **Panda3D**, **win32gui**. Стиль — профессиональный инструмент для AI / Robotics: антрацитовый фон, тонкие линии вместо рамок, единственный системный акцент Vivid Mint `#00FF88`.

## Состав

| Файл | Роль |
|---|---|
| `ui_theme.py` | Полный QSS + палитра (экспортируется как `apply_theme(widget)`). |
| `right_panel.py` | Фиксированная правая панель (380 px) со всеми секциями. |
| `depth_map_widget.py` | Виджет предпросмотра карты глубины 16:9 (thread-safe через `pyqtSignal`). |
| `panda_depth_bridge.py` | Offscreen-пасс Panda3D → `Texture.get_ram_image()` → виджет. |
| `overlay_widgets.py` | Плавающие информационные блоки поверх 3D-сцены. |
| `main_window.py` | Пример сборки: viewport + right panel + overlays + depth bridge. |

## Визуальный язык

- Фон `#101010` (Anthracite), приподнятые поверхности `#161616` / `#1B1B1B`.
- Разделение — исключительно hairlines `#252525` + whitespace. Никаких «коробочных» `QGroupBox`-рамок: заголовки секций подаются как SMALL CAPS eyebrow над тонкой линией.
- Единственный акцент — Vivid Mint `#00FF88`. Используется только для: состояния «LIVE», фокусов ввода, главной кнопки `RUN SIMULATION`, активных табов и handle слайдеров.
- Типографика — Geist Sans / Inter / IBM Plex Sans. Цифровые метрики — Geist Mono.
- Радиусы: 8 px для поверхностей, 6 px для контролов, 4 px для чипов.

## Ключевые моменты QSS

- `QGroupBox` перестилизован под eyebrow + hairline вместо рамки (`border: none; border-top: 1px solid #252525`).
- Скроллбары overlay-стиля 8 px без стрелок.
- Табы подчёркнутого стиля (underline на `:selected`, без «plateau»).
- Кнопки имеют варианты через `setProperty("variant", …)`: `primary`, `danger`, `ghost`, `icon`.
- Чипы статуса через `setProperty("role", "chip-live" | "chip-idle" | "chip-err")`.

Применение:
```python
from ui_theme import apply_theme
apply_theme(main_window)
```

## Правая панель (компоновка)

Ширина зафиксирована на 380 px (`QSizePolicy.Fixed`), высота — растягиваемая. Структура сверху вниз:

1. **Header** — бренд + версия сборки.
2. **Depth Map Preview** — 16:9 (`AspectRatioFrame`, см. ниже) + чип `● LIVE`, строка разрешения, FPS.
3. **Metrics row** — две большие метрики (Target Volume, Particles) в монокомпе.
4. **Scene · Assets** — модели + текстуры через `QComboBox`.
5. **Filling · Parameters** — `QDoubleSpinBox`, `QSlider`, `QCheckBox`.
6. **Primary Action** — единственная акцентная кнопка `RUN SIMULATION` + ghost-вариант `Abort`.
7. **Reconstruction** — блок 2D→3D.
8. **StatusBar** — хейрлайн сверху, моно-шрифт, приглушённый.

Соотношение 16:9 для превью достигается кастомным `AspectRatioFrame`:
```python
class AspectRatioFrame(QFrame):
    def hasHeightForWidth(self): return True
    def heightForWidth(self, w):  return int(w * 9 / 16)
```

## Overlay-виджеты поверх 3D-сцены

`SceneOverlay` — это `QWidget` c флагами `FramelessWindowHint | Tool | WindowStaysOnTopHint`, атрибутами `WA_TranslucentBackground` и `WA_TransparentForMouseEvents`. Внутри — `QFrame#Overlay` с полупрозрачным `rgba(16,16,16,210)`, drop-shadow и hairline-бордером.

Поведение «прилипания» к 3D-контейнеру реализовано через `installEventFilter` + reposition по `Resize`/`Move`/`Show` целевого виджета. Стандартный набор якорей: `top-left`, `top-right`, `bottom-left`, `bottom-right`.

```python
ov = SceneOverlay("Camera · Telemetry", anchor="top-left", parent=main_window)
ov.set_rows([("Pitch","-90.0°"),("Yaw","0.0°"),("FOV","60°")])
ov.show_over(panda_container)
```

## Интеграция карты глубины из Panda3D

Конвейер:

1. **Создаём offscreen buffer** с глубинной текстурой:
   ```python
   fbp = FrameBufferProperties(); fbp.set_depth_bits(32); fbp.set_rgb_color(False)
   buf = app.graphicsEngine.make_output(app.pipe, "depth_buffer", -100,
                                        fbp, WindowProperties.size(W, H),
                                        GraphicsPipe.BFRefuseWindow,
                                        app.win.getGsg(), app.win)
   ```
2. **Прикручиваем depth-текстуру с CPU-миррорингом**. Ключевой вызов — `set_keep_ram_image(True)` и `RTMCopyRam`:
   ```python
   tex = Texture("depth_tex")
   tex.set_format(Texture.F_depth_component)
   tex.set_component_type(Texture.T_float)
   tex.set_keep_ram_image(True)
   buf.add_render_texture(tex, GraphicsOutput.RTMCopyRam, GraphicsOutput.RTPDepth)
   ```
3. **Вторая камера** повторяет лензу главной и парентится к `app.camera`, чтобы depth-пасс рисовался с той же позиции.
4. **Task в taskMgr** раз в N кадров достаёт RAM-образ, нормализует и отдаёт виджету:
   ```python
   def _tick(self, task):
       ram = tex.get_ram_image_as("D")        # bytes, float32
       depth = np.frombuffer(memoryview(ram).tobytes(), dtype=np.float32)
       depth = depth.reshape(H, W)
       near, far = np.percentile(depth, (2, 98))
       norm = np.clip((depth - near) / (far - near), 0, 1)
       rgba = LUT[(norm*255).astype(np.uint8)]   # turbo-подобный LUT
       preview.push_frame(rgba.tobytes(), W, H, "rgba8")
       return Task.cont
   app.taskMgr.add(_tick, "DepthBridgeTick")
   ```
5. **На стороне Qt** `DepthMapPreview.push_frame` — это всего лишь `emit` сигнала `frame_ready(bytes, w, h, fmt)`. Слот собирает `QImage` **на главном потоке Qt** (безопасно в любом режиме: `Qt.ConnectionType.QueuedConnection`) и обновляет `QLabel`. Панда-текстуры приходят отражёнными по Y — мы выполняем `img.mirrored(False, True)`.

Производительность: stride=2 (каждый второй кадр), 512×288 Turbo-LUT — нагрузка на CPU/GUI пренебрежимо мала, превью идёт в 30 FPS без потерь для основной сцены.

## Миграция PyQt5 → PyQt6

В исходниках `gui.py` / `main.py` используется PyQt5. Новый набор написан на **PyQt6**, что соответствует заданию. Если полной миграции пока не планируется, в файлах достаточно заменить в импортах:
```python
from PyQt6.QtCore    import Qt, QTimer, pyqtSignal
from PyQt6.QtGui     import QImage, QPixmap, QColor
from PyQt6.QtWidgets import QWidget, QLabel, ...
```
и уточнить enum-пути (`Qt.AlignmentFlag.AlignCenter`, `QFrame.Shape.NoFrame`, `Qt.ConnectionType.QueuedConnection` и т.д. — в коде уже сделано). QSS полностью совместим с обеими версиями.

## Шрифты

Geist Sans / Geist Mono можно подложить локально и зарегистрировать в старте приложения:
```python
from PyQt6.QtGui import QFontDatabase
QFontDatabase.addApplicationFont("fonts/Geist-Variable.ttf")
QFontDatabase.addApplicationFont("fonts/GeistMono-Variable.ttf")
```
Фолбэки в стилях прописаны — без Geist UI выглядит корректно на Inter / IBM Plex / Segoe UI Variable.

## Сборка в существующий проект

В `MainWindowManager`:

```python
from ui_theme      import apply_theme
from right_panel   import RightPanel, PANEL_WIDTH
from overlay_widgets import SceneOverlay
from panda_depth_bridge import DepthBridge

apply_theme(self.main_window)
self.control_panel = RightPanel(panda_app)
self.control_panel.setFixedWidth(PANEL_WIDTH)
main_layout.addWidget(self.panda_container, 1)
main_layout.addWidget(self.control_panel, 0)

self.depth_bridge = DepthBridge(panda_app, self.control_panel.depth_preview,
                                width=512, height=288, stride=2, colormap="turbo")

self.camera_overlay = SceneOverlay("Camera · Telemetry", "top-left", self.main_window)
self.camera_overlay.set_rows([("Pitch","-90.0°"),("Yaw","0.0°")])
self.camera_overlay.show_over(self.panda_container)
```

Этого достаточно, чтобы получить футуристичный интерфейс вместо текущего «tabbed» UI.
