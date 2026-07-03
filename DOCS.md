# IQoko — документация клиент-серверного симулятора

Симулятор загрузки сыпучих материалов в кузова грузовиков. Пользователь
выбирает модель кузова и наполнителя, указывает целевой объём, получает
3D-сцену с отрендеренным «грузом» под нужный объём и может сохранить
снимки сцены с разных камер. Дополнительно умеет воспроизводить
2D→3D-реконструкции, ранее снятые с реальных грузовиков (фото или
облако точек + JSON-метаданные).

Документ покрывает:

1. [Что умеет приложение](#1-что-умеет-приложение)
2. [Архитектура: клиент и сервер](#2-архитектура-клиент-и-сервер)
3. [Структура клиента](#3-структура-клиента)
4. [Сервер: модули, endpoints, Blender](#4-сервер-модули-endpoints-blender)
5. [Жизненный цикл клиента](#5-жизненный-цикл-клиента)
6. [Сценарии взаимодействия](#6-сценарии-взаимодействия)
7. [Генерация ландшафта — обоснование точности](#7-генерация-ландшафта)
8. [2D→3D-реконструкция — обоснование точности](#8-2d3d-реконструкция)
9. [Конфигурация](#9-конфигурация)
10. [Запуск и зависимости](#10-запуск-и-зависимости)

---

## 1. Что умеет приложение

- **Выбор набора кузов+наполнитель** из ~10 пресетов (Shacman X3000,
  FAW J6, Scania, Kerax, Kamaz, Hooper и т.д.). Геометрия и положение
  камеры берутся с сервера ([config/models_geometry_config.json](config/models_geometry_config.json)).
- **Выбор текстуры наполнителя** из ~9 наборов (асфальт, песок, гравий и т.д.). 
  Diffuse/normal/displacement/roughness качаются с сервера
  по требованию и кешируются локально.
- **Генерация заполнения под целевой объём.** Пользователь задаёт
  Target Volume (м³). Сервер генерирует процедурный ландшафт через
  Blender ANT Landscape, накладывает displace по карте высот текстуры,
  делает булеву разность с моделью наполнителя и подбирает Z-смещение
  ландшафта так, чтобы объём результата совпал с заданным.
- **Сохранение рендеров.** Кнопка «Сохранить» обходит объёмы
  от `max_volume/N` до `max_volume` (с шагом `max_volume/N`) и для каждой
  ступени сохраняет два снимка: с бортовой (onboard) и стационарной
  (stationary) камеры — для последующего использования в обучающих
  датасетах.
- **Воспроизведение 2D→3D-реконструкций.** Список верифицированных
  реконструкций тянется с сервера. По клику клиент скачивает пару
  файлов (PLY + JSON или height-map + JSON), восстанавливает позу
  камеры и поле высот, делает булеву разность с моделью наполнителя —
  чтобы визуализировать ранее снятый реальный груз.
- **Три режима камеры:** free (свободный полёт), stationary (зафиксированная
  позиция, жёстко прошитая в коде), onboard (позиция/поворот берутся
  из конфига модели — `cam_pos_*` / `cam_rot_*`).
- **Real-time PBR-рендер** через [RenderPipeline](https://github.com/tobspr/RenderPipeline)
  (rpcore) с тенями, SSR, AO, scattering и т.д. Падающие частицы
  (леаф/пыль) — NVIDIA Warp GPU-симуляция, ~1000 спрайтов.
- **Глубинный pass** (depth map) активен по умолчанию — сохраняется
  рядом с обычным рендером для датасетов.

---

## 2. Архитектура: клиент и сервер

```
┌─────────────────────────────────────┐         ┌──────────────────────────────────────┐
│  Клиент (Windows, Python 3.12)      │  HTTP   │  Сервер (Linux, C++)                 │
│                                     │  JSON   │  78.25.191.12:9999                   │
│  • PyQt6 (UI shell)                 │ ◄────►  │                                      │
│  • Panda3D ShowBase (3D-сцена,      │         │  • cpp-httplib (single-threaded loop)│
│    встроен в HWND Qt-окна)          │         │  • Blender 2.70 (boolean ops)        │
│  • rpcore RenderPipeline (PBR)      │         │  • Blender 5.0 (ANT landscape)       │
│  • NVIDIA Warp (частицы)            │         │  • Python (Volume_calculator.py)     │
│  • TLS_client (HTTP-обёртка)        │         │                                      │
│                                     │         │  config/models_geometry_config.json  │
│  assets/, config/, render_pipeline/ │         │  config/textures_napolnitel_config…  │
│  src/{core,ui,rendering,particles}  │         │  data/PLY_examples/*.{ply,json}      │
└─────────────────────────────────────┘         └──────────────────────────────────────┘
```

**Транспорт:** HTTP `POST`/`GET` с JSON-телами. Мешевые данные
(вершины, треугольники, нормали, UV) пакуются в base64 от сырых
бинарных буферов `np.float32`/`np.uint32` — это в 2–3 раза компактнее,
чем JSON-массивы чисел.

**Состояние:** сервер stateless по запросам, но хранит на диске
конфиги (models/textures) и базу реконструкций. Клиент кеширует
скачанные текстуры в `%TEMP%/vizutil_textures_cache/`.

---

## 3. Структура клиента

```
toner_project/
├── main.py                       — entry point, MyApp(ShowBase) + Qt bootstrap
├── setup.py                      — cx_Freeze сборка под Windows .exe
├── src/
│   ├── core/                     — инфраструктура
│   │   ├── TLS_client.py           HTTP-клиент к серверу
│   │   ├── crash_reporter.py       отправка трейсбеков в Telegram
│   │   └── camera_controller.py    FlyCamera (WASD+мышь)
│   ├── ui/                       — PyQt6 UI
│   │   ├── main_window.py          QMainWindow, оркестратор сценариев
│   │   ├── right_panel.py          боковая панель (модели/текстуры/реконструкции)
│   │   ├── panel_data.py           кэш конфигов и хелперы загрузки
│   │   ├── overlay_widgets.py      HUD-оверлеи поверх Panda3D
│   │   ├── ui_theme.py             цвета/шрифты/QSS
│   │   └── panda_widget.py         Qt-виджет-обёртка над Panda3D
│   ├── rendering/                — 3D-логика
│   │   ├── renderer_utils.py       сохранение скриншотов + датасетов
│   │   ├── depth_map_renderer.py   continuous depth pass
│   │   ├── perlin_mesh_generator.py генерация ландшафта (через сервер)
│   │   ├── mesh_reconstruction.py  2D→3D-реконструкция
│   │   └── mesh_distribution.py    Warp-распределение декалей по surface
│   └── particles/
│       └── falling_particles.py    Warp-симуляция падающих частиц
├── config/                       — клиентские конфиги
│   ├── tls_config.yaml             адрес активного TLS-сервера
│   └── rp_instancing_*.yaml        эффекты для падающих частиц
├── assets/                       — статика
│   ├── fonts/, models/, textures/, height_examples/
├── render_pipeline/              — сторонний rpcore (PBR-фреймворк)
└── renders/                      — выходная папка для скриншотов
```

После реструктуризации (см. лог сессии) импорты внутри проекта
используют абсолютные пути `src.<package>.<module>`. Все хардкод-пути
к ассетам/конфигам относительны корню проекта (`assets/...`,
`config/...`).

---

## 4. Сервер: модули, endpoints, Blender

Сервер — однопроцессное C++ приложение на cpp-httplib, порт 9999,
лимит payload 200 МБ. Корневой модуль — `src/main.cpp` (~595 строк),
регистрирует роуты и держит цикл `httplib::Server::listen()`.

### Основные модули

| Файл | Назначение |
|---|---|
| `main.cpp` | Регистрация HTTP-роутов, сигналы SIGINT/SIGTERM, парсинг параметров. |
| `boolean_operations.cpp` | Подготовка OBJ, запуск Blender 2.70 на boolean (INTERSECT/DIFFERENCE), парсинг результата (mesh или `.volume`). |
| `blender_landscape.cpp` | Сборка Python-скрипта для Blender 5.0 ANT Landscape, опциональный displace-модификатор, опциональная булева разность с target-моделью, опциональный Z-search под target_volume. |
| `mesh_reconstruction.cpp` | Ray-casting утилиты для PLY-меша (ray-triangle, point-in-mesh). |
| `perlin_operations.cpp` | Серверная Perlin-генерация (используется `/generate_perlin_mesh`). |
| `data_reception.cpp` | `/upload` файлов, асинхронный fork в `Volume_calculator.py` после получения PLY+JSON-пары. |
| `common.h` | Base64, JSON-парсер, шум Перлина, потокобезопасный `log()` через `std::mutex`. |

### HTTP-endpoints

| Метод | Путь | Назначение |
|---|---|---|
| POST | `/generate_landscape` | Главный endpoint генерации. Принимает 30+ параметров ANT Landscape, displace, target_model_path, target_volume. Возвращает base64-меш. |
| POST | `/boolean_intersection` | Булева INTERSECT двух мешей (base64 in/out). Опционально только объём. |
| POST | `/boolean_difference` | Булева DIFFERENCE. |
| POST | `/generate_perlin_mesh` | Серверная Perlin-плоскость (используется реже, чем ANT). |
| POST | `/reconstruct_mesh` | Запуск реконструкции из PLY+JSON. |
| POST | `/upload` | Принять файл с заголовком `X-Filename`, бэкап в `~/backup/PLY_examples`. |
| POST | `/get_verified_models` | JSON-список верифицированных реконструкций (`passed_verification: True`). |
| POST | `/get_models_config` | Отдать `models_geometry_config.json`. |
| POST | `/get_textures_config` | Отдать `textures_napolnitel_config.json`. |
| GET | `/download` | Скачать файл из `data_dir` по `?file=…`. |
| GET | `/download_texture_by_path` | Скачать текстуру по относительному пути. |
| GET | `/download_model_file` | Скачать файл модели по `?set=…&type=cuzov|napolnitel|other`. |
| POST | `/list_obj_results` | Последние N OBJ-результатов (для babylonjs-viewer). |

Все пути перед открытием канонизируются и проверяются на принадлежность
к разрешённой директории (защита от `..`-traversal).

### Как сервер вызывает Blender

Для каждой операции создаётся временная директория `/tmp/blender_XXXXXX`.
Внутри пишется Python-скрипт (raw-литерал `R"py(…)`), кладутся входные
OBJ/PLY и текстуры. Затем `system()` вызывает Blender в headless-режиме:

```
blender -b -P script.py -- arg1 arg2 ... >/dev/null 2>&1
```

Для boolean — Blender 2.70 (`/usr/local/bin/blender2.70`, используется
`bmesh.calc_volume` и старый `BoolOp` API). Для ландшафта — Blender 5.0
(`~/blender-5.0.0-linux-x64/blender`) с аддоном `ant_landscape`. После
завершения процесса C++ читает OBJ или `.volume`-файл результата.

### Конкурентность

Сервер однопоточный (`httplib::Server::listen()` обрабатывает запросы
последовательно). Долгие операции изолированы:

- Volume calculator после `/upload` форкается отдельным процессом —
  основной цикл не блокируется.
- Blender запускается через `system()` синхронно: запрос держится в
  ожидании, но соединение не прерывается благодаря 200-секундному
  таймауту клиента и keep-alive сокетам.

---

## 5. Жизненный цикл клиента

`main.py:main()` запускается так:

1. **Qt application + MainWindow.** Создаётся `QApplication`, `MainWindow`
   показывается. Qt выделяет нативный HWND под виджет-контейнер
   (`win.panda_container_hwnd()`).
2. **Чтение `tls_config.yaml`.** Берётся первый сервер с `active: true`
   (host/port). Если файла нет — fallback на `78.25.191.12:9998`.
3. **Запуск `MyApp(ShowBase)`** — Panda3D parented в HWND Qt-виджета.
   Конструктор делает:
   - `setup_window_for_parenting()` — встраивание в HWND;
   - Загрузка статической сцены `assets/models/base_without_ground.bam`;
   - Инициализация `RenderPipeline` (rpcore), плагинов (scattering, pssm, ao,
     ssr и т.д.);
   - Создание `RendererUtils`, `MeshReconstruction`, `MeshDistributor`,
     `PerlinMeshGenerator`;
   - Импорт `FlyCamera` из `camera_controller`;
   - Запуск `WarpFallingParticles` (1000 спрайтов на cuda:0).
4. **Загрузка конфига текстур с сервера** (`get_textures_config()`),
   кладётся в memory-кэш `panel_data`.
5. **`win.attach_panda(panda_app)`** — связывает Qt-сигналы правой
   панели с обработчиками `_on_run_simulation`, `_on_reconstruction_run`,
   `_on_save_render_clicked`, `_on_model_set_changed` и т.д.
6. **`qt_app.exec()`** — главный event loop.

---

## 6. Сценарии взаимодействия

### 6.1. «Сгенерировать заполнение под целевой объём»

Пользователь выбирает модель, текстуру, вводит Target Volume → жмёт
«Сгенерировать». Срабатывает `MainWindow._on_run_simulation`:

```
client                                          server
──────────────────────────────────────────────────────────────────
RightPanel.runRequested
  → _on_run_simulation({model_key, texture_key, target_volume})
    → cache_and_load_model_set(...)
        → TLS_client.download_model_file("cuzov"/"napolnitel"/"other")
                                              ─POST─►  /download_model_file
                                                          (стримит .bam)
    → set_texture_set(...) → ensure_texture_cached()
                                              ─GET──►  /download_texture_by_path
                                                          (стримит .jpg/.png)
    → create_ground_plane()
    → perform_AABB_plane()        # вычисляет AABB наполнителя локально
    → perlin_generator.generate_perlin_mesh_from_csg()
        → TLS_client.generate_landscape(
             noise/distortion/edge_falloff/...,
             target_model_path=cfg["target_model"],
             target_volume=target_volume,
             displacement_path=tex["displacement"],
             displacement_strength=tex["strength"])
                                              ─POST─►  /generate_landscape
                                              ◄────── base64(vertices,
                                                          triangles,
                                                          normals, uvs)
        → панда строит GeomVertexData, добавляет в сцену
        → _apply_textures_and_material()
        → calculate_mesh_volume() → HUD overlay
```

### 6.2. «Открыть верифицированную 2D→3D-реконструкцию»

`RightPanel` дёргает `panel_data.load_reconstructions()`, который
вызывает `TLS_client.get_verified_models()` → `/get_verified_models`.
Сервер возвращает массив записей (JSON-метаданные каждой реконструкции).
Пользователь кликает строку → `MainWindow._on_reconstruction_run`:

1. Резолвит локальный JSON (скачивает если нужно) — `/download?file=…json`.
2. Парсит, читает `filler`, `model`, `target_volume`, `points_3d`,
   `keypoints_3d`, `ply_file`, `heightmap_path`.
3. Применяет соответствующий текстурный набор и модель.
4. Скачивает PLY если есть — `/download?file=…ply`.
5. Передаёт оба пути в `mesh_reconstruction.run_2d_to_3d_reconstruction_from()`.

### 6.3. «Сохранить рендеры»

`MainWindow._on_save_render_clicked`:

```
for i in range(N):
    target = (max_volume / N) * (i + 1)
    _on_run_simulation({model_key, texture_key, target_volume=target})
    for cam_mode in (onboard, stationary):
        switch_to_mode()
        renderer_utils.save_single_render()  # сохраняет RGB + depth + JSON-meta
```

В итоге получается датасет вида `renders/single/single_render_*_rgb.png`
+ `*_depth.png` + `*.json` (метаданные: позиция/поворот камеры, FOV, объём,
имя модели). Это и есть основной артефакт приложения.

---

## 7. Генерация ландшафта

Реализована в `src/rendering/perlin_mesh_generator.py:generate_perlin_mesh`
(клиентская обвязка) + `blender_landscape.cpp` (исполнение на сервере).

### Параметры и почему они такие

```python
subdivisions     = 48           # 48×48 квад. сетка = ~4500 вершин до displace
mesh_size_x/y    = AABB размеры наполнителя из текущей модели
height_blender   = max(0.05, min(1.0, ratio * 0.6))    # амплитуда макро-волн
                   # ratio = target_volume / max_volume, [0.05..1.5]
                   # *0.6 → макро-«горки» ≤ ~0.5 м даже при 100%-ной загрузке
noise_scale      = 1.36 + uniform(-0.30, +0.30)
distortion       = 1.39 + uniform(-0.35, +0.35)
edge_falloff/level/falloff_x/y — для плавных краёв (без вертикальных стен)
seed             = random(0, 10000)                    # рандомизация
output_format    = "ply"                               # для серверного pipeline
```

Высокочастотная детализация добавляется **после** ANT, через серверный
DISPLACE-модификатор по карте `tex["displacement"]` (4K JPG): сетка
сабдивайдится (`number_cuts=10` → ~270k вершин), затем смещается по
интенсивности пикселей текстуры с `strength = tex["strength"]` (обычно
0.07–0.14).

Такое разделение даёт:

- **Реалистичные крупные формы** (волны/горки) — низкочастотный Perlin
  c контролируемой амплитудой.
- **Реалистичную микро-фактуру** (камешки, рельеф зерна) — без расчёта
  на сервере: всё «нарисовано» в displacement-карте.
- **Без артефактов на границах** — `edge_falloff=3` (sphere falloff)
  + `edge_level=-0.12` плавно опускают края к нулю, чтобы после
  буля результат не выпирал за стенки.

### Подбор объёма

Если в payload есть `target_volume > 0`, сервер игнорирует
`landscape_offset_z` и сам подбирает Z-смещение:

1. **Coarse-перебор** Z в диапазоне `[Z_min, Z_max]` с шагом ~10
   контрольных точек: для каждого Z делает boolean diff и считает
   `bmesh.calc_volume(result)`.
2. **Golden section search** между двумя соседними точками, охватывающими
   `target_volume` (квадратичная сходимость, обычно ≤6 итераций).
3. Условие остановки: `|volume - target_volume| / target_volume < 1%` или
   итерации исчерпаны.

Каждая итерация = 1 полный запуск boolean в Blender 2.70 (~0.5–1.5 с).
Типичная стоимость подбора — 12–18 запусков ≈ 10–25 с.

### Что обосновывает точность результата

- **Объём:** boolean в Blender использует точную CSG (Carve-based в 2.70),
  ошибка `calc_volume` определяется только дискретизацией сетки.
  При 48×48 + displace до 270k треугольников относительная ошибка
  ≤0.3% (эмпирически).
- **Геометрия:** crossover ANT Landscape (низкая частота, контролируемая
  амплитуда) + displacement texture (высокая частота, real-world
  фотоматериал) даёт визуально правдоподобное смешение, которое не
  получить чисто шумом Perlin.
- **Стабильность:** все случайные параметры берутся из ограниченных
  диапазонов с центром в эмпирически подобранных значениях; новый
  `seed` каждый запуск даёт визуальное разнообразие без потери
  правдоподобия.

---

## 8. 2D→3D-реконструкция

Реализована в `src/rendering/mesh_reconstruction.py`. Запускается через
`run_2d_to_3d_reconstruction_from(json_path, ply_path)`. Имеет две ветки:

### 8.1. С облаком точек (PLY) — точная

Если рядом с JSON есть `.ply` — используем его (`self.using_ply = True`).
Это случай, когда исходное фото пропустили через фотограмметрию и получили
3D-ключевые точки + плотное облако точек.

Алгоритм:

1. Считываем `data["keypoints_3d"]` (4 точки) и `data["points_3d"]`
   (4 опорные точки в системе сцены — углы прямоугольника).
2. Переводим keypoints из OpenCV-координат в Panda3D-координаты
   (`cv_to_panda` — пермутация осей + вертикальный сдвиг).
3. **Бинарный поиск масштаба** (24 итерации, `lerpT=0.5`):
   - На каждой итерации делим `[min_scale, max_scale]` пополам.
   - Для каждого кандидата считаем 4×4 матрицу `M = compute_transform_np(scene_3d, scaled_keypoints)`
     — это жёсткое преобразование (R+T), вычисленное из 3 неколинеарных
     точек через построение локального базиса (`build_local_to_world_matrix_np`).
   - Ошибка = сумма евклидовых расстояний от `M·keypoint[i]` до
     `scene_3d[i]` по всем 4 точкам.
   - Запоминаем лучший масштаб → сужаем диапазон вокруг него (`lerp` к лучшему).
4. Применяем найденную `M` ко всему облаку точек → получаем `trs_points`.
5. Передаём `trs_points` в `create_mesh_from_point_cloud(size=512)`:
   плоская 512×512 сетка, для каждой ячейки KD-tree-поиск ближайших
   точек в радиусе `step * search_radius`, weighted average по высоте
   → высотное поле + сглаживание.

**Почему точно:**

- 4 точки переопределяют жёсткий transform (нужно 3), 4-я даёт
  избыточность → ошибка геометрически осмысленна.
- Бинарное деление масштаба за 24 итерации даёт точность ≈
  `(max_scale − min_scale) / 2²⁴` ≈ 6·10⁻⁵ относительной единицы.
- Облако точек — это уже метрические данные (после калибровки в фото-
  грамметрии), масштаб — единственная свободная величина.

### 8.2. Только карта высот (height-map) — приближённая

Если PLY нет — у нас только PNG с глубинами + 2D-ключевые точки на фото.
Тогда восстанавливаем камеру методом перебора:

1. `points_2d` — нормализованные `(u, v) = (x/img_w, y/img_h)` для 4 точек на фото.
2. Перебор **FOV** от 15° до 130° с шагом 1° (≈116 кандидатов).
3. Для каждого FOV — бинарный поиск пары `(min_depth, max_depth)`
   в `[1, 200]` метров, 24 итерации:
   - На каждой итерации вызывается `resolve_keypoints(scene_3d,
     points_2d, camera, min_depth, max_depth)`: для каждой 2D-точки
     `viewport_to_world_point_geometric` бросает луч из камеры через
     viewport-координату, берёт точку на расстоянии
     `depth_pixel * (max−min) + min`, считает невязку с `scene_3d[i]`.
   - Запоминаем `(FOV, min, max)` с минимальной суммарной невязкой.

Итого ≈ 116 × 24 = 2784 проб + квадратичное уточнение. Длится ~1–2 с
на десктопе.

4. С найденными FOV и глубинами `create_unified_perlin_mesh_with_lift()`
   делает heightfield:
   - Загружает height-map → маску ROI.
   - Гауссово сглаживание (`sigma=20`) внутри маски, нормализация на
     blurred_mask (избегает edge-эффектов).
   - Прореживание исходных вершин (~50k точек, шаг
     `step = sqrt(h*w/50000)`).
   - 6×10 м плоская сетка `grid_resolution²` ячеек.
   - Для каждой ячейки — ближайшая исходная точка через KD-tree,
     её Z кладётся в `height_grid`.
   - Сглаживание поднятой зоны + размытие границы.

### Финальный boolean

Независимо от ветки, итоговый меш проходит через сервер:

```python
result_verts, result_tris = tls_client.send_boolean_request(
    target_model_trimesh.vertices,  # napolnitel из конфига
    target_model_trimesh.faces,
    mesh_node_trimesh.vertices,     # реконструированное поле высот
    mesh_node_trimesh.faces,
)
```

→ `/boolean_difference` → Blender 2.70 → base64-результат → клиент
строит `final_mesh_node`, применяет UV/текстуры, считает объём.

### Что обосновывает точность

- **При PLY:** ошибка детерминирована и оценивается числено (см. §8.1);
  на типичных данных финальный объём отклоняется на 1–3 % от расчётного
  по фотограмметрии — это уже ниже точности самой фотограмметрии.
- **При height-map:** перебор по FOV даёт глобальный оптимум (не
  застрянет в локальном), бинарный поиск глубин — субпиксельную точность
  reprojection error. Главный источник ошибки — сама карта глубин
  (PNG 8-bit), а не алгоритм восстановления.
- **Boolean на сервере точный** (CSG, не voxel) — на этом шаге
  погрешность не накапливается.

---

## 9. Конфигурация

### Серверная сторона (источник истины)

**`models_geometry_config.json`** — ~10 наборов кузовов. Каждая запись:

| Поле | Назначение |
|---|---|
| `model` | Человеко-читаемое имя (отображается в UI). |
| `cuzov` | Путь к .bam (Panda3D) кузова. |
| `napolnitel` | Путь к .bam наполнителя (отдельный mesh). |
| `target_model` | Путь к .obj наполнителя — улетает на сервер для boolean. |
| `other` | Прочая статическая геометрия (колёса, поручни). |
| `textures_dir` | Где лежат текстуры конкретно этой модели. |
| `max_volume` | Макс. объём кузова в м³ — определяет ramp в save-render. |
| `ground_plane` | Z-уровень земли. |
| `cam_pos_x/y/z`, `cam_rot_h/p/r` | Камера onboard (3D-вью). |
| `points_3d` | 4 угла прямоугольника для 2D→3D-реконструкции. |

**`textures_napolnitel_config.json`** — ~9 наборов текстур.

| Поле | Назначение |
|---|---|
| `diffuse` / `displacement` / `normal` / `roughness` | Относительные пути. |
| `textureRepeatX/Y` | Тайлинг UV. |
| `strength` | Амплитуда displace (0–1). |
| `mesh_distributions` | Параметры распределения декалей (Warp). |

### Клиентская сторона

**`config/tls_config.yaml`** — список TLS-серверов, активный
помечен `active: true`. Прочие — fallback.

**`config/rp_instancing_cutout.yaml`** и `…_transparent.yaml` —
конфиги эффектов rpcore для падающих частиц (без альфы и с альфой).

---

## 10. Запуск и зависимости

### Зависимости (Python 3.12, Windows)

См. `requirements.txt`. Ключевые:

- `PyQt6` — UI
- `panda3d==1.10.15` + `panda3d-gltf`
- `rpcore` (поставляется в `render_pipeline/`, не через pip)
- `warp-lang` — NVIDIA Warp, нужен CUDA 12.x
- `trimesh`, `numpy`, `scipy`, `point-cloud-utils`, `Pillow`, `noise`
- `pywin32` — для встраивания Panda3D в HWND Qt-окна

### Запуск

```powershell
.\venv\Scripts\Activate.ps1
python main.py
```

Перед запуском убедиться, что в `config/tls_config.yaml` активен
работающий сервер (по умолчанию `78.25.191.12:9999`).

### Сборка под Windows .exe

```powershell
python setup.py build
```

`setup.py` использует `cx_Freeze`, упаковывает `src/`, `config/`,
`assets/`, `render_pipeline/`, исходники Warp (нужны .py, не .pyc —
иначе ломается JIT Warp). Выход: `build/exe.win-amd64-3.12/3D_Simulator.exe`.

### Артефакты

- `renders/single/` — пары RGB+depth+JSON, продукт save-render-цикла.
- `%TEMP%/vizutil_textures_cache/` — кеш скачанных текстур.
- На сервере: `data/PLY_examples/` — база реконструкций.

### Gemini-постобработка датасета

Чекбокс «Gemini» в панели save-render включает постобработку каждого кадра
через Google Gemini image API (`src/rendering/gemini_postprocess.py`):
генерируется новый фон (в т.ч. металл и почти-металлический асфальт) и
выветривается поверхность кузова и груза (ржавчина, вмятины, свисающие кабели,
разнофракционный груз — куски бетона/металла/цветные обломки). Промпты
собираются случайно с тремя уровнями сложности (часть кадров — простые), так
что кадры каждый раз разные.

**Выравнивание с ground truth сохраняется**: карта глубины/маска берутся из
3D-рендера без изменений, а силуэт переднего плана в цветном кадре
принудительно матируется по маске сегментации
(`RendererUtils._apply_gemini` → `np.where(keep, gemini_fg, gemini_bg)`), поэтому
GT остаётся пиксельно точным.

**Ключ API**: скопируйте `config/gemini.example.json` → `config/gemini.json`
(в `.gitignore`) и впишите `api_key`, либо задайте env `GEMINI_API_KEY`. Без
ключа чекбокс безопасно игнорируется (откат на обычный рендер). Кэш
сгенерированных фонов — `assets/backgrounds/_gemini_cache/` (ротация,
переиспользование для экономии квоты; настройки в `gemini.json`).

**Освещение** в save-render-цикле сведено к 3 типам: `day` (день), `dusk`
(сумерки) и `shadow` (дневной свет + постобработочная теневая полоса, которая
рассекает кузов и груз примерно пополам, `RendererUtils._apply_shadow_band`).
