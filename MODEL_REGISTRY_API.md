# Реестр моделей photo-to-volume — HTTP API и интеграция в Python-приложение

Документ для агента/разработчика, который встраивает загрузку и замену моделей
кузовов в Python-приложение.

---

## 1. Что это

`model-registry` — небольшой HTTP-демон (C++, собирается из этого же репозитория),
который умеет **добавлять, заменять, удалять и инспектировать модели грузовиков**:
файлы геометрии, текстуры, запись в `config/models_geometry_config.json` и пресет
камеры в `config/camera_presets.json` — одним запросом, атомарно.

Он **не** обрабатывает фотографии. Обработка — это отдельный бинарник
`photo-to-volume`, и трогать его не нужно (см. §2).

| | |
|---|---|
| Исходник | [src/model_registry.cpp](../src/model_registry.cpp), API — [include/model_registry.h](../include/model_registry.h) |
| Где работает | внутри `operation-3d-service` на порту **9999**, префикс `/registry` (см. §3.1) |
| Отдельный демон | `build/model-registry` — тот же код, режим локальной отладки |
| Формат загрузки | `multipart/form-data`, файлы пишутся на диск потоком |
| Зависимости | нет (vendored `httplib.h` + `json.hpp`) |

---

## 2. Главное про «сервер должен узнать об изменениях»

**Ничего перезапускать не нужно, и никакого сигнала пайплайну слать не нужно.**

`photo-to-volume` — это одноразовый CLI: он запускается на каждое фото и на
каждом запуске заново читает с диска `config/models_geometry_config.json`,
`config/camera_presets.json` и файлы моделей. Долгоживущего процесса, который
держал бы конфиг в памяти, в C++-части нет.

Реестр пишет так, что параллельный запуск пайплайна не может увидеть
промежуточное состояние:

* конфиги пишутся во временный файл рядом → `fsync` → `rename` (атомарная
  подмена inode);
* каталог модели собирается в `data/models/.staging/…` и въезжает на место
  одним `rename`; старая версия уезжает в `data/models/.trash/` (хранятся
  3 последние, потом удаляются);
* все мутации сериализованы `flock` на `data/models/.registry.lock` —
  это защищает и от второго экземпляра сервера, и от ручной правки скриптами;
* при ошибке на любом шаге состояние откатывается целиком.

Итог: **следующий запуск пайплайна после `HTTP 200/201` уже видит новую модель.**
Запуск, стартовавший до подмены, доработает на старых файлах — они живы, пока
процесс держит их открытыми.

Кэш нужен только вашему Python-приложению, если оно само кэширует список моделей
в памяти. Для этого есть `GET /registry/version` (§5.2) и вебхук (§8).

### 2.1 Два дерева и зеркалирование

На этом сервере `data/` и `config/` существуют в **двух независимых копиях**, и
читают их разные потребители:

| Дерево | Кто читает | Зачем |
|---|---|---|
| `/home/leonid/photo-to-volume/{data,config}` | CLI-пайплайн `photo-to-volume` | резолвит `target_model` и файлы кузова, берёт пресет камеры |
| `/home/leonid/operation-3d-service/{data,config}` | TLS-сервер 9999 | `get_models_config` (список кузовов в утилите), `/download_model_file` |

Пути у TLS-9999 захардкожены абсолютными, а пайплайн он запускает через
`env -u …`, специально сбрасывая переопределения, — свести их к одному дереву
настройкой нельзя. Модель, попавшая только в одно дерево, ломается
предсказуемо: в дереве пайплайна — не видна в списке кузовов; в дереве 9999 —
видна, но обработка падает с `napolnitel OBJ not found`.

Поэтому реестр запускается с зеркалом: основное дерево — `photo-to-volume`,
зеркало — `operation-3d-service`. После успешной подмены модель раскладывается
во второе дерево **хардлинками** (обе копии на одной ФС, поэтому комплект в
150–200 МБ появляется мгновенно и место не удваивается), а запись конфига и
пресет камеры пишутся в оба `models_geometry_config.json` /
`camera_presets.json`.

**Инвариант: модель появляется в обоих деревьях или ни в одном.** Если
зеркалирование сорвалось на любом шаге, запрос откатывает всё — каталоги и все
затронутые конфиги — и возвращает `500`; полусостояния не остаётся.

Зеркало трогает только загружаемый ключ. Собственные расхождения второго дерева
сохраняются: у него есть модель `Scania`, которой нет в первом, и свои
`points_3d` у 13 моделей (их правит `setModelPoints` через утилиту). Пайплайн
`points_3d` не читает, так что этот дрейф безвреден.

Карточка модели показывает состояние зеркала в поле `mirror`, и `ready: false`
выставляется в том числе когда модель не доехала до второго дерева.

---

## 3. Запуск сервера

```bash
cd /home/leonid/photo-to-volume/build
cmake .. && make model-registry

MODEL_REGISTRY_TOKEN='<длинный-случайный-токен>' \
  ./model-registry --host 127.0.0.1 --port 8099 --route-prefix /registry \
    --data-dir   /home/leonid/photo-to-volume/data \
    --config-dir /home/leonid/photo-to-volume/config \
    --mirror-data-dir   /home/leonid/operation-3d-service/data \
    --mirror-config-dir /home/leonid/operation-3d-service/config
```

| Флаг | Env | По умолчанию | Смысл |
|---|---|---|---|
| `--host` | — | `127.0.0.1` | интерфейс |
| `--port` | — | `8099` | порт |
| `--data-dir` | `MODEL_REGISTRY_DATA_DIR` | `<exe>/../data` | каталог `data/` |
| `--config-dir` | `MODEL_REGISTRY_CONFIG_DIR` | `<exe>/../config` | каталог `config/` |
| `--token` | `MODEL_REGISTRY_TOKEN` | — | токен для мутаций |
| `--allow-anonymous-write` | — | выкл. | запись без токена (только локальная отладка) |
| `--protect-reads` | — | выкл. | требовать токен и на `GET` |
| `--max-upload-mb` | — | `2048` | лимит на тело запроса |
| `--notify-url` | `MODEL_REGISTRY_NOTIFY_URL` | — | `http://…` — POST после каждого изменения |
| `--mirror-data-dir` | `MODEL_REGISTRY_MIRROR_DATA_DIR` | — | второе дерево `data/` (см. §2.1) |
| `--mirror-config-dir` | `MODEL_REGISTRY_MIRROR_CONFIG_DIR` | — | второе дерево `config/`; задаётся только вместе с предыдущим |
| `--route-prefix` | — | — | префикс всех маршрутов, например `/registry` (см. §3.1) |

### 3.1 Где реестр работает на самом деле: порт 9999

Отдельный демон выше — рабочий режим для локальной отладки. **В продакшене
реестр живёт внутри `operation-3d-service`**, на порту 9999, потому что снаружи
проброшен только он: 9998 проверен и снаружи недоступен (TCP-таймаут при живом
ICMP — правила проброса для него нет).

Один и тот же код собирается в оба режима (см. `include/model_registry.h`):

* `build/model-registry` — самостоятельный демон;
* `model_registry::register_routes(svr)` — маршруты внутри чужого
  httplib-сервера. `operation-3d-service/CMakeLists.txt` подключает
  `photo-to-volume/src/model_registry.cpp` исходником, а не копией, поэтому
  версия одна на оба режима. `httplib.h` и `json.hpp` в обоих репозиториях
  побайтово одинаковы (0.32.0 / 3.12.0), конфликта версий нет.

Реестр отвечает **только** на `/registry/…`; всё остальное на 9999 — это
собственные маршруты сервиса (`/get_models_config`, `/download_model_file`
и прочие), они не затронуты. Корневые `/health` и `/models` отдают `404`.

Что меняется в самом сервисе при включённом реестре (настройки httplib общие
на весь сервер, поэтому иначе нельзя):

| Настройка | Было | Стало |
|---|---|---|
| `set_payload_max_length` | 200 МБ | 2 ГБ — комплект модели 150–200 МБ идёт одним запросом |
| `set_read_timeout` / `write` | дефолт httplib, 5 с | 600 с — иначе многомегабайтная загрузка рвётся на медленном канале |
| `Options(".*")` CORS | `Content-Type` | плюс `Authorization`, `X-Registry-Token` |

Токен сервис берёт из `MODEL_REGISTRY_TOKEN`, а если её нет — из файла
`/etc/model-registry.env` (строка `MODEL_REGISTRY_TOKEN=…`, права 600 root).
Так секрет не приходится вписывать в unit-файл. Если токен не найден или
конфиги недоступны, **реестр просто не включается**, а сервис поднимается как
обычно и пишет в лог `[registry] НЕ включён: …` — сломать 9999 неудачная
инициализация реестра не может. Аварийный выключатель — переменная
`MODEL_REGISTRY_DISABLE=1` в юните.

Итоговый базовый URL для `tls_config.yaml` → `model_registry: url:`

```
http://78.25.191.12:9999/registry
```

Все пути ниже указаны без префикса — подставляйте базовый URL целиком:
`GET /health` → `http://78.25.191.12:9999/registry/health`.

Без токена и без `--allow-anonymous-write` сервер **не стартует**.

### TLS и публичный доступ

Бинарник собран без OpenSSL — он говорит по обычному HTTP. Наружу выставляйте
через nginx, он же даёт TLS, лимит размера тела и, при желании, IP-allowlist:

```nginx
location /registry/ {
    proxy_pass         http://127.0.0.1:9999/registry/;
    proxy_request_buffering off;      # обязательно: аплоады идут потоком
    client_max_body_size 2048m;
    proxy_read_timeout 600s;
    proxy_send_timeout 600s;
}
```

### systemd

```ini
[Unit]
Description=photo-to-volume model registry
After=network.target

[Service]
User=leonid
WorkingDirectory=/home/leonid/photo-to-volume/build
Environment=MODEL_REGISTRY_TOKEN=<токен>
Environment=MODEL_REGISTRY_NOTIFY_URL=http://127.0.0.1:8000/internal/models-changed
ExecStart=/home/leonid/photo-to-volume/build/model-registry \
    --host 127.0.0.1 --port 8099 --route-prefix /registry \
    --data-dir   /home/leonid/photo-to-volume/data \
    --config-dir /home/leonid/photo-to-volume/config \
    --mirror-data-dir   /home/leonid/operation-3d-service/data \
    --mirror-config-dir /home/leonid/operation-3d-service/config
Restart=always

[Install]
WantedBy=multi-user.target
```

### Аутентификация

Мутирующие запросы (`POST` / `PUT` / `DELETE`) требуют один из заголовков:

```
Authorization: Bearer <токен>
X-Registry-Token: <токен>
```

Без него — `401`. `GET` по умолчанию открыт (для этого и «публичный эндпоинт»),
закрывается флагом `--protect-reads`. CORS открыт (`*`), preflight `OPTIONS`
обрабатывается.

---

## 4. Что такое «моделька» — анатомия

Ключ модели (`<key>`) — это одновременно имя каталога в `data/models/`, ключ в
`models_geometry_config.json`, поле `model` в пресете камеры и то, что
прикладной код кладёт в `meta.model` при обработке фото. Разрешены
`[A-Za-z0-9._-]`, до 64 символов.

Реестр раскладывает загруженное так:

```
data/models/<key>/
    <key>-Napolnitel.obj    ← ОБЯЗАТЕЛЬНО: наполнитель, по нему считается объём
    <key>-Cuzov.obj         ← ОБЯЗАТЕЛЬНО: кузов, по нему ставятся анкерные точки
    <key>-Cuzov.bam         ← кузов в debug-preview
    <key>-Napolnitel.bam
    <key>-Other.bam         ← остальная часть машины в debug-preview
    <key>-Napolnitel.mtl    ← опционально
    <key>-Other.obj / -Body.obj / -Full.obj   ← опционально, исходники
    WEB/…                   ← опционально: gltf/bin/текстуры для веб-вьюера
data/textures/<textures_dir>/…                ← опционально: текстуры машины
config/models_geometry_config.json  → запись "<key>": {…}
config/camera_presets.json          → пресет {"model": "<key>", pos, hpr, fov}
```

**Имена файлов задаёт сервер, не клиент.** Клиент присылает файл в поле с
именем роли (`napolnitel_obj`, `cuzov_obj`, …), а сервер сам называет его
`<key>-Napolnitel.obj` и т.д. Так гарантируется, что пайплайн его найдёт: он
ищет файлы по имени, отрезая от `target_model` суффикс `-Napolnitel`
(см. [src/main.cpp](../src/main.cpp)).

### Что из этого реально нужно пайплайну

Проверено по коду C++-пайплайна:

| Сущность | Читает C++-пайплайн | Комментарий |
|---|---|---|
| `<key>-Napolnitel.obj` (`target_model`) | **да** | без него `500 napolnitel OBJ not found` |
| `<key>-Cuzov.obj` | **да** | без него `500 cuzov OBJ missing` |
| пресет камеры в `camera_presets.json` | **да** | без совпадения `500 camera preset lookup failed` |
| `-Cuzov.bam` / `-Other.bam` | только для debug-preview | без них preview рендерится без машины |
| `max_volume`, `ground_plane`, `points_3d`, `cam_pos_*`, `cam_rot_*` | нет | их читает **ваше** приложение (процент заполнения, вьюер) |
| `textures_dir`, `WEB/*` | нет | preview-рендер и веб-вьюер |

То есть модель «поедет» при наличии двух OBJ и пресета камеры; всё остальное —
для качества картинки и для прикладной логики. Реестр отдаёт этот вердикт полем
`ready` (§5.3), не заставляя вас помнить эти правила.

---

## 5. Эндпоинты

Базовый URL ниже — `$BASE` (например `https://host/registry`).

### 5.1 `GET /health`

```json
{ "status": "ok", "service": "photo-to-volume model registry",
  "revision": 42, "data_dir": "…", "config_dir": "…" }
```

### 5.2 `GET /registry/version`

Ревизия реестра — для инвалидации кэша. Поддерживает `ETag` / `If-None-Match`
(при совпадении вернёт `304` без тела).

```json
{ "status": "ok", "revision": 42, "updated_at": "2026-09-09T20:15:17Z",
  "models": { "MAZ-6x4": { "revision": 41, "updated_at": "…", "action": "update" } } }
```

`revision` — монотонный счётчик, растёт на каждое изменение любой модели.

### 5.3 `GET /models` и `GET /models/{key}`

Список / одна модель. Параметр `?hashes=1` добавляет SHA-256 каждого файла
(медленнее — файлы читаются целиком).

Карточка модели:

```json
{
  "key": "Volvo-FMX-8x4",
  "display_name": "Volvo FMX 8x4",
  "config": { "model": "…", "cuzov": "models/…", "target_model": "models/…",
              "max_volume": 28, "ground_plane": 2.1, "points_3d": [[…]] },
  "camera_preset": { "pos": [...], "hpr": [...], "fov": 87, "model": "Volvo-FMX-8x4" },
  "dir": "/home/leonid/photo-to-volume/data/models/Volvo-FMX-8x4",
  "stem": "Volvo-FMX-8x4",
  "files_required": { "napolnitel_obj": {"file": "…", "present": true}, … },
  "files": [ { "name": "Volvo-FMX-8x4-Cuzov.obj", "size": 1771745, "sha256": "…" } ],
  "mirror": { "config_dir": "/home/leonid/operation-3d-service/config",
              "in_config": true, "target_model_present": true, "ok": true },
  "problems": [],          // пусто ⇒ пайплайн отработает И модель видна в списке кузовов
  "warnings": [],          // не блокируют, но стоит прочитать
  "ready": true,
  "revision": 43,
  "updated_at": "2026-09-09T20:15:17Z"
}
```

`ready` вычисляется по тем же правилам, по которым модель ищет сам пайплайн
(включая нечёткий матч ключа и пресета камеры), поэтому это честный ответ на
вопрос «сработает ли обработка фото с этой моделью».

В списке показываются и каталоги без записи в конфиге — у них `ready: false` и
`problems: ["нет записи в models_geometry_config.json"]`.

### 5.4 `POST /models/{key}` — создать или заменить

`PUT` делает ровно то же самое. `Content-Type: multipart/form-data`.

**Текстовые поля**

| Поле | Обяз. | Смысл |
|---|---|---|
| `meta` | да | JSON-объект с метаданными (§6) |
| `mode` | нет | `replace` (по умолчанию) или `patch` |

* `replace` — каталог модели собирается заново: чего не прислали, того в модели
  не будет. Запись конфига пересобирается с нуля.
* `patch` — недостающие файлы доносятся из текущего каталога модели, а запись
  конфига обновляется только присланными полями `meta`. Удобно, чтобы поправить
  один `max_volume` или заменить одну `.bam`.

**Файловые поля** (имя поля = роль, имя файла клиента игнорируется):

| Поле | Станет файлом | Обяз. |
|---|---|---|
| `napolnitel_obj` | `<key>-Napolnitel.obj` | **да** |
| `cuzov_obj` | `<key>-Cuzov.obj` | **да** |
| `cuzov_bam` | `<key>-Cuzov.bam` | нет |
| `napolnitel_bam` | `<key>-Napolnitel.bam` | нет |
| `other_bam` | `<key>-Other.bam` | нет |
| `napolnitel_mtl` | `<key>-Napolnitel.mtl` | нет |
| `other_obj` / `body_obj` / `full_obj` | `<key>-Other.obj` / `-Body.obj` / `-Full.obj` | нет |
| `texture` (можно много раз) | `data/textures/<textures_dir>/<имя файла>` | нет |
| `web` (можно много раз) | `data/models/<key>/WEB/<относительный путь>` | нет |

Для `texture` и `web` имя файла клиента сохраняется (до 4 уровней вложенности,
только `[A-Za-z0-9._- ]` в компонентах пути). Неизвестные поля игнорируются и
перечисляются в ответе как `ignored_parts` — это не ошибка, но повод проверить
опечатку в имени поля.

Ответ `201` (создано) или `200` (заменено):

```json
{ "status": "ok", "created": true, "key": "Volvo-FMX-8x4", "mode": "replace",
  "revision": 43,
  "stored_files": ["Volvo-FMX-8x4-Napolnitel.obj", "…", "textures/paint.png"],
  "ignored_parts": [],
  "warnings": ["не загружен Volvo-FMX-8x4-Other.bam (поле 'other_bam'): …"],
  "model": { …карточка из §5.3… } }
```

**Всегда проверяйте `model.ready` в ответе**, а не только HTTP-код: `200/201`
означает «файлы и конфиг записаны», а `ready` — «пайплайн с этим поедет».

### 5.5 `DELETE /models/{key}`

| Параметр | Смысл |
|---|---|
| `?purge=1` | удалить каталог насовсем (по умолчанию он уезжает в `.trash/`) |
| `?keep_camera_preset=1` | не удалять пресет камеры |

```json
{ "status": "ok", "deleted": "Volvo-FMX-8x4", "purged": false, "revision": 44 }
```

### 5.6 `GET /models/{key}/files/{путь}`

Отдаёт файл модели — удобно, чтобы веб-вьюер тянул `WEB/<key>.gltf` прямо
отсюда, не расшаривая файловую систему.

---

## 6. Поле `meta`

```json
{
  "display_name": "Volvo FMX 8x4",
  "max_volume": 28,
  "ground_plane": 2.1,
  "points_3d": [[1.15, 3.25, 3.65], [-1.15, 3.25, 3.65],
                [-1.15, -2.35, 3.65], [1.15, -2.35, 3.65]],
  "textures_dir": "Volvo-FMX-8x4",
  "camera": { "pos": [0.62, 2.81, 3.26], "hpr": [160.02, -23.34, 0.0], "fov": 87.0 },

  "cam_pos_x": -1.1, "cam_pos_y": 3.2, "cam_pos_z": 4.2,
  "cam_rot_h": -137, "cam_rot_p": -42, "cam_rot_r": 0,
  "extra": { "любое_поле": "уедет в запись конфига как есть" }
}
```

| Поле | Обяз. при `replace` | Что это |
|---|---|---|
| `display_name` | нет (по умолчанию `<key>`) | человекочитаемое имя → поле `model` в конфиге |
| `max_volume` | **да** | полный объём кузова, м³. Пайплайн его не читает; ваше приложение делит на него посчитанный объём, чтобы получить процент загрузки |
| `ground_plane` | **да** | уровень «пола» кузова в координатах модели |
| `points_3d` | **да** | 4 точки `[x,y,z]` — прямоугольник верхней кромки кузова в координатах модели. C++-пайплайн его не читает (анкеры ставятся автоматически по маске кузова), поле нужно прикладным потребителям конфига |
| `camera` | **да** для новой модели | пресет съёмки: `pos` — позиция камеры, `hpr` — heading/pitch/roll в градусах, `fov` — горизонтальный угол. Без него пайплайн падает с `camera preset lookup failed` |
| `textures_dir` | нет (по умолчанию `<key>`) | имя каталога в `data/textures/`; в конфиг уедет как `textures/<имя>` |
| `cam_pos_*`, `cam_rot_*` | нет | легаси-поля вьюера; есть у 8 из 17 существующих моделей |
| `extra` | нет | любые дополнительные ключи записи конфига |

При `mode=patch` обязательных полей нет — присланные перезапишут старые,
остальные останутся.

Если модель с таким ключом уже есть в `camera_presets.json` (точное совпадение
без учёта регистра), `camera` можно не присылать — старый пресет останется.

---

## 7. Готовый Python-клиент

Для больших файлов ставьте `requests-toolbelt` — иначе `requests` соберёт всё
тело мультипарта в память:

```bash
pip install requests requests-toolbelt
```

```python
"""Клиент реестра моделей photo-to-volume."""
from __future__ import annotations

import json
import pathlib
from typing import Iterable

import requests

try:
    from requests_toolbelt.multipart.encoder import MultipartEncoder
except ImportError:                       # можно и без него, но тело уедет в память
    MultipartEncoder = None


class ModelRegistryError(RuntimeError):
    def __init__(self, status: int, payload: dict):
        self.status = status
        self.payload = payload
        super().__init__(f"HTTP {status}: {payload.get('error', payload)}")


class ModelRegistry:
    # роль → имя поля мультипарта
    FILE_ROLES = (
        "napolnitel_obj", "cuzov_obj", "cuzov_bam", "napolnitel_bam",
        "other_bam", "napolnitel_mtl", "other_obj", "body_obj", "full_obj",
    )

    def __init__(self, base_url: str, token: str | None = None, timeout: int = 600):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"

    # ---------- чтение ----------
    def health(self) -> dict:
        return self._json("GET", "/health")

    def version(self) -> dict:
        """{'revision': int, 'updated_at': str, 'models': {...}} — для кэша."""
        return self._json("GET", "/registry/version")

    def list_models(self, with_hashes: bool = False) -> list[dict]:
        params = {"hashes": "1"} if with_hashes else None
        return self._json("GET", "/models", params=params)["models"]

    def get_model(self, key: str, with_hashes: bool = False) -> dict:
        params = {"hashes": "1"} if with_hashes else None
        return self._json("GET", f"/models/{key}", params=params)["model"]

    # ---------- запись ----------
    def upsert_model(
        self,
        key: str,
        meta: dict,
        files: dict[str, str | pathlib.Path],      # роль → путь
        textures: Iterable[str | pathlib.Path] = (),
        web_files: Iterable[str | pathlib.Path] = (),
        mode: str = "replace",
    ) -> dict:
        """Создать или заменить модель. Возвращает ответ сервера целиком.

        Бросает ModelRegistryError на 4xx/5xx.
        Проверьте result['model']['ready'] — HTTP 200 ещё не значит,
        что пайплайн с этой моделью поедет.
        """
        for role in files:
            if role not in self.FILE_ROLES:
                raise ValueError(f"неизвестная роль файла: {role!r}")

        opened: list = []
        try:
            fields: list[tuple[str, object]] = [
                ("meta", json.dumps(meta, ensure_ascii=False)),
                ("mode", mode),
            ]
            for role, path in files.items():
                p = pathlib.Path(path)
                fh = p.open("rb"); opened.append(fh)
                fields.append((role, (p.name, fh, "application/octet-stream")))
            for path in textures:
                p = pathlib.Path(path)
                fh = p.open("rb"); opened.append(fh)
                fields.append(("texture", (p.name, fh, "application/octet-stream")))
            for path in web_files:
                p = pathlib.Path(path)
                fh = p.open("rb"); opened.append(fh)
                fields.append(("web", (p.name, fh, "application/octet-stream")))

            if MultipartEncoder is not None:
                enc = MultipartEncoder(fields=fields)
                resp = self.session.post(
                    f"{self.base}/models/{key}", data=enc,
                    headers={"Content-Type": enc.content_type}, timeout=self.timeout)
            else:
                # Текстовые поля тоже уходят через files=[...] как (None, value):
                # если положить их в data=, а files окажется пустым (mode=patch
                # без файлов), requests отправит urlencoded и сервер ответит 415.
                parts = [(name, (None, val) if isinstance(val, str) else val)
                         for name, val in fields]
                resp = self.session.post(f"{self.base}/models/{key}",
                                         files=parts, timeout=self.timeout)
            return self._parse(resp)
        finally:
            for fh in opened:
                fh.close()

    def delete_model(self, key: str, purge: bool = False) -> dict:
        params = {"purge": "1"} if purge else None
        return self._json("DELETE", f"/models/{key}", params=params)

    # ---------- внутреннее ----------
    def _json(self, method: str, path: str, **kw) -> dict:
        return self._parse(self.session.request(
            method, f"{self.base}{path}", timeout=self.timeout, **kw))

    @staticmethod
    def _parse(resp: requests.Response) -> dict:
        try:
            payload = resp.json()
        except ValueError:
            payload = {"error": resp.text[:500]}
        if resp.status_code >= 400:
            raise ModelRegistryError(resp.status_code, payload)
        return payload
```

Использование:

```python
reg = ModelRegistry("http://78.25.191.12:9999/registry",
                    token=os.environ["MODEL_REGISTRY_TOKEN"])

result = reg.upsert_model(
    key="Volvo-FMX-8x4",
    meta={
        "display_name": "Volvo FMX 8x4",
        "max_volume": 28,
        "ground_plane": 2.1,
        "points_3d": [[1.15, 3.25, 3.65], [-1.15, 3.25, 3.65],
                      [-1.15, -2.35, 3.65], [1.15, -2.35, 3.65]],
        "camera": {"pos": [0.62, 2.81, 3.26], "hpr": [160.02, -23.34, 0.0], "fov": 87.0},
    },
    files={
        "napolnitel_obj": "upload/Volvo-Napolnitel.obj",
        "cuzov_obj":      "upload/Volvo-Cuzov.obj",
        "cuzov_bam":      "upload/Volvo-Cuzov.bam",
        "napolnitel_bam": "upload/Volvo-Napolnitel.bam",
        "other_bam":      "upload/Volvo-Other.bam",
    },
    textures=["upload/paint_color.png", "upload/paint_normal.png"],
)

model = result["model"]
if not model["ready"]:
    raise RuntimeError("модель загружена, но пайплайн с ней не поедет: "
                       + "; ".join(model["problems"]))
for w in result["warnings"]:
    log.warning("registry: %s", w)
```

---

## 8. Инвалидация кэша в вашем приложении

Если приложение держит список моделей в памяти — обновляйте его по `revision`.

**Вариант A: опрос (просто и достаточно).**

```python
class ModelCache:
    def __init__(self, registry: ModelRegistry, ttl: float = 30.0):
        self.reg = registry
        self.ttl = ttl
        self._models: dict[str, dict] = {}
        self._revision = -1
        self._checked_at = 0.0

    def models(self) -> dict[str, dict]:
        now = time.monotonic()
        if now - self._checked_at > self.ttl:
            self._checked_at = now
            rev = self.reg.version()["revision"]
            if rev != self._revision:
                self._models = {m["key"]: m for m in self.reg.list_models()}
                self._revision = rev
        return self._models
```

**Вариант B: вебхук (мгновенно).** Запустите реестр с
`--notify-url http://127.0.0.1:8000/internal/models-changed` — после каждого
изменения он пошлёт (best-effort, без ретраев, только `http://`):

```json
{ "event": "model.created", "key": "Volvo-FMX-8x4",
  "revision": 43, "at": "2026-09-09T20:15:17Z" }
```

`event` — `model.created` | `model.updated` | `model.deleted`.

```python
@app.post("/internal/models-changed")
async def models_changed(payload: dict):
    model_cache.invalidate()      # доставка не гарантирована — оставьте и опрос
    return {"ok": True}
```

Вебхук — оптимизация поверх опроса, а не замена: сервис мог быть недоступен в
момент отправки.

**Что кэшировать нельзя:** результат обработки фото, привязанный к модели.
После замены модели старые расчёты объёма относятся к старой геометрии —
инвалидируйте их вместе с моделью, если храните.

---

## 9. Коды ответов

| Код | Когда | Что делать |
|---|---|---|
| `200` | модель заменена / прочитана / удалена | проверить `model.ready` |
| `201` | модель создана | проверить `model.ready` |
| `304` | `If-None-Match` совпал с текущей ревизией | кэш актуален |
| `400` | плохой ключ, кривой `meta`, отсутствует обязательное поле | чинить запрос; текст в `error` конкретный |
| `401` | нет/неверный токен | проверить заголовок |
| `404` | нет такой модели или файла | — |
| `413` | тело больше `--max-upload-mb` | поднять лимит на сервере **и** в nginx |
| `415` | не `multipart/form-data` | — |
| `422` | не хватает обязательного файла или файл не похож на OBJ | догрузить; в `hint` подсказка про `mode=patch` |
| `500` | ошибка ФС/записи конфига, в т.ч. сбой зеркалирования | смотреть stderr сервера; изменения откатились в обоих деревьях |
| `503` | не удалось взять `flock` | повторить позже |

Тело ошибки: `{"status": "error", "error": "…", "hint": "…"}` — `error`
человекочитаемый, его можно показывать оператору.

---

## 10. Грабли, которые стоит знать

1. **Нечёткий поиск модели по имени.** Пайплайн ищет модель в конфиге не
   строгим совпадением: точное → без учёта регистра → по префиксу → по
   подстроке. Поэтому короткий ключ (`MAZ`) может перехватывать запросы,
   адресованные другому (`MAZ-6x4`, `Kamaz`). Реестр предупреждает об этом при
   создании (`warnings`), но не блокирует. **В `meta.model` при обработке фото
   всегда передавайте ключ точно.**
2. **`ready: false` у уже существующих моделей — это правда, а не баг реестра.**
   На момент написания из 17 моделей в конфиге пайплайн отработает только на
   `MAZ-6x4` и `SCANIA-P8X400-P380CB8X4EHZ`: у остальных нет `-Cuzov.obj`
   и/или пресета камеры. Все примеры в `examples/` используют именно эти две.
3. **Имена файлов внутри каталога модели канонизируются.** У части старых
   моделей стем файлов не совпадает с именем каталога (`Shackman-…/Schackman-…`).
   Реестр это корректно читает, но всё новое пишет как `<key>-*`.
3a. **Файлы в двух деревьях — это один inode (хардлинк), а не две копии.**
   Правка файла модели «на месте» в одном дереве изменит его и во втором.
   Так никто не делает — замена всегда идёт через новый staging и `rename`,
   при котором inode новый, — но если будете чинить модель руками, копируйте
   файл, а не редактируйте по месту.
4. **Реестр наследует стиль файла, который переписывает:** переводы строк
   (у `photo-to-volume` конфиги с CRLF, у `operation-3d-service` — с LF) и шаг
   отступа (2 против 4 пробелов — второй конфиг форматирует `setModelPoints`).
   Порядок ключей сохраняется, модель дописывается в конец. Поэтому добавление
   модели даёт diff на ~30 строк, а не на весь файл.
   Единственный неустранимый шум — рендер дробных чисел: `nlohmann` печатает
   кратчайшую форму, которая читается обратно в тот же double, поэтому
   `-31.800003051757812` может стать `...813`. Это тот же самый IEEE-754 double,
   потери точности нет; после цикла «загрузка + удаление» файл отличается от
   исходного на 2–4 такие строки.
5. **`.trash` растёт.** Хранятся 3 последние версии каталога на ключ, остальные
   удаляются автоматически. При `DELETE` без `?purge=1` каталог тоже уезжает
   туда — место освободится не сразу.
6. **`proxy_request_buffering off` в nginx обязателен**, иначе прокси сначала
   сложит весь аплоад к себе на диск и вы получите двойное время загрузки.
7. **Параллельная обработка фото во время замены модели безопасна**, но запуск,
   стартовавший до подмены, доработает на старой геометрии. Если это важно —
   ставьте загрузку моделей в тот же пул, что и обработку, либо просто не
   заменяйте модель под нагрузкой.

---

## 11. Чеклист приёмки интеграции

- [ ] `GET /health` отвечает, `revision` растёт после загрузки
- [ ] загрузка новой модели даёт `201` и `model.ready == true`
- [ ] `model.mirror.ok == true` — модель доехала во второе дерево, значит
      появится в списке кузовов утилиты (`get_models_config` на TLS-9999)
- [ ] `model.warnings` и `result.warnings` логируются, а не проглатываются
- [ ] `ignored_parts` пуст (иначе — опечатка в имени поля)
- [ ] повторная загрузка того же ключа даёт `200` и `created: false`
- [ ] `mode=patch` с одним `meta` меняет конфиг и не теряет файлы
- [ ] `DELETE` убирает модель из `GET /models` и из обоих конфигов
- [ ] обработка фото с `meta.model = <новый key>` проходит без
      `camera preset lookup failed` / `cuzov OBJ missing`
- [ ] кэш моделей в приложении обновляется по `revision` (и/или по вебхуку)
- [ ] токен лежит в env/секретнице, а не в коде; наружу — только через TLS
