# -*- coding: utf-8 -*-
"""
Какие кузова сервер уже снимает, а сгенерировать их ещё некому.

Откуда берётся список
---------------------
Сервер `operation-3d-service` на каждый снимок кладёт в `data/PLY_examples`
тройку `<base>.json` / `<base>.ply` / `<base>_topdown.jpg`. В JSON оператор
проставляет модель кузова — строку вида «Shacman X3000 8x4 (высокий)». Если
такой строки нет в `models_geometry_config.json`, пайплайн не находит
геометрию наполнителя и подставляет общий `data/Napolnitel.obj`
(`mesh_reconstruction.cpp: get_target_model_path` → «Using fallback napolnitel
path»). Снимок обсчитывается, но по чужому кузову: объём получается не тот.
Ровно эти имена и есть очередь на генерацию.

Почему не `/get_verified_models`
-------------------------------
Этот маршрут отдаёт только записи с `passed_verification == "True"`, а снимок
ПУСТОГО кузова до верификации не доходит: его срезает filler-gate
(`pipeline_skipped_reason: filler=empty`), и поля `passed_verification` в
таком JSON нет вовсе. То есть ровно то, что нам нужно для генерации, из этой
выдачи и выпадает — у JAC 6x4, например, все 24 пустых скана. Поэтому список
собирается по сырым JSON.

Как перечисляются записи
------------------------
Прямого «дай все имена» у сервера нет: `/list_obj_results` отдаёт не больше
200 имён (жёсткий потолок в `getObjResults`), самых свежих по mtime. Зато он
фильтрует по СУФФИКСУ имени. Имена записей оканчиваются на hex-символы, так
что каталог разбирается рекурсивно: `.json` → если ответ упёрся в потолок,
то `0.json`, `1.json`, … `f.json`, и так вглубь, пока шард не перестанет
насыщаться. Два уровня хватает на ~8 тысяч записей, это пара секунд.

Дальше каждый JSON качается через `/download?file=` (~400 байт) и кладётся в
кэш на диске: первый проход — около минуты, последующие берут с диска и
дочитывают только новые имена.

Наличие фотографии проверяется HEAD-запросом к `/download` — тело при этом не
едет, а 404 отличается от 200. Поэтому «есть ли у этого скана topdown» стоит
один round-trip, а не 400 КБ.

Qt здесь нет и быть не должно: модуль обязан работать из CLI и из рабочего
потока UI одинаково.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import requests

#: Кэш сырых JSON записей и скачанных .ply/.jpg. Рядом с кэшем текстур
#: (`panel_data.TEXTURES_CACHE_DIR`) — та же конвенция: временная папка, а не
#: дерево проекта, чтобы сотни мегабайт сканов не оседали в репозитории.
CACHE_DIR = os.path.join(tempfile.gettempdir(), "vizutil_worklist_cache")

#: Разобранные JSON записей: {имя файла: содержимое}. Один файл, а не россыпь
#: из восьми тысяч, — иначе первый проход упирается не в сеть, а в файловую
#: систему Windows.
RECORDS_CACHE = os.path.join(CACHE_DIR, "records.json")

#: Какие сканы имеют topdown-фото: {base: bool}. Отдельный файл, потому что
#: ответ на HEAD не зависит от содержимого записи и переживает её перечитку.
PHOTOS_CACHE = os.path.join(CACHE_DIR, "photos.json")

#: Куда кладутся скачанные облака и фотографии.
ASSETS_DIR = os.path.join(CACHE_DIR, "assets")

#: Потолок `getObjResults` на стороне сервера. Ответ ровно такой длины —
#: признак того, что шард обрезан и его надо дробить дальше.
_LIST_LIMIT = 200

#: Из каких символов складываются окончания имён записей (hex + разделители).
_SHARD_CHARSET = "0123456789abcdef-_"

#: Глубже дробить бессмысленно: 18^3 запросов дороже, чем выгода.
_SHARD_MAX_DEPTH = 3

#: Сколько файлов тянуть параллельно. Больше — сервер начинает отвечать
#: медленнее, чем мы успеваем просить: httplib там однопоточный по соединению.
_FETCH_WORKERS = 16

#: Значения `filler` / `load_state`, означающие пустой кузов.
_EMPTY_MARKS = ("empty",)

#: Модель, которую клиент шлёт, когда не распознал кузов.
_UNKNOWN_MODEL = "unknown"

ProgressFn = Callable[[str, float], None]
CancelFn = Callable[[], bool]


class WorklistCancelled(RuntimeError):
    """Пользователь закрыл диалог посреди обхода."""


def normalize_model_name(name: str) -> str:
    """
    Имя модели в сравнимом виде.

    Кириллическая «х» в обозначении осей (6х4, 8х4) путается с латинской «x»
    постоянно: в реестре одно, в клиенте другое. Сервер нормализует так же
    (`scripts/truck_models.normalize_model_name`), и без этого половина
    моделей выглядела бы отсутствующей в базе.
    """
    s = str(name or "").strip()
    return re.sub(r"(\d)[хХ](\d)", r"\1x\2", s)


#: Хвосты имени, у которых есть принятый в реестре перевод. Порядок важен:
#: «(стандарт)» ищется как есть, а не по подстроке «станд».
_KEY_WORDS = (
    ("(стандарт)", "standard"),
    ("(высокий)", "tall"),
    ("оси", "axle"),
    ("ось", "axle"),
    ("кузов", "body"),
    ("борт", "side"),
)


def suggest_key(display_name: str) -> str:
    """
    Ключ реестра по имени модели у клиента.

    Повторяет стиль уже заведённых ключей (`Shackman-X3000-8x4-tall`,
    `KAMAZ-65201-8x4-standard`): латиница, дефисы, суффикс комплектации
    словом. Это предложение, а не канон — в диалоге ключ правится руками.
    """
    from src.registry.client import suggest_key as _translit

    s = normalize_model_name(display_name)
    for src, dst in _KEY_WORDS:
        s = s.replace(src, dst)
    key = _translit(s)
    # «мод.» → «mod.-tall» читается как опечатка: точка на стыке со словом
    # лишняя. Внутри числа («6.3m») она осмысленна и остаётся.
    key = key.replace(".-", "-").replace("_", "-")
    key = re.sub(r"-+", "-", key).strip("-._")
    return key


def client_name_for_key(key: str, cache_dir: str = CACHE_DIR) -> str:
    """
    Имя модели у клиента, из которого получился бы такой ключ, или "".

    Нужно диалогу загрузки в реестр: сервер ищет геометрию по полю `model`
    точным сравнением строк, так что в реестр должно уехать ровно то имя,
    которое оператор выбирает в клиенте («КАМАЗ 65952-10792-CA 6х6
    (высокий)»), а не имя комплекта на диске. В сеть функция не ходит —
    смотрит в кэш последнего обхода; нет кэша — нет и ответа.
    """
    key = str(key or "").strip().lower()
    if not key:
        return ""
    try:
        with open(os.path.join(cache_dir, "records.json"), "r",
                  encoding="utf-8") as fh:
            records = json.load(fh)
    except Exception:                                     # noqa: BLE001
        return ""
    if not isinstance(records, dict):
        return ""
    # Одно и то же имя встречается сотни раз — ключ считаем по разу на имя.
    counts: Dict[str, int] = {}
    for data in records.values():
        if isinstance(data, dict):
            name = str(data.get("model") or "").strip()
            if name and name.lower() != _UNKNOWN_MODEL:
                counts[name] = counts.get(name, 0) + 1
    hits = [n for n in counts if suggest_key(n).lower() == key]
    # Два написания с одним ключом («6x4» и «6х4») — берём то, которым
    # оператор пользуется чаще: под ним и лежит большинство снимков.
    return max(hits, key=lambda n: counts[n]) if hits else ""


# ---------------------------------------------------------------------------
# Переименования
# ---------------------------------------------------------------------------
#
# Модель на сервере можно переименовать, и тогда снимки, снятые под старым
# именем, перестают находить свою же геометрию: `get_target_model_path`
# сравнивает строки точно. Так случилось с «MAN (8м) (стандарт)»: 02.10.2026
# запись `MAN-8M-standard` стала называться «MAN Gr.T31 (8м) (стандарт)», и
# 224 старых снимка повисли без геометрии, хотя кузов давно смоделирован.
#
# Генерировать такой кузов заново не нужно — достаточно поправить имя в
# реестре. Поэтому такие строки опознаются и в очередь по умолчанию не идут.
#
# Признак из двух частей, обе обязательны:
#   1. ИМЕНА. Набор слов одного имени целиком входит в другой («MAN 8м
#      стандарт» ⊂ «MAN Gr.T31 8м стандарт», «Shacman 8x4 высокий» ⊂ «Shacman
#      X3000 8x4 высокий»). Одного этого мало: «Volvo 3 оси» и «Volvo 4 оси»
#      так не схлопываются, а вот «МАЗ (стандарт)» ⊂ «МАЗ 6501 6х4 (стандарт)»
#      схлопнулось бы, хотя это разные кузова.
#   2. РАЗМЕР. Обмер кузова по облаку совпадает с проёмом заведённой модели
#      (`points_3d`). Это и отсекает ложные пары: у «МАЗ (стандарт)» обмер
#      5.10 м против 5.63 м у «МАЗ 6501», полметра разницы.

#: Допуск сравнения, м. Обмер по облаку гуляет на несколько сантиметров от
#: скана к скану, а проём в конфиге — номинал, поэтому допуск не нулевой. По
#: длине шире: туда попадает задний борт, который лидар видит по-разному.
_RENAME_TOL_LENGTH = 0.35
_RENAME_TOL_WIDTH = 0.20

#: Слова, которые ничего не говорят о кузове и только мешают сравнению.
_NOISE_TOKENS = frozenset({"", "м", "m", "mm", "мм"})


def _rect_from_points(points: Any) -> Tuple[float, float]:
    """
    (длина, ширина) проёма по `points_3d` модели.

    Четыре угла проёма лежат в плоскости крышки; длина — вдоль Y, ширина —
    вдоль X (такова конвенция конфига, см. любую запись
    `models_geometry_config.json`).
    """
    if not isinstance(points, (list, tuple)) or len(points) < 4:
        return 0.0, 0.0
    try:
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
    except (TypeError, ValueError, IndexError):
        return 0.0, 0.0
    return max(ys) - min(ys), max(xs) - min(xs)


def _tokens(name: str) -> frozenset:
    """Имя модели как набор значащих слов."""
    s = normalize_model_name(name).lower()
    s = re.sub(r"[()\[\].,_/\\]+", " ", s)
    return frozenset(t for t in s.split() if t not in _NOISE_TOKENS)


@dataclass(frozen=True)
class KnownModel:
    """Модель, уже заведённая на сервере."""

    key: str
    display_name: str
    #: Проём по `points_3d`, м. Нули — точек в записи нет.
    length: float = 0.0
    width: float = 0.0


@dataclass(frozen=True)
class RenameHint:
    """Подозрение, что модель просто переименовали."""

    known: KnownModel
    #: Обмер по облаку, с которым сравнивали.
    length: float
    width: float

    def explain(self) -> str:
        return (f"похоже на переименование: на сервере есть "
                f"«{self.known.display_name}» (ключ {self.known.key}), проём "
                f"{self.known.length:.2f} × {self.known.width:.2f} м против "
                f"обмера {self.length:.2f} × {self.width:.2f} м")


def find_rename(display_name: str, length: float, width: float,
                known: Sequence[KnownModel]) -> Optional[RenameHint]:
    """
    Та же модель под другим именем — или None.

    Без обмера (нулевые длина/ширина) гипотеза не проверяется и не
    выдвигается: совпадение одних слов слишком часто врёт.
    """
    if length <= 0 or width <= 0:
        return None
    mine = _tokens(display_name)
    if not mine:
        return None

    best: Optional[RenameHint] = None
    best_gap = None
    for entry in known:
        if entry.length <= 0 or entry.width <= 0:
            continue
        theirs = _tokens(entry.display_name)
        if not (mine <= theirs or theirs <= mine):
            continue
        dl = abs(entry.length - length)
        dw = abs(entry.width - width)
        if dl > _RENAME_TOL_LENGTH or dw > _RENAME_TOL_WIDTH:
            continue
        gap = dl + dw
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best = RenameHint(known=entry, length=length, width=width)
    return best


# ---------------------------------------------------------------------------
# Записи
# ---------------------------------------------------------------------------


@dataclass
class Shot:
    """Один снимок — то, что лежит в `<base>.json` на сервере."""

    base: str
    model: str = ""
    car_number: str = ""
    filler: str = ""
    load_state: str = ""
    direction: str = ""
    station: str = ""
    time: str = ""
    ply_file: str = ""
    length: float = 0.0
    width: float = 0.0

    @property
    def json_file(self) -> str:
        return f"{self.base}.json"

    @property
    def photo_file(self) -> str:
        return f"{self.base}_topdown.jpg"

    @property
    def is_empty(self) -> bool:
        return (self.filler.lower() in _EMPTY_MARKS
                or self.load_state.lower() in _EMPTY_MARKS)

    @property
    def day(self) -> str:
        """`YYYYMMDD` из имени файла, иначе из поля `time`. '' — не разобрано."""
        m = re.search(r"_(\d{8})\d{6}$", self.base)
        if m:
            return m.group(1)
        m = re.match(r"(\d{2})\.(\d{2})\.(\d{4})", self.time or "")
        if m:
            return m.group(3) + m.group(2) + m.group(1)
        return ""

    @classmethod
    def from_json(cls, base: str, data: Dict[str, Any]) -> "Shot":
        dims = data.get("body_dimensions")
        length = width = 0.0
        if isinstance(dims, dict):
            try:
                length = float(dims.get("length") or 0.0)
                width = float(dims.get("width") or 0.0)
            except (TypeError, ValueError):
                length = width = 0.0
        model = data.get("model")
        return cls(
            base=base,
            model=str(model or "").strip(),
            car_number=str(data.get("car_number") or "").strip(),
            filler=str(data.get("filler") or "").strip(),
            load_state=str(data.get("load_state") or "").strip(),
            direction=str(data.get("Direction") or "").strip(),
            station=str(data.get("StationName") or "").strip(),
            time=str(data.get("time") or "").strip(),
            ply_file=str(data.get("ply_file") or "").strip(),
            length=length,
            width=width,
        )


@dataclass
class ModelNeed:
    """Модель, которой нет в базе сервера, и всё, что по ней набралось."""

    display_name: str
    suggested_key: str = ""
    #: Все снимки с этим именем модели — «затронуто снимков».
    shots: int = 0
    #: Сканы пустого кузова: из них и собирается модель.
    empty_shots: List[Shot] = field(default_factory=list)
    plates: List[str] = field(default_factory=list)
    first_day: str = ""
    last_day: str = ""
    #: Сканы, для которых HEAD подтвердил фотографию. Заполняется
    #: `WorklistScanner._attach_photos`, порядок — от свежих к старым.
    with_photo: List[Shot] = field(default_factory=list)
    #: Обмеры кузова по облаку со ВСЕХ снимков модели, не только пустых.
    #: Поле `body_dimensions` пишет реконструкция, а пустой скан до неё не
    #: доходит (его режет filler-gate) — меряя только по пустым, мы получили
    #: бы нули ровно у тех моделей, ради которых всё и затевалось.
    _lengths: List[float] = field(default_factory=list, repr=False)
    _widths: List[float] = field(default_factory=list, repr=False)
    #: Заполняется, если это та же модель под другим именем (см. `find_rename`).
    #: Такую строку генерировать не надо — надо поправить имя в реестре.
    rename: Optional[RenameHint] = None

    @property
    def ready(self) -> bool:
        """Есть всё, что нужно для генерации: пустой скан, фото и имя."""
        return bool(self.display_name and self.with_photo)

    @property
    def missing(self) -> str:
        """Чего не хватает — текстом, для колонки «почему скрыт»."""
        if self.ready:
            return ""
        if not self.empty_shots:
            return "нет снимка пустого кузова"
        return "нет topdown-фото"

    @property
    def best(self) -> Optional[Shot]:
        """Скан, с которого стоит собирать: свежий, пустой, с фотографией."""
        return self.with_photo[0] if self.with_photo else None

    def measurement(self) -> Tuple[float, float]:
        """
        Медианный обмер кузова (длина, ширина) по всем снимкам модели.

        Медиана, а не среднее: единичный скан иногда цепляет соседнюю машину
        или козырёк кабины, и такой выброс утащил бы среднее на полметра.
        (0, 0) — обмера нет ни в одной записи.
        """
        pick = lambda v: sorted(v)[len(v) // 2] if v else 0.0   # noqa: E731
        return pick(self._lengths), pick(self._widths)


@dataclass
class Worklist:
    """Результат обхода."""

    needs: List[ModelNeed] = field(default_factory=list)
    #: Сколько моделей уже заведено на сервере — для строки состояния.
    known_models: int = 0
    #: Сколько записей просмотрено.
    records: int = 0
    #: Записи, которые не удалось прочитать (сеть, битый JSON).
    failed: int = 0

    @staticmethod
    def _by_shots(items) -> List[ModelNeed]:
        return sorted(items, key=lambda n: (-n.shots, n.display_name.lower()))

    @property
    def ready(self) -> List[ModelNeed]:
        """
        Что действительно надо сгенерировать, по убыванию числа снимков.

        Переименования сюда не идут: кузов для них уже смоделирован, собирать
        его заново — потерянный день.
        """
        return self._by_shots(n for n in self.needs
                              if n.ready and n.rename is None)

    @property
    def renames(self) -> List[ModelNeed]:
        """Та же модель под другим именем — чинится правкой имени в реестре."""
        return self._by_shots(n for n in self.needs if n.rename is not None)

    @property
    def incomplete(self) -> List[ModelNeed]:
        """Без пустого скана или без фото — собрать не из чего."""
        return self._by_shots(n for n in self.needs
                              if not n.ready and n.rename is None)


# ---------------------------------------------------------------------------
# Обход
# ---------------------------------------------------------------------------


class WorklistScanner:
    """Обход сервера. Один экземпляр — один проход."""

    def __init__(self, host: str, port: int = 9999, timeout: float = 30.0,
                 cache_dir: str = CACHE_DIR):
        self.base = f"http://{host}:{int(port)}"
        self.timeout = float(timeout)
        self.cache_dir = cache_dir
        self.records_cache = os.path.join(cache_dir, "records.json")
        self.photos_cache = os.path.join(cache_dir, "photos.json")
        self.assets_dir = os.path.join(cache_dir, "assets")
        self._local = threading.local()
        self._progress: Optional[ProgressFn] = None
        self._cancelled: Optional[CancelFn] = None

    # ---- транспорт ---------------------------------------------------

    def _session(self) -> requests.Session:
        """
        Своя сессия на поток.

        `requests.Session` не потокобезопасна, а качаем мы в шестнадцать
        потоков. Зато на поток остаётся keep-alive, и восемь тысяч мелких
        запросов не превращаются в восемь тысяч TCP-рукопожатий.
        """
        s = getattr(self._local, "session", None)
        if s is None:
            s = requests.Session()
            self._local.session = s
        return s

    def _check(self) -> None:
        if self._cancelled is not None and self._cancelled():
            raise WorklistCancelled()

    def _say(self, text: str, fraction: float = -1.0) -> None:
        if self._progress is not None:
            self._progress(text, fraction)

    def _download_url(self, name: str) -> str:
        return f"{self.base}/download?file={urllib.parse.quote(name)}"

    # ---- перечисление имён -------------------------------------------

    def _list_suffix(self, suffix: str) -> List[str]:
        resp = self._session().post(
            f"{self.base}/list_obj_results",
            json={"limit": _LIST_LIMIT, "suffix": suffix},
            timeout=self.timeout)
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("status") != "success":
            raise RuntimeError(payload.get("error") or "list_obj_results failed")
        return [str(f.get("name") or "") for f in payload.get("files", [])
                if f.get("name")]

    def list_record_names(self) -> List[str]:
        """
        Имена всех `<base>.json` в `data/PLY_examples`.

        Обход в ширину: уровень целиком уходит в пул, и только насыщенные
        шарды (ответ ровно в `_LIST_LIMIT` — значит сервер обрезал) дробятся
        дальше, по одному символу влево. На ~8000 записей хватает двух
        уровней: имена оканчиваются hex-символами, и шарды выходят по сотне.
        """
        from concurrent.futures import ThreadPoolExecutor

        found: set = set()
        level = [".json"]
        depth = 0
        self._say("перечисление записей…", 0.0)
        with ThreadPoolExecutor(_FETCH_WORKERS) as pool:
            while level and depth <= _SHARD_MAX_DEPTH:
                self._check()
                results = list(pool.map(self._list_suffix, level))
                saturated = []
                for suffix, names in zip(level, results):
                    found.update(names)
                    if len(names) >= _LIST_LIMIT:
                        saturated.append(suffix)
                self._say(f"перечисление записей… {len(found)}")
                if depth == _SHARD_MAX_DEPTH and saturated:
                    print(f"[worklist] {len(saturated)} шард(ов) упёрлись в "
                          f"потолок на предельной глубине: часть записей "
                          f"могла не попасть в список")
                    break
                level = [ch + s for s in saturated for ch in _SHARD_CHARSET]
                depth += 1
        return sorted(found)

    # ---- чтение записей ----------------------------------------------

    @staticmethod
    def _load_json(path: str) -> Dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else {}
        except FileNotFoundError:
            return {}
        except Exception as exc:                          # noqa: BLE001
            print(f"[worklist] кэш {path} не прочитан ({exc}); начинаю с нуля")
            return {}

    def _save_json(self, path: str, data: Dict[str, Any]) -> None:
        """Запись через tmp+rename: оборванный проход не должен бить кэш."""
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False)
            os.replace(tmp, path)
        except Exception as exc:                          # noqa: BLE001
            print(f"[worklist] кэш {path} не сохранён: {exc}")

    def _fetch_record(self, name: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        try:
            resp = self._session().get(self._download_url(name),
                                       timeout=self.timeout)
            if resp.status_code != 200:
                return name, None
            return name, resp.json()
        except Exception:                                 # noqa: BLE001
            return name, None

    def load_records(self, names: Sequence[str]) -> Tuple[Dict[str, Any], int]:
        """
        Содержимое всех записей: из кэша, недостающее — с сервера.

        Запись неизменяема после обсчёта: `<base>.json` переписывается только
        при повторной загрузке того же снимка, а имя тогда остаётся прежним.
        Поэтому кэш по имени без проверки mtime корректен, и второй проход
        стоит ровно столько, сколько появилось новых снимков.
        """
        from concurrent.futures import ThreadPoolExecutor

        cache = self._load_json(self.records_cache)
        wanted = set(names)
        # Записи, которых на сервере больше нет, из кэша убираем: иначе он
        # растёт вечно и тащит в список давно удалённые снимки.
        cache = {k: v for k, v in cache.items() if k in wanted}
        missing = [n for n in names if n not in cache]
        failed = 0

        if missing:
            total = len(missing)
            done = 0
            self._say(f"чтение записей 0 / {total}", 0.0)
            with ThreadPoolExecutor(_FETCH_WORKERS) as pool:
                for name, data in pool.map(self._fetch_record, missing):
                    done += 1
                    if data is None:
                        failed += 1
                    else:
                        cache[name] = data
                    if done % 50 == 0 or done == total:
                        self._say(f"чтение записей {done} / {total}",
                                  done / float(total))
                        # Отмена проверяется здесь, а не в `_fetch_record`:
                        # рвать пул на полпути незачем, достаточно перестать
                        # ждать следующую пачку.
                        if self._cancelled is not None and self._cancelled():
                            pool.shutdown(wait=False, cancel_futures=True)
                            self._save_json(self.records_cache, cache)
                            raise WorklistCancelled()
            self._save_json(self.records_cache, cache)

        return cache, failed

    # ---- фотографии ---------------------------------------------------

    def _has_photo(self, shot: Shot) -> bool:
        """
        Есть ли у скана topdown-фото.

        HEAD, а не GET: сервер отдаёт на него Content-Length и 404 для
        отсутствующего файла, но не шлёт сами 400 КБ. На полусотне проверок
        это разница между секундой и полуминутой.
        """
        try:
            resp = self._session().head(self._download_url(shot.photo_file),
                                        timeout=self.timeout)
            return resp.status_code == 200
        except Exception:                                 # noqa: BLE001
            return False

    def _attach_photos(self, needs: List[ModelNeed], per_model: int) -> None:
        """
        Найти у каждой модели сканы с фотографией.

        Проверяются не все пустые сканы, а первые `per_model` от свежих: для
        галереи в диалоге больше и не нужно, а у ходовых моделей пустых
        сканов под сотню. Ответы складываются в кэш — снимок не отрастит
        фотографию задним числом, так что спрашивать второй раз незачем.
        """
        from concurrent.futures import ThreadPoolExecutor

        cache = self._load_json(self.photos_cache)
        jobs: List[Tuple[ModelNeed, Shot]] = []
        for need in needs:
            for shot in need.empty_shots[:per_model]:
                if shot.base in cache:
                    if cache[shot.base]:
                        need.with_photo.append(shot)
                else:
                    jobs.append((need, shot))

        if jobs:
            total = len(jobs)
            done = 0
            self._say(f"поиск фотографий 0 / {total}", 0.0)
            with ThreadPoolExecutor(_FETCH_WORKERS) as pool:
                for (need, shot), ok in zip(
                        jobs, pool.map(lambda j: self._has_photo(j[1]), jobs)):
                    done += 1
                    cache[shot.base] = bool(ok)
                    if ok:
                        need.with_photo.append(shot)
                    if done % 20 == 0 or done == total:
                        self._say(f"поиск фотографий {done} / {total}",
                                  done / float(total))
            self._save_json(self.photos_cache, cache)

        # Кэшированные сканы добавлялись раньше проверенных — возвращаем
        # порядок `empty_shots` (основная машина сверху, см. `_group`).
        for need in needs:
            order = {s.base: i for i, s in enumerate(need.empty_shots)}
            need.with_photo.sort(key=lambda s: order.get(s.base, 1 << 30))

    # ---- сборка -------------------------------------------------------

    def known_models(self) -> List[KnownModel]:
        """Модели, которые на сервере уже заведены, с размером проёма."""
        resp = self._session().post(f"{self.base}/get_models_config",
                                    json={}, timeout=self.timeout)
        resp.raise_for_status()
        payload = resp.json()
        if payload.get("status") != "success":
            raise RuntimeError(payload.get("error") or "get_models_config failed")
        cfg = payload.get("config") or {}
        out: List[KnownModel] = []
        for key, entry in cfg.items():
            if not isinstance(entry, dict) or not entry.get("model"):
                continue
            length, width = _rect_from_points(entry.get("points_3d"))
            out.append(KnownModel(key=str(key),
                                  display_name=str(entry["model"]),
                                  length=length, width=width))
        return out

    def scan(self, progress: Optional[ProgressFn] = None,
             cancelled: Optional[CancelFn] = None,
             photos_per_model: int = 6) -> Worklist:
        """Полный проход: имена → записи → группировка → фотографии."""
        self._progress = progress
        self._cancelled = cancelled
        try:
            self._say("опрос сервера…", 0.0)
            known = self.known_models()

            names = self.list_record_names()
            self._say(f"записей на сервере: {len(names)}")

            records, failed = self.load_records(names)

            self._say("разбор записей…", -1.0)
            needs = self._group(records, (k.display_name for k in known))

            # Переименования ищутся после группировки: нужен медианный обмер,
            # а он известен, только когда собраны все снимки модели.
            for need in needs:
                length, width = need.measurement()
                need.rename = find_rename(need.display_name, length, width,
                                          known)

            self._attach_photos(needs, photos_per_model)

            return Worklist(needs=needs, known_models=len(known),
                            records=len(records), failed=failed)
        finally:
            self._progress = None
            self._cancelled = None

    @staticmethod
    def _group(records: Dict[str, Any], known: Iterable[str]) -> List[ModelNeed]:
        """
        Сгруппировать снимки по имени модели, оставив отсутствующие в базе.

        «Unknown» и пустое имя отбрасываются сразу: это снимки, у которых
        оператор модель не выбрал вовсе (их режет model-gate на сервере), а
        собирать комплект под безымянный кузов всё равно не из чего — имя
        пришлось бы выдумать, и в реестр оно уедет мусором.
        """
        known_set = {normalize_model_name(k) for k in known}
        by_name: Dict[str, ModelNeed] = {}
        #: {имя модели: {госномер: число снимков}} — чья машина основная.
        plate_shots: Dict[str, Dict[str, int]] = {}

        for name, data in records.items():
            if not isinstance(data, dict):
                continue
            base = name[:-5] if name.endswith(".json") else name
            shot = Shot.from_json(base, data)
            if not shot.model or shot.model.lower() == _UNKNOWN_MODEL:
                continue
            if normalize_model_name(shot.model) in known_set:
                continue

            need = by_name.get(shot.model)
            if need is None:
                need = by_name[shot.model] = ModelNeed(
                    display_name=shot.model,
                    suggested_key=suggest_key(shot.model))
            need.shots += 1
            if shot.car_number:
                per = plate_shots.setdefault(shot.model, {})
                per[shot.car_number] = per.get(shot.car_number, 0) + 1
            if shot.length > 0:
                need._lengths.append(shot.length)
            if shot.width > 0:
                need._widths.append(shot.width)
            if shot.car_number and shot.car_number not in need.plates:
                need.plates.append(shot.car_number)
            day = shot.day
            if day:
                if not need.first_day or day < need.first_day:
                    need.first_day = day
                if day > need.last_day:
                    need.last_day = day
            # Пустой скан без .ply бесполезен: генератор читает именно облако.
            if shot.is_empty and shot.ply_file:
                need.empty_shots.append(shot)

        for need in by_name.values():
            need.plates.sort()
            # Сначала сканы машины, на которую приходится больше всего снимков
            # модели, внутри машины — свежие сверху.
            #
            # Под одним именем модели ездят разные кузова: у «Sitrak C7H-F 8x4
            # (высокий)» четыре машины с кузовом 7.65 м и одна с 6.2 м, и
            # самый свежий пустой скан оказался именно её — комплект вышел на
            # полтора метра короче, чем нужно 30 снимкам из 33. Геометрия
            # модели на сервере одна, поэтому собирать её надо по той машине,
            # которая даёт большинство снимков.
            weight = plate_shots.get(need.display_name, {})
            need.empty_shots.sort(
                key=lambda s: (weight.get(s.car_number, 0), s.base[-14:]),
                reverse=True)
        return list(by_name.values())

    # ---- файлы --------------------------------------------------------

    def fetch_asset(self, name: str,
                    progress: Optional[Callable[[int, int], bool]] = None
                    ) -> str:
        """
        Скачать файл записи в кэш и вернуть локальный путь.

        Уже скачанный файл не перекачивается: облака по пять мегабайт, а
        перебирать сканы одной модели в галерее приходится по многу раз.
        `progress(sent, total)` может вернуть False — тогда закачка рвётся и
        частичный файл удаляется.
        """
        os.makedirs(self.assets_dir, exist_ok=True)
        dest = os.path.join(self.assets_dir, name)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            return dest

        tmp = dest + ".part"
        resp = self._session().get(self._download_url(name),
                                   timeout=self.timeout, stream=True)
        if resp.status_code != 200:
            raise RuntimeError(f"сервер не отдал {name}: HTTP {resp.status_code}")
        total = int(resp.headers.get("Content-Length") or 0)
        sent = 0
        try:
            with open(tmp, "wb") as fh:
                for chunk in resp.iter_content(64 * 1024):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    sent += len(chunk)
                    if progress is not None and not progress(sent, total):
                        raise WorklistCancelled()
            os.replace(tmp, dest)
        except BaseException:
            try:
                os.remove(tmp)
            except OSError:
                pass
            raise
        return dest
