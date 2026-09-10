# -*- coding: utf-8 -*-
"""
Клиент реестра моделей photo-to-volume (HTTP-демон `model-registry`).

Здесь нет ни Qt, ни Panda3D: модуль обязан работать и из CLI, и из рабочего
потока UI одинаково. Всё, что касается диалога, лежит в
`src/ui/registry_dialog.py`.

Почему свой мультипарт, а не `requests_toolbelt`
------------------------------------------------
Комплект кузова — это сотни мегабайт (.obj наполнителя, .bam, текстуры), и
собирать тело запроса в памяти нельзя. `requests` без посторонних пакетов
делает именно это, а `requests-toolbelt` в окружении утилиты не стоит.
`_MultipartBody` ниже — файлоподобный объект, который отдаёт тело кусками
прямо с диска и знает свою длину: значит, уходит нормальный `Content-Length`,
а не chunked (его httplib на той стороне не ждёт). На каждом куске дёргается
колбэк прогресса; он же служит отменой — вернул False, и загрузка обрывается
`UploadCancelled`.
"""

from __future__ import annotations

import json
import mimetypes
import os
import uuid
from typing import (Any, Callable, Dict, Iterable, Iterator, List, Optional,
                    Tuple)

import requests

#: Роли файлов, которые понимает реестр: имя поля мультипарта. Как назвать
#: файл на диске, решает сервер, а не мы, — поэтому имя файла клиента не важно.
FILE_ROLES: Tuple[str, ...] = (
    "napolnitel_obj", "cuzov_obj", "cuzov_bam", "napolnitel_bam",
    "other_bam", "napolnitel_mtl", "other_obj", "body_obj", "full_obj",
)

#: Без этих двух файлов пайплайн на той стороне не поедет (сервер ответит 422).
REQUIRED_ROLES: Tuple[str, ...] = ("napolnitel_obj", "cuzov_obj")

#: Роли, из которых состоит комплект в нашем пайплайне: два обязательных OBJ
#: плюс debug-preview. Остальные роли API (`other_obj`, `body_obj`,
#: `full_obj`) — это исходники чужих раскладок; у наших комплектов таких
#: файлов не бывает, и в диалоге они показываются, только если файл реально
#: нашёлся на диске или его выбрали руками.
COMMON_ROLES: Tuple[str, ...] = (
    "napolnitel_obj", "cuzov_obj", "cuzov_bam", "napolnitel_bam",
    "other_bam", "napolnitel_mtl",
)

#: Человекочитаемые подписи ролей — их же показывает диалог.
ROLE_LABELS: Dict[str, str] = {
    "napolnitel_obj": "Наполнитель, .obj",
    "cuzov_obj":      "Кузов, .obj",
    "cuzov_bam":      "Кузов, .bam",
    "napolnitel_bam": "Наполнитель, .bam",
    "other_bam":      "Остальная машина, .bam",
    "napolnitel_mtl": "Материал наполнителя, .mtl",
    "other_obj":      "Остальная машина, .obj",
    "body_obj":       "Корпус, .obj",
    "full_obj":       "Машина целиком, .obj",
}

#: Зачем роль нужна — подсказка в диалоге.
ROLE_HINTS: Dict[str, str] = {
    "napolnitel_obj": "Обязателен: по нему считается объём груза",
    "cuzov_obj":      "Обязателен: по нему ставятся анкерные точки",
    "cuzov_bam":      "Нужен только для debug-preview на сервере",
    "napolnitel_bam": "Нужен только для debug-preview на сервере",
    "other_bam":      "Шасси и кабина в debug-preview",
    "napolnitel_mtl": "Материал к .obj наполнителя",
    "other_obj":      "Исходник, пайплайну не нужен",
    "body_obj":       "Исходник, пайплайну не нужен",
    "full_obj":       "Исходник, пайплайну не нужен",
}

#: Как сервер назовёт файл роли — показывается в колонке «Станет файлом».
ROLE_TARGET_SUFFIX: Dict[str, str] = {
    "napolnitel_obj": "-Napolnitel.obj",
    "cuzov_obj":      "-Cuzov.obj",
    "cuzov_bam":      "-Cuzov.bam",
    "napolnitel_bam": "-Napolnitel.bam",
    "other_bam":      "-Other.bam",
    "napolnitel_mtl": "-Napolnitel.mtl",
    "other_obj":      "-Other.obj",
    "body_obj":       "-Body.obj",
    "full_obj":       "-Full.obj",
}

#: Разрешённые символы ключа модели — те же, что проверяет сервер.
KEY_ALLOWED = ("abcdefghijklmnopqrstuvwxyz"
               "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
KEY_MAX_LEN = 64

#: Что делать с кодом ответа. Сервер присылает человекочитаемый `error`, но без
#: контекста «а что теперь» он оператору мало о чём говорит.
STATUS_HINTS: Dict[int, str] = {
    400: "Сервер не принял запрос: плохой ключ модели или поля meta.",
    401: "Реестр требует токен на запись. Пропишите его в config/"
         "tls_config.yaml, секция model_registry, поле token — или "
         "попросите администратора сервера выдать токен.",
    403: "Доступ запрещён: у токена нет прав на запись.",
    404: "На сервере нет такой модели.",
    413: "Файлы больше лимита сервера (--max-upload-mb, а также "
         "client_max_body_size в nginx).",
    415: "Сервер ждал multipart/form-data — похоже, запрос переписал прокси.",
    422: "Не хватает обязательного файла или файл не похож на OBJ.",
    500: "Ошибка на стороне сервера; изменения откатились в обоих деревьях. "
         "Смотрите лог демона.",
    502: "Отвечает прокси, а не реестр: nginx не достучался до демона "
         "model-registry. Проверьте, что демон запущен и что proxy_pass "
         "смотрит на его хост и порт.",
    503: "Реестр занят другой операцией (flock) — или прокси не дождался "
         "апстрима. Повторите позже.",
    504: "Прокси не дождался ответа реестра. Для больших загрузок нужны "
         "proxy_read_timeout / proxy_send_timeout по 600s.",
}

#: Признаки того, что ответ пришёл не от реестра, а от промежуточного прокси.
_HTML_MARKS = ("<html", "<!doctype html")


class ModelRegistryError(RuntimeError):
    """Ответ 4xx/5xx. `payload` — тело ошибки, как его прислал сервер."""

    def __init__(self, status: int, payload: Optional[Dict[str, Any]] = None):
        self.status = status
        self.payload = payload or {}
        self.error = str(self.payload.get("error") or "").strip()
        self.hint = str(self.payload.get("hint") or "").strip()
        super().__init__(self.message())

    def message(self) -> str:
        return f"HTTP {self.status}: {self.error}" if self.error \
            else f"HTTP {self.status}"

    def explain(self) -> str:
        """Текст для окна ошибки: что сказал сервер и что с этим делать."""
        bits = [self.message()]
        if self.hint:
            bits.append(self.hint)
        advice = STATUS_HINTS.get(self.status)
        if advice:
            bits.append(advice)
        return "\n\n".join(bits)


class UploadCancelled(RuntimeError):
    """Пользователь нажал «Отмена», пока тело запроса уходило на сервер."""


class RegistryConnectionError(RuntimeError):
    """Сеть: не достучались, таймаут, оборванный TLS."""


#: Колбэк прогресса: (отправлено_байт, всего_байт) -> продолжать ли.
ProgressFn = Callable[[int, int], bool]


def _escape_quotes(value: str) -> str:
    return value.replace('"', "%22")


class _MultipartBody:
    """
    Тело multipart/form-data, которое читается с диска кусками.

    Для `requests` это «поток»: есть `__iter__`, `read()` и `__len__`, поэтому
    длина уходит в `Content-Length`, а тело не собирается в память.
    """

    CHUNK = 256 * 1024

    def __init__(self, fields: Iterable[Tuple[str, Any]]):
        self.boundary = uuid.uuid4().hex
        self._parts: List[Tuple[bytes, Optional[str]]] = []
        self._tail = f"--{self.boundary}--\r\n".encode()
        total = 0

        for name, value in fields:
            if isinstance(value, tuple):                 # файл: (имя, путь)
                filename, path = value
                size = os.path.getsize(path)
                ctype = (mimetypes.guess_type(filename)[0]
                         or "application/octet-stream")
                head = (
                    f"--{self.boundary}\r\n"
                    f'Content-Disposition: form-data; name="{name}"; '
                    f'filename="{_escape_quotes(filename)}"\r\n'
                    f"Content-Type: {ctype}\r\n\r\n"
                ).encode()
                self._parts.append((head, path))
                total += len(head) + size + 2            # +2 — хвостовой CRLF
            else:                                        # текстовое поле
                head = (
                    f"--{self.boundary}\r\n"
                    f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                ).encode()
                data = str(value).encode("utf-8")
                self._parts.append((head + data + b"\r\n", None))
                total += len(head) + len(data) + 2

        self._total = total + len(self._tail)
        self._sent = 0
        self._progress: Optional[ProgressFn] = None
        self._iter: Optional[Iterator[bytes]] = None
        self._buffer = b""

    # -- сервис --------------------------------------------------------
    @property
    def content_type(self) -> str:
        return f"multipart/form-data; boundary={self.boundary}"

    def set_progress(self, fn: Optional[ProgressFn]) -> None:
        self._progress = fn

    def __len__(self) -> int:
        return self._total

    # -- собственно поток ----------------------------------------------
    def _chunks(self) -> Iterator[bytes]:
        for head, path in self._parts:
            yield head
            if path is None:
                continue
            with open(path, "rb") as fh:
                while True:
                    block = fh.read(self.CHUNK)
                    if not block:
                        break
                    yield block
            yield b"\r\n"
        yield self._tail

    def _tick(self, size: int) -> None:
        self._sent += size
        if self._progress is not None and not self._progress(self._sent,
                                                             self._total):
            raise UploadCancelled("загрузка отменена")

    def __iter__(self) -> Iterator[bytes]:
        for block in self._chunks():
            self._tick(len(block))
            yield block

    def read(self, amount: int = -1) -> bytes:
        if self._iter is None:
            self._iter = self._chunks()
        if amount is None or amount < 0:
            rest = [self._buffer]
            self._buffer = b""
            for block in self._iter:
                self._tick(len(block))
                rest.append(block)
            return b"".join(rest)

        while len(self._buffer) < amount:
            try:
                block = next(self._iter)
            except StopIteration:
                break
            self._tick(len(block))
            self._buffer += block
        out, self._buffer = self._buffer[:amount], self._buffer[amount:]
        return out


def validate_key(key: str) -> str:
    """Причина, по которой ключ не годится, или "" — если годится."""
    key = str(key or "")
    if not key:
        return "Ключ модели не может быть пустым"
    if len(key) > KEY_MAX_LEN:
        return f"Ключ длиннее {KEY_MAX_LEN} символов"
    bad = sorted({ch for ch in key if ch not in KEY_ALLOWED})
    if bad:
        return ("Недопустимые символы в ключе: "
                + " ".join(repr(ch) for ch in bad)
                + ". Разрешены латиница, цифры, точка, дефис, подчёркивание")
    return ""


#: Кириллица в ключ не проходит, а имена наборов у нас сплошь русские.
_TRANSLIT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
    "ж": "zh", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "h", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}


def suggest_key(name: str) -> str:
    """Ключ из имени набора: транслит, пробелы в дефис, остальное — прочь."""
    out: List[str] = []
    for ch in str(name or ""):
        low = ch.lower()
        if low in _TRANSLIT:
            piece = _TRANSLIT[low]
            out.append(piece.upper() if ch.isupper() else piece)
        elif ch in KEY_ALLOWED:
            out.append(ch)
        elif ch.isspace() or ch in "/\\":
            out.append("-")
    key = "".join(out).strip("-._")
    while "--" in key:
        key = key.replace("--", "-")
    return key[:KEY_MAX_LEN]


class ModelRegistry:
    """Тонкая обёртка над HTTP API реестра (см. MODEL_REGISTRY_API.md)."""

    def __init__(self, base_url: str, token: str = "",
                 timeout: float = 30.0, upload_timeout: float = 1800.0,
                 verify_tls: bool = True):
        self.base = str(base_url or "").rstrip("/")
        self.timeout = timeout
        self.upload_timeout = upload_timeout
        self.session = requests.Session()
        self.session.verify = bool(verify_tls)
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"

    def close(self) -> None:
        try:
            self.session.close()
        except Exception:
            pass

    # ---- чтение ------------------------------------------------------
    def health(self) -> Dict[str, Any]:
        return self._json("GET", "/health")

    def version(self) -> Dict[str, Any]:
        return self._json("GET", "/registry/version")

    def list_models(self) -> List[Dict[str, Any]]:
        return self._json("GET", "/models").get("models", [])

    def get_model(self, key: str) -> Optional[Dict[str, Any]]:
        """Карточка модели или None, если её на сервере нет."""
        try:
            return self._json("GET", f"/models/{key}").get("model")
        except ModelRegistryError as exc:
            if exc.status == 404:
                return None
            raise

    # ---- запись ------------------------------------------------------
    def upsert_model(self, key: str, meta: Dict[str, Any],
                     files: Dict[str, str],
                     textures: Iterable[str] = (),
                     web_files: Iterable[Any] = (),
                     mode: str = "replace",
                     progress: Optional[ProgressFn] = None) -> Dict[str, Any]:
        """
        Создать или заменить модель. `files` — {роль: путь к файлу}.

        `web_files` — то, что уедет в `data/models/<key>/WEB/`. Здесь важно
        имя, под которым файл приедет: glTF ссылается на буфер и текстуры
        относительными путями (`<стем>.bin`, `<стем>/tex.jpg`), и если
        отправить одни basename'ы, вьюер получит битые ссылки. Поэтому
        элементом может быть либо путь, либо пара `(имя на сервере, путь)`.

        HTTP 200/201 ещё не значит, что пайплайн с моделью поедет: смотрите
        `result["model"]["ready"]`, `result["warnings"]` и `ignored_parts`.
        """
        for role in files:
            if role not in FILE_ROLES:
                raise ValueError(f"неизвестная роль файла: {role!r}")
        for role, path in files.items():
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"{ROLE_LABELS.get(role, role)}: файла нет — {path}")

        fields: List[Tuple[str, Any]] = [
            ("meta", json.dumps(meta, ensure_ascii=False)),
            ("mode", mode),
        ]
        for role, path in files.items():
            fields.append((role, (os.path.basename(path), path)))
        for path in textures:
            fields.append(("texture", (os.path.basename(path), path)))
        for item in web_files:
            name, path = item if isinstance(item, (tuple, list)) \
                else (os.path.basename(item), item)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"WEB: файла нет — {path}")
            fields.append(("web", (str(name).replace("\\", "/"), path)))

        body = _MultipartBody(fields)
        body.set_progress(progress)
        return self._request("POST", f"/models/{key}", data=body,
                             headers={"Content-Type": body.content_type},
                             timeout=self.upload_timeout)

    def delete_model(self, key: str, purge: bool = False,
                     keep_camera_preset: bool = False) -> Dict[str, Any]:
        params: Dict[str, str] = {}
        if purge:
            params["purge"] = "1"
        if keep_camera_preset:
            params["keep_camera_preset"] = "1"
        return self._json("DELETE", f"/models/{key}", params=params or None)

    # ---- внутреннее --------------------------------------------------
    def _json(self, method: str, path: str, **kw) -> Dict[str, Any]:
        kw.setdefault("timeout", self.timeout)
        return self._request(method, path, **kw)

    def _request(self, method: str, path: str, **kw) -> Dict[str, Any]:
        if not self.base:
            raise RegistryConnectionError("не задан адрес реестра моделей")
        url = f"{self.base}{path}"
        try:
            resp = self.session.request(method, url, **kw)
        except UploadCancelled:
            raise
        except requests.exceptions.SSLError as exc:
            raise RegistryConnectionError(f"TLS не сошёлся с {url}: {exc}")
        except requests.exceptions.ConnectTimeout:
            # Соединение даже не установилось: снаружи так выглядит фильтр —
            # SYN дропается вместо отказа. Живой, но не проброшенный наружу
            # реестр даёт ровно эту картину, поэтому подсказываем сразу.
            raise RegistryConnectionError(
                f"порт реестра недоступен: {url} не принял соединение за "
                f"{self.timeout:.0f} с. Похоже, порт закрыт снаружи — "
                "проверьте проброс порта или выставите реестр через nginx и "
                "укажите его адрес в config/tls_config.yaml (model_registry: "
                "url:).")
        except requests.exceptions.Timeout:
            raise RegistryConnectionError(f"сервер {url} не ответил вовремя")
        except requests.exceptions.RequestException as exc:
            raise RegistryConnectionError(f"нет связи с {url}: {exc}")
        return self._parse(resp)

    @staticmethod
    def _parse(resp: requests.Response) -> Dict[str, Any]:
        try:
            payload = resp.json()
        except ValueError:
            text = (resp.text or "").strip()
            if any(mark in text[:200].lower() for mark in _HTML_MARKS):
                # Страница ошибки прокси. Показывать оператору разметку
                # бессмысленно — важно, что ответил не реестр.
                title = ""
                low = text.lower()
                start = low.find("<title>")
                if start >= 0:
                    end = low.find("</title>", start)
                    title = text[start + 7:end if end > 0 else None].strip()
                payload = {"error": "ответ не от реестра, а от прокси"
                                    + (f": {title}" if title else "")}
            else:
                payload = {"error": text[:500]}
        if not isinstance(payload, dict):
            payload = {"models": payload}
        if resp.status_code >= 400:
            raise ModelRegistryError(resp.status_code, payload)
        payload.setdefault("_status_code", resp.status_code)
        return payload
