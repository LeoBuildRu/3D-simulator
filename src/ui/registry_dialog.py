# -*- coding: utf-8 -*-
"""
Диалог загрузки/замены модели в реестре photo-to-volume.

Открывается из списка кузовов (ПКМ по строке -> «Загрузить на сервер…»).
Здесь только интерфейс: протокол живёт в `src/registry/client.py`, а сбор
файлов и метаданных — в `src/registry/payload.py`.

Три вещи, ради которых диалог существует
----------------------------------------
1. **Прогресс.** Комплект кузова — сотни мегабайт, и «программа висит» на
   минуту без единого признака жизни неприемлемо. Тело запроса уходит потоком,
   и полоса показывает реальные проценты, скорость и остаток времени.
2. **Вопрос о замене.** Загрузка по существующему ключу молча затирает модель
   на сервере. Поэтому перед отправкой карточка ключа перечитывается с
   сервера, и если модель есть — показывается, что именно будет заменено
   (ревизия, дата, файлы), с явным подтверждением.
3. **Честный итог.** `HTTP 200` у реестра значит «файлы записаны», а не
   «пайплайн поедет»: за это отвечает `model.ready`. Диалог показывает
   `ready`, `problems`, `warnings` и `ignored_parts`, а не только «готово».
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable, Dict, List, Optional

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox,
                             QDoubleSpinBox, QFileDialog, QFormLayout, QFrame,
                             QGridLayout, QGroupBox, QHBoxLayout, QLabel,
                             QLineEdit, QMessageBox, QPlainTextEdit,
                             QProgressBar, QPushButton, QRadioButton,
                             QScrollArea, QSizePolicy, QToolButton,
                             QVBoxLayout, QWidget)

from src.registry.client import (COMMON_ROLES, FILE_ROLES, REQUIRED_ROLES,
                                 ROLE_HINTS,
                                 ROLE_LABELS, ROLE_TARGET_SUFFIX,
                                 ModelRegistry, ModelRegistryError,
                                 RegistryConnectionError, UploadCancelled,
                                 validate_key)
from src.registry.payload import bam_to_obj, build_upload_plan, meta_from_form
from src.registry.settings import RegistryEndpoint, resolve_registry
from src.ui.ui_theme import (COLOR_ACCENT, COLOR_DANGER, COLOR_HAIRLINE,
                             COLOR_TEXT_MUTED, COLOR_WARN, FONT_MONO,
                             apply_theme)

_EM_DASH = "—"


def _fmt_size(size: float) -> str:
    if size <= 0:
        return "0 Б"
    if size >= 1024 ** 3:
        return f"{size / 1024 ** 3:.2f} ГБ"
    if size >= 1024 ** 2:
        return f"{size / 1024 ** 2:.1f} МБ"
    if size >= 1024:
        return f"{size / 1024:.0f} КБ"
    return f"{size:.0f} Б"


def _fmt_eta(seconds: float) -> str:
    if seconds <= 0 or seconds > 24 * 3600:
        return _EM_DASH
    if seconds < 60:
        return f"{seconds:.0f} с"
    return f"{int(seconds // 60)} мин {int(seconds % 60):02d} с"


def _mirror_note(model: Dict[str, Any]) -> str:
    """
    Что с зеркалом модели, или "" — если всё хорошо либо зеркала нет.

    На проде `data/`+`config/` живут в двух копиях: из первой читает
    CLI-пайплайн, из второй — TLS-сервер 9999, который отдаёт утилите список
    кузовов. Реестр раскладывает модель в обе и возвращает результат в поле
    `mirror`. Если модель не доехала до второго дерева, обработка фото
    работает, а в списке кузовов набора просто нет — симптом настолько
    неочевидный, что о нём надо сказать прямо.
    """
    mirror = model.get("mirror")
    if not isinstance(mirror, dict) or mirror.get("ok"):
        return ""
    bits = []
    if mirror.get("in_config") is False:
        bits.append("нет записи в конфиге второго дерева")
    if mirror.get("target_model_present") is False:
        bits.append("не доехал файл наполнителя")
    detail = ", ".join(bits) or "подробностей сервер не дал"
    return ("модель не зеркалирована во второе дерево "
            f"({mirror.get('config_dir') or 'путь не указан'}): {detail}. "
            "В списке кузовов утилиты она не появится")


def _elide(path: str, keep: int = 52) -> str:
    if len(path) <= keep:
        return path
    return "…" + path[-(keep - 1):]


class _Job(QThread):
    """
    Один сетевой вызов в фоне.

    Диалог не должен замирать ни на проверке связи (сервер может быть
    недоступен и отвалиться только по таймауту), ни на загрузке. Работа
    задаётся колбэком, который получает сам поток — чтобы уметь спросить
    `is_cancelled()` и слать прогресс.
    """

    progress = pyqtSignal(int, int)          # отправлено, всего
    message = pyqtSignal(str)                # строка в журнал
    done = pyqtSignal(object)                # результат колбэка
    failed = pyqtSignal(str, str)            # заголовок, подробности
    cancelled = pyqtSignal()

    def __init__(self, work: Callable[["_Job"], Any], parent=None):
        super().__init__(parent)
        self._work = work
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def is_cancelled(self) -> bool:
        return self._cancel

    def on_progress(self, sent: int, total: int) -> bool:
        """Колбэк для `_MultipartBody`: он же и кнопка отмены."""
        self.progress.emit(sent, total)
        return not self._cancel

    def run(self) -> None:                               # noqa: D102 (Qt API)
        try:
            result = self._work(self)
        except UploadCancelled:
            self.cancelled.emit()
            return
        except ModelRegistryError as exc:
            self.failed.emit("Сервер отклонил запрос", exc.explain())
            return
        except RegistryConnectionError as exc:
            self.failed.emit("Нет связи с реестром", str(exc))
            return
        except FileNotFoundError as exc:
            self.failed.emit("Файл не найден", str(exc))
            return
        except Exception as exc:                          # noqa: BLE001
            import traceback
            traceback.print_exc()
            self.failed.emit("Непредвиденная ошибка",
                             f"{type(exc).__name__}: {exc}")
            return
        if self._cancel:
            self.cancelled.emit()
            return
        self.done.emit(result)


class _FileRow(QWidget):
    """Строка одной роли файла: что отправляем и во что это превратится."""

    changed = pyqtSignal()

    def __init__(self, role: str, parent=None):
        super().__init__(parent)
        self.role = role
        self.path = ""
        self._convert_from = ""

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        required = role in REQUIRED_ROLES
        title = ROLE_LABELS.get(role, role)
        self.label = QLabel(title + (" *" if required else ""))
        self.label.setMinimumWidth(168)
        self.label.setToolTip(ROLE_HINTS.get(role, ""))
        if required:
            self.label.setStyleSheet(f"color: {COLOR_ACCENT};")
        lay.addWidget(self.label)

        self.value = QLabel(_EM_DASH)
        self.value.setStyleSheet(f"color: {COLOR_TEXT_MUTED};"
                                 f" font-family: {FONT_MONO}; font-size: 11px;")
        self.value.setSizePolicy(QSizePolicy.Policy.Ignored,
                                 QSizePolicy.Policy.Preferred)
        lay.addWidget(self.value, 1)

        self.btn_convert = QToolButton()
        self.btn_convert.setText("Собрать из .bam")
        self.btn_convert.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_convert.setVisible(False)
        lay.addWidget(self.btn_convert)

        self.btn_pick = QToolButton()
        self.btn_pick.setText("Обзор…")
        self.btn_pick.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_pick.clicked.connect(self._pick)
        lay.addWidget(self.btn_pick)

        self.btn_clear = QToolButton()
        self.btn_clear.setText("✕")
        self.btn_clear.setToolTip("Не отправлять этот файл")
        self.btn_clear.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_clear.clicked.connect(lambda: self.set_path(""))
        lay.addWidget(self.btn_clear)

    def set_convert_source(self, bam_path: str) -> None:
        """Показать кнопку конвертации: OBJ этой роли можно получить из .bam."""
        self._convert_from = bam_path
        self.btn_convert.setVisible(bool(bam_path) and not self.path)
        if bam_path:
            self.btn_convert.setToolTip(
                f"Выгрузить геометрию из {os.path.basename(bam_path)} в OBJ "
                "средствами Panda3D — координаты совпадут с наполнителем")

    @property
    def convert_source(self) -> str:
        return self._convert_from

    def set_path(self, path: str) -> None:
        self.path = str(path or "")
        if self.path and os.path.isfile(self.path):
            size = os.path.getsize(self.path)
            self.value.setText(f"{os.path.basename(self.path)}  ·  "
                               f"{_fmt_size(size)}")
            self.value.setToolTip(f"{self.path}\n\nна сервере станет: "
                                  f"<ключ>{ROLE_TARGET_SUFFIX.get(self.role, '')}")
            self.value.setStyleSheet(f"color: {COLOR_ACCENT};"
                                     f" font-family: {FONT_MONO};"
                                     f" font-size: 11px;")
        else:
            self.path = ""
            missing = self.role in REQUIRED_ROLES
            self.value.setText("не выбран" if missing else _EM_DASH)
            self.value.setToolTip("")
            self.value.setStyleSheet(
                f"color: {COLOR_DANGER if missing else COLOR_TEXT_MUTED};"
                f" font-family: {FONT_MONO}; font-size: 11px;")
        self.btn_clear.setEnabled(bool(self.path))
        self.btn_convert.setVisible(bool(self._convert_from) and not self.path)
        self.changed.emit()

    def size(self) -> int:
        try:
            return os.path.getsize(self.path) if self.path else 0
        except OSError:
            return 0

    def _pick(self) -> None:
        suffix = ROLE_TARGET_SUFFIX.get(self.role, "")
        ext = os.path.splitext(suffix)[1] or ""
        mask = f"Файлы {ext} (*{ext});;Все файлы (*.*)" if ext \
            else "Все файлы (*.*)"
        start = os.path.dirname(self.path) if self.path else ""
        path, _ = QFileDialog.getOpenFileName(
            self, f"{ROLE_LABELS.get(self.role, self.role)}", start, mask)
        if path:
            self.set_path(path)


class ModelUploadDialog(QDialog):
    """
    Загрузка набора кузова в реестр моделей.

    `info` — `ModelSetInfo` выбранной строки списка (может быть None, тогда
    все поля заполняются вручную). `camera_provider` — колбэк, отдающий
    текущее положение камеры сцены: пресет камеры обязателен для новой
    модели, а взять его удобнее всего прямо из вида.
    """

    def __init__(self, parent=None, info: Any = None,
                 camera_provider: Optional[Callable[[], Optional[dict]]] = None):
        super().__init__(parent)
        apply_theme(self)
        self.setWindowTitle("Модель на сервере")
        self.setModal(True)
        self.setMinimumSize(760, 640)

        self._info = info
        self._camera_provider = camera_provider
        self._endpoint: RegistryEndpoint = resolve_registry()
        self._plan = build_upload_plan(info) if info is not None else None
        self._job: Optional[_Job] = None
        #: Карточка модели с сервера (или None, если её там нет). `False` —
        #: «ещё не спрашивали»: это разные вещи, и вопрос о замене задаётся
        #: только когда мы точно знаем, что модель там есть.
        self._remote: Any = False
        self._textures: List[str] = list(self._plan.textures) if self._plan \
            else []
        self._started_at = 0.0
        self._uploaded = False
        self._uploaded_key = ""

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(10)

        root.addWidget(self._build_server())
        root.addWidget(self._build_identity())

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        body = QWidget()
        body_lay = QVBoxLayout(body)
        body_lay.setContentsMargins(0, 0, 0, 0)
        body_lay.setSpacing(10)
        body_lay.addWidget(self._build_files())
        body_lay.addWidget(self._build_meta())
        body_lay.addStretch(1)
        scroll.setWidget(body)
        root.addWidget(scroll, 1)

        root.addWidget(self._build_progress())

        self.buttons = QDialogButtonBox()
        self.btn_upload = self.buttons.addButton(
            "Загрузить", QDialogButtonBox.ButtonRole.AcceptRole)
        self.btn_close = self.buttons.addButton(
            "Закрыть", QDialogButtonBox.ButtonRole.RejectRole)
        self.btn_upload.clicked.connect(self._start_upload)
        self.btn_close.clicked.connect(self.reject)
        root.addWidget(self.buttons)

        self._fill_from_plan()
        self._sync_state()
        if self._endpoint.ok:
            self._probe(auto=True)

    # ---- секции ------------------------------------------------------
    @staticmethod
    def _group(title: str) -> QGroupBox:
        box = QGroupBox(title)
        box.setStyleSheet(
            f"QGroupBox {{ border: 1px solid {COLOR_HAIRLINE};"
            f" border-radius: 8px; margin-top: 8px; padding-top: 10px; }}"
            f"QGroupBox::title {{ subcontrol-origin: margin; left: 10px;"
            f" padding: 0 4px; color: {COLOR_TEXT_MUTED}; }}")
        return box

    def _build_server(self) -> QWidget:
        """
        Адрес реестра — не настройка, а следствие выбранного сервера.

        Вводить тут нечего: реестр стоит на том же хосте, что и активный
        TLS-сервер из `config/tls_config.yaml`, и адрес складывается сам
        (см. `registry.settings.resolve_registry`). Панель только показывает,
        куда именно пойдёт загрузка и откуда этот адрес взялся, — чтобы при
        промахе было понятно, какой файл править.
        """
        box = self._group("Реестр моделей")
        lay = QGridLayout(box)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setHorizontalSpacing(8)
        lay.setVerticalSpacing(4)

        endpoint = self._endpoint
        self.lbl_url = QLabel(endpoint.url or "адрес не определён")
        self.lbl_url.setStyleSheet(
            f"color: {COLOR_ACCENT if endpoint.ok else COLOR_DANGER};"
            f" font-family: {FONT_MONO}; font-size: 12px;")
        self.lbl_url.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.lbl_url, 0, 0)

        self.btn_probe = QPushButton("Проверить связь")
        self.btn_probe.clicked.connect(lambda: self._probe(auto=False))
        self.btn_probe.setEnabled(endpoint.ok)
        lay.addWidget(self.btn_probe, 0, 1)
        lay.setColumnStretch(0, 1)

        source = QLabel(endpoint.source or endpoint.error)
        source.setWordWrap(True)
        source.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 10px;")
        lay.addWidget(source, 1, 0, 1, 2)

        self.lbl_server = QLabel("связь не проверялась")
        self.lbl_server.setWordWrap(True)
        self.lbl_server.setStyleSheet(
            f"color: {COLOR_TEXT_MUTED}; font-size: 11px;")
        lay.addWidget(self.lbl_server, 2, 0, 1, 2)
        return box

    def _build_identity(self) -> QWidget:
        box = self._group("Модель")
        lay = QGridLayout(box)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setHorizontalSpacing(8)

        self.ed_key = QLineEdit()
        self.ed_key.setToolTip(
            "Ключ модели: имя каталога на сервере и ключ в "
            "models_geometry_config.json.\n"
            "Разрешены латиница, цифры, точка, дефис, подчёркивание.\n"
            "Именно его надо передавать в meta.model при обработке фото.")
        self.ed_key.textChanged.connect(self._on_key_changed)
        lay.addWidget(QLabel("Ключ"), 0, 0)
        lay.addWidget(self.ed_key, 0, 1)

        self.ed_display = QLineEdit()
        self.ed_display.setToolTip("Человекочитаемое имя (поле model в конфиге)")
        lay.addWidget(QLabel("Название"), 0, 2)
        lay.addWidget(self.ed_display, 0, 3)

        self.rb_replace = QRadioButton("Полная замена")
        self.rb_replace.setToolTip(
            "replace: каталог модели собирается заново — чего не прислали, "
            "того на сервере не будет.")
        self.rb_patch = QRadioButton("Дополнить")
        self.rb_patch.setToolTip(
            "patch: недостающие файлы останутся от прежней версии, а в конфиге "
            "обновятся только заполненные поля. Годится, чтобы поправить один "
            "max_volume или заменить одну .bam.")
        self.rb_replace.setChecked(True)
        self.rb_replace.toggled.connect(lambda _: self._sync_state())
        mode_row = QHBoxLayout()
        mode_row.setSpacing(10)
        mode_row.addWidget(QLabel("Режим"))
        mode_row.addWidget(self.rb_replace)
        mode_row.addWidget(self.rb_patch)
        mode_row.addStretch(1)
        holder = QWidget()
        holder.setLayout(mode_row)
        lay.addWidget(holder, 1, 0, 1, 2)

        self.lbl_key = QLabel("")
        self.lbl_key.setWordWrap(True)
        self.lbl_key.setStyleSheet(f"color: {COLOR_TEXT_MUTED};"
                                   f" font-size: 11px;")
        lay.addWidget(self.lbl_key, 1, 2, 1, 2)
        return box

    def _build_files(self) -> QWidget:
        box = self._group("Файлы комплекта")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(4)

        # Показываем роли нашего комплекта. Остальные роли API
        # (`other_obj`, `body_obj`, `full_obj`) — исходники чужих раскладок,
        # у комплектов генератора таких файлов не бывает: строка «Корпус,
        # .obj», которую нечем заполнить, только сбивает с толку. Они
        # разворачиваются кнопкой ниже — на случай ручной сборки набора.
        self.rows: Dict[str, _FileRow] = {}
        for role in FILE_ROLES:
            row = _FileRow(role)
            row.changed.connect(self._sync_state)
            row.btn_convert.clicked.connect(
                lambda _=False, r=role: self._convert_role(r))
            self.rows[role] = row
            row.setVisible(role in COMMON_ROLES)
            lay.addWidget(row)

        self.btn_more_roles = QToolButton()
        self.btn_more_roles.setText("Ещё роли реестра…")
        self.btn_more_roles.setToolTip(
            "Исходники (-Other.obj, -Body.obj, -Full.obj): пайплайн их не "
            "читает, у комплектов генератора их нет — но реестр принимает.")
        self.btn_more_roles.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_more_roles.clicked.connect(self._show_rare_roles)
        rare_row = QHBoxLayout()
        rare_row.addWidget(self.btn_more_roles)
        rare_row.addStretch(1)
        lay.addLayout(rare_row)

        tex_row = QHBoxLayout()
        tex_row.setSpacing(6)
        self.chk_textures = QCheckBox("Текстуры машины")
        self.chk_textures.setToolTip(
            "Уедут в data/textures/<textures_dir>/ на сервере — их берёт "
            "debug-preview и веб-вьюер; пайплайну расчёта объёма они не нужны.")
        self.chk_textures.toggled.connect(self._sync_state)
        tex_row.addWidget(self.chk_textures)
        self.lbl_textures = QLabel("")
        self.lbl_textures.setStyleSheet(f"color: {COLOR_TEXT_MUTED};"
                                        f" font-size: 11px;")
        tex_row.addWidget(self.lbl_textures, 1)
        btn_tex = QToolButton()
        btn_tex.setText("Выбрать…")
        btn_tex.clicked.connect(self._pick_textures)
        tex_row.addWidget(btn_tex)
        lay.addLayout(tex_row)

        web_row = QHBoxLayout()
        web_row.setSpacing(6)
        self.chk_web = QCheckBox("Комплект веб-вьюера (glTF)")
        self.chk_web.setToolTip(
            "Уедет в data/models/<ключ>/WEB/ — оттуда его тянет веб-вьюер.\n"
            "Это .gltf, его .bin и ужатые в JPEG карты; относительные пути "
            "сохраняются, иначе ссылки внутри glTF рвутся.")
        self.chk_web.toggled.connect(self._sync_state)
        web_row.addWidget(self.chk_web)
        self.lbl_web = QLabel("")
        self.lbl_web.setStyleSheet(f"color: {COLOR_TEXT_MUTED};"
                                   f" font-size: 11px;")
        self.lbl_web.setWordWrap(True)
        web_row.addWidget(self.lbl_web, 1)
        lay.addLayout(web_row)

        self.lbl_total = QLabel("")
        self.lbl_total.setStyleSheet(f"color: {COLOR_TEXT_MUTED};"
                                     f" font-size: 11px;")
        lay.addWidget(self.lbl_total)
        return box

    def _build_meta(self) -> QWidget:
        box = self._group("Метаданные (meta)")
        lay = QFormLayout(box)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(6)

        self.sp_volume = QDoubleSpinBox()
        self.sp_volume.setRange(0.0, 1000.0)
        self.sp_volume.setDecimals(3)
        self.sp_volume.setSuffix(" м³")
        self.sp_volume.setSpecialValueText("не задан")
        self.sp_volume.setToolTip(
            "Полный объём кузова. Пайплайн его не читает — на него делит "
            "посчитанный объём наше приложение, показывая процент загрузки.")
        lay.addRow("max_volume", self.sp_volume)

        self.sp_ground = QDoubleSpinBox()
        self.sp_ground.setRange(-100.0, 100.0)
        self.sp_ground.setDecimals(4)
        self.sp_ground.setSuffix(" м")
        self.sp_ground.setToolTip("Уровень «пола» кузова в координатах модели")
        lay.addRow("ground_plane", self.sp_ground)

        self.ed_points = QPlainTextEdit()
        self.ed_points.setFixedHeight(64)
        self.ed_points.setFont(QFont(FONT_MONO.split(",")[0].strip("' "), 9))
        self.ed_points.setToolTip(
            "Четыре точки [x, y, z] — прямоугольник верхней кромки кузова.\n"
            "C++-пайплайн их не читает, но конфиг без них сервер не примет "
            "в режиме полной замены.")
        lay.addRow("points_3d", self.ed_points)
        self.lbl_points = QLabel("")
        self.lbl_points.setStyleSheet(f"color: {COLOR_WARN}; font-size: 11px;")
        self.lbl_points.setWordWrap(True)
        lay.addRow("", self.lbl_points)

        cam = QHBoxLayout()
        cam.setSpacing(4)
        self.cam_fields: List[QDoubleSpinBox] = []
        for title, rng, dec in (("x", 200.0, 3), ("y", 200.0, 3),
                                ("z", 200.0, 3), ("h", 360.0, 2),
                                ("p", 360.0, 2), ("r", 360.0, 2),
                                ("fov", 180.0, 1)):
            sp = QDoubleSpinBox()
            sp.setRange(-rng, rng)
            sp.setDecimals(dec)
            sp.setFixedWidth(78)
            sp.setToolTip({"x": "позиция камеры X", "y": "позиция камеры Y",
                           "z": "позиция камеры Z", "h": "heading, °",
                           "p": "pitch, °", "r": "roll, °",
                           "fov": "горизонтальный угол обзора, °"}[title])
            lbl = QLabel(title)
            lbl.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 10px;")
            cam.addWidget(lbl)
            cam.addWidget(sp)
            self.cam_fields.append(sp)
        self.chk_camera = QCheckBox("задан")
        self.chk_camera.setToolTip(
            "Без пресета камеры обработка фото падает с "
            "«camera preset lookup failed».\n"
            "Если модель уже есть на сервере, галочку можно снять — старый "
            "пресет останется.")
        self.chk_camera.setChecked(True)
        cam.insertWidget(0, self.chk_camera)
        self.btn_cam_scene = QToolButton()
        self.btn_cam_scene.setText("Из сцены")
        self.btn_cam_scene.setToolTip(
            "Взять текущее положение камеры вида как пресет съёмки")
        self.btn_cam_scene.clicked.connect(self._camera_from_scene)
        self.btn_cam_scene.setEnabled(self._camera_provider is not None)
        cam.addWidget(self.btn_cam_scene)
        cam.addStretch(1)
        cam_holder = QWidget()
        cam_holder.setLayout(cam)
        lay.addRow("camera", cam_holder)
        return box

    def _build_progress(self) -> QWidget:
        box = QFrame()
        box.setStyleSheet(f"QFrame {{ border: 1px solid {COLOR_HAIRLINE};"
                          f" border-radius: 8px; }}")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(12, 8, 12, 10)
        lay.setSpacing(6)

        # Почему кнопка недоступна. Раньше это жило в подсказке кнопки — то
        # есть было не видно вовсе: пользователь упирался в серую «Загрузить»
        # и не знал, что именно доложить. Причина должна быть на экране.
        self.lbl_blockers = QLabel("")
        self.lbl_blockers.setWordWrap(True)
        self.lbl_blockers.setStyleSheet(f"color: {COLOR_WARN};"
                                        f" font-size: 11px;")
        lay.addWidget(self.lbl_blockers)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFormat("%p%")
        self.progress.setVisible(False)
        lay.addWidget(self.progress)

        self.lbl_speed = QLabel("")
        self.lbl_speed.setStyleSheet(f"color: {COLOR_TEXT_MUTED};"
                                     f" font-family: {FONT_MONO};"
                                     f" font-size: 11px;")
        self.lbl_speed.setVisible(False)
        lay.addWidget(self.lbl_speed)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(96)
        self.log.setStyleSheet(f"font-family: {FONT_MONO}; font-size: 11px;")
        lay.addWidget(self.log)
        return box

    # ---- заполнение --------------------------------------------------
    def _fill_from_plan(self) -> None:
        plan = self._plan
        if plan is None:
            self.chk_camera.setChecked(False)
            return

        self.ed_key.setText(plan.key)
        self.ed_display.setText(plan.display_name)
        for role, path in plan.roles.items():
            if role in self.rows:
                self.rows[role].set_path(path)
        for role, bam in plan.convertible.items():
            if role in self.rows:
                self.rows[role].set_convert_source(bam)

        self.chk_textures.setChecked(False)
        self._sync_textures_label()
        self._sync_web_label()
        # Роль, для которой файл всё-таки нашёлся, прятать незачем.
        for role, row in self.rows.items():
            if row.path:
                row.setVisible(True)

        meta = plan.meta
        self.sp_volume.setValue(float(meta.get("max_volume") or 0.0))
        self.sp_ground.setValue(float(meta.get("ground_plane") or 0.0))
        points = meta.get("points_3d")
        if points:
            self.ed_points.setPlainText(
                json.dumps(points, ensure_ascii=False))
        if "points_3d" in plan.guessed:
            self.lbl_points.setText(
                "посчитано по габаритам комплекта — проверьте перед загрузкой")
        camera = meta.get("camera")
        if camera:
            values = list(camera.get("pos", [0, 0, 0])) \
                + list(camera.get("hpr", [0, 0, 0])) \
                + [camera.get("fov", 87.0)]
            for sp, value in zip(self.cam_fields, values):
                sp.setValue(float(value or 0.0))
        else:
            self.cam_fields[-1].setValue(87.0)
            self.chk_camera.setChecked(self._camera_provider is None)

        for note in plan.notes:
            self._log(note)

    def _show_rare_roles(self) -> None:
        for role, row in self.rows.items():
            if role not in COMMON_ROLES:
                row.setVisible(True)
        self.btn_more_roles.setVisible(False)

    def _sync_web_label(self) -> None:
        plan = self._plan
        files = list(plan.web_files) if plan else []
        blocked = plan.web_blocked if plan else ""
        if blocked:
            self.chk_web.setEnabled(False)
            self.chk_web.setChecked(False)
            self.lbl_web.setText(blocked)
            self.lbl_web.setStyleSheet(f"color: {COLOR_WARN}; font-size: 11px;")
            return
        if not files:
            self.chk_web.setEnabled(False)
            self.lbl_web.setText("рядом с комплектом нет .gltf")
            return
        size = sum(os.path.getsize(path) for _name, path in files
                   if os.path.isfile(path))
        self.chk_web.setEnabled(True)
        # Веб-вьюер — штатная часть комплекта генератора, и файлы лёгкие
        # (JPEG-копии карт), поэтому по умолчанию отправляем.
        self.chk_web.setChecked(True)
        self.lbl_web.setText(f"{len(files)} файл(ов), {_fmt_size(size)}")
        self.lbl_web.setToolTip("\n".join(name for name, _ in files[:40]))

    def _sync_textures_label(self) -> None:
        if not self._textures:
            self.lbl_textures.setText("рядом с комплектом не найдены")
            self.chk_textures.setEnabled(False)
            return
        self.chk_textures.setEnabled(True)
        size = sum(os.path.getsize(p) for p in self._textures
                   if os.path.isfile(p))
        self.lbl_textures.setText(
            f"{len(self._textures)} файл(ов), {_fmt_size(size)}")
        self.lbl_textures.setToolTip("\n".join(_elide(p, 70)
                                               for p in self._textures[:40]))

    # ---- состояние ---------------------------------------------------
    def _selected_files(self) -> Dict[str, str]:
        return {role: row.path for role, row in self.rows.items() if row.path}

    def _selected_textures(self) -> List[str]:
        return list(self._textures) if self.chk_textures.isChecked() else []

    def _selected_web(self) -> List[Any]:
        if self._plan is None or not self.chk_web.isChecked():
            return []
        return list(self._plan.web_files)

    def _mode(self) -> str:
        return "replace" if self.rb_replace.isChecked() else "patch"

    def _sync_state(self) -> None:
        """Пересчитать подсказки и доступность кнопки загрузки."""
        files = self._selected_files()
        total = sum(os.path.getsize(p) for p in files.values()
                    if os.path.isfile(p))
        total += sum(os.path.getsize(p) for p in self._selected_textures()
                     if os.path.isfile(p))
        total += sum(os.path.getsize(p) for _n, p in self._selected_web()
                     if os.path.isfile(p))
        count = (len(files) + len(self._selected_textures())
                 + len(self._selected_web()))
        self.lbl_total.setText(f"к отправке: {count} файл(ов), "
                               f"{_fmt_size(total)}")

        problems = self._blocking_problems()
        busy = self._job is not None and self._job.isRunning()
        self.btn_upload.setEnabled(not busy and not problems)
        self.btn_upload.setToolTip("\n".join(problems) if problems else
                                   ("Заменить модель на сервере"
                                    if self._remote else
                                    "Загрузить модель на сервер"))

        label = getattr(self, "lbl_blockers", None)
        if label is None:
            return
        if problems:
            label.setText("Загрузить нельзя: "
                          + "; ".join(problems).rstrip(".") + ".")
            label.setStyleSheet(f"color: {COLOR_WARN}; font-size: 11px;")
        else:
            auto = [ROLE_LABELS[r] for r in self._auto_convert_roles()]
            label.setText(("при загрузке будет собран из .bam: "
                           + ", ".join(auto)) if auto else "")
            label.setStyleSheet(f"color: {COLOR_TEXT_MUTED};"
                                f" font-size: 11px;")

    def _auto_convert_roles(self) -> List[str]:
        """
        Обязательные роли, которые соберём из .bam при нажатии «Загрузить».

        Кузова в OBJ у генератора нет никогда — только .bam, — поэтому для
        всех наших комплектов кнопка «Загрузить» упиралась в «не выбран Кузов,
        .obj», хотя файл добывается одним щелчком. Тупик убран: раз путь
        известен и однозначен, диалог проходит его сам, а в журнал пишет, что
        именно собрал.
        """
        return [role for role in REQUIRED_ROLES
                if not self.rows[role].path and self.rows[role].convert_source]

    def _blocking_problems(self) -> List[str]:
        """Что мешает нажать «Загрузить». Пусто — можно отправлять."""
        out: List[str] = []
        if not self._endpoint.ok:
            out.append(self._endpoint.error or "Адрес реестра не определён")
        key_error = validate_key(self.ed_key.text().strip())
        if key_error:
            out.append(key_error)
        if self._mode() == "replace":
            missing = [ROLE_LABELS[r] for r in REQUIRED_ROLES
                       if not self.rows[r].path
                       and not self.rows[r].convert_source]
            if missing:
                out.append("нет обязательных файлов: " + ", ".join(missing)
                           + " — выберите их кнопкой «Обзор…»")
            if self.sp_volume.value() <= 0:
                out.append("не задан max_volume — сервер обязательно требует "
                           "его при полной замене")
            if not self.ed_points.toPlainText().strip():
                out.append("не заданы points_3d — четыре точки верхней кромки "
                           "кузова, сервер требует их при полной замене")
        elif not self._selected_files() and not self._selected_textures() \
                and not self._selected_web() and not self._meta_touched():
            out.append("В режиме «дополнить» нечего отправлять")
        return out

    def _meta_touched(self) -> bool:
        return bool(self.sp_volume.value() > 0
                    or self.ed_points.toPlainText().strip()
                    or self.chk_camera.isChecked()
                    or self.ed_display.text().strip())

    def _on_key_changed(self, text: str) -> None:
        error = validate_key(text.strip())
        if error:
            self.lbl_key.setText(error)
            self.lbl_key.setStyleSheet(f"color: {COLOR_DANGER};"
                                       f" font-size: 11px;")
        else:
            self.lbl_key.setText(
                "на сервере: data/models/<ключ>/ и запись в "
                "models_geometry_config.json")
            self.lbl_key.setStyleSheet(f"color: {COLOR_TEXT_MUTED};"
                                       f" font-size: 11px;")
        # Ключ сменился — то, что мы знали о модели на сервере, больше не
        # про неё: спросим заново перед загрузкой.
        self._remote = False
        self._sync_state()

    # ---- журнал ------------------------------------------------------
    def _log(self, text: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        self.log.appendPlainText(f"{stamp}  {text}")
        self.log.verticalScrollBar().setValue(
            self.log.verticalScrollBar().maximum())

    # ---- сеть --------------------------------------------------------
    def _client(self) -> ModelRegistry:
        return ModelRegistry(
            base_url=self._endpoint.url,
            token=self._endpoint.token,
            verify_tls=self._endpoint.verify_tls,
        )

    def _probe(self, auto: bool) -> None:
        """Спросить у сервера здоровье и карточку текущего ключа."""
        if self._job is not None and self._job.isRunning():
            return
        key = self.ed_key.text().strip()
        client = self._client()

        def work(_job: _Job) -> Dict[str, Any]:
            health = client.health()
            remote = client.get_model(key) if key and not validate_key(key) \
                else None
            return {"health": health, "remote": remote}

        self.btn_probe.setEnabled(False)
        self.lbl_server.setText("проверяем связь…")
        self._run_job(work, self._on_probe_done,
                      on_fail=self._on_probe_failed if not auto
                      else self._on_probe_failed_quiet)

    def _on_probe_done(self, result: Dict[str, Any]) -> None:
        self.btn_probe.setEnabled(True)
        health = result.get("health") or {}
        remote = result.get("remote")
        self._remote = remote
        revision = health.get("revision")
        self.lbl_server.setText(
            f"связь есть · ревизия реестра {revision} · "
            + (f"модель «{self.ed_key.text().strip()}» на сервере есть "
               f"(ревизия {remote.get('revision')}, "
               f"{'готова к обработке' if remote.get('ready') else 'НЕ готова'})"
               if remote else "модели с таким ключом на сервере нет"))
        self.lbl_server.setStyleSheet(f"color: {COLOR_TEXT_MUTED};"
                                      f" font-size: 11px;")
        self._log(f"реестр доступен, ревизия {revision}")
        if remote:
            self._log(f"ключ занят: ревизия {remote.get('revision')}, "
                      f"обновлена {remote.get('updated_at') or _EM_DASH}")
            if not remote.get("ready"):
                for problem in remote.get("problems") or []:
                    self._log(f"на сервере: {problem}")
            mirror = _mirror_note(remote)
            if mirror:
                self._log(f"внимание: {mirror}")
        self._sync_state()

    def _on_probe_failed(self, title: str, text: str) -> None:
        self._on_probe_failed_quiet(title, text)
        QMessageBox.warning(self, title, text)

    def _on_probe_failed_quiet(self, title: str, text: str) -> None:
        self.btn_probe.setEnabled(True)
        self._remote = False
        self.lbl_server.setText(f"{title}: {text.splitlines()[0]}")
        self.lbl_server.setStyleSheet(f"color: {COLOR_WARN}; font-size: 11px;")
        self._log(f"{title}: {text.splitlines()[0]}")
        self._sync_state()

    def _run_job(self, work: Callable[[_Job], Any],
                 on_done: Callable[[Any], None],
                 on_fail: Optional[Callable[[str, str], None]] = None,
                 on_progress: bool = False) -> None:
        job = _Job(work, self)
        self._job = job
        job.done.connect(on_done)
        job.message.connect(self._log)
        job.failed.connect(on_fail or self._on_job_failed)
        job.cancelled.connect(self._on_job_cancelled)
        job.finished.connect(self._on_job_finished)
        if on_progress:
            job.progress.connect(self._on_progress)
        job.start()
        self._sync_state()

    def _on_job_finished(self) -> None:
        # Проверка ключа перед загрузкой запускает следующую работу прямо из
        # обработчика `done` предыдущей — то есть ДО того, как придёт её
        # `finished`. Без сверки с отправителем этот обработчик обнулил бы
        # ссылку на уже идущую загрузку, и «Прервать» перестало бы работать.
        job = self.sender()
        if self._job is not None and job is not self._job:
            return
        self._job = None
        if isinstance(job, _Job):
            job.deleteLater()
        self._sync_state()

    def _on_job_failed(self, title: str, text: str) -> None:
        self._end_progress()
        self._log(f"{title}: {text.splitlines()[0]}")
        QMessageBox.critical(self, title, text)

    def _on_job_cancelled(self) -> None:
        self._end_progress()
        self._log("загрузка прервана пользователем")
        QMessageBox.information(
            self, "Загрузка прервана",
            "Отправка остановлена. На сервере ничего не изменилось: реестр "
            "применяет модель одним куском и откатывает незавершённую "
            "загрузку целиком.")

    # ---- конвертация .bam -> .obj ------------------------------------
    def _convert_role(self, role: str) -> None:
        row = self.rows.get(role)
        if row is None or not row.convert_source:
            return
        self.setCursor(Qt.CursorShape.WaitCursor)
        try:
            path = bam_to_obj(row.convert_source)
        except Exception as exc:                          # noqa: BLE001
            self.unsetCursor()
            self._log(f"конвертация не удалась: {exc}")
            QMessageBox.warning(
                self, "Не удалось собрать OBJ",
                f"{os.path.basename(row.convert_source)} → OBJ:\n{exc}\n\n"
                "Выберите файл вручную кнопкой «Обзор…».")
            return
        self.unsetCursor()
        row.set_path(path)
        self._log(f"собран {os.path.basename(path)} "
                  f"({_fmt_size(os.path.getsize(path))}) из "
                  f"{os.path.basename(row.convert_source)}")

    def _pick_textures(self) -> None:
        start = os.path.dirname(self._textures[0]) if self._textures else ""
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Текстуры машины", start,
            "Изображения (*.png *.jpg *.jpeg *.tga *.dds *.bmp);;"
            "Все файлы (*.*)")
        if paths:
            self._textures = paths
            self.chk_textures.setChecked(True)
            self._sync_textures_label()
            self._sync_state()

    def _camera_from_scene(self) -> None:
        if self._camera_provider is None:
            return
        state = None
        try:
            state = self._camera_provider()
        except Exception as exc:                          # noqa: BLE001
            self._log(f"камера сцены недоступна: {exc}")
        if not state:
            QMessageBox.information(
                self, "Камера сцены",
                "Не удалось прочитать положение камеры — сцена ещё не "
                "загружена.")
            return
        pos = list(state.get("pos") or [0, 0, 0])
        hpr = list(state.get("hpr") or [0, 0, 0])
        fov = state.get("fov") or 87.0
        for sp, value in zip(self.cam_fields, pos + hpr + [fov]):
            sp.setValue(float(value or 0.0))
        self.chk_camera.setChecked(True)
        self._log("пресет камеры взят из сцены")

    # ---- сбор meta ---------------------------------------------------
    def _collect_meta(self) -> Optional[Dict[str, Any]]:
        """`meta` из полей формы или None, если поля не разобрались."""
        points = None
        raw = self.ed_points.toPlainText().strip()
        if raw:
            try:
                points = json.loads(raw)
            except ValueError as exc:
                QMessageBox.warning(self, "points_3d",
                                    f"Это не JSON: {exc}")
                return None
            if (not isinstance(points, list) or len(points) != 4
                    or any(not isinstance(p, list) or len(p) != 3
                           for p in points)):
                QMessageBox.warning(
                    self, "points_3d",
                    "Нужны ровно четыре точки вида [x, y, z] — "
                    "прямоугольник верхней кромки кузова.")
                return None
            points = [[float(v) for v in p] for p in points]

        camera = None
        if self.chk_camera.isChecked():
            values = [sp.value() for sp in self.cam_fields]
            camera = {"pos": values[0:3], "hpr": values[3:6],
                      "fov": values[6] or 87.0}

        volume = self.sp_volume.value() or None
        ground = self.sp_ground.value()
        return meta_from_form(
            display_name=self.ed_display.text().strip(),
            max_volume=round(volume, 4) if volume else None,
            ground_plane=round(ground, 4),
            points_3d=points,
            camera=camera,
            textures_dir=self.ed_key.text().strip()
            if self._selected_textures() else "",
        )

    # ---- загрузка ----------------------------------------------------
    def _start_upload(self) -> None:
        if self._job is not None and self._job.isRunning():
            return
        for role in self._auto_convert_roles():
            self._convert_role(role)
            if not self.rows[role].path:
                return          # причина уже показана окном ошибки
        problems = self._blocking_problems()
        if problems:
            QMessageBox.warning(self, "Нельзя загрузить",
                                "\n".join(f"• {p}" for p in problems))
            return

        key = self.ed_key.text().strip()
        meta = self._collect_meta()
        if meta is None:
            return

        if self._mode() == "replace" and not self.chk_camera.isChecked() \
                and self._remote in (False, None):
            answer = QMessageBox.question(
                self, "Пресет камеры не задан",
                "Пресет камеры не заполнен, а модели с таким ключом на "
                "сервере (насколько мы знаем) нет.\n\n"
                "Без пресета обработка фото упадёт с «camera preset lookup "
                "failed». Всё равно загрузить?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                return

        # Про замену спрашиваем только зная наверняка: если карточку с сервера
        # ещё не читали (или ключ с тех пор поменялся), сперва читаем.
        if self._remote is False:
            self._log("уточняем, занят ли ключ на сервере…")
            client = self._client()

            def probe_then(_job: _Job) -> Any:
                return client.get_model(key)

            self._run_job(
                probe_then,
                lambda remote: self._after_probe_before_upload(remote, meta),
                on_fail=self._on_job_failed)
            return

        self._confirm_and_upload(meta)

    def _after_probe_before_upload(self, remote: Any,
                                   meta: Dict[str, Any]) -> None:
        self._remote = remote
        self._confirm_and_upload(meta)

    def _confirm_and_upload(self, meta: Dict[str, Any]) -> None:
        key = self.ed_key.text().strip()
        remote = self._remote if isinstance(self._remote, dict) else None

        if remote:
            mode_line = (
                "Файлы каталога будут собраны заново: чего нет в списке выше, "
                "того на сервере не останется."
                if self._mode() == "replace" else
                "Недостающие файлы останутся от прежней версии, обновятся "
                "только присланные.")
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Warning)
            box.setWindowTitle("Модель уже есть на сервере")
            box.setText(
                f"На сервере уже есть модель «{key}».\n\n"
                f"Ревизия {remote.get('revision')}, обновлена "
                f"{remote.get('updated_at') or _EM_DASH}.\n{mode_line}\n\n"
                "Заменить её?")
            details = [f"ready: {remote.get('ready')}"]
            for problem in remote.get("problems") or []:
                details.append(f"problem: {problem}")
            for f in remote.get("files") or []:
                details.append(f"{f.get('name')}  {_fmt_size(f.get('size', 0))}")
            box.setDetailedText("\n".join(details))
            yes = box.addButton("Заменить",
                                QMessageBox.ButtonRole.DestructiveRole)
            box.addButton("Отмена", QMessageBox.ButtonRole.RejectRole)
            box.setDefaultButton(yes)
            box.exec()
            if box.clickedButton() is not yes:
                self._log("замена отменена пользователем")
                return

        files = self._selected_files()
        textures = self._selected_textures()
        web_files = self._selected_web()
        client = self._client()
        mode = self._mode()

        def work(job: _Job) -> Dict[str, Any]:
            job.message.emit(
                f"отправляем {len(files) + len(textures) + len(web_files)} "
                f"файл(ов), режим {mode}…")
            return client.upsert_model(key, meta, files, textures=textures,
                                       web_files=web_files, mode=mode,
                                       progress=job.on_progress)

        self._begin_progress()
        self._run_job(work, self._on_upload_done, on_fail=self._on_job_failed,
                      on_progress=True)

    # ---- прогресс ----------------------------------------------------
    def _begin_progress(self) -> None:
        self._started_at = time.monotonic()
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.lbl_speed.setVisible(True)
        self.lbl_speed.setText("подключаемся…")
        self.btn_close.setText("Прервать")
        try:
            self.btn_close.clicked.disconnect()
        except TypeError:
            pass
        self.btn_close.clicked.connect(self._cancel_upload)
        self._sync_state()

    def _end_progress(self) -> None:
        self.progress.setVisible(False)
        self.lbl_speed.setVisible(False)
        self.btn_close.setText("Закрыть")
        try:
            self.btn_close.clicked.disconnect()
        except TypeError:
            pass
        self.btn_close.clicked.connect(self.reject)
        self._sync_state()

    def _cancel_upload(self) -> None:
        if self._job is not None and self._job.isRunning():
            self._job.cancel()
            self.lbl_speed.setText("прерываем…")
            return
        self.reject()

    def _on_progress(self, sent: int, total: int) -> None:
        if total <= 0:
            return
        percent = int(sent * 100 / total)
        self.progress.setValue(percent)
        elapsed = max(1e-3, time.monotonic() - self._started_at)
        speed = sent / elapsed
        left = (total - sent) / speed if speed > 0 else 0
        self.lbl_speed.setText(
            f"{_fmt_size(sent)} из {_fmt_size(total)}  ·  "
            f"{_fmt_size(speed)}/с  ·  осталось {_fmt_eta(left)}")

    # ---- итог --------------------------------------------------------
    def _on_upload_done(self, result: Dict[str, Any]) -> None:
        self._end_progress()
        self._uploaded = True
        self._uploaded_key = self.ed_key.text().strip()
        model = result.get("model") or {}
        created = bool(result.get("created"))
        revision = result.get("revision")
        self._remote = model or None

        self._log(("модель создана" if created else "модель заменена")
                  + f", ревизия реестра {revision}")
        for name in result.get("stored_files") or []:
            self._log(f"записан {name}")
        warnings: List[str] = list(result.get("warnings") or [])
        warnings += list(model.get("warnings") or [])
        for text in warnings:
            self._log(f"предупреждение: {text}")
        ignored = result.get("ignored_parts") or []
        for part in ignored:
            self._log(f"поле проигнорировано сервером: {part}")

        ready = bool(model.get("ready"))
        problems = list(model.get("problems") or [])
        for problem in problems:
            self._log(f"проблема: {problem}")
        mirror = _mirror_note(model)
        if mirror:
            self._log(f"зеркало: {mirror}")
            problems.append(mirror)

        head = ("Модель создана" if created else "Модель заменена") \
            + f" · ревизия реестра {revision}"
        details = []
        if warnings:
            details.append("Предупреждения:\n"
                           + "\n".join(f"• {w}" for w in warnings))
        if ignored:
            details.append("Сервер не понял поля (опечатка в имени?):\n"
                           + "\n".join(f"• {p}" for p in ignored))

        if ready and not mirror:
            body = (f"{head}\n\nПайплайн с этой моделью поедет: следующий "
                    "запуск обработки фото уже увидит новую геометрию, а в "
                    "списке кузовов набор появится после обновления списка."
                    "\n\nВ meta.model при обработке передавайте ключ точно: "
                    f"«{self.ed_key.text().strip()}».")
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Warning if warnings
                        else QMessageBox.Icon.Information)
            box.setWindowTitle("Готово")
            box.setText(body)
            if details:
                box.setDetailedText("\n\n".join(details))
            box.exec()
        else:
            # Модель может быть годной для обработки фото и при этом не
            # доехать до второго дерева — тогда виновато зеркало, а не файлы,
            # и «догрузите недостающее» было бы вредным советом.
            lead = ("Файлы и конфиг записаны, но обработка фото с этой "
                    "моделью не пройдёт:" if not ready else
                    "Файлы и конфиг записаны, обработка фото пройдёт, но:")
            tail = ("\n\nДогрузите недостающее — режим «Дополнить» не "
                    "потеряет уже загруженное." if not ready else
                    "\n\nЭто чинится на сервере: реестр должен быть запущен "
                    "с --mirror-data-dir / --mirror-config-dir на второе "
                    "дерево. Повторная загрузка сама по себе не поможет.")
            body = (f"{head}\n\n{lead}\n"
                    + "\n".join(f"• {p}" for p in problems or
                                ["сервер не объяснил причину"]) + tail)
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Warning)
            box.setWindowTitle("Загружено, но не готово")
            box.setText(body)
            if details:
                box.setDetailedText("\n\n".join(details))
            box.exec()
            if not ready:
                # Догружать недостающее логичнее «дополнением»: уже уехавшие
                # файлы при этом не теряются. Сбой зеркала так не лечится,
                # поэтому режим трогаем только когда дело в файлах.
                self.rb_patch.setChecked(True)

    # ---- закрытие ----------------------------------------------------
    @property
    def uploaded(self) -> bool:
        """Была ли хоть одна успешная загрузка — сигнал обновить списки."""
        return self._uploaded

    @property
    def uploaded_key(self) -> str:
        """Ключ, под которым модель уехала на сервер (для проверки списка)."""
        return self._uploaded_key

    def reject(self) -> None:                            # noqa: D102 (Qt API)
        if self._job is not None and self._job.isRunning():
            answer = QMessageBox.question(
                self, "Идёт загрузка",
                "Файлы ещё отправляются. Прервать и закрыть окно?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if answer != QMessageBox.StandardButton.Yes:
                return
            self._job.cancel()
            self._job.wait(5000)
        super().reject()
