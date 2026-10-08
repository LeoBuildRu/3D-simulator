# right_panel.py
# ---------------------------------------------------------------------------
# Инспектор справа от 3D-вида.
#
# Плавающая панель на всю высоту (FloatingPanel, см. src/ui/hud.py) с
# четырьмя вкладками-иконками вместо одной длинной ленты карточек:
#
#   ▣ Сцена    — кузов, наполнитель (текстура + объём), «Сгенерировать»,
#                генератор кузова
#   ☰ Записи   — список реконструкций, подробности выбранной, «Кино»,
#                «Реконструировать»
#   ◉ Камера   — FOV и крен; снимок стенда поверх вида (прозрачность,
#                показать/скрыть); опорные точки
#   ≋ Датасет  — сводка съёмки, «Настроить», «Снять»
#
# Качество графики — в меню под иконкой дисплея в шапке.
#
# Панель ничего не делает со сценой сама: всё уходит сигналами в
# MainWindow (список сигналов — в классе RightPanel).
# ---------------------------------------------------------------------------

from __future__ import annotations

import json
import os

from PyQt6.QtCore import Qt, QPoint, QSize, QTimer, pyqtSignal
from PyQt6.QtGui import QActionGroup, QColor, QFontMetrics, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QDialog, QDoubleSpinBox, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMenu, QPushButton,
    QScrollArea, QSizePolicy, QSlider, QStackedWidget, QVBoxLayout, QWidget,
)

from src.ui import icons
from src.ui.hud import (
    Card, Chip, FloatingPanel, IconButton, SegmentedControl, Switch,
    TileButton,
    hline, icon_label, label,
)
from src.ui.ui_theme import (
    COLOR_ACCENT, COLOR_DANGER, COLOR_PURPLE, COLOR_SUCCESS, COLOR_TEAL,
    COLOR_TEXT, COLOR_TEXT_DIM, COLOR_TEXT_MUTED, COLOR_WARN, FONT_MONO,
    apply_hud_theme, rgba,
)
from src.ui.model_picker import ModelPickerCombo
from src.ui.panel_data import (
    load_model_sets_detailed, load_texture_sets, get_default_texture_set_key,
    load_reconstructions, Reconstruction, PROJECT_ROOT, HEIGHT_EXAMPLES_DIR,
    get_model_set_config, download_server_image, SERVER_IMAGE_CACHE_DIR,
    RECON_PAGE_SIZE,
)
from src.core import graphics_settings


# ---------------------------------------------------------------------------
# Типы записей: иконка, цвет плитки, подпись для подсказки
# ---------------------------------------------------------------------------
_DTYPES = {
    "height": ("bars",   COLOR_ACCENT, "Карта высот"),
    "ply":    ("points", COLOR_PURPLE, "Облако точек (PLY)"),
    "stand":  ("camera", COLOR_TEAL,   "Снимок стенда"),
    "depth":  ("depth",  COLOR_WARN,   "Карта глубины с сервера"),
}


def _dtype(rec_type: str) -> tuple[str, str, str]:
    return _DTYPES.get(rec_type or "", ("doc", "#8E8E93",
                                        (rec_type or "—").upper()))


def _type_tile(rec_type: str, size: int = 32) -> QLabel:
    """Цветная плитка с иконкой типа записи — тип читается без подписи."""
    ic, color, tip = _dtype(rec_type)
    tile = QLabel()
    tile.setFixedSize(size, size)
    tile.setAlignment(Qt.AlignmentFlag.AlignCenter)
    tile.setPixmap(icons.pixmap(ic, color, int(size * 0.6)))
    tile.setToolTip(tip)
    tile.setStyleSheet(
        f"QLabel {{ background: {rgba(color, 46)}; border-radius: {size // 4 + 1}px; }}")
    return tile


def _elide(text: str, font, max_w: int,
           mode=Qt.TextElideMode.ElideRight) -> str:
    if not text:
        return ""
    return QFontMetrics(font).elidedText(text, mode, max_w)


_CINEMATIC_CFG = os.path.join(PROJECT_ROOT, "config", "cinematic.json")


def _load_cinematic_enabled() -> bool:
    try:
        with open(_CINEMATIC_CFG, "r", encoding="utf-8") as fh:
            return bool(json.load(fh).get("enabled", True))
    except (OSError, ValueError):
        return True


def _save_cinematic_enabled(on: bool) -> None:
    try:
        os.makedirs(os.path.dirname(_CINEMATIC_CFG), exist_ok=True)
        with open(_CINEMATIC_CFG, "w", encoding="utf-8") as fh:
            json.dump({"enabled": bool(on)}, fh)
    except OSError as exc:
        print(f"[RightPanel] cinematic.json не сохранён: {exc}")


def _format_short_dt(rec: Reconstruction) -> str:
    """
    Compact two-line-friendly timestamp ("23 Apr · 15:32"). Falls back
    to the raw `time` string if `datetime` couldn't be parsed.
    """
    from datetime import datetime as _dt
    if rec.datetime and rec.datetime != _dt.min:
        return rec.datetime.strftime("%d %b · %H:%M")
    return rec.time or "—"


def _resolve_image_path(rec: Reconstruction) -> str | None:
    """
    Resolve `rec.img_file` into an absolute path suitable for QPixmap,
    WITHOUT touching the network.

    LOCAL entries:   look under `height_examples/`, then PROJECT_ROOT.
    SERVER entries:  look in the server-image cache directory; if the
                     image was already downloaded for a previous open,
                     reuse it. Otherwise return None and let the caller
                     trigger `download_server_image()` explicitly.
    """
    # Stand snapshots carry an explicit absolute colour-frame path.
    color_path = (getattr(rec, "color_path", "") or "").strip()
    if color_path and os.path.exists(color_path):
        return color_path

    img = (rec.img_file or "").strip()
    if not img:
        return None

    # Already absolute (or absolute-ish) — trust it.
    if os.path.isabs(img):
        return img if os.path.exists(img) else None

    # Local entries: probe the height_examples directory.
    if rec.is_local:
        candidate = os.path.join(HEIGHT_EXAMPLES_DIR, img)
        if os.path.exists(candidate):
            return candidate
        candidate = os.path.join(PROJECT_ROOT, img)
        if os.path.exists(candidate):
            return candidate
        return None

    # Server entries: cache hit?
    cached = os.path.join(SERVER_IMAGE_CACHE_DIR, img)
    if os.path.exists(cached) and os.path.getsize(cached) > 0:
        return cached
    return None


def _resolve_or_fetch_image_path(rec: Reconstruction) -> str | None:
    """
    Same as `_resolve_image_path`, but for SERVER entries it ALSO
    triggers a synchronous download via TLS_client when no cache hit
    exists. Returns the local path on success, or None on any failure.
    """
    p = _resolve_image_path(rec)
    if p is not None:
        return p
    if rec.is_local:
        return None
    # Серверные depth-записи: качаем через /download_depth_file (а не через
    # старый /download?file=). Preview thumbnail в списке показывает
    # ФИНАЛЬНОЕ изображение (kind='masked' — после de-barrel + polygon-crop),
    # с резервами на исходный uploaded и depth-карту.
    if getattr(rec, "data_type", "") == "depth":
        from src.ui.panel_data import resolve_depth_record_files
        try:
            paths = resolve_depth_record_files(rec)
        except Exception as exc:
            print(f"[right_panel] resolve depth preview failed: {exc}")
            return None
        return (paths.get("color")
                or paths.get("uploaded")
                or paths.get("depth")
                or None)
    return download_server_image(rec.img_file or "")



# ---------------------------------------------------------------------------
# Строка списка записей
# ---------------------------------------------------------------------------
#   ┌────────────────────────────────────────────────┐
#   │ [▥]  К906ТС190                            (⌕)  │
#   │      Камаз 6520 · 29 Sep · 18:52                │
#   └────────────────────────────────────────────────┘
# Тип — цветом и иконкой плитки, без текстовой метки.
# ---------------------------------------------------------------------------
class ReconRowWidget(QWidget):
    ROW_FIXED_HEIGHT = 50

    viewClicked = pyqtSignal()

    def __init__(self, rec: Reconstruction, max_text_width: int = 200,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self._rec = rec
        self.setMinimumHeight(self.ROW_FIXED_HEIGHT)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(8, 6, 6, 6)
        outer.setSpacing(10)

        outer.addWidget(_type_tile(rec.data_type, 32), 0,
                        Qt.AlignmentFlag.AlignVCenter)

        text_col = QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        text_col.setSpacing(1)
        self.car = QLabel()
        self.car.setStyleSheet(
            f"color: {COLOR_TEXT}; font-family: {FONT_MONO};"
            "font-size: 13px; font-weight: 600; background: transparent;")
        self.car.setText(_elide(rec.car_number or "—", self.car.font(),
                                max_text_width))
        text_col.addWidget(self.car)

        meta_text = " · ".join(filter(None, [
            (rec.model or "").strip(), _format_short_dt(rec)])) or "—"
        self.meta = QLabel()
        self.meta.setStyleSheet(
            f"color: {COLOR_TEXT_MUTED}; font-size: 11px; background: transparent;")
        self.meta.setText(_elide(meta_text, self.meta.font(), max_text_width))
        self.meta.setToolTip(meta_text)
        text_col.addWidget(self.meta)
        outer.addLayout(text_col, 1)

        self.btn_view = IconButton("zoom_in", "Снимок и подробности",
                                   size=28, icon_size=16,
                                   color=COLOR_TEXT_MUTED)
        self.btn_view.clicked.connect(lambda _c=False: self.viewClicked.emit())
        outer.addWidget(self.btn_view, 0, Qt.AlignmentFlag.AlignVCenter)


# ---------------------------------------------------------------------------
# Просмотр снимка записи — полноэкранная модальная карточка
# ---------------------------------------------------------------------------
class RecordPhotoOverlay(QDialog):
    """Затемнение окна + карточка: шапка, снимок, строка фактов. Esc / клик
    мимо — закрыть."""

    OUTER_MARGIN = 28

    def __init__(self, rec: Reconstruction, parent: QWidget | None = None):
        super().__init__(parent)
        self._rec = rec
        self.setWindowFlags(Qt.WindowType.Dialog
                            | Qt.WindowType.FramelessWindowHint
                            | Qt.WindowType.NoDropShadowWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setModal(True)
        apply_hud_theme(self)

        self._backdrop = QFrame(self)
        self._backdrop.setStyleSheet("background-color: rgba(0, 0, 0, 170);")

        self.card = QFrame(self)
        self.card.setObjectName("PhotoCard")
        self.card.setStyleSheet(
            "QFrame#PhotoCard { background-color: rgba(30, 30, 32, 250);"
            " border: 1px solid rgba(255,255,255,22); border-radius: 16px; }")
        card_lay = QVBoxLayout(self.card)
        card_lay.setContentsMargins(20, 16, 16, 18)
        card_lay.setSpacing(14)
        card_lay.addLayout(self._build_header())
        card_lay.addWidget(self._build_image_preview(), 1)
        card_lay.addLayout(self._build_info_strip())

        outer = QGridLayout(self)
        m = self.OUTER_MARGIN
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._backdrop, 0, 0)
        inner = QVBoxLayout()
        inner.setContentsMargins(m, m, m, m)
        inner.addWidget(self.card)
        outer.addLayout(inner, 0, 0)

    def _build_header(self) -> QHBoxLayout:
        rec = self._rec
        h = QHBoxLayout()
        h.setSpacing(12)
        h.addWidget(_type_tile(rec.data_type, 36))
        col = QVBoxLayout()
        col.setSpacing(0)
        title = QLabel(rec.car_number or rec.name or "—")
        title.setStyleSheet(
            f"color: {COLOR_TEXT}; font-family: {FONT_MONO};"
            " font-size: 17px; font-weight: 600;")
        col.addWidget(title)
        sub = QLabel(" · ".join(filter(None, [
            _dtype(rec.data_type)[2], (rec.model or "").strip()])))
        sub.setProperty("role", "muted")
        col.addWidget(sub)
        h.addLayout(col)
        h.addStretch(1)
        btn_x = IconButton("close", "Закрыть · Esc", size=30, icon_size=16,
                           filled=True)
        btn_x.clicked.connect(self.close)
        h.addWidget(btn_x, 0, Qt.AlignmentFlag.AlignTop)
        return h

    def _build_image_preview(self) -> QWidget:
        rec = self._rec
        canvas = QLabel()
        canvas.setStyleSheet("background: transparent; border: none;")
        canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        canvas.setMinimumHeight(260)
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                             QSizePolicy.Policy.Expanding)
        path = _resolve_or_fetch_image_path(rec)
        pm = QPixmap(path) if path else QPixmap()
        if path and not pm.isNull():
            canvas.setPixmap(self._scale_pixmap(pm, 800, 480))

            def _on_resize(_e, c=canvas, p=pm):
                c.setPixmap(self._scale_pixmap(p, c.width() - 8, c.height() - 8))

            canvas.resizeEvent = _on_resize  # type: ignore[assignment]
        else:
            if path:
                text = "Файл изображения повреждён"
            elif rec.is_local:
                text = "Изображение не найдено локально"
            else:
                text = "Не удалось загрузить изображение\nсервер недоступен или файла нет"
            canvas.setText(text)
            canvas.setStyleSheet(
                f"background: transparent; color: {COLOR_TEXT_MUTED}; font-size: 13px;")
        return canvas

    @staticmethod
    def _scale_pixmap(pm: QPixmap, w: int, h: int) -> QPixmap:
        if w <= 0 or h <= 0:
            return pm
        return pm.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio,
                         Qt.TransformationMode.SmoothTransformation)

    def _build_info_strip(self) -> QHBoxLayout:
        rec = self._rec
        target = (f"{rec.target_volume:.2f} м³"
                  if rec.target_volume is not None else "—")
        cells = (
            ("truck", "Модель", rec.model or "—"),
            ("heap", "Наполнитель", rec.filler or "—"),
            ("cube", "Целевой объём", target),
            ("clock", "Время", _format_short_dt(rec)),
            ("doc", "Файл", rec.name or "—"),
        )
        h = QHBoxLayout()
        h.setSpacing(22)
        for ic, tip, value in cells:
            cell = QHBoxLayout()
            cell.setSpacing(7)
            cell.addWidget(icon_label(ic, COLOR_TEXT_MUTED, 15, tip))
            v = QLabel(value if len(value) <= 36 else "…" + value[-32:])
            v.setToolTip(f"{tip}: {value}")
            v.setStyleSheet(f"color: {COLOR_TEXT}; font-size: 12px;")
            if ic == "doc":
                v.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            cell.addWidget(v)
            h.addLayout(cell)
        h.addStretch(1)
        return h

    def showEvent(self, event):
        parent = self.parentWidget()
        if parent is not None and parent.window() is not None:
            self.setGeometry(parent.window().geometry())
        super().showEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            return
        super().keyPressEvent(event)

    def mousePressEvent(self, event):
        gp = event.globalPosition().toPoint()
        if not self.card.geometry().contains(self.mapFromGlobal(gp)):
            self.close()
            return
        super().mousePressEvent(event)


# ---------------------------------------------------------------------------
# Инспектор
# ---------------------------------------------------------------------------
class RightPanel(FloatingPanel):
    """Правая панель-инспектор с вкладками (см. шапку модуля)."""

    modelSetChanged          = pyqtSignal(str)
    # Удаление набора моделей с диска (генератор / assets/models/trucks).
    # Подтверждение показывает MainWindow: набор может быть сейчас в сцене.
    modelSetDeleteRequested  = pyqtSignal(str)
    # Отправка набора в реестр моделей на сервере (диалог — в MainWindow).
    modelSetUploadRequested  = pyqtSignal(str)
    textureSetChanged        = pyqtSignal(str)
    reconstructionSelected   = pyqtSignal(str)
    # Кнопка «Реконструировать» — payload: Reconstruction.
    reconstructionRunRequested = pyqtSignal(object)
    # Выбрана запись-снимок (stand / depth) — Reconstruction, иначе None.
    standReferenceSelected     = pyqtSignal(object)
    fovChanged                 = pyqtSignal(float)
    rollChanged                = pyqtSignal(float)
    referenceOpacityChanged    = pyqtSignal(float)
    referenceVisibleToggled    = pyqtSignal(bool)
    pointPickingToggled        = pyqtSignal(bool)
    pointsResetRequested       = pyqtSignal()
    pointVizToggled            = pyqtSignal(bool)
    autoPointsRequested        = pyqtSignal()
    # «Сгенерировать»: {"model_key", "texture_key", "target_volume"}.
    runRequested             = pyqtSignal(dict)
    bodyGenRequested         = pyqtSignal()
    #: «Каких кузовов не хватает на сервере» — очередь на генерацию.
    bodyWorklistRequested    = pyqtSignal()
    # ultra / medium / performance — MainWindow сохраняет и просит перезапуск.
    graphicsPresetChanged    = pyqtSignal(str)
    # Вкладка «Датасет».
    datasetSettingsRequested = pyqtSignal()
    datasetStartRequested    = pyqtSignal()

    PANEL_WIDTH = 344

    _FOV_MIN = 20
    _FOV_MAX = 150
    _FOV_DEFAULT = 100
    _ROLL_MIN = -180
    _ROLL_MAX = 180

    def __init__(self, parent: QWidget, margin: int = 16):
        super().__init__(parent, anchor="right-stretch", margin=margin,
                         padding=(14, 14, 14, 14), width=self.PANEL_WIDTH)
        col = self.body_layout
        col.setSpacing(12)

        col.addLayout(self._build_header())

        self.tabs = SegmentedControl([
            ("scene", "cube", "Сцена: кузов и наполнение"),
            ("records", "list", "Записи реконструкций"),
            ("camera", "camera", "Камера и совмещение со снимком"),
            ("dataset", "stack", "Съёмка датасета"),
        ], height=32, icon_size=17)
        col.addWidget(self.tabs)

        self.pages = QStackedWidget()
        self.pages.setStyleSheet("QStackedWidget { background: transparent; }")
        col.addWidget(self.pages, 1)

        self._page_keys: list[str] = []
        self._add_page("scene", self._build_scene_page(), scroll=True)
        self._add_page("records", self._build_records_page(), scroll=False)
        self._add_page("camera", self._build_camera_page(), scroll=True)
        self._add_page("dataset", self._build_dataset_page(), scroll=True)
        self.tabs.changed.connect(self._on_tab)

        # Подписка — только когда построены все вкладки: выбор записи
        # трогает и подробности, и вкладку «Камера».
        self.lst_recon.currentItemChanged.connect(self._on_recon_changed)
        self.lst_recon.itemClicked.connect(self._on_recon_clicked)

        # Стартовое состояние списка записей.
        if self._recons:
            self.lst_recon.setCurrentRow(0)
            self._selected_rec = self._recons[0]
            self._populate_details(self._recons[0])
        else:
            self._selected_rec = None
            self._populate_details(None)
            self.btn_run_recon.setEnabled(False)

    # ==================================================================
    # Каркас
    # ==================================================================
    def _add_page(self, key: str, page: QWidget, scroll: bool) -> None:
        if scroll:
            area = QScrollArea()
            area.setWidgetResizable(True)
            area.setFrameShape(QFrame.Shape.NoFrame)
            area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            area.setWidget(page)
            self.pages.addWidget(area)
        else:
            self.pages.addWidget(page)
        self._page_keys.append(key)

    def _on_tab(self, key: str) -> None:
        if key in self._page_keys:
            self.pages.setCurrentIndex(self._page_keys.index(key))
        if key == "camera":
            self.tabs.set_badge("camera", False)

    def show_tab(self, key: str) -> None:
        self.tabs.set_current(key)
        self._on_tab(key)

    @staticmethod
    def _page() -> tuple[QWidget, QVBoxLayout]:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 2, 0, 0)
        lay.setSpacing(8)
        return w, lay

    @staticmethod
    def _title(text: str) -> QLabel:
        t = QLabel(text)
        t.setProperty("role", "title")
        return t

    @staticmethod
    def _section(text: str, trailing: QWidget | None = None) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(4, 8, 0, 0)
        row.setSpacing(6)
        lbl = QLabel(text)
        lbl.setProperty("role", "eyebrow")
        row.addWidget(lbl, 0, Qt.AlignmentFlag.AlignBottom)
        row.addStretch(1)
        if trailing is not None:
            row.addWidget(trailing, 0, Qt.AlignmentFlag.AlignBottom)
        return row

    @staticmethod
    def _field_row(icon_name: str, tip: str, widget: QWidget,
                   trailing: QWidget | None = None) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(10)
        row.addWidget(icon_label(icon_name, COLOR_TEXT_MUTED, 17, tip), 0,
                      Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(widget, 1, Qt.AlignmentFlag.AlignVCenter)
        if trailing is not None:
            row.addWidget(trailing, 0, Qt.AlignmentFlag.AlignVCenter)
        return row

    @staticmethod
    def _value_label(text: str, width: int = 44) -> QLabel:
        v = label(text, mono=True, size=12, color=COLOR_TEXT)
        v.setFixedWidth(width)
        v.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        return v

    def _build_header(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(2, 0, 0, 0)
        row.setSpacing(10)
        mark = QLabel()
        mark.setFixedSize(30, 30)
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mark.setPixmap(icons.pixmap("cube", "#FFFFFF", 18))
        mark.setStyleSheet(
            "QLabel { border-radius: 8px; background: qlineargradient("
            "x1:0, y1:0, x2:1, y2:1, stop:0 #409CFF, stop:1 #5E5CE6); }")
        row.addWidget(mark)
        names = QVBoxLayout()
        names.setSpacing(0)
        names.addWidget(label("IQoko", size=15, weight=700))
        names.addWidget(label("3D-симулятор", role="caption"))
        row.addLayout(names)
        row.addStretch(1)

        self.btn_graphics = IconButton("display", "Качество графики", size=30)
        self.btn_graphics.clicked.connect(self._show_graphics_menu)
        row.addWidget(self.btn_graphics)
        return row

    # ==================================================================
    # Вкладка «Сцена»
    # ==================================================================
    def _build_scene_page(self) -> QWidget:
        page, lay = self._page()
        lay.addWidget(self._title("Сцена"))

        model_infos = load_model_sets_detailed()
        texture_sets = load_texture_sets()
        default_tex = get_default_texture_set_key()

        # ---- Кузов ------------------------------------------------------
        self._model_count = label("", role="caption")
        lay.addLayout(self._section("Кузов", self._model_count))
        # Не QComboBox: имена наборов длинные, а выбирать приходится по
        # объёму / шасси / комплекту — см. src/ui/model_picker.py.
        self.cmb_model = ModelPickerCombo()
        self.cmb_model.set_details(model_infos)
        for info in model_infos:
            self.cmb_model.addItem(info.display, userData=info.key)
        if model_infos:
            self.cmb_model.setCurrentIndex(0)
        else:
            self.cmb_model.addItem("Модели не найдены", userData=None)
            self.cmb_model.setEnabled(False)
        self.cmb_model.currentIndexChanged.connect(self._on_model_index_changed)
        self.cmb_model.deleteRequested.connect(
            lambda key: self.modelSetDeleteRequested.emit(str(key)))
        self.cmb_model.uploadRequested.connect(
            lambda key: self.modelSetUploadRequested.emit(str(key)))
        self._model_count.setText(self._model_count_label(len(model_infos)))

        # Две кнопки рядом с выбором кузова: «список» — очередь с сервера,
        # «палочка» — ручная сборка по своему файлу. Очередь слева, потому
        # что обычный сценарий начинается именно с неё: сначала смотрим, чего
        # не хватает, и только потом собираем.
        self.btn_worklist = IconButton("list", "Кузова к генерации…",
                                       size=34, icon_size=18)
        self.btn_worklist.setToolTip(
            "Кузова к генерации…\nМодели, которые сервер уже снимает, но "
            "геометрии под них нет: объём у таких снимков считается по "
            "чужому кузову.")
        self.btn_worklist.clicked.connect(self.bodyWorklistRequested.emit)

        self.btn_bodygen = IconButton("wand", "Собрать кузов по скану…",
                                      size=34, icon_size=18, filled=True)
        self.btn_bodygen.clicked.connect(self.bodyGenRequested.emit)
        pick_row = QHBoxLayout()
        pick_row.setSpacing(6)
        pick_row.addWidget(self.cmb_model, 1)
        pick_row.addWidget(self.btn_worklist)
        pick_row.addWidget(self.btn_bodygen)
        lay.addLayout(pick_row)

        # Строка состояния: итог удаления / загрузки, ход сборки кузова.
        self.lbl_model_status = label("", role="caption")
        self.lbl_model_status.setWordWrap(True)
        self.lbl_model_status.setContentsMargins(4, 0, 0, 0)
        self.lbl_model_status.hide()
        lay.addWidget(self.lbl_model_status)
        self.lbl_bodygen = self.lbl_model_status
        self._init_bodygen_state()

        # Характеристики выбранного кузова — то, по чему его и выбирают.
        self._spec = Card(spacing=7)
        self._spec_rows: dict[str, tuple[QLabel, QLabel]] = {}
        for key, ic, tip in (("volume", "cube", "Вместимость кузова"),
                             ("axles", "truck", "Колёсная формула"),
                             ("dims", "expand", "Внутренние габариты, м"),
                             ("kit", "stack", "Состав набора")):
            v = label("—", size=12)
            note = label("", role="caption")
            self._spec.add(self._field_row(ic, tip, v, note))
            self._spec_rows[key] = (v, note)
        lay.addWidget(self._spec)
        self._refresh_model_spec()

        # ---- Наполнитель -------------------------------------------------
        lay.addLayout(self._section("Наполнитель"))
        card = Card(spacing=10)
        self.cmb_texture = QComboBox()
        default_index = 0
        for i, (key, display) in enumerate(texture_sets):
            self.cmb_texture.addItem(display, userData=key)
            if default_tex and key == default_tex:
                default_index = i
        if texture_sets:
            self.cmb_texture.setCurrentIndex(default_index)
        else:
            self.cmb_texture.addItem("Текстуры не найдены", userData=None)
            self.cmb_texture.setEnabled(False)
        self.cmb_texture.currentIndexChanged.connect(self._on_texture_index_changed)
        card.add(self._field_row("texture", "Текстура наполнителя",
                                 self.cmb_texture))

        self.spn_target = QDoubleSpinBox()
        self.spn_target.setDecimals(2)
        self.spn_target.setRange(0.1, 999.0)
        self.spn_target.setSingleStep(0.5)
        self.spn_target.setSuffix(" м³")
        initial_volume = 10.0
        cur_model_key = self.cmb_model.itemData(self.cmb_model.currentIndex())
        if cur_model_key:
            mc = get_model_set_config(str(cur_model_key))
            if mc and mc.get("max_volume") is not None:
                try:
                    initial_volume = float(mc["max_volume"])
                except (TypeError, ValueError):
                    pass
        self.spn_target.setValue(initial_volume)
        card.add(self._field_row("cube", "Объём наполнения", self.spn_target))
        lay.addWidget(card)

        lay.addSpacing(6)
        run_row = QHBoxLayout()
        run_row.setSpacing(8)
        btn_reset = IconButton("rotate_ccw", "Сбросить выбор", size=38,
                               icon_size=18, filled=True)
        btn_reset.clicked.connect(self._reset_selections)
        self.btn_run = QPushButton("Сгенерировать")
        self.btn_run.setProperty("variant", "primary")
        self.btn_run.setIcon(icons.icon("play", "#FFFFFF", 14,
                                        active_color="#FFFFFF"))
        self.btn_run.setIconSize(QSize(14, 14))
        self.btn_run.setMinimumHeight(38)
        self.btn_run.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_run.setToolTip("Насыпать груз в выбранный кузов")
        self.btn_run.clicked.connect(self._emit_run_requested)
        run_row.addWidget(btn_reset)
        run_row.addWidget(self.btn_run, 1)
        lay.addLayout(run_row)
        lay.addStretch(1)
        return page

    # ==================================================================
    # Вкладка «Записи»
    # ==================================================================
    def _build_records_page(self) -> QWidget:
        page, lay = self._page()
        self._recons: list[Reconstruction] = load_reconstructions()

        head = QHBoxLayout()
        head.setSpacing(8)
        head.addWidget(self._title("Записи"))
        self.lbl_recon_count = QLabel("")
        self.lbl_recon_count.setProperty("role", "chip-idle")
        head.addWidget(self.lbl_recon_count, 0, Qt.AlignmentFlag.AlignVCenter)
        head.addStretch(1)
        self.btn_load_more = IconButton("arrow_down_circle", "Загрузить ещё",
                                        size=30, icon_size=18)
        self.btn_load_more.clicked.connect(self._on_load_more)
        head.addWidget(self.btn_load_more)
        lay.addLayout(head)

        self.lst_recon = QListWidget()
        self.lst_recon.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.lst_recon.setStyleSheet(
            "QListWidget::item { padding: 0px; margin: 0px; border-radius: 9px; }")
        self.lst_recon.setSpacing(1)
        self.lst_recon.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.lst_recon.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self.lst_recon.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.lst_recon.customContextMenuRequested.connect(self._on_recon_context_menu)
        lay.addWidget(self.lst_recon, 1)
        self._fill_recon_list()

        # ---- выбранная запись -------------------------------------------
        self._details = Card(padding=(12, 10, 10, 10), spacing=6)
        self._details_rows: dict[str, QLabel] = {}
        for key, ic, tip in (("model", "truck", "Модель"),
                             ("filler", "heap", "Наполнитель"),
                             ("target", "cube", "Целевой объём"),
                             ("time", "clock", "Время")):
            v = label("—", size=12)
            v.setToolTip(tip)
            self._details.add(self._field_row(ic, tip, v))
            self._details_rows[key] = v
        self._file_lbl = label("—", role="caption")
        self._file_lbl.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._btn_copy = IconButton("copy", "Скопировать имя файла", size=24,
                                    icon_size=14, color=COLOR_TEXT_MUTED)
        self._btn_copy.clicked.connect(
            lambda: self._copy_filename_to_clipboard(
                getattr(self._selected_rec, "name", "") or "", self._btn_copy))
        self._details.add(self._field_row("doc", "Файл", self._file_lbl,
                                          self._btn_copy))
        lay.addWidget(self._details)

        # ---- запуск -------------------------------------------------------
        foot = QHBoxLayout()
        foot.setSpacing(8)
        foot.addWidget(icon_label("film", COLOR_TEXT_MUTED, 16,
                                  "Кинематографичный показ"))
        foot.addWidget(label("Кино", size=12, color=COLOR_TEXT_MUTED))
        self.chk_cinematic = Switch(
            _load_cinematic_enabled(),
            "Кино: показывать реконструкцию по проезду кинематографично —\n"
            "снимок, лидар, поиск кузова, этапы расчёта. Esc — пропустить")
        self.chk_cinematic.toggled.connect(_save_cinematic_enabled)
        foot.addWidget(self.chk_cinematic)
        foot.addStretch(1)
        self.btn_run_recon = QPushButton("Реконструировать")
        self.btn_run_recon.setProperty("variant", "primary")
        self.btn_run_recon.setIcon(icons.icon("sparkles", "#FFFFFF", 15,
                                              active_color="#FFFFFF"))
        self.btn_run_recon.setIconSize(QSize(15, 15))
        self.btn_run_recon.setMinimumHeight(36)
        self.btn_run_recon.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_run_recon.clicked.connect(self._emit_recon_run_requested)
        foot.addWidget(self.btn_run_recon)
        lay.addLayout(foot)
        return page

    def _row_text_width(self) -> int:
        return self.PANEL_WIDTH - 28 - 8 - 32 - 20 - 28 - 14

    def _fill_recon_list(self) -> None:
        self.lst_recon.clear()
        if self._recons:
            for idx, rec in enumerate(self._recons):
                row_w = ReconRowWidget(rec, max_text_width=self._row_text_width())
                row_w.viewClicked.connect(lambda i=idx: self._on_view_requested(i))
                item = QListWidgetItem()
                item.setSizeHint(QSize(0, row_w.ROW_FIXED_HEIGHT))
                item.setData(Qt.ItemDataRole.UserRole, idx)
                self.lst_recon.addItem(item)
                self.lst_recon.setItemWidget(item, row_w)
        else:
            placeholder = QListWidgetItem("Записей нет")
            placeholder.setFlags(Qt.ItemFlag.NoItemFlags)
            placeholder.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.lst_recon.addItem(placeholder)
        self._refresh_recon_card_count()

    # ==================================================================
    # Вкладка «Камера»
    # ==================================================================
    def _build_camera_page(self) -> QWidget:
        page, lay = self._page()
        lay.addWidget(self._title("Камера"))

        # ---- Объектив ---------------------------------------------------
        lay.addLayout(self._section("Объектив"))
        lens = Card(spacing=10)
        self.fov_slider = QSlider(Qt.Orientation.Horizontal)
        self.fov_slider.setRange(self._FOV_MIN, self._FOV_MAX)
        self.fov_slider.setValue(self._FOV_DEFAULT)
        self.fov_value_lbl = self._value_label(f"{self._FOV_DEFAULT}°")

        def _on_fov(val: int):
            self.fov_value_lbl.setText(f"{int(val)}°")
            self.fovChanged.emit(float(val))

        self.fov_slider.valueChanged.connect(_on_fov)
        lens.add(self._field_row("angle", "Угол обзора (FOV)", self.fov_slider,
                                 self.fov_value_lbl))

        self.roll_slider = QSlider(Qt.Orientation.Horizontal)
        self.roll_slider.setRange(self._ROLL_MIN, self._ROLL_MAX)
        self.roll_slider.setValue(0)
        self.roll_value_lbl = self._value_label("0°")

        def _on_roll(val: int):
            self.roll_value_lbl.setText(f"{int(val)}°")
            self.rollChanged.emit(float(val))

        self.roll_slider.valueChanged.connect(_on_roll)
        self.btn_roll_reset = IconButton("rotate_ccw", "Выровнять горизонт",
                                         size=24, icon_size=14,
                                         color=COLOR_TEXT_MUTED)
        # setValue(0) -> valueChanged -> rollChanged: камера выравнивается.
        self.btn_roll_reset.clicked.connect(lambda: self.roll_slider.setValue(0))
        roll_trailing = QWidget()
        rt = QHBoxLayout(roll_trailing)
        rt.setContentsMargins(0, 0, 0, 0)
        rt.setSpacing(2)
        rt.addWidget(self.roll_value_lbl)
        rt.addWidget(self.btn_roll_reset)
        lens.add(self._field_row("horizon", "Крен (поворот вокруг центра кадра)",
                                 self.roll_slider, roll_trailing))
        lay.addWidget(lens)

        # ---- Снимок стенда поверх вида -----------------------------------
        self.btn_ref_toggle = IconButton("eye", "Показать / скрыть снимок",
                                         size=26, icon_size=16, checkable=True)
        self.btn_ref_toggle.setChecked(True)

        def _on_toggle(checked: bool):
            self.btn_ref_toggle.set_icon("eye" if checked else "eye_off")
            self.referenceVisibleToggled.emit(bool(checked))

        self.btn_ref_toggle.toggled.connect(_on_toggle)
        lay.addLayout(self._section("Снимок поверх вида", self.btn_ref_toggle))

        self._top_ref_controls = Card()
        self.ref_opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.ref_opacity_slider.setRange(0, 100)
        self.ref_opacity_slider.setValue(50)
        self.ref_opacity_lbl = self._value_label("50%")

        def _on_opacity(val: int):
            self.ref_opacity_lbl.setText(f"{int(val)}%")
            self.referenceOpacityChanged.emit(float(val) / 100.0)

        self.ref_opacity_slider.valueChanged.connect(_on_opacity)
        self._top_ref_controls.add(self._field_row(
            "halfcircle", "Прозрачность снимка", self.ref_opacity_slider,
            self.ref_opacity_lbl))
        lay.addWidget(self._top_ref_controls)

        # ---- Опорные точки ------------------------------------------------
        lay.addLayout(self._section("Опорные точки"))
        self._ref_controls_holder = QWidget()
        grid = QGridLayout(self._ref_controls_holder)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(6)
        self.btn_pick_points = TileButton(
            "cursor", "Отметить",
            "Кликайте опорные точки на кузове (любое число).\n"
            "ПКМ или Esc — завершить и построить наполнение", checkable=True)
        self.btn_pick_points.toggled.connect(
            lambda checked: self.pointPickingToggled.emit(bool(checked)))
        self.btn_auto_points = TileButton(
            "sparkles", "Авто",
            "Найти опорные точки автоматически и построить наполнение")
        self.btn_auto_points.clicked.connect(
            lambda _=False: self.autoPointsRequested.emit())
        self.btn_point_viz = TileButton(
            "target", "Показать", "Показать использованные опорные точки",
            checkable=True)
        self.btn_point_viz.toggled.connect(
            lambda checked: self.pointVizToggled.emit(bool(checked)))
        self.btn_pick_reset = TileButton(
            "rotate_ccw", "Сброс", "Сбросить точки и реконструкцию")
        self.btn_pick_reset.clicked.connect(
            lambda _=False: self.pointsResetRequested.emit())
        for i, b in enumerate((self.btn_pick_points, self.btn_auto_points,
                               self.btn_point_viz, self.btn_pick_reset)):
            grid.addWidget(b, 0, i)
        lay.addWidget(self._ref_controls_holder)

        self._ref_hint = QLabel(
            "Выберите снимок стенда или карту глубины во вкладке «Записи» — "
            "здесь станут доступны подложка и опорные точки.")
        self._ref_hint.setProperty("role", "caption")
        self._ref_hint.setWordWrap(True)
        self._ref_hint.setContentsMargins(4, 4, 4, 0)
        lay.addWidget(self._ref_hint)
        lay.addStretch(1)

        self._set_ref_enabled(False)
        return page

    def _set_ref_enabled(self, on: bool) -> None:
        # Заливка слайдера не реагирует на :disabled из общей таблицы
        # стилей — гасим её явно.
        self.ref_opacity_slider.setStyleSheet(
            "" if on else "QSlider::sub-page:horizontal { background: #48484A; }")
        self._top_ref_controls.setEnabled(on)
        self._ref_controls_holder.setEnabled(on)
        self.btn_ref_toggle.setEnabled(on)
        self._ref_hint.setVisible(not on)

    # ==================================================================
    # Вкладка «Датасет»
    # ==================================================================
    _OUTPUT_META = {
        "color":        ("photo",  "Цвет"),
        "depth":        ("depth",  "Глубина"),
        "segmentation": ("mask",   "Маска"),
        "lidar":        ("lidar",  "Лидар"),
        "json":         ("braces", "JSON"),
    }

    def _build_dataset_page(self) -> QWidget:
        page, lay = self._page()
        lay.addWidget(self._title("Датасет"))
        lay.addLayout(self._section("Что будет снято"))

        card = Card(padding=(14, 12, 12, 12), spacing=10)
        metric_row = QHBoxLayout()
        metric_row.setSpacing(6)
        self.lbl_ds_total = QLabel("—")
        self.lbl_ds_total.setProperty("role", "metric")
        metric_row.addWidget(self.lbl_ds_total, 0, Qt.AlignmentFlag.AlignBottom)
        unit = QLabel("кадров")
        unit.setProperty("role", "metric-unit")
        metric_row.addWidget(unit, 0, Qt.AlignmentFlag.AlignBottom)
        metric_row.addStretch(1)
        self.lbl_ds_formula = label("", mono=True, size=12, color=COLOR_TEXT_MUTED)
        self.lbl_ds_formula.setToolTip("наполнений × кадров с каждого")
        metric_row.addWidget(self.lbl_ds_formula, 0, Qt.AlignmentFlag.AlignBottom)
        card.add(metric_row)

        self._ds_chips = QWidget()
        self._ds_chips_lay = QGridLayout(self._ds_chips)
        self._ds_chips_lay.setContentsMargins(0, 0, 0, 0)
        self._ds_chips_lay.setHorizontalSpacing(6)
        self._ds_chips_lay.setVerticalSpacing(6)
        card.add(self._ds_chips)
        card.add(hline())

        self.lbl_ds_dir = label("", role="caption")
        self.lbl_ds_dir.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.btn_ds_open = IconButton("arrow_up_right", "Открыть папку",
                                      size=24, icon_size=14,
                                      color=COLOR_TEXT_MUTED)
        self.btn_ds_open.clicked.connect(self._open_dataset_dir)
        card.add(self._field_row("folder", "Папка датасета", self.lbl_ds_dir,
                                 self.btn_ds_open))
        lay.addWidget(card)

        lay.addSpacing(6)
        btns = QHBoxLayout()
        btns.setSpacing(8)
        self.btn_dataset_setup = QPushButton("Настроить")
        self.btn_dataset_setup.setIcon(icons.icon("sliders", COLOR_TEXT, 15))
        self.btn_dataset_setup.setIconSize(QSize(15, 15))
        self.btn_dataset_setup.setMinimumHeight(38)
        self.btn_dataset_setup.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_dataset_setup.setToolTip(
            "Что сохранять, куда, как варьировать наполнение, камеру и свет")
        self.btn_dataset_setup.clicked.connect(self.datasetSettingsRequested.emit)
        self.btn_dataset_start = QPushButton("Снять")
        self.btn_dataset_start.setProperty("variant", "primary")
        self.btn_dataset_start.setIcon(icons.icon("record", "#FFFFFF", 15,
                                                  active_color="#FFFFFF"))
        self.btn_dataset_start.setIconSize(QSize(15, 15))
        self.btn_dataset_start.setMinimumHeight(38)
        self.btn_dataset_start.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_dataset_start.setToolTip("Запустить съёмку с текущими настройками")
        self.btn_dataset_start.clicked.connect(self.datasetStartRequested.emit)
        btns.addWidget(self.btn_dataset_setup, 1)
        btns.addWidget(self.btn_dataset_start, 1)
        lay.addLayout(btns)
        lay.addStretch(1)
        self._ds_dir_full = ""
        return page

    def set_dataset_summary(self, count: int, per_fill: int, total: int,
                            outputs, out_dir: str) -> None:
        """Сводка на вкладке «Датасет» (зовёт MainWindow после правки конфига)."""
        self.lbl_ds_total.setText(str(total))
        self.lbl_ds_formula.setText(f"{count} × {per_fill}")
        while self._ds_chips_lay.count():
            item = self._ds_chips_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        row = None
        for i, key in enumerate(outputs):
            ic, text = self._OUTPUT_META.get(key, ("doc", key))
            if i % 3 == 0:
                holder = QWidget()
                row = QHBoxLayout(holder)
                row.setContentsMargins(0, 0, 0, 0)
                row.setSpacing(6)
                row.addStretch(1)
                self._ds_chips_lay.addWidget(holder, i // 3, 0)
                # новые чипы встают перед распоркой — ряд прижат влево
            chip = Chip(ic, text)
            row.insertWidget(row.count() - 1, chip)
        self._ds_dir_full = str(out_dir or "")
        self.lbl_ds_dir.setText(_elide(self._ds_dir_full, self.lbl_ds_dir.font(),
                                       self.PANEL_WIDTH - 130,
                                       Qt.TextElideMode.ElideMiddle))
        self.lbl_ds_dir.setToolTip(self._ds_dir_full)

    def set_dataset_error(self, text: str) -> None:
        self.lbl_ds_total.setText("—")
        self.lbl_ds_formula.setText("")
        self.lbl_ds_dir.setText(text)

    def _open_dataset_dir(self) -> None:
        path = self._ds_dir_full
        if not path:
            return
        if not os.path.isabs(path):
            path = os.path.join(PROJECT_ROOT, path)
        try:
            os.makedirs(path, exist_ok=True)
            os.startfile(path)  # type: ignore[attr-defined]
        except Exception as exc:
            print(f"[RightPanel] папка датасета не открылась: {exc}")

    # ==================================================================
    # Качество графики
    # ==================================================================
    def _show_graphics_menu(self) -> None:
        cur = graphics_settings.load_saved() or graphics_settings.DEFAULT_PRESET
        menu = QMenu(self)
        head = menu.addAction("Качество графики")
        head.setEnabled(False)
        group = QActionGroup(menu)
        group.setExclusive(True)
        for pkey in graphics_settings.PRESET_ORDER:
            act = menu.addAction(graphics_settings.PRESETS[pkey]["name"])
            act.setCheckable(True)
            act.setChecked(pkey == cur)
            act.setData(pkey)
            group.addAction(act)
        menu.addSeparator()
        note = menu.addAction("Применится после перезапуска")
        note.setEnabled(False)
        chosen = menu.exec(self.btn_graphics.mapToGlobal(
            QPoint(self.btn_graphics.width() - 4, self.btn_graphics.height() + 4))
            - QPoint(menu.sizeHint().width(), 0))
        if chosen is not None and chosen.data() and chosen.data() != cur:
            self.graphicsPresetChanged.emit(str(chosen.data()))

    # ==================================================================
    # Состояние выбора точек / объектива (зовёт MainWindow)
    # ==================================================================
    def set_point_count(self, n: int) -> None:
        n = max(0, int(n))
        self.btn_pick_points.set_badge(str(n) if n else "")

    def set_picking_active(self, active: bool) -> None:
        btn = self.btn_pick_points
        blocked = btn.blockSignals(True)
        btn.setChecked(bool(active))
        btn.blockSignals(blocked)
        btn.update()

    def set_fov_value(self, fov: float) -> None:
        """Отразить FOV, выставленный кодом, без повторного fovChanged."""
        try:
            v = int(round(float(fov)))
        except (TypeError, ValueError):
            return
        v = max(self._FOV_MIN, min(self._FOV_MAX, v))
        blocked = self.fov_slider.blockSignals(True)
        self.fov_slider.setValue(v)
        self.fov_slider.blockSignals(blocked)
        self.fov_value_lbl.setText(f"{v}°")

    def set_roll_value(self, roll: float) -> None:
        """Отразить крен, выставленный кодом, без повторного rollChanged."""
        try:
            v = int(round(float(roll)))
        except (TypeError, ValueError):
            return
        while v > self._ROLL_MAX:
            v -= 360
        while v < self._ROLL_MIN:
            v += 360
        blocked = self.roll_slider.blockSignals(True)
        self.roll_slider.setValue(v)
        self.roll_slider.blockSignals(blocked)
        self.roll_value_lbl.setText(f"{v}°")

    # ==================================================================
    # Список записей
    # ==================================================================
    def _on_recon_changed(self, current: QListWidgetItem | None, _prev) -> None:
        if current is None:
            return
        idx = current.data(Qt.ItemDataRole.UserRole)
        if not isinstance(idx, int) or not (0 <= idx < len(self._recons)):
            return
        rec = self._recons[idx]
        self._selected_rec = rec
        self._populate_details(rec)
        self.btn_run_recon.setEnabled(True)
        self.reconstructionSelected.emit(str(rec.name))
        self._emit_stand_reference(rec)

    def _on_recon_clicked(self, item: QListWidgetItem) -> None:
        """Клик только выбирает запись; реконструкцию запускает кнопка."""
        if item is None:
            return
        idx = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(idx, int) or not (0 <= idx < len(self._recons)):
            return
        self._selected_rec = self._recons[idx]
        self._populate_details(self._selected_rec)
        self.btn_run_recon.setEnabled(True)
        self._emit_stand_reference(self._selected_rec)

    def _emit_stand_reference(self, rec: Reconstruction | None) -> None:
        """Снимок стенда или серверная карта глубины включает подложку и
        опорные точки на вкладке «Камера» (на её иконке — точка-подсказка)."""
        is_ref = bool(rec is not None and rec.data_type in ("stand", "depth"))
        self.standReferenceSelected.emit(rec if is_ref else None)
        was = self._top_ref_controls.isEnabled()
        self._set_ref_enabled(is_ref)
        if is_ref and not was and self.tabs.current() != "camera":
            self.tabs.set_badge("camera", True)
        if not is_ref:
            self.tabs.set_badge("camera", False)

    def _on_recon_context_menu(self, pos: QPoint) -> None:
        item = self.lst_recon.itemAt(pos)
        if item is None:
            return
        idx = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(idx, int) or not (0 <= idx < len(self._recons)):
            return
        rec = self._recons[idx]
        name = str(rec.name or "").strip()
        menu = QMenu(self.lst_recon)
        act_view = menu.addAction(icons.icon("zoom_in", COLOR_TEXT, 16),
                                  "Снимок и подробности")
        act_copy = menu.addAction(icons.icon("copy", COLOR_TEXT, 16),
                                  "Скопировать имя файла")
        act_copy.setEnabled(bool(name))
        chosen = menu.exec(self.lst_recon.viewport().mapToGlobal(pos))
        if chosen is act_copy:
            QApplication.clipboard().setText(name)
        elif chosen is act_view:
            self._on_view_requested(idx)

    def _on_view_requested(self, idx: int) -> None:
        if not (0 <= idx < len(self._recons)):
            return
        self.lst_recon.setCurrentRow(idx)
        owner_top = self._owner.window() if self._owner else None
        dlg = RecordPhotoOverlay(self._recons[idx], parent=owner_top)
        try:
            dlg.exec()
        finally:
            self._reposition()
            self.raise_()

    def cinematic_enabled(self) -> bool:
        return bool(self.chk_cinematic.isChecked())

    def _emit_recon_run_requested(self) -> None:
        rec = getattr(self, "_selected_rec", None)
        if rec is not None:
            self.reconstructionRunRequested.emit(rec)

    def _on_load_more(self) -> None:
        """Подтянуть ещё страницу записей, сохранив выбор."""
        new_limit = getattr(self, "_recon_limit", RECON_PAGE_SIZE) + RECON_PAGE_SIZE
        self._recon_limit = new_limit
        self.btn_load_more.setEnabled(False)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            new_recons = load_reconstructions(limit=new_limit)
        except Exception as exc:
            print(f"[RightPanel] load_reconstructions failed: {exc}")
            return
        finally:
            QApplication.restoreOverrideCursor()
            self.btn_load_more.setEnabled(True)

        sel_name = getattr(self._selected_rec, "name", None) if self._selected_rec else None
        self._recons = new_recons
        self._fill_recon_list()
        if self._recons:
            row = 0
            if sel_name:
                for i, r in enumerate(self._recons):
                    if r.name == sel_name:
                        row = i
                        break
            self.lst_recon.setCurrentRow(row)

    def _refresh_recon_card_count(self) -> None:
        self.lbl_recon_count.setText(str(len(self._recons or [])))

    def _populate_details(self, rec: Reconstruction | None) -> None:
        if rec is None:
            self._details.hide()
            return
        self._details.show()
        target = (f"{rec.target_volume:.2f} м³"
                  if rec.target_volume is not None else "—")
        values = {
            "model": rec.model or "—",
            "filler": rec.filler or "—",
            "target": target,
            "time": _format_short_dt(rec),
        }
        width = self.PANEL_WIDTH - 28 - 24 - 30
        for key, v in values.items():
            lbl = self._details_rows[key]
            lbl.setText(_elide(v, lbl.font(), width))
            lbl.setToolTip(v)
        name = rec.name or "—"
        self._file_lbl.setText(_elide(name, self._file_lbl.font(), width - 34,
                                      Qt.TextElideMode.ElideMiddle))
        self._file_lbl.setToolTip(name)
        self._btn_copy.setEnabled(bool(rec.name))

    def _copy_filename_to_clipboard(self, name: str, btn) -> None:
        if not name:
            return
        QApplication.clipboard().setText(name)
        btn.set_icon("check")
        btn.setToolTip("Скопировано")
        QTimer.singleShot(1200, lambda b=btn: (
            b.set_icon("copy"), b.setToolTip("Скопировать имя файла")))

    # ==================================================================
    # Модели / текстуры
    # ==================================================================
    @staticmethod
    def _model_count_label(count: int) -> str:
        """«12 наборов» с правильным падежом."""
        tail = count % 100
        if 11 <= tail <= 14:
            word = "наборов"
        elif count % 10 == 1:
            word = "набор"
        elif count % 10 in (2, 3, 4):
            word = "набора"
        else:
            word = "наборов"
        return f"{count} {word}"

    def _refresh_model_spec(self) -> None:
        spec = getattr(self, "_spec_rows", None)
        if not spec:
            return
        info = self.cmb_model.current_info()
        if info is None:
            self._spec.hide()
            return
        self._spec.show()
        vol = (f"{info.volume:.2f}".rstrip("0").rstrip(".") + " м³"
               if info.volume is not None else "—")
        dims = (" × ".join(f"{v:.2f}" for v in info.dims)
                if info.dims else "—")
        values = {
            "volume": (vol, info.volume_kind or ""),
            "axles": (info.axles or info.chassis or "—", ""),
            "dims": (dims, "Д × Ш × В" if info.dims else ""),
            "kit": (info.kit, info.source_label),
        }
        for key, (text, note) in values.items():
            v, n = spec[key]
            v.setText(text)
            n.setText(note)

    def _on_model_index_changed(self, idx: int) -> None:
        self._refresh_model_spec()
        key = self.cmb_model.itemData(idx)
        if key:
            mc = get_model_set_config(str(key))
            if mc and mc.get("max_volume") is not None:
                try:
                    self.spn_target.setValue(float(mc["max_volume"]))
                except (TypeError, ValueError):
                    pass
            self.modelSetChanged.emit(str(key))

    def _on_texture_index_changed(self, idx: int) -> None:
        key = self.cmb_texture.itemData(idx)
        if key:
            self.textureSetChanged.emit(str(key))

    def update_texture_sets(self, texture_sets_list, default_key=None) -> None:
        """
        Перезалить список текстур (пары key, display). Сигнал на время
        перезалива заблокирован — подписчики увидят только итог.
        """
        items = []
        for entry in (texture_sets_list or []):
            try:
                key, display = entry
            except (TypeError, ValueError):
                continue
            if not key or key == "default":
                continue
            items.append((str(key), str(display) if display else str(key)))

        self.cmb_texture.blockSignals(True)
        try:
            self.cmb_texture.clear()
            if items:
                target_index = 0
                for i, (key, display) in enumerate(items):
                    self.cmb_texture.addItem(display, userData=key)
                    if default_key and key == default_key:
                        target_index = i
                self.cmb_texture.setEnabled(True)
                self.cmb_texture.setCurrentIndex(target_index)
            else:
                self.cmb_texture.addItem("Текстуры не найдены", userData=None)
                self.cmb_texture.setEnabled(False)
        finally:
            self.cmb_texture.blockSignals(False)

    # ==================================================================
    # Генератор кузова
    # ==================================================================
    def _init_bodygen_state(self) -> None:
        try:
            from src.bodygen import probe
            info = probe()
        except Exception as exc:
            info = {"available": False, "reason": f"модуль не загружен: {exc}"}
        if info.get("available"):
            chassis = ", ".join(info.get("chassis") or []) or "нет"
            self.btn_bodygen.setToolTip(
                f"Собрать кузов по скану…\nШасси: {chassis}")
        else:
            self.btn_bodygen.setEnabled(False)
            self.btn_bodygen.setToolTip(
                "Генератор кузова недоступен: "
                + (info.get("reason") or "нет модуля"))

    def set_bodygen_status(self, text: str, busy: bool = False) -> None:
        """Ход сборки под выбором кузова; на время сборки кнопка заблокирована."""
        self.btn_bodygen.setEnabled(not busy)
        self.btn_bodygen.set_icon("clock" if busy else "wand")
        # Пока идёт сборка, из очереди запускать вторую нечем: поток генератора
        # один, и вторая пошла бы писать в тот же каталог.
        self.btn_worklist.setEnabled(not busy)
        self._set_status_line(text)

    def set_model_status(self, text: str) -> None:
        """Короткое сообщение под выбором кузова (итог удаления, загрузки)."""
        self._set_status_line(text)

    def _set_status_line(self, text: str) -> None:
        text = str(text or "").strip()
        self.lbl_model_status.setText(text)
        self.lbl_model_status.setVisible(bool(text))

    def reload_model_sets(self, select_key: str | None = None) -> None:
        """Перечитать наборы моделей (после генерации / удаления / загрузки)."""
        try:
            infos = load_model_sets_detailed()
        except Exception as exc:
            print(f"[RightPanel] не удалось перечитать модели: {exc}")
            return
        self.cmb_model.blockSignals(True)
        self.cmb_model.clear()
        self.cmb_model.set_details(infos)
        for info in infos:
            self.cmb_model.addItem(info.display, userData=info.key)
        self.cmb_model.setEnabled(bool(infos))
        self._model_count.setText(self._model_count_label(len(infos)))
        index = 0
        if select_key:
            found = self.cmb_model.findData(select_key)
            if found >= 0:
                index = found
        self.cmb_model.setCurrentIndex(index)
        self.cmb_model.blockSignals(False)
        self._refresh_model_spec()
        if select_key and self.cmb_model.currentData() == select_key:
            self.modelSetChanged.emit(str(select_key))

    def current_model_key(self):
        return self.cmb_model.itemData(self.cmb_model.currentIndex())

    def model_info(self, key):
        """Характеристики набора (`ModelSetInfo`) или None."""
        return self.cmb_model.info_for(key)

    def set_current_model_key(self, key) -> bool:
        """
        Выставить набор в выборе без сигнала (модель уже загружена
        вызывающим кодом), синхронизировав целевой объём.
        """
        if key is None:
            return False
        for i in range(self.cmb_model.count()):
            if self.cmb_model.itemData(i) == key:
                self.cmb_model.blockSignals(True)
                try:
                    self.cmb_model.setCurrentIndex(i)
                finally:
                    self.cmb_model.blockSignals(False)
                mc = get_model_set_config(str(key))
                if mc and mc.get("max_volume") is not None:
                    try:
                        self.spn_target.setValue(float(mc["max_volume"]))
                    except (TypeError, ValueError):
                        pass
                self._refresh_model_spec()
                return True
        return False

    def current_texture_key(self):
        return self.cmb_texture.itemData(self.cmb_texture.currentIndex())

    def current_target_volume(self) -> float:
        try:
            return float(self.spn_target.value())
        except Exception:
            return 0.0

    def _emit_run_requested(self) -> None:
        self.runRequested.emit({
            "model_key":     self.current_model_key(),
            "texture_key":   self.current_texture_key(),
            "target_volume": self.current_target_volume(),
        })

    def _reset_selections(self) -> None:
        if self.cmb_model.count():
            self.cmb_model.setCurrentIndex(0)
        default_tex = get_default_texture_set_key()
        if default_tex is not None:
            for i in range(self.cmb_texture.count()):
                if self.cmb_texture.itemData(i) == default_tex:
                    self.cmb_texture.setCurrentIndex(i)
                    break
            else:
                self.cmb_texture.setCurrentIndex(0)
        elif self.cmb_texture.count():
            self.cmb_texture.setCurrentIndex(0)
        if self._recons:
            self.lst_recon.setCurrentRow(0)
