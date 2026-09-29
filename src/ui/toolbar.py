# toolbar.py
# ---------------------------------------------------------------------------
# Верхняя плавающая панель вида — всё, что касается «откуда смотрим»:
#
#   [✥ ⌖ 🚚]  │  (1)(2)(3) 🔖  │  ☀ ━━━●━━ 15:00  │  ▣  ⌨
#
#   * режим камеры: свободная / стационарная / бортовая;
#   * три пользовательских вида + «запомнить вид» (слоты мигают, пока
#     ждут выбора, ПКМ по слоту — меню «сохранить / очистить»);
#   * время суток;
#   * показать/скрыть превью глубины;
#   * справка по клавишам (вместо постоянной карточки «Управление»).
#
# Логика (что делать с камерой) остаётся в MainWindow — панель только
# рисует состояние и отдаёт сигналы.
# ---------------------------------------------------------------------------

from __future__ import annotations

from PyQt6.QtCore import QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import (
    QAbstractButton, QGridLayout, QHBoxLayout, QLabel, QSlider, QWidget,
)

from src.ui import icons
from src.ui.hud import (
    FloatingPanel, IconButton, Keycap, Popover, SegmentedControl, _HoverMixin,
    label, vline,
)
from src.ui.ui_theme import (
    COLOR_ACCENT, COLOR_TEXT, COLOR_TEXT_DIM, COLOR_TEXT_MUTED, FONT_MONO,
)


class PresetSlot(_HoverMixin, QAbstractButton):
    """
    Слот пользовательского вида. Состояния (задаёт MainWindow):
      empty | filled | selected | blink_on | blink_off
    """

    def __init__(self, number: int, parent: QWidget | None = None):
        super().__init__(parent)
        self._n = number
        self._state = "empty"
        self._hover = False
        self.setFixedSize(28, 28)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    def set_state(self, state: str) -> None:
        if state != self._state:
            self._state = state
            self.update()

    def state(self) -> str:
        return self._state

    def sizeHint(self) -> QSize:
        return self.size()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        r = QRectF(self.rect()).adjusted(2, 2, -2, -2)
        st = self._state
        accent = QColor(COLOR_ACCENT)
        fg = QColor(COLOR_TEXT)
        pen = Qt.PenStyle.NoPen
        if st == "selected":
            bg = accent
            fg = QColor("#FFFFFF")
        elif st == "filled":
            bg = QColor(255, 255, 255, 44 if self._hover else 28)
        elif st == "blink_on":
            bg = QColor(accent.red(), accent.green(), accent.blue(), 120)
            fg = QColor("#FFFFFF")
        else:  # empty / blink_off
            bg = QColor(255, 255, 255, 18 if self._hover else 0)
            fg = QColor(COLOR_TEXT_DIM if st == "empty" else COLOR_TEXT_MUTED)
            pen = QPen(QColor(255, 255, 255, 60), 1, Qt.PenStyle.DashLine)
            if st == "blink_off":
                pen = QPen(accent, 1.2, Qt.PenStyle.DashLine)
        if self.isDown():
            bg = bg.darker(130) if bg.alpha() else QColor(255, 255, 255, 14)
        p.setBrush(bg)
        p.setPen(pen)
        p.drawEllipse(r)
        f = QFont(self.font())
        f.setPixelSize(12)
        f.setWeight(QFont.Weight.DemiBold)
        p.setFont(f)
        p.setPen(fg)
        p.drawText(r, int(Qt.AlignmentFlag.AlignCenter), str(self._n))
        p.end()


class ViewToolbar(FloatingPanel):
    modeChanged = pyqtSignal(str)          # free | stationary | onboard
    presetClicked = pyqtSignal(int)
    presetMenuRequested = pyqtSignal(int)
    saveArmToggled = pyqtSignal(bool)
    daytimeChanged = pyqtSignal(int)       # минуты от полуночи
    pipToggled = pyqtSignal(bool)

    def __init__(self, owner: QWidget, margin: int = 16):
        super().__init__(owner, anchor="top-center", margin=margin,
                         padding=(5, 5, 9, 5), layout="h")
        lay = self.body_layout
        lay.setSpacing(6)

        # ---- режим камеры -------------------------------------------------
        self.mode_control = SegmentedControl([
            ("free", "move", "Свободная камера · WASD, ПКМ — обзор"),
            ("stationary", "tripod", "Стационарная камера"),
            ("onboard", "truck", "Бортовая камера (из конфига модели)"),
        ], height=30, segment_width=40, icon_size=17)
        self.mode_control.changed.connect(self.modeChanged.emit)
        lay.addWidget(self.mode_control)
        lay.addSpacing(4)
        lay.addWidget(vline(20), 0, Qt.AlignmentFlag.AlignVCenter)
        lay.addSpacing(4)

        # ---- пользовательские виды ---------------------------------------
        self.preset_btns: dict[int, PresetSlot] = {}
        for slot in (0, 1, 2):
            btn = PresetSlot(slot + 1)
            btn.setToolTip(f"Вид {slot + 1}\nЛКМ — перейти · ПКМ — сохранить / очистить")
            btn.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            btn.customContextMenuRequested.connect(
                lambda _p, s=slot: self.presetMenuRequested.emit(s))
            btn.clicked.connect(lambda _c=False, s=slot: self.presetClicked.emit(s))
            self.preset_btns[slot] = btn
            lay.addWidget(btn)
        self.btn_save = IconButton(
            "bookmark_plus",
            "Запомнить вид: нажмите, затем выберите слот 1–3 и\n"
            "отметьте опорные точки на кузове. ПКМ или Esc — готово",
            size=30, icon_size=17, checkable=True)
        self.btn_save.toggled.connect(self.saveArmToggled.emit)
        lay.addWidget(self.btn_save)
        lay.addSpacing(2)
        lay.addWidget(vline(20), 0, Qt.AlignmentFlag.AlignVCenter)
        lay.addSpacing(6)

        # ---- время суток --------------------------------------------------
        self._sun = QLabel()
        self._sun.setFixedSize(18, 18)
        self._sun.setToolTip("Время суток")
        lay.addWidget(self._sun, 0, Qt.AlignmentFlag.AlignVCenter)
        self.daytime_slider = QSlider(Qt.Orientation.Horizontal)
        self.daytime_slider.setRange(0, 1439)
        self.daytime_slider.setFixedWidth(118)
        self.daytime_slider.setToolTip("Время суток")
        self.daytime_slider.setCursor(Qt.CursorShape.PointingHandCursor)
        lay.addWidget(self.daytime_slider, 0, Qt.AlignmentFlag.AlignVCenter)
        self.daytime_label = label("15:00", mono=True, size=12, color=COLOR_TEXT)
        self.daytime_label.setFixedWidth(40)
        self.daytime_label.setAlignment(Qt.AlignmentFlag.AlignRight
                                        | Qt.AlignmentFlag.AlignVCenter)
        lay.addWidget(self.daytime_label, 0, Qt.AlignmentFlag.AlignVCenter)
        self.daytime_slider.valueChanged.connect(self._on_daytime)
        lay.addSpacing(4)
        lay.addWidget(vline(20), 0, Qt.AlignmentFlag.AlignVCenter)
        lay.addSpacing(2)

        # ---- превью и справка --------------------------------------------
        self.btn_pip = IconButton("pip", "Превью глубины", size=30,
                                  icon_size=18, checkable=True)
        self.btn_pip.setChecked(True)
        self.btn_pip.toggled.connect(self.pipToggled.emit)
        lay.addWidget(self.btn_pip)
        self.btn_help = IconButton("keyboard", "Управление", size=30,
                                   icon_size=18)
        lay.addWidget(self.btn_help)
        self._help = self._build_help()
        self.btn_help.clicked.connect(
            lambda: self._help.toggle_for(self.btn_help, "below"))

        self.daytime_slider.setValue(15 * 60)
        self._on_daytime(self.daytime_slider.value(), emit=False)

    # -- API -----------------------------------------------------------------
    def set_mode(self, mode: str) -> None:
        self.mode_control.set_current(mode)

    # -- внутреннее ----------------------------------------------------------
    def _on_daytime(self, mins: int, emit: bool = True) -> None:
        hh, mm = divmod(int(mins), 60)
        self.daytime_label.setText(f"{hh:02d}:{mm:02d}")
        night = hh < 6 or hh >= 20
        self._sun.setPixmap(icons.pixmap("moon" if night else "sun",
                                         COLOR_TEXT_MUTED, 17))
        if emit:
            self.daytimeChanged.emit(int(mins))

    def _build_help(self) -> Popover:
        pop = Popover(self, width=300)
        pop.body_layout.addWidget(label("Управление", role="headline"))
        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)
        rows = (
            (("W", "A", "S", "D"), "Движение"),
            (("Q", "E"), "Вниз / вверх"),
            (("Shift",), "Быстрее"),
            (("ПКМ",), "Осмотреться"),
            (("Esc",), "Готово · пропустить кино"),
        )
        for i, (keys, text) in enumerate(rows):
            keys_box = QHBoxLayout()
            keys_box.setSpacing(3)
            for k in keys:
                keys_box.addWidget(Keycap(k))
            keys_box.addStretch(1)
            holder = QWidget()
            holder.setLayout(keys_box)
            keys_box.setContentsMargins(0, 0, 0, 0)
            grid.addWidget(holder, i, 0)
            t = label(text, size=12, color=COLOR_TEXT_MUTED)
            t.setWordWrap(True)
            grid.addWidget(t, i, 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnMinimumWidth(0, 110)
        pop.body_layout.addLayout(grid)
        return pop
