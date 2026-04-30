# recon_list.py
# ---------------------------------------------------------------------------
# 2D → 3D reconstruction list — compact, information-dense rows.
#
# Row anatomy (height = 44 px):
#   │▎ CAR_NUMBER                                 TYPE │   <- line 1 (bold, chip)
#   │▎ 14:32 · 12 Apr · truck_A · 24.5 m³              │   <- line 2 (muted meta)
#     ^
#     1px accent bar (visible on hover/selection)
#
# Interaction model:
#   * Whole row is selectable.  Double-click / Enter → `item_activated(dict)`.
#   * No always-visible icon button — reduces visual noise, gives metadata
#     the room it needs.
#   * Hover paints a subtle wash; selected row gets the mint accent bar + tint.
# ---------------------------------------------------------------------------

from __future__ import annotations

from typing import Callable, Iterable

from PyQt6.QtCore import Qt, QSize, pyqtSignal, QEvent
from PyQt6.QtGui import QColor, QPainter, QPen, QBrush
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget,
    QListWidgetItem, QAbstractItemView, QStyle, QStyleOptionViewItem,
    QStyledItemDelegate,
)

from ui_theme import (
    COLOR_TEXT, COLOR_TEXT_MUTED, COLOR_ACCENT, COLOR_HAIRLINE,
)


RecordSource = Callable[[int, int], Iterable[dict]]


# ---------------------------------------------------------------------------
# Row widget — lightweight: three labels + a type chip.  All painting of the
# left accent bar and the hover/selection wash is handled by the delegate so
# the row widget doesn't need to know its selection state.
# ---------------------------------------------------------------------------
class _ReconRow(QWidget):
    def __init__(self, entry: dict, parent=None):
        super().__init__(parent)
        self.entry = entry

        v = QVBoxLayout(self)
        v.setContentsMargins(14, 6, 10, 6)      # left padding includes accent bar gutter
        v.setSpacing(2)

        # -- Line 1: car number + data-type chip -----------------------
        line1 = QHBoxLayout()
        line1.setContentsMargins(0, 0, 0, 0)
        line1.setSpacing(8)

        car = QLabel(entry.get("car_number", "—"))
        car.setStyleSheet(
            f"color: {COLOR_TEXT}; font-size: 12px; font-weight: 600;"
            f"letter-spacing: 0.3px; background: transparent;"
        )
        line1.addWidget(car)
        line1.addStretch(1)

        dtype = (entry.get("data_type") or "").upper()
        if dtype:
            chip = QLabel(dtype)
            chip.setProperty("role", "chip-idle" if entry.get("is_local") else "chip-live")
            chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
            line1.addWidget(chip)
        v.addLayout(line1)

        # -- Line 2: meta string ---------------------------------------
        meta_bits = []
        if entry.get("time"):
            meta_bits.append(self._short_time(entry["time"]))
        if entry.get("model"):
            # strip extension for readability
            m = str(entry["model"])
            if "." in m:
                m = m.rsplit(".", 1)[0]
            meta_bits.append(m)
        if entry.get("target_volume") not in (None, ""):
            meta_bits.append(f"{entry['target_volume']} m³")
        if entry.get("filler"):
            meta_bits.append(str(entry["filler"]))

        meta = QLabel("  ·  ".join(meta_bits) if meta_bits else "—")
        meta.setStyleSheet(
            f"color: {COLOR_TEXT_MUTED}; font-size: 10px;"
            f"font-family: 'Geist Mono','JetBrains Mono',monospace;"
            f"letter-spacing: 0.1px; background: transparent;"
        )
        meta.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        v.addWidget(meta)

    @staticmethod
    def _short_time(raw: str) -> str:
        """'2026-04-12 14:32' → '14:32 · 12 Apr'  (best-effort)."""
        try:
            from datetime import datetime
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
                        "%d.%m.%Y %H:%M", "%Y-%m-%dT%H:%M:%S"):
                try:
                    dt = datetime.strptime(raw, fmt)
                    return dt.strftime("%H:%M · %d %b")
                except ValueError:
                    continue
        except Exception:
            pass
        return raw


# ---------------------------------------------------------------------------
# Delegate — paints the left accent bar + selection tint BEHIND the row widget.
# The row widget itself stays simple; the delegate handles state-driven paint.
# ---------------------------------------------------------------------------
class _ReconDelegate(QStyledItemDelegate):
    ACCENT = QColor(COLOR_ACCENT)
    ACCENT_TINT = QColor(0, 255, 136, 18)
    HOVER_TINT = QColor(255, 255, 255, 8)
    HAIR = QColor(COLOR_HAIRLINE)

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index) -> None:
        painter.save()
        r = option.rect

        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)

        # Base wash
        if selected:
            painter.fillRect(r, self.ACCENT_TINT)
        elif hovered:
            painter.fillRect(r, self.HOVER_TINT)

        # Left accent bar (2 px)
        if selected or hovered:
            painter.fillRect(
                r.left() + 4, r.top() + 4,
                2, r.height() - 8,
                self.ACCENT if selected else QColor(COLOR_TEXT_MUTED),
            )

        # Bottom hairline separator (except for last item — handled by list)
        pen = QPen(self.HAIR)
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawLine(r.left() + 10, r.bottom(), r.right() - 6, r.bottom())
        painter.restore()


# ---------------------------------------------------------------------------
# The full section widget — used inside RightPanel.
# ---------------------------------------------------------------------------
class ReconstructionList(QWidget):
    item_activated = pyqtSignal(dict)
    load_failed = pyqtSignal(str)

    def __init__(self, record_source: RecordSource | None = None,
                 page_size: int = 20, parent=None):
        super().__init__(parent)
        self._record_source = record_source
        self._page_size = page_size
        self._cursor = 0
        self._exhausted = False

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        # Header: eyebrow + counter
        header = QHBoxLayout()
        header.setSpacing(8)
        eyebrow = QLabel("RECONSTRUCTION · 2D → 3D")
        eyebrow.setProperty("role", "eyebrow")
        self.counter = QLabel("0")
        self.counter.setProperty("role", "muted")
        self.counter.setStyleSheet(
            "font-family: 'Geist Mono','JetBrains Mono',monospace;"
            "font-size: 10px;"
        )
        self.counter.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        header.addWidget(eyebrow)
        header.addStretch(1)
        header.addWidget(self.counter)
        root.addLayout(header)

        # List
        self.list = QListWidget()
        self.list.setObjectName("ReconList")
        self.list.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection)
        self.list.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.list.setMouseTracking(True)
        self.list.setUniformItemSizes(True)
        self.list.setMinimumHeight(190)
        self.list.setSpacing(0)
        self.list.setFrameShape(self.list.frameShape().NoFrame)
        self.list.setItemDelegate(_ReconDelegate(self.list))
        # Tighter QSS: remove the default 8px padding on items.
        self.list.setStyleSheet("""
            QListWidget#ReconList { background: transparent; border: none; padding: 0; }
            QListWidget#ReconList::item { padding: 0; margin: 0; border-radius: 0; }
            QListWidget#ReconList::item:hover,
            QListWidget#ReconList::item:selected { background: transparent; }
        """)
        self.list.itemDoubleClicked.connect(self._on_double_click)
        self.list.itemActivated.connect(self._on_double_click)
        root.addWidget(self.list, 1)

        # Action row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        self.load_more_btn = QPushButton("LOAD MORE")
        self.load_more_btn.setProperty("variant", "primary")
        self.load_more_btn.setMinimumHeight(32)
        self.load_more_btn.clicked.connect(self.load_more)

        self.refresh_btn = QPushButton("↻")
        self.refresh_btn.setProperty("variant", "icon")
        self.refresh_btn.setToolTip("Reset and reload from server")
        self.refresh_btn.clicked.connect(self.reset_and_load)

        btn_row.addWidget(self.load_more_btn, 1)
        btn_row.addWidget(self.refresh_btn, 0)
        root.addLayout(btn_row)

        # Status line
        self.status = QLabel("")
        self.status.setProperty("role", "muted")
        self.status.setStyleSheet(
            "font-size: 10px; font-family: 'Geist Mono','JetBrains Mono',monospace;"
        )
        self.status.setWordWrap(True)
        root.addWidget(self.status)

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------
    def set_record_source(self, source: RecordSource) -> None:
        self._record_source = source

    def reset_and_load(self) -> None:
        self.list.clear()
        self._cursor = 0
        self._exhausted = False
        self.load_more_btn.setEnabled(True)
        self.load_more_btn.setText("LOAD MORE")
        self._update_counter()
        self.load_more()

    def load_more(self) -> None:
        if self._record_source is None:
            self.status.setText("no record source bound")
            return
        if self._exhausted:
            return

        self.load_more_btn.setEnabled(False)
        self.load_more_btn.setText("LOADING…")
        self.status.setText("fetching from server…")
        # Let the UI repaint the "LOADING…" state before we block on IO.
        from PyQt6.QtCore import QCoreApplication
        QCoreApplication.processEvents()

        try:
            batch = list(self._record_source(self._cursor, self._page_size))
        except Exception as e:
            self.status.setText(f"error: {e}")
            self.load_failed.emit(str(e))
            self.load_more_btn.setEnabled(True)
            self.load_more_btn.setText("LOAD MORE")
            return

        added = 0
        for entry in batch:
            self._append(entry)
            added += 1

        self._cursor += added
        self._update_counter()

        if added < self._page_size:
            self._exhausted = True
            self.load_more_btn.setText("NO MORE")
            self.load_more_btn.setEnabled(False)
            self.status.setText(f"all records loaded · total {self._cursor}")
        else:
            self.load_more_btn.setText("LOAD MORE")
            self.load_more_btn.setEnabled(True)
            self.status.setText(f"+{added} loaded · cursor {self._cursor}")

    # ----------------------------------------------------------------
    def _append(self, entry: dict) -> None:
        item = QListWidgetItem(self.list)
        item.setData(Qt.ItemDataRole.UserRole, entry)
        row = _ReconRow(entry)
        item.setSizeHint(QSize(0, 44))
        self.list.addItem(item)
        self.list.setItemWidget(item, row)

    def _on_double_click(self, item: QListWidgetItem) -> None:
        entry = item.data(Qt.ItemDataRole.UserRole)
        if isinstance(entry, dict):
            self.item_activated.emit(entry)

    def _update_counter(self) -> None:
        self.counter.setText(f"{self.list.count():>3}")
