# -*- coding: utf-8 -*-
"""
Диалог «Кузова к генерации».

Показывает модели, которые сервер уже снимает, но геометрии под них в
`models_geometry_config.json` нет (см. `src.registry.worklist` — там же
разобрано, почему такие снимки считаются по чужому кузову). Список
отсортирован по числу затронутых снимков: сверху то, что чаще всего портит
объём.

Что отсюда уходит наружу
------------------------
Диалог НЕ собирает кузов сам. Он качает облако выбранного пустого скана в
кэш и отдаёт `WorklistRequest` — набор полей для `BodyGenDialog`. Сборкой,
как и раньше, занимается `MainWindow`: у него поток генератора, сцена для
проверки и кнопка загрузки в реестр. Так цепочка «список → параметры →
сборка → проверка → реестр» собирается из уже существующих кусков, а не
дублируется здесь.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (QAbstractItemView, QCheckBox, QDialog,
                             QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                             QMessageBox, QProgressBar, QPushButton,
                             QSizePolicy, QSplitter, QTableWidget,
                             QTableWidgetItem, QVBoxLayout, QWidget)

from src.registry.worklist import (ModelNeed, Shot, Worklist,
                                   WorklistCancelled, WorklistScanner)
from src.ui.ui_theme import (COLOR_HAIRLINE, COLOR_SURFACE, COLOR_TEXT_DIM,
                             COLOR_TEXT_MUTED, COLOR_WARN, FONT_MONO,
                             apply_theme)

#: Колонки таблицы: заголовок, ширина, подсказка.
_COLUMNS = (
    ("Модель у клиента", 260, "Строка, которую оператор выбирает в клиенте"),
    ("Снимков", 78, "Сколько снимков обсчитано по чужой геометрии"),
    ("Пустых", 70, "Сканы пустого кузова — из них и собирается модель"),
    ("Обмер, м", 100, "Медиана по полю body_dimensions всех снимков"),
    ("Период", 168, "Первый и последний снимок с этим именем"),
    ("Ключ для реестра", 220, "Предложение; правится при загрузке"),
)


class _SortItem(QTableWidgetItem):
    """
    Ячейка, которая сортируется по числу, а не по тексту.

    QTableWidget сравнивает ячейки строками: «224» оказалось бы меньше «48», и
    главная колонка диалога — число затронутых снимков — сортировалась бы
    бессмысленно. Ключ кладётся в UserRole+1; где его нет, остаётся обычное
    строковое сравнение.
    """

    _KEY = Qt.ItemDataRole.UserRole + 1

    def __lt__(self, other: QTableWidgetItem) -> bool:    # noqa: D105 (Qt API)
        mine = self.data(self._KEY)
        theirs = other.data(self._KEY) if isinstance(other, QTableWidgetItem) \
            else None
        if mine is None or theirs is None:
            return super().__lt__(other)
        return mine < theirs


@dataclass
class WorklistRequest:
    """Что пользователь выбрал для сборки."""

    need: ModelNeed
    shot: Shot
    #: Локальный путь к скачанному облаку пустого кузова.
    ply_path: str
    #: Предлагаемое имя комплекта (оно же будущий ключ реестра).
    name: str
    #: Обмер кузова, м. Нули — обмера не нашлось ни в одной записи.
    length: float = 0.0
    width: float = 0.0


class _ScanThread(QThread):
    """Обход сервера. Отмена — флагом: рвать HTTP на полуслове незачем."""

    progressed = pyqtSignal(str, float)
    finishedWith = pyqtSignal(object)      # Worklist | Exception

    def __init__(self, host: str, port: int, parent=None):
        super().__init__(parent)
        self._scanner = WorklistScanner(host, port)
        self._stop = False

    def cancel(self) -> None:
        self._stop = True

    def run(self) -> None:                                # noqa: D102 (Qt API)
        try:
            result = self._scanner.scan(
                progress=lambda text, frac: self.progressed.emit(text, frac),
                cancelled=lambda: self._stop)
        except WorklistCancelled:
            return
        except Exception as exc:                          # noqa: BLE001
            result = exc
        self.finishedWith.emit(result)


class _FetchThread(QThread):
    """Скачивание одного файла записи (фото или облако)."""

    finishedWith = pyqtSignal(str, object)   # имя файла, путь | Exception

    def __init__(self, scanner: WorklistScanner, name: str, parent=None):
        super().__init__(parent)
        self._scanner = scanner
        self._name = name

    def run(self) -> None:                                # noqa: D102 (Qt API)
        try:
            path = self._scanner.fetch_asset(self._name)
        except Exception as exc:                          # noqa: BLE001
            path = exc
        self.finishedWith.emit(self._name, path)


class BodyWorklistDialog(QDialog):
    """Список кузовов, которых не хватает на сервере."""

    def __init__(self, parent=None, host: str = "", port: int = 9999):
        super().__init__(parent)
        apply_theme(self)
        self.setWindowTitle("Кузова к генерации")
        self.setModal(True)
        self.resize(1180, 680)

        self._host = host
        self._port = int(port)
        #: Отдельный сканер для докачки файлов: у него свой пул сессий, и он
        #: переживает поток обхода.
        self._scanner = WorklistScanner(host, port) if host else None
        self._worklist: Optional[Worklist] = None
        self._rows: List[ModelNeed] = []
        self._scan_thread: Optional[_ScanThread] = None
        self._fetch_thread: Optional[_FetchThread] = None
        self._photo_index = 0
        self._request: Optional[WorklistRequest] = None

        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(10)
        root.addLayout(self._build_head())

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self._build_table())
        split.addWidget(self._build_preview())
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        root.addWidget(split, 1)

        root.addWidget(self._build_footer())

        if not host:
            self._set_status("в config/tls_config.yaml нет активного сервера "
                             "— список брать неоткуда", warn=True)
        else:
            self.refresh()

    # ---- разметка ------------------------------------------------------

    def _build_head(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)

        self.ed_filter = QLineEdit()
        self.ed_filter.setPlaceholderText("фильтр по имени модели или госномеру")
        self.ed_filter.setClearButtonEnabled(True)
        self.ed_filter.textChanged.connect(self._refill)
        row.addWidget(self.ed_filter, 1)

        # Требование простое: показывать то, что можно собрать прямо сейчас.
        # Остальное прячется, но не исчезает — иначе непонятно, почему модель
        # из логов сервера в списке не значится.
        self.chk_incomplete = QCheckBox("показать неполные")
        self.chk_incomplete.setToolTip(
            "Модели без снимка пустого кузова или без topdown-фото. Собрать "
            "их нельзя: генератору нужно облако пустого кузова.")
        self.chk_incomplete.toggled.connect(self._refill)
        row.addWidget(self.chk_incomplete)

        # Переименования — отдельная галочка, а не «неполные»: там кузова нет
        # вовсе, а тут он есть и собирать его заново не надо.
        self.chk_renames = QCheckBox("показать переименования")
        self.chk_renames.setToolTip(
            "Имена, под которыми на сервере уже лежит та же модель: её просто "
            "переименовали, и старые снимки перестали находить свою "
            "геометрию. Чинится правкой имени в реестре, а не генерацией.")
        self.chk_renames.toggled.connect(self._refill)
        row.addWidget(self.chk_renames)

        self.btn_refresh = QPushButton("Обновить")
        self.btn_refresh.clicked.connect(self.refresh)
        row.addWidget(self.btn_refresh)
        return row

    def _build_table(self) -> QWidget:
        self.table = QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels([c[0] for c in _COLUMNS])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setSortingEnabled(True)
        header = self.table.horizontalHeader()
        for i, (_, width, tip) in enumerate(_COLUMNS):
            self.table.setColumnWidth(i, width)
            self.table.horizontalHeaderItem(i).setToolTip(tip)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.itemSelectionChanged.connect(self._on_row_changed)
        self.table.doubleClicked.connect(self._on_generate)
        return self.table

    def _build_preview(self) -> QWidget:
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(12, 0, 0, 0)
        lay.setSpacing(8)

        self.lbl_title = QLabel("—")
        self.lbl_title.setWordWrap(True)
        self.lbl_title.setStyleSheet("font-size: 15px; font-weight: 600;")
        lay.addWidget(self.lbl_title)

        self.lbl_photo = QLabel("выберите модель слева")
        self.lbl_photo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_photo.setMinimumHeight(240)
        self.lbl_photo.setSizePolicy(QSizePolicy.Policy.Expanding,
                                     QSizePolicy.Policy.Expanding)
        self.lbl_photo.setStyleSheet(
            f"background: {COLOR_SURFACE}; border: 1px solid {COLOR_HAIRLINE};"
            f" border-radius: 8px; color: {COLOR_TEXT_DIM};")
        lay.addWidget(self.lbl_photo, 1)

        nav = QHBoxLayout()
        nav.setSpacing(6)
        self.btn_prev = QPushButton("◀")
        self.btn_prev.setFixedWidth(34)
        self.btn_prev.clicked.connect(lambda: self._step_photo(-1))
        self.btn_next = QPushButton("▶")
        self.btn_next.setFixedWidth(34)
        self.btn_next.clicked.connect(lambda: self._step_photo(+1))
        self.lbl_shot = QLabel("—")
        self.lbl_shot.setStyleSheet(
            f"color: {COLOR_TEXT_MUTED}; font-size: 11px;"
            f" font-family: {FONT_MONO};")
        nav.addWidget(self.btn_prev)
        nav.addWidget(self.btn_next)
        nav.addWidget(self.lbl_shot, 1)
        lay.addLayout(nav)

        self.lbl_details = QLabel("")
        self.lbl_details.setWordWrap(True)
        self.lbl_details.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        self.lbl_details.setStyleSheet(
            f"color: {COLOR_TEXT_MUTED}; font-size: 11px;")
        lay.addWidget(self.lbl_details)
        return box

    def _build_footer(self) -> QWidget:
        box = QWidget()
        lay = QHBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet(
            f"color: {COLOR_TEXT_MUTED}; font-size: 11px;")
        lay.addWidget(self.lbl_status, 1)

        self.bar = QProgressBar()
        self.bar.setFixedWidth(180)
        self.bar.setTextVisible(False)
        self.bar.hide()
        lay.addWidget(self.bar)

        self.btn_generate = QPushButton("Собрать кузов…")
        self.btn_generate.setDefault(True)
        self.btn_generate.setEnabled(False)
        self.btn_generate.clicked.connect(self._on_generate)
        lay.addWidget(self.btn_generate)

        btn_close = QPushButton("Закрыть")
        btn_close.clicked.connect(self.reject)
        lay.addWidget(btn_close)
        return box

    # ---- обход ---------------------------------------------------------

    def refresh(self) -> None:
        """Перечитать список с сервера."""
        if self._scanner is None or self._scan_thread is not None:
            return
        self.btn_refresh.setEnabled(False)
        self.bar.setRange(0, 0)
        self.bar.show()
        self._set_status("опрос сервера…")

        self._scan_thread = _ScanThread(self._host, self._port, self)
        self._scan_thread.progressed.connect(self._on_scan_progress)
        self._scan_thread.finishedWith.connect(self._on_scan_done)
        self._scan_thread.finished.connect(self._on_scan_finished)
        self._scan_thread.start()

    def _on_scan_progress(self, text: str, fraction: float) -> None:
        if fraction < 0:
            self.bar.setRange(0, 0)
        else:
            self.bar.setRange(0, 1000)
            self.bar.setValue(int(fraction * 1000))
        self._set_status(text)

    def _on_scan_done(self, result) -> None:
        if isinstance(result, Exception):
            self._set_status(f"сервер не ответил: {result}", warn=True)
            return
        self._worklist = result
        self._refill()

        # Скрытое надо назвать вслух: иначе непонятно, куда делась модель,
        # которая в логах сервера есть, а в списке её нет.
        tail = []
        if result.renames:
            tail.append(f"{len(result.renames)} — переименования")
        if result.incomplete:
            tail.append(f"{len(result.incomplete)} без пустого скана или фото")
        # Нечитаемые записи — это почти всегда нулевые .json на сервере
        # (обрыв записи при заполнении диска). Молчать о них нельзя: вместе с
        # ними из подсчёта выпадают и снимки.
        if result.failed:
            tail.append(f"{result.failed} записей пустые или битые")
        note = ("; скрыто: " + ", ".join(tail)) if tail else ""
        self._set_status(
            f"на сервере {result.known_models} моделей, просмотрено "
            f"{result.records} снимков: сгенерировать надо "
            f"{len(result.ready)} кузовов{note}")

    def _on_scan_finished(self) -> None:
        self.bar.hide()
        self.btn_refresh.setEnabled(True)
        self._scan_thread = None

    # ---- таблица -------------------------------------------------------

    def _visible_needs(self) -> List[ModelNeed]:
        if self._worklist is None:
            return []
        needs = list(self._worklist.ready)
        if self.chk_renames.isChecked():
            needs += self._worklist.renames
        if self.chk_incomplete.isChecked():
            needs += self._worklist.incomplete
        text = self.ed_filter.text().strip().lower()
        if text:
            needs = [n for n in needs
                     if text in n.display_name.lower()
                     or text in n.suggested_key.lower()
                     or any(text in p.lower() for p in n.plates)]
        return needs

    def _refill(self) -> None:
        needs = self._visible_needs()
        self._rows = needs

        # Сортировка выключается на время заполнения: с включённой QTableWidget
        # переставляет строки после каждого setItem, и индексы в _rows разъезжаются
        # с тем, что видит пользователь.
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(needs))
        for row, need in enumerate(needs):
            length, width = need.measurement()
            dims = (f"{length:.2f} × {width:.2f}"
                    if length > 0 and width > 0 else "—")
            period = (f"{_day(need.first_day)} – {_day(need.last_day)}"
                      if need.first_day else "—")
            title = need.display_name
            if need.rename is not None:
                title += "  · переименование"
            cells = (
                (title, None),
                (str(need.shots), need.shots),
                (str(len(need.empty_shots)), len(need.empty_shots)),
                (dims, length),
                (period, need.last_day),
                (need.suggested_key, None),
            )
            for col, (text, sort_key) in enumerate(cells):
                item = _SortItem(text)
                if sort_key is not None:
                    item.setData(_SortItem._KEY, sort_key)
                if col in (1, 2, 3):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight
                                          | Qt.AlignmentFlag.AlignVCenter)
                if need.rename is not None:
                    item.setForeground(_dim_brush())
                    item.setToolTip("Генерировать не надо — "
                                    + need.rename.explain())
                elif not need.ready:
                    item.setForeground(_dim_brush())
                    item.setToolTip(f"Собрать нельзя: {need.missing}")
                if col == 0:
                    item.setData(Qt.ItemDataRole.UserRole, row)
                self.table.setItem(row, col, item)
        self.table.setSortingEnabled(True)
        self.table.sortItems(1, Qt.SortOrder.DescendingOrder)
        if needs:
            self.table.selectRow(0)
        else:
            self._show_need(None)

    def _current_need(self) -> Optional[ModelNeed]:
        row = self.table.currentRow()
        if row < 0:
            return None
        item = self.table.item(row, 0)
        if item is None:
            return None
        index = item.data(Qt.ItemDataRole.UserRole)
        if index is None or not (0 <= int(index) < len(self._rows)):
            return None
        return self._rows[int(index)]

    def _on_row_changed(self) -> None:
        self._photo_index = 0
        self._show_need(self._current_need())

    # ---- карточка ------------------------------------------------------

    def _show_need(self, need: Optional[ModelNeed]) -> None:
        # Переименование собрать технически можно, но это заведомо лишняя
        # работа: кузов уже есть. Кнопку не гасим совсем — бывает, что
        # «та же» модель на деле другой кузов, — но решение принимает человек,
        # прочитав пояснение в карточке.
        self.btn_generate.setEnabled(bool(need and need.ready))
        self.btn_generate.setText(
            "Собрать всё равно…" if (need and need.rename is not None)
            else "Собрать кузов…")
        if need is None:
            self.lbl_title.setText("—")
            self.lbl_details.setText("")
            self.lbl_shot.setText("—")
            self._clear_photo("выберите модель слева")
            self.btn_prev.setEnabled(False)
            self.btn_next.setEnabled(False)
            return

        self.lbl_title.setText(need.display_name)
        length, width = need.measurement()
        plates = ", ".join(need.plates[:6])
        if len(need.plates) > 6:
            plates += f" и ещё {len(need.plates) - 6}"
        # У переименования свой ключ предлагать нечего: модель на сервере уже
        # лежит под старым ключом, и показывать рядом «новый» значит толкать
        # к созданию дубля.
        key_line = (f"Ключ на сервере: {need.rename.known.key}"
                    if need.rename is not None
                    else f"Ключ: {need.suggested_key}")
        bits = [key_line,
                f"Снимков по чужой геометрии: {need.shots}",
                f"Сканов пустого кузова: {len(need.empty_shots)}"]
        if length > 0:
            bits.append(f"Обмер кузова: {length:.2f} × {width:.2f} м")
        if plates:
            bits.append(f"Госномера: {plates}")
        if need.rename is not None:
            bits.append(
                "⟲ Генерировать не надо: " + need.rename.explain()
                + ". Кузов уже смоделирован — достаточно поправить имя "
                  "модели в реестре, и снимки пересчитаются по нему.")
        elif not need.ready:
            bits.append(f"Собрать нельзя: {need.missing}")
        self.lbl_details.setText("\n".join(bits))

        has = len(need.with_photo)
        self.btn_prev.setEnabled(has > 1)
        self.btn_next.setEnabled(has > 1)
        if not has:
            self.lbl_shot.setText("фотографий нет")
            self._clear_photo("нет topdown-фото")
            return
        self._load_photo(need)

    def _step_photo(self, delta: int) -> None:
        need = self._current_need()
        if need is None or not need.with_photo:
            return
        self._photo_index = ((self._photo_index + delta)
                             % len(need.with_photo))
        self._load_photo(need)

    def _load_photo(self, need: ModelNeed) -> None:
        shot = need.with_photo[self._photo_index]
        self.lbl_shot.setText(
            f"{self._photo_index + 1}/{len(need.with_photo)}  "
            f"{_day(shot.day)}  {shot.car_number or '—'}  "
            f"{shot.direction or '—'}")
        if self._scanner is None:
            return
        cached = os.path.join(self._scanner.assets_dir, shot.photo_file)
        if os.path.exists(cached) and os.path.getsize(cached) > 0:
            self._set_photo(cached)
            return
        self._clear_photo("загрузка фото…")
        self._start_fetch(shot.photo_file, self._on_photo_fetched)

    def _on_photo_fetched(self, name: str, path) -> None:
        if isinstance(path, Exception):
            self._clear_photo("фото не скачалось")
            print(f"[worklist] {name}: {path}")
            return
        # Пока файл ехал, пользователь мог уйти на другую строку: показываем
        # только то, что сейчас выбрано.
        need = self._current_need()
        if (need is None or not need.with_photo
                or need.with_photo[self._photo_index].photo_file != name):
            return
        self._set_photo(path)

    def _set_photo(self, path: str) -> None:
        pix = QPixmap(path)
        if pix.isNull():
            self._clear_photo("фото не читается")
            return
        self._photo_pixmap = pix
        self.lbl_photo.setPixmap(pix.scaled(
            self.lbl_photo.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

    def _clear_photo(self, text: str) -> None:
        self._photo_pixmap = None
        self.lbl_photo.setPixmap(QPixmap())
        self.lbl_photo.setText(text)

    def resizeEvent(self, event):                         # noqa: D102 (Qt API)
        super().resizeEvent(event)
        pix = getattr(self, "_photo_pixmap", None)
        if pix is not None and not pix.isNull():
            self.lbl_photo.setPixmap(pix.scaled(
                self.lbl_photo.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))

    # ---- сборка --------------------------------------------------------

    def _start_fetch(self, name: str, slot) -> None:
        """
        Поставить одну закачку. Предыдущая не ждётся: она либо уже закончилась,
        либо её результат всё равно не нужен (пользователь ушёл со строки).
        """
        thread = _FetchThread(self._scanner, name, self)
        thread.finishedWith.connect(slot)
        thread.finished.connect(thread.deleteLater)
        self._fetch_thread = thread
        thread.start()

    def _on_generate(self) -> None:
        need = self._current_need()
        if need is None or not need.ready:
            return
        shot = (need.with_photo[self._photo_index]
                if need.with_photo else None)
        if shot is None or not shot.ply_file:
            QMessageBox.warning(self, "Кузова к генерации",
                                "У этого скана нет облака .ply.")
            return

        self.btn_generate.setEnabled(False)
        self.bar.setRange(0, 0)
        self.bar.show()
        self._set_status(f"скачиваю облако {shot.ply_file}…")
        self._pending = (need, shot)
        self._start_fetch(shot.ply_file, self._on_ply_fetched)

    def _on_ply_fetched(self, name: str, path) -> None:
        self.bar.hide()
        self.btn_generate.setEnabled(True)
        need, shot = getattr(self, "_pending", (None, None))
        if need is None:
            return
        if isinstance(path, Exception):
            self._set_status(f"облако не скачалось: {path}", warn=True)
            return
        length, width = need.measurement()
        self._request = WorklistRequest(
            need=need, shot=shot, ply_path=str(path),
            name=need.suggested_key, length=length, width=width)
        self.accept()

    def request(self) -> Optional[WorklistRequest]:
        """Что выбрано для сборки. None — диалог закрыли без выбора."""
        return self._request

    # ---- мелочи --------------------------------------------------------

    def _set_status(self, text: str, warn: bool = False) -> None:
        color = COLOR_WARN if warn else COLOR_TEXT_MUTED
        self.lbl_status.setStyleSheet(f"color: {color}; font-size: 11px;")
        self.lbl_status.setText(text)

    def done(self, result: int) -> None:                  # noqa: D102 (Qt API)
        # Поток обхода держит сокеты и пул: закрывать диалог, не сказав ему
        # «стоп», значит оставить его качать записи в пустоту.
        if self._scan_thread is not None:
            self._scan_thread.cancel()
            self._scan_thread.wait(3000)
        super().done(result)


def _day(value: str) -> str:
    """`20261006` → `06.10.2026`. Непонятное — как есть."""
    if len(value) == 8 and value.isdigit():
        return f"{value[6:]}.{value[4:6]}.{value[:4]}"
    return value or "—"


_DIM_BRUSH = None


def _dim_brush():
    """Кисть для строк, которые генерировать не надо. Создаётся один раз."""
    global _DIM_BRUSH
    if _DIM_BRUSH is None:
        from PyQt6.QtGui import QBrush, QColor
        _DIM_BRUSH = QBrush(QColor(COLOR_TEXT_DIM))
    return _DIM_BRUSH
