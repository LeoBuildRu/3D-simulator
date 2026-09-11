# -*- coding: utf-8 -*-
"""
Диалог генератора кузовов.

Только сбор параметров: считает `src.bodygen.service`, запускает поток
`MainWindow`. Здесь нет ни Panda3D, ни обращений к сцене — так модуль остаётся
отключаемым, а расчётная часть пригодной для сервера.
"""

from __future__ import annotations

import json
import os

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                             QDoubleSpinBox, QFileDialog, QFormLayout, QFrame,
                             QHBoxLayout, QLabel, QLineEdit, QMessageBox,
                             QPushButton, QSpinBox, QStackedWidget,
                             QVBoxLayout, QWidget)

from src.bodygen import BodyGenParams, list_chassis, list_models, probe
from src.bodygen.service import DEFAULT_OUT_DIR
from src.ui.ui_theme import (COLOR_HAIRLINE, COLOR_TEXT_MUTED, COLOR_WARN,
                             FONT_MONO, apply_theme)

#: Пресеты качества: имя -> (плотность текселя px/m, размер атласа, AO).
#: «Черновик» существует не для экономии диска, а для итераций: полный прогон
#: на 4K занимает минуты, и подбирать по нему цвет краски невозможно.
QUALITY_PRESETS = {
    "Черновик (быстро, ~25 с)": (250.0, 2048, False),
    "Рабочее (~1.5 мин)": (320.0, 4096, True),
    "Полное (4K, ~2.5 мин)": (400.0, 4096, True),
}

# src/ui/bodygen_dialog.py -> src/ui -> src -> <корень проекта>
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: Последние параметры сборки. Подбор кузова — это несколько прогонов подряд с
#: правкой одного-двух полей (цвет, износ, рельеф борта), и заполнять форму
#: заново каждый раз бессмысленно. Рядом с config/dataset.json, тем же
#: способом: у проекта уже есть эта конвенция.
SETTINGS_PATH = os.path.join(PROJECT_ROOT, "config", "bodygen.json")

#: Как генератор называет файлы комплекта `<name>` (`pipeline._write_files`).
#: Список точный, а не «всё, что начинается с имени»: иначе комплект «KAMAZ»
#: утащил бы за собой файлы «KAMAZ-6520».
_KIT_SUFFIXES = ("", "-Cuzov.bam", "-Napolnitel.bam", "-Napolnitel.obj",
                 "-Other.bam", ".bam", ".glb", ".obj", ".gltf", ".bin",
                 ".set.json", ".spec.json")


def existing_kit_paths(out_dir: str, name: str) -> list:
    """
    Файлы уже собранного комплекта `name` в папке `out_dir`.

    Пустой список — комплекта нет. Проверяется по именам, которые пишет сам
    генератор, поэтому работает в любой папке, в том числе выбранной вручную,
    и не зависит от того, попал ли комплект в список моделей.
    """
    if not out_dir or not name:
        return []
    found = []
    for suffix in _KIT_SUFFIXES:
        path = os.path.join(out_dir, f"{name}{suffix}")
        if os.path.exists(path):
            found.append(path)
    return found


class BodyGenDialog(QDialog):
    """Модальный диалог параметров сборки кузова."""

    def __init__(self, parent=None, default_out: str = DEFAULT_OUT_DIR):
        super().__init__(parent)
        apply_theme(self)
        self.setWindowTitle("Генератор кузова")
        self.setModal(True)
        self.setMinimumWidth(560)

        self._probe = probe()
        #: Файлы комплекта, который пользователь согласился перезаписать.
        #: Читает `MainWindow` и удаляет их перед сборкой.
        self._overwrite_paths: list = []
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(12)

        root.addWidget(self._build_status())
        root.addWidget(self._build_source())
        root.addWidget(self._build_assembly(default_out))
        root.addWidget(self._build_look())

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Собрать")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Отмена")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self._ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self._ok_button.setEnabled(bool(self._probe.get("available")))

        self._restore_state()

    # ---- секции ------------------------------------------------------- #

    def _build_status(self) -> QWidget:
        box = QFrame()
        box.setStyleSheet(f"QFrame {{ border: 1px solid {COLOR_HAIRLINE};"
                          f" border-radius: 8px; }}")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(12, 10, 12, 10)

        if self._probe.get("available"):
            bits = [f"шасси: {', '.join(self._probe['chassis']) or '—'}"]
            if not self._probe.get("draco"):
                bits.append("без Draco (.gltf будет крупным)")
            if not self._probe.get("volume_calculator"):
                bits.append("без расчёта объёма — сборка из облака недоступна")
            text = " · ".join(bits)
            color = COLOR_TEXT_MUTED
        else:
            text = self._probe.get("reason") or "генератор недоступен"
            color = COLOR_WARN

        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setStyleSheet(f"color: {color}; font-size: 11px;"
                          f" font-family: {FONT_MONO}; border: none;")
        lay.addWidget(lbl)
        return box

    def _build_source(self) -> QWidget:
        box = QFrame()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        self.cmb_source = QComboBox()
        self.cmb_source.addItem("Справочник truck_models", "catalog")
        self.cmb_source.addItem("Съёмка пустого кузова (.ply)", "ply")
        self.cmb_source.addItem("Готовое описание (.spec.json)", "spec")
        self.cmb_source.currentIndexChanged.connect(
            lambda i: self.stack.setCurrentIndex(i))

        head = QFormLayout()
        head.addRow("Источник", self.cmb_source)
        lay.addLayout(head)

        self.stack = QStackedWidget()

        # -- справочник
        page_cat = QWidget()
        f = QFormLayout(page_cat)
        f.setContentsMargins(0, 0, 0, 0)
        self.cmb_model = QComboBox()
        for key in list_models():
            self.cmb_model.addItem(key, key)
        if self.cmb_model.count() == 0:
            self.cmb_model.addItem("— справочник недоступен —", "")
        f.addRow("Модель", self.cmb_model)
        self.stack.addWidget(page_cat)

        # -- облако
        page_ply = QWidget()
        f = QFormLayout(page_ply)
        f.setContentsMargins(0, 0, 0, 0)
        self.ed_ply, row = self._file_row("Облако .ply", "PLY (*.ply)")
        f.addRow("Файл", row)
        self.cmb_cloud_model = QComboBox()
        self.cmb_cloud_model.addItem("автоподбор", "")
        for key in list_models():
            self.cmb_cloud_model.addItem(key, key)
        f.addRow("Модель скана", self.cmb_cloud_model)

        rect_row = QHBoxLayout()
        self.spn_rw = QDoubleSpinBox()
        self.spn_rw.setRange(0.0, 4.0)
        self.spn_rw.setSingleStep(0.05)
        self.spn_rw.setSpecialValueText("авто")
        self.spn_rw.setSuffix(" м")
        self.spn_rl = QDoubleSpinBox()
        self.spn_rl.setRange(0.0, 14.0)
        self.spn_rl.setSingleStep(0.1)
        self.spn_rl.setSpecialValueText("авто")
        self.spn_rl.setSuffix(" м")
        rect_row.addWidget(self.spn_rw)
        rect_row.addWidget(self.spn_rl)
        holder = QWidget()
        holder.setLayout(rect_row)
        f.addRow("Прямоугольник", holder)
        hint = QLabel("Задайте, если кузова нет в справочнике: иначе скан "
                      "может зацепиться за ложный прямоугольник.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 10px;")
        f.addRow("", hint)
        self.stack.addWidget(page_ply)

        # -- спек
        page_spec = QWidget()
        f = QFormLayout(page_spec)
        f.setContentsMargins(0, 0, 0, 0)
        self.ed_spec, row = self._file_row("Описание", "JSON (*.json)")
        f.addRow("Файл", row)
        self.stack.addWidget(page_spec)

        lay.addWidget(self.stack)
        return box

    def _build_assembly(self, default_out: str) -> QWidget:
        box = QFrame()
        f = QFormLayout(box)
        f.setContentsMargins(0, 0, 0, 0)

        self.ed_name = QLineEdit()
        self.ed_name.setPlaceholderText("имя комплекта (по умолчанию — из источника)")
        f.addRow("Имя", self.ed_name)

        self.cmb_chassis = QComboBox()
        self.cmb_chassis.addItem("подобрать по длине", "auto")
        for key in list_chassis():
            self.cmb_chassis.addItem(key, key)
        self.cmb_chassis.addItem("без шасси", "none")
        f.addRow("Шасси", self.cmb_chassis)

        self.spn_heap = QDoubleSpinBox()
        self.spn_heap.setRange(0.2, 3.0)
        self.spn_heap.setSingleStep(0.1)
        self.spn_heap.setValue(1.0)
        self.spn_heap.setSuffix(" м")
        f.addRow("Запас на горку", self.spn_heap)

        self.cmb_quality = QComboBox()
        for label in QUALITY_PRESETS:
            self.cmb_quality.addItem(label)
        self.cmb_quality.setCurrentIndex(1)
        f.addRow("Качество", self.cmb_quality)

        self.spn_decimate = QDoubleSpinBox()
        self.spn_decimate.setRange(0.0, 0.95)
        self.spn_decimate.setSingleStep(0.05)
        self.spn_decimate.setValue(0.0)
        self.spn_decimate.setSpecialValueText("без прореживания")
        f.addRow("Прореживание .gltf", self.spn_decimate)

        self.cmb_texmax = QComboBox()
        self.cmb_texmax.addItem("как есть", 0)
        self.cmb_texmax.addItem("до 2048", 2048)
        self.cmb_texmax.addItem("до 1024", 1024)
        f.addRow("Карты в .gltf", self.cmb_texmax)

        self.ed_out, row = self._dir_row(default_out)
        f.addRow("Каталог", row)
        return box

    def _build_look(self) -> QWidget:
        box = QFrame()
        f = QFormLayout(box)
        f.setContentsMargins(0, 0, 0, 0)

        # Рельеф борта и обвязка — ВЫБОР, а не замер.
        #
        # Наружную грань борта лидар с мачты не видит вовсе: она в тени самого
        # кузова. Поэтому при сборке из облака генератор её не выдумывает, и
        # без этих полей кузов выходит гладким корытом. Здесь выбирается, чем
        # его закрыть, — а замеренные размеры остаются как есть.
        self.cmb_side = QComboBox()
        for label, key in (("как в описании", ""),
                           ("решётка квадратных ячеек", "panels"),
                           ("наклонные стойки (как передний борт)", "slanted"),
                           ("один длинный вырез с брусьями", "belt"),
                           ("гладкий борт", "none")):
            self.cmb_side.addItem(label, key)
        self.cmb_side.setCurrentIndex(1)
        self.cmb_side.setToolTip(
            "Чем закрыта наружная грань борта. В облаке этой формы нет — борт "
            "снаружи в тени кузова, — поэтому она выбирается, а не меряется. "
            "Наклонные стойки идут под тем же углом, что передний борт.")
        f.addRow("Рельеф борта", self.cmb_side)

        self.chk_decor = QCheckBox("обвязка, задняя дверь и стойки")
        self.chk_decor.setChecked(True)
        self.chk_decor.setToolTip(
            "Достроить нижнюю обвязку, петли задней створки и задние угловые "
            "стойки. Лидар их не показывает, но на настоящем кузове они есть.")
        f.addRow("Достроить", self.chk_decor)

        self.spn_visor = QDoubleSpinBox()
        self.spn_visor.setRange(-0.01, 1.20)
        self.spn_visor.setSingleStep(0.05)
        self.spn_visor.setDecimals(2)
        self.spn_visor.setSpecialValueText("как измерено")
        self.spn_visor.setValue(-0.01)
        self.spn_visor.setSuffix(" м")
        self.spn_visor.setToolTip(
            "Вылет полки козырька ВПЕРЁД, над кабиной. «Как измерено» — взять "
            "из облака: полка видна как сплошная плита во всю ширину за "
            "передним бортом. Если сверху не полка, а скат (дефлектор кабины), "
            "замер её не ставит — тогда вылет можно задать здесь вручную.")
        f.addRow("Вылет козырька", self.spn_visor)

        # Поле намеренно ПУСТОЕ: умолчание живёт в спеке (тёмный нейтральный,
        # как у ручных моделей проекта), и дублировать его здесь значит завести
        # второй источник правды, который рано или поздно разойдётся с первым.
        self.ed_paint = QLineEdit("")
        self.ed_paint.setPlaceholderText("#RRGGBB — пусто: как у ручных моделей")
        self.ed_paint.setToolTip(
            "Цвет краски под слоем грязи. Пустое поле — тёмный нейтральный, "
            "в котором сделаны готовые модели проекта.")
        f.addRow("Цвет краски", self.ed_paint)

        self.spn_wear = QDoubleSpinBox()
        self.spn_wear.setRange(0.0, 1.0)
        self.spn_wear.setSingleStep(0.05)
        self.spn_wear.setValue(0.45)
        f.addRow("Износ", self.spn_wear)

        self.spn_dirt = QDoubleSpinBox()
        self.spn_dirt.setRange(0.0, 1.0)
        self.spn_dirt.setSingleStep(0.05)
        self.spn_dirt.setValue(0.62)
        f.addRow("Грязь", self.spn_dirt)

        self.spn_seed = QSpinBox()
        self.spn_seed.setRange(0, 9999)
        f.addRow("Зерно шума", self.spn_seed)

        self.chk_install = QCheckBox(
            "Показать в списке моделей после сборки")
        self.chk_install.setChecked(True)
        f.addRow("", self.chk_install)
        return box

    # ---- мелочи ------------------------------------------------------- #

    def _file_row(self, title: str, mask: str):
        edit = QLineEdit()
        edit.setPlaceholderText("путь к файлу")
        btn = QPushButton("Обзор…")
        btn.clicked.connect(
            lambda: self._pick_file(edit, title, mask))
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(edit, 1)
        h.addWidget(btn, 0)
        return edit, row

    def _dir_row(self, default: str):
        edit = QLineEdit(default)
        btn = QPushButton("Обзор…")

        def pick():
            path = QFileDialog.getExistingDirectory(
                self, "Куда сложить комплект", edit.text() or default)
            if path:
                edit.setText(path)

        btn.clicked.connect(pick)
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(edit, 1)
        h.addWidget(btn, 0)
        return edit, row

    def _pick_file(self, edit: QLineEdit, title: str, mask: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, title, edit.text(), mask)
        if path:
            edit.setText(path)

    # ---- запоминание полей -------------------------------------------- #

    def _state(self) -> dict:
        """Снимок всех полей формы — ровно то, что уходит в файл настроек."""
        return {
            "source": self.cmb_source.currentData(),
            "model_key": self.cmb_model.currentData() or "",
            "ply_path": self.ed_ply.text(),
            "spec_path": self.ed_spec.text(),
            "cloud_model": self.cmb_cloud_model.currentData() or "",
            "rect_width": float(self.spn_rw.value()),
            "rect_length": float(self.spn_rl.value()),
            "name": self.ed_name.text(),
            "out_dir": self.ed_out.text(),
            "chassis": self.cmb_chassis.currentData() or "auto",
            "heap": float(self.spn_heap.value()),
            "quality": self.cmb_quality.currentText(),
            "decimate": float(self.spn_decimate.value()),
            "texmax": int(self.cmb_texmax.currentData() or 0),
            "side_style": self.cmb_side.currentData() or "",
            "decor": bool(self.chk_decor.isChecked()),
            "visor": float(self.spn_visor.value()),
            "paint": self.ed_paint.text(),
            "wear": float(self.spn_wear.value()),
            "dirt": float(self.spn_dirt.value()),
            "seed": int(self.spn_seed.value()),
            "install": bool(self.chk_install.isChecked()),
        }

    def _restore_state(self) -> None:
        """
        Вернуть поля к прошлой сборке.

        Каждое поле восстанавливается ОТДЕЛЬНО и молча пропускается, если
        значения больше нет: справочник моделей и каталог шасси приходят
        снаружи и между запусками меняются, а из-за одной пропавшей записи
        форма не должна терять остальные.
        """
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except FileNotFoundError:
            return
        except Exception as exc:
            print(f"[BodyGen] {SETTINGS_PATH} не прочитан ({exc}); "
                  f"беру умолчания.")
            return
        if not isinstance(data, dict):
            return

        def pick(combo: QComboBox, key: str) -> None:
            idx = combo.findData(data.get(key))
            if idx >= 0:
                combo.setCurrentIndex(idx)

        def text(edit: QLineEdit, key: str) -> None:
            if isinstance(data.get(key), str):
                edit.setText(data[key])

        def number(spin, key: str) -> None:
            value = data.get(key)
            if isinstance(value, (int, float)):
                spin.setValue(type(spin.value())(value))

        def flag(check: QCheckBox, key: str) -> None:
            if isinstance(data.get(key), bool):
                check.setChecked(data[key])

        pick(self.cmb_source, "source")
        self.stack.setCurrentIndex(self.cmb_source.currentIndex())
        pick(self.cmb_model, "model_key")
        pick(self.cmb_cloud_model, "cloud_model")
        pick(self.cmb_chassis, "chassis")
        pick(self.cmb_texmax, "texmax")
        pick(self.cmb_side, "side_style")

        text(self.ed_ply, "ply_path")
        text(self.ed_spec, "spec_path")
        text(self.ed_name, "name")
        text(self.ed_paint, "paint")
        if data.get("out_dir"):
            self.ed_out.setText(str(data["out_dir"]))

        number(self.spn_rw, "rect_width")
        number(self.spn_rl, "rect_length")
        number(self.spn_heap, "heap")
        number(self.spn_decimate, "decimate")
        number(self.spn_visor, "visor")
        number(self.spn_wear, "wear")
        number(self.spn_dirt, "dirt")
        number(self.spn_seed, "seed")

        flag(self.chk_decor, "decor")
        flag(self.chk_install, "install")

        quality = data.get("quality")
        if quality in QUALITY_PRESETS:
            self.cmb_quality.setCurrentText(quality)

    def _save_state(self) -> None:
        try:
            os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
            with open(SETTINGS_PATH, "w", encoding="utf-8") as fh:
                json.dump(self._state(), fh, indent=2, ensure_ascii=False)
        except Exception as exc:
            print(f"[BodyGen] настройки не сохранены: {exc}")

    # ---- подтверждение ------------------------------------------------ #

    def accept(self) -> None:
        """
        Перед закрытием — спросить про перезапись, если имя уже занято.

        Проверка возможна только при ЯВНО заданном имени: пустое поле означает
        «взять имя из источника», а оно известно лишь после разбора облака —
        то есть после самой долгой части сборки.

        Старый комплект именно удаляется, а не перезаписывается поверх: набор
        файлов зависит от параметров (без шасси не будет `-Other.bam`, без
        частей — .gltf), и остатки прошлой сборки собрались бы в комплект-химеру.
        """
        name = self.ed_name.text().strip()
        out_dir = self.ed_out.text().strip() or DEFAULT_OUT_DIR
        existing = existing_kit_paths(out_dir, name)
        if existing:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Warning)
            box.setWindowTitle("Комплект уже существует")
            box.setText(
                f"«{name}» уже собран в этой папке.\n\n"
                f"Перезаписать? Старые файлы ({len(existing)} шт.) будут "
                f"удалены перед сборкой, отменить это нельзя.")
            box.setDetailedText("\n".join(existing))
            yes = box.addButton("Перезаписать",
                                QMessageBox.ButtonRole.DestructiveRole)
            back = box.addButton("Изменить имя",
                                 QMessageBox.ButtonRole.RejectRole)
            box.setDefaultButton(back)
            box.exec()
            if box.clickedButton() is not yes:
                self.ed_name.setFocus()
                self.ed_name.selectAll()
                return
            self._overwrite_paths = existing
        else:
            self._overwrite_paths = []

        self._save_state()
        super().accept()

    def overwrite_paths(self) -> list:
        """Файлы прошлого комплекта, которые пользователь согласился стереть."""
        return list(self._overwrite_paths)

    # ---- результат ---------------------------------------------------- #

    def params(self) -> BodyGenParams:
        """Собрать параметры из полей диалога."""
        density, atlas, ao = QUALITY_PRESETS[self.cmb_quality.currentText()]
        source = self.cmb_source.currentData()
        return BodyGenParams(
            source=source,
            model_key=(self.cmb_model.currentData() or ""),
            ply_path=self.ed_ply.text().strip(),
            spec_path=self.ed_spec.text().strip(),
            cloud_model=(self.cmb_cloud_model.currentData() or ""),
            rect_width=float(self.spn_rw.value()),
            rect_length=float(self.spn_rl.value()),
            name=self.ed_name.text().strip(),
            out_dir=self.ed_out.text().strip() or DEFAULT_OUT_DIR,
            chassis=(self.cmb_chassis.currentData() or "auto"),
            heap=float(self.spn_heap.value()),
            density=density, atlas=atlas, with_ao=ao,
            gltf_decimate=float(self.spn_decimate.value()),
            gltf_texture_max=int(self.cmb_texmax.currentData() or 0),
            side_style=(self.cmb_side.currentData() or ""),
            decor=bool(self.chk_decor.isChecked()),
            visor_overhang=float(self.spn_visor.value()),
            paint=self.ed_paint.text().strip(),
            wear=float(self.spn_wear.value()),
            dirt=float(self.spn_dirt.value()),
            seed=int(self.spn_seed.value()),
        )

    def show_in_list(self) -> bool:
        return bool(self.chk_install.isChecked())
