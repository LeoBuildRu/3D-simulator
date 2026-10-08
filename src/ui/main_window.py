# main_window.py
# ---------------------------------------------------------------------------
# Minimal Qt main window for the Toner simulator.
# ---------------------------------------------------------------------------

from __future__ import annotations

import win32gui
import win32con

from panda3d.core import WindowProperties

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QFrame,
)

from src.ui.ui_theme import apply_theme
from src.ui.overlay_widgets import (
    TelemetryHUD, DepthMapOverlay, CameraReferenceOverlay,
)
from src.ui.right_panel import RightPanel
from src.ui.toolbar import ViewToolbar
from src.ui import dataset_config
from src.ui.depth_preview import depth_to_qimage
import os
import json
import math
import random
import time
import shutil
import tempfile
import traceback
from typing import Any

from src.ui.panel_data import (
    get_model_set_config, get_texture_set_config,
    load_texture_sets, Reconstruction, download_server_image,
    ensure_texture_cached, TEXTURE_PATH_KEYS, get_default_texture_set_key,
    resolve_depth_record_files,
    PROJECT_ROOT,
)

# Where the 3 user camera presets (position + FOV) are persisted.
CAMERA_PRESETS_PATH = os.path.join(PROJECT_ROOT, "presets", "camera_presets.json")
# Опорные точки (world 3D) для авто-реконструкции по depth-записям.
DEPTH_ANCHORS_PATH  = os.path.join(PROJECT_ROOT, "presets", "depth_anchors_world.json")


def _is_child_of(hwnd: int, parent_hwnd: int) -> bool:
    """True iff `hwnd` is (transitively) a child of `parent_hwnd`."""
    try:
        cur = win32gui.GetParent(hwnd)
        while cur:
            if int(cur) == int(parent_hwnd):
                return True
            cur = win32gui.GetParent(cur)
    except Exception:
        return False
    return False


class MainWindow(QMainWindow):
    """Qt main window shell. Panda3D ShowBase attaches AFTER show()."""

    #: Ход реконструкции по проезду: (стадия, данные). Точка подключения для
    #: визуального сопровождения. Стадии по порядку:
    #:   "started"  {"rec"}
    #:   "fetched"  {"rec", "json", "ply_path", "model_key", "texture_key",
    #:               "vertices", "faces", "volume"} — всё скачано и разобрано,
    #:               сцена ещё не тронута
    #:   "applied"  {"rec", "node", "volume"} — меш стоит в сцене
    #:   "failed"   {"rec", "error"}
    reconstructionStage = pyqtSignal(str, object)

    def __init__(self):
        super().__init__()

        self.panda_app = None
        self._panda_hwnd: int | None = None

        # Настройки съёмки датасета живут в config/dataset.json и правятся
        # в отдельном диалоге (вкладка «Датасет» инспектора → «Настроить»).
        self._dataset_cfg = dataset_config.load()

        self.setWindowTitle("IQoko · 3D Симулятор")
        self.resize(1920, 1080)
        self.setMinimumSize(1280, 720)
        apply_theme(self)

        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.panda_container = QFrame()
        self.panda_container.setStyleSheet("background-color: #000000;")
        self.panda_container.setMinimumSize(800, 600)
        self.panda_container.setAttribute(
            Qt.WidgetAttribute.WA_NativeWindow, True
        )
        self.panda_container.setAttribute(
            Qt.WidgetAttribute.WA_DontCreateNativeAncestors, True
        )
        self.panda_container.setAttribute(
            Qt.WidgetAttribute.WA_NoSystemBackground, True
        )

        root.addWidget(self.panda_container, 1)
        self.setCentralWidget(central)

    def panda_container_hwnd(self) -> int:
        return int(self.panda_container.winId())

    def attach_panda(self, panda_app) -> None:
        if self.panda_app is not None:
            raise RuntimeError("attach_panda() called twice")
        self.panda_app = panda_app

        parent_hwnd = self.panda_container_hwnd()
        self._panda_hwnd = self._resolve_panda_hwnd(panda_app, parent_hwnd)
        print(
            f"[MainWindow] panda_container_hwnd = {parent_hwnd:#x}, "
            f"panda_hwnd = "
            f"{(f'{self._panda_hwnd:#x}' if self._panda_hwnd else 'None')}"
        )

        self._reposition_panda()

        # Кадры гоним ЧЕРЕЗ FramePump, а не напрямую taskMgr.step. Пока идёт
        # съёмка датасета (она сама крутит кадры), тик этого таймера, доехавший
        # через QApplication.processEvents, отсекается защитой от повторного
        # входа — вместо «Ignoring recursive poll()» и молча пропущенного кадра.
        #
        # Точный таймер с интервалом в период обновления экрана: грубый
        # (по умолчанию) QTimer на Windows округляет до ~15.6 мс, и ровные
        # 16 мс давали потолок 62 FPS с дрожанием интервала. Если кадр дольше
        # периода, следующий тик приходит сразу — упираемся в GPU, а не в
        # таймер; если короче — не гоняем сотни лишних кадров.
        self._panda_timer = QTimer(self)
        self._panda_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._panda_timer.timeout.connect(self._pump_frame)
        self._panda_timer.start(self._frame_interval_ms())

        # ---- Раскладка интерфейса над 3D-видом -------------------------
        #   слева сверху  — превью глубины («картинка в картинке»);
        #   сверху        — панель вида: режим камеры, виды 1-3, время
        #                   суток, превью, справка по клавишам;
        #   слева снизу   — строка телеметрии камеры;
        #   справа        — инспектор с вкладками (RightPanel).
        # Все панели — отдельные окна Qt.Tool (см. src/ui/hud.py).

        # ---- Превью глубины ------------------------------------------
        self.depth_overlay = DepthMapOverlay(
            parent=self.panda_container, anchor="top-left", margin=16,
            width=300,
        )
        self.depth_overlay.attach()
        self.depth_overlay.toggleRequested.connect(self._on_depth_toggle)
        # Диапазон глубины (ближняя / дальняя плоскость и градиент) — в
        # поповере под кнопкой на превью.
        try:
            from PyQt6.QtWidgets import (
                QDoubleSpinBox as _QDSB, QGridLayout as _QGL, QWidget as _QW,
            )
            from src.ui.hud import label as _hud_label, icon_label as _hud_icon

            depth_settings = _QW()
            grid = _QGL(depth_settings)
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setHorizontalSpacing(8)
            grid.setVerticalSpacing(8)

            def _make_spin(rng, val, step, decimals=2, suffix=""):
                sp = _QDSB()
                sp.setRange(*rng)
                sp.setSingleStep(step)
                sp.setDecimals(decimals)
                sp.setValue(val)
                if suffix:
                    sp.setSuffix(suffix)
                return sp

            self.spn_near = _make_spin((0.01, 1000.0), 0.1, 0.1, suffix=" м")
            self.spn_far  = _make_spin((0.1, 10000.0), 100.0, 1.0,
                                       decimals=1, suffix=" м")
            self.spn_g_a  = _make_spin((0.0, 1.0), 0.2, 0.05)
            self.spn_g_b  = _make_spin((0.0, 1.0), 0.4, 0.05)

            grid.addWidget(_hud_label("Плоскости", role="eyebrow"), 0, 0, 1, 2)
            grid.addWidget(_hud_icon("chevron_left", tooltip="Ближняя"), 1, 0)
            grid.addWidget(self.spn_near, 1, 1)
            grid.addWidget(_hud_icon("chevron_right", tooltip="Дальняя"), 2, 0)
            grid.addWidget(self.spn_far, 2, 1)
            grid.addWidget(_hud_label("Градиент", role="eyebrow"), 3, 0, 1, 2)
            grid.addWidget(_hud_icon("depth", tooltip="Начало градиента"), 4, 0)
            grid.addWidget(self.spn_g_a, 4, 1)
            grid.addWidget(_hud_icon("bars", tooltip="Конец градиента"), 5, 0)
            grid.addWidget(self.spn_g_b, 5, 1)
            grid.setColumnStretch(1, 1)
            self.spn_near.setToolTip("Ближняя плоскость")
            self.spn_far.setToolTip("Дальняя плоскость")
            self.spn_g_a.setToolTip("Начало градиента (0…1)")
            self.spn_g_b.setToolTip("Конец градиента (0…1)")

            self.spn_near.valueChanged.connect(self._on_depth_min_changed)
            self.spn_far.valueChanged.connect(self._on_depth_max_changed)
            self.spn_g_a.valueChanged.connect(self._on_depth_grad_a_changed)
            self.spn_g_b.valueChanged.connect(self._on_depth_grad_b_changed)

            self.depth_overlay.attach_extra(depth_settings)
        except Exception as exc:
            print(f"[MainWindow] depth-settings init failed: {exc}")
        # Кадры для превью берутся из panda_app.depth_renderer.depth_texture
        # (DepthMapRenderer сам обновляет её каждый кадр).
        self._depth_capture_w = 320
        self._depth_capture_h = 180   # 16:9
        self._depth_in_main = False   # default: main = normal, widget = depth
        self._color_mirror_tex = None
        self._color_mirror_buf = None
        self._color_mirror_cam = None
        self._depth_timer = QTimer(self)
        self._depth_timer.timeout.connect(self._tick_depth_overlay)
        # Первый тик — с задержкой, чтобы RenderPipeline успел загрузиться:
        # иначе заставка не закрывается и get_screenshot врёт. ~8 FPS для
        # превью достаточно и не мешает RP.
        QTimer.singleShot(3000, lambda: self._depth_timer.start(120))

        # ---- Телеметрия камеры ---------------------------------------
        self.telemetry = TelemetryHUD(self.panda_container, margin=16)
        self.telemetry.attach()

        # ---- Панель вида ---------------------------------------------
        self._camera_mode = "free"   # free | stationary | onboard
        # Пользовательские виды: dict {"pos", "hpr", "fov", ...} или None,
        # с диска — чтобы переживали перезапуск.
        self._cam_presets = self._load_cam_presets()
        self._preset_save_armed = False
        self._selected_preset = None
        # Пока ждём выбора слота для сохранения, все три мигают.
        self._preset_blink_on = False
        self._preset_blink_timer = QTimer(self)
        self._preset_blink_timer.timeout.connect(self._on_preset_blink_tick)

        self.toolbar = ViewToolbar(self.panda_container, margin=16)
        self._preset_btns = self.toolbar.preset_btns
        self._btn_preset_save = self.toolbar.btn_save
        self.daytime_slider = self.toolbar.daytime_slider
        self.toolbar.modeChanged.connect(self._on_camera_mode)
        self.toolbar.presetClicked.connect(self._on_preset_clicked)
        self.toolbar.presetMenuRequested.connect(self._on_preset_context_menu)
        self.toolbar.saveArmToggled.connect(self._on_preset_save_armed)
        self.toolbar.daytimeChanged.connect(self._on_daytime_changed)
        self.toolbar.pipToggled.connect(self._on_pip_toggled)
        self.toolbar.attach()
        self._apply_preset_styles()
        # Синхронизировать время суток в рендере с положением ползунка.
        self._on_daytime_changed(self.daytime_slider.value())

        self.right_panel = RightPanel(parent=self.panda_container)
        self.right_panel.attach()
        # Панель вида центрируется в свободной полосе между превью и
        # инспектором, чтобы не заходить ни под одно из них.
        self._update_toolbar_insets()
        # Съёмка датасета — вкладка «Датасет» инспектора.
        self.btn_dataset_setup = self.right_panel.btn_dataset_setup
        self.btn_save_render = self.right_panel.btn_dataset_start
        self.right_panel.datasetSettingsRequested.connect(
            self._on_dataset_settings_clicked)
        self.right_panel.datasetStartRequested.connect(
            self._on_save_render_clicked)
        self._refresh_dataset_summary()
        # Если конфиг текстур уже подтянут с сервера (см. main.py), сразу
        # перезаливаем выпадающий список. Безопасно вызывать и в случае,
        # когда конфига нет — метод просто оставит комбо как есть.
        try:
            server_tex_cfg = getattr(panda_app, "texture_sets", None) or {}
            if server_tex_cfg and hasattr(self.right_panel, "update_texture_sets"):
                texture_sets_list = [
                    (k, (v.get("name") or k) if isinstance(v, dict) else k)
                    for k, v in server_tex_cfg.items()
                    if k != "default" and isinstance(v, dict)
                ]
                self.right_panel.update_texture_sets(
                    texture_sets_list,
                    get_default_texture_set_key(),
                )
        except Exception as exc:
            print(f"[MainWindow] update_texture_sets failed: {exc}")
        # NOTE: depth_renderer is created lazily by MyApp.init_depth_renderer
        # (taskMgr.do_method_later(0.5, ...)), so it is still None right
        # now. The actual depth-camera reparent + lens copy happens on the
        # first depth tick where depth_renderer becomes available - see
        # `_sync_depth_camera_once`.
        self._depth_synced = False

        # Палитра классов сегментации из config/dataset.json: применяем один
        # раз при подключении рендера, чтобы маска и легенда в json совпадали
        # с тем, что выбрано в диалоге, ещё до первой съёмки.
        self._apply_dataset_palette(self._dataset_cfg)

        self.right_panel.runRequested.connect(self._on_run_simulation)
        # When the user picks a model from the combo we have to download
        # the cuzov/napolnitel/other .bam files into the temp cache and
        # load them into the Panda3D scene - otherwise perform_AABB_plane
        # has nothing to intersect against.
        self.right_panel.modelSetChanged.connect(self._on_model_set_changed)
        # Same for texture sets - set_texture_set on MyApp is the legacy
        # hook that drives the perlin generator's PBR slots.
        self.right_panel.textureSetChanged.connect(self._on_texture_set_changed)
        self.right_panel.reconstructionRunRequested.connect(
            self._on_reconstruction_run
        )
        # Graphics preset: persist the choice and prompt for a restart
        # (the rendering engine is chosen before the Panda window exists).
        self.right_panel.graphicsPresetChanged.connect(
            self._on_graphics_preset_changed
        )
        self.right_panel.bodyGenRequested.connect(self._on_bodygen_requested)
        self.right_panel.bodyWorklistRequested.connect(
            self._on_worklist_requested)
        self.right_panel.modelSetDeleteRequested.connect(
            self._on_model_delete_requested)
        self.right_panel.modelSetUploadRequested.connect(
            self._on_model_upload_requested)

        # ---- Camera-alignment reference overlay --------------------
        # Full-viewport translucent layer that shows a captured stand
        # snapshot's colour frame so the user can match the live camera.
        self.reference_overlay = CameraReferenceOverlay(
            parent=self.panda_container
        )
        self.reference_overlay.attach()
        self.right_panel.standReferenceSelected.connect(
            self._on_stand_reference_selected
        )
        self.right_panel.fovChanged.connect(self._on_fov_changed)
        self.right_panel.rollChanged.connect(self._on_roll_changed)
        self.right_panel.referenceOpacityChanged.connect(
            self._on_reference_opacity_changed
        )
        self.right_panel.referenceVisibleToggled.connect(
            self._on_reference_visible_toggled
        )

        # ---- Depth-fill reconstruction (N-point picking) -----------
        self._active_stand_rec = None
        self._last_auto_recon_depth = ""   # de-dupe auto-reconstruct calls
        try:
            from src.rendering.depth_reconstruction import DepthReconstructor
            self.depth_reconstructor = DepthReconstructor(panda_app)
            self.depth_reconstructor.on_count = self._on_pick_count
            self.depth_reconstructor.on_finished = self._on_reconstruct_finished
            self.depth_reconstructor.on_picking_state = self._on_picking_state
        except Exception as exc:
            self.depth_reconstructor = None
            print(f"[MainWindow] DepthReconstructor init failed: {exc}")
        self.right_panel.pointPickingToggled.connect(
            self._on_point_picking_toggled
        )
        self.right_panel.pointsResetRequested.connect(
            self._on_points_reset
        )
        self.right_panel.pointVizToggled.connect(self._on_point_viz_toggled)
        self.right_panel.autoPointsRequested.connect(
            self._on_auto_points_requested
        )

        # Fire an initial load so the default model is on the scene before
        # the user even picks anything.
        try:
            initial_key = self.right_panel.current_model_key()
            if initial_key:
                self._on_model_set_changed(str(initial_key))
            initial_tex = self.right_panel.current_texture_key()
            if initial_tex:
                self._on_texture_set_changed(str(initial_tex))
        except Exception as exc:
            print(f"[MainWindow] initial model/texture preload failed: {exc}")

        self._telemetry_timer = QTimer(self)
        self._telemetry_timer.timeout.connect(self._update_telemetry)
        self._telemetry_timer.start(80)

        # Кинематограф: заранее скомпилировать шейдеры и выделить буферы,
        # чтобы запуск сцены не начинался с секундного рывка.
        QTimer.singleShot(8000, self._warmup_cinematic)

    # ==================================================================
    # Панель вида: время суток, превью, раскладка
    # ==================================================================
    def _on_daytime_changed(self, mins: int) -> None:
        """Время суток: MyApp.set_time_of_day ведёт и daytime-менеджер
        RenderPipeline (ultra / medium), и солнце simplepbr (performance)."""
        app = self.panda_app
        if app is None:
            return
        try:
            if hasattr(app, "set_time_of_day"):
                app.set_time_of_day(int(mins))
            else:
                rp = getattr(app, "render_pipeline", None)
                dt_mgr = getattr(rp, "daytime_mgr", None) if rp else None
                if dt_mgr is not None:
                    dt_mgr.time = f"{int(mins) // 60:02d}:{int(mins) % 60:02d}"
        except Exception as exc:
            print(f"[Daytime] set failed: {exc}")

    def _on_pip_toggled(self, visible: bool) -> None:
        ov = getattr(self, "depth_overlay", None)
        if ov is not None:
            ov.set_user_visible(bool(visible))
        self._update_toolbar_insets()

    def _update_toolbar_insets(self) -> None:
        tb = getattr(self, "toolbar", None)
        if tb is None:
            return
        gap = 12
        pip = getattr(self, "depth_overlay", None)
        rp = getattr(self, "right_panel", None)
        tb.inset_left = (16 + pip.card_rect().width() + gap
                         if pip is not None and not pip.user_hidden else 0)
        tb.inset_right = (16 + rp.card_rect().width() + gap
                          if rp is not None else 0)
        tb._reposition()

    # ==================================================================
    # Graphics preset
    # ==================================================================
    def _on_graphics_preset_changed(self, preset_key: str) -> None:
        """
        Persist the chosen graphics preset and tell the user a restart is
        needed. The rendering engine (RenderPipeline vs simplepbr) is built
        before the Panda3D window exists, so it cannot be swapped live.
        """
        from src.core import graphics_settings
        from PyQt6.QtWidgets import QMessageBox

        graphics_settings.save(str(preset_key))
        name = graphics_settings.get_preset(str(preset_key)).get(
            "name", preset_key
        )
        print(f"[Graphics] preset saved: {preset_key}")
        QMessageBox.information(
            self,
            "Графика",
            f"Выбран пресет: {name}.\n\n"
            "Изменения вступят в силу после перезапуска приложения.",
        )

    # ==================================================================
    # Model / texture combo handlers
    # ==================================================================
    def _on_model_set_changed(self, model_key: str) -> None:
        """
        User picked a model in the right panel.  Download (or reuse the
        cached copy of) cuzov/napolnitel/other .bam files in
        %TEMP%/vizutil_models_cache and load them into the Panda3D scene
        via MyApp.cache_and_load_model_set.
        """
        if self.panda_app is None or not model_key:
            return
        cfg = get_model_set_config(str(model_key))
        if cfg is None:
            print(f"[ModelSet] config not found for {model_key!r}")
            return

        # Local truck models (assets/models/trucks) load straight from disk:
        # no server download, and no reference points / camera presets.
        if cfg.get("local"):
            if not hasattr(self.panda_app, "load_model_set"):
                print("[ModelSet] panda_app.load_model_set missing.")
                return
            print(f"[ModelSet] loading local model '{model_key}' ...")
            try:
                ok = bool(self.panda_app.load_model_set(cfg, str(model_key)))
                print(f"[ModelSet] {'OK' if ok else 'FAILED'} '{model_key}'")
            except Exception as exc:
                print(f"[ModelSet] local load_model_set raised: {exc}")
            return

        if not hasattr(self.panda_app, "cache_and_load_model_set"):
            print(f"[ModelSet] panda_app.cache_and_load_model_set missing.")
            return
        print(f"[ModelSet] caching + loading '{model_key}' ...")
        try:
            ok = bool(self.panda_app.cache_and_load_model_set(
                str(model_key), cfg
            ))
            print(f"[ModelSet] {'OK' if ok else 'FAILED'} '{model_key}'")
        except Exception as exc:
            print(f"[ModelSet] cache_and_load_model_set raised: {exc}")

    def _on_texture_set_changed(self, texture_key: str) -> None:
        """
        Пользователь выбрал текстурный набор в правой панели.

        Конфиг набора берётся из in-memory кэша (его наполняет main.py
        при старте). Перед тем как передавать набор в `panda_app.set_texture_set`,
        каждый ключ-путь к файлу текстуры (diffuse / normal / displacement /
        roughness / albedo / metallic / height) лениво докачивается в
        локальный кэш `%TEMP%/vizutil_textures_cache` и заменяется
        на абсолютный локальный путь, который Panda3D сможет открыть
        напрямую без обращения к серверу при каждом кадре.
        """
        if self.panda_app is None or not texture_key:
            return
        tex_cfg = get_texture_set_config(str(texture_key))
        if tex_cfg is None:
            print(f"[TextureSet] config not found for {texture_key!r}")
            return

        # Сохраняем СЫРОЙ конфиг с относительными путями — он нужен
        # серверу для displace-карты, передаётся через
        # tls_client.generate_landscape(displacement_path=...).
        # Materialized-версия в current_texture_set заменит пути на
        # локальный кэш, который сервер использовать не сможет.
        self.panda_app.current_texture_set_raw = dict(tex_cfg)

        resolved = self._materialize_texture_set(tex_cfg)
        if not hasattr(self.panda_app, "set_texture_set"):
            return
        try:
            self.panda_app.set_texture_set(resolved)
            print(f"[TextureSet] '{texture_key}' applied")
        except Exception as exc:
            print(f"[TextureSet] set_texture_set raised: {exc}")

    def _materialize_texture_set(self, tex_cfg: dict) -> dict:
        """
        Скопировать `tex_cfg` и подменить относительные пути к текстурам
        (по списку TEXTURE_PATH_KEYS) на локальные абсолютные пути из
        кэша, при необходимости скачав файлы с сервера.

        Не валит весь набор, если какая-то одна текстура не скачалась —
        просто оставляет в этом ключе исходный относительный путь, и
        дальше Panda3D отработает по своим резервным веткам.
        """
        out = dict(tex_cfg)
        tls = getattr(self.panda_app, "tls_client", None)
        for key in TEXTURE_PATH_KEYS:
            val = tex_cfg.get(key)
            if not isinstance(val, str) or not val:
                continue
            local = ensure_texture_cached(tls, val)
            if local:
                out[key] = local
            else:
                print(f"[TextureSet] не удалось закэшировать '{key}' "
                      f"({val}) — оставляем исходный путь")
        return out

    # ==================================================================
    # run_full_process - port of legacy gui.py
    # ==================================================================
    def _on_run_simulation(self, payload: dict) -> None:
        if self.panda_app is None:
            print("[Run] panda_app not attached - aborting.")
            return

        target_volume = float(payload.get("target_volume") or 0.0)
        model_key     = payload.get("model_key")
        texture_key   = payload.get("texture_key")
        # Пустой кузов — отдельная ветка, а не «наполнение объёмом 0»:
        # сервер объём 0 подобрать не может (см. _run_empty_body).
        empty         = bool(payload.get("empty"))

        print("=" * 60)
        print(f"[Run] Pipeline start. target_volume={target_volume:.2f}  "
              f"model_key={model_key!r}  texture_key={texture_key!r}"
              f"{'  EMPTY' if empty else ''}")

        # Resolve model + ground_plane_z BEFORE side-effects so we can
        # bail out early with a clear message.
        if not model_key:
            print(f"[Run] no model selected - abort.")
            print("=" * 60)
            return
        mc = get_model_set_config(str(model_key))
        if mc is None:
            print(f"[Run] model config '{model_key}' not in YAML - abort.")
            print("=" * 60)
            return
        try:
            ground_plane_z = float(mc.get("ground_plane", 0))
        except (TypeError, ValueError):
            ground_plane_z = 0.0
        print(f"[Run] ground_plane_z = {ground_plane_z}")

        # 0) Make sure the model set is loaded into the scene.
        already_loaded = (
            getattr(self.panda_app, "current_model_set", None) == model_key
            and getattr(self.panda_app, "loaded_models", None)
        )
        if (not already_loaded
                and hasattr(self.panda_app, "cache_and_load_model_set")):
            print(f"[Run] model set '{model_key}' not on scene - loading...")
            try:
                if not self.panda_app.cache_and_load_model_set(
                        str(model_key), mc):
                    print(f"[Run] cache_and_load_model_set FAILED - abort.")
                    print("=" * 60)
                    return
            except Exception as exc:
                print(f"[Run] ERR cache_and_load_model_set: {exc}")
                print("=" * 60)
                return

        # 1) Target volume
        try:
            self.panda_app.Target_Volume = target_volume
            print(f"[Run] OK Target_Volume = {target_volume}")
        except Exception as exc:
            print(f"[Run] ERR Target_Volume: {exc}")

        # 2) Texture set
        if texture_key and hasattr(self.panda_app, "set_texture_set"):
            tex_cfg = get_texture_set_config(str(texture_key))
            if tex_cfg is None:
                print(f"[Run] WARN texture '{texture_key}' not in server config")
            else:
                # Сохраняем сырой конфиг для серверного displace.
                self.panda_app.current_texture_set_raw = dict(tex_cfg)
                try:
                    self.panda_app.set_texture_set(
                        self._materialize_texture_set(tex_cfg)
                    )
                    print(f"[Run] OK texture set '{texture_key}' applied")
                except Exception as exc:
                    print(f"[Run] ERR set_texture_set: {exc}")

        # 2.5) Пустой кузов: ни ландшафта, ни boolean — просто снимаем меш
        # наполнения и уходим. Ground plane трогать нельзя (create_ground_plane
        # создаёт ВИДИМУЮ зелёную плоскость, а прячет её только удачный
        # perform_AABB_plane) — вместо этого прячем то, что есть.
        if empty:
            gen = getattr(self.panda_app, "perlin_generator", None)
            if gen is not None and hasattr(gen, "clear_fill_mesh"):
                try:
                    gen.clear_fill_mesh()
                    # Прокси-габарит наполнителя обычно прячет сам Perlin-этап;
                    # без него на свежезагруженном наборе в кадр попал бы
                    # сплошной блок вместо пустого кузова.
                    gen.hide_napolnitel_proxy()
                    print("[Run] OK пустой кузов: меш наполнения снят")
                except Exception as exc:
                    print(f"[Run] ERR clear_fill_mesh: {exc}")
            else:
                print("[Run] ERR perlin_generator не подключён — меш "
                      "наполнения снять нечем, кадр НЕ будет пустым.")
            try:
                gp = getattr(self.panda_app, "ground_plane", None)
                if gp is not None and not gp.is_empty():
                    gp.hide()
            except Exception as exc:
                print(f"[Run] WARN ground_plane.hide: {exc}")
            try:
                self.panda_app.update_overlay_info(volume=0.0)
            except Exception:
                pass
            print("=" * 60)
            return

        # 3) Ground plane (GREEN constant, then position)
        if hasattr(self.panda_app, "create_ground_plane"):
            try:
                self.panda_app.create_ground_plane()
                print(f"[Run] OK ground plane created (green constant)")
            except Exception as exc:
                print(f"[Run] ERR create_ground_plane: {exc}")
        try:
            gp = getattr(self.panda_app, "ground_plane", None)
            if gp is not None:
                gp.setPos(0, 0, ground_plane_z)
                print(f"[Run] OK ground_plane.setPos(0,0,{ground_plane_z})")
            else:
                print(f"[Run] WARN panda_app.ground_plane is None")
        except Exception as exc:
            print(f"[Run] ERR ground_plane.setPos: {exc}")

        # 4) AABB plane
        success_aabb = False
        if hasattr(self.panda_app, "perform_AABB_plane"):
            try:
                success_aabb = bool(self.panda_app.perform_AABB_plane())
                print(f"[Run] AABB plane -> {success_aabb}")
            except Exception as exc:
                print(f"[Run] ERR perform_AABB_plane: {exc}")
        else:
            print(f"[Run] SKIP perform_AABB_plane not implemented.")

        # 5) Perlin mesh from CSG
        if not success_aabb:
            print(f"[Run] AABB unsuccessful - skipping Perlin.")
            print("=" * 60)
            return

        gen = getattr(self.panda_app, "perlin_generator", None)
        if gen is not None and hasattr(gen, "generate_perlin_mesh_from_csg"):
            try:
                ok = bool(gen.generate_perlin_mesh_from_csg())
                if ok:
                    print(f"[Run] OK pipeline finished. "
                          f"Target Volume={target_volume}, "
                          f"ground_plane_z={ground_plane_z}")
                else:
                    print(f"[Run] ERR Perlin mesh generation failed.")
            except Exception as exc:
                print(f"[Run] ERR Perlin: {exc}")
        else:
            print(f"[Run] SKIP perlin_generator not connected.")
        print("=" * 60)

    # ==================================================================
    # Reconstruction (2D->3D) trigger - port of legacy gui.py
    # ==================================================================
    def _on_reconstruction_run(self, rec: Reconstruction) -> None:
        """
        User clicked a reconstruction row in the right panel.
        Mirrors gui.on_recon_file_clicked: resolve / fetch JSON and PLY,
        apply texture-set by 'filler', load the matching model set, then
        delegate to panda_app.mesh_reconstruction.run_2d_to_3d_reconstruction_from.
        """
        if self.panda_app is None:
            print("[Recon] panda_app not attached - aborting.")
            return
        if rec is None:
            return

        # Stand snapshots / серверные depth-записи используют depth-пайплайн
        # (anchor points + depth map), не серверную JSON/PLY реконструкцию.
        if getattr(rec, "data_type", "") == "depth":
            # Серверные depth-записи приходят с именами файлов;
            # скачиваем их и подставляем локальные абсолютные пути.
            self._materialize_depth_record_paths(rec)
            # Для depth-записей используем жёстко заданные 16 опорных
            # точек (presets/depth_anchors_world.json) + первый camera
            # preset — пользователю достаточно одной кнопки.
            self._run_depth_reconstruction(rec)
            return
        if getattr(rec, "data_type", "") == "stand":
            self._run_stand_reconstruction(rec)
            return

        # Кинематографичный показ (src/cinematic): снимок, лидар, поиск
        # кузова, этапы расчёта. Нужен RenderPipeline.
        if (getattr(rec, "data_type", "") == "ply"
                and getattr(self, "right_panel", None) is not None
                and self.right_panel.cinematic_enabled()):
            try:
                from src.cinematic.session import CineSession
                if CineSession.supported(self.panda_app):
                    CineSession(self, rec).start()
                    return
            except Exception as exc:
                traceback.print_exc()
                print(f"[Recon] кинематограф недоступен: {exc}")
        self._on_reconstruction_run_plain(rec)

    def _warmup_cinematic(self) -> None:
        app = self.panda_app
        rp = getattr(self, "right_panel", None)
        if app is None or rp is None or not rp.cinematic_enabled():
            return
        try:
            from src.cinematic.session import CineSession
            from src.cinematic import scan_analysis
            from src.vfx.compositor import Compositor
            if CineSession.supported(app) and CineSession.current is None:
                Compositor.shared(app, app.render_pipeline).warmup()
                # Depth Anything, open3d, детектор кузова — в постоянном
                # процессе, прогретые заранее
                scan_analysis.warm_up()
        except Exception as exc:
            print(f"[cine] прогрев не удался: {exc}")

    # ------------------------------------------------------------------
    def fade_overlays(self, visible: bool, duration_ms: int = 450) -> None:
        """
        Плавно спрятать / вернуть весь интерфейс поверх 3D-вида на время
        кинематографичной сцены. Карточки и панели — отдельные окна-Tool,
        принадлежащие главному окну (см. overlay_widgets), поэтому гасятся
        прозрачностью окна, а не графическим эффектом.
        """
        from PyQt6.QtCore import QPropertyAnimation, QEasingCurve
        from PyQt6.QtWidgets import QApplication

        if not visible:
            self._faded_overlays = [
                w for w in QApplication.topLevelWidgets()
                if w is not self and w.isVisible()
                and getattr(w, "_owner", None) is self.panda_container]
        widgets = list(getattr(self, "_faded_overlays", []))
        self._overlay_anims = []
        for w in widgets:
            if visible:
                w.setWindowOpacity(0.0)
                w.show()
            anim = QPropertyAnimation(w, b"windowOpacity", self)
            anim.setDuration(duration_ms)
            anim.setStartValue(w.windowOpacity())
            anim.setEndValue(1.0 if visible else 0.0)
            anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
            if not visible:
                anim.finished.connect(w.hide)
            anim.start()
            self._overlay_anims.append(anim)
        if visible:
            self._faded_overlays = []

    def _on_reconstruction_run_plain(self, rec: Reconstruction) -> None:
        """Штатная реконструкция: загрузка в фоне, сцена — в главном потоке."""
        recon_module = getattr(self.panda_app, "mesh_reconstruction", None)
        if recon_module is None:
            print("[Recon] panda_app.mesh_reconstruction not available.")
            return

        print("=" * 60)
        print(f"[Recon] click '{rec.name}' (data_type={rec.data_type}, "
              f"is_local={rec.is_local})")

        # Сеть, диск и разбор меша — в фоновом потоке: холодный запуск —
        # это десятки секунд скачиваний, и всё это время рендер раньше стоял.
        # В сцену результат ставится уже в главном потоке
        # (_apply_reconstruction). Повторный клик во время загрузки
        # перекрывает прошлый запуск: его результат просто выбрасывается.
        self._recon_seq = getattr(self, "_recon_seq", 0) + 1
        seq = self._recon_seq
        worker = _CallInThread(lambda: self._fetch_reconstruction(rec), self)
        worker.finishedWith.connect(
            lambda out, _seq=seq: self._apply_reconstruction(rec, out, _seq))
        worker.finished.connect(worker.deleteLater)
        self._recon_worker = worker
        self.reconstructionStage.emit("started", {"rec": rec})
        worker.start()

    # ------------------------------------------------------------------
    def _fetch_reconstruction(self, rec: Reconstruction) -> dict:
        """
        Фоновая часть реконструкции: всё, что не трогает сцену.

        Скачивает JSON, текстуры наполнителя, файлы модели, PLY и готовый
        меш, разбирает меш. Возвращает словарь для _apply_reconstruction; при
        отказе — {"error": ...}. Исключения ловит сам поток (_CallInThread).
        """
        recon_module = self.panda_app.mesh_reconstruction

        # ---- 1) Resolve JSON path (local or download) ---------------
        local_json_path = self._resolve_recon_json(rec)
        if not local_json_path or not os.path.exists(local_json_path):
            return {"error": f"could not resolve JSON for {rec.name!r}"}

        # ---- 2) Parse JSON ------------------------------------------
        try:
            with open(local_json_path, "r", encoding="utf-8") as fp:
                json_data = json.load(fp)
        except Exception as exc:
            return {"error": f"failed to read JSON: {exc}"}

        out: dict = {"json_path": local_json_path, "json": json_data}
        filler = json_data.get("filler") or rec.filler
        model_name = json_data.get("model") or rec.model

        # ---- 3) Texture set by filler: скачать файлы ----------------
        if filler:
            tex_key, tex_cfg = self._find_texture_by_filler(filler)
            if tex_cfg is not None:
                out["texture_key"] = tex_key
                out["texture_raw"] = dict(tex_cfg)
                out["texture_set"] = self._materialize_texture_set(tex_cfg)
            else:
                print(f"[Recon] no texture set found for filler='{filler}'")

        # ---- 4) Model set: скачать файлы ----------------------------
        if model_name:
            model_key = self._find_model_key_by_name(model_name)
            cfg = get_model_set_config(model_key) if model_key else None
            if cfg is not None and hasattr(
                    self.panda_app, "download_and_cache_model_set"):
                try:
                    out["model_key"] = model_key
                    out["model_cfg"] = cfg
                    files = self.panda_app.download_and_cache_model_set(
                        model_key, cfg)
                    out["model_files"] = files
                    # Прочитать модели и их текстуры здесь же, в потоке, —
                    # главному останется только повесить их в сцену.
                    self.panda_app.preload_model_files(
                        files.get(k) for k in ("other", "cuzov", "napolnitel"))
                except Exception as exc:
                    print(f"[Recon] model set '{model_key}' not cached: {exc}")
            elif model_key:
                print(f"[Recon] config for '{model_key}' missing.")
            else:
                print(f"[Recon] no model key for '{model_name}'.")

        # ---- 5) Resolve PLY path (download if SERVER) ---------------
        ply_filename = json_data.get("ply_file") or rec.ply_file
        if ply_filename:
            local_ply_path = self._resolve_recon_ply(rec, ply_filename,
                                                    local_json_path)
            out["ply_path"] = local_ply_path
            if local_ply_path:
                print(f"[Recon] PLY ready: {local_ply_path}")
            else:
                print(f"[Recon] PLY '{ply_filename}' could not be resolved")

        # ---- 6) Resolve heightmap (only for data_type='height') -----
        if rec.data_type == "height":
            heightmap_filename = json_data.get("heightmap_path", "")
            if heightmap_filename:
                local_hm_path = self._resolve_recon_heightmap(
                    rec, heightmap_filename, local_json_path
                )
                if not local_hm_path or not os.path.exists(local_hm_path):
                    return {"error": f"heightmap '{heightmap_filename}' "
                                     f"could not be resolved"}

        # ---- 7) Готовый меш: скачать и разобрать --------------------
        # Повторы текстуры — от набора, который встанет в сцену (с теми же
        # умолчаниями, что добавляет MyApp.set_texture_set), иначе — от
        # текущего.
        from src.rendering.mesh_reconstruction import uv_scale_for
        tex_set = out.get("texture_set")
        if tex_set is not None:
            tex_set = {"textureRepeatX": 1.35, "textureRepeatY": 3.2,
                       **tex_set}
        else:
            tex_set = getattr(self.panda_app, "current_texture_set", None)
        out["mesh"] = recon_module.prepare(local_json_path,
                                           uv_scale=uv_scale_for(tex_set))
        return out

    # ------------------------------------------------------------------
    def _apply_reconstruction(self, rec: Reconstruction, out, seq: int) -> None:
        """Главный поток: поставить скачанное и разобранное в сцену."""
        if seq != getattr(self, "_recon_seq", 0):
            print(f"[Recon] '{rec.name}': результат устарел — пропускаю")
            return
        if isinstance(out, BaseException) or "error" in out:
            err = out if isinstance(out, BaseException) else out["error"]
            print(f"[Recon] ERR {err}")
            print("=" * 60)
            self.reconstructionStage.emit("failed",
                                          {"rec": rec, "error": str(err)})
            return

        json_data = out["json"]
        prepared = out.get("mesh")
        self.reconstructionStage.emit("fetched", {
            "rec": rec,
            "json": json_data,
            "ply_path": out.get("ply_path"),
            "model_key": out.get("model_key"),
            "texture_key": out.get("texture_key"),
            "vertices": getattr(prepared, "vertices", None),
            "faces": getattr(prepared, "faces", None),
            "volume": json_data.get("target_volume"),
        })

        filler = json_data.get("filler") or rec.filler

        # ---- 3) Apply texture set by filler -------------------------
        if out.get("texture_set") is not None and hasattr(
                self.panda_app, "set_texture_set"):
            # Сохраняем сырой конфиг для серверного displace.
            self.panda_app.current_texture_set_raw = out["texture_raw"]
            try:
                self.panda_app.set_texture_set(out["texture_set"])
                print(f"[Recon] texture set by filler: '{out['texture_key']}'")
            except Exception as exc:
                print(f"[Recon] set_texture_set failed: {exc}")

        # ---- 4) Load model set (файлы уже в кэше) -------------------
        model_key = out.get("model_key")
        if model_key and out.get("model_files") is not None:
            cached = out["model_files"]
            model_config = {
                "cuzov":        cached.get("cuzov"),
                "napolnitel":   cached.get("napolnitel"),
                "other":        cached.get("other"),
                "max_volume":   cached.get("max_volume"),
                "ground_plane": cached.get("ground_plane"),
                "target_model": out["model_cfg"].get("target_model"),
            }
            try:
                ok = bool(self.panda_app.load_model_set(model_config,
                                                        model_key))
                print(f"[Recon] model set '{model_key}' loaded: {ok}")
                # Синхронизируем выбор в правой панели: иначе
                # right_panel.current_model_key() продолжает
                # возвращать прежний (дефолтный) ключ, и
                # _apply_onboard_camera берёт камеру не той модели.
                if ok:
                    rp = getattr(self, "right_panel", None)
                    if rp is not None and hasattr(rp, "set_current_model_key"):
                        rp.set_current_model_key(model_key)
                    # Если на момент реконструкции уже включён
                    # бортовой вид — пересобираем pos/hpr камеры
                    # под новую модель сразу, чтобы пользователю
                    # не пришлось переключать режим вручную.
                    if getattr(self, "_camera_mode", None) == "onboard":
                        try:
                            self._apply_onboard_camera()
                        except Exception as exc:
                            print(f"[Recon] reapply onboard: {exc}")
            except Exception as exc:
                print(f"[Recon] load_model_set: {exc}")
        self.panda_app.drop_preloaded_models()

        # ---- 7) Push overlay info if MyApp supports it --------------
        try:
            if hasattr(self.panda_app, "update_overlay_info"):
                self.panda_app.update_overlay_info(
                    texture=filler,
                    car_number=json_data.get("car_number") or rec.car_number,
                    initial_volume=json_data.get("target_volume"),
                    time=json_data.get("time") or rec.time,
                )
        except Exception as exc:
            print(f"[Recon] update_overlay_info failed: {exc}")

        # ---- 8) Put the mesh into the scene -------------------------
        node = None
        if prepared is not None:
            try:
                node = self.panda_app.mesh_reconstruction.apply(prepared)
                print("[Recon] OK pipeline finished.")
            except Exception as exc:
                print(f"[Recon] ERR mesh_reconstruction.apply: {exc}")
        print("=" * 60)
        if node is None:
            self.reconstructionStage.emit(
                "failed", {"rec": rec, "error": "меш не построен"})
            return
        self.reconstructionStage.emit("applied", {
            "rec": rec, "node": node,
            "volume": json_data.get("target_volume")})

    # ------------------------------------------------------------------
    # Recon helpers
    # ------------------------------------------------------------------
    def _resolve_recon_json(self, rec: Reconstruction) -> str | None:
        """Return a local path to the JSON for `rec`. Downloads if SERVER."""
        if rec.is_local and rec.path and os.path.exists(rec.path):
            return rec.path
        # SERVER entry - cache under %TEMP%/vizutil_recon.
        temp_dir = os.path.join(tempfile.gettempdir(), "vizutil_recon")
        os.makedirs(temp_dir, exist_ok=True)
        local_path = os.path.join(temp_dir, rec.name)
        if os.path.exists(local_path) and os.path.getsize(local_path) > 0:
            return local_path
        try:
            self.panda_app.tls_client.download_file(rec.name, local_path)
        except Exception as exc:
            print(f"[Recon] download_file('{rec.name}') failed: {exc}")
            return None
        return local_path if os.path.exists(local_path) else None

    def _resolve_recon_ply(self, rec: Reconstruction,
                            ply_filename: str,
                            local_json_path: str) -> str | None:
        target_dir = os.path.dirname(local_json_path)
        local_ply = os.path.join(target_dir, ply_filename)
        if os.path.exists(local_ply):
            return local_ply
        if rec.is_local:
            src_dir = os.path.dirname(rec.path)
            src_ply = os.path.join(src_dir, ply_filename)
            if os.path.exists(src_ply):
                try:
                    shutil.copy2(src_ply, local_ply)
                    return local_ply
                except Exception as exc:
                    print(f"[Recon] copy local PLY failed: {exc}")
                    return None
            return None
        # SERVER PLY
        try:
            self.panda_app.tls_client.download_file(ply_filename, local_ply)
            return local_ply if os.path.exists(local_ply) else None
        except Exception as exc:
            print(f"[Recon] download PLY '{ply_filename}' failed: {exc}")
            return None

    def _resolve_recon_heightmap(self, rec: Reconstruction,
                                  heightmap_filename: str,
                                  local_json_path: str) -> str | None:
        """
        MeshReconstruction.load_height_map expects the heightmap right
        next to the JSON, so copy / download it there.
        """
        target_dir = os.path.dirname(local_json_path)
        local_hm = os.path.join(target_dir, heightmap_filename)
        if os.path.exists(local_hm):
            return local_hm
        if rec.is_local:
            src_dir = os.path.dirname(rec.path)
            src_hm = os.path.join(src_dir, heightmap_filename)
            if os.path.exists(src_hm):
                try:
                    shutil.copy2(src_hm, local_hm)
                    return local_hm
                except Exception as exc:
                    print(f"[Recon] copy local heightmap failed: {exc}")
                    return None
            return None
        # SERVER heightmap
        try:
            self.panda_app.tls_client.download_file(heightmap_filename,
                                                    local_hm)
            return local_hm if os.path.exists(local_hm) else None
        except Exception as exc:
            print(f"[Recon] download heightmap "
                  f"'{heightmap_filename}' failed: {exc}")
            return None

    @staticmethod
    def _find_texture_by_filler(filler: str) -> tuple[str | None, dict | None]:
        """Find a texture set whose 'name' matches `filler`."""
        if not filler:
            return None, None
        for key, _disp in load_texture_sets():
            if key == "default":
                continue
            cfg = get_texture_set_config(key)
            if cfg and cfg.get("name") == filler:
                return key, cfg
        return None, None

    @staticmethod
    def _find_model_key_by_name(model_name: str) -> str | None:
        """Find a model set key whose 'model' field matches `model_name`."""
        if not model_name:
            return None
        from src.ui.panel_data import load_model_sets
        for key, _disp in load_model_sets():
            cfg = get_model_set_config(key)
            if cfg and cfg.get("model") == model_name:
                return key
        return None

    # ==================================================================
    # Telemetry
    # ==================================================================
    # ==================================================================
    # Depth-map overlay tick
    # ==================================================================
    # ------------------------------------------------------------------
    def _on_depth_toggle(self) -> None:
        """
        Swap main viewport contents.  Calls panda_app.toggle_depth_overlay()
        which shows / hides the depth fullscreen quad on render2d, and
        flips our internal flag so the depth widget switches its source.
        """
        if self.panda_app is None:
            return
        try:
            depth_now_in_main = bool(self.panda_app.toggle_depth_overlay())
        except Exception as exc:
            print(f"[Depth] toggle failed: {exc}")
            return
        self._depth_in_main = depth_now_in_main
        try:
            self.depth_overlay.set_toggle_state(depth_now_in_main)
        except Exception:
            pass
        # If switching to "depth in main, normal in widget", make sure
        # the color mirror is up.
        if depth_now_in_main:
            self._ensure_color_mirror()
        print(f"[Depth] toggled. depth_in_main={depth_now_in_main}")

    # ------------------------------------------------------------------
    def _ensure_color_mirror(self) -> None:
        """
        Build a low-res offscreen buffer that mirrors the 3D scene from
        the main camera's POV.  The buffer renders the SAME 3D scene
        (from a clone of the main camera's lens, parented to main camera
        so it follows it), but does NOT include the render2d depth
        overlay - which is exactly what we need so toggling the overlay
        on the main window swaps what the widget shows.

        Trade-off: no RenderPipeline post-effects (RP attaches its
        passes to base.cam only).  The widget shows a basic lit version
        of the scene, but the swap is visually unambiguous: depth pass
        on one side, lit scene on the other.
        """
        if self._color_mirror_tex is not None:
            return
        try:
            from panda3d.core import Texture
            tex = Texture("color_mirror")
            # Кадр читается асинхронно (src/core/gl_sync.AsyncReadback):
            # to_ram=True копировал его в RAM каждый кадр синхронно, с
            # ожиданием всей очереди GPU. Без PyOpenGL — старый путь.
            reader_ok = True
            try:
                from src.core.gl_sync import available as _gl_ok
                reader_ok = _gl_ok()
            except Exception:
                reader_ok = False
            if not reader_ok:
                tex.set_keep_ram_image(True)
            buf = self.panda_app.win.make_texture_buffer(
                "color_mirror_buf",
                self._depth_capture_w, self._depth_capture_h,
                tex, to_ram=not reader_ok,
            )
            if buf is None:
                print("[Depth] make_texture_buffer returned None.")
                return
            cam = self.panda_app.makeCamera(
                buf,
                lens=self.panda_app.cam.node().get_lens(),
            )
            cam.reparent_to(self.panda_app.camera)
            cam.set_pos(0, 0, 0)
            cam.set_hpr(0, 0, 0)
            self._color_mirror_reader = None
            if reader_ok:
                from src.core.gl_sync import AsyncReadback
                self._color_mirror_reader = AsyncReadback(
                    buf.get_display_region(buf.get_num_display_regions() - 1),
                    self._depth_capture_w, self._depth_capture_h,
                    channels=4, dtype="uint8")
            self._color_mirror_tex = tex
            self._color_mirror_buf = buf
            self._color_mirror_cam = cam
            print(f"[Depth] color mirror created "
                  f"({self._depth_capture_w}x{self._depth_capture_h})")
        except Exception as exc:
            print(f"[Depth] color mirror init failed: {exc}")

    # ------------------------------------------------------------------
    def _arm_continuous_depth_pass(self, dr) -> None:
        """
        One-shot setup that makes the depth pass cheap to consume:
          1. Reparent depth_camera_np onto panda_app.camera so it
             auto-tracks the main camera (no manual set_pos every frame).
          2. Mirror the main lens's FOV.
          3. Set depth_buffer permanently active so it renders every
             frame as part of Panda's normal loop.
          4. Override update_depth_texture into a no-op so the existing
             overlay-task (which still fires when overlay is visible)
             doesn't keep deactivating the buffer or fighting our
             reparented transform.
        """
        try:
            cam_np = getattr(dr, "depth_camera_np", None)
            if cam_np is not None:
                cam_np.reparent_to(self.panda_app.camera)
                cam_np.set_pos(0, 0, 0)
                cam_np.set_hpr(0, 0, 0)
                cam_np.set_scale(1, 1, 1)
                dn = cam_np.node()
                if dn is not None and dn.get_lens() is not None:
                    main_lens = self.panda_app.cam.node().get_lens()
                    if main_lens is not None and hasattr(main_lens, "get_fov"):
                        dn.get_lens().set_fov(main_lens.get_fov())
            buf = getattr(dr, "depth_buffer", None)
            if buf is not None:
                buf.set_active(True)
            def _noop_update_depth_texture(_dr=dr):
                # Buffer renders continuously, no manual update needed.
                return True
            dr.update_depth_texture = _noop_update_depth_texture
            print("[Depth] continuous depth pass armed.")
        except Exception as exc:
            print(f"[Depth] arm_continuous_depth_pass failed: {exc}")

    # ==================================================================
    # Depth-pass parameter handlers (wired to the QDoubleSpinBoxes
    # in the DepthMapOverlay's settings strip)
    # ==================================================================
    def _on_depth_min_changed(self, value: float) -> None:
        if self.panda_app is None:
            return
        dr = getattr(self.panda_app, "depth_renderer", None)
        if dr is None:
            return
        try:
            dr.min_depth = float(value)
            if getattr(dr, "depth_camera_np", None) is not None:
                lens = dr.depth_camera_np.node().get_lens()
                if lens is not None:
                    lens.set_near_far(float(value), float(dr.max_depth))
            if getattr(dr, "overlay_node", None) is not None:
                dr.overlay_node.setShaderInput("near", float(value))
        except Exception as exc:
            print(f"[Depth] min_depth update failed: {exc}")

    def _on_depth_max_changed(self, value: float) -> None:
        if self.panda_app is None:
            return
        dr = getattr(self.panda_app, "depth_renderer", None)
        if dr is None:
            return
        try:
            dr.max_depth = float(value)
            if getattr(dr, "depth_camera_np", None) is not None:
                lens = dr.depth_camera_np.node().get_lens()
                if lens is not None:
                    lens.set_near_far(float(dr.min_depth), float(value))
            if getattr(dr, "overlay_node", None) is not None:
                dr.overlay_node.setShaderInput("far", float(value))
        except Exception as exc:
            print(f"[Depth] max_depth update failed: {exc}")

    def _on_depth_grad_a_changed(self, value: float) -> None:
        dr = getattr(self.panda_app, "depth_renderer", None) if self.panda_app else None
        if dr is None:
            return
        try:
            dr.set_gradient_start(float(value))
        except Exception as exc:
            print(f"[Depth] gradient_start update failed: {exc}")

    def _on_depth_grad_b_changed(self, value: float) -> None:
        dr = getattr(self.panda_app, "depth_renderer", None) if self.panda_app else None
        if dr is None:
            return
        try:
            dr.set_gradient_end(float(value))
        except Exception as exc:
            print(f"[Depth] gradient_end update failed: {exc}")

    def _tick_depth_overlay(self) -> None:
        """
        Push a frame to the depth widget.  Source depends on toggle:
            self._depth_in_main = False -> show DEPTH (default)
            self._depth_in_main = True  -> show NORMAL render mirror
        """
        if self.panda_app is None or not hasattr(self, "depth_overlay"):
            return
        if self._depth_in_main:
            self._tick_color_mirror()
            return
        dr = getattr(self.panda_app, "depth_renderer", None)
        if dr is None or getattr(dr, "depth_texture", None) is None:
            return

        # First time depth_renderer is available - reparent the depth
        # camera onto the main camera and turn the depth buffer into a
        # permanent render so we don't have to drive it from the tick.
        if not getattr(self, "_depth_pass_armed", False):
            self._arm_continuous_depth_pass(dr)
            self._depth_pass_armed = True

        # Keep the depth camera's lens FOV mirrored to the main lens. The
        # depth camera has its OWN lens (reparented onto the main camera in
        # _arm_continuous_depth_pass), so FOV-slider / camera-mode changes
        # don't reach it automatically. Cheap, and covers every FOV source.
        self._mirror_depth_camera_fov(dr)

        # Раскраска — в src/ui/depth_preview: тот же код кормит превью в
        # диалоге настроек датасета, поэтому картинки гарантированно
        # совпадают.
        img = depth_to_qimage(
            dr, self._depth_capture_w, self._depth_capture_h,
            grayscale=bool(getattr(dr, "grayscale", False)),
        )
        if img is not None:
            self.depth_overlay.set_image(img)

    def _tick_color_mirror(self) -> None:
        """
        Read the offscreen color-mirror texture's RAM copy and push it
        to the depth widget as RGBA.  Cheap: no render_frame, no PNG
        encode - just a memcpy of (W*H*4) bytes.
        """
        if self._color_mirror_tex is None:
            self._ensure_color_mirror()
            if self._color_mirror_tex is None:
                return
        try:
            import numpy as np
            from PyQt6.QtGui import QImage

            reader = getattr(self, "_color_mirror_reader", None)
            if reader is not None:
                reader.request()
                arr = reader.latest
                if arr is None:
                    return
                th, tw = arr.shape[:2]
                img = QImage(np.ascontiguousarray(arr).tobytes(), tw, th,
                             tw * 4, QImage.Format.Format_RGBA8888)
                # GL отдаёт строки снизу вверх.
                self.depth_overlay.set_image(img.mirrored(False, True).copy())
                return

            tex = self._color_mirror_tex
            if not tex.has_ram_image():
                return
            ram = tex.get_ram_image_as("RGBA")
            if ram is None:
                return
            buf = memoryview(ram).tobytes()
            if not buf:
                return
            tw = tex.get_x_size()
            th = tex.get_y_size()
            if tw * th * 4 != len(buf):
                return
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(th, tw, 4)
            arr = np.ascontiguousarray(arr)
            data = arr.tobytes()
            img = QImage(data, tw, th, tw * 4,
                         QImage.Format.Format_RGBA8888)
            img = img.mirrored(False, True)   # Panda flips Y
            img = img.copy()
            self.depth_overlay.set_image(img)
        except Exception as exc:
            print(f"[Depth] color mirror tick failed: {exc}")

    # ==================================================================
    # Camera mode handlers (free / stationary / onboard)
    # ==================================================================
    # Stationary preset — pinned by the user as a known-good viewpoint.
    _STATIONARY_POS = (1.0, 1.1, 8.0)
    _STATIONARY_HPR = (-627.9, -74.1, 0.0)   # h=yaw, p=pitch, r=roll
    _STATIONARY_FOV = 100.0
    # Стандартное FOV из RenderPipeline (см. rpcore/render_pipeline.py:476 —
    # self._showbase.camLens.set_fov(125)). При входе в бортовой режим
    # _apply_stationary_camera мог оставить FOV=100 от предыдущего STATIC,
    # из-за чего бортовая камера выглядела неправильно. Возвращаем на pipeline-дефолт.
    _ONBOARD_FOV = 125.0

    # Depth-pass presets per camera mode: (near, far, gradient_start, gradient_end).
    _STATIONARY_DEPTH = (0.01, 64.0, 0.10, 0.25)
    _ONBOARD_DEPTH    = (0.01, 58.0, 0.01, 0.19)

    def _apply_depth_preset(self, preset: tuple) -> None:
        """Drive the depth-settings spin boxes; their valueChanged signals
        propagate to the depth_renderer."""
        try:
            near, far, g_a, g_b = preset
        except (TypeError, ValueError):
            return
        # Order: far before near to avoid a transient where new near > old far
        # would clamp the spin box; same logic for grad start/end.
        for spin, value in (
            (getattr(self, "spn_far", None),  far),
            (getattr(self, "spn_near", None), near),
            (getattr(self, "spn_g_b", None),  g_b),
            (getattr(self, "spn_g_a", None),  g_a),
        ):
            if spin is None:
                continue
            try:
                spin.setValue(float(value))
            except Exception as exc:
                print(f"[Depth] preset set failed: {exc}")

    def _on_camera_mode(self, mode: str) -> None:
        if self.panda_app is None:
            return
        if mode not in ("free", "stationary", "onboard"):
            return
        self._camera_mode = mode

        # Отразить режим на панели вида (сигнал не повторяется).
        tb = getattr(self, "toolbar", None)
        if tb is not None:
            tb.set_mode(mode)

        if mode == "free":
            self._apply_free_camera()
        elif mode == "stationary":
            self._apply_stationary_camera()
        else:  # onboard
            self._apply_onboard_camera()

    # ------------------------------------------------------------------
    def _apply_free_camera(self) -> None:
        fc = getattr(self.panda_app, "fly_cam", None)
        if fc is not None and hasattr(fc, "set_frozen"):
            fc.set_frozen(False)
        # Default presets straighten the camera (roll = 0).
        self._apply_camera_roll(0.0)
        self._sync_roll_dial(0.0)

    # ------------------------------------------------------------------
    def _apply_stationary_camera(self) -> None:
        fc = getattr(self.panda_app, "fly_cam", None)
        if fc is not None and hasattr(fc, "set_frozen"):
            fc.set_frozen(True)
        cam = getattr(self.panda_app, "camera", None)
        if cam is None:
            return
        try:
            cam.set_pos(*self._STATIONARY_POS)
            cam.set_hpr(*self._STATIONARY_HPR)
            lens = self.panda_app.cam.node().get_lens()
            if lens is not None and hasattr(lens, "set_fov"):
                lens.set_fov(self._STATIONARY_FOV)
            self._sync_fov_slider(self._STATIONARY_FOV)
            # Default preset → roll = 0 (also resets the fly cam's stored
            # roll so a later free-fly doesn't inherit a stale value).
            self._apply_camera_roll(0.0)
            self._sync_roll_dial(0.0)
            print(f"[Camera] STATIC pos={self._STATIONARY_POS} "
                  f"hpr={self._STATIONARY_HPR} fov={self._STATIONARY_FOV}")
        except Exception as exc:
            print(f"[Camera] stationary preset failed: {exc}")
        self._apply_depth_preset(self._STATIONARY_DEPTH)

    # ------------------------------------------------------------------
    def _apply_onboard_camera(self) -> None:
        """
        On-board view depends on the currently-selected model set:
        cam_pos_x/y/z + cam_rot_h/p/r from models_config.yaml.
        """
        rp = getattr(self, "right_panel", None)
        key = rp.current_model_key() if rp is not None else None
        if not key:
            print("[Camera] onboard: no model set selected.")
            return
        cfg = get_model_set_config(str(key))
        if not cfg:
            print(f"[Camera] onboard: config for '{key}' missing.")
            return
        try:
            cx = float(cfg.get("cam_pos_x", 0))
            cy = float(cfg.get("cam_pos_y", 0))
            cz = float(cfg.get("cam_pos_z", 0))
            ch = float(cfg.get("cam_rot_h", 0))
            cp = float(cfg.get("cam_rot_p", 0))
            cr = float(cfg.get("cam_rot_r", 0))
        except (TypeError, ValueError) as exc:
            print(f"[Camera] onboard: bad cam_* in '{key}': {exc}")
            return

        fc = getattr(self.panda_app, "fly_cam", None)
        if fc is not None and hasattr(fc, "set_frozen"):
            fc.set_frozen(True)
        cam = getattr(self.panda_app, "camera", None)
        if cam is None:
            return
        try:
            cam.set_pos(cx, cy, cz)
            cam.set_hpr(ch, cp, cr)
            # Восстанавливаем pipeline-дефолтный FOV (RenderPipeline
            # инициализирует camLens c set_fov(125)). Без этого после
            # STATIC-режима, который ставил 100, бортовая камера наследовала
            # его FOV и картинка выглядела неверно.
            lens = self.panda_app.cam.node().get_lens()
            if lens is not None and hasattr(lens, "set_fov"):
                lens.set_fov(self._ONBOARD_FOV)
            self._sync_fov_slider(self._ONBOARD_FOV)
            # On-board roll comes from the model config (cam_rot_r, normally
            # 0). Keep the fly cam + dial in sync with it so no custom roll
            # is carried over from a previous view.
            self._apply_camera_roll(cr)
            self._sync_roll_dial(cr)
            print(f"[Camera] ONBOARD '{key}' pos=({cx},{cy},{cz}) "
                  f"hpr=({ch},{cp},{cr}) fov={self._ONBOARD_FOV}")
        except Exception as exc:
            print(f"[Camera] onboard preset failed: {exc}")
        self._apply_depth_preset(self._ONBOARD_DEPTH)

    # ==================================================================
    # Custom camera presets (3 user slots: position + FOV)
    # ==================================================================
    def _load_cam_presets(self) -> list:
        """Read the 3 user presets from disk. Always returns a 3-element
        list of (dict | None)."""
        presets: list = [None, None, None]
        try:
            if os.path.exists(CAMERA_PRESETS_PATH):
                with open(CAMERA_PRESETS_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                items = data.get("presets") if isinstance(data, dict) else data
                if isinstance(items, list):
                    for i in range(min(3, len(items))):
                        item = items[i]
                        if isinstance(item, dict) and "pos" in item:
                            presets[i] = item
        except Exception as exc:
            print(f"[Preset] load failed: {exc}")
        return presets

    def _save_cam_presets(self) -> None:
        try:
            os.makedirs(os.path.dirname(CAMERA_PRESETS_PATH), exist_ok=True)
            with open(CAMERA_PRESETS_PATH, "w", encoding="utf-8") as f:
                json.dump({"presets": self._cam_presets}, f,
                          ensure_ascii=False, indent=2)
        except Exception as exc:
            print(f"[Preset] save failed: {exc}")

    def _apply_preset_styles(self) -> None:
        """Состояние слотов видов на панели: пустой / сохранён / выбран /
        мигает (ждёт выбора слота для сохранения)."""
        try:
            armed = getattr(self, "_preset_save_armed", False)
            blink_on = getattr(self, "_preset_blink_on", False)
            for slot, btn in getattr(self, "_preset_btns", {}).items():
                if armed:
                    state = "blink_on" if blink_on else "blink_off"
                elif self._selected_preset == slot:
                    state = "selected"
                elif self._cam_presets[slot] is not None:
                    state = "filled"
                else:
                    state = "empty"
                btn.set_state(state)
        except Exception:
            pass

    def _on_preset_blink_tick(self) -> None:
        self._preset_blink_on = not getattr(self, "_preset_blink_on", False)
        self._apply_preset_styles()

    def _set_preset_save_armed(self, armed: bool) -> None:
        """Arm/disarm save mode: while armed all 3 slots blink, inviting
        the user to pick a slot to write the current camera into."""
        self._preset_save_armed = bool(armed)
        sb = getattr(self, "_btn_preset_save", None)
        if sb is not None and sb.isChecked() != self._preset_save_armed:
            blocked = sb.blockSignals(True)
            sb.setChecked(self._preset_save_armed)
            sb.blockSignals(blocked)
        timer = getattr(self, "_preset_blink_timer", None)
        if self._preset_save_armed:
            self._preset_blink_on = True
            if timer is not None:
                timer.start(450)
        else:
            if timer is not None:
                timer.stop()
            self._preset_blink_on = False
        self._apply_preset_styles()

    def _capture_camera_state(self) -> dict | None:
        if self.panda_app is None:
            return None
        cam = getattr(self.panda_app, "camera", None)
        if cam is None:
            return None
        try:
            pos = cam.get_pos()
            hpr = cam.get_hpr()
            fov = None
            camnode = getattr(self.panda_app, "cam", None)
            if camnode is not None:
                lens = camnode.node().get_lens()
                if lens is not None and hasattr(lens, "get_fov"):
                    fov = float(lens.get_fov().x)
            # Remember the active truck model so recall restores it too.
            model_key = None
            rp = getattr(self, "right_panel", None)
            if rp is not None and hasattr(rp, "current_model_key"):
                try:
                    model_key = rp.current_model_key()
                except Exception:
                    model_key = None
            return {
                "pos": [float(pos.x), float(pos.y), float(pos.z)],
                "hpr": [float(hpr[0]), float(hpr[1]), float(hpr[2])],
                "fov": fov,
                "model": model_key,
            }
        except Exception as exc:
            print(f"[Preset] capture failed: {exc}")
            return None

    # Opacity of the original frame shown while binding a preset's anchor
    # points (lets the user see the reference photo under their clicks).
    _PRESET_PICK_OPACITY = 0.30

    def _begin_preset_capture(self, slot: int) -> None:
        """Saving a preset is a two-step flow: snapshot the camera pose, then
        let the user click any number of anchor points on the truck (with the
        reference frame overlaid at 30 %). The pose + picked film coords are
        persisted together when picking finishes."""
        if not (0 <= slot < 3):
            return
        state = self._capture_camera_state()
        if state is None:
            print("[Preset] nothing to save (no camera).")
            return
        self._pending_preset_slot = slot
        self._pending_preset_state = state

        dr = getattr(self, "depth_reconstructor", None)
        if dr is None or getattr(dr, "is_picking", lambda: False)():
            # No reconstructor (or already picking) — save pose only.
            self._commit_preset_points([])
            return

        # Point the reconstructor at the active stand snapshot (for the truck
        # collider + the overlay frame), then arm the 30 %-opacity overlay.
        rec = getattr(self, "_active_stand_rec", None)
        if rec is not None:
            try:
                dr.set_source(
                    (getattr(rec, "depth_path", "") or "").strip(),
                    (getattr(rec, "color_path", "") or "").strip(),
                )
            except Exception:
                pass
        self._begin_preset_overlay(rec)

        dr.start_picking(commit_cb=self._commit_preset_points)
        if not dr.is_picking():
            # Couldn't start (e.g. truck model missing) — save pose only.
            self._restore_preset_overlay()
            self._commit_preset_points([])

    def _begin_preset_overlay(self, rec) -> None:
        """Show the reference frame at 30 % over the viewport so the user can
        see where to click anchor points. Remembers the prior overlay state so
        it can be restored afterwards."""
        ov = getattr(self, "reference_overlay", None)
        if ov is None:
            return
        path = ""
        if rec is not None:
            path = (getattr(rec, "color_path", "") or "").strip() \
                or (getattr(rec, "path", "") or "").strip()
        try:
            self._preset_overlay_prev_visible = bool(ov.isVisible())
            self._preset_overlay_prev_opacity = float(ov.windowOpacity())
        except Exception:
            self._preset_overlay_prev_visible = False
            self._preset_overlay_prev_opacity = 0.5
        if path:
            ov.set_image(path)
        ov.set_opacity(self._PRESET_PICK_OPACITY)
        ov.show_overlay()
        self._raise_huds_above_reference()

    def _restore_preset_overlay(self) -> None:
        """Undo _begin_preset_overlay: restore opacity from the panel slider
        and hide the overlay unless a stand row is still the active selection."""
        ov = getattr(self, "reference_overlay", None)
        if ov is None:
            return
        rp = getattr(self, "right_panel", None)
        try:
            if rp is not None and hasattr(rp, "ref_opacity_slider"):
                ov.set_opacity(rp.ref_opacity_slider.value() / 100.0)
            else:
                ov.set_opacity(getattr(self, "_preset_overlay_prev_opacity", 0.5))
        except Exception:
            pass
        if not getattr(self, "_preset_overlay_prev_visible", False):
            ov.hide_overlay()

    def _commit_preset_points(self, films) -> None:
        """Finish a preset capture: store the pose + picked film coords into
        the slot and persist them to disk."""
        slot = getattr(self, "_pending_preset_slot", None)
        state = getattr(self, "_pending_preset_state", None)
        self._pending_preset_slot = None
        self._pending_preset_state = None
        self._restore_preset_overlay()
        if slot is None or state is None or not (0 <= slot < 3):
            return
        pts = []
        for f in films or []:
            try:
                pts.append([float(f[0]), float(f[1])])
            except (TypeError, ValueError, IndexError):
                continue
        state = dict(state)
        state["points"] = pts
        self._cam_presets[slot] = state
        self._selected_preset = slot          # saving selects the slot
        self._save_cam_presets()
        self._apply_preset_styles()
        print(f"[Preset] saved slot {slot + 1}: поза + {len(pts)} опорных точек")

    # Tolerances for deciding the live camera is "at" a saved preset, so its
    # bound anchor points can drive an automatic reconstruction.
    _PRESET_MATCH_POS_TOL = 0.08      # world units
    _PRESET_MATCH_ANG_TOL = 1.0       # degrees (per H/P/R axis)

    @staticmethod
    def _angle_close(a: float, b: float, tol: float) -> bool:
        d = (float(a) - float(b) + 180.0) % 360.0 - 180.0
        return abs(d) <= tol

    def _camera_matches_preset(self, state: dict, preset: dict) -> bool:
        """True if the current camera pose (position + rotation) matches a
        saved preset within tolerance."""
        try:
            sp, pp = state.get("pos"), preset.get("pos")
            sh, ph = state.get("hpr"), preset.get("hpr")
            if not (sp and pp and sh and ph):
                return False
            for a, b in zip(sp, pp):
                if abs(float(a) - float(b)) > self._PRESET_MATCH_POS_TOL:
                    return False
            for a, b in zip(sh, ph):
                if not self._angle_close(a, b, self._PRESET_MATCH_ANG_TOL):
                    return False
        except (TypeError, ValueError):
            return False
        return True

    def _matching_preset_points(self) -> list | None:
        """If the live camera is at a saved preset that carries enough anchor
        points, return those points (list of [fx, fy]); else None."""
        dr = getattr(self, "depth_reconstructor", None)
        min_pts = getattr(dr, "MIN_POINTS", 2) if dr is not None else 2
        state = self._capture_camera_state()
        if state is None:
            return None
        for preset in getattr(self, "_cam_presets", []) or []:
            if not isinstance(preset, dict):
                continue
            pts = preset.get("points") or []
            if len(pts) < min_pts:
                continue
            if self._camera_matches_preset(state, preset):
                return pts
        return None

    def _clear_preset(self, slot: int) -> None:
        if not (0 <= slot < 3):
            return
        self._cam_presets[slot] = None
        if self._selected_preset == slot:
            self._selected_preset = None
        self._save_cam_presets()
        self._apply_preset_styles()
        print(f"[Preset] cleared slot {slot + 1}")

    def _recall_preset(self, slot: int) -> None:
        """Apply a saved preset's position + FOV and drop into free-fly so
        the user can immediately look around from the saved vantage point."""
        if self.panda_app is None or not (0 <= slot < 3):
            return
        preset = self._cam_presets[slot]
        if not preset:
            return
        cam = getattr(self.panda_app, "camera", None)
        if cam is None:
            return
        # Unfreeze the fly camera (STATIC/BOARD pin it) and reflect FREE in
        # the mode segment, so the recalled pose is the new free-cam origin.
        try:
            if getattr(self, "_camera_mode", None) != "free":
                self._on_camera_mode("free")
        except Exception as exc:
            print(f"[Preset] switch to free failed: {exc}")
        # Restore the truck model bound to the preset (if any), loading it only
        # when it differs from the one already on the scene.
        self._apply_preset_model(preset.get("model"))
        try:
            px, py, pz = preset.get("pos", [0.0, 0.0, 0.0])
            h, p, r = preset.get("hpr", [0.0, 0.0, 0.0])
            cam.set_pos(float(px), float(py), float(pz))
            cam.set_hpr(float(h), float(p), float(r))
            # Restore roll through the fly cam (so mouse-look keeps it) and
            # reflect it on the dial. Must run AFTER the free-mode switch
            # above, which resets roll to 0.
            self._apply_camera_roll(float(r))
            self._sync_roll_dial(float(r))
            fov = preset.get("fov")
            if fov is not None:
                lens = self.panda_app.cam.node().get_lens()
                if lens is not None and hasattr(lens, "set_fov"):
                    lens.set_fov(float(fov))
                self._sync_fov_slider(float(fov))
                self._mirror_depth_camera_fov()
            self._selected_preset = slot      # loading selects the slot
            self._apply_preset_styles()
            print(f"[Preset] recalled slot {slot + 1}")
        except Exception as exc:
            print(f"[Preset] recall failed: {exc}")

    def _apply_preset_model(self, model_key) -> None:
        """Select + load the truck model bound to a preset. No-op when the key
        is empty or that model set is already the active one."""
        if not model_key:
            return
        model_key = str(model_key)
        rp = getattr(self, "right_panel", None)
        already = False
        if rp is not None and hasattr(rp, "current_model_key"):
            try:
                already = (rp.current_model_key() == model_key)
            except Exception:
                already = False
        # Reflect the choice in the combo (blocks its signal — no double load).
        if rp is not None and hasattr(rp, "set_current_model_key"):
            try:
                rp.set_current_model_key(model_key)
            except Exception as exc:
                print(f"[Preset] set_current_model_key failed: {exc}")
        if already:
            return
        try:
            self._on_model_set_changed(model_key)
            print(f"[Preset] model loaded: {model_key}")
        except Exception as exc:
            print(f"[Preset] model load failed: {exc}")

    def _on_preset_save_armed(self, checked: bool) -> None:
        self._set_preset_save_armed(bool(checked))

    def _on_preset_clicked(self, slot: int) -> None:
        # Save mode → write the current camera into this slot, select it,
        # and leave save mode (stops the blinking).
        if getattr(self, "_preset_save_armed", False):
            self._set_preset_save_armed(False)
            self._begin_preset_capture(slot)  # captures pose, then anchor points
            return
        # Normal mode → load the preset if the slot holds one. Empty slots
        # do nothing (no accidental auto-save).
        if self._cam_presets[slot] is not None:
            self._recall_preset(slot)

    def _on_preset_context_menu(self, slot: int) -> None:
        from PyQt6.QtWidgets import QMenu
        btn = self._preset_btns.get(slot)
        if btn is None:
            return
        menu = QMenu(btn)
        act_save = menu.addAction(
            f"Сохранить камеру + опорные точки в слот {slot + 1}")
        act_clear = menu.addAction("Очистить слот")
        act_clear.setEnabled(self._cam_presets[slot] is not None)
        chosen = menu.exec(btn.mapToGlobal(btn.rect().bottomLeft()))
        if chosen is act_save:
            self._begin_preset_capture(slot)
        elif chosen is act_clear:
            self._clear_preset(slot)

    # ==================================================================
    # FOV slider + camera-alignment reference overlay
    # ==================================================================
    def _sync_fov_slider(self, fov: float) -> None:
        """Reflect a programmatically-applied FOV on the panel slider
        (without re-triggering _on_fov_changed)."""
        rp = getattr(self, "right_panel", None)
        if rp is not None and hasattr(rp, "set_fov_value"):
            try:
                rp.set_fov_value(float(fov))
            except Exception:
                pass

    def _sync_roll_dial(self, roll: float) -> None:
        """Reflect a programmatically-applied roll on the panel dial
        (without re-triggering _on_roll_changed)."""
        rp = getattr(self, "right_panel", None)
        if rp is not None and hasattr(rp, "set_roll_value"):
            try:
                rp.set_roll_value(float(roll))
            except Exception:
                pass

    def _apply_camera_roll(self, roll: float) -> None:
        """Set the camera roll. Routes through the fly cam (so mouse-look
        keeps the roll) when present; otherwise sets the node directly."""
        if self.panda_app is None:
            return
        fc = getattr(self.panda_app, "fly_cam", None)
        if fc is not None and hasattr(fc, "set_roll"):
            fc.set_roll(float(roll))
            return
        cam = getattr(self.panda_app, "camera", None)
        if cam is not None:
            try:
                cam.set_r(float(roll))
            except Exception as exc:
                print(f"[Camera] roll set failed: {exc}")

    def _on_roll_changed(self, roll: float) -> None:
        """Drive the live camera roll from the right-panel dial."""
        self._apply_camera_roll(roll)

    def _mirror_depth_camera_fov(self, dr=None) -> None:
        """Mirror the main camera lens FOV onto the depth-preview camera's
        own lens (only writes when it actually changed)."""
        if self.panda_app is None:
            return
        if dr is None:
            dr = getattr(self.panda_app, "depth_renderer", None)
        if dr is None:
            return
        cam_np = getattr(dr, "depth_camera_np", None)
        cam = getattr(self.panda_app, "cam", None)
        if cam_np is None or cam is None:
            return
        try:
            main_lens = cam.node().get_lens()
            dlens = cam_np.node().get_lens()
            if main_lens is None or dlens is None:
                return
            mf = main_lens.get_fov()
            df = dlens.get_fov()
            if abs(mf.x - df.x) > 1e-3 or abs(mf.y - df.y) > 1e-3:
                dlens.set_fov(mf)
        except Exception as exc:
            print(f"[Depth] FOV mirror failed: {exc}")

    def _on_fov_changed(self, fov: float) -> None:
        """Drive the live camera lens FOV from the right-panel slider."""
        if self.panda_app is None:
            return
        try:
            lens = self.panda_app.cam.node().get_lens()
            if lens is not None and hasattr(lens, "set_fov"):
                lens.set_fov(float(fov))
        except Exception as exc:
            print(f"[Camera] FOV set failed: {exc}")
        # Reflect the change on the depth-preview camera immediately
        # (the periodic depth tick also mirrors it as a safety net).
        self._mirror_depth_camera_fov()

    def _on_stand_reference_selected(self, rec) -> None:
        """Selecting a stand snapshot. Jumps the camera to the FIRST saved
        preset (so its bound anchor points line up with the live view) and
        feeds the snapshot's depth/colour to the reconstructor. It does NOT
        reconstruct — that happens only when the user presses the
        "Реконструировать" button (_on_reconstruction_run).

        `rec` is a stand Reconstruction, or None."""
        ov = getattr(self, "reference_overlay", None)
        if ov is None:
            return
        # Switching snapshots invalidates any in-progress point picking.
        self._active_stand_rec = rec
        self._stop_point_picking()
        if rec is None:
            ov.hide_overlay()
            return
        # Серверные depth-записи — резолвим имена файлов в локальные пути.
        if getattr(rec, "data_type", "") == "depth":
            self._materialize_depth_record_paths(rec)
        # Prefer the explicit colour-frame path; fall back to .path.
        path = (getattr(rec, "color_path", "") or "").strip() \
            or (getattr(rec, "path", "") or "").strip()
        if not path:
            ov.hide_overlay()
            return
        # Feed the reconstructor the snapshot's depth + colour paths.
        dr = getattr(self, "depth_reconstructor", None)
        if dr is not None:
            meta = rec.raw if getattr(rec, "data_type", "") == "depth" else None
            dr.set_source(
                (getattr(rec, "depth_path", "") or "").strip(),
                (getattr(rec, "color_path", "") or "").strip(),
                meta=meta,
            )
        ov.set_image(path)
        # Manual alignment needs a movable camera — drop into free-fly so
        # WASD / RMB-look work (STATIC / BOARD freeze the camera). Это часть
        # «как было раньше» — overlay показывается ВСЕГДА, без зависимости от
        # пресетов: пользователь видит снимок поверх рендера и подгоняет
        # камеру руками (или потом сохраняет пресет).
        if getattr(self, "_camera_mode", None) != "free":
            try:
                self._on_camera_mode("free")
            except Exception as exc:
                print(f"[Camera] auto free-mode for alignment failed: {exc}")
        # Для серверной depth-записи сразу переводим камеру в первый
        # пресет: пользователь жмёт «Реконструировать», и всё работает
        # без ручного выравнивания.
        if getattr(rec, "data_type", "") == "depth":
            presets = getattr(self, "_cam_presets", None) or []
            if presets and isinstance(presets[0], dict):
                try:
                    self._recall_preset(0)
                except Exception as exc:
                    print(f"[Preset] авто-применение первого пресета упало: {exc}")
        rp = getattr(self, "right_panel", None)
        try:
            if rp is not None and hasattr(rp, "ref_opacity_slider"):
                ov.set_opacity(rp.ref_opacity_slider.value() / 100.0)
            if rp is not None and hasattr(rp, "btn_ref_toggle"):
                blocked = rp.btn_ref_toggle.blockSignals(True)
                rp.btn_ref_toggle.setChecked(True)
                rp.btn_ref_toggle.setText("Скрыть снимок")
                rp.btn_ref_toggle.blockSignals(blocked)
        except Exception:
            pass
        ov.show_overlay()
        self._raise_huds_above_reference()

    def _on_reference_opacity_changed(self, value: float) -> None:
        ov = getattr(self, "reference_overlay", None)
        if ov is not None:
            ov.set_opacity(float(value))

    def _on_reference_visible_toggled(self, visible: bool) -> None:
        ov = getattr(self, "reference_overlay", None)
        if ov is None:
            return
        if visible:
            ov.show_overlay()
            self._raise_huds_above_reference()
        else:
            ov.hide_overlay()

    def _raise_huds_above_reference(self) -> None:
        """Keep the interactive panel + read-only HUDs above the
        click-through reference layer (the telemetry card shows camera
        pos/rot/FOV the user reads while aligning)."""
        for name in ("telemetry", "toolbar", "depth_overlay", "right_panel"):
            w = getattr(self, name, None)
            if w is not None:
                try:
                    w.raise_()
                except Exception:
                    pass

    # ==================================================================
    # Depth-fill reconstruction (4-point picking)
    # ==================================================================
    def _stop_point_picking(self) -> None:
        """Cancel any in-progress picking and reset the toggle/label."""
        dr = getattr(self, "depth_reconstructor", None)
        if dr is not None:
            try:
                dr.stop_picking()
                dr.clear_points()
            except Exception:
                pass
        rp = getattr(self, "right_panel", None)
        if rp is not None:
            try:
                rp.set_picking_active(False)
                rp.set_point_count(0)
            except Exception:
                pass

    def _on_point_picking_toggled(self, active: bool) -> None:
        dr = getattr(self, "depth_reconstructor", None)
        if dr is None:
            return
        rec = getattr(self, "_active_stand_rec", None)
        if active and rec is None:
            # No stand snapshot selected — nothing to pick against.
            rp = getattr(self, "right_panel", None)
            if rp is not None:
                rp.set_picking_active(False)
            print("[DepthRecon] выберите снимок стенда перед выбором точек.")
            return
        if active:
            dr.start_picking()
            # If picking couldn't start (e.g. no depth map), snap the toggle
            # back so the UI doesn't look armed.
            if not dr.is_picking():
                rp = getattr(self, "right_panel", None)
                if rp is not None:
                    rp.set_picking_active(False)
        else:
            dr.stop_picking()

    def _on_points_reset(self) -> None:
        dr = getattr(self, "depth_reconstructor", None)
        if dr is not None:
            try:
                dr.stop_picking()
                dr.clear_points()
                dr.clear_saved_points()   # also stop auto-reconstructing
                dr.dispose_mesh()
            except Exception as exc:
                print(f"[DepthRecon] reset failed: {exc}")
        self._last_auto_recon_depth = ""
        rp = getattr(self, "right_panel", None)
        if rp is not None:
            rp.set_picking_active(False)
            rp.set_point_count(0)

    def _on_pick_count(self, n: int) -> None:
        rp = getattr(self, "right_panel", None)
        if rp is not None:
            rp.set_point_count(int(n))

    def _materialize_depth_record_paths(self, rec) -> None:
        """Для серверной depth-записи скачивает её файлы в локальный кеш
        и подменяет на абсолютные локальные пути поля `rec.depth_path`,
        `rec.color_path`, `rec.path`. Идемпотентно: если оба пути уже
        существуют локально — ничего не делает."""
        if rec is None or getattr(rec, "data_type", "") != "depth":
            return
        depth_p = (getattr(rec, "depth_path", "") or "").strip()
        color_p = (getattr(rec, "color_path", "") or "").strip()
        if (depth_p and os.path.isabs(depth_p) and os.path.exists(depth_p)
                and color_p and os.path.isabs(color_p) and os.path.exists(color_p)):
            return
        try:
            paths = resolve_depth_record_files(rec)
        except Exception as exc:
            print(f"[Recon] resolve depth-record files failed: {exc}")
            return
        print(f"[Recon] depth-record paths resolved: "
              f"depth={paths.get('depth','')!r} "
              f"color={paths.get('color','')!r} "
              f"uploaded={paths.get('uploaded','')!r}")
        depth_local = paths.get("depth", "")
        # Overlay поверх рендера — показываем ФИНАЛЬНОЕ обработанное
        # изображение (после de-barrel и polygon-crop — `masked`). Это
        # «та же картинка, что мы используем для восстановления по
        # depth_map-е». Резервы: uploaded (исходный кадр) → depth-карта.
        color_local = (paths.get("color", "")
                       or paths.get("uploaded", "")
                       or paths.get("depth", ""))
        if depth_local:
            rec.depth_path = depth_local
        if color_local:
            rec.color_path = color_local
            rec.path = color_local  # reference-overlay показывает rec.path

    def _load_depth_anchors_world(self) -> list:
        """Загружает 16 опорных 3D-точек из presets/depth_anchors_world.json.
        Возвращает список (x, y, z); при сбое — пустой список."""
        try:
            if not os.path.exists(DEPTH_ANCHORS_PATH):
                return []
            with open(DEPTH_ANCHORS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            pts = data.get("points") or []
            cleaned = []
            for p in pts:
                if len(p) == 3:
                    cleaned.append((float(p[0]), float(p[1]), float(p[2])))
            return cleaned
        except Exception as exc:
            print(f"[DepthRecon] не удалось прочитать {DEPTH_ANCHORS_PATH}: {exc}")
            return []

    def _run_depth_reconstruction(self, rec) -> None:
        """Авто-реконструкция для серверной depth-записи: применяем первый
        camera-preset, проецируем сохранённые 3D-точки в film-координаты
        текущего вида, заполняем DepthReconstructor и запускаем
        reconstruct() — пользователь жмёт одну кнопку, и всё."""
        dr = getattr(self, "depth_reconstructor", None)
        if dr is None:
            print("[DepthRecon] реконструктор недоступен.")
            return
        if dr.is_picking():
            print("[DepthRecon] идёт выбор точек — завершите его сначала.")
            return

        depth_p = (getattr(rec, "depth_path", "") or "").strip()
        if not depth_p or not os.path.exists(depth_p):
            print("[DepthRecon] у записи нет карты глубины.")
            return

        meta = rec.raw if getattr(rec, "data_type", "") == "depth" else None
        dr.set_source(
            depth_p,
            (getattr(rec, "color_path", "") or "").strip(),
            meta=meta,
        )

        # Применяем первый camera-preset, если есть. Это гарантирует, что
        # 3D-точки проецируются на ту же камеру, для которой они снимались.
        presets = getattr(self, "_cam_presets", None) or []
        if presets and isinstance(presets[0], dict):
            try:
                self._recall_preset(0)
            except Exception as exc:
                print(f"[DepthRecon] не удалось применить пресет 0: {exc}")

        world_points = self._load_depth_anchors_world()
        if not world_points:
            print(f"[DepthRecon] нет 3D-точек в {DEPTH_ANCHORS_PATH}.")
            return

        # Проецируем каждую world-точку в film-координаты текущей камеры
        # (lens.project). Те, что не попадают в frustum — пропускаем.
        from panda3d.core import Point2, Point3
        cam_np = self.panda_app.cam
        render = self.panda_app.render
        lens = cam_np.node().get_lens()

        films: list[tuple[float, float]] = []
        hits: list[Point3] = []
        for (wx, wy, wz) in world_points:
            p_world = Point3(wx, wy, wz)
            p_cam = cam_np.getRelativePoint(render, p_world)
            film_pt = Point2()
            try:
                ok = bool(lens.project(p_cam, film_pt))
            except Exception:
                ok = False
            if not ok:
                continue
            films.append((float(film_pt.x), float(film_pt.y)))
            hits.append(p_world)

        if len(films) < dr.MIN_POINTS:
            print(f"[DepthRecon] точек спроецировано {len(films)}, "
                  f"нужно ≥ {dr.MIN_POINTS}.")
            return

        # Подставляем напрямую в внутренние буферы и запускаем reconstruct.
        dr._films = films
        dr._hits = hits
        dr._saved_films = list(films)
        dr._auto_mode = False
        try:
            dr._emit_count()
        except Exception:
            pass

        print(f"[DepthRecon] авто-реконструкция depth-записи "
              f"по {len(films)} опорным точкам.")
        try:
            dr.reconstruct()
        except Exception as exc:
            print(f"[DepthRecon] реконструкция упала: {exc}")

    def _run_stand_reconstruction(self, rec) -> None:
        """Run the depth reconstruction for a stand snapshot when the user
        presses "Реконструировать". Uses the anchor points bound to the camera
        preset the view is parked at (or a prior manual pick as a fallback)."""
        dr = getattr(self, "depth_reconstructor", None)
        if dr is None:
            print("[DepthRecon] реконструктор недоступен.")
            return
        if dr.is_picking():
            print("[DepthRecon] идёт выбор точек — завершите его сначала.")
            return
        depth_p = (getattr(rec, "depth_path", "") or "").strip()
        if not depth_p or not os.path.exists(depth_p):
            print("[DepthRecon] у снимка нет карты глубины.")
            return
        # Для серверных depth-записей meta хранится в rec.raw — пробрасываем
        # его в DepthReconstructor (там лежит {"type":"depth","model":"MAZ",...}).
        meta = rec.raw if getattr(rec, "data_type", "") == "depth" else None
        dr.set_source(
            depth_p,
            (getattr(rec, "color_path", "") or "").strip(),
            meta=meta,
        )

        preset_pts = self._matching_preset_points()
        if preset_pts is not None:
            try:
                dr.set_saved_films(preset_pts)
                print(f"[DepthRecon] восстановление по {len(preset_pts)} "
                      f"опорным точкам пресета.")
            except Exception as exc:
                print(f"[DepthRecon] set_saved_films failed: {exc}")
        elif not dr.has_manual_saved_points():
            print("[DepthRecon] нет опорных точек: камера не в пресете и нет "
                  "ручного выбора. Примените пресет с точками или выберите "
                  "точки в «Дополнительно».")
            return

        self._last_auto_recon_depth = depth_p
        try:
            dr.reconstruct_saved(depth_p)
        except Exception as exc:
            print(f"[DepthRecon] реконструкция упала: {exc}")

    def _on_auto_points_requested(self) -> None:
        """Explicit automatic anchor-point search + rebuild for the active
        snapshot (triggered by the "Авто-точки" button)."""
        dr = getattr(self, "depth_reconstructor", None)
        if dr is None:
            return
        rec = getattr(self, "_active_stand_rec", None)
        if rec is None:
            print("[DepthRecon] выберите снимок стенда перед авто-поиском точек.")
            return
        if dr.is_picking():
            return
        depth_p = (getattr(rec, "depth_path", "") or "").strip()
        if not depth_p:
            print("[DepthRecon] у снимка нет карты глубины.")
            return
        self._last_auto_recon_depth = depth_p
        try:
            dr.reconstruct_auto(depth_p)
        except Exception as exc:
            print(f"[DepthRecon] авто-поиск точек упал: {exc}")

    def _on_point_viz_toggled(self, on: bool) -> None:
        dr = getattr(self, "depth_reconstructor", None)
        if dr is not None and hasattr(dr, "set_visualize"):
            try:
                dr.set_visualize(bool(on))
            except Exception as exc:
                print(f"[DepthRecon] viz toggle failed: {exc}")

    # Panel / HUD windows hidden while picking for a clean view. The
    # reference-photo overlay is intentionally NOT in this list — it keeps
    # rendering during picking so the user can see where the bed corners are.
    _PICK_HIDE_WIDGETS = (
        "telemetry", "depth_overlay", "toolbar", "right_panel",
    )

    def _on_picking_state(self, active: bool) -> None:
        """Hide the panels/HUDs while picking bed corners (keeping the
        reference photo), and restore them afterwards."""
        if active:
            self._hide_ui_for_picking()
        else:
            self._show_ui_after_picking()

    def _hide_ui_for_picking(self) -> None:
        self._ui_prev_visible = {}
        for name in self._PICK_HIDE_WIDGETS:
            w = getattr(self, name, None)
            if w is None:
                continue
            try:
                self._ui_prev_visible[name] = bool(w.isVisible())
                w.hide()
            except Exception:
                pass
        # Keep the reference photo visible + on top of the viewport while the
        # panels are gone (it's click-through, so picks still reach the scene).
        ov = getattr(self, "reference_overlay", None)
        if ov is not None and ov.isVisible():
            try:
                ov.raise_()
            except Exception:
                pass

    def _show_ui_after_picking(self) -> None:
        prev = getattr(self, "_ui_prev_visible", None) or {}
        # Bring back the chrome (HUDs + panel).
        for name in self._PICK_HIDE_WIDGETS:
            w = getattr(self, name, None)
            if w is None:
                continue
            was_visible = prev.get(name, True)
            if not was_visible:
                continue
            try:
                w.show()
                w.raise_()
            except Exception:
                pass
        # Keep the interactive chrome above the click-through photo layer.
        self._raise_huds_above_reference()

    def _on_reconstruct_finished(self, success: bool, info: dict) -> None:
        rp = getattr(self, "right_panel", None)
        if rp is not None:
            rp.set_picking_active(False)
            rp.set_point_count(0)
        if success:
            print(f"[DepthRecon] готово: {info}")
            # Mark this snapshot as already built so re-selecting it doesn't
            # trigger a redundant auto-reconstruction.
            rec = getattr(self, "_active_stand_rec", None)
            if rec is not None:
                self._last_auto_recon_depth = (
                    getattr(rec, "depth_path", "") or "").strip()
        else:
            print("[DepthRecon] реконструкция не выполнена.")

    # ==================================================================
    # Съёмка датасета
    # ==================================================================
    # Раньше здесь было два почти одинаковых прогона — «обычный» датасет и
    # «случайная сегментация», — которые расходились ровно в трёх местах: как
    # выбирается объём, как двигается камера и как ставится свет. Всё
    # остальное они дублировали, и любое исправление приходилось вносить
    # дважды. Теперь прогон один, а эти три решения приходят из конфига
    # (src/ui/dataset_config.py) независимыми осями.

    # Окна времени суток для типов освещения (минуты от полуночи).
    _DATASET_DAY_WINDOW = (600, 960)                    # 10:00–16:00
    _DATASET_DUSK_WINDOWS = ((300, 375), (1170, 1275))  # утро / вечер

    def _on_dataset_settings_clicked(self) -> None:
        """Открыть диалог настроек; «Начать съёмку» запускает прогон."""
        try:
            from src.ui.dataset_dialog import DatasetSettingsDialog
        except Exception as exc:
            print(f"[Dataset] диалог настроек недоступен: {exc}")
            return

        # HUD-карточки — это отдельные окна Qt.Tool поверх главного окна, и
        # без этого они всплыли бы поверх модального диалога.
        self._hide_huds_for_modal()
        try:
            dlg = DatasetSettingsDialog(
                self._dataset_cfg,
                parent=self.window(),
                panda_app=self.panda_app,
            )
            dlg.exec()
        finally:
            self._restore_huds_after_modal()

        if dlg.action is None:
            return

        self._dataset_cfg = dlg.config
        dataset_config.save(self._dataset_cfg)
        self._apply_dataset_palette(self._dataset_cfg)
        self._refresh_dataset_summary()

        if dlg.action == "start":
            # Через таймер, чтобы диалог успел закрыться: съёмка блокирует
            # цикл событий на всё время прогона.
            QTimer.singleShot(0, self._on_save_render_clicked)

    def _hide_huds_for_modal(self) -> None:
        self._modal_hidden_huds = []
        for name in ("telemetry", "toolbar", "depth_overlay", "right_panel"):
            widget = getattr(self, name, None)
            if widget is None:
                continue
            try:
                if widget.isVisible():
                    widget.hide()
                    self._modal_hidden_huds.append(widget)
            except Exception:
                pass

    def _restore_huds_after_modal(self) -> None:
        for widget in getattr(self, "_modal_hidden_huds", []) or []:
            try:
                widget.show()
                widget.raise_()
            except Exception:
                pass
        self._modal_hidden_huds = []

    def _apply_dataset_palette(self, cfg) -> None:
        """Отдать палитру классов рендереру сегментации."""
        palette = (cfg.get("segmentation") or {}).get("palette") or {}
        if not palette:
            return
        seg = (getattr(self.panda_app, "segmentation_renderer", None)
               if self.panda_app is not None else None)
        if seg is None or not hasattr(seg, "apply_palette"):
            return
        try:
            seg.apply_palette(palette)
        except Exception as exc:
            print(f"[Dataset] палитра сегментации не применена: {exc}")

    def _refresh_dataset_summary(self) -> None:
        """Сводка на вкладке «Датасет»: сколько кадров и что сохранится."""
        rp = getattr(self, "right_panel", None)
        if rp is None or not hasattr(rp, "set_dataset_summary"):
            return
        try:
            cfg = dataset_config.normalize(self._dataset_cfg)
            rp.set_dataset_summary(
                count=cfg["count"],
                per_fill=dataset_config.frames_per_fill(cfg),
                total=dataset_config.total_frames(cfg),
                outputs=list(dataset_config.output_list(cfg)),
                out_dir=cfg["output_dir"],
            )
        except Exception as exc:
            rp.set_dataset_error(f"настройки не прочитаны: {exc}")

    def _on_save_render_clicked(self) -> None:
        """Запустить съёмку датасета с текущим конфигом."""
        if self.panda_app is None:
            return

        # Съёмка САМА крутит кадры, поэтому запускаться изнутри кадра нельзя:
        # насос подавил бы вложенные шаги (иначе — рекурсивный poll), кадры бы
        # не продвигались, и ожидание сцены зависло бы. Слот может приехать
        # именно так — через QApplication.processEvents(), вызванный из-под
        # идущего кадра. Если это наш случай — откладываем старт до момента,
        # когда кадр завершится (singleShot(0) = следующая итерация цикла
        # событий, уже вне кадра).
        pump = getattr(self.panda_app, "frame_pump", None)
        if pump is not None and pump.busy:
            print("[Dataset] запуск изнутри кадра — откладываю на "
                  "следующий цикл событий.")
            QTimer.singleShot(0, self._on_save_render_clicked)
            return

        ru = getattr(self.panda_app, "renderer_utils", None)
        if ru is None or not hasattr(ru, "save_single_render"):
            print("[Dataset] renderer_utils.save_single_render missing.")
            return

        self._run_dataset(dataset_config.normalize(self._dataset_cfg), ru)

    # ------------------------------------------------------------------
    # План кадров: поза камеры и объём наполнения
    # ------------------------------------------------------------------
    def _dataset_camera_plan(self, cfg, lights) -> list:
        """Список кадров, снимаемых с ОДНОГО наполнения.

        Каждый элемент — dict с именем варианта и отклонениями от базовой
        позы (dh/dp — рысканье/тангаж в градусах, lat/vert — смещения в
        метрах). Ключ "light" задаёт освещение принудительно; без него тип
        света назначается по кругу уже на этапе съёмки.
        """
        cam = cfg["camera"]
        mode = cam["mode"]
        ang = float(cam["angle_deg"])
        off = float(cam["offset_m"])

        if mode == "fixed":
            return [{"name": "base"}]

        if mode == "random":
            # Конкретные значения берутся в момент съёмки — иначе все
            # наполнения получили бы один и тот же набор «случайных» поз.
            return [{"name": "random", "randomize": True}
                    for _ in range(int(cam["samples"]))]

        variants = cam["variants"]
        plan: list = []
        if variants.get("originals"):
            # Базовая поза — по кадру на каждый тип света, чтобы «эталон»
            # был представлен в каждом освещении.
            for light in lights:
                plan.append({"name": f"{light}_orig", "light": light})
        if variants.get("angles"):
            plan += [
                {"name": "h_plus",  "dh": +ang},
                {"name": "h_minus", "dh": -ang},
                {"name": "p_plus",  "dp": +ang},
                {"name": "p_minus", "dp": -ang},
            ]
        if variants.get("offsets"):
            plan += [
                {"name": "lat_plus",   "lat":  +off},
                {"name": "lat_minus",  "lat":  -off},
                {"name": "vert_plus",  "vert": +off},
                {"name": "vert_minus", "vert": -off},
            ]
        if variants.get("random_combined"):
            plan.append({"name": "random_combined", "randomize": True})
        return plan or [{"name": "base"}]

    def _dataset_volume_for(self, cfg, index, count, max_volume):
        """Объём наполнения для итерации: (target, класс кадра)."""
        vol = cfg["volume"]
        if max_volume is None:
            rp = getattr(self, "right_panel", None)
            target = (float(rp.current_target_volume())
                      if rp is not None else 0.0)
            return target, ("empty" if target <= 0.0 else "random")

        if vol["mode"] == "ramp":
            # Объём равномерно растёт: шаг = max_volume / count.
            return (float(max_volume) / count) * (index + 1), "ramp"

        ceiling = float(vol["ceiling_k"]) * float(max_volume)
        roll = random.uniform(0.0, 100.0)
        if roll < vol["full_pct"]:
            return random.uniform(0.95, 1.0) * ceiling, "full"
        if roll < vol["full_pct"] + vol["empty_pct"]:
            return 0.0, "empty"
        # Нижняя граница строго > 0: ровно 0 доезжает до сервера как «объём
        # не задан», и вместо пустого кузова получается случайный НЕпустой с
        # ярлыком vol0000.00. Пустой кузов делается отдельной веткой.
        MIN_FRACTION = 0.02
        return random.uniform(MIN_FRACTION * ceiling, ceiling), "random"

    def _dataset_daytime_for(self, light_mode) -> int:
        if light_mode == "dusk":
            lo, hi = random.choice(self._DATASET_DUSK_WINDOWS)
            return random.randint(lo, hi)
        return random.randint(*self._DATASET_DAY_WINDOW)

    # ------------------------------------------------------------------
    # Сам прогон
    # ------------------------------------------------------------------
    def _run_dataset(self, cfg, ru) -> None:
        from PyQt6.QtWidgets import QApplication

        count = int(cfg["count"])
        outputs = set(dataset_config.output_list(cfg))
        depth_settings = dict(cfg["depth"]) if "depth" in outputs else None
        lidar_settings = dict(cfg["lidar"]) if "lidar" in outputs else None
        scene = cfg["scene"]
        lights = dataset_config.enabled_lights(cfg)
        plan = self._dataset_camera_plan(cfg, lights)
        cam_cfg = cfg["camera"]
        ang = float(cam_cfg["angle_deg"])
        off = float(cam_cfg["offset_m"])

        out_dir = cfg["output_dir"]
        if not os.path.isabs(out_dir):
            out_dir = os.path.join(PROJECT_ROOT, out_dir)

        # Палитра классов могла измениться в диалоге или приехать с диска при
        # старте — применяем её прямо перед съёмкой.
        self._apply_dataset_palette(cfg)

        # Текущая модель + текстура из правой панели; max_volume — из YAML.
        rp = getattr(self, "right_panel", None)
        model_key = rp.current_model_key() if rp is not None else None
        texture_key = rp.current_texture_key() if rp is not None else None
        max_volume = None
        if model_key:
            mc = get_model_set_config(str(model_key))
            if mc and mc.get("max_volume") is not None:
                try:
                    max_volume = float(mc["max_volume"])
                except (TypeError, ValueError):
                    max_volume = None
        # Локальные наборы в YAML не хранят max_volume — берём эффективный
        # (унаследованный от донора), который load_model_set положил в
        # panda_app.current_max_volume. Без этого объём наполнения
        # фиксировался бы значением из панели.
        if not max_volume:
            try:
                max_volume = float(
                    getattr(self.panda_app, "current_max_volume", 0) or 0)
            except (TypeError, ValueError):
                max_volume = None
        if not max_volume:
            print("[Dataset] max_volume недоступен для текущего набора — "
                  "объём берётся из панели.")
            max_volume = None

        total = count * len(plan)
        print(f"[Dataset] старт: {count} наполнений x {len(plan)} кадров = "
              f"{total}; выходы={sorted(outputs)}; каталог={out_dir}")

        self.btn_save_render.setEnabled(False)
        btn_setup = getattr(self, "btn_dataset_setup", None)
        if btn_setup is not None:
            btn_setup.setEnabled(False)
        original_text = self.btn_save_render.text()
        ok_count = 0

        # Замораживаем fly-cam на весь прогон, чтобы наши setPos / setHpr на
        # каждом варианте не сбивались его тиком.
        fly_cam = getattr(self.panda_app, "fly_cam", None)
        prev_frozen = None
        if fly_cam is not None and hasattr(fly_cam, "set_frozen"):
            try:
                prev_frozen = (fly_cam.is_frozen()
                               if hasattr(fly_cam, "is_frozen") else None)
                fly_cam.set_frozen(True)
            except Exception:
                prev_frozen = None

        base_daytime_mins = 6 * 60 + 40
        if hasattr(self, "daytime_slider"):
            try:
                base_daytime_mins = int(self.daytime_slider.value())
            except Exception:
                pass

        def _set_daytime(mins) -> None:
            mins = int(mins) % 1440
            try:
                pipeline = getattr(self.panda_app, "render_pipeline", None)
                mgr = (getattr(pipeline, "daytime_mgr", None)
                       if pipeline else None)
                if mgr is not None:
                    mgr.time = f"{mins // 60:02d}:{mins % 60:02d}"
            except Exception as exc:
                print(f"[Dataset] время суток не выставлено: {exc}")

        def _set_sun_overhead(enable=True) -> None:
            """Солнце жёстко в зенит — не временем суток, а направлением."""
            app = self.panda_app
            if app is not None and hasattr(app, "set_sun_overhead"):
                try:
                    app.set_sun_overhead(enable)
                except Exception as exc:
                    print(f"[Dataset] солнце в зените не выставлено: {exc}")

        # На время съёмки ОСТАНАВЛИВАЕМ Qt-таймеры, которые сами крутят
        # taskMgr.step()/рендер. Иначе наши ручные settle-шаги пересекаются с
        # тиком _panda_timer (он доезжает через QApplication.processEvents),
        # Panda ругается «Ignoring recursive poll()», часть шагов ИГНОРИРУЕТСЯ,
        # счётчик кадров сбивается — и цветной кадр рассинхронизируется с
        # маской. Таймеры возвращаем в finally.
        paused_timers = []
        for name in ("_panda_timer", "_depth_timer", "_telemetry_timer"):
            timer = getattr(self, name, None)
            try:
                if timer is not None and timer.isActive():
                    timer.stop()
                    paused_timers.append(timer)
            except Exception:
                pass

        # Ожидание сходимости сцены: ждём по ОБОИМ условиям — не меньше
        # WAIT_FRAMES реально выполненных кадров И не меньше WAIT_SECONDS
        # секунд. Свежий меш наполнения и его 8K-текстуры доезжают до GPU
        # лениво, а голый sleep кадры НЕ гонит.
        WAIT_FRAMES = 60
        WAIT_SECONDS = 1.0

        def _settle_wait(frames=WAIT_FRAMES, seconds=WAIT_SECONDS):
            """Прокрутить кадры до сходимости сцены.

            Кадры считаем по фактически ВЫПОЛНЕННЫМ (pump.step возвращает их
            число): часть шагов Panda молча игнорирует как рекурсивные.

            КРИТИЧНО: у цикла ОБЯЗАН быть выход, не зависящий от числа
            выполненных кадров. Если мы сами оказались внутри кадра (насос
            подавляет повторный вход и возвращает 0), кадры отсюда не
            продвинутся НИКОГДА — и цикл висел бы вечно. Поэтому есть и
            детектор простоя, и жёсткий дедлайн: лучше выйти рано и дать
            захвату честно отказаться, чем повесить приложение.
            """
            pump = getattr(self.panda_app, "frame_pump", None)
            start = time.perf_counter()
            deadline = start + max(seconds * 4.0, seconds + 15.0)
            done_frames = 0
            stalled = 0
            while True:
                if pump is not None:
                    done = pump.step(1)
                    done_frames += done
                    stalled = 0 if done else stalled + 1
                else:
                    self.panda_app.taskMgr.step()
                    done_frames += 1
                QApplication.processEvents()

                now = time.perf_counter()
                if done_frames >= frames and (now - start) >= seconds:
                    break
                if stalled >= 50:
                    print("[Dataset] кадры не продвигаются (вызов изнутри "
                          "кадра); выхожу из ожидания, чтобы не зависнуть.")
                    break
                if now >= deadline:
                    print(f"[Dataset] дедлайн ожидания: {done_frames} из "
                          f"{frames} кадров за {now - start:.1f} c — иду "
                          f"дальше.")
                    break
                time.sleep(0.005)

        frame_no = 0
        try:
            for i in range(count):
                target, fill_class = self._dataset_volume_for(
                    cfg, i, count, max_volume)
                is_empty = (fill_class == "empty")

                self.btn_save_render.setText(f"{i + 1}/{count}")
                QApplication.processEvents()

                # Новое наполнение для этой итерации (для пустого кузова —
                # снятие старого меша без обращения к серверу).
                try:
                    self._on_run_simulation({
                        "model_key":     model_key,
                        "texture_key":   texture_key,
                        "target_volume": float(target),
                        "empty":         is_empty,
                    })
                except Exception as exc:
                    print(f"[Dataset] пайплайн {i + 1} упал: {exc}")
                    QApplication.processEvents()
                    continue

                _settle_wait()

                cam = self.panda_app.camera
                base_pos = cam.getPos()
                base_hpr = cam.getHpr()
                base_pos_t = (float(base_pos.x), float(base_pos.y),
                              float(base_pos.z))
                base_hpr_t = (float(base_hpr.x), float(base_hpr.y),
                              float(base_hpr.z))

                for v_idx, variant in enumerate(plan):
                    # --- поза камеры ------------------------------------
                    if variant.get("randomize"):
                        dh = random.uniform(-ang, ang)
                        dp = random.uniform(-ang, ang)
                        lat = random.uniform(-off, off)
                        vert = random.uniform(-off, off)
                    else:
                        dh = float(variant.get("dh", 0.0))
                        dp = float(variant.get("dp", 0.0))
                        lat = float(variant.get("lat", 0.0))
                        vert = float(variant.get("vert", 0.0))

                    cam.setPos(*base_pos_t)
                    cam.setHpr(base_hpr_t[0] + dh,
                               base_hpr_t[1] + dp,
                               base_hpr_t[2])
                    if lat or vert:
                        # Смещения в локальном фрейме камеры: +X вправо,
                        # +Z вверх.
                        cam.setPos(cam, lat, 0.0, vert)

                    # --- освещение --------------------------------------
                    light_mode = (variant.get("light")
                                  or lights[frame_no % len(lights)])
                    shadow_band = (light_mode == "shadow")
                    applied_time = int(base_daytime_mins)
                    if light_mode == "overhead":
                        _set_sun_overhead(True)
                    elif light_mode != "current":
                        applied_time = self._dataset_daytime_for(light_mode)
                        _set_daytime(applied_time)

                    _settle_wait()

                    frame_no += 1
                    self.btn_save_render.setText(f"{frame_no}/{total}")
                    QApplication.processEvents()

                    extra_meta = {
                        "render_type": "dataset",
                        "dataset_config": {
                            "outputs": sorted(outputs),
                            "volume_mode": cfg["volume"]["mode"],
                            "camera_plan": cam_cfg["mode"],
                            "lighting_mode": cfg["lighting"]["mode"],
                        },
                        "random_background": scene["random_background"],
                        "light_mode": light_mode,
                        "shadow_band": shadow_band,
                        "sun_overhead": (light_mode == "overhead"),
                        "iteration": i,
                        "iteration_total": count,
                        "variant": variant["name"],
                        "variant_index": v_idx,
                        "variant_params": {"dh": dh, "dp": dp,
                                           "lat": lat, "vert": vert},
                        "camera_mode": getattr(self, "_camera_mode", None),
                        "base_camera_position": {
                            "x": base_pos_t[0], "y": base_pos_t[1],
                            "z": base_pos_t[2],
                        },
                        "base_camera_rotation": {
                            "h": base_hpr_t[0], "p": base_hpr_t[1],
                            "r": base_hpr_t[2],
                        },
                        "base_daytime_minutes": int(base_daytime_mins),
                        "applied_daytime_minutes": int(applied_time),
                        "target_volume": float(target),
                        "fill_class": fill_class,
                        "max_volume": (float(max_volume)
                                       if max_volume is not None else None),
                        "model_key":   model_key,
                        "texture_key": texture_key,
                    }
                    if depth_settings is not None:
                        extra_meta["depth_settings"] = dict(depth_settings)
                    if lidar_settings is not None:
                        extra_meta["lidar_settings"] = dict(lidar_settings)
                    if is_empty:
                        # В сцене нет final_model, поэтому save_single_render
                        # объём посчитать не может и записал бы null. Для
                        # пустого кузова это именно 0, а не «неизвестно».
                        extra_meta["actual_volume"] = 0.0

                    prefix = (f"i{i:04d}_vol{target:07.2f}_"
                              f"v{v_idx:02d}_{variant['name']}")
                    try:
                        ok = ru.save_single_render(
                            output_dir=out_dir,
                            filename_prefix=prefix,
                            extra_metadata=extra_meta,
                            outputs=outputs,
                            depth_settings=depth_settings,
                            lidar_settings=lidar_settings,
                            random_background=scene["random_background"],
                            gemini=False,
                            shadow_band=shadow_band,
                            cloth=scene["cloth"],
                            cloth_probability=scene["cloth_probability"],
                        )
                        if ok:
                            ok_count += 1
                        else:
                            print(f"[Dataset] {frame_no}/{total} "
                                  f"({variant['name']}) кадр не подтверждён")
                    except Exception as exc:
                        print(f"[Dataset] {frame_no}/{total} "
                              f"({variant['name']}) не сохранён: {exc}")
                    QApplication.processEvents()

                # Вернуть базовую позу, чтобы следующее наполнение стартовало
                # из того же состояния, что видит пользователь.
                cam.setPos(*base_pos_t)
                cam.setHpr(*base_hpr_t)
        finally:
            _set_sun_overhead(False)
            _set_daytime(base_daytime_mins)
            # Вернуть Qt-таймеры — .start() без аргумента переиспользует
            # прежний интервал.
            for timer in paused_timers:
                try:
                    timer.start()
                except Exception:
                    pass
            if (fly_cam is not None and hasattr(fly_cam, "set_frozen")
                    and prev_frozen is not None):
                try:
                    fly_cam.set_frozen(bool(prev_frozen))
                except Exception:
                    pass
            self.btn_save_render.setText(original_text)
            self.btn_save_render.setEnabled(True)
            if btn_setup is not None:
                btn_setup.setEnabled(True)

        print(f"[Dataset] готово: сохранено {ok_count} из {total} кадров; "
              f"каталог {out_dir}")

    def _update_telemetry(self) -> None:
        if self.panda_app is None:
            return
        try:
            cam = self.panda_app.camera
            hpr = cam.get_hpr()
            yaw, pitch, roll = float(hpr[0]), float(hpr[1]), float(hpr[2])
            lens = (
                self.panda_app.cam.node().get_lens()
                if self.panda_app.cam else None
            )
            fov = float(lens.get_fov().x) if lens is not None else 0.0
            self.telemetry.update_row("PITCH", f"{pitch:+6.1f}")
            self.telemetry.update_row("YAW",   f"{yaw:+6.1f}")
            self.telemetry.update_row("ROLL",  f"{roll:+6.1f}")
            self.telemetry.update_row("FOV",   f"{fov:6.1f}")
            # One-time: align the FOV slider with the live lens so the
            # control starts in sync with whatever the pipeline booted with.
            if not getattr(self, "_fov_slider_synced", False) and fov > 0:
                rp = getattr(self, "right_panel", None)
                if rp is not None and hasattr(rp, "set_fov_value"):
                    rp.set_fov_value(fov)
                    self._fov_slider_synced = True
            # One-time: align the roll dial with the live camera roll.
            if not getattr(self, "_roll_dial_synced", False):
                rp = getattr(self, "right_panel", None)
                if rp is not None and hasattr(rp, "set_roll_value"):
                    rp.set_roll_value(roll)
                    self._roll_dial_synced = True
            try:
                pos = cam.get_pos()
                self.telemetry.update_row("X", f"{float(pos.x):+7.1f}")
                self.telemetry.update_row("Y", f"{float(pos.y):+7.1f}")
                self.telemetry.update_row("Z", f"{float(pos.z):+7.1f}")
            except Exception:
                pass
        except Exception:
            pass

    # ==================================================================
    # Panda HWND resolution + resize sync
    # ==================================================================
    @staticmethod
    def _resolve_panda_hwnd(panda_app,
                            parent_hwnd: int | None = None) -> int | None:
        win = getattr(panda_app, "win", None)
        if win is not None:
            wh = None
            try:
                wh = win.getWindowHandle()
            except Exception:
                wh = None

            if wh is not None:
                for getter in ("getIntHandle", "get_int_handle"):
                    fn = getattr(wh, getter, None)
                    if callable(fn):
                        try:
                            v = fn()
                            if v:
                                hwnd = int(v)
                                if (parent_hwnd is None
                                        or _is_child_of(hwnd, parent_hwnd)):
                                    return hwnd
                        except Exception:
                            pass

            if wh is not None:
                os_handle = None
                for getter in ("getOSHandle", "get_os_handle"):
                    fn = getattr(wh, getter, None)
                    if callable(fn):
                        try:
                            os_handle = fn()
                        except Exception:
                            os_handle = None
                        if os_handle is not None:
                            break
                if os_handle is not None:
                    for getter in ("getHandle", "get_handle"):
                        fn = getattr(os_handle, getter, None)
                        if callable(fn):
                            try:
                                v = fn()
                                if v:
                                    hwnd = int(v)
                                    if (parent_hwnd is None
                                            or _is_child_of(hwnd, parent_hwnd)):
                                        return hwnd
                            except Exception:
                                pass

        if parent_hwnd:
            children: list[int] = []

            def _cb(child_hwnd, _):
                children.append(int(child_hwnd))
                return True

            try:
                win32gui.EnumChildWindows(parent_hwnd, _cb, None)
            except Exception:
                pass

            if children:
                def _area(h):
                    try:
                        l, t, r, b = win32gui.GetWindowRect(h)
                        return max(0, r - l) * max(0, b - t)
                    except Exception:
                        return 0
                children.sort(key=_area, reverse=True)
                return children[0]

        return None

    def _frame_interval_ms(self) -> int:
        """Период обновления экрана в целых мс (потолок частоты кадров)."""
        try:
            hz = float(self.screen().refreshRate())
        except Exception:
            hz = 0.0
        if not (30.0 <= hz <= 500.0):
            hz = 60.0
        return max(1, int(1000.0 / hz))

    def _pump_frame(self) -> None:
        """Тик Qt-таймера: один кадр Panda через защищённый насос."""
        app = getattr(self, "panda_app", None)
        if app is None:
            return
        pump = getattr(app, "frame_pump", None)
        if pump is not None:
            pump.step(1)
        else:
            app.taskMgr.step()

    def _reposition_panda(self) -> None:
        if self.panda_app is None:
            return
        dpr = self.devicePixelRatio()
        w = max(1, round(self.panda_container.width() * dpr))
        h = max(1, round(self.panda_container.height() * dpr))
        hwnd = self._panda_hwnd
        if hwnd:
            try:
                flags = (win32con.SWP_NOZORDER
                         | win32con.SWP_NOACTIVATE
                         | win32con.SWP_SHOWWINDOW)
                win32gui.SetWindowPos(hwnd, 0, 0, 0, w, h, flags)
            except Exception as e:
                print(f"[Resize] SetWindowPos failed: {e}")
        try:
            props = WindowProperties()
            props.setOrigin(0, 0)
            props.setSize(w, h)
            self.panda_app.win.requestProperties(props)
        except Exception as e:
            print(f"[Resize] requestProperties failed: {e}")
        # Keep the lens aspect in sync with the new window so the rendered
        # view stays undistorted. set_fov() pinned the HORIZONTAL FOV, so the
        # vertical FOV follows the aspect — the view scales by window WIDTH.
        # The reference-photo overlay scales to width to match this exactly.
        try:
            lens = self.panda_app.cam.node().get_lens()
            if lens is not None and hasattr(lens, "set_aspect_ratio"):
                lens.set_aspect_ratio(float(w) / float(h))
        except Exception as e:
            print(f"[Resize] lens aspect update failed: {e}")

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.panda_app is not None:
            try:
                self._reposition_panda()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Генератор кузовов (опциональный модуль src/bodygen)
    # ------------------------------------------------------------------

    def _on_worklist_requested(self) -> None:
        """
        Очередь на генерацию: что сервер снимает, а геометрии под это нет.

        Диалог только выбирает кузов и качает облако пустого скана; дальше
        начинается обычная сборка — тот же `BodyGenDialog` с подставленными
        файлом, именем и прямоугольником. Прямоугольник подставляется не для
        красоты: кузова нет в справочнике, автоподбор по облаку цепляется за
        ложный прямоугольник, и обмер с сервера — единственное, что его
        удерживает (см. подсказку в самом диалоге).
        """
        from PyQt6.QtWidgets import QMessageBox

        try:
            from src.ui.panel_data import active_tls_server
            from src.ui.worklist_dialog import BodyWorklistDialog
        except Exception as exc:                          # noqa: BLE001
            QMessageBox.critical(
                self, "Кузова к генерации",
                f"Модуль очереди не загрузился: {exc}\n\n"
                "Проверьте, что установлен пакет requests "
                "(pip install requests).")
            return

        server = active_tls_server()
        host, port = server if server else ("", 9999)
        dialog = BodyWorklistDialog(self, host=host, port=port)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        request = dialog.request()
        if request is None:
            return

        print(f"[worklist] собираю «{request.need.display_name}» по скану "
              f"{request.shot.base} ({request.need.shots} снимков ждут)")
        self._worklist_request = request
        preset = {
            "source": "ply",
            "ply_path": request.ply_path,
            # Пусто = автоподбор: имени этой модели в справочнике нет по
            # определению, иначе она не попала бы в очередь.
            "cloud_model": "",
            "rect_width": float(request.width or 0.0),
            "rect_length": float(request.length or 0.0),
            "name": request.name,
        }
        # Колёсная формула почти всегда записана в самом имени модели
        # («HOWO T5G 6x4 (стандарт)»). Подставить её надёжнее, чем оставить
        # подбор по длине: он выбирает шасси по кузову, а у «высоких» кузов
        # длинный при трёх осях. Нет формулы в имени — пусть подбирает.
        chassis = _chassis_from_name(request.need.display_name)
        if chassis:
            preset["chassis"] = chassis
        self._on_bodygen_requested(preset=preset)

    def _on_bodygen_requested(self, preset: dict | None = None) -> None:
        """Показать диалог параметров и запустить сборку в отдельном потоке."""
        try:
            from src.ui.bodygen_dialog import BodyGenDialog
        except Exception as exc:
            print(f"[BodyGen] диалог недоступен: {exc}")
            self.right_panel.set_bodygen_status(f"диалог недоступен: {exc}")
            return

        if preset is None:
            # Ручной запуск: прошлый выбор из очереди больше не при чём, и
            # предлагать загрузку «той» модели после сборки нельзя.
            self._worklist_request = None

        dlg = BodyGenDialog(self, preset=preset)
        if dlg.exec() != dlg.DialogCode.Accepted:
            self._worklist_request = None
            return

        params = dlg.params()
        self._bodygen_select = bool(dlg.show_in_list())

        # Пересборка под тем же именем: диалог уже спросил разрешения, здесь
        # только стираем прошлый комплект. Делать это до запуска потока
        # обязательно — иначе удаление догонит уже записанные новые файлы.
        if not self._bodygen_clear_previous(dlg.overwrite_paths(), params.name):
            return

        # Сборка идёт минуты и держит GIL в numpy-циклах: в главном потоке она
        # заморозила бы и окно, и рендер Panda3D. Поэтому — отдельный поток, а
        # в панель только строка состояния.
        self._bodygen_thread = _BodyGenWorker(params, self)
        self._bodygen_thread.progressed.connect(
            lambda msg: self.right_panel.set_bodygen_status(msg, busy=True))
        self._bodygen_thread.finishedWith.connect(self._on_bodygen_finished)
        self.right_panel.set_bodygen_status("запуск…", busy=True)
        self._bodygen_thread.start()

    def _bodygen_clear_previous(self, paths, name: str) -> bool:
        """
        Стереть файлы прошлого комплекта. False — сборку начинать нельзя.

        Если перезаписывается набор, который сейчас в сцене, она сначала
        освобождается: смотреть на модель, файлов которой уже нет, незачем, а
        сборка идёт минуты. Чужой набор в сцене не трогается.
        """
        paths = [str(p) for p in (paths or [])]
        if not paths:
            return True

        from src.ui.panel_data import GENERATED_MODEL_PREFIX
        current_key = ""
        try:
            current_key = str(self.right_panel.current_model_key() or "")
        except Exception:
            pass
        if current_key == f"{GENERATED_MODEL_PREFIX}{name}":
            panda = getattr(self, "panda_app", None)
            if panda is not None and hasattr(panda, "clear_scene"):
                try:
                    panda.clear_scene()
                except Exception as exc:
                    print(f"[BodyGen] сцена не очищена перед перезаписью: "
                          f"{exc}")

        failed = []
        for path in paths:
            try:
                if os.path.isdir(path) and not os.path.islink(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
            except FileNotFoundError:
                pass
            except OSError as exc:
                failed.append(f"{os.path.basename(path)}: "
                              f"{exc.strerror or exc}")

        if failed:
            from PyQt6.QtWidgets import QMessageBox
            text = ("Не удалось удалить прошлый комплект — сборка отменена, "
                    "иначе он смешался бы с новым.\n\n" + "\n".join(failed[:5]))
            print(f"[BodyGen] перезапись не удалась: {'; '.join(failed)}")
            QMessageBox.warning(self, "Перезапись комплекта", text)
            self.right_panel.set_bodygen_status("перезапись не удалась")
            self._worklist_request = None
            return False

        print(f"[BodyGen] прошлый комплект удалён: {len(paths)} объект(ов)")
        return True

    def _on_bodygen_finished(self, result) -> None:
        """Обработать результат сборки: показать итог и подхватить комплект."""
        if not getattr(result, "ok", False):
            msg = getattr(result, "error", "неизвестная ошибка")
            print(f"[BodyGen] ошибка: {msg}")
            self.right_panel.set_bodygen_status(f"ошибка: {msg}"[:180])
            return

        print(f"[BodyGen] готово за {result.seconds:.1f} с")
        print(result.summary)
        self._remember_client_model(result)
        self.right_panel.set_bodygen_status(
            f"{result.name}: готово за {result.seconds:.0f} с")

        # Пересборка под тем же именем даёт те же пути к картам, а TexturePool
        # кеширует по пути и на диск больше не смотрит. Без сброса в сцену
        # вернулись бы СТАРЫЕ текстуры: цвет краски, износ и грязь остались бы
        # от прошлой сборки. Сбрасываем до перезагрузки списка — она сразу же
        # грузит модель. Сбрасывается только папка карт ЭТОГО комплекта
        # (`<каталог>/<имя>`, см. pipeline._write_files) — соседние комплекты
        # не изменились, и перечитывать их по сотне мегабайт незачем.
        panda = getattr(self, "panda_app", None)
        if panda is not None and hasattr(panda, "forget_cached_textures"):
            try:
                panda.forget_cached_textures(
                    os.path.join(result.out_dir, result.name))
            except Exception as exc:
                print(f"[BodyGen] сброс кеша текстур не удался: {exc}")

        if not getattr(self, "_bodygen_select", True):
            return
        model_key = ""
        try:
            from src.ui.panel_data import GENERATED_MODEL_PREFIX
            model_key = f"{GENERATED_MODEL_PREFIX}{result.name}"
            self.right_panel.reload_model_sets(select_key=model_key)
        except Exception as exc:
            print(f"[BodyGen] не удалось обновить список моделей: {exc}")
            return

        self._offer_worklist_upload(result, model_key)

    def _remember_client_model(self, result) -> None:
        """
        Записать в `.set.json` комплекта имя, которым модель зовёт клиент.

        Сервер ищет геометрию по этому имени точным сравнением строк, а
        комплект на диске называется латинским ключом. Связь между ними
        известна только сейчас, пока жив запрос из очереди; загрузку же в
        реестр могут открыть и через неделю. Поэтому имя кладётся рядом с
        комплектом — диалог загрузки подставит его в поле «Название».
        """
        request = getattr(self, "_worklist_request", None)
        if request is None:
            return
        path = os.path.join(result.out_dir, f"{result.name}.set.json")
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            data["client_model"] = request.need.display_name
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
        except Exception as exc:                          # noqa: BLE001
            print(f"[worklist] имя модели не записано в {path}: {exc}")

    def _offer_worklist_upload(self, result, model_key: str) -> None:
        """
        Замкнуть цепочку очереди: собрано — предложить отправку в реестр.

        Только для сборок, начатых из очереди: там заранее известно, какой
        модели не хватало на сервере, и отправлять её туда — следующий шаг по
        смыслу. Ручная сборка этим не беспокоится.

        Предложение, а не автоматика: загруженная модель сразу идёт в работу
        и пересчитывает чужие снимки (`model_webhook_listener` на сервере
        запускает rerun_by_model.sh), так что кузов сначала надо посмотреть в
        сцене. Поэтому кнопка по умолчанию — «Позже».
        """
        request = getattr(self, "_worklist_request", None)
        self._worklist_request = None
        if request is None or not model_key:
            return

        from PyQt6.QtWidgets import QMessageBox

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle("Кузов собран")
        box.setText(
            f"«{result.name}» собран за {result.seconds:.0f} с и выбран в "
            f"сцене.\n\nЭто кузов «{request.need.display_name}» — его ждут "
            f"{request.need.shots} снимков на сервере. Отправить в реестр?")
        box.setInformativeText(
            "Сначала посмотрите модель в сцене: после загрузки сервер "
            "пересчитает по ней все снимки с этим именем.")
        send = box.addButton("Отправить в реестр…",
                             QMessageBox.ButtonRole.AcceptRole)
        later = box.addButton("Позже", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(later)
        box.exec()
        if box.clickedButton() is send:
            self._on_model_upload_requested(model_key)

    def _on_model_upload_requested(self, model_key: str) -> None:
        """
        Открыть диалог загрузки набора в реестр моделей photo-to-volume.

        Диалог живёт здесь, а не в панели, по двум причинам: ему нужен пресет
        камеры (реестр без него не примет новую модель, а взять его удобнее
        всего из текущего вида сцены), и после успешной загрузки список
        наборов стоит перечитать — серверный конфиг изменился, и набор,
        который до сих пор был «локальным», приедет уже с сервера.
        """
        from PyQt6.QtWidgets import QMessageBox

        key = str(model_key or "")
        if not key:
            return

        info = None
        try:
            info = self.right_panel.model_info(key)
        except Exception as exc:
            print(f"[Registry] характеристики набора недоступны: {exc}")
        if info is None:
            QMessageBox.warning(self, "Загрузка модели",
                                f"Набор «{key}» не найден в списке.")
            return

        try:
            from src.ui.registry_dialog import ModelUploadDialog
        except Exception as exc:                          # noqa: BLE001
            QMessageBox.critical(
                self, "Загрузка модели",
                f"Модуль реестра не загрузился: {exc}\n\n"
                "Проверьте, что установлен пакет requests "
                "(pip install requests).")
            return

        dialog = ModelUploadDialog(self, info=info,
                                   camera_provider=self._capture_camera_state)
        dialog.exec()
        if not dialog.uploaded:
            return

        # Список наборов после загрузки меняется: тот же кузов теперь есть и
        # на сервере. Выбор сохраняем — набор в сцене трогать незачем.
        current_key = ""
        try:
            current_key = str(self.right_panel.current_model_key() or "")
        except Exception:
            pass
        try:
            self.right_panel.reload_model_sets(select_key=current_key or None)
            self.right_panel.set_model_status("модель отправлена в реестр")
        except Exception as exc:
            print(f"[Registry] список моделей не обновлён: {exc}")
            return

        # Сквозная проверка вместо доверия к коду ответа: список кузовов
        # приходит с TLS-сервера 9999, который читает СВОЮ копию конфигов, а
        # реестр пишет в основное дерево и зеркалит во второе. Если модель
        # уехала только в основное, обработка фото пройдёт, а в списке набора
        # не будет — молчать об этом нельзя.
        uploaded_key = str(getattr(dialog, "uploaded_key", "") or "")
        if not uploaded_key:
            return
        try:
            seen = self.right_panel.model_info(uploaded_key) is not None
        except Exception:
            return
        if seen:
            print(f"[Registry] '{uploaded_key}' виден в списке кузовов")
            return
        QMessageBox.warning(
            self, "Модель не появилась в списке",
            f"Реестр принял модель «{uploaded_key}», но сервер по-прежнему не "
            "отдаёт её в списке кузовов.\n\n"
            "Обычно это значит, что реестр не зеркалит модель во второе "
            "дерево конфигов — то, из которого читает сервер списка. "
            "Проверьте на сервере флаги --mirror-data-dir / "
            "--mirror-config-dir у демона model-registry.")

    def _on_model_delete_requested(self, model_key: str) -> None:
        """
        Удалить набор моделей с диска по запросу из списка кузовов.

        Удаляются только «свои» наборы: комплекты генератора из
        `assets/models/generated` и локальные модели из `assets/models/trucks`.
        Серверные наборы сюда не попадают — список их и не предлагает.

        Диалог показывает ПОЛНЫЙ список файлов и папок: комплект генератора это
        не один .bam, а россыпь из кузова, наполнителя, шасси, glTF и папки с
        текстурами на сотню мегабайт, и пользователь должен видеть, что именно
        исчезнет. Отмена — кнопка по умолчанию: операция необратима.
        """
        from PyQt6.QtWidgets import QMessageBox
        from src.ui.panel_data import (delete_model_set,
                                       plan_model_set_removal)

        key = str(model_key or "")
        if not key:
            return

        plan = plan_model_set_removal(key)
        if not plan.ok:
            QMessageBox.warning(self, "Удаление набора",
                                plan.error or "нечего удалять")
            return

        current_key = ""
        try:
            current_key = str(self.right_panel.current_model_key() or "")
        except Exception:
            pass
        was_current = (current_key == key)

        text = (f"Удалить набор «{plan.name}»?\n\n"
                f"С диска будет удалено {len(plan.paths)} объект(ов), "
                f"{plan.size_label}. Отменить удаление нельзя.")
        if was_current:
            text += ("\n\nЭтот набор сейчас выбран — после удаления в сцену "
                     "загрузится первый из оставшихся.")

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Удаление набора")
        box.setText(text)
        box.setDetailedText("\n".join(plan.paths))
        yes = box.addButton("Удалить", QMessageBox.ButtonRole.DestructiveRole)
        cancel = box.addButton("Отмена", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(cancel)
        box.exec()
        if box.clickedButton() is not yes:
            return

        ok, message = delete_model_set(key)
        print(f"[ModelSet] удаление '{key}': {message}")
        if not ok:
            QMessageBox.warning(self, "Удаление набора", message)

        try:
            self.right_panel.reload_model_sets()
            self.right_panel.set_model_status(message[:120])
        except Exception as exc:
            print(f"[ModelSet] не удалось обновить список моделей: {exc}")
            return

        if not was_current:
            # В сцене чужой набор — перечитанный список сбросил выбор на
            # первую строку, возвращаем его на место (без перезагрузки модели:
            # set_current_model_key блокирует сигналы).
            if current_key:
                try:
                    self.right_panel.set_current_model_key(current_key)
                except Exception as exc:
                    print(f"[ModelSet] выбор не восстановлен: {exc}")
            return

        # Удалённый набор мог быть в сцене: его файлов больше нет, поэтому
        # подтягиваем то, что панель выбрала вместо него.
        if ok:
            try:
                new_key = self.right_panel.current_model_key()
                if new_key:
                    self._on_model_set_changed(str(new_key))
            except Exception as exc:
                print(f"[ModelSet] загрузка замены не удалась: {exc}")

    def closeEvent(self, e):
        # 1. Stop the Qt-driven loops first, so neither taskMgr.step() nor the
        #    telemetry callback runs against a Panda app we're tearing down.
        try:
            if hasattr(self, "_panda_timer"):
                self._panda_timer.stop()
        except Exception:
            pass
        try:
            if hasattr(self, "_telemetry_timer"):
                self._telemetry_timer.stop()
        except Exception:
            pass
        # 2. Clean, controlled stop of Panda subsystems (particles / Warp).
        try:
            if self.panda_app is not None:
                self.panda_app.shutdown()
        except Exception:
            pass
        super().closeEvent(e)
        # 3. Hard-exit to avoid a 10-30 s system-wide stall on shared-memory
        #    (integrated) GPUs. CPython would otherwise finalize Panda's C++
        #    object graph and delete RenderPipeline's entire GL context one
        #    object at a time, which saturates the GPU driver / DWM and
        #    stutters the cursor and audio across the whole system. The window
        #    is already gone and nothing critical runs at a normal exit
        #    (graphics.json is saved on change; crash_reporter only fires via
        #    sys.excepthook), so let the OS reclaim the GL context and the
        #    address space in a single operation.
        import sys
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        os._exit(0)


class _CallInThread(QThread):
    """
    Выполнить функцию в отдельном потоке и отдать результат в главный.

    `finishedWith` получает возвращённое значение либо исключение, которым
    функция упала. Для работы с сетью и диском: сокеты и файловый ввод-вывод
    отпускают GIL, поэтому кадры Panda в это время идут без задержек.
    """

    finishedWith = pyqtSignal(object)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self) -> None:
        try:
            result = self._fn()
        except Exception as exc:
            traceback.print_exc()
            result = exc
        self.finishedWith.emit(result)


def _chassis_from_name(display_name: str) -> str:
    """
    Колёсная формула из имени модели: «6x4», «8x4» или "" — не нашлась.

    Кириллическая «х» в этих обозначениях встречается не реже латинской
    («Sitrak 8х4 (6.3м)»), поэтому ищутся обе. «4 оси» / «3 оси» тоже
    считаются: так подписаны модели, у которых марку шасси не опознали.
    """
    import re as _re

    low = str(display_name or "").lower()
    m = _re.search(r"\b([68])\s*[xх]\s*([24])\b", low)
    if m:
        return f"{m.group(1)}x{m.group(2)}"
    if _re.search(r"\b4\s*(оси|ось)\b", low):
        return "8x4"
    if _re.search(r"\b3\s*(оси|ось)\b", low):
        return "6x4"
    return ""


class _BodyGenWorker(QThread):
    """
    Поток сборки кузова.

    Наружу отдаёт только строки состояния и итоговый объект: в поток не
    передаётся ничего из сцены, поэтому Panda3D продолжает рисовать всё время
    сборки.
    """

    progressed = pyqtSignal(str)
    finishedWith = pyqtSignal(object)

    def __init__(self, params, parent=None):
        super().__init__(parent)
        self._params = params

    def run(self) -> None:
        from src.bodygen import generate
        result = generate(self._params, progress=self.progressed.emit)
        self.finishedWith.emit(result)
