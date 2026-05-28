# main_window.py
# ---------------------------------------------------------------------------
# Minimal Qt main window for the Toner simulator.
# ---------------------------------------------------------------------------

from __future__ import annotations

import win32gui
import win32con

from panda3d.core import WindowProperties

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QFrame,
)

from src.ui.ui_theme import apply_theme
from src.ui.overlay_widgets import SceneOverlay, DepthMapOverlay
from src.ui.right_panel import RightPanel
import os
import json
import math
import random
import time
import shutil
import tempfile
from typing import Any

from src.ui.panel_data import (
    get_model_set_config, get_texture_set_config,
    load_texture_sets, Reconstruction, download_server_image,
    ensure_texture_cached, TEXTURE_PATH_KEYS, get_default_texture_set_key,
)


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

    def __init__(self):
        super().__init__()

        self.panda_app = None
        self._panda_hwnd: int | None = None

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

        self._panda_timer = QTimer(self)
        self._panda_timer.timeout.connect(panda_app.taskMgr.step)
        self._panda_timer.start(16)

        # ---- Depth-map overlay (top-LEFT, live image only) ---------
        # Minimal card: only the depth image is rendered, no chrome.
        # Anchored to the top-left so it sits exactly where the user
        # asked for it.
        self.depth_overlay = DepthMapOverlay(
            parent=self.panda_container,
            anchor="top-left",
            margin=16,
            width=320,
        )
        self.depth_overlay.attach()
        self.depth_overlay.toggleRequested.connect(self._on_depth_toggle)
        # ---- Depth settings strip (NEAR / FAR / GRAD START / GRAD END)
        # Lives inside the same DepthMapOverlay card, below the canvas,
        # mirroring the gui.py behaviour for tuning the depth pass.
        try:
            from PyQt6.QtWidgets import (
                QDoubleSpinBox as _QDSB, QLabel as _QL,
                QGridLayout as _QGL, QFrame as _QFr,
            )
            from src.ui.ui_theme import (
                COLOR_TEXT_MUTED as _DTM, COLOR_HAIRLINE as _DCH,
                COLOR_TEXT as _DCT, FONT_MONO as _DFM,
            )

            depth_settings = _QFr()
            depth_settings.setStyleSheet(
                "QFrame { background: transparent; border: none; }"
            )
            grid = _QGL(depth_settings)
            grid.setContentsMargins(0, 8, 0, 0)
            grid.setHorizontalSpacing(8)
            grid.setVerticalSpacing(4)

            def _make_lbl(text: str):
                lbl = _QL(text)
                lbl.setStyleSheet(
                    f"color: {_DTM}; font-size: 10px;"
                    f" letter-spacing: 0.6px; background: transparent;"
                )
                return lbl

            def _make_spin(rng, val, step, decimals=2):
                sp = _QDSB()
                sp.setRange(*rng)
                sp.setSingleStep(step)
                sp.setDecimals(decimals)
                sp.setValue(val)
                sp.setFixedHeight(22)
                sp.setStyleSheet(
                    "QDoubleSpinBox {"
                    "  background: rgba(255,255,255,4);"
                    f"  color: {_DCT};"
                    f"  border: 1px solid {_DCH};"
                    "  border-radius: 4px;"
                    "  padding: 1px 4px;"
                    f"  font-family: {_DFM};"
                    "  font-size: 11px;"
                    "}"
                    "QDoubleSpinBox::up-button,"
                    "QDoubleSpinBox::down-button { width: 0; }"
                )
                return sp

            self.spn_near = _make_spin((0.01, 1000.0), 0.1, 0.1)
            self.spn_far  = _make_spin((0.1, 10000.0), 100.0, 1.0, decimals=1)
            self.spn_g_a  = _make_spin((0.0, 1.0), 0.2, 0.05)
            self.spn_g_b  = _make_spin((0.0, 1.0), 0.4, 0.05)

            grid.addWidget(_make_lbl("Ближняя"), 0, 0)
            grid.addWidget(self.spn_near,        0, 1)
            grid.addWidget(_make_lbl("Дальняя"), 0, 2)
            grid.addWidget(self.spn_far,         0, 3)
            grid.addWidget(_make_lbl("Начало"),  1, 0)
            grid.addWidget(self.spn_g_a,         1, 1)
            grid.addWidget(_make_lbl("Конец"),   1, 2)
            grid.addWidget(self.spn_g_b,         1, 3)
            grid.setColumnStretch(1, 1)
            grid.setColumnStretch(3, 1)

            self.spn_near.valueChanged.connect(self._on_depth_min_changed)
            self.spn_far.valueChanged.connect(self._on_depth_max_changed)
            self.spn_g_a.valueChanged.connect(self._on_depth_grad_a_changed)
            self.spn_g_b.valueChanged.connect(self._on_depth_grad_b_changed)

            self.depth_overlay.attach_extra(depth_settings)
        except Exception as exc:
            print(f"[MainWindow] depth-settings strip init failed: {exc}")
        # Feed it from panda_app.depth_renderer.depth_texture (this is
        # the texture DepthMapRenderer already populates each frame via
        # its own offscreen camera + display region).
        self._depth_capture_w = 320
        self._depth_capture_h = 180   # 16:9
        self._depth_in_main = False   # default: main = normal, widget = depth
        self._color_mirror_tex = None
        self._color_mirror_buf = None
        self._color_mirror_cam = None
        self._depth_timer = QTimer(self)
        self._depth_timer.timeout.connect(self._tick_depth_overlay)
        # Defer the first tick so RenderPipeline has time to boot - if
        # we start banging on it from frame 0 the splash screen never
        # closes and we get bogus get_screenshot results.
        QTimer.singleShot(3000, lambda: self._depth_timer.start(120))
        # ~8 FPS preview is plenty and stays out of RP's way.

        # ---- Camera telemetry (BOTTOM-LEFT) -------------------------
        self.telemetry = SceneOverlay(
            "Камера · Телеметрия",
            anchor="bottom-left",
            parent=self.panda_container,
            margin=16,
        )
        self.telemetry.set_rows([
            ("PITCH", "  0.0"),
            ("YAW",   "  0.0"),
            ("ROLL",  "  0.0"),
            ("FOV",   " 60.0"),
            ("X",     "  0.0"),
            ("Y",     "  0.0"),
            ("Z",     "  0.0"),
        ])
        self.telemetry.attach()

        # ---- Time-of-day slider (sits inside the telemetry card) ----
        try:
            from PyQt6.QtWidgets import QSlider, QLabel as _QLabel, QHBoxLayout as _QHB
            from src.ui.ui_theme import (
                COLOR_TEXT_MUTED as _CTM, COLOR_TEXT as _CT,
                COLOR_ACCENT as _CA, COLOR_HAIRLINE as _CH,
                FONT_MONO as _FM,
            )

            self.daytime_slider_holder = QFrame()
            self.daytime_slider_holder.setStyleSheet(
                "QFrame { background: transparent; border: none; }"
            )
            holder_lay = QHBoxLayout(self.daytime_slider_holder)
            holder_lay.setContentsMargins(0, 8, 0, 0)
            holder_lay.setSpacing(8)

            label = _QLabel("TIME")
            label.setStyleSheet(
                f"color: {_CTM}; font-size: 10px;"
                f" letter-spacing: 1.0px; background: transparent;"
            )
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 1439)              # minutes in a day
            slider.setValue(6 * 60 + 40)          # 06:40 default
            slider.setFixedHeight(18)
            slider.setStyleSheet(
                "QSlider::groove:horizontal {"
                f"  background: {_CH};"
                "  height: 3px; border-radius: 1px;"
                "}"
                "QSlider::sub-page:horizontal {"
                f"  background: {_CA}; height: 3px; border-radius: 1px;"
                "}"
                "QSlider::handle:horizontal {"
                f"  background: {_CA};"
                "  width: 10px; height: 10px;"
                "  margin: -4px 0; border-radius: 5px;"
                "}"
                "QSlider::handle:horizontal:hover {"
                "  background: #00FFAA;"
                "}"
            )

            value_lbl = _QLabel("06:40")
            value_lbl.setStyleSheet(
                f"color: {_CT}; font-family: {_FM};"
                f"font-size: 11px; background: transparent;"
            )
            value_lbl.setMinimumWidth(38)
            value_lbl.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )

            holder_lay.addWidget(label, 0, Qt.AlignmentFlag.AlignVCenter)
            holder_lay.addWidget(slider, 1, Qt.AlignmentFlag.AlignVCenter)
            holder_lay.addWidget(value_lbl, 0, Qt.AlignmentFlag.AlignVCenter)

            def _on_daytime_changed(mins: int,
                                     _vlbl=value_lbl,
                                     _app=panda_app):
                hh = mins // 60
                mm = mins % 60
                txt = f"{hh:02d}:{mm:02d}"
                _vlbl.setText(txt)
                # Push to RenderPipeline's daytime manager.
                try:
                    rp = getattr(_app, "render_pipeline", None)
                    dt_mgr = getattr(rp, "daytime_mgr", None) if rp else None
                    if dt_mgr is not None:
                        dt_mgr.time = txt
                except Exception as exc:
                    print(f"[Daytime] set failed: {exc}")

            slider.valueChanged.connect(_on_daytime_changed)
            self.daytime_slider = slider
            self.daytime_value_lbl = value_lbl

            # Append the row inside the telemetry card body.
            self.telemetry.attach_extra(self.daytime_slider_holder)
            # Trigger an initial sync so RP's daytime matches the slider.
            _on_daytime_changed(slider.value())
        except Exception as exc:
            print(f"[MainWindow] daytime slider init failed: {exc}")

        # ---- Camera-mode buttons (FREE / STATIC / BOARD) ------------
        try:
            from PyQt6.QtWidgets import (
                QPushButton as _QPB,
                QHBoxLayout as _QHB,
                QFrame as _QFr,
            )
            from src.ui.ui_theme import (
                COLOR_TEXT as _MCT,
                COLOR_TEXT_MUTED as _MCTM,
                COLOR_HAIRLINE as _MCH,
                COLOR_ACCENT as _MCA,
            )
            self._camera_mode = "free"   # free | stationary | onboard

            mode_holder = _QFr()
            mode_holder.setStyleSheet(
                "QFrame { background: transparent; border: none; }"
            )
            mh_lay = _QHB(mode_holder)
            mh_lay.setContentsMargins(0, 6, 0, 0)
            mh_lay.setSpacing(4)

            def _seg_button_qss(active: bool) -> str:
                if active:
                    return (
                        "QPushButton {"
                        f"  background-color: rgba(0, 255, 136, 50);"
                        f"  color: {_MCT};"
                        f"  border: 1px solid {_MCA};"
                        "  border-radius: 5px;"
                        "  padding: 4px 6px;"
                        "  font-size: 10px;"
                        "  font-weight: 700;"
                        "  letter-spacing: 0.6px;"
                        "}"
                    )
                return (
                    "QPushButton {"
                    "  background: transparent;"
                    f"  color: {_MCTM};"
                    f"  border: 1px solid {_MCH};"
                    "  border-radius: 5px;"
                    "  padding: 4px 6px;"
                    "  font-size: 10px;"
                    "  font-weight: 600;"
                    "  letter-spacing: 0.6px;"
                    "}"
                    "QPushButton:hover {"
                    "  background: rgba(255,255,255,8);"
                    f"  color: {_MCT};"
                    "}"
                )

            self._mode_btns: dict = {}
            for code, label in (
                ("free",       "СВОБ"),
                ("stationary", "СТАЦ"),
                ("onboard",    "БОРТ"),
            ):
                btn = _QPB(label)
                btn.setCursor(Qt.CursorShape.PointingHandCursor)
                btn.setFixedHeight(22)
                btn.setStyleSheet(_seg_button_qss(code == self._camera_mode))
                btn.clicked.connect(
                    lambda _checked=False, c=code: self._on_camera_mode(c)
                )
                self._mode_btns[code] = btn
                mh_lay.addWidget(btn, 1)

            self.telemetry.attach_extra(mode_holder)
            # Cache the qss-builder so _on_camera_mode can re-style.
            self._seg_button_qss = _seg_button_qss
        except Exception as exc:
            print(f"[MainWindow] camera-mode buttons init failed: {exc}")

        # ---- Save-render row (count + button) ----------------------
        try:
            from PyQt6.QtWidgets import (
                QSpinBox as _QSB,
                QPushButton as _QPB2,
                QHBoxLayout as _QHB2,
                QLabel as _QL2,
                QFrame as _QFr2,
            )
            from src.ui.ui_theme import (
                COLOR_TEXT as _RCT,
                COLOR_TEXT_MUTED as _RCTM,
                COLOR_HAIRLINE as _RCH,
                FONT_MONO as _RFM,
            )

            save_holder = _QFr2()
            save_holder.setStyleSheet(
                "QFrame { background: transparent; border: none; }"
            )
            sh_lay = _QHB2(save_holder)
            sh_lay.setContentsMargins(0, 6, 0, 0)
            sh_lay.setSpacing(6)

            sr_label = _QL2("СНИМОК")
            sr_label.setStyleSheet(
                f"color: {_RCTM}; font-size: 10px;"
                f" letter-spacing: 0.6px; background: transparent;"
            )

            self.spn_render_count = _QSB()
            self.spn_render_count.setRange(1, 50)
            self.spn_render_count.setValue(1)
            self.spn_render_count.setFixedHeight(22)
            self.spn_render_count.setFixedWidth(56)
            self.spn_render_count.setStyleSheet(
                "QSpinBox {"
                "  background: rgba(255,255,255,4);"
                f"  color: {_RCT};"
                f"  border: 1px solid {_RCH};"
                "  border-radius: 4px;"
                "  padding: 1px 4px;"
                f"  font-family: {_RFM};"
                "  font-size: 11px;"
                "}"
                "QSpinBox::up-button, QSpinBox::down-button { width: 0; }"
            )

            self.btn_save_render = _QPB2("Сохранить")
            self.btn_save_render.setCursor(Qt.CursorShape.PointingHandCursor)
            self.btn_save_render.setFixedHeight(22)
            self.btn_save_render.setStyleSheet(
                "QPushButton {"
                "  background-color: rgba(0, 255, 136, 30);"
                f"  color: {_RCT};"
                "  border: 1px solid #00FF88;"
                "  border-radius: 5px;"
                "  padding: 2px 12px;"
                "  font-size: 10px;"
                "  font-weight: 600;"
                "  letter-spacing: 0.4px;"
                "}"
                "QPushButton:hover {"
                "  background-color: rgba(0, 255, 136, 55);"
                "}"
                "QPushButton:pressed {"
                "  background-color: rgba(0, 255, 136, 90);"
                "}"
                "QPushButton:disabled {"
                "  background: rgba(255, 255, 255, 4);"
                f"  color: {_RCTM};"
                f"  border: 1px solid {_RCH};"
                "}"
            )
            self.btn_save_render.clicked.connect(self._on_save_render_clicked)

            sh_lay.addWidget(sr_label, 0, Qt.AlignmentFlag.AlignVCenter)
            sh_lay.addWidget(self.spn_render_count, 0, Qt.AlignmentFlag.AlignVCenter)
            sh_lay.addStretch(1)
            sh_lay.addWidget(self.btn_save_render, 0, Qt.AlignmentFlag.AlignVCenter)

            self.telemetry.attach_extra(save_holder)
        except Exception as exc:
            print(f"[MainWindow] save-render row init failed: {exc}")

        self.controls = SceneOverlay(
            "Управление",
            anchor="top-left",
            parent=self.panda_container,
            width=240,
            margin=16,
        )
        self.controls.set_rows([
            ("WASD",  "Движение"),
            ("Q / E", "Вниз / Вверх"),
            ("Shift", "Ускорение"),
            ("ПКМ",   "Обзор"),
        ])
        self.controls.attach()
        # Anchor the Controls card to the right of the depth overlay
        # (same trick the telemetry used to use before the swap).
        try:
            from PyQt6.QtCore import QPoint as _QP
            _depth = self.depth_overlay
            _top_y = 16
            _x_off = 16 + _depth.width() + 12
            def _controls_reposition(_self=self.controls,
                                      _owner=self.panda_container,
                                      _x=_x_off, _y=_top_y):
                w = _self.width()
                h = _self.sizeHint().height()
                gp = _owner.mapToGlobal(_QP(_x, _y))
                _self.setGeometry(gp.x(), gp.y(), w, h)
                _self.raise_()
            import types as _types
            self.controls._reposition = _types.MethodType(
                lambda s, _f=_controls_reposition: _f(),
                self.controls,
            )
            self.controls._reposition()
        except Exception as exc:
            print(f"[MainWindow] controls reposition patch failed: {exc}")

        self.right_panel = RightPanel(parent=self.panda_container)
        self.right_panel.attach()
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

        print("=" * 60)
        print(f"[Run] Pipeline start. target_volume={target_volume:.2f}  "
              f"model_key={model_key!r}  texture_key={texture_key!r}")

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

        recon_module = getattr(self.panda_app, "mesh_reconstruction", None)
        if recon_module is None:
            print("[Recon] panda_app.mesh_reconstruction not available.")
            return

        print("=" * 60)
        print(f"[Recon] click '{rec.name}' (data_type={rec.data_type}, "
              f"is_local={rec.is_local})")

        # ---- 1) Resolve JSON path (local or download) ---------------
        local_json_path = self._resolve_recon_json(rec)
        if not local_json_path or not os.path.exists(local_json_path):
            print(f"[Recon] could not resolve JSON for {rec.name!r}")
            print("=" * 60)
            return

        # ---- 2) Parse JSON ------------------------------------------
        try:
            with open(local_json_path, "r", encoding="utf-8") as fp:
                json_data = json.load(fp)
        except Exception as exc:
            print(f"[Recon] failed to read JSON: {exc}")
            print("=" * 60)
            return

        filler        = json_data.get("filler") or rec.filler
        model_name    = json_data.get("model")  or rec.model
        target_volume = json_data.get("target_volume")
        car_number    = json_data.get("car_number") or rec.car_number
        time_str      = json_data.get("time") or rec.time

        # ---- 3) Apply texture set by filler -------------------------
        if filler:
            tex_key, tex_cfg = self._find_texture_by_filler(filler)
            if tex_cfg is not None and hasattr(self.panda_app, "set_texture_set"):
                # Сохраняем сырой конфиг для серверного displace.
                self.panda_app.current_texture_set_raw = dict(tex_cfg)
                try:
                    self.panda_app.set_texture_set(
                        self._materialize_texture_set(tex_cfg)
                    )
                    print(f"[Recon] texture set by filler: '{tex_key}'")
                except Exception as exc:
                    print(f"[Recon] set_texture_set failed: {exc}")
            else:
                print(f"[Recon] no texture set found for filler='{filler}'")

        # ---- 4) Load model set --------------------------------------
        if model_name:
            model_key = self._find_model_key_by_name(model_name)
            if (model_key
                    and hasattr(self.panda_app, "cache_and_load_model_set")):
                cfg = get_model_set_config(model_key)
                if cfg is not None:
                    try:
                        ok = bool(self.panda_app.cache_and_load_model_set(
                            model_key, cfg
                        ))
                        print(f"[Recon] model set "
                              f"'{model_key}' loaded: {ok}")
                        # Синхронизируем выбор в правой панели: иначе
                        # right_panel.current_model_key() продолжает
                        # возвращать прежний (дефолтный) ключ, и
                        # _apply_onboard_camera берёт камеру не той модели.
                        if ok:
                            rp = getattr(self, "right_panel", None)
                            if rp is not None and hasattr(
                                rp, "set_current_model_key"
                            ):
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
                        print(f"[Recon] cache_and_load_model_set: {exc}")
                else:
                    print(f"[Recon] config for '{model_key}' missing.")
            else:
                print(f"[Recon] no model key for '{model_name}'.")

        # ---- 5) Resolve PLY path (download if SERVER) ---------------
        local_ply_path = None
        ply_filename = json_data.get("ply_file") or rec.ply_file
        if ply_filename:
            local_ply_path = self._resolve_recon_ply(rec, ply_filename,
                                                    local_json_path)
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
                    print(f"[Recon] heightmap '{heightmap_filename}' "
                          f"could not be resolved - aborting.")
                    print("=" * 60)
                    return

        # ---- 7) Push overlay info if MyApp supports it --------------
        try:
            if hasattr(self.panda_app, "update_overlay_info"):
                self.panda_app.update_overlay_info(
                    texture=filler,
                    car_number=car_number,
                    initial_volume=target_volume,
                    time=time_str,
                )
        except Exception as exc:
            print(f"[Recon] update_overlay_info failed: {exc}")

        # ---- 8) Run the reconstruction ------------------------------
        try:
            if rec.data_type == "height":
                print("[Recon] launching height-map reconstruction...")
                recon_module.run_2d_to_3d_reconstruction_from(
                    json_path=local_json_path
                )
            else:
                print("[Recon] launching PLY reconstruction...")
                recon_module.run_2d_to_3d_reconstruction_from(
                    json_path=local_json_path,
                    ply_path=local_ply_path,
                )
            print("[Recon] OK pipeline finished.")
        except Exception as exc:
            print(f"[Recon] ERR run_2d_to_3d_reconstruction_from: {exc}")

        print("=" * 60)

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
            tex.set_keep_ram_image(True)
            buf = self.panda_app.win.make_texture_buffer(
                "color_mirror_buf",
                self._depth_capture_w, self._depth_capture_h,
                tex, to_ram=True,
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

        try:
            import numpy as np
            from PyQt6.QtGui import QImage

            tex = dr.depth_texture
            if not tex.has_ram_image():
                return
            ram = tex.get_ram_image_as("D")
            if ram is None:
                return
            buf = memoryview(ram).tobytes()
            if not buf:
                return
            tw = tex.get_x_size()
            th = tex.get_y_size()
            if tw * th * 4 != len(buf):
                return

            depth = np.frombuffer(buf, dtype=np.float32).reshape(th, tw)

            # Linearise non-linear z-buffer using the same formula as
            # depth_renderer's overlay shader:
            #   linear = (2 * near) / (far + near - depth*(far-near))
            near = float(getattr(dr, "min_depth", 0.1))
            far  = float(getattr(dr, "max_depth", 100.0))
            den = (far + near) - depth * (far - near)
            den = np.where(np.abs(den) < 1e-6, 1e-6, den)
            linear = (2.0 * near) / den

            # Map linear depth into the gradient window using the same
            # gradientStart / gradientEnd that the overlay shader uses.
            gs = float(getattr(dr, "gradient_start", 0.2))
            ge = float(getattr(dr, "gradient_end",   0.4))
            if abs(ge - gs) < 1e-6:
                ge = gs + 1.0
            n = np.clip((linear - gs) / (ge - gs), 0.0, 1.0)
            t = 1.0 - n   # close = "hot" colours (red), far = blue

            # Stride downscale to preview resolution.
            tw_out = self._depth_capture_w
            th_out = self._depth_capture_h
            sx = max(1, tw // tw_out)
            sy = max(1, th // th_out)
            t_small = t[::sy, ::sx]
            if t_small.shape[0] > th_out:
                t_small = t_small[:th_out, :]
            if t_small.shape[1] > tw_out:
                t_small = t_small[:, :tw_out]
            sh, sw = t_small.shape

            # Build / reuse the rainbow LUT (256 entries, RGBA u8).
            lut = self._get_rainbow_lut()
            idx = (np.clip(t_small, 0.0, 1.0) * 255.0).astype(np.uint8)
            rgba = lut[idx]   # (sh, sw, 4)
            rgba = np.ascontiguousarray(rgba)

            data = rgba.tobytes()
            img = QImage(data, sw, sh, sw * 4,
                         QImage.Format.Format_RGBA8888)
            img = img.mirrored(False, True)
            img = img.copy()
            self.depth_overlay.set_image(img)
        except Exception as exc:
            print(f"[Depth] tick failed: {exc}")

    @staticmethod
    def _get_rainbow_lut():
        """
        256-entry RGBA LUT mirroring depth_renderer's overlay shader
        gradient (red -> orange -> yellow -> emerald -> blue -> dark blue).
        Cached on the class for cheap repeated lookups.
        """
        import numpy as np
        cls = MainWindow
        lut = getattr(cls, "_rainbow_lut", None)
        if lut is not None:
            return lut

        out = np.zeros((256, 4), dtype=np.uint8)
        # Stop list mirrors the shader segments exactly.
        # t in [0..1]; format: (t_low, color_low_rgb, t_high, color_high_rgb)
        stops = [
            (0.00, (0.0, 0.0, 0.3), 0.10, (0.0, 0.0, 1.0)),     # dark blue -> blue
            (0.10, (0.0, 0.0, 1.0), 0.30, (0.1, 0.7, 0.4)),     # blue -> emerald
            (0.30, (0.1, 0.7, 0.4), 0.50, (1.0, 1.0, 0.0)),     # emerald -> yellow
            (0.50, (1.0, 1.0, 0.0), 0.70, (1.0, 0.5, 0.0)),     # yellow -> orange
            (0.70, (1.0, 0.5, 0.0), 0.90, (1.0, 0.0, 0.0)),     # orange -> red
            (0.90, (1.0, 0.0, 0.0), 1.01, (0.5, 0.0, 0.0)),     # red -> dark red
        ]
        for i in range(256):
            t = i / 255.0
            for tl, cl, th, ch in stops:
                if tl <= t < th:
                    a = (t - tl) / (th - tl)
                    r = cl[0] + (ch[0] - cl[0]) * a
                    g = cl[1] + (ch[1] - cl[1]) * a
                    b = cl[2] + (ch[2] - cl[2]) * a
                    out[i, 0] = int(np.clip(r * 255.0, 0, 255))
                    out[i, 1] = int(np.clip(g * 255.0, 0, 255))
                    out[i, 2] = int(np.clip(b * 255.0, 0, 255))
                    out[i, 3] = 255
                    break
        cls._rainbow_lut = out
        return out

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

        # Repaint the segment buttons.
        try:
            for code, btn in self._mode_btns.items():
                btn.setStyleSheet(self._seg_button_qss(code == mode))
        except Exception:
            pass

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
            print(f"[Camera] ONBOARD '{key}' pos=({cx},{cy},{cz}) "
                  f"hpr=({ch},{cp},{cr}) fov={self._ONBOARD_FOV}")
        except Exception as exc:
            print(f"[Camera] onboard preset failed: {exc}")
        self._apply_depth_preset(self._ONBOARD_DEPTH)

    # ==================================================================
    # Save-render handler
    # ==================================================================
    def _on_save_render_clicked(self) -> None:
        """
        Walk the filling from `step` -> `max_volume` in N evenly-spaced
        target_volume steps, run the full pipeline at each step and
        save a render. Step formula matches the user's spec: for N=10
        and max_volume=20 -> targets [2, 4, 6, 8, 10, 12, 14, 16, 18, 20].
        """
        if self.panda_app is None:
            return
        ru = getattr(self.panda_app, "renderer_utils", None)
        if ru is None or not hasattr(ru, "save_single_render"):
            print("[SaveRender] renderer_utils.save_single_render missing.")
            return
        try:
            count = int(self.spn_render_count.value())
        except Exception:
            count = 1
        count = max(1, count)

        # Resolve current model + texture from the right panel and pull
        # max_volume from the model's YAML config.
        rp = getattr(self, "right_panel", None)
        model_key   = rp.current_model_key()   if rp is not None else None
        texture_key = rp.current_texture_key() if rp is not None else None
        max_volume  = None
        if model_key:
            cfg = get_model_set_config(str(model_key))
            if cfg and cfg.get("max_volume") is not None:
                try:
                    max_volume = float(cfg["max_volume"])
                except (TypeError, ValueError):
                    max_volume = None
        if max_volume is None or max_volume <= 0:
            print("[SaveRender] max_volume not available for current "
                  "model set - falling back to spinbox-only count loop.")
            max_volume = None

        from PyQt6.QtWidgets import QApplication
        self.btn_save_render.setEnabled(False)
        original_text = self.btn_save_render.text()
        ok_count = 0

        # Freeze fly-cam for the whole dataset run, чтобы наши setPos /
        # setHpr на каждом варианте не сбивались тиком fly_cam.
        fc = getattr(self.panda_app, "fly_cam", None)
        prev_frozen = None
        if fc is not None and hasattr(fc, "set_frozen"):
            try:
                prev_frozen = (
                    fc.is_frozen() if hasattr(fc, "is_frozen") else None
                )
                fc.set_frozen(True)
            except Exception:
                prev_frozen = None

        base_daytime_mins = 6 * 60 + 40
        if hasattr(self, "daytime_slider"):
            try:
                base_daytime_mins = int(self.daytime_slider.value())
            except Exception:
                pass

        def _set_daytime(mins: int) -> None:
            mins = int(mins) % 1440
            hh, mm = mins // 60, mins % 60
            txt = f"{hh:02d}:{mm:02d}"
            try:
                rp = getattr(self.panda_app, "render_pipeline", None)
                dt_mgr = getattr(rp, "daytime_mgr", None) if rp else None
                if dt_mgr is not None:
                    dt_mgr.time = txt
            except Exception as exc:
                print(f"[SaveRender] daytime set failed: {exc}")

        try:
            for i in range(count):
                # Target ramp: step = max_volume/N, target = step*(i+1).
                # Чем больше N (текущий индекс), тем больше объём
                # наполнения — та же логика, что и раньше.
                if max_volume is not None:
                    target = (max_volume / count) * (i + 1)
                else:
                    target = float(rp.current_target_volume()) if rp else 0.0

                self.btn_save_render.setText(f"{i+1}/{count}")
                QApplication.processEvents()

                # Сгенерировать новое наполнение для этой итерации.
                try:
                    self._on_run_simulation({
                        "model_key":     model_key,
                        "texture_key":   texture_key,
                        "target_volume": float(target),
                    })
                except Exception as exc:
                    print(f"[SaveRender] pipeline {i+1} failed: {exc}")
                    QApplication.processEvents()
                    continue

                # Подождать, пока финальные кадры пайплайна успеют
                # отрисоваться.
                for _ in range(4):
                    QApplication.processEvents()
                    time.sleep(0.05)

                # Базовое состояние камеры — то, что выбрал пользователь
                # (free / stationary / onboard уже выставил позицию).
                cam = self.panda_app.camera
                base_pos = cam.getPos()
                base_hpr = cam.getHpr()
                base_pos_t = (float(base_pos.x),
                              float(base_pos.y),
                              float(base_pos.z))
                base_hpr_t = (float(base_hpr.x),
                              float(base_hpr.y),
                              float(base_hpr.z))

                # 10 равномерно распределённых временных меток
                # (0:00, 2:24, ... 21:36).
                tod_list = [int(round(k * 1440.0 / 10.0)) for k in range(10)]

                # Один "альтернативный" момент времени, гарантированно
                # отличающийся от базового хотя бы на 2 часа.
                far_choices = [
                    t for t in range(0, 1440, 30)
                    if abs(t - base_daytime_mins) >= 120
                ]
                alt_time = random.choice(far_choices) if far_choices else 0

                # 5 см → 0.05 единицы Panda (проект работает в метрах).
                OFFSET_M = 0.05
                ANG_DEG = 10.0

                variants: list[tuple[str, dict]] = [
                    ("orig",          {}),
                    ("light_alt",     {"time": alt_time}),
                    ("h_plus10",      {"dh":  +ANG_DEG}),
                    ("h_minus10",     {"dh":  -ANG_DEG}),
                    ("p_plus10",      {"dp":  +ANG_DEG}),
                    ("p_minus10",     {"dp":  -ANG_DEG}),
                    ("lat_plus5cm",   {"lat": +OFFSET_M}),
                    ("lat_minus5cm",  {"lat": -OFFSET_M}),
                    ("vert_plus5cm",  {"vert": +OFFSET_M}),
                    ("vert_minus5cm", {"vert": -OFFSET_M}),
                ]
                for t in tod_list:
                    variants.append((f"tod_{t:04d}m", {"time": t}))
                variants.append((
                    "random_combined",
                    {
                        "dh":   random.uniform(-ANG_DEG,  ANG_DEG),
                        "dp":   random.uniform(-ANG_DEG,  ANG_DEG),
                        "lat":  random.uniform(-OFFSET_M, OFFSET_M),
                        "vert": random.uniform(-OFFSET_M, OFFSET_M),
                        "time": random.randint(0, 1439),
                    },
                ))

                for v_idx, (v_name, p) in enumerate(variants):
                    # 1) Восстанавливаем базовую позу
                    cam.setPos(*base_pos_t)
                    cam.setHpr(*base_hpr_t)

                    # 2) Угловые отклонения (heading = горизонталь,
                    #    pitch = вертикаль).
                    dh = float(p.get("dh", 0.0))
                    dp = float(p.get("dp", 0.0))
                    if dh or dp:
                        cam.setHpr(
                            base_hpr_t[0] + dh,
                            base_hpr_t[1] + dp,
                            base_hpr_t[2],
                        )

                    # 3) Смещения в локальном фрейме камеры:
                    #    +X — вправо, +Z — вверх.
                    lat = float(p.get("lat", 0.0))
                    vert = float(p.get("vert", 0.0))
                    if lat or vert:
                        cam.setPos(cam, lat, 0.0, vert)

                    # 4) Время суток
                    t_val = p.get("time", None)
                    if t_val is None:
                        _set_daytime(base_daytime_mins)
                    else:
                        _set_daytime(int(t_val))

                    # Дать UI/Panda обработать setPos/setHpr и обновлённое
                    # освещение перед тем как звать save_single_render
                    # (внутри он сам делает дополнительные ручные тики
                    # против motion blur).
                    for _ in range(3):
                        QApplication.processEvents()
                        time.sleep(0.05)

                    self.btn_save_render.setText(
                        f"{i+1}/{count} · {v_idx+1}/{len(variants)}"
                    )
                    QApplication.processEvents()

                    applied_time = (
                        int(t_val) if t_val is not None
                        else int(base_daytime_mins)
                    )
                    extra_meta = {
                        "render_type": "dataset",
                        "iteration": i,
                        "iteration_total": count,
                        "variant": v_name,
                        "variant_index": v_idx,
                        "variant_params": p,
                        "camera_mode": getattr(self, "_camera_mode", None),
                        "base_camera_position": {
                            "x": base_pos_t[0],
                            "y": base_pos_t[1],
                            "z": base_pos_t[2],
                        },
                        "base_camera_rotation": {
                            "h": base_hpr_t[0],
                            "p": base_hpr_t[1],
                            "r": base_hpr_t[2],
                        },
                        "base_daytime_minutes": int(base_daytime_mins),
                        "applied_daytime_minutes": applied_time,
                        "target_volume": float(target),
                        "model_key":   model_key,
                        "texture_key": texture_key,
                    }

                    prefix = (
                        f"i{i:03d}_vol{target:07.2f}_"
                        f"v{v_idx:02d}_{v_name}"
                    )

                    try:
                        ok = ru.save_single_render(
                            output_dir="renders/dataset",
                            filename_prefix=prefix,
                            extra_metadata=extra_meta,
                        )
                        if ok:
                            ok_count += 1
                            print(f"[SaveRender] {i+1}/{count} "
                                  f"v={v_name} target={target:.2f} saved")
                        else:
                            print(f"[SaveRender] {i+1}/{count} "
                                  f"v={v_name} returned False")
                    except Exception as exc:
                        print(f"[SaveRender] {i+1}/{count} "
                              f"v={v_name} save failed: {exc}")
                    QApplication.processEvents()

                # Восстановить базовую позу и время после всех вариантов
                # текущей итерации, чтобы следующий _on_run_simulation
                # стартовал с того же состояния, что и пользователь видит.
                cam.setPos(*base_pos_t)
                cam.setHpr(*base_hpr_t)
                _set_daytime(base_daytime_mins)
        finally:
            # Вернуть fly_cam в его прежнее состояние.
            if fc is not None and hasattr(fc, "set_frozen") and prev_frozen is not None:
                try:
                    fc.set_frozen(bool(prev_frozen))
                except Exception:
                    pass
            self.btn_save_render.setText(original_text)
            self.btn_save_render.setEnabled(True)
        print(f"[SaveRender] saved {ok_count} render(s) across "
              f"{count} iteration(s); max_volume={max_volume}")

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

    def _reposition_panda(self) -> None:
        if self.panda_app is None:
            return
        w = max(1, self.panda_container.width())
        h = max(1, self.panda_container.height())
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

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self.panda_app is not None:
            try:
                self._reposition_panda()
            except Exception:
                pass

    def closeEvent(self, e):
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
        super().closeEvent(e)
