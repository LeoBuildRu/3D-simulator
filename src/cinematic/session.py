# -*- coding: utf-8 -*-
"""
Кинематографичная реконструкция по проезду: запуск, связь с окном, выход.

    session = CineSession(main_window, rec)
    session.start()

Пока идёт сцена, камерой управляет режиссёр (src/cinematic/director.py), а
сцена утилиты готовится под покровом «абстрактного мира»: штатное применение
реконструкции (`MainWindow._apply_reconstruction`) вызывается, когда её не
видно, и готовый PBR-наполнитель прячется до финальной смены текстуры.

Esc — пропустить: всё, что не успело примениться, применяется сразу, эффекты
убираются, камера остаётся на месте.
"""

from __future__ import annotations

import os
import traceback
from typing import Optional

from panda3d.core import Vec4

from src.vfx.camera import CinematicCamera
from src.vfx.compositor import Compositor
from src.vfx.sequencer import Sequencer

from .data import Loader
from .director import Director

CLIP_EFFECT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "effects", "cine_clip.yaml")


class CineSession:
    #: активная сессия (одна на приложение)
    current: Optional["CineSession"] = None

    def __init__(self, window, rec):
        self.win = window
        self.app = window.panda_app
        self.rec = rec
        self.comp = None
        self.seq = None
        self.rig = None
        self.loader = None
        self.data = None
        self.director = None
        self.pbr_applied = False
        self._clip_on = False
        self._done = False

    @staticmethod
    def supported(app) -> bool:
        """Нужен RenderPipeline: сведение берёт готовый кадр его FinalStage."""
        return getattr(app, "render_pipeline", None) is not None

    # ------------------------------------------------------------------ #
    #: сколько ждать подготовки, прежде чем сдаться и показать без кино, с
    PREPARE_TIMEOUT = 240.0

    def start(self) -> None:
        """
        Подготовка: всё скачивается и считается ДО показа (интерфейс на
        месте, на кнопке — ход подготовки), чтобы в самой сцене не было ни
        одного ожидания. Показ начинается, когда готово всё.
        """
        if CineSession.current is not None:
            CineSession.current.skip()
        CineSession.current = self
        app = self.app
        # этот запуск перекрывает любой штатный, начатый раньше
        self.win._recon_seq = getattr(self.win, "_recon_seq", 0) + 1
        self.recon_seq = self.win._recon_seq
        self.loader = Loader(self.win, self.rec)
        self.data = self.loader.start()
        self.playing = False
        self._prep_time = 0.0
        btn = getattr(getattr(self.win, "right_panel", None), "btn_run_recon", None)
        self._btn = btn
        self._btn_text = btn.text() if btn is not None else ""
        app.accept("escape", self.skip)
        app.taskMgr.add(self._prepare_task, "cine_prepare")
        print(f"[cine] подготовка: {self.rec.name}")

    _PARTS = (("photo_ready", "снимок"), ("scene_ready", "анализ снимка и облака"),
              ("fetch_ready", "модель и наполнение"), ("fill_ready", "этапы наполнения"),
              ("bodies_ready", "кузова"))

    def _prepare_task(self, task):
        if self._done:
            return task.done
        from panda3d.core import ClockObject
        self._prep_time += ClockObject.get_global_clock().get_dt()
        d = self.data
        ready = [flag for flag, _ in self._PARTS if getattr(d, flag)]
        if self._btn is not None:
            left = [name for flag, name in self._PARTS if not getattr(d, flag)]
            self._btn.setText(f"Подготовка {len(ready)}/{len(self._PARTS)}"
                              + (f": {d.status or left[0]}" if left else ""))
            self._btn.setEnabled(False)
        if d.error and not d.fetch_ready:
            return task.cont                  # штатная выборка ещё может успеть
        if len(ready) == len(self._PARTS):
            self._restore_button()
            self._play()
            return task.done
        if self._prep_time > self.PREPARE_TIMEOUT or (d.error and d.fetch_ready and not d.fetched):
            print(f"[cine] подготовка не удалась ({d.error or 'таймаут'}) — без кино")
            self.skip()
            return task.done
        return task.cont

    def _restore_button(self) -> None:
        if self._btn is not None:
            self._btn.setText(self._btn_text)
            self._btn.setEnabled(True)

    def _play(self) -> None:
        """Всё готово: гасим интерфейс, забираем камеру, запускаем сцену."""
        app = self.app
        self.playing = True
        speed = float(os.environ.get("IQOKO_CINE_SPEED", "1") or 1)
        try:
            self.win.fade_overlays(False)
        except Exception:
            traceback.print_exc()
        self.comp = Compositor.shared(app, app.render_pipeline)
        self.comp._claimed = True
        self.comp._owner = self
        self.comp.reset()
        self.comp.set_enabled(True)
        self.seq = Sequencer(app)
        self.seq.speed = speed
        self.seq.every_frame(lambda t, dt: self.comp.set_time(t))
        self.rig = CinematicCamera(app)
        self.rig.acquire()
        self.director = Director(self)
        self.seq.spawn(self._run(), "cinematic")
        print(f"[cine] показ: {self.rec.name} (подготовка {self._prep_time:.1f} с)")

    def _run(self):
        try:
            yield from self.director.main()
        finally:
            self.finish()

    # ------------------------------------------------------------------ #
    def apply_pbr_hidden(self) -> None:
        """Поставить реконструкцию в сцену утилиты (пока её не видно)."""
        if self.pbr_applied:
            return
        d = self.data
        out = d.fetched if d is not None else None
        if not d or not d.fetch_ready or out is None:
            return
        if self.win._recon_seq != self.recon_seq:
            return                        # пользователь уже запустил другое
        try:
            self.win._apply_reconstruction(self.rec, out, self.recon_seq)
        except Exception:
            traceback.print_exc()
        self.pbr_applied = True
        fill = self.pbr_fill_node()
        if fill is not None and not self._done:
            fill.hide()
            # эффект отсечения готовится сейчас (генерация шейдеров RP), а не
            # в момент смены текстуры
            self.clip_pbr_fill(None)

    def pbr_fill_node(self):
        node = getattr(self.app, "final_mesh_node", None)
        return None if node is None or node.is_empty() else node

    def clip_pbr_fill(self, plane: Optional[Vec4]) -> None:
        """Отсечь PBR-наполнитель плоскостью (None — показать целиком)."""
        fill = self.pbr_fill_node()
        rp = getattr(self.app, "render_pipeline", None)
        if fill is None or rp is None:
            return
        if not self._clip_on:
            try:
                rp.set_effect(fill, CLIP_EFFECT, {})
                self._clip_on = True
            except Exception:
                traceback.print_exc()
                return
        fill.set_shader_input("cineClip", plane if plane is not None else Vec4(0, 0, 1, 1e6))

    # ------------------------------------------------------------------ #
    def skip(self) -> None:
        """Пропустить: применить результат и закрыть сцену."""
        if self._done:
            return
        print("[cine] пропуск")
        if self.seq is not None:
            self.seq.cancel_all()
        self.finish()

    def finish(self) -> None:
        if self._done:
            return
        self._done = True
        app = self.app
        app.ignore("escape")
        self._restore_button()
        if getattr(self, "playing", False):
            try:
                self.win.fade_overlays(True, 700)
            except Exception:
                traceback.print_exc()
        d = self.data
        if d is not None and not self.pbr_applied and d.fetch_ready:
            self.apply_pbr_hidden()
        elif d is not None and not d.fetch_ready and self.loader is not None:
            # данные не успели — штатная реконструкция доделает сама
            self.loader.shutdown()
            self.loader = None
            try:
                self.win._on_reconstruction_run_plain(self.rec)
            except Exception:
                traceback.print_exc()
        fill = self.pbr_fill_node()
        if fill is not None:
            if self._clip_on:
                fill.set_shader_input("cineClip", Vec4(0, 0, 1, 1e6))
            fill.show()
        if self.rig is not None:
            self.rig.release(keep_pose=True)
        if self.seq is not None:
            # последний кадр сценария ещё идёт — убираем со следующего;
            # компоновщик общий: гасится и чистится, но буферы и шейдеры живут
            seq, comp = self.seq, self.comp

            def cleanup(task):
                seq.destroy()
                # компоновщик мог уже перейти к новой сцене (клик по другому
                # проезду во время этой) — тогда его не трогаем
                if getattr(comp, "_owner", None) is self:
                    comp.reset()
                    comp._claimed = False
                    comp._owner = None
                    comp.set_enabled(False)
                return task.done
            app.taskMgr.do_method_later(0.0, cleanup, "cine_cleanup")
        if self.loader is not None:
            self.loader.shutdown()
        if CineSession.current is self:
            CineSession.current = None
        print("[cine] конец")
