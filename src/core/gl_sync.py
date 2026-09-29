# -*- coding: utf-8 -*-
"""
Синхронизация CPU и GPU без простоев: ограничитель очереди кадров и
асинхронное чтение пикселей.

Зачем
-----
Panda 1.10 умеет читать картинку с GPU только синхронно (`RTM_copy_ram`,
`trigger_copy`, `get_screenshot`): драйвер ждёт, пока GPU доделает ВСЁ, что ему
уже отправлено. А отправлено бывает много: CPU готовит кадр за ~4 мс, GPU рисует
его ~15 мс, и драйвер копит до трёх кадров впрок. Поэтому одно чтение даже
крошечного превью стоило ~50 мс — видимый рывок, — а камера отставала от мыши на
те же ~50 мс.

Что здесь
---------
* `FrameLatencyLimiter` — в конце каждого кадра ставит GL-fence и ждёт fence
  предыдущего кадра. GPU всё так же работает внахлёст с CPU (один кадр в
  запасе), но очередь не растёт: задержка ввода — один кадр, интервалы ровные.
* `AsyncReadback` — чтение цвета буфера через PBO: `glReadPixels` в буфер на
  стороне GPU не блокирует, а забираем данные через кадр-другой, когда fence
  уже сработал. Ни одного ожидания GPU.

Оба работают из draw-callback'ов Panda: в этот момент GL-контекст Panda
текущий, а рисование идёт в главном потоке (`support-threads #f`).
"""

from __future__ import annotations

import ctypes
from collections import deque
from typing import Optional

import numpy as np

try:
    from OpenGL import GL as _gl
except Exception as _exc:  # pragma: no cover - без PyOpenGL просто не включаемся
    _gl = None
    _GL_IMPORT_ERROR = _exc
else:
    _GL_IMPORT_ERROR = None


def available() -> bool:
    return _gl is not None


# --------------------------------------------------------------------------- #

class FrameLatencyLimiter:
    """
    Не пускать CPU дальше чем на `max_in_flight` кадров впереди GPU.

    Ставится последним display region'ом окна: к моменту его отрисовки все
    команды кадра уже отправлены.
    """

    def __init__(self, base, max_in_flight: int = 1, sort: int = 10_000):
        if _gl is None:
            raise RuntimeError(f"PyOpenGL недоступен: {_GL_IMPORT_ERROR}")
        from panda3d.core import Camera, NodePath, OrthographicLens

        self.max_in_flight = max(1, int(max_in_flight))
        self._fences: deque = deque()
        #: суммарное время ожидания GPU, с (для диагностики)
        self.waited = 0.0
        self.frames = 0

        # Пустая сцена: display region нужен только ради draw-callback'а.
        self._root = NodePath("frame_sync_root")
        cam = self._root.attach_new_node(Camera("frame_sync_cam",
                                                OrthographicLens()))
        self._dr = base.win.make_display_region(0, 1, 0, 1)
        self._dr.set_sort(sort)
        self._dr.set_clear_color_active(False)
        self._dr.set_clear_depth_active(False)
        self._dr.set_camera(cam)
        self._dr.set_draw_callback(self._on_draw)

    def _on_draw(self, cbdata) -> None:
        cbdata.upcall()
        import time
        gl = _gl
        self._fences.append(gl.glFenceSync(gl.GL_SYNC_GPU_COMMANDS_COMPLETE, 0))
        self.frames += 1
        while len(self._fences) > self.max_in_flight:
            fence = self._fences.popleft()
            t0 = time.perf_counter()
            # Таймаут — страховка от зависшего драйвера: лучше кадр с
            # очередью, чем вечное ожидание.
            gl.glClientWaitSync(fence, gl.GL_SYNC_FLUSH_COMMANDS_BIT,
                                200_000_000)
            self.waited += time.perf_counter() - t0
            gl.glDeleteSync(fence)

    def destroy(self) -> None:
        if self._dr is not None:
            self._dr.clear_draw_callback()
            win = self._dr.get_window()
            if win is not None:
                win.remove_display_region(self._dr)
            self._dr = None
        # Незавершённые fence остаются драйверу: удалять их можно только при
        # текущем контексте, а он есть лишь внутри кадра.
        self._fences.clear()


# --------------------------------------------------------------------------- #

class AsyncReadback:
    """
    Асинхронное чтение цвета offscreen-буфера Panda в numpy.

    `display_region` — регион буфера, чья отрисовка заканчивается нужной
    картинкой (его draw-callback занимается этот класс). `request()` заказывает
    копию: она снимается в ближайшем кадре и становится доступна в `latest`
    через кадр-другой, без ожидания GPU.

    `channels`/`dtype`: 1 + float32 — GL_RED/GL_FLOAT (сырая глубина),
    4 + uint8 — GL_RGBA/GL_UNSIGNED_BYTE (цвет).
    """

    def __init__(self, display_region, width: int, height: int,
                 channels: int = 1, dtype=np.float32):
        if _gl is None:
            raise RuntimeError(f"PyOpenGL недоступен: {_GL_IMPORT_ERROR}")
        gl = _gl
        self.width, self.height = int(width), int(height)
        self.channels = int(channels)
        self.dtype = np.dtype(dtype)
        self._format = {1: gl.GL_RED, 3: gl.GL_RGB, 4: gl.GL_RGBA}[self.channels]
        self._type = {np.dtype(np.float32): gl.GL_FLOAT,
                      np.dtype(np.uint8): gl.GL_UNSIGNED_BYTE}[self.dtype]
        self._nbytes = (self.width * self.height * self.channels
                        * self.dtype.itemsize)

        #: последняя пришедшая картинка: (h, w) или (h, w, c), строки снизу
        #: вверх, как их отдаёт GL
        self.latest: Optional[np.ndarray] = None
        #: номер кадра, в котором снята `latest`
        self.latest_frame = -1

        self._pbo = None
        self._fence = None
        self._fence_frame = -1
        self._requested = False
        self._frame = 0
        self._dr = display_region
        self._dr.set_draw_callback(self._on_draw)

    def request(self) -> None:
        """Заказать копию. Повторные заказы до её снятия схлопываются."""
        self._requested = True

    def _on_draw(self, cbdata) -> None:
        cbdata.upcall()
        self._frame += 1
        gl = _gl
        if self._pbo is None:
            self._pbo = gl.glGenBuffers(1)
            gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, self._pbo)
            gl.glBufferData(gl.GL_PIXEL_PACK_BUFFER, self._nbytes, None,
                            gl.GL_STREAM_READ)
            gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, 0)

        # 1) Забрать прошлую копию, если GPU её уже сделал.
        if self._fence is not None:
            state = gl.glClientWaitSync(self._fence, 0, 0)
            if state in (gl.GL_ALREADY_SIGNALED, gl.GL_CONDITION_SATISFIED):
                gl.glDeleteSync(self._fence)
                self._fence = None
                self._collect()

        # 2) Снять новую — только если прошлая уже забрана (один PBO).
        if self._requested and self._fence is None:
            self._requested = False
            gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, self._pbo)
            gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
            gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT0)
            gl.glReadPixels(0, 0, self.width, self.height, self._format,
                            self._type, ctypes.c_void_p(0))
            gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, 0)
            self._fence = gl.glFenceSync(gl.GL_SYNC_GPU_COMMANDS_COMPLETE, 0)
            self._fence_frame = self._frame

    def _collect(self) -> None:
        gl = _gl
        gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, self._pbo)
        ptr = gl.glMapBufferRange(gl.GL_PIXEL_PACK_BUFFER, 0, self._nbytes,
                                  gl.GL_MAP_READ_BIT)
        try:
            if ptr:
                raw = ctypes.string_at(ptr, self._nbytes)
                arr = np.frombuffer(raw, dtype=self.dtype)
                shape = ((self.height, self.width) if self.channels == 1
                         else (self.height, self.width, self.channels))
                self.latest = arr.reshape(shape)
                self.latest_frame = self._fence_frame
        finally:
            gl.glUnmapBuffer(gl.GL_PIXEL_PACK_BUFFER)
            gl.glBindBuffer(gl.GL_PIXEL_PACK_BUFFER, 0)

    def destroy(self) -> None:
        if self._dr is not None:
            self._dr.clear_draw_callback()
            self._dr = None
