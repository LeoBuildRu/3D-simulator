# height_map_widget.py
# ---------------------------------------------------------------------------
# Real-time Height-Map preview — mirrors DepthMapRenderer.depth_texture.
#
# The existing `DepthMapRenderer` (depth_map_renderer.py) already renders an
# orthographic top-down depth pass into a Panda3D Texture.  We reuse that
# texture and mirror its RAM image into a Qt widget.  No changes required to
# DepthMapRenderer itself — we only flip `set_keep_ram_image(True)` on its
# texture and poll it from a taskMgr task.
#
# Pipeline:
#   DepthMapRenderer.depth_texture (F_depth_component32, T_float)
#         |
#         v
#   HeightMapBridge._tick (taskMgr, every N-th frame)
#         |   tex.get_ram_image_as("D")   -- float32 bytes
#         v
#   np.frombuffer → reshape → normalise → colourise (mint gradient)
#         |
#         v
#   signal frame_ready(bytes, w, h, fmt)  (queued)
#         |
#         v
#   HeightMapPreview._on_frame_ready (Qt main thread)
# ---------------------------------------------------------------------------

from __future__ import annotations

import time

from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QWidget, QFrame, QLabel, QVBoxLayout, QHBoxLayout,
)


# -- Shared 16:9 / arbitrary ratio frame --------------------------------------
class _AspectFrame(QFrame):
    def __init__(self, rw: int = 16, rh: int = 9, parent=None):
        super().__init__(parent)
        self._rw, self._rh = rw, rh
        self.setObjectName("DepthMapFrame")

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, w: int) -> int:
        return int(w * self._rh / self._rw)

    def sizeHint(self) -> QSize:
        return QSize(320, int(320 * self._rh / self._rw))


class HeightMapPreview(QWidget):
    """
    Embeddable top-down Height-Map preview.  Identical visual language as
    DepthMapPreview so the two read as a pair inside the right panel.
    Uses a square frame (1:1) since height maps are typically square.
    """

    frame_ready = pyqtSignal(bytes, int, int, str)

    def __init__(self, parent=None, ratio: tuple[int, int] = (1, 1)):
        super().__init__(parent)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        # Header
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)

        eyebrow = QLabel("HEIGHT MAP / TOP-DOWN")
        eyebrow.setProperty("role", "eyebrow")

        self.status_chip = QLabel("IDLE")
        self.status_chip.setProperty("role", "chip-idle")
        self.status_chip.setAlignment(Qt.AlignmentFlag.AlignCenter)

        header.addWidget(eyebrow)
        header.addStretch(1)
        header.addWidget(self.status_chip)

        root.addLayout(header)

        # Canvas
        self.frame = _AspectFrame(*ratio, parent=self)
        frame_layout = QVBoxLayout(self.frame)
        frame_layout.setContentsMargins(6, 6, 6, 6)
        frame_layout.setSpacing(0)

        self.canvas = QLabel("NO SIGNAL", self.frame)
        self.canvas.setObjectName("DepthMapCanvas")
        self.canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.canvas.setMinimumHeight(120)
        frame_layout.addWidget(self.canvas)

        root.addWidget(self.frame)

        # Footer
        footer = QHBoxLayout()
        footer.setContentsMargins(2, 0, 2, 0)
        footer.setSpacing(12)

        self.res_label = QLabel("—×—")
        self.res_label.setProperty("role", "muted")
        self.fps_label = QLabel("0.0 FPS")
        self.fps_label.setProperty("role", "muted")

        footer.addWidget(self.res_label)
        footer.addStretch(1)
        footer.addWidget(self.fps_label)
        root.addLayout(footer)

        self.frame_ready.connect(
            self._on_frame_ready, Qt.ConnectionType.QueuedConnection)

        self._frame_count = 0
        self._last_ts = 0.0

    # --- Public API ---------------------------------------------------
    def push_frame(self, raw: bytes, w: int, h: int, fmt: str = "rgba8") -> None:
        self.frame_ready.emit(raw, w, h, fmt)

    def set_live(self, live: bool) -> None:
        if live:
            self.status_chip.setText("● LIVE")
            self.status_chip.setProperty("role", "chip-live")
        else:
            self.status_chip.setText("IDLE")
            self.status_chip.setProperty("role", "chip-idle")
        self.status_chip.style().unpolish(self.status_chip)
        self.status_chip.style().polish(self.status_chip)

    # --- Slot ---------------------------------------------------------
    def _on_frame_ready(self, raw: bytes, w: int, h: int, fmt: str) -> None:
        if fmt == "gray8":
            img = QImage(raw, w, h, w, QImage.Format.Format_Grayscale8)
        elif fmt == "rgba8":
            img = QImage(raw, w, h, w * 4, QImage.Format.Format_RGBA8888)
        else:
            return

        img = img.mirrored(False, True)

        pm = QPixmap.fromImage(img).scaled(
            self.canvas.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.canvas.setPixmap(pm)
        self.res_label.setText(f"{w}×{h}")

        now = time.monotonic()
        self._frame_count += 1
        if self._last_ts == 0.0:
            self._last_ts = now
        elif now - self._last_ts >= 0.5:
            fps = self._frame_count / (now - self._last_ts)
            self.fps_label.setText(f"{fps:5.1f} FPS")
            self._frame_count = 0
            self._last_ts = now

        if self.status_chip.text() != "● LIVE":
            self.set_live(True)
