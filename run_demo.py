# run_demo.py
# ---------------------------------------------------------------------------
# Standalone demo of the Digital Engineering 2026 UI — works WITHOUT Panda3D.
#
# What you get:
#   * Redesigned right panel (depth + height previews, metrics, sections,
#     2D→3D reconstruction list with "LOAD MORE" button).
#   * Stub 3D viewport (plain dark QFrame).
#   * Two floating HUD overlays (camera telemetry, data information) —
#     now implemented as CHILD widgets of the viewport, so they DO NOT
#     stay above other desktop apps when the main window is hidden.
#   * Synthetic depth-map and height-map frames piped through the real
#     signal/slot pipeline so the UI behaves identically to the Panda3D
#     integration.
#   * Stub reconstruction record source that simulates a paginated server,
#     so the LOAD MORE button actually appends records batch-by-batch.
#
# Run:
#     python run_demo.py
# ---------------------------------------------------------------------------

from __future__ import annotations

import math
import random
import sys
import time

import numpy as np
from PyQt6.QtCore import Qt, QTimer, QSize
from PyQt6.QtGui import QColor, QPainter, QLinearGradient
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QFrame, QLabel,
)

from ui_theme import apply_theme, COLOR_BG
from right_panel import RightPanel
from overlay_widgets import SceneOverlay


# ---------------------------------------------------------------------------
# Stub "3D viewport" — dark gradient frame.
# ---------------------------------------------------------------------------
class StubViewport(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: #050505;")
        self.setMinimumSize(1100, 700)

        lbl = QLabel("PANDA3D  VIEWPORT  (stub)", self)
        lbl.setStyleSheet(
            "color: #2E2E2E; font-size: 22px; font-weight: 700;"
            "letter-spacing: 6px; background: transparent;"
        )
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        lay = QHBoxLayout(self)
        lay.addWidget(lbl)

    def paintEvent(self, e):
        p = QPainter(self)
        g = QLinearGradient(0, 0, 0, self.height())
        g.setColorAt(0.0, QColor("#050505"))
        g.setColorAt(1.0, QColor("#0A0A0A"))
        p.fillRect(self.rect(), g)
        super().paintEvent(e)


# ---------------------------------------------------------------------------
# LUTs
# ---------------------------------------------------------------------------
def _build_lut(stops):
    stops = np.array(stops, dtype=np.float32)
    xs = np.linspace(0, 1, len(stops))
    grid = np.linspace(0, 1, 256)
    lut = np.empty((256, 4), dtype=np.uint8)
    for c in range(4):
        lut[:, c] = np.interp(grid, xs, stops[:, c]).astype(np.uint8)
    return lut


TURBO_LUT = _build_lut([
    [  6,   8,  12, 255],
    [  0, 140, 160, 255],
    [  0, 255, 136, 255],
    [210, 255, 120, 255],
    [255, 255, 255, 255],
])

MINT_LUT = _build_lut([
    [  6,   8,  12, 255],
    [ 10,  60,  60, 255],
    [  0, 160, 110, 255],
    [  0, 255, 136, 255],
    [230, 255, 200, 255],
])


# ---------------------------------------------------------------------------
# Synthetic feeders — prove the pipeline without Panda3D.
# ---------------------------------------------------------------------------
class SyntheticDepthSource:
    """Moving concentric-wave depth map, 16:9."""

    def __init__(self, preview, w: int = 512, h: int = 288):
        self.preview = preview
        self.w, self.h = w, h
        self._t0 = time.monotonic()

        xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)
        ys = np.linspace(-1.0, 1.0, h, dtype=np.float32) * (h / w)
        self.gx, self.gy = np.meshgrid(xs, ys)

        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self.timer.start(33)

    def _tick(self):
        t = time.monotonic() - self._t0
        r = np.sqrt(self.gx ** 2 + self.gy ** 2)
        pattern = 0.5 + 0.5 * np.sin(6.0 * r - 2.0 * t) * np.cos(1.5 * t)
        pattern = np.clip(pattern, 0.0, 1.0)
        rgba = TURBO_LUT[(pattern * 255).astype(np.uint8)]
        self.preview.push_frame(rgba.tobytes(), self.w, self.h, "rgba8")


class SyntheticHeightSource:
    """Square top-down height map with drifting Gaussians."""

    def __init__(self, preview, size: int = 256):
        self.preview = preview
        self.n = size
        self._t0 = time.monotonic()

        xs = np.linspace(-1.0, 1.0, size, dtype=np.float32)
        self.gx, self.gy = np.meshgrid(xs, xs)

        self.timer = QTimer()
        self.timer.timeout.connect(self._tick)
        self.timer.start(80)            # ~12 fps — matches realistic renderer

    def _tick(self):
        t = time.monotonic() - self._t0
        cx1 = 0.5 * math.cos(t * 0.3)
        cy1 = 0.5 * math.sin(t * 0.4)
        cx2 = 0.3 * math.cos(t * 0.7 + 1.0)
        cy2 = 0.3 * math.sin(t * 0.6 + 2.0)

        d1 = (self.gx - cx1) ** 2 + (self.gy - cy1) ** 2
        d2 = (self.gx - cx2) ** 2 + (self.gy - cy2) ** 2
        h = np.exp(-d1 * 6.0) + 0.6 * np.exp(-d2 * 10.0)
        h = np.clip(h / h.max(), 0.0, 1.0)

        rgba = MINT_LUT[(h * 255).astype(np.uint8)]
        self.preview.push_frame(rgba.tobytes(), self.n, self.n, "rgba8")


# ---------------------------------------------------------------------------
# Fake server — emulates tls_client.get_verified_models() pagination.
# ---------------------------------------------------------------------------
def make_stub_record_source(total: int = 73):
    rng = random.Random(42)
    models = ["truck_A.glb", "truck_B.glb", "wagon_01.glb", "tender_02.glb"]
    fillers = ["sand", "gravel", "coal", "toner"]

    entries = []
    for i in range(total):
        hh = rng.randint(8, 22)
        mm = rng.randint(0, 59)
        entries.append({
            "car_number":    f"{rng.choice(['А','Б','В','М','Т'])}"
                             f"{rng.randint(100, 999)}"
                             f"{rng.choice(['МА','НН','ТР','ВЛ'])}",
            "time":          f"2026-04-{rng.randint(1,23):02d} {hh:02d}:{mm:02d}",
            "data_type":     rng.choice(["ply", "height"]),
            "model":         rng.choice(models),
            "filler":        rng.choice(fillers),
            "target_volume": round(rng.uniform(12.0, 65.0), 1),
            "is_local":      rng.random() < 0.2,
            "img_file":      None,
        })

    def source(start: int, count: int):
        # Simulate network delay indirectly — just slice.
        return entries[start:start + count]

    return source, len(entries)


# ---------------------------------------------------------------------------
# Demo window
# ---------------------------------------------------------------------------
class DemoWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Toner · UI Redesign Demo")
        self.resize(1720, 1000)
        self.setMinimumSize(1280, 820)
        self.setStyleSheet(f"background-color: {COLOR_BG};")

        # Stub record source for the reconstruction list
        record_source, self._total = make_stub_record_source(73)

        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.viewport = StubViewport()
        self.right_panel = RightPanel(panda_app=None, record_source=record_source)

        root.addWidget(self.viewport, 1)
        root.addWidget(self.right_panel, 0)
        self.setCentralWidget(central)

        apply_theme(self)

        # Feed both previews
        self.depth_src = SyntheticDepthSource(self.right_panel.depth_preview)
        self.height_src = SyntheticHeightSource(self.right_panel.height_preview)

        # --- Overlays as CHILDREN of the viewport (not top-level windows) --
        # This is the critical fix: when the main window is hidden / minimised,
        # the overlays are hidden with it because they're parented to a widget
        # inside the main window, not to the screen.
        self.telemetry = SceneOverlay("Camera · Telemetry",
                                      anchor="top-left", parent=self.viewport)
        self.telemetry.set_rows([
            ("Pitch", "-90.0°"),
            ("Yaw",   "  0.0°"),
            ("Roll",  "  0.0°"),
            ("FOV",   " 60°"),
        ])
        self.telemetry.attach()

        self.data = SceneOverlay("Data · Information",
                                 anchor="top-right", parent=self.viewport)
        self.data.set_rows([
            ("Scene",     "truck_A"),
            ("Particles", "128 234"),
            ("Volume",    " 20.0 m³"),
            ("GPU",       "RTX 4090"),
        ])
        self.data.attach()

        # Animate overlay values
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._animate)
        self._anim_timer.start(80)
        self._t0 = time.monotonic()

        # Wire actions
        self.right_panel.run_btn.clicked.connect(self._on_run)
        self.right_panel.recon_list.item_activated.connect(self._on_recon_activated)

        # Kick off first page load
        self.right_panel.recon_list.reset_and_load()

    # ------------------------------------------------------------------
    def _animate(self):
        t = time.monotonic() - self._t0
        yaw = math.sin(t * 0.4) * 45.0
        pitch = -90.0 + math.cos(t * 0.3) * 10.0
        self.telemetry.update_row("Yaw",   f"{yaw:+6.1f}°")
        self.telemetry.update_row("Pitch", f"{pitch:+6.1f}°")

        parts = 128000 + int(2000 * math.sin(t * 0.8))
        self.data.update_row("Particles", f"{parts:>7,}".replace(",", " "))

    def _on_run(self):
        self.right_panel.status_text.setText(
            "● running simulation  ·  please wait"
        )

    def _on_recon_activated(self, entry: dict):
        self.right_panel.status_text.setText(
            f"● opened {entry.get('car_number','?')}  ·  "
            f"{entry.get('data_type','?').upper()}  ·  "
            f"vol {entry.get('target_volume','?')} m³"
        )


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Toner UI Demo")
    w = DemoWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
