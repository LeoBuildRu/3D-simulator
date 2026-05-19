# height_map_bridge.py
# ---------------------------------------------------------------------------
# Bridge: DepthMapRenderer.depth_texture → HeightMapPreview widget.
#
# Unlike panda_depth_bridge.py (which creates its OWN offscreen buffer),
# this bridge piggybacks on the existing `DepthMapRenderer` instance you
# already have.  It:
#
#   1. Flips `set_keep_ram_image(True)` on renderer.depth_texture so the
#      pixels are mirrored into RAM after each update_depth_texture() call.
#   2. Adds a taskMgr task that polls the ram image and forwards it to the
#      Qt widget through a queued signal.
#   3. Optionally re-triggers depth_map_renderer.update_depth_texture()
#      itself on a slower cadence, so the top-down map refreshes while the
#      scene evolves (the original renderer was one-shot on demand).
# ---------------------------------------------------------------------------

from __future__ import annotations

import numpy as np
from direct.task import Task


class HeightMapBridge:
    """
    Parameters
    ----------
    panda_app : ShowBase
    renderer  : DepthMapRenderer         (your existing instance)
    preview   : HeightMapPreview         (the Qt widget)
    auto_refresh_hz : float | None
        If > 0, the bridge will call `renderer.update_depth_texture()`
        at that rate.  Leave as None if your code already updates the
        depth texture elsewhere.
    stride : int
        Push every N-th tick to the widget.
    """

    def __init__(self, panda_app, renderer, preview, *,
                 auto_refresh_hz: float | None = 4.0,
                 stride: int = 1, colormap: str = "mint"):
        self.app = panda_app
        self.renderer = renderer
        self.preview = preview
        self.stride = max(1, stride)
        self._counter = 0
        self.colormap = colormap

        self._last_refresh = 0.0
        self._refresh_period = (1.0 / auto_refresh_hz) if auto_refresh_hz else None

        # Ensure the existing texture keeps a CPU-side copy.
        tex = renderer.depth_texture
        tex.set_keep_ram_image(True)

        self.app.taskMgr.add(self._tick, "HeightMapBridgeTick", sort=95)

    # ------------------------------------------------------------------
    def _tick(self, task):
        # Optional auto-refresh of the renderer itself
        if self._refresh_period is not None:
            now = task.time
            if now - self._last_refresh >= self._refresh_period:
                try:
                    self.renderer.update_depth_texture()
                except Exception as e:
                    print(f"[HeightMapBridge] update_depth_texture failed: {e}")
                self._last_refresh = now

        self._counter += 1
        if self._counter % self.stride != 0:
            return Task.cont

        tex = self.renderer.depth_texture
        if tex is None or not tex.has_ram_image():
            return Task.cont

        w, h = tex.get_x_size(), tex.get_y_size()
        ram = tex.get_ram_image_as("D")
        if ram is None:
            return Task.cont
        try:
            buf = memoryview(ram).tobytes()
        except Exception:
            return Task.cont
        if not buf:
            return Task.cont

        depth = np.frombuffer(buf, dtype=np.float32)
        if depth.size != w * h:
            return Task.cont
        depth = depth.reshape(h, w)

        near, far = np.percentile(depth, (2, 98))
        if far - near < 1e-6:
            return Task.cont
        norm = np.clip((depth - near) / (far - near), 0.0, 1.0)

        if self.colormap == "gray":
            frame = (norm * 255.0).astype(np.uint8)
            self.preview.push_frame(frame.tobytes(), w, h, "gray8")
        else:
            rgba = _mint_lut()[(norm * 255).astype(np.uint8)]
            self.preview.push_frame(rgba.tobytes(), w, h, "rgba8")

        return Task.cont

    def stop(self) -> None:
        self.app.taskMgr.remove("HeightMapBridgeTick")


# -- Mint-accent LUT (lazily built once) --------------------------------------
_LUT_CACHE: np.ndarray | None = None


def _mint_lut() -> np.ndarray:
    global _LUT_CACHE
    if _LUT_CACHE is not None:
        return _LUT_CACHE
    stops = np.array([
        [  6,   8,  12, 255],
        [ 10,  60,  60, 255],
        [  0, 160, 110, 255],
        [  0, 255, 136, 255],      # accent mint
        [230, 255, 200, 255],
    ], dtype=np.float32)
    xs = np.linspace(0, 1, len(stops))
    grid = np.linspace(0, 1, 256)
    lut = np.empty((256, 4), dtype=np.uint8)
    for c in range(4):
        lut[:, c] = np.interp(grid, xs, stops[:, c]).astype(np.uint8)
    _LUT_CACHE = lut
    return lut
